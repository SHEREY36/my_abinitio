# Tier 1 / Tier 2 — Hands-On Playbook (Field Surrogate + Active Learning)

This is the concrete operating manual for the `surrogate/` package. It assumes Tier 0 is done and calibrated. Everything here runs on **CPU**; no GPU is needed for this POD+GP approach at the ~100-run scale.

---

## 1. What the surrogate produces

A trained surrogate maps an operating point **θ** = (T_wafer, pressure, DCS sccm, H₂ sccm, HCl sccm, rotation) to:

- **Full fields** (Tier 2): u, v, p, T, X_DCS, X_HCl, X_SiCl2, Dep_Si_B, cov_H_S, cov_Cl_S — visualization-ready, on the ACE+ mesh.
- **Per-node uncertainty** for every field (drives active learning).
- **Gradients**: ∂field/∂θ (`Tier2FieldSurrogate.field_sensitivity`) and ∂QoI/∂θ (`Tier1Surrogate.sensitivity`); spatial ∇field comes from the mesh.
- **Scalar QoIs** (Tier 1, and Tier 2 via `qois()`): GR(0), GR(edge), ΔGR, mean growth, within-wafer nonuniformity — recomputed from the *predicted* deposition field, so the numbers and the pictures come from one model.

**Two low-fidelities, by design:**
- Growth-rate QoIs are corrected on top of the **Tier 0 analytic prior** → `Tier1`.
- Flow/thermal **fields** are learned from cheap **flow+heat-only** runs and reused across all chemistry conditions → `Tier2` (flow fields train on *all* runs; chemistry/deposition fields train on full-chemistry runs only).

---

## 2. Package map

```
surrogate/
  tier0_backbone.py  dependency-light Tier 0 two-resistance physics backbone
  run_tier0_jackel_si.py  writes the calibrated Jackel Tier 0 profile/summary
  paramspace.py   ParameterSpace, jackel_si_space()   -- design vars, bounds, Sobol/LHS, transforms
  snapshots.py    Snapshot, SnapshotDataset, loaders   -- ingest ACE+ exports; radial-average -> QoIs
  core.py         POD, GP, MultiGP                      -- reduction + GP with std and d/dinput
  tier1.py        Tier1Surrogate, make_tier0_prior      -- MF QoI/profile correction of Tier 0
  tier2.py        Tier2FieldSurrogate                   -- POD-GP per field; fields, std, sensitivities, QoIs
  active.py       propose_batch, active_learning_step   -- ParEGO EI + field-uncertainty, role quotas
  runner.py       write_run_table, inlet_bc             -- proposals -> CFD-ACE+ run table + manifest
  demo_synthetic.py  end-to-end on fabricated data (run this first)
```

Install the Tier 1/Tier 2 dependencies:
```bash
python -m pip install ".[surrogate]"
```

Run the demo to see the whole loop with no ACE+ data:
```bash
python3 -W ignore surrogate/demo_synthetic.py
```

---

## 3. Sampling plan (explicit)

| Phase | Runs | Fidelity | How |
|---|---|---|---|
| Initial DoE | **16** | full_chem | `space.sobol(16, seed=0)` |
| Initial DoE | **16** | flow_heat | `space.sobol(16, seed=100)` |
| Each AL iter | **2** | full_chem | top ParEGO-EI (exploit growth/uniformity) |
| Each AL iter | **2** | full_chem | top deposition/species uncertainty (explore) |
| Each AL iter | **2** | flow_heat | top flow-field uncertainty (cheap coverage) |

- **Candidate pool:** 2000 Sobol points scored per iteration (`pool_size`).
- **Budget bookkeeping:** cost weights `{flow_heat: 1, full_chem: 12}`. One AL iter ≈ 4·12 + 2·1 = **50 cost units**. Starting 16 HF + 16 LF ≈ 208 units; ~6 iters ≈ 300 more → ~**64 full-chem runs total**, inside your ~100 budget.
- Sobol warns unless n is a power of two — use 16/32/64 to keep the low-discrepancy property.

---

## 4. Active-learning loop (explicit steps)

**Step 1 — initial runs.** Generate and run the initial DoE:
```python
import numpy as np, surrogate as S
space = S.jackel_si_space()
hf = [{n: v for n, v in zip(space.names, row)} for row in space.sobol(16, seed=0)]
lf = [{n: v for n, v in zip(space.names, row)} for row in space.sobol(16, seed=100)]
S.write_run_table([S.Proposal(p, "full_chem", 0, "init") for p in hf] +
                  [S.Proposal(p, "flow_heat", 0, "init") for p in lf],
                  "surrogate/outputs/init_batch.csv")
```
Feed `surrogate/outputs/init_batch.csv` to CFD-ACE+ parametric mode. `flow_heat` rows enable **Flow+Heat**; `full_chem` rows enable **Flow+Heat+Chemistry** (column `modules`). The inlet BC columns (`x_*`, `w_*`, `mdot_total_kg_s`) are pre-computed.

**Step 2 — export fields.** After each run converges, export its solution (see §6) to `exports/run_XXX.dat` (name per `*.manifest.json`).

**Step 3 — ingest.**
```python
from surrogate import Snapshot, SnapshotDataset, load_tecplot_ascii
snaps = []
for rid, info in manifest.items():
    s = load_tecplot_ascii(info["output_file"], info["params"], info["fidelity"])
    s.wafer_mask = (s.coords[:, 1] < 1e-6)      # <-- set your wafer patch here
    snaps.append(s)
ds = SnapshotDataset(snaps)
```

**Step 4 — fit + propose.**
```python
radial = np.linspace(0, 0.15, 31)               # wafer radius grid
prior  = S.make_tier0_prior(radial)             # wraps tier0_backbone.py
t1, t2, batch = S.active_learning_step(space, ds, prior, radial,
                                       n_exploit=2, n_chem=2, n_flow=2)
S.write_run_table(batch, "surrogate/outputs/iter_batch.csv", start_id=len(ds.snapshots))
```

**Step 5 — run the batch** in ACE+ (the 2 `flow_heat` rows are fast), export, append to `ds`, **go to Step 4.**

**Step 6 — stop when** any holds: held-out `Dep_Si_B` L2 error < 5% AND Tier-1 GRmean error < 3%; **or** Pareto-hypervolume gain < 1% for 2 consecutive iters; **or** budget spent. Validate the final optimum by one confirming ACE+ run.

---

## 5. Using the trained surrogate

```python
theta = dict(wafer_temperature_C=640, pressure_Torr=300, main_dcs_sccm=450,
             main_h2_sccm=9000, hcl_sccm=30, rotation_rpm=60)

fields = t2.predict_all(theta)                  # {name: FieldPrediction(values, std)}
dep    = t2.predict_field(theta, "Dep_Si_B")    # .values, .std on the mesh
qoi    = t2.qois(theta)                         # GR0/GRedge/dGR/GRmean/nonuniformity from the field
q1     = t1.predict(theta)                      # corrected QoIs + std + GR(r) profile
dGR_dθ = t1.sensitivity(theta)["GRmean_nm_min"] # optimization gradient (per physical param)
dfield = t2.field_sensitivity(theta, "Dep_Si_B")# (N, d): how the deposition field moves with θ
```
- **Visualize** by writing `dep.values` back onto the mesh (same node order as the ACE+ export) and loading in CFD-VIEW/ParaView — this is the side-by-side check against the real ACE+ plot.
- **Optimize** with the gradients: gradient ascent on GRmean subject to a nonuniformity cap, or feed Tier-1 QoI GPs to a multi-objective optimizer (ParEGO is already the acquisition; `botorch` qNEHVI is the drop-in scale-up).

---

## 6. Getting fields out of ACE+ (.DTF → loadable)

DTF is binary/proprietary, so **export**, don't parse:
1. CFD-VIEW → open the `.DTF` → **File ▸ Export ▸ Tecplot (ASCII)** (or CSV). Select nodal variables: `Velocity` (U,V), `Pressure`, `Temperature`, the gas mole fractions, `Dep_SI(B)`, `H(S)`, `Cl(S)`.
2. Name each export per the manifest (`exports/run_XXX.dat`).
3. `load_tecplot_ascii` maps common ACE+ names (`DEP_SI(B)→Dep_Si_B`, `H(S)→cov_H_S`, …). For CSV use `load_csv` with a `field_cols` map.
4. **Wafer mask:** either export an `is_wafer` flag column, or set `s.wafer_mask` by geometry after loading (nodes on the wafer patch).
5. **One mesh:** parametric runs keep the mesh fixed, so nodal arrays stack directly for POD. If a mesh ever differs, interpolate to a reference grid before ingesting (the loader raises if node counts disagree).

To script it, record a CFD-VIEW macro of one export and replay it over the batch, or wire the ESI DTF API into `load_dtf` (left as a stub).

---

## 7. Validation & gotchas

- **Held-out field error:** refit on all-but-k full-chem runs, predict them, report `‖pred−true‖/‖true‖` on wafer nodes (the demo does this). Track it per iteration; active learning should drive it down.
- **POD energy:** `POD.fit(..., energy_tol=0.9999)`. Watch `pod.n_modes`; if one field needs many modes it's rougher — give it more runs.
- **GP data hunger:** GPs in 6-D want ≳ 2·(d+1) points per field before you trust them; that's why the initial DoE is 16 full-chem. Early predictions carry wide std — that's the point, and the acquisition uses it.
- **Flow reuse is the payoff:** every `flow_heat` run improves u/v/p/T for *all* chemistry conditions at 1/12 the cost — keep feeding a few each iteration.
- **Scale-up path (only if needed):** swap `MultiGP` for `botorch`/`gpytorch` (GPU, qNEHVI multi-objective) when runs exceed a few hundred or you add geometry variation; then the POD-GP becomes a DeepONet/FNO for mesh/geometry generalization. Not needed for the fixed-mesh parametric study.

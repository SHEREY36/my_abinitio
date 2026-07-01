"""End-to-end demo of the Tier 1 / Tier 2 framework on SYNTHETIC ACE+-like data.

No CFD-ACE+ needed: we fabricate smooth fields with realistic parameter trends
(rotation flattens the radial growth profile -- the effect Tier 0 cannot see),
then run the real pipeline: POD + GP surrogates, held-out validation, field &
QoI prediction with sensitivities, and one cost-aware active-learning step that
writes the next ACE+ run table.  Swap `synthetic_dataset` for real Tecplot/CSV
loads and everything else is unchanged.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from surrogate.paramspace import jackel_si_space
from surrogate.snapshots import Snapshot, SnapshotDataset, growth_profile_from_deposition, qois_from_profile
from surrogate.tier1 import Tier1Surrogate
from surrogate.tier2 import Tier2FieldSurrogate
from surrogate.active import propose_batch, observed_stats
from surrogate.runner import write_run_table

RHO_SI = 2329.0
L = 0.25                 # wafer radius / domain length (m)
NX, NY = 40, 12
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "surrogate" / "outputs"


# ----------------------------------------------------------------------------- synthetic "truth"
def build_mesh():
    x = np.linspace(0, L, NX)
    y = np.linspace(0, 0.05, NY)
    X, Y = np.meshgrid(x, y, indexing="ij")
    coords = np.stack([X.ravel(), Y.ravel()], 1)
    wafer = coords[:, 1] < 1e-9            # bottom row = wafer
    return coords, wafer


def truth_fields(params, coords, wafer, fidelity):
    x, y = coords[:, 0], coords[:, 1]
    T = params["wafer_temperature_C"] + 273.15
    pdcs = params["main_dcs_sccm"]; phcl = params["hcl_sccm"]
    rot = params["rotation_rpm"]; h2 = params["main_h2_sccm"]
    v_in = 0.1145 * h2 / 9000.0
    ymax = y.max()
    # ---- flow / thermal (available at both fidelities) ----
    vel = v_in * (0.25 + 0.75 * np.exp(-y / 0.015)) * (1 + 0.4 * (x / L))
    vel *= 1 + 0.15 * rot / 120.0
    u = vel * (0.9 - 0.5 * y / ymax)
    v = -0.3 * vel * (x / L)
    p = params["pressure_Torr"] * 133.322 - 5.0 * vel
    Tf = T * (1 - 0.06 * y / ymax)
    fields = {"u": u, "v": v, "p": p, "T": Tf, "vel_mag": vel}
    if fidelity == "full_chem":
        r = x
        base_gr = 1.8 * (pdcs / 400.0) ** 0.45 * np.exp(9000 * (1 / 923.15 - 1 / T)) \
            * (1.0 - 0.25 * np.log1p((phcl / max(pdcs, 1e-9)) / 0.01))
        base_gr = float(np.clip(base_gr, 0.2, 6.0))
        shape = 1 + 0.11 * (r / L) ** 2 * (1 - 0.7 * rot / 120.0)   # rotation flattens edge rise
        gr = base_gr * shape
        dep = np.where(wafer, gr / (1e9 * 60.0) * RHO_SI, 0.0)      # kg/m^2/s, wafer only
        x_dcs = (pdcs / 15000.0) * (0.4 + 0.6 * np.exp(-(ymax - y) / 0.02)) * (1 - 0.2 * x / L)
        x_hcl = (phcl / 15000.0 + 0.02 * gr / base_gr) * (0.5 + 0.5 * y / ymax)
        fields.update({"Dep_Si_B": dep, "X_DCS": x_dcs, "X_HCl": x_hcl})
    return fields


def synthetic_dataset(space, n_hf=32, n_lf=24, seed=1):
    coords, wafer = build_mesh()
    snaps = []
    Xhf = space.sobol(n_hf, seed=seed)
    Xlf = space.sobol(n_lf, seed=seed + 100)
    for row in Xhf:
        d = space.array_to_dicts(row)[0]
        snaps.append(Snapshot(d, "full_chem", coords, truth_fields(d, coords, wafer, "full_chem"), wafer))
    for row in Xlf:
        d = space.array_to_dicts(row)[0]
        snaps.append(Snapshot(d, "flow_heat", coords, truth_fields(d, coords, wafer, "flow_heat"), wafer))
    return SnapshotDataset(snaps), coords, wafer


# ----------------------------------------------------------------------------- Tier 0 prior (rotation-blind, like the real backbone)
def synthetic_prior(radial_grid):
    """Mimics Tier 0: same law but WITHOUT the rotation effect, so Tier 1 must
    learn the rotation correction from data."""
    def prior(params):
        T = params["wafer_temperature_C"] + 273.15
        pdcs = params["main_dcs_sccm"]; phcl = params["hcl_sccm"]
        base = 1.8 * (pdcs / 400.0) ** 0.45 * np.exp(9000 * (1 / 923.15 - 1 / T)) \
            * (1.0 - 0.25 * np.log1p((phcl / max(pdcs, 1e-9)) / 0.01))
        base = float(np.clip(base, 0.2, 6.0))
        gr = base * (1 + 0.11 * (radial_grid / L) ** 2)            # rotation=0 assumption
        q = {"GR0_nm_min": float(gr[0]), "GRedge_nm_min": float(gr[-1]),
             "dGR_nm_min": float(gr[-1] - gr[0]), "GRmean_nm_min": float(gr.mean()),
             "nonuniformity_pct": float(100 * (gr.max() - gr.min()) / max(gr.mean(), 1e-9))}
        return {"r": radial_grid, "GR_r_nm_min": gr, "qois": q}
    return prior


# ----------------------------------------------------------------------------- run
def main():
    space = jackel_si_space()
    ds, coords, wafer = synthetic_dataset(space)
    radial = np.linspace(0, L, 31)
    prior = synthetic_prior(radial)

    print(f"dataset: {len(ds.by_fidelity('full_chem'))} full_chem + "
          f"{len(ds.by_fidelity('flow_heat'))} flow_heat snapshots\n")

    # ---- fit Tier 1 & Tier 2 ----
    t1 = Tier1Surrogate(space, prior, radial, wafer_radius_m=L).fit(ds)
    t2 = Tier2FieldSurrogate(space, wafer_radius_m=L).fit(ds)
    print("Tier 2 fields trained (name: #modes, #train, fidelity):")
    for n, m in t2.models.items():
        print(f"   {n:10s}: {m.pod.n_modes:2d} modes, {m.n_train:2d} runs, {m.fidelity_used}")

    # ---- held-out validation (fit on all but 4 HF, predict them) ----
    hf = ds.by_fidelity("full_chem"); hold = hf[-4:]
    ds_tr = SnapshotDataset(hf[:-4] + ds.by_fidelity("flow_heat"))
    t2v = Tier2FieldSurrogate(space, wafer_radius_m=L).fit(ds_tr)
    t1v = Tier1Surrogate(space, prior, radial, wafer_radius_m=L).fit(ds_tr)
    dep_err, grm_err = [], []
    for s in hold:
        pred = t2v.predict_field(s.params, "Dep_Si_B").values
        rel = np.linalg.norm(pred[wafer] - s.fields["Dep_Si_B"][wafer]) / \
              np.linalg.norm(s.fields["Dep_Si_B"][wafer])
        dep_err.append(rel)
        q_true = qois_from_profile(growth_profile_from_deposition(coords, wafer, s.fields["Dep_Si_B"], L))
        q_pred = t1v.predict(s.params).qois
        grm_err.append(abs(q_pred["GRmean_nm_min"] - q_true["GRmean_nm_min"]) / q_true["GRmean_nm_min"])
    print(f"\nheld-out (4 runs):  Dep_Si_B field L2 err = {np.mean(dep_err)*100:.1f}%   "
          f"GRmean err = {np.mean(grm_err)*100:.1f}%")

    # ---- predict at a new operating point: fields + QoIs + sensitivities ----
    test = {"wafer_temperature_C": 640.0, "pressure_Torr": 300.0, "main_dcs_sccm": 450.0,
            "main_h2_sccm": 9000.0, "hcl_sccm": 30.0, "rotation_rpm": 60.0}
    q = t2.qois(test)
    p1 = t1.predict(test)
    print(f"\nprediction @ test point:")
    print(f"   Tier2 fields->QoI : GRmean={q['GRmean_nm_min']:.3f} nm/min, "
          f"nonunif={q['nonuniformity_pct']:.2f}%")
    print(f"   Tier1 corrected   : GRmean={p1.qois['GRmean_nm_min']:.3f} "
          f"+/- {p1.qois_std['GRmean_nm_min']:.3f} nm/min")
    sens = t1.sensitivity(test)["GRmean_nm_min"]
    print("   dGRmean/dparam    : " +
          ", ".join(f"{n}={sens[i]:+.2e}" for i, n in enumerate(space.names)))
    depfld = t2.predict_field(test, "Dep_Si_B")
    print(f"   Dep_Si_B field    : peak={np.abs(depfld.values[wafer]).max():.3e} kg/m2/s, "
          f"mean node std={depfld.std[wafer].mean():.2e}")

    # ---- one active-learning step: propose next ACE+ runs ----
    batch = propose_batch(space, ds, t1, t2, n_exploit=2, n_chem=2, n_flow=2)
    print("\nproposed next ACE+ batch (cost-aware):")
    for b in batch:
        print(f"   [{b.fidelity:9s}] {b.reason:16s} score={b.score:.3f}  "
              f"T={b.params['wafer_temperature_C']:.0f}C P={b.params['pressure_Torr']:.0f}Torr "
              f"DCS={b.params['main_dcs_sccm']:.0f} rot={b.params['rotation_rpm']:.0f}rpm")
    out_path = OUT_DIR / "next_batch.csv"
    rows = write_run_table(batch, out_path, out_dir="exports")
    n_hf = sum(1 for b in batch if b.fidelity == "full_chem")
    print(f"\nwrote run table: {len(rows)} runs ({n_hf} full_chem, {len(rows)-n_hf} flow_heat) "
          f"-> {out_path}")


if __name__ == "__main__":
    main()

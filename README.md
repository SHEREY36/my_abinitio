# my_abinitio

Standalone workflow for generating temperature-dependent surface reaction rate
constants for carbon incorporation in SiGeC epitaxy.

The project is deliberately separate from the Epitaxy reactor model.  It
produces intrinsic surface-chemistry parameters that ACE+ can consume:

```text
MACE/MLIP NEB -> DFT correction -> harmonic TST -> Arrhenius fit -> ACE+ surface rates
```

Optional layer:

```text
kMC -> effective Csub/Cint branching validation
```

## What This Project Produces

For the two carbon pathways:

```text
C(S) + Si(B) -> C_sub + Si(S)
C(S)         -> C_int
```

the target output is:

```text
A_C_sub, n_C_sub, E_C_sub/R
A_C_int, n_C_int, E_C_int/R
```

in the modified Arrhenius form used by ACE+:

```text
k(T) = A * T^n * exp(-(E/R)/T)
```

## Quick Validation

The toy pipeline runs without external scientific packages:

```bash
python3 scripts/00_run_toy_pipeline.py
python3 -m unittest discover -s tests
```

It creates:

```text
results/toy_neb.json
results/toy_dft_corrected.json
results/toy_arrhenius.json
results/toy_kmc.json
results/ace_surface_rates.c
```

## Real Workflow On Cluster/Laptop With Scientific Stack

1. Install optional atomistic tools:

```bash
python -m pip install -e ".[atomistics]"
```

2. Build Si/SiGeC surface endpoint structures:

```bash
python scripts/00_build_sige_surface.py \
  --ge-fraction 0.10 \
  --termination H \
  --termination-fraction 0.25
```

Visualize the resulting initial/final structures:

```bash
ase gui structures/C_surface_IS.traj structures/C_sub_FS.traj structures/C_int_FS.traj
```

The scientific assumptions behind these starting structures are documented in
`docs/sige_surface_rationale.md`.

3. Run MACE/MLIP NEB screening:

```bash
python scripts/01_mace_neb_paths.py \
  --config configs/c_paths.json \
  --device auto \
  --default-dtype float32
```

On Apple Silicon, `--device auto` prefers PyTorch/MPS acceleration.  For a
fully portable CPU run on an office laptop or cluster login node:

```bash
python scripts/01_mace_neb_paths.py \
  --config configs/c_paths.json \
  --device cpu \
  --default-dtype float32
```

After MACE NEB, write the DFT single-point manifest:

```bash
python scripts/01b_write_dft_manifest.py \
  --neb results/mace_neb_summary.json \
  --output results/dft_points_manifest.csv
```

4. Run or parse DFT single-point corrections:

```bash
python scripts/02_qe_singlepoint_corrections.py \
  --neb results/mace_neb_summary.json \
  --dft results/qe_singlepoints.json
```

5. Convert corrected barriers to rate constants:

```bash
python scripts/03_harmonic_tst_arrhenius.py \
  --barriers results/dft_corrected_barriers.json
```

6. Validate branching with kMC:

```bash
python scripts/04_kmc_branching.py --rates results/arrhenius_rates.json
```

7. Write ACE+ constants:

```bash
python scripts/05_make_ace_rates.py --rates results/arrhenius_rates.json
```

## Directory Layout

```text
configs/              input examples
scripts/              runnable workflow steps
src/my_abinitio/      reusable library code
tests/                toy validation tests
results/              generated outputs, ignored by git except .gitkeep
```

## Important Scientific Boundary

Without experimental data, these constants are not "calibrated to reactor data."
They are predictive, first-principles/MLIP-assisted surface kinetics.  ACE+
should still handle reactor-specific gas transport, pressure, residence time,
wall temperature, and boundary-layer effects.

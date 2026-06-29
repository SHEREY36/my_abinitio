#!/bin/bash
#SBATCH --job-name=c_rates_export
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=logs/rates_%j.out
#SBATCH --error=logs/rates_%j.err

set -euo pipefail

mkdir -p logs

python scripts/02_qe_singlepoint_corrections.py \
  --neb results/mace_neb_summary.json \
  --dft results/qe_singlepoints.json \
  --output results/dft_corrected_barriers.json

python scripts/03_harmonic_tst_arrhenius.py \
  --barriers results/dft_corrected_barriers.json \
  --output results/arrhenius_rates.json

python scripts/04_kmc_branching.py \
  --rates results/arrhenius_rates.json \
  --output results/kmc_branching.json

python scripts/05_make_ace_rates.py \
  --rates results/arrhenius_rates.json \
  --output results/ace_surface_rates.c

#!/bin/bash
#SBATCH --job-name=c_neb_mace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=logs/mace_neb_%j.out
#SBATCH --error=logs/mace_neb_%j.err

set -euo pipefail

mkdir -p logs

# Edit these for your cluster.
# module load python/3.11
# module load cuda
# source .venv/bin/activate

python scripts/01_mace_neb_paths.py \
  --config configs/c_paths.json \
  --output results/mace_neb_summary.json

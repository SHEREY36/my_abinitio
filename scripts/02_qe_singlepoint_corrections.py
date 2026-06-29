#!/usr/bin/env python3
"""Apply DFT single-point corrections to MACE NEB barriers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from my_abinitio.dft_correction import correct_barrier_table
from my_abinitio.io_utils import read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--neb", default="results/mace_neb_summary.json")
    parser.add_argument("--dft", default="results/qe_singlepoints.json")
    parser.add_argument("--output", default="results/dft_corrected_barriers.json")
    args = parser.parse_args()
    corrected = correct_barrier_table(read_json(args.neb), read_json(args.dft))
    write_json(args.output, corrected)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

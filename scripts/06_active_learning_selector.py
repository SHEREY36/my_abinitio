#!/usr/bin/env python3
"""Select atomistic structures/pathways that most need DFT labels."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from my_abinitio.io_utils import read_json, write_json
from my_abinitio.surrogate import select_by_uncertainty


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="results/barrier_candidates.json")
    parser.add_argument("--output", default="results/active_learning_selected.json")
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()
    selected = select_by_uncertainty(read_json(args.candidates), args.k)
    write_json(args.output, selected)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

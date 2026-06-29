#!/usr/bin/env python3
"""Convert corrected barriers to TST rates and ACE+ Arrhenius parameters."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from my_abinitio.io_utils import read_json, write_json
from my_abinitio.tst import rates_from_barrier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--barriers", default="results/dft_corrected_barriers.json")
    parser.add_argument("--output", default="results/arrhenius_rates.json")
    parser.add_argument(
        "--temperatures",
        default="773.15,798.15,823.15,848.15,873.15,898.15,923.15",
        help="Comma-separated Kelvin values",
    )
    args = parser.parse_args()
    temps = [float(x) for x in args.temperatures.split(",")]
    payload = read_json(args.barriers)
    rates = {
        row["name"]: rates_from_barrier(row["name"], float(row["barrier_ev"]), temps)
        for row in payload["barriers"]
    }
    write_json(args.output, rates)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

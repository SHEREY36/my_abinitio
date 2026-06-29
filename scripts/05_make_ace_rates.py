#!/usr/bin/env python3
"""Write ACE+/C constants from Arrhenius rate JSON."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from my_abinitio.ace_writer import write_ace_surface_rates
from my_abinitio.io_utils import read_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rates", default="results/arrhenius_rates.json")
    parser.add_argument("--output", default="results/ace_surface_rates.c")
    args = parser.parse_args()
    write_ace_surface_rates(args.output, read_json(args.rates))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

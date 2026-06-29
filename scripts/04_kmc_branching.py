#!/usr/bin/env python3
"""Validate Csub/Cint branching with a small Gillespie race."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from my_abinitio.io_utils import read_json, write_json
from my_abinitio.kmc import gillespie_branching


def rate_at_T(fit: dict, T: float) -> float:
    return fit["A"] * (T ** fit["n"]) * math.exp(-fit["E_over_R_K"] / T)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rates", default="results/arrhenius_rates.json")
    parser.add_argument("--output", default="results/kmc_branching.json")
    parser.add_argument("--temperature", type=float, default=823.15)
    parser.add_argument("--k-bury", type=float, default=0.0)
    parser.add_argument("--events", type=int, default=10000)
    args = parser.parse_args()

    rates = read_json(args.rates)
    k_sub = rate_at_T(rates["C_sub"]["modified_arrhenius"], args.temperature)
    k_int = rate_at_T(rates["C_int"]["modified_arrhenius"], args.temperature)
    result = gillespie_branching(k_sub, k_int, args.k_bury, args.events)
    result.update({"T_K": args.temperature, "k_sub_s^-1": k_sub, "k_int_s^-1": k_int})
    write_json(args.output, result)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

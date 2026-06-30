#!/usr/bin/env python3
"""Run the Tier 0 Jackel DCS Si epitaxy surrogate."""

from __future__ import annotations

import argparse
from pathlib import Path

from tier0_backbone import evaluate_config, write_profile_csv, write_result_json


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "surrogate" / "configs" / "jackel_si_tier0.json"))
    parser.add_argument("--csv", default=str(ROOT / "surrogate" / "outputs" / "jackel_si_tier0_profile.csv"))
    parser.add_argument("--json", default=str(ROOT / "surrogate" / "outputs" / "jackel_si_tier0_summary.json"))
    args = parser.parse_args()

    result = evaluate_config(args.config)
    write_profile_csv(args.csv, result.profile)
    write_result_json(args.json, result)

    print("Tier 0 Jackel Si surrogate complete")
    print(f"  profile CSV : {args.csv}")
    print(f"  summary JSON: {args.json}")
    print(f"  mean growth : {result.summary['mean_growth_nm_min']:.4g} nm/min")
    print(f"  center/edge : {result.summary['center_growth_nm_min']:.4g} / {result.summary['edge_growth_nm_min']:.4g} nm/min")
    print(f"  regime      : {result.summary['dominant_regime']}")
    print(f"  HCl/DCS fit : {result.summary['calibrated_hcl_to_dcs_ratio']:.4g}")


if __name__ == "__main__":
    main()

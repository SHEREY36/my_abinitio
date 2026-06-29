#!/usr/bin/env python3
"""Write a CSV manifest of DFT single-point structures from MACE NEB output."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from my_abinitio.io_utils import read_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--neb", default="results/mace_neb_summary.json")
    parser.add_argument("--output", default="results/dft_points_manifest.csv")
    args = parser.parse_args()

    neb = read_json(args.neb)
    rows = []
    for barrier in neb["barriers"]:
        name = barrier["name"]
        points = barrier["dft_points"]
        rows.extend(
            [
                {
                    "pathway": name,
                    "point": "IS",
                    "purpose": "required for barrier correction",
                    "mace_energy_ev": barrier["mace_is_ev"],
                    "traj": points["is_traj"],
                    "xyz": points["is_xyz"],
                },
                {
                    "pathway": name,
                    "point": "TS",
                    "purpose": "required for barrier correction",
                    "mace_energy_ev": barrier["mace_ts_ev"],
                    "traj": points["ts_traj"],
                    "xyz": points["ts_xyz"],
                },
                {
                    "pathway": name,
                    "point": "FS",
                    "purpose": "optional sanity check for reaction energy",
                    "mace_energy_ev": barrier["mace_fs_ev"],
                    "traj": points["fs_traj"],
                    "xyz": points["fs_xyz"],
                },
            ]
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["pathway", "point", "purpose", "mace_energy_ev", "traj", "xyz"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output}")
    print("DFT barrier correction requires IS and TS for each pathway; FS is optional.")


if __name__ == "__main__":
    main()

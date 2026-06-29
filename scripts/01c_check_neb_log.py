#!/usr/bin/env python3
"""Summarize ASE/FIRE NEB convergence from a log file."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


LINE_RE = re.compile(
    r"^FIRE:\s+(?P<step>\d+)\s+\S+\s+(?P<energy>[-+0-9.]+)\s+(?P<fmax>[-+0-9.]+)"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", nargs="?", default="results/C_sub/neb.log")
    parser.add_argument("--target-fmax", type=float, default=0.08)
    args = parser.parse_args()

    rows = []
    for line in Path(args.log).read_text().splitlines():
        match = LINE_RE.match(line.strip())
        if match:
            rows.append(
                {
                    "step": int(match.group("step")),
                    "energy": float(match.group("energy")),
                    "fmax": float(match.group("fmax")),
                }
            )
    if not rows:
        raise SystemExit(f"No FIRE rows found in {args.log}")

    first = rows[0]
    best = min(rows, key=lambda row: row["fmax"])
    last = rows[-1]
    recent = rows[-10:]
    recent_best = min(recent, key=lambda row: row["fmax"])
    print(f"log            : {args.log}")
    print(f"steps recorded : {len(rows)}")
    print(f"first fmax     : {first['fmax']:.6f} at step {first['step']}")
    print(f"best fmax      : {best['fmax']:.6f} at step {best['step']}")
    print(f"last fmax      : {last['fmax']:.6f} at step {last['step']}")
    print(f"recent best    : {recent_best['fmax']:.6f} over last {len(recent)} rows")
    print(f"target fmax    : {args.target_fmax:.6f}")
    if last["fmax"] <= args.target_fmax:
        print("status         : converged")
    elif best["fmax"] < last["fmax"] and last["fmax"] > 2.0 * args.target_fmax:
        print("status         : not converging cleanly; likely bad band/endpoints or needs restart")
    else:
        print("status         : still running or slowly improving")


if __name__ == "__main__":
    main()

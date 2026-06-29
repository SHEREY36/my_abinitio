#!/usr/bin/env python3
"""Build realistic starting structures for SiGeC C-sub/C-int NEB work."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from my_abinitio.io_utils import write_json
from my_abinitio.slab_builder import SlabBuildConfig, write_sige_c_endpoints


def parse_repetitions(text: str) -> tuple[int, int, int]:
    parts = text.lower().replace(",", "x").split("x")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected NxMxL, for example 4x4x6")
    try:
        values = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("repetitions must be integers") from exc
    if any(value < 1 for value in values):
        raise argparse.ArgumentTypeError("all repetitions must be positive")
    return values  # type: ignore[return-value]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build Si/SiGe surface endpoints for carbon surface, substitutional, "
            "and interstitial NEB calculations."
        )
    )
    parser.add_argument("--output-dir", default="structures")
    parser.add_argument("--repetitions", type=parse_repetitions, default="4x4x6")
    parser.add_argument("--ge-fraction", type=float, default=0.10)
    parser.add_argument("--ge-layers", type=int, default=2)
    parser.add_argument("--termination", choices=["H", "Cl", "none", "bare", "clean"], default="H")
    parser.add_argument("--termination-fraction", type=float, default=0.25)
    parser.add_argument(
        "--c-sub-layer-from-top",
        type=int,
        default=2,
        help="Layer receiving substitutional C: 1=top surface, 2=first subsurface layer.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--vacuum", type=float, default=14.0)
    parser.add_argument("--lattice-constant", type=float, default=5.431)
    args = parser.parse_args()

    if not 0 <= args.ge_fraction <= 1:
        raise SystemExit("--ge-fraction must be between 0 and 1")
    if not 0 <= args.termination_fraction <= 1:
        raise SystemExit("--termination-fraction must be between 0 and 1")

    cfg = SlabBuildConfig(
        repetitions=args.repetitions,
        lattice_constant_A=args.lattice_constant,
        vacuum_A=args.vacuum,
        ge_fraction=args.ge_fraction,
        ge_layers=args.ge_layers,
        termination=args.termination,
        termination_fraction=args.termination_fraction,
        c_sub_layer_from_top=args.c_sub_layer_from_top,
        seed=args.seed,
    )
    output_dir = Path(args.output_dir)
    metadata = write_sige_c_endpoints(output_dir, cfg)
    metadata_path = output_dir / "sige_surface_metadata.json"
    write_json(
        metadata_path,
        {
            "rationale": {
                "surface": "Si(001)-like growth front with top-layer Ge substitutions.",
                "carbon_initial_state": "C surface species after MMS decomposition, above an open top site.",
                "c_sub_final_state": "C occupies a selected subsurface lattice site; displaced Si/Ge becomes a surface adatom.",
                "c_int_final_state": "C moves to a tetrahedral-like subsurface interstitial starting guess.",
                "termination": "Optional H/Cl represents residual hydride/chloride surface passivation.",
            },
            "config": cfg.__dict__,
            "metadata": metadata,
        },
    )
    print(f"Wrote structures to {output_dir}")
    print(f"Wrote metadata to {metadata_path}")
    print("Visualize with:")
    print(f"  ase gui {output_dir / 'C_surface_IS.traj'} {output_dir / 'C_sub_FS.traj'} {output_dir / 'C_int_FS.traj'}")


if __name__ == "__main__":
    main()

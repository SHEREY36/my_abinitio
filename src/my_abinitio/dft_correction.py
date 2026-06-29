"""DFT correction utilities for MLIP/MACE NEB barriers."""

from __future__ import annotations


def corrected_barrier(
    mace_is_ev: float,
    mace_ts_ev: float,
    dft_is_ev: float,
    dft_ts_ev: float,
) -> float:
    """Return DFT-corrected barrier using DFT single-points on MACE images."""
    mace_barrier = mace_ts_ev - mace_is_ev
    correction = (dft_ts_ev - mace_ts_ev) - (dft_is_ev - mace_is_ev)
    return mace_barrier + correction


def correct_barrier_table(neb_summary: dict, dft_singlepoints: dict) -> dict:
    out = {"barriers": []}
    for row in neb_summary["barriers"]:
        name = row["name"]
        dft = dft_singlepoints[name]
        barrier = corrected_barrier(
            row["mace_is_ev"],
            row["mace_ts_ev"],
            dft["dft_is_ev"],
            dft["dft_ts_ev"],
        )
        out["barriers"].append(
            {
                "name": name,
                "reaction": row.get("reaction", name),
                "barrier_ev": barrier,
                "source": "DFT single-point corrected MACE NEB",
            }
        )
    return out

#!/usr/bin/env python3
"""Run the complete toy workflow end-to-end."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from my_abinitio.ace_writer import write_ace_surface_rates
from my_abinitio.dft_correction import correct_barrier_table
from my_abinitio.io_utils import write_json
from my_abinitio.kmc import gillespie_branching
from my_abinitio.neb import run_toy_neb
from my_abinitio.toy_potential import toy_reactions
from my_abinitio.tst import rates_from_barrier


def main() -> None:
    results = ROOT / "results"
    results.mkdir(exist_ok=True)

    neb_rows = []
    for name, potential in toy_reactions().items():
        res = run_toy_neb(name, potential, n_images=9)
        neb_rows.append(
            {
                "name": name,
                "reaction": name,
                "barrier_ev": res.barrier_ev,
                "mace_is_ev": res.energies_ev[0],
                "mace_ts_ev": res.energies_ev[res.ts_index],
                "ts_index": res.ts_index,
                "images": res.images,
                "energies_ev": res.energies_ev,
            }
        )
    neb_summary = {"barriers": neb_rows}
    write_json(results / "toy_neb.json", neb_summary)

    # Toy DFT correction: pretend DFT raises the interstitial barrier by 0.05 eV
    # and leaves the substitutional barrier unchanged.
    dft_singlepoints = {
        "C_sub": {"dft_is_ev": 0.0, "dft_ts_ev": neb_rows[0]["mace_ts_ev"]},
        "C_int": {"dft_is_ev": 0.0, "dft_ts_ev": neb_rows[1]["mace_ts_ev"] + 0.05},
    }
    write_json(results / "toy_qe_singlepoints.json", dft_singlepoints)
    corrected = correct_barrier_table(neb_summary, dft_singlepoints)
    write_json(results / "toy_dft_corrected.json", corrected)

    T_grid = [773.15, 798.15, 823.15, 848.15, 873.15, 898.15, 923.15]
    rate_table = {}
    for row in corrected["barriers"]:
        rate_table[row["name"]] = rates_from_barrier(row["name"], row["barrier_ev"], T_grid)
    write_json(results / "toy_arrhenius.json", rate_table)

    T_ref = 823.15
    # The fitted form is used here so the toy validates the exact ACE+ export.
    sub = rate_table["C_sub"]["modified_arrhenius"]
    cint = rate_table["C_int"]["modified_arrhenius"]
    import math

    k_sub = sub["A"] * (T_ref ** sub["n"]) * math.exp(-sub["E_over_R_K"] / T_ref)
    k_int = cint["A"] * (T_ref ** cint["n"]) * math.exp(-cint["E_over_R_K"] / T_ref)
    kmc = gillespie_branching(k_sub, k_int, k_bury=0.0, n_events=5000, seed=3)
    write_json(results / "toy_kmc.json", kmc)

    write_ace_surface_rates(results / "ace_surface_rates.c", rate_table)

    print("Toy pipeline complete")
    print(f"  C_sub barrier = {corrected['barriers'][0]['barrier_ev']:.3f} eV")
    print(f"  C_int barrier = {corrected['barriers'][1]['barrier_ev']:.3f} eV")
    print(f"  f_sub exact at 550 C = {kmc['f_sub_exact']:.4f}")
    print(f"  wrote {results}")


if __name__ == "__main__":
    main()

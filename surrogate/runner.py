"""Turn active-learning proposals into a CFD-ACE+ parametric run table.

Emits one row per run with the physical operating point, the derived inlet
boundary-condition values (mole fractions + mass flows), which modules to
enable for that fidelity, and the output filename to ingest afterwards.  Feed
this table to CFD-ACE+ parametric mode (or a scripted DTF-prep) and, after the
runs finish, export each solution to Tecplot/CSV named per `output_file`.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

# sccm -> mol/s at 0 C, 1 atm (matches CFD-ACE+ 'standard'); use 6.832e-7 for 25 C
SCCM_TO_MOL_S = 7.436e-7
M = {"H2": 2.01588e-3, "DCS": 101.007e-3, "HCl": 36.4607e-3}  # kg/mol


def inlet_bc(params: dict) -> dict:
    """Derived single-inlet BC values from an operating point."""
    dcs = params.get("main_dcs_sccm", 400.0)
    h2 = params.get("main_h2_sccm", 9000.0)
    hcl = params.get("hcl_sccm", 0.0)
    total = dcs + h2 + hcl
    mol = {"H2": h2, "DCS": dcs, "HCl": hcl}
    mdot = {k: mol[k] * SCCM_TO_MOL_S * M[k] for k in mol}          # kg/s
    mdot_tot = sum(mdot.values())
    return {
        "x_H2": h2 / total, "x_DCS": dcs / total, "x_HCl": hcl / total,
        "w_H2": mdot["H2"] / mdot_tot, "w_DCS": mdot["DCS"] / mdot_tot,
        "w_HCl": mdot["HCl"] / mdot_tot,
        "mdot_total_kg_s": mdot_tot,
    }


def modules_for(fidelity: str) -> str:
    return "Flow+Heat" if fidelity == "flow_heat" else "Flow+Heat+Chemistry"


def write_run_table(proposals, path, start_id: int = 0, out_dir: str = "exports"):
    """proposals: list of Proposal (or dicts with .params/.fidelity)."""
    rows = []
    manifest = {}
    for i, p in enumerate(proposals):
        params = p.params if hasattr(p, "params") else p["params"]
        fid = p.fidelity if hasattr(p, "fidelity") else p["fidelity"]
        rid = f"run_{start_id + i:03d}"
        bc = inlet_bc(params)
        rows.append({
            "run_id": rid, "fidelity": fid, "modules": modules_for(fid),
            **{k: round(v, 6) for k, v in params.items()},
            **{k: round(v, 8) for k, v in bc.items()},
            "output_file": f"{out_dir}/{rid}.dat",
        })
        manifest[rid] = {
            "fidelity": fid,
            "params": {k: rows[-1][k] for k in params},
            "output_file": rows[-1]["output_file"],
        }
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("")
        Path(str(p) + ".manifest.json").write_text(json.dumps({}, indent=2))
        return rows
    with p.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    Path(str(p) + ".manifest.json").write_text(json.dumps(manifest, indent=2))
    return rows

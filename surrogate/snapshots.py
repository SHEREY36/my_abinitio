"""Snapshot data model + ingestion + QoI extraction.

A *snapshot* is one converged CFD-ACE+ solution: the mesh node coordinates, a
dict of nodal field arrays, the operating point that produced it, and its
fidelity ('flow_heat' = cheap, no chemistry; 'full_chem' = expensive).

ACE+ writes binary .DTF.  DTF is proprietary, so the robust path is to *export*
each solution from CFD-VIEW (or the solver) to Tecplot-ASCII or CSV and load
that here.  `load_dtf` is left as a stub to wire the ESI DTF API if you have it;
`load_tecplot_ascii` and `load_csv` are the working loaders.

QoI extraction turns a wafer deposition field into the growth-rate profile
GR(r) and the scalars the whole study targets: GR(0), GR(edge), dGR, mean,
nonuniformity -- using area-weighted radial averaging (paper Fig. 2).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

RHO_SI = 2329.0  # kg/m^3

# canonical field names we expect from ACE+ (rename map your export to these)
FLOW_FIELDS = ("u", "v", "p", "T", "vel_mag")
CHEM_FIELDS = ("X_DCS", "X_HCl", "X_SiCl2", "X_H2")
SURFACE_FIELDS = ("Dep_Si_B", "cov_H_S", "cov_Cl_S")  # wafer-nodes only


@dataclass
class Snapshot:
    params: dict[str, float]          # operating point that produced this solution
    fidelity: str                     # 'flow_heat' or 'full_chem'
    coords: np.ndarray                # (N, 2) node coordinates [x, y] (meters)
    fields: dict[str, np.ndarray]     # name -> (N,) nodal values
    wafer_mask: np.ndarray            # (N,) bool, True on wafer surface nodes
    meta: dict[str, Any] = field(default_factory=dict)

    def field_names(self) -> list[str]:
        return list(self.fields.keys())


# ----------------------------------------------------------------------------- IO
def load_csv(path: str | Path, params: dict[str, float], fidelity: str,
             x_col="x", y_col="y", wafer_flag_col="is_wafer",
             field_cols: dict[str, str] | None = None) -> Snapshot:
    """Load one exported CSV (one row per node).  field_cols maps canonical
    name -> column name in the file; unspecified numeric columns are ignored."""
    import csv
    rows = list(csv.DictReader(open(path, newline="")))
    x = np.array([float(r[x_col]) for r in rows])
    y = np.array([float(r[y_col]) for r in rows])
    wafer = (np.array([float(r.get(wafer_flag_col, 0)) for r in rows]) > 0.5)
    fields = {}
    if field_cols:
        for canon, col in field_cols.items():
            if col in rows[0]:
                fields[canon] = np.array([float(r[col]) for r in rows])
    return Snapshot(params, fidelity, np.stack([x, y], 1), fields, wafer)


def load_tecplot_ascii(path: str | Path, params: dict[str, float], fidelity: str,
                       wafer_var: str | None = None) -> Snapshot:
    """Minimal Tecplot POINT-format ASCII reader (VARIABLES + one ZONE).
    Recognizes columns named X/Y and maps common ACE+ variable names."""
    text = Path(path).read_text().splitlines()
    varnames, data_start = [], 0
    for i, line in enumerate(text):
        u = line.upper()
        if u.strip().startswith("VARIABLES"):
            raw = line.split("=", 1)[1]
            varnames = [v.strip().strip('"') for v in raw.replace(",", " ").split()]
        if u.strip().startswith("ZONE"):
            data_start = i + 1
            break
    vals = []
    for line in text[data_start:]:
        parts = line.split()
        if not parts:
            continue
        try:
            vals.append([float(p) for p in parts])
        except ValueError:
            continue
    arr = np.array(vals)
    col = {v.upper(): arr[:, i] for i, v in enumerate(varnames)}
    x = col.get("X", arr[:, 0]); y = col.get("Y", arr[:, 1])
    rename = {"U": "u", "V": "v", "P": "p", "T": "T", "VELOCITYMAGNITUDE": "vel_mag",
              "DEP_SI(B)": "Dep_Si_B", "DEP_SI_B": "Dep_Si_B",
              "H(S)": "cov_H_S", "CL(S)": "cov_Cl_S"}
    fields = {rename.get(k, k): v for k, v in col.items() if k not in ("X", "Y")}
    if wafer_var and wafer_var.upper() in col:
        wafer = col[wafer_var.upper()] > 0.5
    else:
        wafer = np.zeros(len(x), bool)  # set later via geometry
    return Snapshot(params, fidelity, np.stack([x, y], 1), fields, wafer)


def load_dtf(path: str | Path, params: dict[str, float], fidelity: str) -> Snapshot:  # pragma: no cover
    """STUB. Wire the ESI DTF Python/C API here if available; otherwise export
    the .DTF to Tecplot/CSV from CFD-VIEW and use the loaders above."""
    raise NotImplementedError(
        "DTF is binary/proprietary. Export to Tecplot-ASCII or CSV from CFD-VIEW "
        "(File > Export) and use load_tecplot_ascii / load_csv.")


# ----------------------------------------------------------------------------- dataset
@dataclass
class SnapshotDataset:
    snapshots: list[Snapshot]

    def by_fidelity(self, fidelity: str) -> list[Snapshot]:
        return [s for s in self.snapshots if s.fidelity == fidelity]

    def with_field(self, name: str) -> list[Snapshot]:
        return [s for s in self.snapshots if name in s.fields]

    def param_matrix(self, space, snaps: list[Snapshot] | None = None) -> np.ndarray:
        snaps = snaps if snaps is not None else self.snapshots
        return np.concatenate([space.dict_to_array(s.params) for s in snaps], 0)

    def field_matrix(self, name: str) -> tuple[np.ndarray, list[Snapshot]]:
        """Stack a field across all snapshots that have it -> (M, N)."""
        snaps = self.with_field(name)
        if not snaps:
            raise KeyError(f"no snapshot has field {name!r}")
        N = snaps[0].fields[name].shape[0]
        for s in snaps:
            if s.fields[name].shape[0] != N:
                raise ValueError(f"field {name!r} has inconsistent node count "
                                 "(snapshots must share one parametric mesh; "
                                 "interpolate to a reference grid otherwise)")
        return np.stack([s.fields[name] for s in snaps], 0), snaps


# ----------------------------------------------------------------------------- QoI
def growth_profile_from_deposition(coords: np.ndarray, wafer_mask: np.ndarray,
                                   dep_si_b: np.ndarray, wafer_radius_m: float,
                                   axis_coord: int = 0, axis_origin: float = 0.0,
                                   n_bins: int = 30) -> dict[str, np.ndarray]:
    """Radial-average a wafer deposition field [kg/m^2/s] into GR(r) [nm/min].

    axis_coord: which coord column is the radial direction from the symmetry
    axis; axis_origin: the axis location in that coordinate.
    """
    r_all = np.abs(coords[:, axis_coord] - axis_origin)
    r = r_all[wafer_mask]
    gr = np.abs(dep_si_b[wafer_mask]) / RHO_SI * 1e9 * 60.0   # m/s -> nm/min
    edges = np.linspace(0.0, wafer_radius_m, n_bins + 1)
    centers, gr_r = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (r >= a) & (r < b)
        if m.any():
            centers.append(0.5 * (a + b))
            gr_r.append(gr[m].mean())
    return {"r": np.array(centers), "GR_r_nm_min": np.array(gr_r)}


def qois_from_profile(prof: dict[str, np.ndarray]) -> dict[str, float]:
    r, gr = prof["r"], prof["GR_r_nm_min"]
    if r.size == 0:
        return {}
    w = 2.0 * np.pi * r                      # axisymmetric area weight
    mean = float(np.trapezoid(gr * w, r) / np.trapezoid(w, r))
    return {
        "GR0_nm_min": float(gr[0]),
        "GRedge_nm_min": float(gr[-1]),
        "dGR_nm_min": float(gr[-1] - gr[0]),
        "GRmean_nm_min": mean,
        "nonuniformity_pct": float(100.0 * (gr.max() - gr.min()) / max(mean, 1e-30)),
    }

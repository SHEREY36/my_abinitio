"""Tier 2 -- full-field surrogate (POD + GP per field).

Maps the operating point theta to every ACE+ nodal field: flow/thermal
(u, v, p, T, vel_mag), gas species (X_DCS, X_HCl, ...), and wafer-surface
fields (Dep_Si_B, cov_H_S, cov_Cl_S).  Multi-fidelity by construction:

  * flow/thermal fields train on ALL runs (cheap flow_heat + full_chem),
  * chemistry/surface fields train on full_chem runs only.

Outputs are visualization-ready fields, a per-node uncertainty field (drives
active learning), parameter-sensitivity fields d(field)/d(theta), and scalar
QoIs recomputed from the *predicted* deposition field -- so a single surrogate
gives both the pictures you validate against ACE+ and the numbers you optimize.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dfield

import numpy as np

from .core import MultiGP, POD
from .paramspace import ParameterSpace
from .snapshots import (SnapshotDataset, growth_profile_from_deposition,
                        qois_from_profile, FLOW_FIELDS)


@dataclass
class FieldModel:
    pod: POD
    gp: MultiGP
    n_train: int
    fidelity_used: str


@dataclass
class FieldPrediction:
    values: np.ndarray        # (N,)
    std: np.ndarray           # (N,) per-node uncertainty
    name: str


class Tier2FieldSurrogate:
    def __init__(self, space: ParameterSpace, wafer_radius_m: float = 0.15):
        self.space = space
        self.wafer_radius_m = wafer_radius_m
        self.models: dict[str, FieldModel] = {}
        self.coords: np.ndarray | None = None
        self.wafer_mask: np.ndarray | None = None

    def fit(self, ds: SnapshotDataset, fields: list[str] | None = None,
            energy_tol: float = 0.9999) -> "Tier2FieldSurrogate":
        # reference mesh (parametric study shares one mesh)
        ref = ds.snapshots[0]
        self.coords, self.wafer_mask = ref.coords, ref.wafer_mask
        names = fields or sorted({n for s in ds.snapshots for n in s.fields})
        for name in names:
            F, snaps = ds.field_matrix(name)
            X = self.space.to_unit(ds.param_matrix(self.space, snaps))
            if F.shape[0] < 3:
                continue  # not enough runs for this field yet
            pod = POD.fit(F, energy_tol=energy_tol)
            gp = MultiGP().fit(X, pod.project(F))
            fid = "all" if name in FLOW_FIELDS else "full_chem"
            self.models[name] = FieldModel(pod, gp, F.shape[0], fid)
        return self

    # ---- field prediction ----
    def predict_field(self, params: dict, name: str) -> FieldPrediction:
        m = self.models[name]
        u = self.space.to_unit(self.space.dict_to_array(params))
        cm, cs = m.gp.predict(u)
        vals = m.pod.reconstruct(cm)[0]
        std = np.sqrt((cs[0] ** 2) @ (m.pod.modes.T ** 2))
        return FieldPrediction(vals, std, name)

    def predict_all(self, params: dict) -> dict[str, FieldPrediction]:
        return {n: self.predict_field(params, n) for n in self.models}

    def field_sensitivity(self, params: dict, name: str) -> np.ndarray:
        """d(field)/d(physical theta): (N, d)."""
        m = self.models[name]
        u = self.space.to_unit(self.space.dict_to_array(params))
        dc = m.gp.dmean_dx(u)[0]                      # (k, d) unit-coord
        phys = self.space.dict_to_array(params)[0]
        scale = np.array([
            1.0 / ((np.log10(p.high) - np.log10(p.low)) * phys[i] * np.log(10))
            if p.log else 1.0 / (p.high - p.low)
            for i, p in enumerate(self.space.params)])
        dc_phys = dc * scale                          # (k, d)
        return m.pod.modes @ dc_phys                  # (N, d)

    # ---- fields -> QoIs (close the loop) ----
    def qois(self, params: dict) -> dict[str, float]:
        if "Dep_Si_B" not in self.models:
            raise KeyError("deposition field not trained yet")
        dep = self.predict_field(params, "Dep_Si_B").values
        prof = growth_profile_from_deposition(
            self.coords, self.wafer_mask, dep, self.wafer_radius_m)
        return qois_from_profile(prof)

    # ---- reduced-space uncertainty (active learning signal) ----
    def field_uncertainty(self, U: np.ndarray, name: str) -> np.ndarray:
        """Energy-weighted coefficient std per candidate: (M,)."""
        m = self.models[name]
        _, cs = m.gp.predict(U)
        w = m.pod.singular_values / np.sum(m.pod.singular_values)
        return np.sqrt((cs ** 2) @ (w ** 2))

    def trained_fields(self) -> list[str]:
        return list(self.models)

"""Tier 1 -- multi-fidelity scalar / profile surrogate.

Low fidelity = the Tier 0 analytic backbone (cheap prior).  High fidelity =
ACE+ full-chemistry runs.  For each growth QoI we fit a GP to the *residual*
(HF - Tier0), so the surrogate inherits Tier 0's correct asymptotics and only
learns what Tier 0 misses (e.g. wafer rotation, multi-inlet transport).  The
GR(r) profile is handled the same way through a POD of residual profiles.

    corrected(theta) = Tier0(theta) + GP_residual(theta)     (with uncertainty)

`prior_fn(params_dict) -> {'r', 'GR_r_nm_min', 'qois'}` decouples this module
from the exact Tier 0 implementation; `make_tier0_prior` wires the shipped
tier0_backbone.py, and a synthetic prior can be injected for testing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .core import GP, MultiGP, POD
from .paramspace import ParameterSpace
from .snapshots import (SnapshotDataset, growth_profile_from_deposition,
                        qois_from_profile)

QOI_KEYS = ("GR0_nm_min", "GRedge_nm_min", "dGR_nm_min", "GRmean_nm_min", "nonuniformity_pct")
PriorFn = Callable[[dict], dict]


def make_tier0_prior(radial_grid_m: np.ndarray, wafer_radius_m: float = 0.15) -> PriorFn:
    """Adapter around the shipped tier0_backbone.evaluate_profile."""
    from . import tier0_backbone as t0

    def prior(params: dict) -> dict:
        proc = t0.ProcessConditions(
            wafer_temperature_C=params.get("wafer_temperature_C", 650.0),
            pressure_Torr=params.get("pressure_Torr", 300.0),
            main_h2_sccm=params.get("main_h2_sccm", 9000.0),
            main_dcs_sccm=params.get("main_dcs_sccm", 400.0),
            hcl_sccm=params.get("hcl_sccm", 0.0),
        )
        res = t0.evaluate_profile(process=proc, n_points=max(31, radial_grid_m.size))
        r = np.array([p.r_over_R for p in res.profile]) * wafer_radius_m
        gr = np.array([p.growth_nm_min for p in res.profile])
        gr_on_grid = np.interp(radial_grid_m, r, gr)
        qois = {
            "GR0_nm_min": float(gr_on_grid[0]),
            "GRedge_nm_min": float(gr_on_grid[-1]),
            "dGR_nm_min": float(gr_on_grid[-1] - gr_on_grid[0]),
            "GRmean_nm_min": float(np.trapezoid(gr_on_grid * 2 * np.pi * radial_grid_m, radial_grid_m)
                                   / np.trapezoid(2 * np.pi * radial_grid_m + 1e-30, radial_grid_m)),
            "nonuniformity_pct": float(100 * (gr_on_grid.max() - gr_on_grid.min())
                                       / max(gr_on_grid.mean(), 1e-30)),
        }
        return {"r": radial_grid_m, "GR_r_nm_min": gr_on_grid, "qois": qois}

    return prior


@dataclass
class Tier1Prediction:
    qois: dict[str, float]
    qois_std: dict[str, float]
    r: np.ndarray
    GR_r_nm_min: np.ndarray
    GR_r_std: np.ndarray


class Tier1Surrogate:
    def __init__(self, space: ParameterSpace, prior_fn: PriorFn,
                 radial_grid_m: np.ndarray, wafer_radius_m: float = 0.15):
        self.space = space
        self.prior_fn = prior_fn
        self.r = np.asarray(radial_grid_m)
        self.wafer_radius_m = wafer_radius_m
        self.qoi_gp: dict[str, GP] = {}
        self.prof_pod: POD | None = None
        self.prof_gp: MultiGP | None = None

    # ---- training ----
    def _extract_hf(self, ds: SnapshotDataset):
        X, dqoi, dprof = [], {k: [] for k in QOI_KEYS}, []
        for s in ds.by_fidelity("full_chem"):
            if "Dep_Si_B" not in s.fields:
                continue
            prof = growth_profile_from_deposition(
                s.coords, s.wafer_mask, s.fields["Dep_Si_B"], self.wafer_radius_m)
            gr_on_grid = np.interp(self.r, prof["r"], prof["GR_r_nm_min"])
            qoi = qois_from_profile({"r": self.r, "GR_r_nm_min": gr_on_grid})
            pri = self.prior_fn(s.params)
            X.append(self.space.to_unit(self.space.dict_to_array(s.params))[0])
            for k in QOI_KEYS:
                dqoi[k].append(qoi[k] - pri["qois"][k])       # residual
            dprof.append(gr_on_grid - pri["GR_r_nm_min"])     # profile residual
        return np.array(X), {k: np.array(v) for k, v in dqoi.items()}, np.array(dprof)

    def fit(self, ds: SnapshotDataset) -> "Tier1Surrogate":
        X, dqoi, dprof = self._extract_hf(ds)
        if len(X) < 3:
            raise ValueError("need >=3 full-chemistry runs to fit Tier 1")
        for k in QOI_KEYS:
            self.qoi_gp[k] = GP().fit(X, dqoi[k])
        self.prof_pod = POD.fit(dprof, energy_tol=0.999)
        self.prof_gp = MultiGP().fit(X, self.prof_pod.project(dprof))
        return self

    # ---- prediction ----
    def predict(self, params: dict) -> Tier1Prediction:
        u = self.space.to_unit(self.space.dict_to_array(params))
        pri = self.prior_fn(params)
        qois, stds = {}, {}
        for k in QOI_KEYS:
            m, s = self.qoi_gp[k].predict(u)
            qois[k] = float(pri["qois"][k] + m[0]); stds[k] = float(s[0])
        cm, cs = self.prof_gp.predict(u)
        prof = pri["GR_r_nm_min"] + self.prof_pod.reconstruct(cm)[0]
        prof_std = np.sqrt((cs[0] ** 2) @ (self.prof_pod.modes.T ** 2))
        return Tier1Prediction(qois, stds, self.r, prof, prof_std)

    def predict_unit(self, U: np.ndarray) -> tuple[dict, dict]:
        """Vectorized QoI mean/std over normalized points (for acquisition)."""
        U = np.atleast_2d(U)
        phys = self.space.from_unit(U)
        priors = [self.prior_fn(d) for d in self.space.array_to_dicts(phys)]
        mean, std = {}, {}
        for k in QOI_KEYS:
            m, s = self.qoi_gp[k].predict(U)
            mean[k] = np.array([p["qois"][k] for p in priors]) + m
            std[k] = s
        return mean, std

    def sensitivity(self, params: dict) -> dict[str, np.ndarray]:
        """d(QoI)/d(physical param) at a point (chain rule through the cube)."""
        u = self.space.to_unit(self.space.dict_to_array(params))
        phys = self.space.dict_to_array(params)[0]
        out = {}
        for k in QOI_KEYS:
            g_unit = self.qoi_gp[k].dmean_dx(u)[0]
            scale = np.array([  # d(unit)/d(phys) for each param
                1.0 / ((np.log10(p.high) - np.log10(p.low)) * phys[i] * np.log(10))
                if p.log else 1.0 / (p.high - p.low)
                for i, p in enumerate(self.space.params)])
            out[k] = g_unit * scale
        return out

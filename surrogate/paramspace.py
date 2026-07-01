"""Operating-parameter space for the RP-CVD Si-epitaxy surrogate.

Defines the design variables (the operating vector theta), their bounds and
which are best sampled on a log scale, and provides normalized <-> physical
transforms plus space-filling samplers (Sobol / LHS).  Everything downstream
(GP inputs, acquisition search, ACE+ run tables) works in the *normalized*
[0, 1]^d cube and converts to physical units only at the boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.stats import qmc


@dataclass(frozen=True)
class Param:
    name: str
    low: float
    high: float
    log: bool = False      # sample/interpolate in log10 space
    unit: str = ""

    def to_unit(self, x_phys: np.ndarray) -> np.ndarray:
        lo, hi = (np.log10(self.low), np.log10(self.high)) if self.log else (self.low, self.high)
        v = np.log10(x_phys) if self.log else x_phys
        return (v - lo) / (hi - lo)

    def from_unit(self, u: np.ndarray) -> np.ndarray:
        lo, hi = (np.log10(self.low), np.log10(self.high)) if self.log else (self.low, self.high)
        v = lo + u * (hi - lo)
        return np.power(10.0, v) if self.log else v


@dataclass(frozen=True)
class ParameterSpace:
    params: Sequence[Param]

    @property
    def dim(self) -> int:
        return len(self.params)

    @property
    def names(self) -> list[str]:
        return [p.name for p in self.params]

    # ---- transforms (physical dict/array <-> normalized cube) ----
    def to_unit(self, X_phys: np.ndarray) -> np.ndarray:
        X_phys = np.atleast_2d(X_phys)
        return np.stack([p.to_unit(X_phys[:, i]) for i, p in enumerate(self.params)], axis=1)

    def from_unit(self, U: np.ndarray) -> np.ndarray:
        U = np.atleast_2d(U)
        return np.stack([p.from_unit(U[:, i]) for i, p in enumerate(self.params)], axis=1)

    def dict_to_array(self, d: dict) -> np.ndarray:
        return np.array([[d[p.name] for p in self.params]], dtype=float)

    def array_to_dicts(self, X_phys: np.ndarray) -> list[dict]:
        X_phys = np.atleast_2d(X_phys)
        return [{p.name: float(row[i]) for i, p in enumerate(self.params)} for row in X_phys]

    # ---- samplers (return PHYSICAL points) ----
    def sobol(self, n: int, seed: int | None = 0) -> np.ndarray:
        U = qmc.Sobol(d=self.dim, scramble=True, seed=seed).random(n)
        return self.from_unit(U)

    def lhs(self, n: int, seed: int | None = 0) -> np.ndarray:
        U = qmc.LatinHypercube(d=self.dim, seed=seed).random(n)
        return self.from_unit(U)

    def candidate_pool(self, n: int, seed: int = 12345) -> np.ndarray:
        """Dense pool the acquisition function scans over (normalized cube)."""
        return qmc.Sobol(d=self.dim, scramble=True, seed=seed).random(n)


def jackel_si_space() -> ParameterSpace:
    """Operating box for the Jaeckel et al. low-T DCS Si case.

    Chosen around the paper's set point (300 Torr, <700 C, ~9000 sccm H2 main,
    ~400 sccm DCS).  Flows are log-sampled; rotation is included because it
    reshapes the flow field (paper's most effective uniformity knob).
    """
    return ParameterSpace([
        Param("wafer_temperature_C", 580.0, 700.0, unit="C"),
        Param("pressure_Torr", 50.0, 300.0, unit="Torr"),
        Param("main_dcs_sccm", 150.0, 800.0, log=True, unit="sccm"),
        Param("main_h2_sccm", 6000.0, 14000.0, log=True, unit="sccm"),
        Param("hcl_sccm", 1.0, 300.0, log=True, unit="sccm"),
        Param("rotation_rpm", 0.0, 120.0, unit="rpm"),
    ])

"""Reduced-order core: POD field reduction + a thin GP wrapper.

POD (via SVD) compresses each (M snapshots x N nodes) field matrix to a few
spatial modes and per-snapshot coefficients.  A GP then learns
operating-point -> coefficients.  The GP wrapper standardizes I/O, returns
predictive std (needed by active learning), and exposes d mean / d input
(needed for parameter-sensitivity fields and gradient-based optimization).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel


# ----------------------------------------------------------------------------- POD
@dataclass
class POD:
    mean: np.ndarray          # (N,)
    modes: np.ndarray         # (N, k)
    singular_values: np.ndarray
    energy: np.ndarray        # cumulative energy fraction per mode

    @classmethod
    def fit(cls, F: np.ndarray, n_modes: int | None = None, energy_tol: float = 0.9999) -> "POD":
        """F: (M, N) snapshots x nodes."""
        mean = F.mean(0)
        A = F - mean
        U, S, Vt = np.linalg.svd(A, full_matrices=False)   # A = U S Vt ; rows of Vt are spatial modes
        cum = np.cumsum(S**2) / max(np.sum(S**2), 1e-30)
        k = n_modes or int(np.searchsorted(cum, energy_tol) + 1)
        k = max(1, min(k, Vt.shape[0]))
        return cls(mean, Vt[:k].T, S[:k], cum[:k])

    def project(self, F: np.ndarray) -> np.ndarray:
        """(M, N) -> (M, k) coefficients."""
        return (np.atleast_2d(F) - self.mean) @ self.modes

    def reconstruct(self, C: np.ndarray) -> np.ndarray:
        """(M, k) -> (M, N) fields."""
        return self.mean + np.atleast_2d(C) @ self.modes.T

    @property
    def n_modes(self) -> int:
        return self.modes.shape[1]


# ----------------------------------------------------------------------------- GP
class GP:
    """Single-output GP on normalized inputs with standardized targets."""

    def __init__(self, length_scale=0.4, nu=2.5, noise=1e-6, seed=0):
        kernel = (ConstantKernel(1.0, (1e-3, 1e3))
                  * Matern(length_scale=length_scale * np.ones(1), nu=nu,
                           length_scale_bounds=(1e-2, 1e4))
                  + WhiteKernel(noise, (1e-10, 1e-1)))
        self._proto = kernel
        self.gp = None
        self.y_mean = 0.0
        self.y_std = 1.0
        self.seed = seed

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GP":
        X = np.atleast_2d(X)
        d = X.shape[1]
        kernel = (ConstantKernel(1.0, (1e-3, 1e3))
                  * Matern(length_scale=0.4 * np.ones(d), nu=2.5,
                           length_scale_bounds=(1e-2, 1e4))
                  + WhiteKernel(1e-6, (1e-12, 1e-1)))
        self.y_mean = float(np.mean(y)); self.y_std = float(np.std(y) or 1.0)
        yy = (y - self.y_mean) / self.y_std
        self.gp = GaussianProcessRegressor(kernel=kernel, normalize_y=False,
                                           n_restarts_optimizer=4,
                                           random_state=self.seed).fit(X, yy)
        return self

    def predict(self, X: np.ndarray, return_std=True):
        X = np.atleast_2d(X)
        m, s = self.gp.predict(X, return_std=True)
        m = m * self.y_std + self.y_mean
        s = s * self.y_std
        return (m, s) if return_std else m

    def dmean_dx(self, X: np.ndarray, eps=1e-4) -> np.ndarray:
        """Finite-difference d(mean)/d(input) in normalized coords: (M, d)."""
        X = np.atleast_2d(X); d = X.shape[1]
        g = np.zeros_like(X)
        for j in range(d):
            Xp = X.copy(); Xm = X.copy()
            Xp[:, j] += eps; Xm[:, j] -= eps
            g[:, j] = (self.predict(Xp, False) - self.predict(Xm, False)) / (2 * eps)
        return g


class MultiGP:
    """Independent GP per column of a coefficient matrix (POD-GP)."""

    def __init__(self, **kw):
        self.kw = kw
        self.gps: list[GP] = []

    def fit(self, X: np.ndarray, C: np.ndarray) -> "MultiGP":
        C = np.atleast_2d(C)
        self.gps = [GP(**self.kw).fit(X, C[:, k]) for k in range(C.shape[1])]
        return self

    def predict(self, X: np.ndarray, return_std=True):
        outs = [g.predict(X, True) for g in self.gps]
        M = np.stack([o[0] for o in outs], 1)
        S = np.stack([o[1] for o in outs], 1)
        return (M, S) if return_std else M

    def dmean_dx(self, X: np.ndarray) -> np.ndarray:
        """(M, k, d) coefficient sensitivities."""
        return np.stack([g.dmean_dx(X) for g in self.gps], 1)

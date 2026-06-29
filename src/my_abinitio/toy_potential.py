"""Analytic toy potentials for local validation.

The potential is a 2D double well:

    E(x, y) = barrier * (1 - x^2)^2 + 0.5 * k_y * y^2

Initial state: x=-1, y=0
Final state:   x=+1, y=0
Transition:    x=0,  y=0

The exact barrier is therefore `barrier`.
"""

from __future__ import annotations


class DoubleWell2D:
    def __init__(self, barrier_ev: float, k_y: float = 5.0):
        self.barrier_ev = float(barrier_ev)
        self.k_y = float(k_y)

    def energy(self, point: list[float]) -> float:
        x, y = point
        return self.barrier_ev * (1.0 - x * x) ** 2 + 0.5 * self.k_y * y * y

    def gradient(self, point: list[float]) -> list[float]:
        x, y = point
        d_x = -4.0 * self.barrier_ev * x * (1.0 - x * x)
        d_y = self.k_y * y
        return [d_x, d_y]


def toy_reactions() -> dict[str, DoubleWell2D]:
    """Return two toy pathways with known exact barriers."""
    return {
        "C_sub": DoubleWell2D(1.20),
        "C_int": DoubleWell2D(0.85),
    }

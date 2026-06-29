"""Transition-state theory and ACE+ Arrhenius fitting."""

from __future__ import annotations

import math

from .constants import KB_EV_PER_K, KB_OVER_H_PER_K_S, ev_to_kelvin


def eyring_rate(T_K: float, barrier_ev: float, entropy_factor: float = 1.0) -> float:
    """Harmonic-TST/Eyring rate with optional entropy prefactor multiplier."""
    return entropy_factor * KB_OVER_H_PER_K_S * T_K * math.exp(-barrier_ev / (KB_EV_PER_K * T_K))


def solve_3x3(a: list[list[float]], b: list[float]) -> list[float]:
    """Gaussian elimination for a 3x3 linear system."""
    m = [row[:] + [rhs] for row, rhs in zip(a, b)]
    n = 3
    for i in range(n):
        pivot = max(range(i, n), key=lambda r: abs(m[r][i]))
        m[i], m[pivot] = m[pivot], m[i]
        div = m[i][i]
        if abs(div) < 1e-20:
            raise ValueError("Singular normal equation")
        for j in range(i, n + 1):
            m[i][j] /= div
        for r in range(n):
            if r == i:
                continue
            factor = m[r][i]
            for j in range(i, n + 1):
                m[r][j] -= factor * m[i][j]
    return [m[i][n] for i in range(n)]


def fit_modified_arrhenius(T_K: list[float], rates_s: list[float]) -> dict[str, float]:
    """Fit ln k = ln A + n ln T - (E/R)/T."""
    rows = [[1.0, math.log(T), -1.0 / T] for T in T_K]
    y = [math.log(k) for k in rates_s]
    xtx = [[0.0] * 3 for _ in range(3)]
    xty = [0.0] * 3
    for row, yi in zip(rows, y):
        for i in range(3):
            xty[i] += row[i] * yi
            for j in range(3):
                xtx[i][j] += row[i] * row[j]
    ln_a, n, e_over_r = solve_3x3(xtx, xty)
    return {"A": math.exp(ln_a), "n": n, "E_over_R_K": e_over_r}


def rates_from_barrier(
    name: str,
    barrier_ev: float,
    temperature_grid_K: list[float],
    entropy_factor: float = 1.0,
) -> dict:
    rates = [eyring_rate(T, barrier_ev, entropy_factor) for T in temperature_grid_K]
    fit = fit_modified_arrhenius(temperature_grid_K, rates)
    return {
        "name": name,
        "barrier_ev": barrier_ev,
        "barrier_E_over_R_K": ev_to_kelvin(barrier_ev),
        "temperature_grid_K": temperature_grid_K,
        "rates_s^-1": rates,
        "modified_arrhenius": fit,
    }

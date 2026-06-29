"""Minimal dependency-free NEB-style path evaluator for toy validation.

For real atomistic NEB, use scripts/01_mace_neb_paths.py with ASE/MACE.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class NEBResult:
    name: str
    barrier_ev: float
    ts_index: int
    images: list[list[float]]
    energies_ev: list[float]


def interpolate(start: list[float], end: list[float], n_images: int) -> list[list[float]]:
    total = n_images + 2
    images = []
    for i in range(total):
        f = i / (total - 1)
        images.append([(1.0 - f) * a + f * b for a, b in zip(start, end)])
    return images


def norm(vec: list[float]) -> float:
    return math.sqrt(sum(v * v for v in vec))


def run_toy_neb(
    name: str,
    potential,
    start: list[float] | None = None,
    end: list[float] | None = None,
    n_images: int = 9,
    steps: int = 0,
    step_size: float = 0.02,
) -> NEBResult:
    """Run a tiny NEB-like relaxation and return the maximum image barrier.

    The default toy path is already the exact MEP, so `steps=0` validates the
    barrier extraction cleanly.  A few gradient steps can be used if a perturbed
    path is supplied.
    """
    start = start or [-1.0, 0.0]
    end = end or [1.0, 0.0]
    images = interpolate(start, end, n_images)

    for _ in range(steps):
        new_images = [images[0]]
        for point in images[1:-1]:
            grad = potential.gradient(point)
            new_images.append([p - step_size * g for p, g in zip(point, grad)])
        new_images.append(images[-1])
        images = new_images

    energies = [potential.energy(p) for p in images]
    e0 = energies[0]
    rel = [e - e0 for e in energies]
    ts_index = max(range(len(rel)), key=lambda i: rel[i])
    return NEBResult(
        name=name,
        barrier_ev=rel[ts_index],
        ts_index=ts_index,
        images=images,
        energies_ev=energies,
    )

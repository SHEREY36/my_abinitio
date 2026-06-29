"""Small Gillespie/kMC tools for Csub/Cint branching validation."""

from __future__ import annotations

import math
import random


def branching_fraction(k_sub: float, k_int: float, k_bury: float = 0.0) -> float:
    r_sub = max(k_sub, 0.0) + max(k_bury, 0.0)
    r_int = max(k_int, 0.0)
    denom = r_sub + r_int
    return 0.0 if denom <= 0.0 else r_sub / denom


def gillespie_branching(
    k_sub: float,
    k_int: float,
    k_bury: float = 0.0,
    n_events: int = 10000,
    seed: int = 7,
) -> dict[str, float]:
    rng = random.Random(seed)
    p_sub = branching_fraction(k_sub, k_int, k_bury)
    sub = 0
    total_time = 0.0
    total_rate = k_sub + k_int + k_bury
    for _ in range(n_events):
        if rng.random() < p_sub:
            sub += 1
        if total_rate > 0.0:
            total_time += -math.log(max(rng.random(), 1e-16)) / total_rate
    return {
        "n_events": n_events,
        "substitutional_events": sub,
        "interstitial_events": n_events - sub,
        "f_sub_kmc": sub / n_events,
        "f_sub_exact": p_sub,
        "mean_event_time_s": total_time / max(n_events, 1),
    }

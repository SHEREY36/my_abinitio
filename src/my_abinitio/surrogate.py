"""Tiny uncertainty-ranking helper for active learning."""

from __future__ import annotations


def select_by_uncertainty(candidates: list[dict], k: int = 5) -> list[dict]:
    """Select candidates with largest ensemble standard deviation."""
    scored = []
    for row in candidates:
        preds = [float(x) for x in row.get("barrier_predictions_ev", [])]
        if len(preds) < 2:
            std = 0.0
        else:
            mean = sum(preds) / len(preds)
            std = (sum((x - mean) ** 2 for x in preds) / (len(preds) - 1)) ** 0.5
        out = dict(row)
        out["uncertainty_ev"] = std
        scored.append(out)
    return sorted(scored, key=lambda r: r["uncertainty_ev"], reverse=True)[:k]

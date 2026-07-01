"""Active learning -- decide *which* runs to spend, and at *which* fidelity.

Two drivers, combined per candidate and normalized by ACE+ cost:

  * exploit: push the growth/uniformity Pareto front  (ParEGO MC-EI on Tier 1 QoIs)
  * explore: cut field-surrogate uncertainty          (POD-coeff std from Tier 2)

Cost model: flow_heat runs are cheap and yield only flow fields; full_chem runs
are expensive but yield everything.  So a candidate's flow-field uncertainty is
scored against the cheap cost, while its chemistry/deposition uncertainty and
all optimization value are scored against the expensive cost.  We pick, per
candidate, the fidelity with the higher value-per-cost, then greedily build a
diverse batch.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .paramspace import ParameterSpace
from .snapshots import SnapshotDataset
from .tier1 import Tier1Surrogate
from .tier2 import Tier2FieldSurrogate
from .snapshots import FLOW_FIELDS

DEFAULT_COST = {"flow_heat": 1.0, "full_chem": 12.0}   # relative wall-clock


# ----------------------------------------------------------------------------- acquisition
def parego_mc_ei(qoi_mean: dict, qoi_std: dict, observed: dict,
                 rng: np.random.Generator, n_weight=8, n_mc=64, rho=0.05) -> np.ndarray:
    """ParEGO Monte-Carlo Expected Improvement over (maximize GRmean, minimize
    nonuniformity_pct).  Returns (M,) acquisition per candidate."""
    o1_m, o1_s = qoi_mean["GRmean_nm_min"], qoi_std["GRmean_nm_min"]
    o2_m, o2_s = qoi_mean["nonuniformity_pct"], qoi_std["nonuniformity_pct"]
    # normalization ranges from observed data (fallback to candidate spread)
    o1_lo, o1_hi = observed["GRmean_min"], observed["GRmean_max"]
    o2_lo, o2_hi = observed["nonunif_min"], observed["nonunif_max"]
    def n1(x): return (x - o1_lo) / max(o1_hi - o1_lo, 1e-9)                 # higher better
    def n2(x): return 1.0 - (x - o2_lo) / max(o2_hi - o2_lo, 1e-9)          # lower better
    M = o1_m.shape[0]
    acq = np.zeros(M)
    for _ in range(n_weight):
        w = rng.random(2); w /= w.sum()
        # current best scalar over observed points
        so = np.minimum(w[0] * n1(observed["GRmean_obs"]), w[1] * n2(observed["nonunif_obs"])) \
            + rho * (w[0] * n1(observed["GRmean_obs"]) + w[1] * n2(observed["nonunif_obs"]))
        s_best = so.max()
        # MC samples of candidate objectives
        z1 = o1_m[:, None] + o1_s[:, None] * rng.standard_normal((M, n_mc))
        z2 = o2_m[:, None] + o2_s[:, None] * rng.standard_normal((M, n_mc))
        s = np.minimum(w[0] * n1(z1), w[1] * n2(z2)) \
            + rho * (w[0] * n1(z1) + w[1] * n2(z2))
        acq += np.maximum(s - s_best, 0.0).mean(1)
    return acq / n_weight


def observed_stats(t1: Tier1Surrogate, ds: SnapshotDataset) -> dict:
    from .snapshots import growth_profile_from_deposition, qois_from_profile
    grm, nu = [], []
    for s in ds.by_fidelity("full_chem"):
        if "Dep_Si_B" not in s.fields:
            continue
        prof = growth_profile_from_deposition(s.coords, s.wafer_mask,
                                              s.fields["Dep_Si_B"], t1.wafer_radius_m)
        q = qois_from_profile({"r": t1.r,
                               "GR_r_nm_min": np.interp(t1.r, prof["r"], prof["GR_r_nm_min"])})
        grm.append(q["GRmean_nm_min"]); nu.append(q["nonuniformity_pct"])
    grm, nu = np.array(grm), np.array(nu)
    return {"GRmean_obs": grm, "nonunif_obs": nu,
            "GRmean_min": grm.min(), "GRmean_max": grm.max(),
            "nonunif_min": nu.min(), "nonunif_max": nu.max()}


# ----------------------------------------------------------------------------- selection
@dataclass
class Proposal:
    params: dict
    fidelity: str
    score: float
    reason: str


def _greedy(U, scores, k, taken, sigma=0.15):
    """Pick k diverse maxima of `scores`, avoiding indices already in `taken`."""
    picks, s = [], scores.copy()
    s[list(taken)] = -np.inf
    for _ in range(k):
        idx = int(np.argmax(s))
        if not np.isfinite(s[idx]):
            break
        picks.append(idx); taken.add(idx)
        s = s - (s.max() + 1.0) * 0.0  # no-op keep dtype
        d2 = np.sum((U - U[idx]) ** 2, 1)
        s = s * (1.0 - np.exp(-d2 / (2 * sigma ** 2)))
        s[list(taken)] = -np.inf
    return picks


def propose_batch(space: ParameterSpace, ds: SnapshotDataset,
                  t1: Tier1Surrogate, t2: Tier2FieldSurrogate,
                  n_exploit=2, n_chem=2, n_flow=2, pool_size=2000,
                  cost=DEFAULT_COST, seed=0) -> list[Proposal]:
    """Explicit role-quota batch:
      * n_exploit  full_chem runs at top ParEGO-EI  (push growth/uniformity Pareto)
      * n_chem     full_chem runs at top deposition/species uncertainty
      * n_flow     flow_heat runs at top flow-field uncertainty  (cheap coverage)
    Quotas make the cost trade-off explicit and prevent the cheap fidelity from
    starving out the chemistry runs that carry all the deposition truth."""
    rng = np.random.default_rng(seed)
    U = space.candidate_pool(pool_size, seed=seed)

    qm, qs = t1.predict_unit(U)
    ei = parego_mc_ei(qm, qs, observed_stats(t1, ds), rng)

    chem_names = [n for n in t2.trained_fields() if n not in FLOW_FIELDS]
    flow_names = [n for n in t2.trained_fields() if n in FLOW_FIELDS]
    chem_unc = np.mean([t2.field_uncertainty(U, n) for n in chem_names], 0) if chem_names else np.zeros(len(U))
    flow_unc = np.mean([t2.field_uncertainty(U, n) for n in flow_names], 0) if flow_names else np.zeros(len(U))

    taken: set[int] = set()
    out: list[Proposal] = []
    def emit(idxs, fid, score, reason):
        for i in idxs:
            out.append(Proposal(space.array_to_dicts(space.from_unit(U[i:i+1]))[0],
                                fid, float(score[i]), reason))
    emit(_greedy(U, ei, n_exploit, taken), "full_chem", ei, "exploit-pareto")
    emit(_greedy(U, chem_unc, n_chem, taken), "full_chem", chem_unc, "explore-chem")
    emit(_greedy(U, flow_unc, n_flow, taken), "flow_heat", flow_unc, "explore-flow")
    total_cost = sum(cost[p.fidelity] for p in out)
    for p in out:
        p.__dict__["batch_cost"] = total_cost
    return out


# ----------------------------------------------------------------------------- one AL step
def active_learning_step(space, ds, prior_fn, radial_grid, **kw):
    """Fit Tier 1 + Tier 2 on current data and propose the next ACE+ batch."""
    t1 = Tier1Surrogate(space, prior_fn, radial_grid).fit(ds)
    t2 = Tier2FieldSurrogate(space).fit(ds)
    batch = propose_batch(space, ds, t1, t2, **kw)
    return t1, t2, batch

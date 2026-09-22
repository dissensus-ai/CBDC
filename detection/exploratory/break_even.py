"""Gain-threshold crossing: the smallest lambda at which the identity block
reduces misses by more than a threshold tau. Pure functions, no I/O.

(Called "break-even" in the first build. Renamed: nothing breaks even here --
tau is a reference gain level, not a cost-benefit balance point. The module
keeps its file name for now.)

Input is a replicate x grid matrix D[r, j] = Delta missed-per-10k (T2 minus
T4, or T2 minus T3; positive = identity helps) for replicate r at lambda_j,
NaN where that replicate failed. Two estimands; the addendum (E10, A0) fixes:

crossing_mean  PRIMARY. First lambda at which mean Delta exceeds tau. Estimates
               a property of the generator; carries no "demonstrably" claim.
               Its uncertainty is the bootstrap band.

crossing_LB    secondary. First lambda at which the lower endpoint of the
               two-sided (1-alpha) Student-t interval for mean Delta exceeds
               tau. DEPENDS ON R: more replicates tighten the interval and move
               crossing_LB down toward crossing_mean. Reported, not headlined.

Both interpolate linearly between the bracketing grid points. Censoring is
reported, never filled: "left_censored" if the curve already exceeds tau at
the first grid point (crossing <= lambda_min; value lambda_min), "not_reached"
if it never exceeds tau within the grid (value NaN). "recrossed" flags a curve
that drops back to <= tau after its first crossing.

Uncertainty: a replicate bootstrap resampling replicate ROWS (all grid points
of a replicate move together, preserving the lambda pairing). The interval is
CENSORING-AWARE and computed over ALL B draws: a not-reached draw is
right-censored at +infinity and a left-censored draw sits at lambda_min. A
limit that falls in censored mass is reported as a bound with a label
(">lambda_max (not reached within tested range)" / "<=lambda_min (already
above tau at the lowest tested lambda)"), never as a number. P(no crossing in
range) = n_not_reached / B is reported. The finite-draws-only interval is kept
as "conditional_on_crossing" -- it answers a different question (where does it
cross, given that it crosses in range) and must not be read as the band.
Why: the first build used finite draws only; on a toy where 1,107/2,000 draws
never crossed it reported a finite ~0.88-0.98 band, i.e. a conditional interval
dressed as the unconditional one.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

__all__ = ["curve_stats", "first_crossing", "gain_threshold_crossing",
           "bootstrap_gain_threshold_crossing", "RIGHT_CENSORED_LABEL",
           "LEFT_CENSORED_LABEL"]

RIGHT_CENSORED_LABEL = ">lambda_max (not reached within tested range)"
LEFT_CENSORED_LABEL = "<=lambda_min (already above tau at lowest tested lambda)"


def curve_stats(D, alpha=0.10):
    """Per-grid-point mean, sd, n, and two-sided (1-alpha) t interval."""
    D = np.asarray(D, dtype=float)
    if D.ndim != 2:
        raise ValueError("D must be a 2-D replicate x grid matrix")
    G = D.shape[1]
    mean = np.full(G, np.nan)
    lo = np.full(G, np.nan)
    hi = np.full(G, np.nan)
    sd = np.full(G, np.nan)
    n = np.zeros(G, dtype=int)
    for j in range(G):
        x = D[:, j][~np.isnan(D[:, j])]
        n[j] = len(x)
        if len(x) == 0:
            continue
        mean[j] = x.mean()
        if len(x) < 2:
            continue
        sd[j] = x.std(ddof=1)
        half = stats.t.ppf(1 - alpha / 2, df=len(x) - 1) * sd[j] / math.sqrt(len(x))
        lo[j], hi[j] = mean[j] - half, mean[j] + half
    return {"mean": mean, "sd": sd, "n": n, "ci_lo": lo, "ci_hi": hi}


def first_crossing(grid, curve, tau):
    """Smallest grid-interpolated x at which curve first exceeds tau.

    A NaN curve point is treated as not exceeding tau (an undetermined bound
    is not evidence), and interpolation is only across a bracketing pair of
    finite points; a crossing adjacent to a NaN is placed at the grid point.
    Returns (value, flag) with flag in {None, "left_censored", "not_reached"}
    plus a separate recrossed bool.
    """
    grid = np.asarray(grid, dtype=float)
    curve = np.asarray(curve, dtype=float)
    if grid.shape != curve.shape or grid.ndim != 1 or len(grid) == 0:
        raise ValueError("grid and curve must be equal-length 1-D arrays")
    if np.any(np.diff(grid) <= 0):
        raise ValueError("grid must be strictly increasing")
    above = np.where(np.isnan(curve), False, curve > tau)
    if not above.any():
        return math.nan, "not_reached", False
    j = int(np.argmax(above))
    recrossed = bool((~above[j:]).any())
    if j == 0:
        return float(grid[0]), "left_censored", recrossed
    c0, c1 = curve[j - 1], curve[j]
    if np.isnan(c0):
        return float(grid[j]), None, recrossed
    x = grid[j - 1] + (tau - c0) * (grid[j] - grid[j - 1]) / (c1 - c0)
    return float(x), None, recrossed


def gain_threshold_crossing(grid, D, tau, alpha=0.10):
    """Both crossing estimands on one replicate x grid matrix."""
    cs = curve_stats(D, alpha)
    out = {"tau": tau, "alpha": alpha, "primary": "crossing_mean",
           "secondary": "crossing_LB"}
    for name, curve in (("mean", cs["mean"]), ("LB", cs["ci_lo"])):
        v, flag, rec = first_crossing(grid, curve, tau)
        out[f"crossing_{name}"] = v
        out[f"flag_{name}"] = flag
        out[f"recrossed_{name}"] = rec
    out["curve"] = {k: v.tolist() for k, v in cs.items()}
    return out


def _limit(v, lam_min):
    """One bootstrap limit: a number, or a labelled censored bound."""
    if math.isinf(v):
        return {"value": None, "censored": "right", "label": RIGHT_CENSORED_LABEL}
    if v == lam_min:
        return {"value": None, "censored": "left", "label": LEFT_CENSORED_LABEL}
    return {"value": float(v), "censored": None, "label": None}


def bootstrap_gain_threshold_crossing(grid, D, tau, alpha=0.10, B=2000, seed=0,
                                      level=0.90):
    """Censoring-aware replicate-row bootstrap band for both estimands.

    Percentiles use method="inverted_cdf", so each limit is an actual draw and
    never an interpolation between a finite value and +infinity. A limit equal
    to lambda_min is labelled left-censored: a draw can only sit exactly at
    lambda_min by being left-censored or by crossing exactly there, and both
    mean "crossing <= lambda_min".
    """
    grid = np.asarray(grid, dtype=float)
    D = np.asarray(D, dtype=float)
    rng = np.random.default_rng(seed)
    R = D.shape[0]
    draws = {"mean": [], "LB": []}
    flags = {"mean": {"left_censored": 0, "not_reached": 0},
             "LB": {"left_censored": 0, "not_reached": 0}}
    for _ in range(B):
        idx = rng.integers(0, R, size=R)
        cs = curve_stats(D[idx], alpha)
        for name, curve in (("mean", cs["mean"]), ("LB", cs["ci_lo"])):
            v, flag, _ = first_crossing(grid, curve, tau)
            if flag:
                flags[name][flag] += 1
            draws[name].append(math.inf if flag == "not_reached" else v)
    q = ((1 - level) / 2, 1 - (1 - level) / 2)
    out = {"B": B, "seed": seed, "level": level,
           "method": "censoring-aware percentile over all draws"}
    for name in ("mean", "LB"):
        x = np.asarray(draws[name], dtype=float)
        fin = x[np.isfinite(x)]
        lo, hi = (float(np.quantile(x, qq, method="inverted_cdf")) for qq in q)
        out[name] = {
            "lo": _limit(lo, grid[0]),
            "hi": _limit(hi, grid[0]),
            "p_no_crossing_in_range": flags[name]["not_reached"] / B,
            "p_left_censored": flags[name]["left_censored"] / B,
            "n_not_reached": flags[name]["not_reached"],
            "n_left_censored": flags[name]["left_censored"],
            "conditional_on_crossing": {
                "note": "finite draws only; conditional on a crossing within "
                        "the tested range -- NOT the band",
                "n": int(len(fin)),
                "lo": (float(np.quantile(fin, q[0], method="inverted_cdf"))
                       if len(fin) else None),
                "hi": (float(np.quantile(fin, q[1], method="inverted_cdf"))
                       if len(fin) else None),
            },
        }
    return out

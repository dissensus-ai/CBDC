"""Break-even identity signal: the smallest lambda at which the identity block
demonstrably reduces misses by more than a threshold. Pure functions, no I/O.

Input is a replicate x grid matrix D[r, j] = Delta missed-per-10k (T2 minus
T4, or T2 minus T3; positive = identity helps) for replicate r at lambda_j,
NaN where that replicate failed. Two estimands are computed; which one a
reported run uses is for the addendum to fix, not for this code.

lambda*_LB   first lambda at which the lower endpoint of the two-sided
             (1-alpha) Student-t interval for mean Delta exceeds tau --
             "demonstrably above tau". This DEPENDS ON R: with more
             replicates the interval tightens and lambda*_LB moves down
             toward lambda*_mean. It is a detection threshold for a given
             design, not a property of the generator alone.

lambda*_mean first lambda at which mean Delta itself exceeds tau. Estimates a
             generator property; carries no "demonstrably" claim.

Both use linear interpolation between the bracketing grid points on the
relevant curve (lower bound or mean). Censoring is reported, never filled:
"left_censored" if the curve is already above tau at the first grid point
(lambda* <= lambda_0; value lambda_0), "not_reached" if it never exceeds tau
(value NaN). "recrossed" flags a curve that drops back to <= tau after its first
crossing, where "smallest lambda" and "the lambda beyond which" differ.

Uncertainty for lambda*: a replicate bootstrap that resamples replicate ROWS
(all grid points of a replicate move together), recomputes the estimand, and
reports percentile limits plus how many draws were censored. Row resampling
keeps whatever dependence exists across lambda; under the current DGP same-seed
worlds at different lambda share labels and wallet counts but not transactions
(see identity_signal), so that dependence is weak. With a bootstrap on
lambda*_LB the interval is re-derived inside each draw, so its band describes
the estimator at this R, consistent with the R-dependence above.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

__all__ = ["curve_stats", "first_crossing", "break_even", "bootstrap_break_even"]


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


def break_even(grid, D, tau, alpha=0.10):
    """Both estimands on one replicate x grid matrix."""
    cs = curve_stats(D, alpha)
    out = {"tau": tau, "alpha": alpha}
    for name, curve in (("LB", cs["ci_lo"]), ("mean", cs["mean"])):
        v, flag, rec = first_crossing(grid, curve, tau)
        out[f"lambda_star_{name}"] = v
        out[f"flag_{name}"] = flag
        out[f"recrossed_{name}"] = rec
    out["curve"] = {k: v.tolist() for k, v in cs.items()}
    return out


def bootstrap_break_even(grid, D, tau, alpha=0.10, B=2000, seed=0,
                         level=0.90):
    """Replicate-row bootstrap band for both estimands.

    Percentile limits are computed over draws with a finite value; the count
    of censored draws is reported alongside, because a band that silently
    drops half its draws is not the band it looks like. Left-censored draws
    enter at lambda_0 (their value) and are also counted.
    """
    D = np.asarray(D, dtype=float)
    rng = np.random.default_rng(seed)
    R = D.shape[0]
    draws = {"LB": [], "mean": []}
    flags = {"LB": {"left_censored": 0, "not_reached": 0},
             "mean": {"left_censored": 0, "not_reached": 0}}
    for _ in range(B):
        idx = rng.integers(0, R, size=R)
        cs = curve_stats(D[idx], alpha)
        for name, curve in (("LB", cs["ci_lo"]), ("mean", cs["mean"])):
            v, flag, _ = first_crossing(grid, curve, tau)
            if flag:
                flags[name][flag] += 1
            draws[name].append(v)
    q = ((1 - level) / 2, 1 - (1 - level) / 2)
    out = {"B": B, "seed": seed, "level": level}
    for name in ("LB", "mean"):
        x = np.asarray(draws[name], dtype=float)
        fin = x[~np.isnan(x)]
        out[name] = {
            "lo": float(np.quantile(fin, q[0])) if len(fin) else math.nan,
            "hi": float(np.quantile(fin, q[1])) if len(fin) else math.nan,
            "n_finite": int(len(fin)),
            **{f"n_{k}": v for k, v in flags[name].items()},
        }
    return out

"""Arm A: two-stage screening. EXPLORATORY -- no reported result comes from here
until a protocol addendum fixes seeds and grids.

Question: how much of the single-stage T2 -> T4 reduction in missed illicit
entities survives if identity attributes are disclosed for only a shortlist?

    stage 1  rank ALL test entities by the T2 score, shortlist the top K'
    stage 2  re-rank ONLY the shortlist by the T4 (or T3) score, review top k

K' is the disclosure budget: identity attributes are looked up for K' entities
per 10,000 instead of all of them. Both stages reuse `endpoint.select_alerts`
and its lower-id tiebreak, so two exact identities hold and are tested:

    K' = k  ->  the reviewed set IS the single-stage T2 top-k set
    K' = N  ->  the reviewed set IS the single-stage T4 top-k set

STAGE-2 TRAINING ASSUMPTION (default: "full"). The stage-2 scores are the ones
`run_replicate` already computed: a T4 model fit on the WHOLE training world,
i.e. identity attributes observed for every training entity. The disclosure
count therefore covers operational (test-time) lookups only; historical
identity access for model training is assumed and not counted. Default because
it holds the model fixed across the single-stage and two-stage arms, so the
gap-recovered fraction changes only the disclosure pattern at scoring time.

Variant "shortlist": stage-2 models are refit on only the training entities
stage 1 would have shortlisted on the training world (same K' rate per 10k),
which is the regime where identity is never collected outside a shortlist,
historically included. Stage-1 training scores for that shortlist come from
entity 5-fold out-of-fold T2 scores ("oof", default) or the in-sample T2 fit
("insample"). OOF is the default because in-sample boosted scores memorize the
training world: in the 22 Sep dev smoke (n_train=2000) the in-sample gboost
shortlist at K'*=500 was 100/100 positives in every replicate, so there was no
negative class to fit stage 2 on, and at larger K' the training shortlist is
far richer in positives than any test-time shortlist. Neither training variant
is a bound on the other.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "confirmatory"))

from endpoint import budget_for, select_alerts  # noqa: E402

DEFAULT_KPRIME_GRID = (500, 750, 1000, 1500, 2000, 3000, 5000, 10000)
STAGE2_TIERS = ("T4", "T3")
TRAIN_VARIANTS = ("full", "shortlist")
GAP_EPS = 1e-9


def shortlist_size(kprime_star: int, n: int, k: int) -> int:
    """K' on a population of n from a rate per 10,000; must be >= k.

    Uses budget_for (round to nearest) and clips at n, so K'*=10000 is the whole
    population on any n.
    """
    kp = min(n, budget_for(kprime_star, n))
    if kp < k:
        raise ValueError(f"shortlist K'={kp} (K'*={kprime_star}) is below the "
                         f"review budget k={k}")
    return kp


def two_stage_select(score_s1, score_s2, tiebreak, k: int, kprime: int):
    """Indices (into the full population) of the k reviewed entities.

    Returns (reviewed, shortlist). Stage 2 sees only shortlist members and uses
    the same tiebreak keys they carry in the full population.
    """
    score_s1 = np.asarray(score_s1, dtype=float)
    score_s2 = np.asarray(score_s2, dtype=float)
    tiebreak = np.asarray(tiebreak)
    if not (k <= kprime <= len(score_s1)):
        raise ValueError(f"need k <= K' <= n, got k={k} K'={kprime} "
                         f"n={len(score_s1)}")
    shortlist = select_alerts(score_s1, tiebreak, kprime)
    inner = select_alerts(score_s2[shortlist], tiebreak[shortlist], k)
    return shortlist[inner], shortlist


def missed_per_10k_from_set(y, reviewed) -> float:
    """Same quantity as endpoint.missed_per_10k, for an explicit reviewed set."""
    y = np.asarray(y)
    return 10_000.0 * (int(y.sum()) - int(y[reviewed].sum())) / len(y)


def gap_recovered(miss_t2: float, miss_two_stage: float, miss_hi: float):
    """(miss_T2 - miss_2s) / (miss_T2 - miss_hi), with flags.

    Returns (value, flag). flag is None, "gap_zero" (|denominator| < GAP_EPS,
    value NaN: there is no single-stage gap to recover), or "gap_negative"
    (the single-stage hi tier missed MORE than T2; value computed but its sign
    no longer reads as "fraction recovered").
    """
    denom = miss_t2 - miss_hi
    if abs(denom) < GAP_EPS:
        return math.nan, "gap_zero"
    val = (miss_t2 - miss_two_stage) / denom
    return val, ("gap_negative" if denom < 0 else None)


def _oof_t2_scores(model_name, x, y, seed):
    """Entity 5-fold OOF T2 scores on the training world (variant "oof")."""
    from sklearn.model_selection import StratifiedKFold
    from detection_experiment import N_FOLDS, _models

    out = np.empty(len(y), dtype=float)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for tr_idx, te_idx in skf.split(x, y):
        m = _models(seed)[model_name]
        m.fit(x[tr_idx], y[tr_idx])
        out[te_idx] = m.predict_proba(x[te_idx])[:, 1]
    return out


def make_hook(kprime_grid=DEFAULT_KPRIME_GRID, stage2_tiers=STAGE2_TIERS,
              train_variants=("full",), stage1_train_scores="oof"):
    """score_hook for replicate.run_replicate: writes rec["two_stage"].

    One row per (model, stage2_tier, train_variant, K'*). Every row carries
    the single-stage T2 and hi-tier misses it is compared against, so the
    record is self-contained.
    """
    from detection_experiment import _models
    from features import TIER_COLS
    from replicate import model_seed

    for v in train_variants:
        if v not in TRAIN_VARIANTS:
            raise ValueError(f"unknown train variant {v!r}")
    if stage1_train_scores not in ("insample", "oof"):
        raise ValueError("stage1_train_scores must be 'insample' or 'oof'")

    def hook(rec, ctx):
        y, tb, k = ctx["y_test"], ctx["tiebreak"], ctx["k"]
        scores, fitted = ctx["scores"], ctx["fitted"]
        n = len(y)
        rows, fails = [], []
        x_tr_t2 = ctx["train_ef"][TIER_COLS["T2"]].to_numpy(dtype=float)
        y_tr = ctx["y_train"]
        n_tr = len(y_tr)
        mseed = model_seed(ctx["train_seed"])
        s1_train = {}
        for model_name in rec["models"]:
            miss_t2 = rec["models"][model_name]["T2"]["MissedPer10k"]
            for tier in stage2_tiers:
                miss_hi = rec["models"][model_name][tier]["MissedPer10k"]
                x_tr_hi = ctx["train_ef"][TIER_COLS[tier]].to_numpy(dtype=float)
                x_te_hi = ctx["test_ef"][TIER_COLS[tier]].to_numpy(dtype=float)
                for variant in train_variants:
                    for kps in kprime_grid:
                        kp = shortlist_size(kps, n, k)
                        s2 = scores[(model_name, tier)]
                        n_fit, n_pos_fit = n_tr, int(y_tr.sum())
                        if variant == "shortlist":
                            if model_name not in s1_train:
                                if stage1_train_scores == "insample":
                                    s1_train[model_name] = fitted[
                                        (model_name, "T2")].predict_proba(
                                        x_tr_t2)[:, 1]
                                else:
                                    s1_train[model_name] = _oof_t2_scores(
                                        model_name, x_tr_t2, y_tr, mseed)
                            kp_tr = min(n_tr, budget_for(kps, n_tr))
                            sl_tr = np.sort(select_alerts(
                                s1_train[model_name], ctx["train_tiebreak"],
                                kp_tr))
                            n_fit = len(sl_tr)
                            n_pos_fit = int(y_tr[sl_tr].sum())
                            if n_pos_fit in (0, n_fit):
                                fails.append({
                                    "model": model_name, "stage2_tier": tier,
                                    "kprime_star": kps,
                                    "error": f"shortlist train labels "
                                             f"{n_pos_fit}/{n_fit}"})
                                continue
                            if kp_tr == n_tr:
                                # whole training world: identical fit
                                s2 = scores[(model_name, tier)]
                            else:
                                m = _models(mseed)[model_name]
                                m.fit(x_tr_hi[sl_tr], y_tr[sl_tr])
                                s2 = m.predict_proba(x_te_hi)[:, 1]
                        reviewed, _ = two_stage_select(
                            scores[(model_name, "T2")], s2, tb, k, kp)
                        miss = missed_per_10k_from_set(y, reviewed)
                        gap, flag = gap_recovered(miss_t2, miss, miss_hi)
                        rows.append({
                            "model": model_name,
                            "stage2_tier": tier,
                            "train_variant": variant,
                            "kprime_star": int(kps),
                            "kprime": int(kp),
                            "k": int(k),
                            "n_fit_stage2": int(n_fit),
                            "n_pos_fit_stage2": n_pos_fit,
                            "TP": int(y[reviewed].sum()),
                            "MissedPer10k": miss,
                            "DisclosedPer10k": 10_000.0 * kp / n,
                            "miss_T2": miss_t2,
                            "miss_hi": miss_hi,
                            "gap_recovered": gap,
                            "gap_flag": flag,
                        })
        rec["two_stage"] = rows
        rec["two_stage_config"] = {
            "kprime_grid": [int(x) for x in kprime_grid],
            "stage2_tiers": list(stage2_tiers),
            "train_variants": list(train_variants),
            "stage1_train_scores": stage1_train_scores,
            "default_train_variant": "full",
        }
        if fails:
            rec["two_stage_failures"] = fails

    return hook


NOT_APPLICABLE = "not_applicable"


def recovery_summary(reps, grid, qs=(0.5, 0.9), B=2000, seed=0, level=0.90,
                     alpha=0.10):
    """Reference gain, recovered-share curve and K'_q -- a SECONDARY annotation.

    The E9 primary output is absolute (missed per 10k vs test-time identity
    lookups per 10k, built by the driver). A "recovered share" only means
    something when single-stage identity access actually reduces misses, so
    everything here is gated on the reference gain G = miss_T2 - miss_hi:

      mean G <= 0          every share and K'_q = "not_applicable",
                           reason "nonpositive_reference_gain" (full identity
                           does not help, or makes misses worse: there is no
                           gain to recover, and the ratio would read nonsense)
      t-interval of G      flag "reference_gain_not_separated"; the ratio and
      includes 0           its bootstrap band are kept but role is
                           "secondary_annotation"

    `reps`: replicate id -> {K'*: (miss_T2, miss_2s, miss_hi)}; only
    replicates carrying every grid point enter. Share at K'* = ratio of
    replicate means (A0 primary estimator):

        (mean miss_T2 - mean miss_2s) / mean G

    Bootstrap: whole replicates resampled across all K'*. A draw whose
    resampled mean G <= 0 is NOT APPLICABLE; such draws are counted
    (n_not_applicable), never dropped silently. Two bands are reported:
      band_all_draws     percentile over ALL B draws with not-applicable draws
                         entering at -infinity ("no recoverable gain"); a limit
                         in that mass is reported as None + label. K'_q_LB uses
                         this lower limit.
      band_conditional   applicable draws only, labelled
                         conditional_on_positive_reference_gain.
    K'_q = smallest grid K'* whose share >= q (point curve); K'_q_LB the same
    on the all-draws lower limit (R-dependent). None = no grid point reaches q.
    """
    from inference import mean_ci

    grid = [int(g) for g in grid]
    ids = sorted(r for r, d in reps.items() if all(g in d for g in grid))
    out = {"n_replicates": len(ids), "role": "secondary_annotation",
           "estimator": "ratio_of_replicate_means"}
    if not ids:
        out.update(status=NOT_APPLICABLE, reason="no_complete_replicates")
        return out
    arr = np.array([[reps[r][g] for g in grid] for r in ids], dtype=float)
    gain = arr[:, 0, 0] - arr[:, 0, 2]       # identical at every K'*
    ref = mean_ci(gain.tolist(), alpha)
    out["reference_gain"] = ref
    na_all = {"curve": NOT_APPLICABLE, "mean_of_ratios": NOT_APPLICABLE,
              "K_q": {f"{q:g}": {"mean": NOT_APPLICABLE, "LB": NOT_APPLICABLE}
                      for q in qs}}
    if not ref["mean"] > GAP_EPS:
        out.update(status=NOT_APPLICABLE, reason="nonpositive_reference_gain",
                   **na_all)
        return out
    flags = []
    if ref.get("ci_lo") is None or ref["ci_lo"] <= 0:
        flags.append("reference_gain_not_separated")
    out.update(status="computed", flags=flags)

    def share(a):
        m = a.mean(axis=0)                       # (G, 3)
        den = m[0, 0] - m[0, 2]
        if not den > GAP_EPS:
            return None                          # not applicable in this draw
        return (m[:, 0] - m[:, 1]) / den

    point = share(arr)
    rng = np.random.default_rng(seed)
    draws, n_na = [], 0
    for _ in range(B):
        sh = share(arr[rng.integers(0, len(ids), len(ids))])
        if sh is None:
            n_na += 1
            sh = np.full(len(grid), -np.inf)
        draws.append(sh)
    draws = np.array(draws).reshape(B, len(grid))
    a = (1 - level) / 2

    def q_all(qq):
        return np.quantile(draws, qq, axis=0, method="inverted_cdf")

    appl = draws[np.isfinite(draws[:, 0])] if B else draws
    lo_all, hi_all = (q_all(a), q_all(1 - a)) if B else (None, None)
    lo_c = (np.quantile(appl, a, axis=0, method="inverted_cdf")
            if len(appl) else None)
    hi_c = (np.quantile(appl, 1 - a, axis=0, method="inverted_cdf")
            if len(appl) else None)

    def lim(v):
        if v is None:
            return None
        return None if np.isinf(v) else float(v)

    def first(vals, q):
        for g, v in zip(grid, vals):
            if np.isfinite(v) and v >= q:
                return g
        return None

    per_rep = [(g_r, (arr[i, :, 0] - arr[i, :, 1]) / g_r)
               for i, g_r in enumerate(gain) if g_r > GAP_EPS]
    mor = [mean_ci([r[1][j] for r in per_rep], alpha)
           for j in range(len(grid))]
    out.update({
        "bootstrap": {"B": B, "seed": seed, "level": level,
                      "n_not_applicable": n_na,
                      "p_not_applicable": n_na / B if B else None,
                      "not_applicable_label": "resampled mean reference gain "
                                              "<= 0 (no recoverable gain)"},
        "curve": [{
            "kprime_star": g, "share": float(point[j]),
            "band_all_draws": {
                "lo": lim(lo_all[j]) if B else None,
                "hi": lim(hi_all[j]) if B else None,
                "lo_label": ("in not-applicable mass"
                             if B and np.isinf(lo_all[j]) else None)},
            "band_conditional_on_positive_reference_gain": {
                "lo": lim(lo_c[j]) if lo_c is not None else None,
                "hi": lim(hi_c[j]) if hi_c is not None else None,
                "n": int(len(appl))},
        } for j, g in enumerate(grid)],
        "mean_of_ratios": {
            "role": "secondary to ratio_of_means",
            "n_replicates_used": len(per_rep),
            "n_excluded_nonpositive_replicate_gain": len(ids) - len(per_rep),
            "per_kprime": [dict(kprime_star=g, **mor[j])
                           for j, g in enumerate(grid)]},
        "K_q": {f"{q:g}": {"mean": first(point, q),
                           "LB": first(lo_all, q) if B else None}
                for q in qs},
    })
    return out

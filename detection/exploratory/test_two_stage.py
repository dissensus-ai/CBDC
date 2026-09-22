"""Arm A tests: the two exact identities, gap guard, and the replicate hook.

K'=k must reproduce single-stage T2 and K'=N single-stage T4 EXACTLY -- same
reviewed set, not just the same count -- including under score ties, because
both stages share endpoint.select_alerts and its lower-id tiebreak.
"""

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "confirmatory"))

from endpoint import missed_per_10k, select_alerts  # noqa: E402
from two_stage import (gap_recovered, make_hook, recovery_summary,  # noqa: E402
                       missed_per_10k_from_set, shortlist_size,
                       two_stage_select)


def _synthetic(n=400, seed=1, ties=True):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.08).astype(int)
    s2 = rng.random(n) + 0.8 * y
    s4 = rng.random(n) + 1.5 * y
    if ties:   # coarse rounding forces many exact ties at both stages
        s2, s4 = np.round(s2, 1), np.round(s4, 1)
    ids = rng.permutation(n) + 17
    return y, s2, s4, ids


def test_kprime_equals_k_is_single_stage_t2():
    for ties in (False, True):
        y, s2, s4, ids = _synthetic(ties=ties)
        k = 20
        reviewed, shortlist = two_stage_select(s2, s4, ids, k, k)
        assert set(reviewed) == set(select_alerts(s2, ids, k))
        assert set(shortlist) == set(reviewed)
        assert (missed_per_10k_from_set(y, reviewed)
                == missed_per_10k(y, s2, ids, k))


def test_kprime_equals_n_is_single_stage_t4():
    for ties in (False, True):
        y, s2, s4, ids = _synthetic(ties=ties)
        k, n = 20, len(y)
        reviewed, _ = two_stage_select(s2, s4, ids, k, n)
        assert set(reviewed) == set(select_alerts(s4, ids, k))
        assert (missed_per_10k_from_set(y, reviewed)
                == missed_per_10k(y, s4, ids, k))


def test_reviewed_is_subset_of_shortlist_and_size_k():
    y, s2, s4, ids = _synthetic()
    reviewed, shortlist = two_stage_select(s2, s4, ids, 20, 60)
    assert len(reviewed) == 20 and len(set(reviewed)) == 20
    assert set(reviewed) <= set(shortlist) and len(shortlist) == 60


def test_kprime_bounds_enforced():
    y, s2, s4, ids = _synthetic()
    for bad in (19, len(y) + 1):
        try:
            two_stage_select(s2, s4, ids, 20, bad)
        except ValueError:
            continue
        raise AssertionError(f"K'={bad} should raise")
    try:
        shortlist_size(400, 10_000, 500)
    except ValueError:
        pass
    else:
        raise AssertionError("K'* below k* should raise")
    assert shortlist_size(10_000, 7_777, 1) == 7_777   # clips to n
    assert shortlist_size(500, 10_000, 500) == 500


def test_gap_recovered_guard_and_flags():
    v, f = gap_recovered(10.0, 10.0, 10.0)
    assert math.isnan(v) and f == "gap_zero"
    assert gap_recovered(10.0, 6.0, 2.0) == (0.5, None)
    assert gap_recovered(10.0, 10.0, 2.0) == (0.0, None)
    assert gap_recovered(10.0, 2.0, 2.0) == (1.0, None)
    v, f = gap_recovered(2.0, 3.0, 4.0)
    assert f == "gap_negative" and v == 0.5


def _hooked_replicate(**hook_kw):
    from replicate import run_replicate
    return run_replicate(0, 700_001, 700_001 + 10_000_003, n_train=1200,
                         n_test=1500, k_star=500,
                         score_hook=make_hook(**hook_kw))


def test_hook_identities_on_real_replicate():
    rec = _hooked_replicate(kprime_grid=(500, 1000, 10_000),
                            train_variants=("full", "shortlist"))
    assert rec["status"] == "OK", rec.get("error")
    rows = rec["two_stage"]
    assert rows, "hook wrote no rows"
    for row in rows:
        if row["kprime_star"] == 500 and row["train_variant"] == "full":
            assert row["MissedPer10k"] == row["miss_T2"], row
        if row["kprime_star"] == 10_000:
            assert row["kprime"] == 1500
            assert row["MissedPer10k"] == row["miss_hi"], row
            assert row["DisclosedPer10k"] == 10_000.0
    # K'=k: the shortlist is the review set whatever stage 2 was trained on
    for row in rows:
        if row["kprime_star"] == 500:
            assert row["MissedPer10k"] == row["miss_T2"], row


def test_hooks_are_inert():
    """score_hook / cfg_factory must not change the confirmatory record."""
    from dgp import default_config
    from replicate import run_replicate
    kw = dict(n_train=900, n_test=1000, k_star=500, k_star_grid=[500, 1000])
    base = run_replicate(3, 700_010, 700_010 + 10_000_003, **kw)
    hooked = run_replicate(3, 700_010, 700_010 + 10_000_003,
                           score_hook=make_hook(kprime_grid=(500,)),
                           cfg_factory=lambda s, n: default_config(s), **kw)
    for key in ("two_stage", "two_stage_config"):
        hooked.pop(key)
    assert base == hooked


def test_full_refit_matches_fitted_model():
    """Justifies the hook's shortcut: a shortlist covering the whole training
    world refits on identical rows, so it reuses the full-world scores."""
    from detection_experiment import _models
    rng = np.random.default_rng(5)
    x = rng.normal(size=(300, 4))
    y = (x[:, 0] + rng.normal(size=300) > 1.2).astype(int)
    for name in ("gboost", "logit"):
        a = _models(7)[name].fit(x, y).predict_proba(x)[:, 1]
        idx = np.sort(np.arange(300))
        b = _models(7)[name].fit(x[idx], y[idx]).predict_proba(x)[:, 1]
        assert np.array_equal(a, b)


GRID4 = [500, 1000, 2000, 10000]


def _reps(miss_t2, miss_hi, miss_2s_by_k, R=6, jitter=0.0, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for r in range(R):
        e = rng.normal(scale=jitter) if jitter else 0.0
        out[r] = {k: (miss_t2 + e, m2s + e, miss_hi + e)
                  for k, m2s in miss_2s_by_k.items()}
    return out


def test_recovery_known_curve():
    # miss_T2 = 20, miss_hi = 10 in every replicate; ratio-of-means shares
    # 0, 0.4, 0.95, 1.0
    reps = _reps(20.0, 10.0, {500: 20.0, 1000: 16.0, 2000: 10.5, 10000: 10.0})
    out = recovery_summary(reps, GRID4, B=200, seed=1)
    assert out["status"] == "computed" and out["flags"] == []
    assert out["role"] == "secondary_annotation"
    assert [c["share"] for c in out["curve"]] == [0.0, 0.4, 0.95, 1.0]
    assert out["K_q"]["0.5"]["mean"] == 2000
    assert out["K_q"]["0.9"]["mean"] == 2000
    assert out["K_q"]["0.5"]["LB"] == 2000        # no variation: band = point
    assert out["bootstrap"]["n_not_applicable"] == 0
    # a replicate missing a grid point is excluded from every point
    reps[99] = {500: (20.0, 20.0, 10.0)}
    assert recovery_summary(reps, GRID4, B=10)["n_replicates"] == 6


def test_recovery_not_applicable_when_identity_increases_misses():
    """Toy from review: misses go 10 -> 20 under full identity. There is no
    gain to recover; no share and no K'_q may be reported."""
    reps = _reps(10.0, 20.0, {500: 10.0, 1000: 12.0, 2000: 15.0, 10000: 20.0})
    out = recovery_summary(reps, GRID4, B=100, seed=1)
    assert out["status"] == "not_applicable"
    assert out["reason"] == "nonpositive_reference_gain"
    assert out["reference_gain"]["mean"] == -10.0
    assert out["curve"] == "not_applicable"
    assert out["mean_of_ratios"] == "not_applicable"
    assert out["K_q"] == {"0.5": {"mean": "not_applicable",
                                  "LB": "not_applicable"},
                          "0.9": {"mean": "not_applicable",
                                  "LB": "not_applicable"}}
    # zero gain is not applicable either
    z = recovery_summary(_reps(10.0, 10.0, {g: 10.0 for g in GRID4}), GRID4,
                         B=10)
    assert z["status"] == "not_applicable"


def test_recovery_not_separated_is_flagged_and_bootstrap_counts_na():
    # mean gain slightly positive, noisy across replicates: the t-interval of
    # the gain includes 0 and some bootstrap draws have nonpositive gain
    rng = np.random.default_rng(4)
    reps = {}
    for r in range(8):
        g = 0.5 + rng.normal(scale=3.0)
        reps[r] = {k: (20.0, 20.0 - f * g, 20.0 - g)
                   for k, f in zip(GRID4, (0.0, 0.3, 0.8, 1.0))}
    out = recovery_summary(reps, GRID4, B=500, seed=2)
    ref = out["reference_gain"]
    if ref["mean"] <= 0:       # guard the fixture itself
        raise AssertionError("fixture should have positive mean gain")
    assert ref["ci_lo"] <= 0 <= ref["ci_hi"]
    assert out["status"] == "computed"
    assert out["flags"] == ["reference_gain_not_separated"]
    na = out["bootstrap"]["n_not_applicable"]
    assert 0 < na < 500
    assert out["bootstrap"]["p_not_applicable"] == na / 500
    c = out["curve"][-1]
    assert c["band_conditional_on_positive_reference_gain"]["n"] == 500 - na
    # when not-applicable draws exceed the lower tail, the all-draws lower
    # limit is in that mass: None + label, and K'_q_LB cannot be reached
    if na > 0.05 * 500:
        assert c["band_all_draws"]["lo"] is None
        assert c["band_all_draws"]["lo_label"] == "in not-applicable mass"
        assert out["K_q"]["0.5"]["LB"] is None


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS: {fn.__name__}")
    print(f"\nall {len(fns)} two-stage tests passed")

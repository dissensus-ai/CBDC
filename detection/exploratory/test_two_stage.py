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
from two_stage import (gap_recovered, make_hook, missed_per_10k_from_set,  # noqa: E402
                       shortlist_size, two_stage_select)


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


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS: {fn.__name__}")
    print(f"\nall {len(fns)} two-stage tests passed")

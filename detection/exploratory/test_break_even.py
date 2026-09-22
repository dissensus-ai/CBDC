"""Gain-threshold crossing tests on hand-built inputs with known answers."""

import math
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from break_even import (LEFT_CENSORED_LABEL, RIGHT_CENSORED_LABEL,  # noqa: E402
                        bootstrap_gain_threshold_crossing, curve_stats,
                        first_crossing, gain_threshold_crossing)

GRID = np.array([0.0, 0.25, 0.5, 0.75, 1.0])


def test_first_crossing_interpolates():
    v, flag, rec = first_crossing(GRID, [-2.0, -1.0, 1.0, 3.0, 5.0], 0.0)
    assert flag is None and rec is False
    assert abs(v - 0.375) < 1e-12        # halfway between 0.25 and 0.5
    v, _, _ = first_crossing(GRID, [-2.0, -1.0, 1.0, 3.0, 5.0], 1.0)
    assert abs(v - 0.5) < 1e-12          # exactly at a grid point: > not >=
    # touching tau (0.5, 0.75) is not exceeding it; the crossing starts at
    # the last touch point
    v, _, _ = first_crossing(GRID, [-2.0, -1.0, 1.0, 1.0, 5.0], 1.0)
    assert v == 0.75


def test_censoring_flags():
    v, flag, _ = first_crossing(GRID, [2.0, 3, 4, 5, 6], 0.0)
    assert flag == "left_censored" and v == 0.0
    v, flag, _ = first_crossing(GRID, [-3.0, -2, -1, -0.5, 0.0], 0.0)
    assert flag == "not_reached" and math.isnan(v)


def test_recrossing_flagged():
    v, flag, rec = first_crossing(GRID, [-1.0, 1.0, -1.0, 2.0, 3.0], 0.0)
    assert rec is True and abs(v - 0.125) < 1e-12


def test_nan_points_are_not_evidence():
    v, flag, _ = first_crossing(GRID, [-1.0, np.nan, 1.0, 2.0, 3.0], 0.0)
    assert v == 0.5 and flag is None      # placed at the grid point
    v, flag, _ = first_crossing(GRID, [np.nan] * 5, 0.0)
    assert flag == "not_reached"


def test_bad_grid_rejected():
    for g in ([0.0, 0.5, 0.5], [1.0, 0.0]):
        try:
            first_crossing(np.array(g), np.zeros(len(g)), 0.0)
        except ValueError:
            continue
        raise AssertionError("non-increasing grid should raise")


def test_curve_stats_matches_t_formula():
    rng = np.random.default_rng(0)
    D = rng.normal(size=(12, 5)) + GRID * 4
    cs = curve_stats(D, alpha=0.10)
    for j in range(5):
        x = D[:, j]
        half = stats.t.ppf(0.95, 11) * x.std(ddof=1) / math.sqrt(12)
        assert abs(cs["ci_lo"][j] - (x.mean() - half)) < 1e-12
    D[3, 2] = np.nan
    cs = curve_stats(D)
    assert cs["n"][2] == 11 and cs["n"][1] == 12


def test_crossing_known_linear_curve():
    """Delta = 10*lambda - 3 exactly plus symmetric +-1 noise: mean crosses 0
    at 0.3; the LB crossing is later and moves toward 0.3 as R grows."""
    def mat(R):
        noise = np.tile([1.0, -1.0], R // 2)[:, None]
        return 10 * GRID[None, :] - 3 + noise
    small = gain_threshold_crossing(GRID, mat(4), 0.0)
    large = gain_threshold_crossing(GRID, mat(400), 0.0)
    assert abs(small["crossing_mean"] - 0.3) < 1e-12
    assert abs(large["crossing_mean"] - 0.3) < 1e-12
    assert small["crossing_LB"] > large["crossing_LB"] > 0.3
    # threshold 1.0: mean crosses at 0.4
    assert abs(gain_threshold_crossing(GRID, mat(4), 1.0)["crossing_mean"]
               - 0.4) < 1e-12


def test_bootstrap_all_crossing_is_reproducible_and_brackets():
    rng = np.random.default_rng(3)
    D = 10 * GRID[None, :] - 3 + rng.normal(scale=1.0, size=(30, 5))
    a = bootstrap_gain_threshold_crossing(GRID, D, 0.0, B=400, seed=11)
    b = bootstrap_gain_threshold_crossing(GRID, D, 0.0, B=400, seed=11)
    assert a == b
    pt = gain_threshold_crossing(GRID, D, 0.0)["crossing_mean"]
    m = a["mean"]
    assert m["p_no_crossing_in_range"] == 0.0
    assert m["lo"]["value"] <= pt <= m["hi"]["value"]
    # with no censoring the conditional interval IS the band
    assert m["conditional_on_crossing"]["n"] == 400
    assert m["conditional_on_crossing"]["lo"] == m["lo"]["value"]
    assert m["conditional_on_crossing"]["hi"] == m["hi"]["value"]


def mixed_toy():
    """Mean Delta just above tau at the last grid point only, with noise:
    a large share of bootstrap draws never cross within the grid."""
    z = np.random.default_rng(7).normal(scale=3.0, size=20)
    D = np.full((20, 5), -5.0)
    D[:, 4] = 0.1 + (z - z.mean())       # sample mean exactly 0.1 > tau = 0
    return D


def test_bootstrap_mixed_censoring_reports_right_censored_limit():
    D = mixed_toy()
    out = bootstrap_gain_threshold_crossing(GRID, D, 0.0, B=2000, seed=5)
    m = out["mean"]
    assert 0.2 < m["p_no_crossing_in_range"] < 0.8
    assert m["n_not_reached"] == round(m["p_no_crossing_in_range"] * 2000)
    # the upper limit lies in the censored mass: a bound, never a number
    assert m["hi"] == {"value": None, "censored": "right",
                       "label": RIGHT_CENSORED_LABEL}
    # the conditional interval is finite but explicitly labelled as such
    c = m["conditional_on_crossing"]
    assert c["n"] == 2000 - m["n_not_reached"]
    assert 0.75 <= c["lo"] <= c["hi"] <= 1.0
    assert "NOT the band" in c["note"]


def test_bootstrap_all_censored():
    D = np.full((12, 5), -100.0) + np.random.default_rng(0).normal(size=(12, 5))
    m = bootstrap_gain_threshold_crossing(GRID, D, 0.0, B=300, seed=1)["LB"]
    assert m["p_no_crossing_in_range"] == 1.0
    assert m["lo"]["censored"] == m["hi"]["censored"] == "right"
    assert m["conditional_on_crossing"] == {
        "note": m["conditional_on_crossing"]["note"], "n": 0, "lo": None,
        "hi": None}


def test_bootstrap_all_left_censored():
    D = np.full((12, 5), 100.0)
    m = bootstrap_gain_threshold_crossing(GRID, D, 0.0, B=100)["mean"]
    assert m["p_left_censored"] == 1.0
    assert m["lo"]["label"] == m["hi"]["label"] == LEFT_CENSORED_LABEL


def test_mean_is_primary_and_listed_first():
    D = 10 * GRID[None, :] - 3 + np.tile([1.0, -1.0], 3)[:, None]
    out = gain_threshold_crossing(GRID, D, 0.0)
    assert out["primary"] == "crossing_mean"
    assert out["secondary"] == "crossing_LB"
    keys = list(out)
    assert keys.index("crossing_mean") < keys.index("crossing_LB")
    assert list(bootstrap_gain_threshold_crossing(GRID, D, 0.0, B=10))[4:] \
        == ["mean", "LB"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS: {fn.__name__}")
    print(f"\nall {len(fns)} gain-threshold crossing tests passed")

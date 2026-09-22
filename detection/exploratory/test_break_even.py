"""Break-even estimator tests on hand-built inputs with known answers."""

import math
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from break_even import (bootstrap_break_even, break_even,  # noqa: E402
                        curve_stats, first_crossing)

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


def test_break_even_known_linear_curve():
    """Delta = 10*lambda - 3 exactly plus symmetric +-1 noise: mean crosses 0
    at 0.3; the LB crossing is later and moves toward 0.3 as R grows."""
    def mat(R):
        noise = np.tile([1.0, -1.0], R // 2)[:, None]
        return 10 * GRID[None, :] - 3 + noise
    small, large = break_even(GRID, mat(4), 0.0), break_even(GRID, mat(400), 0.0)
    assert abs(small["lambda_star_mean"] - 0.3) < 1e-12
    assert abs(large["lambda_star_mean"] - 0.3) < 1e-12
    assert small["lambda_star_LB"] > large["lambda_star_LB"] > 0.3
    # threshold 1.0: mean crosses at 0.4
    assert abs(break_even(GRID, mat(4), 1.0)["lambda_star_mean"] - 0.4) < 1e-12


def test_bootstrap_band_brackets_and_is_reproducible():
    rng = np.random.default_rng(3)
    D = 10 * GRID[None, :] - 3 + rng.normal(scale=1.0, size=(30, 5))
    a = bootstrap_break_even(GRID, D, 0.0, B=400, seed=11)
    b = bootstrap_break_even(GRID, D, 0.0, B=400, seed=11)
    assert a == b
    pt = break_even(GRID, D, 0.0)["lambda_star_mean"]
    assert a["mean"]["lo"] <= pt <= a["mean"]["hi"]
    assert a["mean"]["n_finite"] == 400
    # a never-reached curve reports censoring rather than a fake band
    c = bootstrap_break_even(GRID, D - 100, 0.0, B=50, seed=1)
    assert c["LB"]["n_not_reached"] == 50 and math.isnan(c["LB"]["lo"])


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS: {fn.__name__}")
    print(f"\nall {len(fns)} break-even tests passed")

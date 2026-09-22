"""E10 path tests: lambda = 0, 1, 2 are s = low, mid, high exactly; the path is
piecewise-linear with its kink at the default world."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from dgp import DGPConfig  # noqa: E402
from identity_signal import (DEFAULT_LAMBDA_GRID, IDENTITY_FIELDS,  # noqa: E402
                             cfg_factory, lambda_block, lambda_config,
                             mid_position, resolved_block)
from surface_configs import world_config  # noqa: E402


def _fields(cfg):
    d = cfg.to_dict()
    d.pop("label")
    return d


def test_identity_fields_cover_every_identity_parameter():
    from surface_configs import _S_IDENTITY
    touched = set().union(*(blk.keys() for blk in _S_IDENTITY.values()))
    assert touched <= set(IDENTITY_FIELDS)
    assert set(IDENTITY_FIELDS) <= set(DGPConfig().to_dict())


def test_anchors_reproduce_s_levels_field_for_field():
    for lam, s in ((0.0, "low"), (1.0, "mid"), (2.0, "high")):
        for b in ("low", "mid", "high"):
            for p in (0.01, 0.05):
                got = lambda_config(lam, seed=700_001, n_entities=5000, b=b,
                                    prevalence=p)
                ref = world_config(b, s, p, seed=700_001, n_entities=5000)
                assert _fields(got) == _fields(ref), (lam, s, b, p)
                for f in IDENTITY_FIELDS:
                    assert type(getattr(got, f)) is type(getattr(ref, f)), f


def test_lambda_one_is_the_default_world():
    from dgp import default_config
    got = _fields(lambda_config(1.0, seed=19, n_entities=8000))
    ref = default_config(19).to_dict()
    ref["n_entities"] = 8000
    ref.pop("label")
    assert got == ref


def test_lambda_zero_has_no_class_signal():
    blk = lambda_block(0.0)
    assert blk["watchlist_tpr"] == blk["watchlist_fpr"]
    for base in ("sar_lambda", "kyc_low_prob", "juris_beta", "acct_age_mu"):
        assert blk[f"{base}_launderer"] == blk[f"{base}_legit"], base


def test_paired_stream_flag_only_differs_in_that_field():
    a = _fields(lambda_config(0.7, seed=3))
    b = lambda_config(0.7, seed=3, identity_rng_stream=True).to_dict()
    b.pop("label")
    assert b.pop("identity_rng_stream") is True
    assert a == b
    # the E10 factory turns the paired stream on
    assert cfg_factory(1.5)(700_001, 100).identity_rng_stream is True


def test_behavior_untouched_along_lambda():
    ref = _fields(world_config("mid", "low", 0.05, seed=1))
    for lam in (0.0, 0.3, 1.0, 1.77, 2.0):
        got = _fields(lambda_config(lam, seed=1))
        for f in ref:
            if f not in IDENTITY_FIELDS:
                assert got[f] == ref[f], f


def test_each_segment_is_linear_and_elementwise():
    for (lam, t, a, b) in ((0.25, 0.25, "low", "mid"),
                           (1.5, 0.5, "mid", "high")):
        lo, hi, got = resolved_block(a), resolved_block(b), lambda_block(lam)
        for f in IDENTITY_FIELDS:
            if isinstance(lo[f], tuple):
                for i in range(len(lo[f])):
                    assert abs(got[f][i] - ((1 - t) * lo[f][i] + t * hi[f][i])) < 1e-12
            else:
                assert abs(got[f] - ((1 - t) * lo[f] + t * hi[f])) < 1e-12


def test_kink_segments_move_different_parameters():
    """[0,1] moves only launderer-side fields; [1,2] also moves the legit side
    and leaves launderer account age fixed. Pinned so the DESIGN note's
    description of the kink cannot drift from the config."""
    a, m, h = lambda_block(0.0), lambda_block(1.0), lambda_block(2.0)
    moved_1 = {f for f in IDENTITY_FIELDS if a[f] != m[f]}
    moved_2 = {f for f in IDENTITY_FIELDS if m[f] != h[f]}
    assert moved_1 == {"watchlist_tpr", "sar_lambda_launderer",
                       "kyc_low_prob_launderer", "juris_beta_launderer",
                       "acct_age_mu_launderer"}
    assert moved_2 == {"watchlist_tpr", "sar_lambda_launderer",
                       "sar_lambda_legit", "kyc_low_prob_launderer",
                       "kyc_low_prob_legit", "juris_beta_launderer",
                       "juris_beta_legit"}


def test_lambda_out_of_range_rejected():
    for bad in (-0.01, 2.01):
        try:
            lambda_block(bad)
        except ValueError:
            continue
        raise AssertionError(f"lambda={bad} should raise")


def test_default_grid():
    assert DEFAULT_LAMBDA_GRID == tuple(0.25 * i for i in range(9))


def test_mid_position_confirms_anchor_and_pins_straight_line_finding():
    rep = mid_position()
    assert rep["lambda_1_equals_mid"] is True and rep["all_anchors_exact"]
    st = rep["straight_line_lambda_for_mid"]
    assert rep["straight_line_collinear"] is False
    assert abs(st["watchlist_tpr"] - 0.38 / 0.70) < 1e-12
    assert abs(st["sar_lambda_launderer"] - 0.3 / 1.1) < 1e-12
    assert st["acct_age_mu_launderer"] == 1.0
    assert st["sar_lambda_legit"] == 0.0 and st["watchlist_fpr"] is None


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS: {fn.__name__}")
    print(f"\nall {len(fns)} identity-signal tests passed")

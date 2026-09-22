"""Arm B config tests: lambda endpoints are exact, mid is off the line."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from dgp import DGPConfig  # noqa: E402
from identity_signal import (IDENTITY_FIELDS, lambda_block,  # noqa: E402
                             lambda_config, mid_position)
from surface_configs import world_config  # noqa: E402


def _fields(cfg):
    d = cfg.to_dict()
    d.pop("label")
    return d


def test_identity_fields_cover_every_identity_parameter():
    # every field any s block touches must be interpolated
    from surface_configs import _S_IDENTITY
    touched = set().union(*(blk.keys() for blk in _S_IDENTITY.values()))
    assert touched <= set(IDENTITY_FIELDS)
    assert set(IDENTITY_FIELDS) <= set(DGPConfig().to_dict())


def test_lambda_zero_is_s_low_field_for_field():
    for b in ("low", "mid", "high"):
        for p in (0.01, 0.05):
            a = lambda_config(0.0, seed=700_001, n_entities=5000, b=b,
                              prevalence=p)
            ref = world_config(b, "low", p, seed=700_001, n_entities=5000)
            assert _fields(a) == _fields(ref)
            for f in IDENTITY_FIELDS:
                assert type(getattr(a, f)) is type(getattr(ref, f)), f


def test_lambda_one_is_s_high_field_for_field():
    for b in ("low", "mid", "high"):
        a = lambda_config(1.0, seed=700_002, n_entities=8000, b=b)
        ref = world_config(b, "high", 0.05, seed=700_002, n_entities=8000)
        assert _fields(a) == _fields(ref)


def test_lambda_zero_has_no_class_signal():
    blk = lambda_block(0.0)
    for base in ("watchlist", "sar_lambda", "kyc_low_prob", "juris_beta",
                 "acct_age_mu"):
        if base == "watchlist":
            assert blk["watchlist_tpr"] == blk["watchlist_fpr"]
        else:
            assert blk[f"{base}_launderer"] == blk[f"{base}_legit"], base


def test_behavior_untouched_along_lambda():
    ref = _fields(world_config("mid", "low", 0.05, seed=1))
    for lam in (0.0, 0.3, 0.77, 1.0):
        got = _fields(lambda_config(lam, seed=1))
        for f in ref:
            if f not in IDENTITY_FIELDS:
                assert got[f] == ref[f], f


def test_interpolation_is_linear_and_elementwise():
    lo, hi = lambda_block(0.0), lambda_block(1.0)
    mid = lambda_block(0.25)
    for f in IDENTITY_FIELDS:
        if isinstance(lo[f], tuple):
            for i in range(len(lo[f])):
                assert abs(mid[f][i] - (0.75 * lo[f][i] + 0.25 * hi[f][i])) < 1e-12
        else:
            assert abs(mid[f] - (0.75 * lo[f] + 0.25 * hi[f])) < 1e-12


def test_lambda_out_of_range_rejected():
    for bad in (-0.01, 1.01):
        try:
            lambda_block(bad)
        except ValueError:
            continue
        raise AssertionError(f"lambda={bad} should raise")


def test_mid_is_not_collinear():
    rep = mid_position()
    assert rep["collinear"] is False
    by = {c["component"]: c for c in rep["components"]}
    # the reported finding, pinned so a silent config change trips the test
    assert abs(by["watchlist_tpr"]["lambda_mid"] - 0.38 / 0.70) < 1e-12
    assert abs(by["sar_lambda_launderer"]["lambda_mid"] - 0.3 / 1.1) < 1e-12
    assert abs(by["kyc_low_prob_launderer"]["lambda_mid"] - 0.375) < 1e-12
    assert by["acct_age_mu_launderer"]["lambda_mid"] == 1.0
    assert by["sar_lambda_legit"]["lambda_mid"] == 0.0
    assert by["watchlist_fpr"]["status"] == "constant_on_line"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS: {fn.__name__}")
    print(f"\nall {len(fns)} identity-signal tests passed")

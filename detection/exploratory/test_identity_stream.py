"""DGPConfig.identity_rng_stream tests.

(a) OFF (default): generate() output is byte-identical to the d28bef0
    generator. The golden digests below were computed on 22 Sep 2026 by running
    the d28bef0 tree (`git archive d28bef0 detection`) with the pinned
    environment (Python 3.14.7, numpy 2.3.5, pandas 2.3.3). They pin this
    environment; a different numpy/libm may legitimately change them.
(b) ON: at a fixed seed, labels, wallets and transactions are identical across
    lambda; only the identity columns change.
(c) ON at lambda = 1: a valid world (schema, ranges, audit-clean), which is a
    different draw from the option-off default world at the same seed.
"""

import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from dgp import default_config, generate, surveillance_strong_config  # noqa: E402
from identity_signal import lambda_config  # noqa: E402
from surface_configs import world_config  # noqa: E402

GOLDEN_D28BEF0 = {
    "default_700001": "f005c60d3c971a6385440f01db59c054365a404c7b298a8e02a381a6b1e43c82",
    "default_700002": "9b67b0061973a91d30cf311cf75a270bde1a39ff730992ec6b02df24e8d8dc70",
    "strong_700003": "9becccf9a9480d2f7768a69052855fe44a0292f5590c6a7fe69dc23759e86964",
    "surface_high_low_0.01": "536895f554b184be82d8fba6279f68bddf28975dbd5d97183d0704f191a850a2",
    "surface_low_high_0.03": "d4855619fba0fc0cedf19447509d30d1ef8bb891ce1acb79d9cbb975b30b041d",
    "surface_mid_mid_0.05": "37dbc887d94f83161005c0cbaec93de5b17a4c2b06e80ce50cf4e87e990584bd",
}

IDENTITY_COLS = ["kyc_tier", "account_age_days", "prior_sar_count",
                 "jurisdiction_risk", "on_watchlist"]


def _cases():
    yield "default_700001", default_config(700001)
    yield "default_700002", default_config(700002)
    yield "strong_700003", surveillance_strong_config(700003)
    yield "surface_high_low_0.01", world_config("high", "low", 0.01, 700004, 800)
    yield "surface_low_high_0.03", world_config("low", "high", 0.03, 700005, 800)
    yield "surface_mid_mid_0.05", world_config("mid", "mid", 0.05, 700006, 800)


def _digest(cfg):
    d = generate(cfg)
    h = hashlib.sha256()
    for k in ("entities", "wallets", "transactions"):
        h.update(d[k].to_csv(index=True).encode())
    h.update(json.dumps(cfg.to_dict(), sort_keys=True).encode())
    return h.hexdigest()


def test_off_is_byte_identical_to_d28bef0():
    for name, cfg in _cases():
        assert cfg.identity_rng_stream is False
        assert _digest(cfg) == GOLDEN_D28BEF0[name], name


def test_off_config_serialization_unchanged():
    d = default_config(1).to_dict()
    assert "identity_rng_stream" not in d
    cfg = default_config(1)
    cfg.identity_rng_stream = True
    assert cfg.to_dict()["identity_rng_stream"] is True


def test_on_behavior_identical_across_lambda():
    worlds = [generate(lambda_config(lam, seed=700_011, n_entities=600,
                                     identity_rng_stream=True))
              for lam in (0.0, 0.5, 1.0, 1.25, 2.0)]
    ref = worlds[0]
    for w in worlds[1:]:
        assert w["transactions"].equals(ref["transactions"])
        assert w["wallets"].equals(ref["wallets"])
        assert w["entities"][["entity_id", "is_launderer"]].equals(
            ref["entities"][["entity_id", "is_launderer"]])
    # identity columns do move with lambda (0 -> 2 changes every block)
    lo, hi = worlds[0]["entities"], worlds[-1]["entities"]
    changed = [c for c in IDENTITY_COLS if not lo[c].equals(hi[c])]
    assert set(changed) == set(IDENTITY_COLS), changed


def test_off_behavior_does_change_with_lambda():
    """The reason the option exists: without it, lambda moves transactions."""
    a = generate(lambda_config(0.0, seed=700_011, n_entities=600))
    b = generate(lambda_config(2.0, seed=700_011, n_entities=600))
    assert not a["transactions"].equals(b["transactions"])


def test_on_at_lambda_one_is_a_valid_different_world():
    from degeneracy_audit import audit
    from features import (T1_WALLET_COLS, TIER_COLS, build_entity_features,
                          build_wallet_features)
    on = generate(lambda_config(1.0, seed=700_012, n_entities=2000,
                                identity_rng_stream=True))
    off = generate(lambda_config(1.0, seed=700_012, n_entities=2000))
    e = on["entities"]
    assert list(e.columns) == list(off["entities"].columns)
    assert e.kyc_tier.isin([0, 1, 2]).all()
    assert (e.account_age_days > 0).all() and (e.prior_sar_count >= 0).all()
    assert e.jurisdiction_risk.between(0, 1).all()
    assert e.on_watchlist.isin([0, 1]).all()
    assert e.is_launderer.equals(off["entities"].is_launderer)   # drawn first
    assert not on["transactions"].equals(off["transactions"])    # new draw
    wf = build_wallet_features(on)
    rep = audit(build_entity_features(on, wf), wf, TIER_COLS, T1_WALLET_COLS)
    assert rep["gate1_pass"] and rep["gate2_pass"]
    # class-conditional identity signal is still there at the default level
    lau = e.is_launderer.astype(bool)
    assert e.on_watchlist[lau].mean() > e.on_watchlist[~lau].mean()
    assert np.isfinite(e.account_age_days).all()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS: {fn.__name__}")
    print(f"\nall {len(fns)} identity-stream tests passed")

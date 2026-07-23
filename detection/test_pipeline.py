"""Regression tests for the honest detection pipeline.

    python3 test_pipeline.py

Test 1 reconstructs the ORIGINAL paper's artifact — launderers defined
to hold 3-6 wallets, legit 1-2, no overlap — and asserts the degeneracy
audit hard-fails on num_wallets at AUC ~1.0. This is the check that
would have blocked the AUC = 1.000 headline before it reached a
permanent DOI.

Test 2 asserts the shipped DGP passes the audit and that num_wallets is
uninformative (AUC < 0.6) under the shared wallet-count distribution.
"""

import sys

import numpy as np

from data_loaders import load_synthetic
from degeneracy_audit import DegeneracyError, audit
from dgp import default_config
from features import (T1_WALLET_COLS, TIER_COLS, build_entity_features,
                      build_wallet_features)


def _featurize(data):
    wf = build_wallet_features(data)
    ef = build_entity_features(data, wf)
    return wf, ef


def test_audit_catches_original_artifact():
    """Rig the world the way the old §4.5 did: wallet count IS the label."""
    cfg = default_config(seed=7)
    cfg.n_entities = 400
    data = load_synthetic(cfg)
    rng = np.random.default_rng(7)

    ent = data["entities"]
    wal = data["wallets"]
    # reassign wallet counts definitionally: launderers 3-6, legit 1-2
    rows, widx = [], 0
    for _, e in ent.iterrows():
        n = rng.integers(3, 7) if e.is_launderer else rng.integers(1, 3)
        for _ in range(n):
            rows.append((f"W{widx}", e.entity_id))
            widx += 1
    import pandas as pd
    rigged = dict(data, wallets=pd.DataFrame(
        rows, columns=["wallet_id", "entity_id"]))
    # keep only transactions whose internal wallets still exist
    keep = set(rigged["wallets"].wallet_id)
    tx = data["transactions"]
    rigged["transactions"] = tx[
        (tx.src.str.startswith("X") | tx.src.isin(keep))
        & (tx.dst.str.startswith("X") | tx.dst.isin(keep))]

    wf, ef = _featurize(rigged)
    try:
        audit(ef, wf, TIER_COLS, T1_WALLET_COLS)
    except DegeneracyError as e:
        assert "num_wallets" in str(e), \
            f"audit failed but not on num_wallets:\n{e}"
        print("PASS: audit hard-fails on the original num_wallets artifact")
        return
    raise AssertionError(
        "audit PASSED a definitionally-rigged DGP — the gate is broken")


def test_shipped_dgp_is_clean():
    data = load_synthetic(default_config(seed=11))
    wf, ef = _featurize(data)
    rep = audit(ef, wf, TIER_COLS, T1_WALLET_COLS)   # raises on failure
    nw = next(r for r in rep["table"] if r["feature"] == "num_wallets")
    assert nw["auc"] < 0.6, \
        f"num_wallets should be uninformative, got AUC {nw['auc']:.3f}"
    print(f"PASS: shipped DGP audit-clean "
          f"(worst {rep['worst_feature']} = {rep['worst_auc']:.3f}; "
          f"num_wallets = {nw['auc']:.3f})")


if __name__ == "__main__":
    test_audit_catches_original_artifact()
    test_shipped_dgp_is_clean()
    print("all tests passed")
    sys.exit(0)

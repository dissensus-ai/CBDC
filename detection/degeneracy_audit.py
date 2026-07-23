"""Pre-registered degeneracy audit (design pack §4.1, gates 1-2).

Runs BEFORE any headline number. Hard-fails (raises / exits non-zero) if
the data-generating process contains a definitional separator:

Gate 1  single-feature separability cap: for EVERY feature available to
        any tier, a one-feature classifier must not reach entity-level
        AUC > 0.95. This is exactly the check that would have caught the
        original paper's num_wallets artifact (which scores AUC = 1.000
        here by construction of the old DGP).
Gate 2  marginal-overlap: no feature may be class-disjoint (two-sample
        KS statistic == 1.0 means the class-conditional supports do not
        overlap at all).

Wallet-level (T1) features are audited the way a one-feature detector
would be deployed: score each wallet by the raw feature, aggregate to
entity by max. Entity-level features are scored directly.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score

AUC_CAP = 0.95


class DegeneracyError(RuntimeError):
    """The DGP contains a (near-)definitional single-feature separator."""


def _single_feature_auc(y, x):
    x = np.nan_to_num(np.asarray(x, dtype=float))
    if len(np.unique(x)) < 2:
        return 0.5
    auc = roc_auc_score(y, x)
    return max(auc, 1.0 - auc)


def audit(entity_feats, wallet_feats, tier_cols, t1_cols) -> dict:
    """Returns the audit report dict; raises DegeneracyError on failure."""
    y = entity_feats.is_launderer.to_numpy()
    records = []

    # entity-level features (everything any of T2-T4 can see)
    all_entity_cols = tier_cols["T4"]
    for c in all_entity_cols:
        x = entity_feats[c].to_numpy(dtype=float)
        ks = ks_2samp(x[y == 1], x[y == 0])
        records.append({"feature": c, "level": "entity",
                        "auc": _single_feature_auc(y, x),
                        "ks_stat": float(ks.statistic)})

    # wallet-level features, max-aggregated to entity (how a one-feature
    # wallet detector would be scored)
    wf = wallet_feats.merge(
        entity_feats[["entity_id", "is_launderer"]], on="entity_id")
    for c in t1_cols:
        for direction, xw in (("hi", wf[c]), ("lo", -wf[c])):
            agg = xw.groupby(wf.entity_id).max()
            ye = entity_feats.set_index("entity_id").loc[agg.index,
                                                         "is_launderer"]
            auc = _single_feature_auc(ye.to_numpy(), agg.to_numpy())
            records.append({"feature": f"{c}[wallet-max-{direction}]",
                            "level": "wallet", "auc": auc,
                            "ks_stat": float("nan")})

    rep = pd.DataFrame(records).sort_values("auc", ascending=False)
    worst = rep.iloc[0]
    violations = rep[rep.auc > AUC_CAP]
    disjoint = rep[(rep.level == "entity") & (rep.ks_stat >= 1.0)]

    report = {
        "n_entities": int(len(y)),
        "n_launderers": int(y.sum()),
        "auc_cap": AUC_CAP,
        "worst_feature": worst.feature,
        "worst_auc": float(worst.auc),
        "gate1_pass": violations.empty,
        "gate2_pass": disjoint.empty,
        "violations": violations.to_dict("records"),
        "class_disjoint": disjoint.to_dict("records"),
        "table": rep.to_dict("records"),
    }
    if not violations.empty or not disjoint.empty:
        raise DegeneracyError(
            "DEGENERACY AUDIT FAILED — run is void, re-draw the DGP.\n"
            + violations.to_string(index=False)
            + ("\nclass-disjoint features:\n" + disjoint.to_string(index=False)
               if not disjoint.empty else ""))
    return report


def print_report(report, top=12):
    tab = pd.DataFrame(report["table"]).head(top)
    print(f"degeneracy audit: {report['n_entities']} entities "
          f"({report['n_launderers']} launderers), cap AUC={report['auc_cap']}")
    print(f"  gate 1 (single-feature AUC cap): "
          f"{'PASS' if report['gate1_pass'] else 'FAIL'} "
          f"(worst: {report['worst_feature']} = {report['worst_auc']:.3f})")
    print(f"  gate 2 (no class-disjoint marginal): "
          f"{'PASS' if report['gate2_pass'] else 'FAIL'}")
    print("  top single-feature AUCs:")
    for _, r in tab.iterrows():
        ks = "" if np.isnan(r.ks_stat) else f"  KS={r.ks_stat:.3f}"
        print(f"    {r.auc:.3f}  {r.feature}{ks}")


if __name__ == "__main__":
    from dgp import default_config, generate
    from features import (T1_WALLET_COLS, TIER_COLS, build_entity_features,
                          build_wallet_features)

    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 20260707
    data = generate(default_config(seed))
    wf = build_wallet_features(data)
    ef = build_entity_features(data, wf)
    try:
        rep = audit(ef, wf, TIER_COLS, T1_WALLET_COLS)
    except DegeneracyError as e:
        print(str(e))
        sys.exit(1)
    print_report(rep)
    with open("results/degeneracy_audit.json", "w") as f:
        json.dump(rep, f, indent=2)

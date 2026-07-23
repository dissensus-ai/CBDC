"""Four-tier marginal-value experiment (design pack E1).

Tiers: T1 structure-only -> T2 +pseudonymous linkage -> T3 +identity
attributes -> T4 +watchlist. Two model families (logistic regression and
gradient boosting — sklearn HistGradientBoostingClassifier substitutes
for XGBoost, which is not in this environment; same family).

Protocol (fixes the pseudoreplication class of errors):
* entity-DISJOINT cross-validation: folds are splits over ENTITIES; for
  the wallet-level T1 tier every wallet inherits its entity's fold, so no
  entity ever straddles train/test.
* effective N = number of entities, reported explicitly.
* entity-clustered bootstrap CIs: the evaluation unit resampled with
  replacement is the entity (each entity contributes exactly one
  out-of-fold score), so the CI never sees wallet-level pseudoreplicates.
* per-entity out-of-fold scores are saved so the TOST equivalence test
  (equivalence_test.py) can do a PAIRED entity-level bootstrap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import T1_WALLET_COLS, TIER_COLS

N_FOLDS = 5
N_BOOT = 1000


def _models(seed):
    return {
        "logit": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced",
                               random_state=seed)),
        "gboost": HistGradientBoostingClassifier(
            class_weight="balanced", random_state=seed),
    }


def _entity_folds(entities, seed):
    """Fold assignment per entity — the entity is the CV unit."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    fold = np.empty(len(entities), dtype=int)
    for k, (_, test_idx) in enumerate(
            skf.split(entities.entity_id, entities.is_launderer)):
        fold[test_idx] = k
    return pd.Series(fold, index=entities.entity_id.values, name="fold")


def _oof_t1(wallet_feats, entities, folds, model_name, seed):
    """T1: train on wallets (labels inherited from the owning entity,
    folds inherited too), evaluate at entity level via max wallet score."""
    wf = wallet_feats.merge(
        entities[["entity_id", "is_launderer"]], on="entity_id")
    wf["fold"] = wf.entity_id.map(folds)
    X = wf[T1_WALLET_COLS].to_numpy(dtype=float)
    y = wf.is_launderer.to_numpy()
    scores = np.full(len(wf), np.nan)
    for k in range(N_FOLDS):
        tr, te = wf.fold != k, wf.fold == k
        m = _models(seed)[model_name]
        m.fit(X[tr.values], y[tr.values])
        scores[te.values] = m.predict_proba(X[te.values])[:, 1]
    wf["score"] = scores
    ent_score = wf.groupby("entity_id").score.max()
    return ent_score.reindex(entities.entity_id.values).to_numpy()


def _oof_entity_tier(entity_feats, cols, folds, model_name, seed):
    """T2-T4: one row per entity; entity-disjoint by construction."""
    ef = entity_feats.set_index("entity_id").loc[folds.index]
    X = ef[cols].to_numpy(dtype=float)
    y = ef.is_launderer.to_numpy()
    scores = np.full(len(ef), np.nan)
    for k in range(N_FOLDS):
        tr, te = (folds != k).values, (folds == k).values
        m = _models(seed)[model_name]
        m.fit(X[tr], y[tr])
        scores[te] = m.predict_proba(X[te])[:, 1]
    return scores


def clustered_bootstrap_ci(y, score, metric, n_boot=N_BOOT, seed=0,
                           alpha=0.05):
    """Percentile bootstrap over ENTITIES (one score per entity, so
    resampling rows = resampling clusters)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    stats = []
    while len(stats) < n_boot:
        idx = rng.integers(0, n, n)
        if y[idx].min() == y[idx].max():
            continue                       # degenerate resample, redraw
        stats.append(metric(y[idx], score[idx]))
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def run_experiment(data, wallet_feats, entity_feats, seed,
                   n_boot=N_BOOT) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (results table, per-entity OOF score matrix)."""
    entities = data["entities"]
    folds = _entity_folds(entities, seed)
    y = entities.is_launderer.to_numpy()

    oof = pd.DataFrame({"entity_id": entities.entity_id,
                        "is_launderer": y})
    rows = []
    for model_name in ("logit", "gboost"):
        tier_scores = {"T1": _oof_t1(wallet_feats, entities, folds,
                                     model_name, seed)}
        for tier, cols in TIER_COLS.items():
            tier_scores[tier] = _oof_entity_tier(entity_feats, cols, folds,
                                                 model_name, seed)
        for tier, s in tier_scores.items():
            oof[f"{tier}_{model_name}"] = s
            auc = roc_auc_score(y, s)
            ap = average_precision_score(y, s)
            auc_lo, auc_hi = clustered_bootstrap_ci(
                y, s, roc_auc_score, n_boot, seed)
            ap_lo, ap_hi = clustered_bootstrap_ci(
                y, s, average_precision_score, n_boot, seed)
            rows.append({
                "model": model_name, "tier": tier,
                "auc": auc, "auc_ci_lo": auc_lo, "auc_ci_hi": auc_hi,
                "ap": ap, "ap_ci_lo": ap_lo, "ap_ci_hi": ap_hi,
                "n_entities": len(y), "n_launderers": int(y.sum()),
            })
    return pd.DataFrame(rows), oof


def label_permutation_control(data, wallet_feats, entity_feats, seed):
    """Negative control: permuted entity labels must give AUC ~ 0.5.
    Sanity check that the pipeline does not leak."""
    rng = np.random.default_rng(seed + 999)
    entities = data["entities"].copy()
    entities["is_launderer"] = rng.permutation(
        entities.is_launderer.to_numpy())
    ef = entity_feats.drop(columns=["is_launderer"]).merge(
        entities[["entity_id", "is_launderer"]], on="entity_id")
    folds = _entity_folds(entities, seed)
    s = _oof_entity_tier(ef, TIER_COLS["T4"], folds, "gboost", seed)
    return float(roc_auc_score(entities.is_launderer.to_numpy(), s))

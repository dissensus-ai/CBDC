"""M2 (additive boosting) for E10, fitted through run_replicate's score_hook.

run_replicate fits the confirmatory pair only (logit = M1, gboost = M4). E10
adds additive boosting because the four-model ladder puts most of the identity
effect at linear -> additive. Rather than touch the confirmatory model loop,
this hook fits M2 on the same training world, scores the same test world, and
writes the same record fields under the name "additive".

Specification: HistGradientBoostingClassifier(interaction_cst="no_interactions",
class_weight="balanced") at sklearn defaults otherwise (learning_rate 0.1,
max_leaf_nodes 31) -- the untuned counterpart of the confirmatory gboost, as the
ladder's M2 is the no-interaction counterpart of its M4. It is NOT the ladder
M2: the ladder tunes a 4-point grid by nested CV, the replicate design fits
fixed hyperparameters. The protocol addendum must say which it means.
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "confirmatory"))

ADDITIVE_MODEL = "additive"


def additive_model(seed):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        interaction_cst="no_interactions", class_weight="balanced",
        random_state=seed)


def make_additive_hook():
    """score_hook adding rec["models"]["additive"] and its delta fields."""
    from endpoint import delta_miss, missed_per_10k, true_positives
    from features import TIER_COLS
    from replicate import TIERS, model_seed

    def hook(rec, ctx):
        y, tb, k = ctx["y_test"], ctx["tiebreak"], ctx["k"]
        y_tr = ctx["y_train"]
        scores = {}
        rec["models"][ADDITIVE_MODEL] = {}
        for tier in TIERS:
            cols = TIER_COLS[tier]
            m = additive_model(model_seed(ctx["train_seed"]))
            m.fit(ctx["train_ef"][cols].to_numpy(dtype=float), y_tr)
            s = m.predict_proba(ctx["test_ef"][cols].to_numpy(dtype=float))[:, 1]
            scores[tier] = s
            rec["models"][ADDITIVE_MODEL][tier] = {
                "TP": true_positives(y, s, tb, k),
                "MissedPer10k": missed_per_10k(y, s, tb, k),
            }
        for hi in ("T4", "T3"):
            rec[f"delta_miss_T2_minus_{hi}_{ADDITIVE_MODEL}"] = delta_miss(
                y, scores["T2"], scores[hi], tb, k)

    return hook

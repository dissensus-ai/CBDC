"""TOST equivalence test: pseudonymous linkage (T2) vs full surveillance (T4).

The paper's thesis is "identity + watchlist add no meaningful detection
value beyond ZK-achievable pseudonymous linkage." That is an EQUIVALENCE
claim, so it needs an equivalence test with a pre-registered margin, not
"1.00 = 1.00" on a saturated toy.

Method: paired entity-clustered percentile bootstrap of
    dAUC = AUC(T4) - AUC(T2)
using the same resampled entities for both tiers (per-entity out-of-fold
scores from detection_experiment.py). Two one-sided tests at level alpha
== the (1 - 2*alpha) percentile CI lying inside (-delta, +delta).

Verdicts — the harness must be able to produce ALL of these:
  EQUIVALENT               CI entirely inside (-delta, +delta): thesis holds.
  SURVEILLANCE_SUPERIOR    CI lower bound > +delta: identity/watchlist beat
                           linkage beyond the margin — THESIS FALSIFIED.
  LINKAGE_SUPERIOR         CI upper bound < -delta (would be odd; flags a
                           harness or DGP problem worth investigating).
  INCONCLUSIVE             CI straddles a margin bound: underpowered; no
                           equivalence claim may be made.

Default margin delta = 0.03 AUC (design pack §8 Q2 — pre-registered
default, an open call for MF; must be fixed before real-data runs).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

DELTA_DEFAULT = 0.03
ALPHA = 0.05
N_BOOT = 2000


def tost_equivalence(oof, tier_lo="T2", tier_hi="T4", model="gboost",
                     delta=DELTA_DEFAULT, alpha=ALPHA, n_boot=N_BOOT,
                     seed=0) -> dict:
    y = oof.is_launderer.to_numpy()
    s_lo = oof[f"{tier_lo}_{model}"].to_numpy()
    s_hi = oof[f"{tier_hi}_{model}"].to_numpy()
    d_point = roc_auc_score(y, s_hi) - roc_auc_score(y, s_lo)

    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = []
    while len(deltas) < n_boot:
        idx = rng.integers(0, n, n)          # same entities for both tiers
        if y[idx].min() == y[idx].max():
            continue
        deltas.append(roc_auc_score(y[idx], s_hi[idx])
                      - roc_auc_score(y[idx], s_lo[idx]))
    deltas = np.asarray(deltas)
    ci_lo, ci_hi = np.percentile(deltas, [100 * alpha, 100 * (1 - alpha)])

    if ci_lo > delta:
        verdict = "SURVEILLANCE_SUPERIOR"    # thesis falsified
    elif ci_hi < -delta:
        verdict = "LINKAGE_SUPERIOR"
    elif -delta < ci_lo and ci_hi < delta:
        verdict = "EQUIVALENT"
    else:
        verdict = "INCONCLUSIVE"

    return {
        "comparison": f"{tier_hi} vs {tier_lo} ({model})",
        "delta_margin": delta,
        "alpha": alpha,
        "n_boot": n_boot,
        "n_entities": int(n),
        "n_launderers": int(y.sum()),
        "dAUC_point": float(d_point),
        "dAUC_ci90": [float(ci_lo), float(ci_hi)],
        "boot_frac_above_margin": float((deltas > delta).mean()),
        "boot_frac_below_neg_margin": float((deltas < -delta).mean()),
        "verdict": verdict,
    }


def print_tost(res):
    print(f"TOST {res['comparison']}: dAUC = {res['dAUC_point']:+.4f}, "
          f"90% CI [{res['dAUC_ci90'][0]:+.4f}, {res['dAUC_ci90'][1]:+.4f}], "
          f"margin ±{res['delta_margin']}")
    print(f"  effective N = {res['n_entities']} entities "
          f"({res['n_launderers']} launderers)")
    print(f"  verdict: {res['verdict']}")

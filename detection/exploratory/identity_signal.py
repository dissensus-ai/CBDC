"""E10 (Arm B): a piecewise identity-signal path lambda in [0, 2]. EXPLORATORY.

surface_configs exposes the identity-signal axis s as three parameter blocks.
This module runs a path through all three (protocol addendum E9-E11, section 2):

    lambda in [0, 1]:  (1 - t) * param(s=low) + t * param(s=mid),  t = lambda
    lambda in [1, 2]:  (1 - t) * param(s=mid) + t * param(s=high), t = lambda-1

for every identity-attribute field of DGPConfig (tuples element-wise), with
behaviour parameters, b and prevalence exactly as surface_configs sets them.
The (1-t)*a + t*b form returns a and b bit-exactly, so lambda = 0, 1, 2 ARE
s = low, mid, high field for field (tested). lambda = 1 is the paper's default
world.

Why piecewise. The 22 Sep dev build used a straight low -> high line and found
s = mid is not on it: the lambda reproducing mid differs per parameter (0 for
the legit-side fields, 0.27-0.54 for the launderer rates, 1 for account age),
because s = high also moves legit-side parameters and leaves launderer account
age at its default. A straight line would have excluded the headline world.

THE KINK AT lambda = 1. The path is continuous but not differentiable there,
and the two segments move different parameters: on [0, 1] only the launderer
side moves (legit-side values are equal at low and mid) and launderer account
age travels its whole range 6.3 -> 5.9; on [1, 2] account age is fixed while
the legit side moves too (sar_lambda_legit 0.10 -> 0.05, kyc_low_prob_legit
0.30 -> 0.20, juris_beta_legit[0] 2.0 -> 1.8). Equal lambda steps on the two
segments are not equal steps in any common signal measure, so a lambda* on
either side of 1 is read on that segment's own scale.

RNG. With DGPConfig.identity_rng_stream False, identity parameters change the
behavioural draws too (the Poisson/beta samplers consume a parameter-dependent
number of uniforms before behaviour is drawn). The E10 driver therefore sets
identity_rng_stream=True, under which worlds at every lambda share labels,
wallets and transactions at a given seed and differ only in identity columns:
paired counterfactuals. Those paired worlds are a different realisation from
the option-off worlds at the same seed, including at lambda = 1.
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from dgp import DGPConfig  # noqa: E402
from surface_configs import world_config  # noqa: E402

# Every identity-attribute field of DGPConfig consumed by dgp._identity_attrs.
IDENTITY_FIELDS = (
    "watchlist_tpr", "watchlist_fpr",
    "sar_lambda_launderer", "sar_lambda_legit",
    "kyc_low_prob_launderer", "kyc_low_prob_legit",
    "juris_beta_launderer", "juris_beta_legit",
    "acct_age_mu_launderer", "acct_age_mu_legit", "acct_age_sigma",
)

DEFAULT_LAMBDA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
LAMBDA_MAX = 2.0
# lambda at each named block
ANCHORS = {"low": 0.0, "mid": 1.0, "high": 2.0}


def resolved_block(s: str) -> dict:
    """Identity field values world_config produces at signal level s."""
    cfg = world_config("mid", s, 0.05, seed=0)
    return {f: getattr(cfg, f) for f in IDENTITY_FIELDS}


def _lerp(a: float, b: float, t: float) -> float:
    return (1.0 - t) * a + t * b


def lambda_block(lam: float) -> dict:
    lam = float(lam)
    if not 0.0 <= lam <= LAMBDA_MAX:
        raise ValueError(f"lambda must be in [0, {LAMBDA_MAX:g}], got {lam}")
    if lam <= 1.0:
        a_blk, b_blk, t = resolved_block("low"), resolved_block("mid"), lam
    else:
        a_blk, b_blk, t = resolved_block("mid"), resolved_block("high"), lam - 1.0
    out = {}
    for f in IDENTITY_FIELDS:
        a, b = a_blk[f], b_blk[f]
        if isinstance(a, tuple):
            out[f] = tuple(_lerp(x, y, t) for x, y in zip(a, b))
        else:
            out[f] = _lerp(a, b, t)
    return out


def lambda_config(lam: float, seed: int, n_entities: int = 8000,
                  b: str = "mid", prevalence: float = 0.05,
                  identity_rng_stream: bool = False) -> DGPConfig:
    """World at identity-signal lambda. Everything but the identity fields,
    identity_rng_stream and the label comes from world_config(b, s=low, p)."""
    cfg = world_config(b, "low", prevalence, seed=seed, n_entities=n_entities)
    for f, v in lambda_block(lam).items():
        setattr(cfg, f, v)
    cfg.identity_rng_stream = bool(identity_rng_stream)
    cfg.label = f"b={b}|lambda={float(lam):g}|p={prevalence:g}"
    return cfg


def cfg_factory(lam: float, b: str = "mid", prevalence: float = 0.05,
                identity_rng_stream: bool = True):
    """(seed, n) -> DGPConfig, the shape replicate.run_replicate expects.

    Paired stream ON by default: this is the E10 driver's factory."""
    def make(seed, n_entities):
        return lambda_config(lam, seed, n_entities, b, prevalence,
                             identity_rng_stream)
    return make


def mid_position(tol: float = 0.0) -> dict:
    """Confirms the anchors: lambda = 0, 1, 2 reproduce s = low, mid, high.

    Returns per-anchor field mismatches (empty when exact) and, for the
    record, the straight-line finding that motivated the piecewise path: the
    lambda on a low -> high line that would reproduce each mid component.
    """
    anchors = {}
    for s_name, lam in ANCHORS.items():
        ref, got = resolved_block(s_name), lambda_block(lam)
        anchors[s_name] = {
            "lambda": lam,
            "mismatched_fields": [f for f in IDENTITY_FIELDS
                                  if got[f] != ref[f]],
        }
    lo, mid, hi = (resolved_block(s) for s in ("low", "mid", "high"))
    straight = {}
    for f in IDENTITY_FIELDS:
        parts = (list(zip(lo[f], mid[f], hi[f])) if isinstance(lo[f], tuple)
                 else [(lo[f], mid[f], hi[f])])
        for i, (a, m, b) in enumerate(parts):
            name = f"{f}[{i}]" if isinstance(lo[f], tuple) else f
            straight[name] = (None if b == a
                              else (m - a) / (b - a) + 0.0)  # no -0.0
    defined = [v for v in straight.values() if v is not None]
    return {
        "anchors": anchors,
        "lambda_1_equals_mid": not anchors["mid"]["mismatched_fields"],
        "all_anchors_exact": all(not a["mismatched_fields"]
                                 for a in anchors.values()),
        "straight_line_lambda_for_mid": straight,
        "straight_line_collinear": max(defined) - min(defined) <= tol,
    }


if __name__ == "__main__":
    rep = mid_position()
    for s_name, a in rep["anchors"].items():
        print(f"lambda={a['lambda']:g} == s={s_name}: "
              f"{'exact' if not a['mismatched_fields'] else a['mismatched_fields']}")
    print("\nstraight low->high line (rejected), lambda that reproduces mid:")
    for name, v in rep["straight_line_lambda_for_mid"].items():
        print(f"  {name:<26}{'-' if v is None else f'{v:.4f}'}")

"""Arm B: a continuous identity-signal scale lambda in [0, 1]. EXPLORATORY.

surface_configs exposes the identity-signal axis s as three parameter blocks.
This module puts a scalar between the two ends:

    param(lambda) = (1 - lambda) * param(s=low) + lambda * param(s=high)

for every identity-attribute field of DGPConfig (tuples element-wise), with
behavior parameters, b and prevalence exactly as surface_configs sets them.
The (1-l)*a + l*b form, not a + l*(b-a), is deliberate: it returns a and b
bit-exactly at the endpoints, so lambda=0 IS s=low and lambda=1 IS s=high
field-for-field (tested), not merely close in floating point.

Interpolation is over the RESOLVED blocks -- the field values world_config
actually produces, defaults included -- not over the override dicts. s=low
leaves the legit-side fields at their defaults; s=high overrides some of them
(sar_lambda_legit 0.10 -> 0.05 etc.) and leaves acct_age_mu_launderer at the
default 5.9, so along the line the legit side moves and the account-age gap
opens fully by lambda=1 while other launderer parameters are still partway.

s=mid is NOT on this line. `mid_position` computes, per scalar component,
which lambda would reproduce mid; they disagree (0 for the legit-side fields,
0.27-0.54 for the launderer rates, 1 for account age). A lambda curve is a new
one-parameter family through the two endpoint worlds, not a refinement of the
three-level surface, and must not be read as passing through the default world.

RNG caveat: at a fixed seed, changing identity parameters changes the
behavioral transactions too. dgp.generate draws identity attributes BEFORE
behavior, and the Poisson/beta samplers consume a parameter-dependent number of
uniforms. Worlds at different lambda with the same seed share labels and wallet
counts, but not transactions -- they are not common random numbers.
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

DEFAULT_LAMBDA_GRID = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6,
                       0.7, 0.8, 0.9, 1.0)


def resolved_block(s: str) -> dict:
    """Identity field values world_config produces at signal level s."""
    cfg = world_config("mid", s, 0.05, seed=0)
    return {f: getattr(cfg, f) for f in IDENTITY_FIELDS}


def _lerp(a: float, b: float, lam: float) -> float:
    return (1.0 - lam) * a + lam * b


def lambda_block(lam: float) -> dict:
    lam = float(lam)
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lambda must be in [0, 1], got {lam}")
    lo, hi = resolved_block("low"), resolved_block("high")
    out = {}
    for f in IDENTITY_FIELDS:
        a, b = lo[f], hi[f]
        if isinstance(a, tuple):
            out[f] = tuple(_lerp(x, y, lam) for x, y in zip(a, b))
        else:
            out[f] = _lerp(a, b, lam)
    return out


def lambda_config(lam: float, seed: int, n_entities: int = 8000,
                  b: str = "mid", prevalence: float = 0.05) -> DGPConfig:
    """World at identity-signal lambda. Everything but the identity fields and
    the label comes from world_config(b, s=low, prevalence)."""
    cfg = world_config(b, "low", prevalence, seed=seed, n_entities=n_entities)
    for f, v in lambda_block(lam).items():
        setattr(cfg, f, v)
    cfg.label = f"b={b}|lambda={float(lam):g}|p={prevalence:g}"
    return cfg


def cfg_factory(lam: float, b: str = "mid", prevalence: float = 0.05):
    """(seed, n) -> DGPConfig, the shape replicate.run_replicate expects."""
    def make(seed, n_entities):
        return lambda_config(lam, seed, n_entities, b, prevalence)
    return make


def mid_position(tol: float = 1e-9) -> dict:
    """Where s=mid sits relative to the low->high line, per scalar component.

    For each component: lambda_mid = (mid - low) / (high - low) when high !=
    low; when high == low the component is constant along the line and is
    either on it (mid == low) or off it for every lambda. `collinear` is True
    only if every defined lambda_mid agrees within tol and no constant
    component is off the line.
    """
    lo, mid, hi = (resolved_block(s) for s in ("low", "mid", "high"))
    comps = []
    for f in IDENTITY_FIELDS:
        if isinstance(lo[f], tuple):
            parts = [(f"{f}[{i}]", lo[f][i], mid[f][i], hi[f][i])
                     for i in range(len(lo[f]))]
        else:
            parts = [(f, lo[f], mid[f], hi[f])]
        for name, a, m, b in parts:
            if abs(b - a) < tol:
                comps.append({"component": name, "low": a, "mid": m, "high": b,
                              "lambda_mid": None,
                              "status": ("constant_on_line" if abs(m - a) < tol
                                         else "constant_off_line")})
            else:
                lm = (m - a) / (b - a) + 0.0   # no -0.0 in the report
                comps.append({"component": name, "low": a, "mid": m, "high": b,
                              "lambda_mid": lm,
                              "status": ("in_segment" if -tol <= lm <= 1 + tol
                                         else "outside_segment")})
    defined = [c["lambda_mid"] for c in comps if c["lambda_mid"] is not None]
    off = any(c["status"] == "constant_off_line" for c in comps)
    collinear = (not off and bool(defined)
                 and max(defined) - min(defined) < tol)
    return {
        "components": comps,
        "collinear": collinear,
        "lambda_mid_min": min(defined) if defined else None,
        "lambda_mid_max": max(defined) if defined else None,
        "lambda_mid_common": defined[0] if collinear else None,
    }


if __name__ == "__main__":
    rep = mid_position()
    print(f"{'component':<26}{'low':>8}{'mid':>8}{'high':>8}  lambda_mid")
    for c in rep["components"]:
        lm = "-" if c["lambda_mid"] is None else f"{c['lambda_mid']:.4f}"
        print(f"{c['component']:<26}{c['low']:>8.3g}{c['mid']:>8.3g}"
              f"{c['high']:>8.3g}  {lm:<8} {c['status']}")
    print(f"\ncollinear: {rep['collinear']}  lambda_mid range "
          f"[{rep['lambda_mid_min']:.4f}, {rep['lambda_mid_max']:.4f}]")

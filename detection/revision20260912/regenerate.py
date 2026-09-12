"""Audit immutable raw results, recompute inference, generate manuscript tables.

Run with the pinned analysis environment. No original result is overwritten.
All intervals are 90% pointwise Student-t intervals for replicate means.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "revision-2026-09-12/evidence"
TEX = OUT.parent / "manuscript/tables"
sys.path.insert(0, str(ROOT / "detection/confirmatory"))
from inference import gated_did


def ci(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    m = float(x.mean())
    sd = float(x.std(ddof=1)) if n > 1 else None
    h = float(t.ppf(0.95, n - 1) * sd / np.sqrt(n)) if n > 1 else None
    return dict(
        mean=m,
        sd=sd,
        n=n,
        ci_lo=m - h if h is not None else None,
        ci_hi=m + h if h is not None else None,
        median=float(np.median(x)),
        q25=float(np.quantile(x, 0.25)),
        q75=float(np.quantile(x, 0.75)),
    )


def f(d, digits=3):
    return f"{d['mean']:.{digits}f} [{d['ci_lo']:.{digits}f}, {d['ci_hi']:.{digits}f}]"


def write_rows(name, rows):
    (TEX / name).write_text(
        "\n".join(" & ".join(r) + r" \\" for r in rows) + "\n\\bottomrule\n"
    )


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    TEX.mkdir(exist_ok=True, parents=True)
    sys.path.insert(0, str(ROOT / "detection"))
    from dgp import default_config
    from surface_configs import world_config

    base = default_config(19).to_dict()
    surface_default = world_config("mid", "mid", 0.05, 19, 8000).to_dict()
    base["n_entities"] = 8000
    base["label"] = surface_default["label"]
    assert base == surface_default
    cpath = ROOT / "detection/confirmatory/results/confirmatory_summary.json"
    c = json.loads(cpath.read_text())
    rs = c["replicates"]
    assert len(rs) == 52 and all(r["status"] == "OK" for r in rs)
    assert len({r["train_seed"] for r in rs}) == 52
    assert not ({r["train_seed"] for r in rs} & {r["test_seed"] for r in rs})
    for r in rs:
        for m in ("gboost", "logit"):
            for tier in ("T2", "T3", "T4"):
                d = r["models"][m][tier]
                assert d["MissedPer10k"] == r["N_positive_test"] - d["TP"]
            for hi in ("T3", "T4"):
                assert (
                    r[f"delta_miss_T2_minus_{hi}_{m}"]
                    == r["models"][m][hi]["TP"] - r["models"][m]["T2"]["TP"]
                )
        assert (
            r["DiD_L0_minus_L3"]
            == r["delta_miss_T2_minus_T4_logit"] - r["delta_miss_T2_minus_T4_gboost"]
        )
    conf = {}
    for m in ("logit", "gboost"):
        conf[m] = {
            tier: ci([r["models"][m][tier]["MissedPer10k"] for r in rs])
            for tier in ("T2", "T3", "T4")
        }
        for key, hi, lo in [("dI", "T2", "T3"), ("dW", "T3", "T4"), ("dF", "T2", "T4")]:
            conf[m][key] = ci(
                [
                    r["models"][m][hi]["MissedPer10k"]
                    - r["models"][m][lo]["MissedPer10k"]
                    for r in rs
                ]
            )
    conf["moderation"] = ci([r["DiD_L0_minus_L3"] for r in rs])
    conf["H1_verdict"] = "ESTIMATE_ONLY"
    conf["H2"] = gated_did([r["DiD_L0_minus_L3"] for r in rs], "ESTIMATE_ONLY")
    assert conf["H2"]["confirmatory"] is False
    write_rows(
        "independent.tex",
        [
            [{"logit": "Logistic", "gboost": "Boosted"}[m]]
            + [f(conf[m][k], 2) for k in ("dI", "dW", "dF")]
            for m in ("logit", "gboost")
        ],
    )
    write_rows(
        "levels.tex",
        [
            [{"logit": "Logistic", "gboost": "Boosted"}[m]]
            + [f(conf[m][k], 2) for k in ("T2", "T3", "T4")]
            for m in ("logit", "gboost")
        ],
    )
    packs = {
        name: [
            json.loads(x)
            for x in (ROOT / f"detection/results/ladder/frozen/{name}.jsonl")
            .read_text()
            .splitlines()
        ]
        for name in ("default_ladder", "surface", "prevalence", "controls")
    }
    assert {k: len(v) for k, v in packs.items()} == {
        "default_ladder": 20,
        "surface": 80,
        "prevalence": 20,
        "controls": 2,
    }
    allr = sum(packs.values(), [])
    assert len({r["seed"] for r in allr}) == 122
    assert all(r["gate"] == "pass" for r in allr)
    assert all(r["config"]["base_rate"] == 0.05 for r in packs["surface"])
    modelnames = {
        "M1_linear": "Linear",
        "M2_additive": "Additive",
        "M3_forest": "Forest",
        "M4_boosted": "Boosted",
    }
    groups = {}
    for r in allr:
        groups.setdefault(r["label"], []).append(r)
        for d in r["models"].values():
            for metric in ("ap", "recall_at_budget"):
                z = d["increments"][metric]
                assert abs(z["dF"] - z["dI"] - z["dW"]) < 1e-12
                assert (
                    abs(z["dF"] - d["tiers"]["T4"][metric] + d["tiers"]["T2"][metric])
                    < 1e-12
                )
    ladder = {}
    for label, rr in sorted(groups.items()):
        ladder[label] = {}
        for m in modelnames:
            dd = {}
            for metric in ("ap", "recall_at_budget"):
                for inc in ("dI", "dW", "dF", "dL"):
                    dd[inc + "_" + metric] = ci(
                        [r["models"][m]["increments"][metric][inc] for r in rr]
                    )
            for tier in ("T1", "T2", "T3", "T4"):
                for metric in ("ap", "missed_per_10k"):
                    dd[tier + "_" + metric] = ci(
                        [r["models"][m]["tiers"][tier][metric] for r in rr]
                    )
            dd["dF_missed_per_10k"] = ci(
                [
                    r["models"][m]["tiers"]["T2"]["missed_per_10k"]
                    - r["models"][m]["tiers"]["T4"]["missed_per_10k"]
                    for r in rr
                ]
            )
            ladder[label][m] = dd
    default = ladder["b=mid|s=mid|p=0.05"]
    pairs = {}
    for a, b in [
        ("M1_linear", "M2_additive"),
        ("M2_additive", "M4_boosted"),
        ("M1_linear", "M4_boosted"),
    ]:
        pairs[a + " minus " + b] = ci(
            [
                r["models"][a]["increments"]["ap"]["dF"]
                - r["models"][b]["increments"]["ap"]["dF"]
                for r in packs["default_ladder"]
            ]
        )
    write_rows(
        "ladder.tex",
        [
            [modelnames[m]] + [f(default[m][k], 4) for k in ("dI_ap", "dW_ap", "dF_ap")]
            for m in modelnames
        ],
    )
    write_rows(
        "ladder-ops.tex",
        [
            [modelnames[m]]
            + [
                f(default[m][k], 2)
                for k in ("T2_missed_per_10k", "T4_missed_per_10k", "dF_missed_per_10k")
            ]
            for m in modelnames
        ],
    )
    write_rows(
        "surface.tex",
        [
            [b.title(), s.title()]
            + [f(ladder[f"b={b}|s={s}|p=0.05"][m]["dF_ap"], 3) for m in modelnames]
            for b in ("low", "mid", "high")
            for s in ("low", "mid", "high")
        ],
    )
    write_rows(
        "prevalence.tex",
        [
            [f"{100 * p:.0f}\\%"]
            + [f(ladder[f"b=mid|s=mid|p={p:g}"][m]["dF_ap"], 3) for m in modelnames]
            for p in (0.01, 0.03, 0.05)
        ],
    )
    write_rows(
        "positive.tex",
        [
            [modelnames[m], f(ladder["surveillance_strong"][m]["dF_ap"], 3)]
            for m in modelnames
        ],
    )
    pilot = json.loads(
        (ROOT / "detection/confirmatory/results/pilot_summary.json").read_text()
    )
    write_rows(
        "resolution.tex",
        [
            [
                str(z["k_star"]),
                f"{z['L3']['delta_miss']:.2f}",
                f"{z['L0']['delta_miss']:.2f}",
                f"{100 * z['L3']['sat_frac']:.0f}\\%",
            ]
            for z in pilot["resolution_scan"]
            if z["k_star"] in (50, 100, 200, 300, 400, 500, 625, 1000)
        ],
    )
    replaypath = OUT / "confirmatory-rerun/confirmatory_summary.json"
    replay = {}
    if replaypath.exists():
        new = json.loads(replaypath.read_text())
        replay = {
            "replicates_exactly_equal": rs == new["replicates"],
            "H2_confirmatory": new["H2_DiD"]["confirmatory"],
            "R_ok": new["R_ok"],
        }
        assert replay["replicates_exactly_equal"] and not replay["H2_confirmatory"]
    negpath = OUT / "permutation-controls.jsonl"
    negative = {}
    if negpath.exists():
        neg = [json.loads(x) for x in negpath.read_text().splitlines()]
        assert len({r["seed"] for r in neg}) == len(neg)
        assert not ({r["seed"] for r in neg} & {r["seed"] for r in allr})
        negative["n"] = len(neg)
        negative["failed"] = sum(r["status"] != "OK" for r in neg)
        for m in modelnames:
            z = [r["models"][m]["tiers"]["T4"] for r in neg if r["status"] == "OK"]
            if z:
                negative[m] = {
                    "auc_min": min(a["auc"] for a in z),
                    "auc_max": max(a["auc"] for a in z),
                    "auc_mean": float(np.mean([a["auc"] for a in z])),
                }
        if len(neg) == 22:
            write_rows(
                "negative.tex",
                [
                    [
                        modelnames[m],
                        f"{negative[m]['auc_mean']:.3f}",
                        f"{negative[m]['auc_min']:.3f}--{negative[m]['auc_max']:.3f}",
                    ]
                    for m in modelnames
                ],
            )
    sources = [cpath] + [
        ROOT / f"detection/results/ladder/frozen/{name}.jsonl" for name in packs
    ]
    sources += [
        p for p in (negpath, replaypath, OUT / "ladder-replay.json") if p.exists()
    ]
    out = {
        "date": "2026-09-12",
        "interval": "90% pointwise Student-t mean intervals",
        "confirmatory": conf,
        "confirmatory_reproduction": replay,
        "ladder": ladder,
        "capacity_contrasts": pairs,
        "permutation_controls": negative,
        "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in sources},
        "checks": {
            "raw_ladder_units": 122,
            "gate_exclusions": 0,
            "paired_arithmetic": True,
            "surface_actual_prevalence": 0.05,
            "surface_protocol_prevalence": 0.03,
        },
        "protocol_deviations": [
            "H2 gate bypass corrected; H2 descriptive",
            "Surface ran at 5%, protocol named 3%",
            "E2 factorial and default within-replicate bootstrap absent",
            "Per-world ladder permutation controls added retrospectively 12 Sep",
        ],
    }
    (OUT / "adjudicated-results.json").write_text(json.dumps(out, indent=2) + "\n")
    print("Independent-population results", json.dumps(conf, indent=2))
    print("Capacity contrasts", json.dumps(pairs, indent=2))
    print("Replay", replay)
    print("Negative controls", negative)


if __name__ == "__main__":
    main()

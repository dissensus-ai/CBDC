"""EXPLORATORY noisy-linkage arm: identity increments under corrupted linkage.

Question: as wallet->entity linkage degrades (false splits, false merges),
does the identity increment (T2 -> T3, T2 -> T4 missed illicit entities per
10,000 at the fixed review budget) grow -- i.e. does identity access
substitute for broken linkage?

Each replicate draws an independent train world and test world (same seed
scheme as confirmatory/replicate.py: test seed = train seed + 10,000,003),
then for every (eps_split, eps_merge) cell corrupts BOTH worlds' linkage
independently at that cell's rates, builds T2/T3/T4 features on the observed
clusters, fits logit and gboost on the training clusters, ranks the test
clusters, and scores the top k under both the entity-coverage and the
conservative rule. Misses are always per TRUE entity.

Nothing this script writes is a reported result. Seeds outside the DEV range
700001-700999 are refused unless --addendum-lock names a file recording the
seed base, i.e. a written addendum has frozen the run.

    python3 run_noisy_linkage.py --seed-base 700501 --R 3 --n-train 2000 \
        --n-test 2000 --out-dir _dev_smoke_linkage
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import traceback  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "confirmatory"))

from degeneracy_audit import DegeneracyError, audit  # noqa: E402
from detection_experiment import _models  # noqa: E402
from features import T1_WALLET_COLS, TIER_COLS  # noqa: E402
from endpoint import budget_for  # noqa: E402
from inference import mean_ci  # noqa: E402
from replicate import SKLEARN_SEED_MAX, _world, model_seed  # noqa: E402
import noisy_linkage as nl  # noqa: E402

TIERS = ("T2", "T3", "T4")
MODELS = ("gboost", "logit")
RULES = ("coverage", "conservative")
SEED_OFFSET = 10_000_003          # same test-stream offset as confirmatory
DEFAULT_EPS = (0.0, 0.05, 0.10, 0.20, 0.30)

DEV_SEED_LO, DEV_SEED_HI = 700001, 700999
# Seeds spent elsewhere in this project, as inclusive ranges. The dated bases
# are blocks (base + replicate index); 1000 covers every block ever run.
FORBIDDEN_SEED_RANGES = (
    (900001, 900020),                       # pilot, burned
    (20260707, 20260707),                   # development seed
    (2026080501, 2026080501 + 999),         # confirmatory
    (2026081951, 2026081951 + 999),         # ladder
    (2026082051, 2026082051 + 999),         # ladder
    (2026091201, 2026091201 + 999),         # September diagnostics
)


class SeedGuardError(ValueError):
    pass


def replicate_seeds(seed_base: int, R: int):
    return [(seed_base + i, seed_base + i + SEED_OFFSET) for i in range(R)]


def check_seeds(seed_base: int, R: int, addendum_lock=None):
    """Refuse any seed that is not DEV unless an addendum lock names it.

    Rules, all enforced before anything is generated:
      * train seeds must lie in DEV 700001-700999, unless --addendum-lock is a
        file that exists and contains the seed base as a token;
      * no train or test seed may fall in a range spent elsewhere, nor in the
        test stream (range + SEED_OFFSET) of one, even under a lock;
      * every derived sklearn random_state must be <= 2**32-1 with no
        wrap-around, so the model seed is the protocol seed itself.
    """
    if R < 1:
        raise SeedGuardError("R must be >= 1")
    pairs = replicate_seeds(seed_base, R)
    train = [a for a, _ in pairs]
    if addendum_lock is None:
        if train[0] < DEV_SEED_LO or train[-1] > DEV_SEED_HI:
            raise SeedGuardError(
                f"seeds {train[0]}..{train[-1]} leave the DEV range "
                f"{DEV_SEED_LO}-{DEV_SEED_HI}; a non-DEV run needs "
                f"--addendum-lock <file recording the seed base>")
    else:
        if not os.path.isfile(addendum_lock):
            raise SeedGuardError(f"addendum lock {addendum_lock} not found")
        with open(addendum_lock) as f:
            tokens = set(f.read().replace(",", " ").replace(":", " ").split())
        if str(seed_base) not in tokens:
            raise SeedGuardError(
                f"addendum lock {addendum_lock} does not record seed base "
                f"{seed_base}")
    spent = [(lo, hi) for lo, hi in FORBIDDEN_SEED_RANGES]
    spent += [(lo + SEED_OFFSET, hi + SEED_OFFSET)
              for lo, hi in FORBIDDEN_SEED_RANGES]
    for s in [x for p in pairs for x in p]:
        for lo, hi in spent:
            if lo <= s <= hi:
                raise SeedGuardError(f"seed {s} is spent elsewhere "
                                     f"({lo}-{hi})")
        if not 0 <= s <= SKLEARN_SEED_MAX or model_seed(s) != s:
            raise SeedGuardError(f"seed {s} exceeds sklearn random_state "
                                 f"ceiling 2**32-1")
    return pairs


def eps_grid(eps=DEFAULT_EPS, kind="axes"):
    """(eps_split, eps_merge) cells.

    axes: oracle, split-only axis, merge-only axis, and the diagonal
          (eps_s = eps_m) -- 1 + 3 * (len(eps) - 1) cells.
    full: the full factorial.
    """
    eps = sorted(set(float(e) for e in eps))
    if kind == "full":
        return [(s, m) for s in eps for m in eps]
    cells = [(0.0, 0.0)]
    for e in eps:
        if e == 0.0:
            continue
        cells += [(e, 0.0), (0.0, e), (e, e)]
    return cells


def _fit_score(tr_X, tr_y, te_X, model_name, seed):
    m = _models(model_seed(seed))[model_name]
    m.fit(tr_X, tr_y)
    return m.predict_proba(te_X)[:, 1]


def run_linkage_replicate(replicate_id, train_seed, test_seed, *, n_train,
                          n_test, k_star, cells, split_mode="wallet",
                          merge_mode="uniform", attr_rule="max_risk",
                          coverage_min_frac=0.0):
    """One replicate over every cell. Failures are recorded, never redrawn."""
    rec = {"replicate_id": int(replicate_id), "train_seed": int(train_seed),
           "test_seed": int(test_seed), "status": "OK", "cells": []}
    t0 = time.time()
    try:
        tr_data, tr_wf, tr_ef, _ = _world(train_seed, n_train)
        te_data, te_wf, te_ef, _ = _world(test_seed, n_test)
        # gate on the ORACLE training world, as confirmatory/replicate.py does:
        # the audit is about the generator, not about the resolver
        try:
            report = audit(tr_ef, tr_wf, TIER_COLS, T1_WALLET_COLS)
            rec["audit"] = {k: report[k] for k in
                            ("gate1_pass", "gate2_pass", "worst_feature",
                             "worst_auc")}
        except DegeneracyError as e:
            rec["status"] = "FAIL_AUDIT"
            rec["error"] = str(e)[:400]
            return rec
        y_te = te_data["entities"].is_launderer.to_numpy()
        n_pos = int(y_te.sum())
        rec["N_positive_test"] = n_pos
        if n_pos == 0 or n_pos == len(y_te):
            rec["status"] = "FAIL_LABEL"
            rec["error"] = f"degenerate test labels: {n_pos}/{len(y_te)}"
            return rec
        # budget in CLUSTER alerts, sized on the TRUE population (protocol v3
        # 7.3/7.5): the same k at every cell, so a cell's misses move only
        # because linkage changed, never because the budget did
        k = budget_for(k_star, len(y_te))
        rec["k"] = k
        wx = {}
        if merge_mode == "counterparty":
            wx = {"tr": nl.counterparty_incidence(tr_data["transactions"],
                                                  tr_data["wallets"]),
                  "te": nl.counterparty_incidence(te_data["transactions"],
                                                  te_data["wallets"])}
        for eps_s, eps_m in cells:
            tc = time.time()
            links, feats = {}, {}
            for tag, data, wf, seed in (("tr", tr_data, tr_wf, train_seed),
                                        ("te", te_data, te_wf, test_seed)):
                link = nl.corrupt_linkage(
                    data["wallets"], eps_split=eps_s, eps_merge=eps_m,
                    seed=seed, split_mode=split_mode, merge_mode=merge_mode,
                    wallet_x=wx.get(tag))
                ef, _ = nl.observed_entity_features(data, wf, link, attr_rule)
                links[tag], feats[tag] = link, ef
            y_tr = feats["tr"].is_launderer.to_numpy()
            cell = {"eps_split": eps_s, "eps_merge": eps_m,
                    "linkage_train": links["tr"].stats,
                    "linkage_test": links["te"].stats,
                    "train_cluster_prevalence": float(y_tr.mean()),
                    "models": {}}
            for model_name in MODELS:
                out = {}
                for tier in TIERS:
                    cols = TIER_COLS[tier]
                    s = _fit_score(feats["tr"][cols].to_numpy(dtype=float),
                                   y_tr,
                                   feats["te"][cols].to_numpy(dtype=float),
                                   model_name, train_seed)
                    out[tier] = {
                        "coverage": nl.score_coverage(
                            y_te, links["te"], s, k, coverage_min_frac),
                        "conservative": nl.score_conservative(
                            y_te, links["te"], s, k),
                    }
                for rule in RULES:
                    m2 = out["T2"][rule]["MissedPer10k"]
                    out[f"delta_T2_minus_T4_{rule}"] = (
                        m2 - out["T4"][rule]["MissedPer10k"])
                    out[f"delta_T2_minus_T3_{rule}"] = (
                        m2 - out["T3"][rule]["MissedPer10k"])
                cell["models"][model_name] = out
            cell["seconds"] = round(time.time() - tc, 2)
            rec["cells"].append(cell)
        rec["seconds"] = round(time.time() - t0, 2)
        return rec
    except Exception as e:  # noqa: BLE001 - any crash is a counted failure
        rec["status"] = "FAIL_NUM"
        rec["error"] = f"{type(e).__name__}: {e}"[:400]
        rec["traceback"] = traceback.format_exc()[-1200:]
        return rec


def _cell_key(c):
    return (c["eps_split"], c["eps_merge"])


def summarize(records, alpha=0.10):
    """Per cell x model x rule: t-intervals over replicates for each tier's
    absolute misses, the T2->T3 and T2->T4 increments, and -- the arm's key
    contrast -- each increment's change relative to the oracle cell of the
    SAME replicate (paired, so between-world variation cancels):

        growth_T4(eps) = delta_T2_minus_T4(eps) - delta_T2_minus_T4(0, 0)

    growth > 0 means identity is worth more when linkage is worse. Also the
    T2 degradation Missed_T2(eps) - Missed_T2(oracle).
    """
    ok = [r for r in records if r["status"] == "OK"]
    rows = []
    if not ok:
        return rows
    for cell0 in ok[0]["cells"]:
        key = _cell_key(cell0)
        for model_name in MODELS:
            for rule in RULES:
                vals = {}
                for r in ok:
                    cmap = {_cell_key(c): c for c in r["cells"]}
                    if key not in cmap or (0.0, 0.0) not in cmap:
                        continue
                    m = cmap[key]["models"][model_name]
                    o = cmap[(0.0, 0.0)]["models"][model_name]
                    for tier in TIERS:
                        vals.setdefault(f"missed_{tier}", []).append(
                            m[tier][rule]["MissedPer10k"])
                    for d in ("T4", "T3"):
                        dk = f"delta_T2_minus_{d}_{rule}"
                        vals.setdefault(f"delta_T2_minus_{d}", []).append(m[dk])
                        vals.setdefault(f"growth_T2_minus_{d}_vs_oracle",
                                        []).append(m[dk] - o[dk])
                    vals.setdefault("T2_degradation_vs_oracle", []).append(
                        m["T2"][rule]["MissedPer10k"]
                        - o["T2"][rule]["MissedPer10k"])
                    for tag in ("linkage_test",):
                        for sk in ("frac_entities_split",
                                   "frac_entities_in_merged_cluster"):
                            vals.setdefault(f"{tag}_{sk}", []).append(
                                cmap[key][tag][sk])
                for name, v in vals.items():
                    ci = mean_ci(v, alpha)
                    rows.append({"eps_split": key[0], "eps_merge": key[1],
                                 "model": model_name, "rule": rule,
                                 "quantity": name, "mean": ci["mean"],
                                 "sd": ci["sd"], "R_ok": ci["R_ok"],
                                 "ci_lo": ci["ci_lo"], "ci_hi": ci["ci_hi"]})
    return rows


def _job(a):
    i, (tr, te), kw = a
    return run_linkage_replicate(i, tr, te, **kw)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed-base", type=int, required=True)
    ap.add_argument("--R", type=int, required=True)
    ap.add_argument("--addendum-lock", default=None,
                    help="file that records the seed base; required outside "
                         "DEV seeds 700001-700999")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-train", type=int, default=8000)
    ap.add_argument("--n-test", type=int, default=10000)
    ap.add_argument("--k-star", type=int, default=500)
    ap.add_argument("--eps", type=float, nargs="+", default=list(DEFAULT_EPS))
    ap.add_argument("--grid", choices=("axes", "full"), default="axes")
    ap.add_argument("--split-mode", choices=nl.SPLIT_MODES, default="wallet")
    ap.add_argument("--merge-mode", choices=nl.MERGE_MODES, default="uniform")
    ap.add_argument("--attr-rule", choices=nl.ATTR_RULES, default="max_risk")
    ap.add_argument("--coverage-min-frac", type=float, default=0.0)
    ap.add_argument("--workers", type=int,
                    default=min(8, max(1, (os.cpu_count() or 4) - 2)))
    args = ap.parse_args(argv)

    try:
        pairs = check_seeds(args.seed_base, args.R, args.addendum_lock)
    except SeedGuardError as e:
        print(f"REFUSING TO RUN: {e}", file=sys.stderr)
        sys.exit(2)
    out_dir = args.out_dir
    rec_path = os.path.join(out_dir, "records.jsonl")
    if os.path.exists(rec_path):
        print(f"REFUSING TO RUN: {rec_path} exists; use a fresh --out-dir",
              file=sys.stderr)
        sys.exit(2)
    os.makedirs(out_dir, exist_ok=True)

    cells = eps_grid(args.eps, args.grid)
    kw = dict(n_train=args.n_train, n_test=args.n_test, k_star=args.k_star,
              cells=cells, split_mode=args.split_mode,
              merge_mode=args.merge_mode, attr_rule=args.attr_rule,
              coverage_min_frac=args.coverage_min_frac)
    config = {"seed_base": args.seed_base, "R": args.R,
              "seed_offset": SEED_OFFSET,
              "dev_seeds": args.addendum_lock is None,
              "addendum_lock": args.addendum_lock, "cells": cells,
              **{k: v for k, v in kw.items() if k != "cells"},
              "workers": args.workers}
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"noisy-linkage arm: R={args.R} seeds {pairs[0][0]}..{pairs[-1][0]}"
          f"  cells={len(cells)}  n_train={args.n_train} n_test={args.n_test}"
          f"  k*={args.k_star}  {args.split_mode}/{args.merge_mode}/"
          f"{args.attr_rule}", flush=True)

    t0 = time.time()
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex, \
            open(rec_path, "w") as fout:
        futs = [ex.submit(_job, (i, p, kw)) for i, p in enumerate(pairs)]
        for done, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            records.append(rec)
            fout.write(json.dumps(rec, default=float) + "\n")
            fout.flush()
            print(f"  [{done}/{len(pairs)}] rep {rec['replicate_id']:>3} "
                  f"{rec['status']:<10} {rec.get('seconds', 0):>7.1f}s",
                  flush=True)
    wall = time.time() - t0
    records.sort(key=lambda r: r["replicate_id"])
    rows = summarize(records)
    import csv
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["empty"])
        w.writeheader()
        w.writerows(rows)
    ok = [r for r in records if r["status"] == "OK"]
    timing = {"wall_seconds": round(wall, 1), "workers": args.workers,
              "R_ok": len(ok), "R_failed": len(records) - len(ok),
              "replicate_seconds": [r.get("seconds") for r in records],
              "mean_cell_seconds": (float(np.mean(
                  [c["seconds"] for r in ok for c in r["cells"]]))
                  if ok else None)}
    with open(os.path.join(out_dir, "timing.json"), "w") as f:
        json.dump(timing, f, indent=2)
    print(f"\n{len(ok)}/{len(records)} OK in {wall:.0f}s wall -> {out_dir}",
          flush=True)


if __name__ == "__main__":
    main()

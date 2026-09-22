"""EXPLORATORY noisy-linkage arm (E11): identity increments under corrupted linkage.

Question: as wallet->entity linkage degrades (false splits, false merges),
does the identity increment (T2 -> T3, T2 -> T4 missed illicit entities per
10,000 at the fixed review budget) grow -- i.e. does identity access
substitute for broken linkage? And, next to that contrast, how much worse do
the tiers get in absolute terms?

Each replicate draws an independent train world and test world (same seed
scheme as confirmatory/replicate.py: test seed = train seed + 10,000,003),
then for every (split_mode, eps_split, eps_merge) cell corrupts BOTH worlds'
linkage independently at that cell's rates, and for every attribute rule
builds T2/T3/T4 features on the observed clusters, fits each model on the
training clusters, ranks the test clusters, and scores the top k under both
the entity-coverage and the conservative rule. Misses are always per TRUE
entity. Identity attributes are attached through TRUE wallet ownership
(semi-oracle): this arm tests noisy aggregation with stipulated attribute
access, not identity discovery or linkage repair.

Nothing this script writes is a reported result. Seeds outside the DEV range
700001-700999 are refused unless --addendum-lock names a file recording the
seed base, i.e. a written addendum has frozen the run. Every run writes a
single `run_spec.json` with the fields a lock is meant to pin.

    python3 run_noisy_linkage.py --seed-base 700501 --R 3 --n-train 2000 \
        --n-test 2000 --out-dir _dev_smoke_linkage
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
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

ARM = "E11"
TIERS = ("T2", "T3", "T4")
MODELS = ("gboost", "logit")
RULES = ("coverage", "conservative")
SEED_OFFSET = 10_000_003          # same test-stream offset as confirmatory
DEFAULT_EPS = (0.0, 0.05, 0.10, 0.20, 0.30)
V2_EPS = (0.05, 0.15)             # protocol v2 section 6 subset
GRIDS = ("axes", "full", "v2", "axes+v2")
DEFAULT_ATTR_RULES = ("attrwise_max_risk", "majority")
REQUIREMENTS = os.path.join(os.path.dirname(HERE), "requirements.txt")

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


def replicate_seeds(seed_base: int, R: int, offset: int = SEED_OFFSET):
    return [(seed_base + i, seed_base + i + offset) for i in range(R)]


def check_seeds(seed_base: int, R: int, addendum_lock=None,
                offset: int = SEED_OFFSET):
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
    if offset < R:
        raise SeedGuardError("test offset must exceed R (streams overlap)")
    pairs = replicate_seeds(seed_base, R, offset)
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


def eps_grid(eps=DEFAULT_EPS, kind="axes", split_mode="wallet"):
    """(split_mode, eps_split, eps_merge) cells.

    axes:    oracle, split-only axis, merge-only axis, and the diagonal
             (eps_s = eps_m) -- 1 + 3 * (len(eps) - 1) cells, all with
             `split_mode`.
    full:    the full factorial over `eps`, with `split_mode`.
    v2:      protocol v2 section 6 as written: bipartition splits x merges
             at eps in {0.05, 0.15}^2, plus the oracle -- 5 cells.
    axes+v2: both, sharing ONE oracle cell (at eps = 0 every split mode is
             the identity, so a second oracle would repeat the same numbers)
             -- 17 cells at the default eps.
    The oracle is always first; paired contrasts reference it.
    """
    if kind not in GRIDS:
        raise ValueError(f"grid must be one of {GRIDS}")
    eps = sorted(set(float(e) for e in eps))
    cells = [(split_mode, 0.0, 0.0)]
    if kind == "full":
        cells += [(split_mode, s, m) for s in eps for m in eps
                  if (s, m) != (0.0, 0.0)]
    if kind in ("axes", "axes+v2"):
        for e in eps:
            if e == 0.0:
                continue
            cells += [(split_mode, e, 0.0), (split_mode, 0.0, e),
                      (split_mode, e, e)]
    if kind in ("v2", "axes+v2"):
        cells += [("bipartition", s, m) for s in V2_EPS for m in V2_EPS]
    return cells


def _fit_score(tr_X, tr_y, te_X, model_name, seed):
    m = _models(model_seed(seed))[model_name]
    m.fit(tr_X, tr_y)
    return m.predict_proba(te_X)[:, 1]


def run_linkage_replicate(replicate_id, train_seed, test_seed, *, n_train,
                          n_test, k_star, cells, merge_mode="uniform",
                          attr_rules=DEFAULT_ATTR_RULES, models=MODELS,
                          coverage_min_frac=0.0):
    """One replicate over every cell x attribute rule. Failures are recorded,
    never redrawn. `cells` are (split_mode, eps_split, eps_merge) triples."""
    attr_rules = [nl.canonical_attr_rule(a) for a in attr_rules]
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
        for split_mode, eps_s, eps_m in cells:
            tc = time.time()
            links = {}
            for tag, data, seed in (("tr", tr_data, train_seed),
                                    ("te", te_data, test_seed)):
                links[tag] = nl.corrupt_linkage(
                    data["wallets"], eps_split=eps_s, eps_merge=eps_m,
                    seed=seed, split_mode=split_mode, merge_mode=merge_mode,
                    wallet_x=wx.get(tag))
            cell = {"split_mode": split_mode, "eps_split": eps_s,
                    "eps_merge": eps_m,
                    "linkage_train": links["tr"].stats,
                    "linkage_test": links["te"].stats, "attr": {}}
            for attr_rule in attr_rules:
                feats = {tag: nl.observed_entity_features(
                            data, wf, links[tag], attr_rule)[0]
                         for tag, data, wf in (("tr", tr_data, tr_wf),
                                               ("te", te_data, te_wf))}
                y_tr = feats["tr"].is_launderer.to_numpy()
                block = {"train_cluster_prevalence": float(y_tr.mean()),
                         "models": {}}
                for model_name in models:
                    out = {}
                    for tier in TIERS:
                        cols = TIER_COLS[tier]
                        s = _fit_score(
                            feats["tr"][cols].to_numpy(dtype=float), y_tr,
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
                    block["models"][model_name] = out
                cell["attr"][attr_rule] = block
            cell["seconds"] = round(time.time() - tc, 2)
            rec["cells"].append(cell)
        rec["seconds"] = round(time.time() - t0, 2)
        return rec
    except Exception as e:  # noqa: BLE001 - any crash is a counted failure
        rec["status"] = "FAIL_NUM"
        rec["error"] = f"{type(e).__name__}: {e}"[:400]
        rec["traceback"] = traceback.format_exc()[-1200:]
        return rec


# ---------------------------------------------------------------------------
# summaries
# ---------------------------------------------------------------------------

def _cell_key(c):
    return (c["split_mode"], c["eps_split"], c["eps_merge"])


def _oracle(cells):
    for c in cells:
        if c["eps_split"] == 0.0 and c["eps_merge"] == 0.0:
            return c
    return None


def _paired_values(ok, key, attr_rule, model_name, rule):
    """Per-replicate values for one (cell, attr rule, model, scoring rule),
    each paired with the SAME replicate's oracle cell."""
    vals = {}
    for r in ok:
        cmap = {_cell_key(c): c for c in r["cells"]}
        orc = _oracle(r["cells"])
        if key not in cmap or orc is None:
            continue
        cell = cmap[key]
        m = cell["attr"][attr_rule]["models"][model_name]
        o = orc["attr"][attr_rule]["models"][model_name]

        def add(name, v):
            vals.setdefault(name, []).append(v)
        for tier in TIERS:
            mt = m[tier][rule]["MissedPer10k"]
            add(f"missed_{tier}", mt)
            # absolute deterioration of each tier vs the oracle, paired
            add(f"{tier}_degradation_vs_oracle",
                mt - o[tier][rule]["MissedPer10k"])
        for d in ("T4", "T3"):
            dk = f"delta_T2_minus_{d}_{rule}"
            add(f"delta_T2_minus_{d}", m[dk])
            add(f"growth_T2_minus_{d}_vs_oracle", m[dk] - o[dk])
        for sk in ("frac_entities_split", "frac_entities_in_merged_cluster"):
            add(f"linkage_test_{sk}", cell["linkage_test"][sk])
    return vals


def _iter_slices(ok):
    if not ok:
        return
    c0 = ok[0]["cells"][0]
    models = list(c0["attr"][next(iter(c0["attr"]))]["models"])
    for cell in ok[0]["cells"]:
        for attr_rule in cell["attr"]:
            for model_name in models:
                for rule in RULES:
                    yield _cell_key(cell), attr_rule, model_name, rule


def summarize(records, alpha=0.10):
    """Long table: one row per (cell, attr rule, model, scoring rule,
    quantity) with the mean over replicates and a two-sided (1-alpha)
    Student-t interval. Paired quantities (`*_vs_oracle`) are differences
    within replicate against that replicate's oracle cell:

        growth_T2_minus_T4_vs_oracle = delta_T2_minus_T4(eps) - delta(0, 0)
        T{2,3,4}_degradation_vs_oracle = missed_T{k}(eps) - missed_T{k}(0, 0)

    growth > 0: identity is worth more when linkage is worse. A growth
    compatible with zero can coexist with every tier getting much worse --
    the degradation rows are what show that.
    """
    ok = [r for r in records if r["status"] == "OK"]
    rows = []
    for key, attr_rule, model_name, rule in _iter_slices(ok):
        vals = _paired_values(ok, key, attr_rule, model_name, rule)
        for name, v in vals.items():
            ci = mean_ci(v, alpha)
            rows.append({"split_mode": key[0], "eps_split": key[1],
                         "eps_merge": key[2], "attr_rule": attr_rule,
                         "model": model_name, "rule": rule,
                         "quantity": name, "mean": ci["mean"], "sd": ci["sd"],
                         "R_ok": ci["R_ok"], "ci_lo": ci["ci_lo"],
                         "ci_hi": ci["ci_hi"],
                         "half_width": ci.get("half_width")})
    return rows


PRIMARY_QUANTITIES = (
    ("G_T4", "growth_T2_minus_T4_vs_oracle"),
    ("G_T3", "growth_T2_minus_T3_vs_oracle"),
    ("deg_T2", "T2_degradation_vs_oracle"),
    ("deg_T3", "T3_degradation_vs_oracle"),
    ("deg_T4", "T4_degradation_vs_oracle"),
)


def primary_summary(rows):
    """Wide primary block: per (cell, attr rule, model, scoring rule), the
    key contrast G next to the absolute deterioration of every tier, each as
    mean [ci_lo, ci_hi] with its achieved half-width, plus realised test-
    world error rates. G is never reported without these."""
    idx = {}
    for r in rows:
        k = (r["split_mode"], r["eps_split"], r["eps_merge"], r["attr_rule"],
             r["model"], r["rule"])
        idx.setdefault(k, {})[r["quantity"]] = r
    out = []
    for k, q in idx.items():
        row = dict(zip(("split_mode", "eps_split", "eps_merge", "attr_rule",
                        "model", "rule"), k))
        row["R_ok"] = q["missed_T2"]["R_ok"]
        for short, name in PRIMARY_QUANTITIES:
            for f in ("mean", "ci_lo", "ci_hi", "half_width"):
                row[f"{short}_{f}"] = q[name][f]
        for sk in ("frac_entities_split", "frac_entities_in_merged_cluster"):
            row[f"realised_{sk}"] = q[f"linkage_test_{sk}"]["mean"]
        out.append(row)
    return out


def _fmt(row, short):
    m, lo, hi = (row[f"{short}_{f}"] for f in ("mean", "ci_lo", "ci_hi"))
    if m is None:
        return f"{short} n/a"
    if lo is None:
        return f"{short} {m:+.1f}"
    return f"{short} {m:+.1f} [{lo:+.1f},{hi:+.1f}]"


def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["empty"])
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# run spec: the fields an addendum lock pins, in one place
# ---------------------------------------------------------------------------

def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def _git(*args):
    try:
        return subprocess.run(["git", "-C", HERE, *args], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_head():
    """(HEAD commit, whether detection/ has uncommitted changes). A dirty
    tree means `commit` does not fully identify the code that ran."""
    status = _git("status", "--porcelain", "--", os.path.dirname(HERE))
    return _git("rev-parse", "HEAD"), (None if status is None
                                       else bool(status))


def run_spec(args, cells):
    """Everything that defines a run, as one dict. Field names follow the
    E9-E11 addendum-lock schema (arm, seed_base, R, grid, models) so the lock
    can be adopted once the branches merge; the remaining fields are what it
    exact-matches (n_train, n_test, k_star, offset, flags, requirements
    hash). Here `grid` is the list of [split_mode, eps_split, eps_merge]
    cells, not a list of floats."""
    head, dirty = _git_head()
    return {
        "arm": ARM,
        "seed_base": args.seed_base,
        "R": args.R,
        "n_train": args.n_train,
        "n_test": args.n_test,
        "k_star": args.k_star,
        "test_offset": args.test_offset,
        "grid_name": args.grid,
        "eps": sorted(set(float(e) for e in args.eps)),
        "grid": [list(c) for c in cells],
        "split_mode": args.split_mode,
        "merge_mode": args.merge_mode,
        "attr_rules": [nl.canonical_attr_rule(a) for a in args.attr_rules],
        "models": list(args.models),
        "scoring_rules": list(RULES),
        "coverage_min_frac": args.coverage_min_frac,
        "alpha": 0.10,
        "dev_seeds": args.addendum_lock is None,
        "addendum_lock": args.addendum_lock,
        "requirements_sha256": _sha256(REQUIREMENTS),
        "commit": head,
        "commit_dirty": dirty,
    }


def _job(a):
    i, (tr, te), kw = a
    return run_linkage_replicate(i, tr, te, **kw)


def build_parser():
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
    ap.add_argument("--test-offset", type=int, default=SEED_OFFSET)
    ap.add_argument("--eps", type=float, nargs="+", default=list(DEFAULT_EPS))
    ap.add_argument("--grid", choices=GRIDS, default="axes")
    ap.add_argument("--split-mode", choices=nl.SPLIT_MODES, default="wallet",
                    help="split mode of the axes/full cells (v2 cells are "
                         "always bipartition)")
    ap.add_argument("--merge-mode", choices=nl.MERGE_MODES, default="uniform")
    ap.add_argument("--attr-rules", "--attr-rule", dest="attr_rules",
                    nargs="+", default=list(DEFAULT_ATTR_RULES),
                    choices=list(nl.ATTR_RULES) + list(nl.ATTR_RULE_ALIASES))
    ap.add_argument("--models", nargs="+", choices=MODELS,
                    default=list(MODELS))
    ap.add_argument("--coverage-min-frac", type=float, default=0.0)
    ap.add_argument("--workers", type=int,
                    default=min(8, max(1, (os.cpu_count() or 4) - 2)))
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        pairs = check_seeds(args.seed_base, args.R, args.addendum_lock,
                            args.test_offset)
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

    cells = eps_grid(args.eps, args.grid, args.split_mode)
    spec = run_spec(args, cells)
    with open(os.path.join(out_dir, "run_spec.json"), "w") as f:
        json.dump(spec, f, indent=2)
    kw = dict(n_train=args.n_train, n_test=args.n_test, k_star=args.k_star,
              cells=cells, merge_mode=args.merge_mode,
              attr_rules=spec["attr_rules"], models=spec["models"],
              coverage_min_frac=args.coverage_min_frac)
    print(f"noisy-linkage arm: R={args.R} seeds {pairs[0][0]}..{pairs[-1][0]}"
          f"  grid={args.grid} ({len(cells)} cells)  n_train={args.n_train} "
          f"n_test={args.n_test}  k*={args.k_star}  merges={args.merge_mode}"
          f"  attr={','.join(spec['attr_rules'])}", flush=True)

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
    prim = primary_summary(rows)
    _write_csv(os.path.join(out_dir, "summary.csv"), rows)
    _write_csv(os.path.join(out_dir, "summary_primary.csv"), prim)
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
    if prim:
        print("\nprimary block (gboost, conservative): G next to absolute "
              "deterioration, 90% t-intervals", flush=True)
        for r in prim:
            if r["model"] == "gboost" and r["rule"] == "conservative":
                print(f"  {r['split_mode']:<11} s={r['eps_split']:.2f} "
                      f"m={r['eps_merge']:.2f} {r['attr_rule']:<17} "
                      + "  ".join(_fmt(r, s) for s, _ in PRIMARY_QUANTITIES),
                      flush=True)


if __name__ == "__main__":
    main()

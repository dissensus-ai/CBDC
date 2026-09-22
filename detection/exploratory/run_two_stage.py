"""E9 driver: two-stage screening across R independent train/test pairs.

EXPLORATORY. Refuses seeds outside the DEV block 700001-700999 unless a valid
addendum lock for arm E9 is supplied (seed_guard, addendum_lock). Primary cell
per addendum A0: gboost (M4), stage 2 = T4, full-training-world stage-2 model,
recovered share as the ratio of replicate means. Stop rule: the run halts if
more than 10% of replicates fail.

    python3 run_two_stage.py --seed-base 700001 --R 3 --n-train 2000 \
        --n-test 2000 --out-dir _dev_smoke/two_stage

Writes <out-dir>/two_stage_replicates.jsonl (one record per replicate, the
run_replicate record plus rec["two_stage"]) and two_stage_summary.json.
Refuses to overwrite an existing summary.
"""

from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "confirmatory"))

import addendum_lock  # noqa: E402
import seed_guard  # noqa: E402
from _driver_common import MAX_WORKERS, environment, provenance  # noqa: E402
from inference import mean_ci  # noqa: E402
from replicate import run_replicate  # noqa: E402
from two_stage import (DEFAULT_KPRIME_GRID, STAGE2_TIERS,  # noqa: E402
                       TRAIN_VARIANTS, make_hook, recovery_summary)

ARM = "E9"
MODELS = ["gboost", "logit"]   # what run_replicate fits
STOP_FAIL_RATE = 0.10
PRIMARY_CELL = {"model": "gboost", "stage2_tier": "T4", "train_variant": "full"}


def _job(a):
    i, tr, te, kw, hook_kw = a
    t0 = time.time()
    rec = run_replicate(i, tr, te, score_hook=make_hook(**hook_kw), **kw)
    rec["wall_seconds"] = time.time() - t0
    return rec


def _cells(records):
    cells = {}
    for r in records:
        if r["status"] != "OK":
            continue
        for row in r.get("two_stage", []):
            key = (row["model"], row["stage2_tier"], row["train_variant"])
            cells.setdefault(key, []).append((r["replicate_id"], row))
    return cells


def primary_absolute(records, alpha=0.10):
    """THE E9 PRIMARY OUTPUT: missed per 10k against test-time identity lookups
    per 10k (K'), per (model, stage-2 tier, training variant), with 90%
    t-intervals, plus the two single-stage reference points (T2 = no identity
    lookups, hi tier = identity for all n). Makes no ratio, so it is defined
    whatever the sign of the single-stage identity gain."""
    out = []
    for (model, tier, variant), pairs in sorted(_cells(records).items()):
        by_k = {}
        for rid, row in pairs:
            by_k.setdefault(row["kprime_star"], []).append(row)
        rows0 = next(iter(by_k.values()))
        out.append({
            "model": model, "stage2_tier": tier, "train_variant": variant,
            "is_primary_cell": {"model": model, "stage2_tier": tier,
                                "train_variant": variant} == PRIMARY_CELL,
            "single_stage_T2_MissedPer10k": mean_ci(
                [x["miss_T2"] for x in rows0], alpha),
            f"single_stage_{tier}_MissedPer10k": mean_ci(
                [x["miss_hi"] for x in rows0], alpha),
            "curve": [{
                "kprime_star": kps,
                "IdentityLookupsPer10k": (sum(x["DisclosedPer10k"] for x in rows)
                                          / len(rows)),
                "MissedPer10k": mean_ci([x["MissedPer10k"] for x in rows],
                                        alpha),
                "R_rows": len(rows),
                "n_gap_zero": sum(x["gap_flag"] == "gap_zero" for x in rows),
                "n_gap_negative": sum(x["gap_flag"] == "gap_negative"
                                      for x in rows),
            } for kps, rows in sorted(by_k.items())],
        })
    return out


def recovery(records, grid, B, seed, alpha=0.10):
    """Secondary: reference gain + recovered share + K'_q, gated on the sign
    of the single-stage identity gain (two_stage.recovery_summary)."""
    out = {}
    for (model, tier, variant), pairs in sorted(_cells(records).items()):
        reps = {}
        for rid, row in pairs:
            reps.setdefault(rid, {})[row["kprime_star"]] = (
                row["miss_T2"], row["MissedPer10k"], row["miss_hi"])
        out[f"{model}|{tier}|{variant}"] = recovery_summary(
            reps, grid, B=B, seed=seed, alpha=alpha)
    return out


def build_spec(args) -> dict:
    """Everything that can change an E9 number; the lock must match it."""
    return {
        "arm": ARM, "seed_base": args.seed_base, "R": args.R,
        "grid": list(args.kprime_grid), "models": list(MODELS),
        "n_train": args.n_train, "n_test": args.n_test, "k_star": args.k_star,
        "test_seed_offset": seed_guard.TEST_SEED_OFFSET,
        "identity_rng_stream": False,       # confirmatory default world
        "flags": {"stage2_tiers": list(args.stage2_tiers),
                  "train_variants": list(args.train_variants),
                  "stage1_train_scores": args.stage1_train_scores,
                  "alpha": args.alpha, "bootstrap_B": args.bootstrap_B},
        "environment": environment(),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seed-base", type=int, required=True)
    ap.add_argument("--R", type=int, required=True)
    ap.add_argument("--addendum-lock", default=None)
    # D2/D1 of the confirmatory lock: train 8000, test 10000
    ap.add_argument("--n-train", type=int, default=8000)
    ap.add_argument("--n-test", type=int, default=10000)
    ap.add_argument("--k-star", type=int, default=500)
    ap.add_argument("--kprime-grid", type=int, nargs="+",
                    default=list(DEFAULT_KPRIME_GRID))
    ap.add_argument("--stage2-tiers", nargs="+", default=list(STAGE2_TIERS),
                    choices=list(STAGE2_TIERS))
    ap.add_argument("--train-variants", nargs="+", default=["full"],
                    choices=list(TRAIN_VARIANTS))
    ap.add_argument("--stage1-train-scores", default="oof",
                    choices=["insample", "oof"])
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--bootstrap-B", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--print-spec", action="store_true",
                    help="print the run spec (the lock's `spec`) and exit")
    args = ap.parse_args(argv)
    spec = build_spec(args)
    if args.print_spec:
        print(json.dumps(spec, indent=2))
        return
    if args.out_dir is None:
        ap.error("--out-dir is required")

    try:
        mode = seed_guard.check_seeds(
            args.seed_base, args.R, args.addendum_lock,
            lock_spec=spec)
    except seed_guard.SeedGuardError as e:
        print(f"REFUSING TO RUN\n\n{e}", file=sys.stderr)
        sys.exit(2)
    if not 1 <= args.workers <= MAX_WORKERS:
        sys.exit(f"--workers must be in 1..{MAX_WORKERS}")
    if min(args.kprime_grid) < args.k_star:
        sys.exit("every K'* must be >= k*")

    os.makedirs(args.out_dir, exist_ok=True)
    summ_path = os.path.join(args.out_dir, "two_stage_summary.json")
    raw_path = os.path.join(args.out_dir, "two_stage_replicates.jsonl")
    if os.path.exists(summ_path):
        sys.exit(f"{summ_path} exists; refusing to overwrite a finished run")

    seeds = seed_guard.replicate_seeds(args.seed_base, args.R)
    kw = dict(n_train=args.n_train, n_test=args.n_test, k_star=args.k_star)
    hook_kw = dict(kprime_grid=tuple(args.kprime_grid),
                   stage2_tiers=tuple(args.stage2_tiers),
                   train_variants=tuple(args.train_variants),
                   stage1_train_scores=args.stage1_train_scores)
    print(f"two-stage [{mode} seeds] R={args.R} seeds "
          f"{seeds[0][0]}..{seeds[-1][0]} (+{seed_guard.TEST_SEED_OFFSET} test) "
          f"n_train={args.n_train} n_test={args.n_test} k*={args.k_star}",
          flush=True)

    prov = provenance()
    if mode == "ADDENDUM":
        # burn before generating anything: a crash still spends the seeds
        addendum_lock.register_start(ARM, args.seed_base, args.R,
                                     seed_guard.TEST_SEED_OFFSET,
                                     os.path.abspath(args.out_dir),
                                     prov["git_commit"])

    t0 = time.time()
    records = []
    halted = None
    n_fail = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex, \
            open(raw_path, "w") as raw:
        futs = [ex.submit(_job, (i, tr, te, kw, hook_kw))
                for i, (tr, te) in enumerate(seeds)]
        for done, fut in enumerate(as_completed(futs), 1):
            if fut.cancelled():
                continue
            rec = fut.result()
            records.append(rec)
            raw.write(json.dumps(rec, default=float) + "\n")
            raw.flush()
            print(f"  [{done}/{len(seeds)}] rep {rec['replicate_id']:>3} "
                  f"{rec['status']:<10} {rec['wall_seconds']:.1f}s", flush=True)
            n_fail += rec["status"] != "OK"
            if n_fail > STOP_FAIL_RATE * args.R and halted is None:
                halted = (f"{n_fail}/{args.R} replicates failed (> "
                          f"{STOP_FAIL_RATE:.0%}); run halted per addendum A0")
                print(f"\n** {halted} **\n", flush=True)
                for f in futs:
                    f.cancel()
    wall = time.time() - t0

    records.sort(key=lambda r: r["replicate_id"])
    ok = [r for r in records if r["status"] == "OK"]
    # key order is the reading order: the absolute table comes first
    out = {
        "primary_absolute_missed_vs_identity_lookups": {
            "description": "E9 primary output: missed illicit entities per "
                           "10k vs test-time identity lookups per 10k (K'), "
                           "replicate means with 90% t-intervals. Recovery "
                           "shares are a secondary annotation (below).",
            "primary_cell": PRIMARY_CELL,
            "cells": primary_absolute(records, args.alpha),
        },
        "arm": ARM,
        "status": "EXPLORATORY",
        "halted": halted,
        "R": args.R,
        "R_planned": args.R, "R_ok": len(ok),
        "seed_mode": mode,
        "seed_base": args.seed_base,
        "test_seed_offset": seed_guard.TEST_SEED_OFFSET,
        "addendum_lock": args.addendum_lock,
        "run_spec": spec,
        "args": vars(args),
        "provenance": prov,
        "stage2_training_default": "full",
        "failures": [{k: r.get(k) for k in ("replicate_id", "train_seed",
                                            "test_seed", "status", "error")}
                     for r in records if r["status"] != "OK"],
        "wall_seconds_total": wall,
        "wall_seconds_per_replicate": [r["wall_seconds"] for r in records],
        "recovery_secondary": recovery(records, args.kprime_grid,
                                       args.bootstrap_B, args.seed_base,
                                       args.alpha),
    }
    with open(summ_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    if mode == "ADDENDUM":
        addendum_lock.register_complete(args.seed_base)
    print(f"\n{len(ok)}/{len(records)} OK; wall {wall:.1f}s; wrote {summ_path}",
          flush=True)


if __name__ == "__main__":
    main()

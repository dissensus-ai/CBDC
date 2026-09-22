"""E10 driver: T2 -> T4 and T2 -> T3 Delta missed-per-10k along lambda.

EXPLORATORY. Refuses seeds outside the DEV block 700001-700999 unless a valid
addendum lock for arm E10 is supplied (seed_guard, addendum_lock).

For each lambda on the piecewise path (identity_signal) and each replicate r,
one independent train/test pair (train seed = base + r, test seed = base + r +
10000003) is generated with identity_rng_stream=True and run through
replicate.run_replicate; M2 ("additive") is fitted through score_hook. The same
seed pair is used at every lambda and, with the paired identity stream, the
worlds share labels, wallets and transactions: Delta(lambda) contrasts are
paired counterfactuals.

    python3 run_signal_scale.py --seed-base 700401 --R 3 --n-train 2000 \
        --n-test 2000 --out-dir _dev_smoke/e10_paired

Writes <out-dir>/signal_scale_replicates.jsonl and signal_scale_summary.json.
The summary leads with two top-level gain-threshold crossing tables of equal
standing -- T2->T4 (attributes + watchlist) and T2->T3 (attributes only) --
per model x threshold, crossing_mean primary, each reported crossing carrying
the identity-parameter values at that lambda so a threshold has a substantive
meaning; then per-lambda Delta intervals. Stop rule (addendum A0): the run halts if
more than 10% of the R units at any single lambda fail. Refuses to overwrite
an existing summary.
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

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "confirmatory"))

import addendum_lock  # noqa: E402
import seed_guard  # noqa: E402
from _driver_common import MAX_WORKERS, environment, provenance  # noqa: E402
from break_even import (bootstrap_gain_threshold_crossing,  # noqa: E402
                        gain_threshold_crossing)
from identity_signal import (DEFAULT_LAMBDA_GRID, cfg_factory,  # noqa: E402
                             lambda_block, mid_position)
from extra_models import ADDITIVE_MODEL, make_additive_hook  # noqa: E402
from inference import mean_ci  # noqa: E402
from replicate import CONTRAST_MODEL, PRIMARY_MODEL, run_replicate  # noqa: E402

ARM = "E10"
MODELS = (PRIMARY_MODEL, CONTRAST_MODEL, ADDITIVE_MODEL)
CONTRASTS = ("T2_minus_T4", "T2_minus_T3")
CONTRAST_LABELS = {
    "T2_minus_T4": "identity attributes + watchlist (T2 -> T4)",
    "T2_minus_T3": "identity attributes only, no watchlist (T2 -> T3)",
}
STOP_FAIL_RATE = 0.10


def _job(a):
    j, lam, i, tr, te, kw, b, p, additive = a
    t0 = time.time()
    rec = run_replicate(i, tr, te, cfg_factory=cfg_factory(lam, b, p, True),
                        score_hook=make_additive_hook() if additive else None,
                        **kw)
    rec["lambda"] = lam
    rec["lambda_index"] = j
    rec["wall_seconds"] = time.time() - t0
    return rec


def delta_matrix(records, grid, R, model, contrast):
    """R x G matrix of Delta, NaN where the (replicate, lambda) unit failed."""
    D = np.full((R, len(grid)), np.nan)
    for r in records:
        if r["status"] == "OK":
            D[r["replicate_id"], r["lambda_index"]] = \
                r[f"delta_miss_{contrast}_{model}"]
    return D


def params_at(lam):
    """Identity-parameter values at lambda (tuples as lists), with the path
    segment, so a crossing reads as a parameter setting, not just a number."""
    if lam is None or (isinstance(lam, float) and np.isnan(lam)):
        return None
    lam = float(lam)
    seg = ("anchor s=low" if lam == 0 else "anchor s=mid" if lam == 1
           else "anchor s=high" if lam == 2
           else "segment s=low->s=mid" if lam < 1 else "segment s=mid->s=high")
    return {"lambda": lam, "segment": seg,
            **{k: (list(v) if isinstance(v, tuple) else v)
               for k, v in lambda_block(lam).items()}}


def crossing_entry(grid, D, tau, alpha, B, boot_seed):
    est = gain_threshold_crossing(grid, D, tau, alpha)
    for name in ("mean", "LB"):
        v, flag = est[f"crossing_{name}"], est[f"flag_{name}"]
        est[f"params_at_crossing_{name}"] = (
            None if flag == "not_reached" else params_at(v))
    boot = bootstrap_gain_threshold_crossing(grid, D, tau, alpha, B=B,
                                             seed=boot_seed)
    for name in ("mean", "LB"):
        for side in ("lo", "hi"):
            lim = boot[name][side]
            lim["params"] = (params_at(lim["value"])
                             if lim["value"] is not None else None)
    est["bootstrap"] = boot
    return est


def summarize(records, grid, R, thresholds, alpha, B, boot_seed, models):
    """Crossing tables (one per contrast, both top-level) and per-lambda Delta."""
    crossings = {c: {"contrast": CONTRAST_LABELS[c], "primary": "crossing_mean",
                     "by_model": {}} for c in CONTRASTS}
    per_lambda = {}
    for model in models:
        for contrast in CONTRASTS:
            D = delta_matrix(records, grid, R, model, contrast)
            per_lambda[f"{model}|{contrast}"] = [
                dict(lam=float(lam), **mean_ci(
                    [v for v in D[:, j] if not np.isnan(v)], alpha))
                for j, lam in enumerate(grid)]
            crossings[contrast]["by_model"][model] = [
                crossing_entry(grid, D, tau, alpha, B, boot_seed)
                for tau in thresholds]
    return crossings, per_lambda


def build_spec(args, models) -> dict:
    """Everything that can change an E10 number; the lock must match it."""
    return {
        "arm": ARM, "seed_base": args.seed_base, "R": args.R,
        "grid": list(args.lambda_grid), "models": list(models),
        "n_train": args.n_train, "n_test": args.n_test, "k_star": args.k_star,
        "test_seed_offset": seed_guard.TEST_SEED_OFFSET,
        "identity_rng_stream": True,
        "flags": {"path": "piecewise low->mid->high",
                  "b": args.b, "prevalence": args.prevalence,
                  "thresholds": list(args.thresholds), "alpha": args.alpha,
                  "bootstrap_B": args.bootstrap_B,
                  "bootstrap_seed": (args.seed_base if args.bootstrap_seed
                                     is None else args.bootstrap_seed)},
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
    ap.add_argument("--lambda-grid", type=float, nargs="+",
                    default=list(DEFAULT_LAMBDA_GRID))
    ap.add_argument("--models", nargs="+", default=list(MODELS),
                    choices=list(MODELS),
                    help="gboost and logit are always fitted by run_replicate;"
                         " dropping 'additive' skips M2")
    ap.add_argument("--b", default="mid", choices=["low", "mid", "high"])
    ap.add_argument("--prevalence", type=float, default=0.05)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.0, 1.0])
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--bootstrap-B", type=int, default=2000)
    ap.add_argument("--bootstrap-seed", type=int, default=None,
                    help="default: the seed base (not a DGP seed)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--print-spec", action="store_true",
                    help="print the run spec (the lock's `spec`) and exit")
    args = ap.parse_args(argv)

    for m in (PRIMARY_MODEL, CONTRAST_MODEL):
        if m not in args.models:
            sys.exit(f"{m} is always fitted and must be listed in --models")
    models = [m for m in MODELS if m in args.models]
    spec = build_spec(args, models)
    if args.print_spec:
        print(json.dumps(spec, indent=2))
        return
    if args.out_dir is None:
        ap.error("--out-dir is required")
    try:
        mode = seed_guard.check_seeds(
            args.seed_base, args.R, args.addendum_lock, lock_spec=spec)
    except seed_guard.SeedGuardError as e:
        print(f"REFUSING TO RUN\n\n{e}", file=sys.stderr)
        sys.exit(2)
    if not 1 <= args.workers <= MAX_WORKERS:
        sys.exit(f"--workers must be in 1..{MAX_WORKERS}")
    grid = sorted(set(args.lambda_grid))
    if grid != args.lambda_grid:
        sys.exit("--lambda-grid must be strictly increasing without repeats")
    for lam in grid:
        lambda_block(lam)   # raises outside [0, 2]
    boot_seed = (args.seed_base if args.bootstrap_seed is None
                 else args.bootstrap_seed)

    os.makedirs(args.out_dir, exist_ok=True)
    summ_path = os.path.join(args.out_dir, "signal_scale_summary.json")
    raw_path = os.path.join(args.out_dir, "signal_scale_replicates.jsonl")
    if os.path.exists(summ_path):
        sys.exit(f"{summ_path} exists; refusing to overwrite a finished run")

    seeds = seed_guard.replicate_seeds(args.seed_base, args.R)
    kw = dict(n_train=args.n_train, n_test=args.n_test, k_star=args.k_star)
    additive = ADDITIVE_MODEL in models
    units = [(j, lam, i, tr, te, kw, args.b, args.prevalence, additive)
             for j, lam in enumerate(grid) for i, (tr, te) in enumerate(seeds)]
    prov = provenance()
    if mode == "ADDENDUM":
        # burn before generating anything: a crash still spends the seeds
        addendum_lock.register_start(ARM, args.seed_base, args.R,
                                     seed_guard.TEST_SEED_OFFSET,
                                     os.path.abspath(args.out_dir),
                                     prov["git_commit"])
    print(f"signal scale [{mode} seeds] R={args.R} x {len(grid)} lambda = "
          f"{len(units)} units, n_train={args.n_train} n_test={args.n_test} "
          f"k*={args.k_star}", flush=True)

    t0 = time.time()
    records = []
    halted = None
    fails_at = [0] * len(grid)
    with ProcessPoolExecutor(max_workers=args.workers) as ex, \
            open(raw_path, "w") as raw:
        futs = [ex.submit(_job, u) for u in units]
        for done, fut in enumerate(as_completed(futs), 1):
            if fut.cancelled():
                continue
            rec = fut.result()
            records.append(rec)
            raw.write(json.dumps(rec, default=float) + "\n")
            raw.flush()
            print(f"  [{done}/{len(units)}] lambda={rec['lambda']:<5g} rep "
                  f"{rec['replicate_id']:>3} {rec['status']:<10} "
                  f"{rec['wall_seconds']:.1f}s", flush=True)
            if rec["status"] != "OK":
                fails_at[rec["lambda_index"]] += 1
                j = rec["lambda_index"]
                if fails_at[j] > STOP_FAIL_RATE * args.R and halted is None:
                    halted = (f"lambda={grid[j]:g}: {fails_at[j]}/{args.R} "
                              f"units failed (> {STOP_FAIL_RATE:.0%}); run "
                              f"halted per addendum A0 stop rule")
                    print(f"\n** {halted} **\n", flush=True)
                    for f in futs:
                        f.cancel()
    wall = time.time() - t0

    records.sort(key=lambda r: (r["lambda_index"], r["replicate_id"]))
    crossings, per_lambda = summarize(records, grid, args.R, args.thresholds,
                                      args.alpha, args.bootstrap_B, boot_seed,
                                      models)
    # key order is the reading order: both crossing tables lead, side by side
    out = {
        "gain_threshold_crossing_T2_minus_T4": crossings["T2_minus_T4"],
        "gain_threshold_crossing_T2_minus_T3": crossings["T2_minus_T3"],
        "per_lambda_delta": per_lambda,
        "arm": ARM,
        "status": "EXPLORATORY",
        "halted": halted,
        "primary_estimand": "crossing_mean",
        "secondary_estimand": "crossing_LB",
        "path": "piecewise: [0,1] s=low->mid, [1,2] s=mid->high; kink at 1",
        "identity_rng_stream": True,
        "models": models,
        "R": args.R,
        "seed_mode": mode,
        "seed_base": args.seed_base,
        "test_seed_offset": seed_guard.TEST_SEED_OFFSET,
        "addendum_lock": args.addendum_lock,
        "run_spec": spec,
        "args": vars(args),
        "provenance": prov,
        "lambda_grid": grid,
        "mid_position": mid_position(),
        "n_units": len(units),
        "n_ok": sum(r["status"] == "OK" for r in records),
        "failures": [{k: r.get(k) for k in ("lambda", "replicate_id",
                                            "train_seed", "test_seed",
                                            "status", "error")}
                     for r in records if r["status"] != "OK"],
        "wall_seconds_total": wall,
        "wall_seconds_per_unit": [r["wall_seconds"] for r in records],
        "failures_per_lambda": dict(zip((f"{g:g}" for g in grid), fails_at)),
    }
    with open(summ_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    if mode == "ADDENDUM":
        addendum_lock.register_complete(args.seed_base)
    print(f"\n{out['n_ok']}/{len(units)} OK; wall {wall:.1f}s; wrote "
          f"{summ_path}", flush=True)


if __name__ == "__main__":
    main()

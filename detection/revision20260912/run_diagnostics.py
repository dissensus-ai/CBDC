"""Retrospective diagnostic runs. See SUPPLEMENTAL_PLAN.md, fixed before runs."""

import os

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import argparse
import json
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "detection"))
from detection_experiment import _entity_folds
from dgp import generate
from features import build_entity_features, build_wallet_features
from ladder_config import LADDER
from ladder_experiment import _oof_ladder, budget_metrics, run_replicate
from surface_configs import world_config

OUT = ROOT / "revision-2026-09-12/evidence"


def work(args):
    b, s, p, seed = args
    cfg = world_config(b, s, p, seed, 8000)
    rec = {
        "label": cfg.label,
        "seed": seed,
        "permutation_seed": seed + 999,
        "config": cfg.to_dict(),
        "status": "OK",
        "models": {},
    }
    try:
        data = generate(cfg)
        wf = build_wallet_features(data)
        ef = build_entity_features(data, wf)
        entities = data["entities"].copy()
        entities["is_launderer"] = np.random.default_rng(seed + 999).permutation(
            entities.is_launderer.to_numpy()
        )
        ef = ef.drop(columns="is_launderer").merge(
            entities[["entity_id", "is_launderer"]],
            on="entity_id",
            validate="one_to_one",
        )
        folds = _entity_folds(entities, seed)
        for m in LADDER:
            y, ss, params = _oof_ladder(entities, wf, ef, folds, m, seed)
            rec["models"][m] = {
                "chosen_params": params,
                "tiers": {
                    t: {
                        "auc": float(roc_auc_score(y, v)),
                        "ap": float(average_precision_score(y, v)),
                        **budget_metrics(y, v),
                    }
                    for t, v in ss.items()
                },
                "n_launderers": int(y.sum()),
            }
    except Exception:
        rec["status"] = "FAILED"
        rec["error"] = traceback.format_exc()
    return rec


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    units = []
    for b in ("low", "mid", "high"):
        for s in ("low", "mid", "high"):
            for r in range(2):
                units.append((b, s, 0.05, 2026091201 + len(units)))
    for p in (0.01, 0.03):
        for r in range(2):
            units.append(("mid", "mid", p, 2026091201 + len(units)))
    output = OUT / "permutation-controls.jsonl"
    done = set()
    if output.exists():
        if not args.resume:
            raise SystemExit(
                "Refusing overwrite; use --resume or preserve the file elsewhere."
            )
        planned = {u[3]: u for u in units}
        for line in output.read_text().splitlines():
            r = json.loads(line)
            if r["seed"] in done:
                raise SystemExit("Duplicate completed seed; investigate.")
            if r["seed"] not in planned:
                raise SystemExit("Unexpected seed; investigate.")
            b, s, p, seed = planned[r["seed"]]
            expected = json.loads(
                json.dumps(world_config(b, s, p, seed, 8000).to_dict())
            )
            if r["config"] != expected:
                raise SystemExit("Checkpoint configuration mismatch; investigate.")
            if r["status"] not in ("OK", "FAILED"):
                raise SystemExit("Invalid checkpoint status.")
            done.add(seed)
    print("Resume:", len(done), "completed units; workers:", args.workers, flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        fs = {ex.submit(work, u): u for u in units if u[3] not in done}
        for i, f in enumerate(as_completed(fs), len(done) + 1):
            r = f.result()
            with output.open("a") as o:
                o.write(json.dumps(r) + "\n")
            print(i, len(units), r["label"], r["seed"], r["status"], flush=True)
    cfg = world_config("mid", "mid", 0.05, 2026082051, 8000)
    # Compare at the JSON artifact boundary: dataclass tuples are lists on disk.
    replay = json.loads(json.dumps(run_replicate(cfg, tuple(LADDER), cfg.seed)))
    original = next(
        json.loads(x)
        for x in (ROOT / "detection/results/ladder/frozen/default_ladder.jsonl")
        .read_text()
        .splitlines()
        if json.loads(x)["seed"] == cfg.seed
    )
    fields = ("label", "seed", "config", "gate", "models")
    exact = {k: replay[k] == original[k] for k in fields}
    (OUT / "ladder-replay.json").write_text(
        json.dumps({"fields_equal": exact, "reproduced_record": replay}, indent=2)
    )
    assert all(exact.values()), exact
    print("Ladder replay exact match", exact, flush=True)

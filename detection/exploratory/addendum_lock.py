"""Minimal addendum lock for E9-E11, plus the spent-seed registry.

A reported (non-DEV) exploratory run needs `--addendum-lock <path>`, a JSON
file written after MF approves the addendum:

    {"spec": {<the run spec, below>},
     "protocol_file": "detection/protocols/E9_E11_PROTOCOL_<date>.md",
     "protocol_sha256": "<sha256 of that file>",
     "freeze_commit": "<commit that added the protocol file>",
     "written": "YYYY-MM-DD", "lock_sha256": "<hash of everything else>"}

The run spec is built by the driver (`--print-spec` prints it) and holds
EVERYTHING that could change a number: arm, seed_base, R, grid, models,
n_train, n_test, k_star, test_seed_offset, identity_rng_stream, the arm's
variant/attribute flags (E9: stage-2 tiers, training variants, stage-1
training scores, alpha, bootstrap; E10: b, prevalence, thresholds, alpha,
bootstrap), and the environment (Python + numpy/scipy/sklearn/pandas versions
+ sha256 of requirements.txt). `validate` requires the run's spec to equal the
lock's EXACTLY, key by key -- no field is left for the command line to drift.
It also requires the self-hash to hold, the protocol file on disk to still hash
to protocol_sha256, the freeze commit to be an ancestor of HEAD, R <= 10,000,
and neither the spent-seed registry nor any result file in the repository
(seed_scan, structured pass) to record a seed in the run's train or test block.

Like protocol_lock, this makes "pre-specified" checkable; it is not a
registration. No lock for the reported bases exists yet -- writing one is a
post-approval step.

Spent-seed registry (SPENT_SEEDS.json beside this file). A locked run records
its train and test blocks as "in_progress" BEFORE generating anything and
flips them to "completed" when it finishes, so a crashed run still burns its
seeds. Later runs refuse any overlap with either status. DEV-block runs are not
registered: the whole DEV block is burned unconditionally.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import date, datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
REGISTRY = os.path.join(HERE, "SPENT_SEEDS.json")
MAX_SEEDS_PER_ARM = 10_000
LOCK_KEYS = ("spec", "protocol_file", "protocol_sha256", "freeze_commit")
SPEC_KEYS = ("arm", "seed_base", "R", "grid", "models", "n_train", "n_test",
             "k_star", "test_seed_offset", "identity_rng_stream", "flags",
             "environment")
ARMS = ("E9", "E10", "E11")


class AddendumLockError(RuntimeError):
    """Raised when a lock does not authorise the requested run."""


def repo_root() -> str:
    try:
        return subprocess.run(["git", "-C", HERE, "rev-parse", "--show-toplevel"],
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return os.path.dirname(os.path.dirname(HERE))


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_hash(lock: dict) -> str:
    body = {k: v for k, v in lock.items() if k != "lock_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def normalize_spec(spec: dict) -> dict:
    """Canonical form for exact comparison: JSON round trip, grid as floats,
    models sorted (the fitted set, not an order)."""
    missing = [k for k in SPEC_KEYS if k not in spec]
    if missing:
        raise AddendumLockError(f"run spec is missing {missing}")
    extra = [k for k in spec if k not in SPEC_KEYS]
    if extra:
        raise AddendumLockError(f"run spec has unknown keys {extra}")
    d = json.loads(json.dumps(spec, sort_keys=True))
    d["grid"] = [float(x) for x in d["grid"]]
    d["models"] = sorted(d["models"])
    return d


def spec_diff(lock_spec: dict, run_spec: dict) -> dict:
    """{dotted key: (lock value, run value)} for every mismatch, recursing into
    the flags and environment dicts."""
    a, b = normalize_spec(lock_spec), normalize_spec(run_spec)
    diff = {}
    for k in SPEC_KEYS:
        if isinstance(a[k], dict) and isinstance(b[k], dict):
            for kk in sorted(set(a[k]) | set(b[k])):
                if a[k].get(kk, "<absent>") != b[k].get(kk, "<absent>"):
                    diff[f"{k}.{kk}"] = (a[k].get(kk, "<absent>"),
                                         b[k].get(kk, "<absent>"))
        elif a[k] != b[k]:
            diff[k] = (a[k], b[k])
    return diff


def blocks(seed_base: int, R: int, offset: int) -> list[tuple[int, int]]:
    """Train and test blocks a run consumes, inclusive."""
    return [(seed_base, seed_base + R - 1),
            (seed_base + offset, seed_base + offset + R - 1)]


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

def load_registry(path: str = REGISTRY) -> dict:
    if not os.path.exists(path):
        return {"entries": []}
    with open(path) as f:
        return json.load(f)


def registry_overlap(lo: int, hi: int, path: str = REGISTRY) -> list[dict]:
    return [e for e in load_registry(path)["entries"]
            if e["lo"] <= hi and e["hi"] >= lo]


def _save_registry(reg: dict, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(reg, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def register_start(arm, seed_base, R, offset, out_dir, commit,
                   path: str = REGISTRY) -> None:
    """Burn the run's blocks before any world is generated."""
    reg = load_registry(path)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for (lo, hi), what in zip(blocks(seed_base, R, offset), ("train", "test")):
        if registry_overlap(lo, hi, path):
            raise AddendumLockError(f"{what} block {lo}..{hi} already in the "
                                    f"spent-seed registry")
        reg["entries"].append({"lo": lo, "hi": hi, "block": what, "arm": arm,
                               "seed_base": seed_base, "R": R,
                               "status": "in_progress", "started": now,
                               "finished": None, "commit": commit,
                               "out_dir": out_dir})
    _save_registry(reg, path)


def register_complete(seed_base, path: str = REGISTRY) -> None:
    reg = load_registry(path)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for e in reg["entries"]:
        if e["seed_base"] == seed_base and e["status"] == "in_progress":
            e["status"], e["finished"] = "completed", now
    _save_registry(reg, path)


# ---------------------------------------------------------------------------
# lock
# ---------------------------------------------------------------------------

def _git(*a, root):
    return subprocess.run(["git", "-C", root, *a], capture_output=True,
                          text=True)


def check_block_unused(seed_base, R, offset, *, root=None,
                       registry=REGISTRY) -> None:
    """Refuse if the registry or any repository result file records a seed in
    the run's train or test block."""
    from seed_scan import structured_hits
    root = root or repo_root()
    for lo, hi in blocks(seed_base, R, offset):
        reg = registry_overlap(lo, hi, registry)
        if reg:
            raise AddendumLockError(
                f"block {lo}..{hi} overlaps the spent-seed registry: "
                f"{[(e['arm'], e['lo'], e['hi'], e['status']) for e in reg]}")
        hits = structured_hits(root, lo, hi)
        if hits:
            raise AddendumLockError(
                f"block {lo}..{hi} is already recorded in result files: "
                + "; ".join(f"{h['file']} {h['lo']}..{h['hi']}"
                            for h in hits[:5]))


def validate(path, spec, *, root=None, registry=REGISTRY) -> dict:
    """Load the lock at `path`; raise unless it authorises exactly `spec`."""
    root = root or repo_root()
    if not os.path.isfile(path):
        raise AddendumLockError(f"addendum lock {path} does not exist")
    try:
        with open(path) as f:
            lock = json.load(f)
    except ValueError as e:
        raise AddendumLockError(f"addendum lock {path} is not JSON: {e}")
    missing = [k for k in LOCK_KEYS + ("lock_sha256",) if k not in lock]
    if missing:
        raise AddendumLockError(f"lock is missing {missing}")
    if lock["lock_sha256"] != compute_hash(lock):
        raise AddendumLockError("lock hash mismatch: edited after it was written")
    if lock["spec"].get("arm") not in ARMS:
        raise AddendumLockError(f"unknown arm {lock['spec'].get('arm')!r}")
    diff = spec_diff(lock["spec"], spec)
    if diff:
        raise AddendumLockError(f"run does not match lock (lock, run): {diff}")
    R = spec["R"]
    if not 1 <= R <= MAX_SEEDS_PER_ARM:
        raise AddendumLockError(f"R={R} outside 1..{MAX_SEEDS_PER_ARM}")

    proto = os.path.join(root, lock["protocol_file"])
    if not os.path.isfile(proto):
        raise AddendumLockError(f"protocol file {proto} not found")
    if file_sha256(proto) != lock["protocol_sha256"]:
        raise AddendumLockError("protocol file changed since the lock was "
                                "written (sha256 mismatch)")
    fc = lock["freeze_commit"]
    if _git("cat-file", "-e", f"{fc}^{{commit}}", root=root).returncode:
        raise AddendumLockError(f"freeze commit {fc} not in this repository")
    if _git("merge-base", "--is-ancestor", fc, "HEAD", root=root).returncode:
        raise AddendumLockError(f"freeze commit {fc} is not an ancestor of "
                                f"HEAD: running from code that predates or "
                                f"forks away from the freeze")
    check_block_unused(spec["seed_base"], R, spec["test_seed_offset"],
                       root=root, registry=registry)
    return lock


def write_lock(path, spec, *, protocol_file, freeze_commit, root=None,
               registry=REGISTRY) -> dict:
    """Write a lock for `spec` after checking its seed blocks are unused.
    Post-approval only. `spec` should come from the driver's --print-spec."""
    root = root or repo_root()
    spec = normalize_spec(spec)
    check_block_unused(spec["seed_base"], spec["R"], spec["test_seed_offset"],
                       root=root, registry=registry)
    lock = {"spec": spec,
            "protocol_file": protocol_file,
            "protocol_sha256": file_sha256(os.path.join(root, protocol_file)),
            "freeze_commit": freeze_commit,
            "written": date.today().isoformat()}
    lock["lock_sha256"] = compute_hash(lock)
    with open(path, "w") as f:
        json.dump(lock, f, indent=2)
    return lock

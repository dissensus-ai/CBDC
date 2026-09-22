"""Seed guard, addendum lock, spent-seed registry and recorded-seed scan tests.

Lock tests build a throwaway git repository (protocol file + freeze commit) in
a temporary directory, so they exercise the real validator end to end without
touching this repository or its registry.
"""

import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import addendum_lock as al  # noqa: E402
from seed_guard import (DEV_SEED_MAX, SeedGuardError,  # noqa: E402
                        TEST_SEED_OFFSET, check_seeds, replicate_seeds)
from seed_scan import raw_hits, structured_hits  # noqa: E402

BASE = 2_026_120_001
SPEC = {"arm": "E10", "grid": [0.0, 1.0, 2.0],
        "models": ["gboost", "logit", "additive"]}


def _refuses(*a, **kw):
    try:
        check_seeds(*a, **kw)
    except SeedGuardError:
        return True
    return False


class _Repo:
    """Temporary git repo holding a protocol file, a freeze commit, a lock."""

    def __init__(self, base=BASE, R=5, spec=SPEC):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.registry = os.path.join(self.root, "SPENT_SEEDS.json")
        git = ["git", "-C", self.root, "-c", "user.name=t", "-c",
               "user.email=t@t", "-c", "commit.gpgsign=false"]
        subprocess.run(["git", "init", "-q", self.root], check=True)
        with open(os.path.join(self.root, "PROTO.md"), "w") as f:
            f.write("protocol text\n")
        subprocess.run(git + ["add", "PROTO.md"], check=True)
        subprocess.run(git + ["commit", "-qm", "freeze"], check=True)
        self.commit = subprocess.run(
            ["git", "-C", self.root, "rev-parse", "HEAD"], capture_output=True,
            text=True, check=True).stdout.strip()
        self.lock = os.path.join(self.root, "e10_lock.json")
        al.write_lock(self.lock, seed_base=base, R=R,
                      protocol_file="PROTO.md", freeze_commit=self.commit,
                      offset=TEST_SEED_OFFSET, root=self.root,
                      registry=self.registry, **spec)

    def check(self, base=BASE, R=5, spec=SPEC):
        return check_seeds(base, R, self.lock, lock_spec=spec, root=self.root,
                           registry=self.registry)

    def refuses(self, base=BASE, R=5, spec=SPEC):
        try:
            self.check(base, R, spec)
        except SeedGuardError:
            return True
        return False

    def close(self):
        self.tmp.cleanup()


def test_dev_block_accepted():
    assert check_seeds(700_001, 52) == "DEV"
    assert check_seeds(700_948, 52) == "DEV"   # last seed == 700999


def test_block_must_stay_inside_dev():
    assert _refuses(700_000, 3)
    assert _refuses(700_950, 52)               # walks past 700999
    assert _refuses(DEV_SEED_MAX + 1, 1)


def test_non_dev_needs_lock():
    assert _refuses(123_456, 5)


def test_forbidden_seeds_refused_even_with_lock():
    for base in (900_001, 900_015, 20_260_707, 2_026_080_501, 2_026_080_530,
                 2_026_081_951, 2_026_082_051, 2_026_091_201):
        assert _refuses(base, 3, addendum_lock="/nonexistent", lock_spec=SPEC)


def test_block_that_runs_into_forbidden_refused():
    assert _refuses(899_990, 20, addendum_lock="/nonexistent", lock_spec=SPEC)


def test_valid_lock_accepted():
    r = _Repo()
    try:
        assert r.check() == "ADDENDUM"
    finally:
        r.close()


def test_lock_must_match_run_exactly():
    r = _Repo()
    try:
        assert r.refuses(R=6)
        assert r.refuses(base=BASE + 1)
        assert r.refuses(spec={**SPEC, "arm": "E9"})
        assert r.refuses(spec={**SPEC, "grid": [0.0, 1.0]})
        assert r.refuses(spec={**SPEC, "models": ["gboost", "logit"]})
        assert r.check(spec={**SPEC, "models": ["additive", "logit",
                                                "gboost"]}) == "ADDENDUM"
    finally:
        r.close()


def test_lock_missing_or_tampered_refused():
    r = _Repo()
    try:
        assert _refuses(BASE, 5, addendum_lock=os.path.join(r.root, "none"),
                        lock_spec=SPEC, root=r.root, registry=r.registry)
        with open(r.lock) as f:
            lock = json.load(f)
        lock["R"] = 6                              # edit without re-hashing
        with open(r.lock, "w") as f:
            json.dump(lock, f)
        assert r.refuses(R=6)
    finally:
        r.close()


def test_protocol_edit_after_lock_refused():
    r = _Repo()
    try:
        with open(os.path.join(r.root, "PROTO.md"), "a") as f:
            f.write("quietly amended\n")
        assert r.refuses()
    finally:
        r.close()


def test_unknown_freeze_commit_refused():
    r = _Repo()
    try:
        with open(r.lock) as f:
            lock = json.load(f)
        lock["freeze_commit"] = "0" * 40
        lock["lock_sha256"] = al.compute_hash(lock)
        with open(r.lock, "w") as f:
            json.dump(lock, f)
        assert r.refuses()
    finally:
        r.close()


def test_lock_on_dev_seeds_refused():
    assert _refuses(700_001, 3, addendum_lock="/nonexistent", lock_spec=SPEC)


def test_registry_burns_and_blocks_reuse():
    r = _Repo()
    try:
        al.register_start("E10", BASE, 5, TEST_SEED_OFFSET, "/out", "abc",
                          path=r.registry)
        reg = al.load_registry(r.registry)["entries"]
        assert [(e["lo"], e["hi"], e["status"]) for e in reg] == [
            (BASE, BASE + 4, "in_progress"),
            (BASE + TEST_SEED_OFFSET, BASE + TEST_SEED_OFFSET + 4,
             "in_progress")]
        assert r.refuses()                          # in-progress already burns
        al.register_complete(BASE, path=r.registry)
        assert {e["status"] for e in
                al.load_registry(r.registry)["entries"]} == {"completed"}
        assert r.refuses()
        try:
            al.register_start("E10", BASE + 2, 5, TEST_SEED_OFFSET, "/o", "x",
                              path=r.registry)
        except al.AddendumLockError:
            pass
        else:
            raise AssertionError("overlapping registration should raise")
    finally:
        r.close()


def test_recorded_seed_in_results_blocks_lock():
    r = _Repo()
    try:
        os.makedirs(os.path.join(r.root, "results"))
        with open(os.path.join(r.root, "results", "old.jsonl"), "w") as f:
            f.write(json.dumps({"train_seed": BASE + 3, "x": 1}) + "\n")
        assert r.refuses()
        try:
            al.write_lock(os.path.join(r.root, "again_lock.json"),
                          seed_base=BASE, R=5, protocol_file="PROTO.md",
                          freeze_commit=r.commit, offset=TEST_SEED_OFFSET,
                          root=r.root, registry=r.registry, **SPEC)
        except al.AddendumLockError:
            pass
        else:
            raise AssertionError("write_lock should refuse a used block")
    finally:
        r.close()


def test_seed_scan_structured_and_raw():
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "summary.json"), "w") as f:
            json.dump({"conf_seed_base": 5_000_000, "R": 10,
                       "seed_offset": 100, "nested": {"pilot_seeds": [7_000_001]},
                       "unrelated": 5_000_050}, f)
        with open(os.path.join(d, "fails.csv"), "w") as f:
            f.write("replicate_id,train_seed,status\n0,6000000,OK\n")
        with open(os.path.join(d, "x_lock.json"), "w") as f:
            json.dump({"seed_base": 8_000_000, "R": 5}, f)   # declaration
        with open(os.path.join(d, "notes.md"), "w") as f:
            f.write("planned base 8000000 and 5000009\n")
        assert structured_hits(d, 5_000_009, 5_000_009)          # block end
        assert structured_hits(d, 5_000_100, 5_000_100)          # test image
        assert not structured_hits(d, 5_000_010, 5_000_050)      # past block
        assert structured_hits(d, 7_000_001, 7_000_001)          # list
        assert structured_hits(d, 6_000_000, 6_000_000)          # csv
        assert not structured_hits(d, 8_000_000, 8_000_004)      # lock skipped
        assert {h["value"] for h in raw_hits(d, 8_000_000, 8_000_000)} == \
            {8_000_000}


def test_sklearn_ceiling_enforced():
    base = 2 ** 32 - TEST_SEED_OFFSET - 2
    assert _refuses(base, 3, addendum_lock="/nonexistent", lock_spec=SPEC)
    r = _Repo(base=base, R=1)
    try:
        assert r.check(base=base, R=1) == "ADDENDUM"
    finally:
        r.close()


def test_bad_types_refused():
    assert _refuses(700_001.0, 3)
    assert _refuses(True, 3)
    assert _refuses(700_001, 0)


def test_replicate_seed_layout():
    assert replicate_seeds(700_001, 2) == [
        (700_001, 700_001 + TEST_SEED_OFFSET),
        (700_002, 700_002 + TEST_SEED_OFFSET)]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS: {fn.__name__}")
    print(f"\nall {len(fns)} seed-guard tests passed")

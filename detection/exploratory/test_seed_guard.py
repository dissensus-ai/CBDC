"""Seed guard tests: DEV block, forbidden seeds, addendum lock, sklearn range."""

import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from seed_guard import (DEV_SEED_MAX, SeedGuardError,  # noqa: E402
                        TEST_SEED_OFFSET, check_seeds, replicate_seeds)


def _refuses(*a, **kw):
    try:
        check_seeds(*a, **kw)
    except SeedGuardError:
        return True
    return False


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
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as d:
        for base in (900_001, 900_015, 20_260_707, 2_026_080_501,
                     2_026_080_530, 2_026_081_951, 2_026_082_051,
                     2_026_091_201):
            p = os.path.join(d, f"lock{base}.txt")
            with open(p, "w") as f:
                f.write(f"seed_base: {base}\n")
            assert _refuses(base, 3, addendum_lock=p), base


def test_block_that_runs_into_forbidden_refused():
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as d:
        p = os.path.join(d, "lock.txt")
        with open(p, "w") as f:
            f.write("seed_base 899990\n")
        assert _refuses(899_990, 20, addendum_lock=p)   # reaches 900001


def test_lock_must_exist_and_name_base():
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as d:
        assert _refuses(800_001, 5, addendum_lock=os.path.join(d, "nope"))
        p = os.path.join(d, "lock.txt")
        with open(p, "w") as f:
            f.write("seed_base: 8000011\n")   # contains the digits, not the token
        assert _refuses(800_001, 5, addendum_lock=p)
        with open(p, "w") as f:
            f.write('{"seed_base": 800001}\n')
        assert check_seeds(800_001, 5, addendum_lock=p) == "ADDENDUM"


def test_sklearn_ceiling_enforced():
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as d:
        base = 2 ** 32 - TEST_SEED_OFFSET - 2
        p = os.path.join(d, "lock.txt")
        with open(p, "w") as f:
            f.write(f"{base}\n")
        assert check_seeds(base, 1, addendum_lock=p) == "ADDENDUM"
        assert _refuses(base, 3, addendum_lock=p)   # test seed crosses 2^32-1


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

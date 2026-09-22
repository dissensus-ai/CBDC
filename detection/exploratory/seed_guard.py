"""Seed guard for the exploratory arms (two-stage screening, signal scale).

These drivers have no protocol lock of their own yet. Until an addendum fixes
the reported seeds, they may run ONLY on the development block 700001-700999,
and every seed they would touch must be disjoint from every seed already spent
elsewhere in this repository. A seed base outside the DEV block is accepted
only with `--addendum-lock <path>`: the file must exist and must contain the
seed base as a literal token. That is a tripwire, not a freeze -- the real
lock format belongs to the addendum, which does not exist yet.

Why refuse rather than warn: a dev run on a confirmatory seed burns that seed
(protocol 6.1 step 3), and nothing downstream can tell a burned seed from a
fresh one. The check has to happen before any world is generated.
"""

from __future__ import annotations

import os
import re

DEV_SEED_MIN = 700_001
DEV_SEED_MAX = 700_999

# Train/test offset, the same constant protocol v3 uses (protocol_lock
# `seed_offset`). Test seed = train seed + TEST_SEED_OFFSET.
TEST_SEED_OFFSET = 10_000_003

SKLEARN_SEED_MAX = 2 ** 32 - 1

# Seeds already spent. "base.." entries in the repo docs have no stated end, so
# each is reserved for 1000 seeds, and the confirmatory/pilot blocks are also
# reserved at their +offset test-seed images. Inclusive (lo, hi, why).
_SPAN = 1000
FORBIDDEN_RANGES = (
    (20_260_707, 20_260_707, "pilot/dev seed 20260707 (run_all.py)"),
    (900_001, 900_020, "burned pilot seeds (protocol_lock.pilot_seeds_burned)"),
    (900_001 + TEST_SEED_OFFSET, 900_020 + TEST_SEED_OFFSET,
     "burned pilot test seeds"),
    (2_026_080_501, 2_026_080_501 + _SPAN - 1, "confirmatory conf_seed_base"),
    (2_026_080_501 + TEST_SEED_OFFSET,
     2_026_080_501 + TEST_SEED_OFFSET + _SPAN - 1, "confirmatory test seeds"),
    (2_026_081_901, 2_026_081_901 + _SPAN - 1, "ladder pilot base (burned)"),
    (2_026_081_951, 2_026_081_951 + _SPAN - 1, "ladder frozen base"),
    (2_026_082_051, 2_026_082_051 + _SPAN - 1, "ladder frozen replay seed"),
    (2_026_091_201, 2_026_091_201 + _SPAN - 1,
     "September permutation diagnostics"),
)


class SeedGuardError(RuntimeError):
    """Raised when a run would touch a seed it is not entitled to."""


def replicate_seeds(seed_base: int, R: int) -> list[tuple[int, int]]:
    """(train_seed, test_seed) for replicates 0..R-1."""
    return [(seed_base + i, seed_base + i + TEST_SEED_OFFSET)
            for i in range(R)]


def _lock_names_base(path: str, seed_base: int) -> bool:
    with open(path) as f:
        text = f.read()
    return re.search(rf"(?<!\d){int(seed_base)}(?!\d)", text) is not None


def check_seeds(seed_base: int, R: int, addendum_lock: str | None = None) -> str:
    """Raise SeedGuardError unless this run may use these seeds.

    Returns "DEV" or "ADDENDUM" for the output record. The whole train block
    seed_base..seed_base+R-1 must sit inside the DEV range, not just its first
    seed, or a large R would walk out of the block unnoticed.
    """
    if not isinstance(seed_base, int) or isinstance(seed_base, bool):
        raise SeedGuardError(f"seed base must be an int, got {seed_base!r}")
    if not isinstance(R, int) or R < 1:
        raise SeedGuardError(f"R must be a positive int, got {R!r}")

    last = seed_base + R - 1
    in_dev = DEV_SEED_MIN <= seed_base and last <= DEV_SEED_MAX
    if in_dev:
        mode = "DEV"
    elif addendum_lock is None:
        raise SeedGuardError(
            f"seed block {seed_base}..{last} is outside the DEV range "
            f"{DEV_SEED_MIN}-{DEV_SEED_MAX}. Non-dev seeds need "
            f"--addendum-lock <path> naming this seed base.")
    else:
        if not os.path.isfile(addendum_lock):
            raise SeedGuardError(f"addendum lock {addendum_lock} does not exist")
        if not _lock_names_base(addendum_lock, seed_base):
            raise SeedGuardError(
                f"addendum lock {addendum_lock} does not contain seed base "
                f"{seed_base}")
        mode = "ADDENDUM"

    for tr, te in replicate_seeds(seed_base, R):
        for s in (tr, te):
            if s < 0 or s > SKLEARN_SEED_MAX:
                raise SeedGuardError(
                    f"seed {s} is outside sklearn's random_state range "
                    f"[0, {SKLEARN_SEED_MAX}]; exploratory arms pass seeds "
                    f"through unnarrowed, so this would not be reproducible "
                    f"from the protocol seed")
            for lo, hi, why in FORBIDDEN_RANGES:
                if lo <= s <= hi:
                    raise SeedGuardError(
                        f"seed {s} collides with already-used seeds "
                        f"{lo}..{hi} ({why})")
    return mode

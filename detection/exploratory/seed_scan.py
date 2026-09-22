"""Find every seed already recorded in the repository's result files.

The addendum (section 0) requires that reported seed ranges be checked against
every seed recorded in existing result manifests, not just the bases someone
remembered. Two passes:

structured (the GATE)  JSON / JSONL / CSV files. Any key or column whose name
                       contains "seed" contributes its integer value(s);
                       a "seed_base"/"conf_seed_base" with an "R" beside it
                       also contributes the whole block base..base+R-1 and,
                       when a "seed_offset"/"test_seed_offset" is present,
                       its test-seed image. Lock files and the spent-seed
                       registry are declarations, not results, and are skipped.

raw (informational)    every 6-10 digit integer in any text file that falls
                       in the queried ranges, with file:line. Protocol text
                       naming a planned base shows up here by design; a hit
                       is a pointer to read, not a collision.

    python3 seed_scan.py [--root DIR] [--range LO HI ...]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re

TEXT_EXT = {".json", ".jsonl", ".csv", ".md", ".txt", ".log", ".py", ".tex",
            ".yaml", ".yml", ".toml"}
STRUCT_EXT = {".json", ".jsonl", ".csv"}
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".ruff_cache",
             "node_modules", ".pytest_cache"}
MAX_BYTES = 200 * 1024 * 1024
_INT_RE = re.compile(r"(?<![\d.])\d{6,10}(?![\d.])")


def is_declaration(path: str) -> bool:
    """Files that declare seeds rather than record their use."""
    base = os.path.basename(path).lower()
    return base == "spent_seeds.json" or "lock" in base


def _iter_files(root, exts):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            p = os.path.join(dirpath, fn)
            if os.path.splitext(fn)[1].lower() in exts:
                try:
                    if os.path.getsize(p) <= MAX_BYTES:
                        yield p
                except OSError:
                    continue


def _as_int(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, str) and v.strip().lstrip("-").isdigit():
        return int(v.strip())
    return None


def _walk_json(obj, out):
    """Collect (lo, hi) inclusive ranges of recorded seeds from a JSON tree."""
    if isinstance(obj, dict):
        base = None
        for key in ("seed_base", "conf_seed_base"):
            if _as_int(obj.get(key)) is not None:
                base = _as_int(obj[key])
        R = _as_int(obj.get("R"))
        if base is not None and R:
            out.append((base, base + R - 1))
            off = _as_int(obj.get("seed_offset", obj.get("test_seed_offset")))
            if off:
                out.append((base + off, base + off + R - 1))
        for k, v in obj.items():
            if "seed" in str(k).lower():
                vals = v if isinstance(v, list) else [v]
                for x in vals:
                    xi = _as_int(x)
                    if xi is not None:
                        out.append((xi, xi))
            if isinstance(v, (dict, list)):
                _walk_json(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _walk_json(v, out)


def structured_ranges(root):
    """{path: [(lo, hi), ...]} for every result file that records a seed."""
    found = {}
    for p in _iter_files(root, STRUCT_EXT):
        if is_declaration(p):
            continue
        rngs = []
        ext = os.path.splitext(p)[1].lower()
        try:
            if ext == ".json":
                with open(p) as f:
                    _walk_json(json.load(f), rngs)
            elif ext == ".jsonl":
                with open(p) as f:
                    for line in f:
                        if line.strip():
                            _walk_json(json.loads(line), rngs)
            else:
                with open(p, newline="") as f:
                    for row in csv.DictReader(f):
                        for k, v in row.items():
                            if k and "seed" in k.lower():
                                xi = _as_int(v)
                                if xi is not None:
                                    rngs.append((xi, xi))
        except (ValueError, UnicodeDecodeError, OSError):
            continue
        if rngs:
            found[p] = sorted(set(rngs))
    return found


def structured_hits(root, lo, hi):
    """Recorded seeds/blocks in result files that intersect [lo, hi]."""
    hits = []
    for p, rngs in structured_ranges(root).items():
        for a, b in rngs:
            if a <= hi and b >= lo:
                hits.append({"file": p, "lo": a, "hi": b})
    return hits


def raw_hits(root, lo, hi):
    hits = []
    for p in _iter_files(root, TEXT_EXT):
        try:
            with open(p, errors="ignore") as f:
                for i, line in enumerate(f, 1):
                    for m in _INT_RE.finditer(line):
                        v = int(m.group())
                        if lo <= v <= hi:
                            hits.append({"file": p, "line": i, "value": v})
        except OSError:
            continue
    return hits


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    ap.add_argument("--range", type=int, nargs=2, action="append",
                    metavar=("LO", "HI"))
    args = ap.parse_args()
    ranges = args.range or [(2_026_110_001, 2_026_139_999)]
    n_files = len(structured_ranges(args.root))
    print(f"root {args.root}: {n_files} result files record seeds")
    for lo, hi in ranges:
        s, r = structured_hits(args.root, lo, hi), raw_hits(args.root, lo, hi)
        print(f"\n[{lo}, {hi}]  structured (gate): "
              f"{'CLEAR' if not s else f'{len(s)} HIT(S)'}   raw text: "
              f"{len(r)} mention(s)")
        for h in s:
            print(f"  GATE {h['file']}: {h['lo']}..{h['hi']}")
        for h in r:
            print(f"  raw  {h['file']}:{h['line']}: {h['value']}")


if __name__ == "__main__":
    main()

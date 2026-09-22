# DEV SEEDS, NOT REPORTABLE

Smoke output of the exploratory noisy-linkage arm, E11 (`../run_noisy_linkage.py`).
It exists to show that the driver runs end to end and to time it. None of these
numbers is a finding, and none may be quoted, plotted, or used to choose a grid,
a rule, or a seed for a reported run.

- Seeds: DEV range only (train 700501–700503 for the main smoke, 700504–700505
  for the variant). Test seed = train seed + 10,000,003.
- Size: n_train = n_test = 2,000 entities (the reported design is 8,000 /
  10,000), k* = 500 per 10,000, so k = 100 cluster alerts. R = 3 (main), R = 2
  (variant). Resolution at this size is too coarse to support any comparison.
- Main smoke: the proposed reported configuration at toy size: `axes+v2` grid
  (13 per-wallet cells + 4 bipartition v2 §6 cells, one shared oracle), uniform
  merges, attribute bracket `attrwise_max_risk` + `majority`.
- `variant_bipartition_counterparty_riskiest_first/`: bipartition splits on the
  axes grid, counterparty-weighted merges, attribute rules `riskiest_member` +
  `first_wallet`. Shows the non-default options execute; nothing more.

Regenerated 22 Sep 2026 after the pre-freeze fixes (schema change: `attr` level
in records, `summary_primary.csv`, `run_spec.json`). The earlier smoke output is
in git history only.

Command (from `detection/exploratory/`, pinned environment):

```bash
python3 run_noisy_linkage.py --seed-base 700501 --R 3 --n-train 2000 \
    --n-test 2000 --workers 3 --grid axes+v2 --out-dir _dev_smoke_linkage
python3 run_noisy_linkage.py --seed-base 700504 --R 2 --n-train 2000 \
    --n-test 2000 --workers 2 --merge-mode counterparty \
    --attr-rules riskiest_member first_wallet --split-mode bipartition \
    --out-dir _dev_smoke_linkage/variant_bipartition_counterparty_riskiest_first
```

Files: `run_spec.json` (every run-defining field), `records.jsonl` (one raw
record per replicate), `summary_primary.csv` (G next to the T2/T3/T4
deterioration curves), `summary.csv` (long form), `timing.json`.

# DEV SEEDS, NOT REPORTABLE

Smoke output of the exploratory noisy-linkage arm (`../run_noisy_linkage.py`).
It exists to show that the driver runs end to end and to time it. None of these
numbers is a finding, and none may be quoted, plotted, or used to choose a grid,
a rule, or a seed for a reported run.

- Seeds: DEV range only (train 700501–700503 for the main smoke, 700504–700505
  for the variant). Test seed = train seed + 10,000,003.
- Size: n_train = n_test = 2,000 entities (the reported design is 8,000 /
  10,000), k* = 500 per 10,000, so k = 100 cluster alerts. R = 3 (main), R = 2
  (variant). Resolution at this size is too coarse to support any comparison.
- Main smoke: wallet-level splits, uniform merges, max-risk attribute pooling,
  13-cell axes+diagonal grid over eps in {0, 0.05, 0.10, 0.20, 0.30}.
- `variant_bipartition_counterparty_majority/`: bipartition splits,
  counterparty-weighted merges, majority attribute rule. Shows the non-default
  options execute; nothing more.

Command (from `detection/exploratory/`, pinned environment):

```bash
python3 run_noisy_linkage.py --seed-base 700501 --R 3 --n-train 2000 \
    --n-test 2000 --workers 3 --out-dir _dev_smoke_linkage
python3 run_noisy_linkage.py --seed-base 700504 --R 2 --n-train 2000 \
    --n-test 2000 --workers 2 --merge-mode counterparty --attr-rule majority \
    --split-mode bipartition \
    --out-dir _dev_smoke_linkage/variant_bipartition_counterparty_majority
```

Files: `config.json` (every argument), `records.jsonl` (one raw record per
replicate), `summary.csv` (t-interval summaries), `timing.json`.

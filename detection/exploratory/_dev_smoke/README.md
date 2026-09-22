# DEV SEEDS. NOT REPORTABLE.

Everything in this directory was generated on the development seed block
700001-700999 at toy or single-unit sizes, to show that the two exploratory
drivers run end-to-end and to time them. None of it is a finding. Do not quote,
plot, or summarize these numbers anywhere. The seeds used here are spent for
reporting purposes. No reported run may use the DEV block (`seed_guard.py`).

Generated 22 Sep 2026 from commit 082bf6c (clean tree, recorded in each
summary's `provenance`), Python 3.14.7, numpy 2.3.5, scipy 1.16.3,
scikit-learn 1.8.0, pandas 2.3.3.

| directory | command (from `detection/exploratory/`) | purpose |
|---|---|---|
| `two_stage/` | `run_two_stage.py --seed-base 700001 --R 3 --n-train 2000 --n-test 2000 --train-variants full shortlist --workers 3 --out-dir _dev_smoke/two_stage` | Arm A pipeline check |
| `signal_scale/` | `run_signal_scale.py --seed-base 700101 --R 3 --n-train 2000 --n-test 2000 --workers 5 --out-dir _dev_smoke/signal_scale` | Arm B pipeline check (full 14-point lambda grid) |
| `timing_probe/two_stage/` | `run_two_stage.py --seed-base 700201 --R 2 --train-variants full shortlist --workers 2 --out-dir _dev_smoke/timing_probe/two_stage` | full-size (8000/10000) wall time only |
| `timing_probe/signal_scale/` | `run_signal_scale.py --seed-base 700301 --R 2 --lambda-grid 0.0 1.0 --bootstrap-B 200 --workers 4 --out-dir _dev_smoke/timing_probe/signal_scale` | full-size wall time only |

The two smoke runs ran concurrently (8 workers in total), and so did the two
timing probes (6). Per-unit wall seconds are in each summary
(`wall_seconds_per_replicate` / `wall_seconds_per_unit`).

What was checked here (pipeline properties, not results):

- all 3 + 42 smoke units and 2 + 4 probe units returned `status: OK`;
- every Arm A row at K'* = k* (500) equals single-stage T2 exactly, and every
  row at K'* = 10000 equals single-stage T4/T3 exactly (80 rows, 0 violations);
- with `--stage1-train-scores oof` (default) no shortlist-trained stage-2 fit
  failed. An earlier uncommitted run with `insample` failed every gboost
  K'* = 500 unit (training shortlist 100/100 positive), which is why OOF is
  the default.

Logs are in `logs/`. The repository `.gitignore` excludes `*.log`, so they are
not committed.

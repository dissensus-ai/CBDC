# DEV SEEDS. NOT REPORTABLE.

Everything in this directory was generated on the development seed block
700001–700999, at toy or single-unit sizes. The runs show that the exploratory
drivers work end to end and measure how long they take. None of it is a
finding. Do not quote, plot or summarize these numbers anywhere. The DEV block
is burned: no reported run may use it (`seed_guard.py`).

Environment: Python 3.14.7, numpy 2.3.5, scipy 1.16.3, scikit-learn 1.8.0,
pandas 2.3.3. Each summary's `provenance` records the commit and whether the
tree was clean.

## Round 2: E9/E10 redesign (22 Sep 2026)

These runs used the piecewise λ path, the paired identity stream and three
models.

| directory | commit | command (run from `detection/exploratory/`) | purpose |
|---|---|---|---|
| `e10_paired/` | 6cc9c49 | `run_signal_scale.py --seed-base 700401 --R 3 --n-train 2000 --n-test 2000 --workers 8 --out-dir _dev_smoke/e10_paired` | E10 pipeline check: 9-point λ grid, gboost + logit + additive |
| `timing_probe/e10_paired/` | 6cc9c49 | `run_signal_scale.py --seed-base 700411 --R 2 --lambda-grid 0.0 2.0 --bootstrap-B 200 --workers 4 --out-dir _dev_smoke/timing_probe/e10_paired` | full-size (10,000/10,000) wall time, 4 concurrent units |
| `timing_probe/e10_paired_8workers/` | 6cc9c49 | `run_signal_scale.py --seed-base 700421 --R 8 --lambda-grid 1.0 --bootstrap-B 50 --workers 8 --out-dir _dev_smoke/timing_probe/e10_paired_8workers` | full-size wall time, 8 concurrent units |
| `e9_driver_check/` | 464c552 | `run_two_stage.py --seed-base 700451 --R 3 --n-train 1500 --n-test 1500 --train-variants full shortlist --workers 3 --out-dir _dev_smoke/e9_driver_check` | the E9 driver after the lock, stop-rule and K'_q changes |

What these runs checked. These are pipeline properties, not results.

- All 27 + 4 + 8 E10 units and all 3 E9 replicates returned `status: OK`.
  No λ tripped the stop rule.
- Pairing: T2 true positives were identical across all 9 λ values in every
  (replicate, model), 9 of 9. With the paired stream, λ changes only the
  identity columns.
- Timing: a full-size unit took 29.3 s at 4 concurrent units and 30.8 s mean
  (31.0 s max) at 8.
- `run_signal_scale.py --seed-base 2026120001 --R 52` with no lock is refused
  before anything is generated.

Seeds used by the unit tests: 700001, 700010, 700011, 700012, 700020.

Also burned, by runs whose output was deleted and superseded:

- 700431–700432: an E9 driver check from 6cc9c49, run before the K'_q summary
  existed.
- 700441–700443: the same check with uncommitted code (a dirty tree).

The check was re-run from 464c552 on 700451–700453.

## Round 1: first build, straight low→high λ line (superseded)

These runs predate the redesign. The λ line they used was replaced by the
piecewise path, and the E9 defaults were n_train = 8000.

| directory | commit | command | purpose |
|---|---|---|---|
| `two_stage/` | 082bf6c | `run_two_stage.py --seed-base 700001 --R 3 --n-train 2000 --n-test 2000 --train-variants full shortlist --workers 3 --out-dir _dev_smoke/two_stage` | E9 pipeline check |
| `signal_scale/` | 082bf6c | `run_signal_scale.py --seed-base 700101 --R 3 --n-train 2000 --n-test 2000 --workers 5 --out-dir _dev_smoke/signal_scale` | straight-line λ path, single RNG stream, 2 models (superseded) |
| `timing_probe/two_stage/` | 082bf6c | `run_two_stage.py --seed-base 700201 --R 2 --train-variants full shortlist --workers 2 --out-dir _dev_smoke/timing_probe/two_stage` | full-size (8000/10000) wall time |
| `timing_probe/signal_scale/` | 082bf6c | `run_signal_scale.py --seed-base 700301 --R 2 --lambda-grid 0.0 1.0 --bootstrap-B 200 --workers 4 --out-dir _dev_smoke/timing_probe/signal_scale` | full-size wall time (superseded path) |

Round 1 checks:

- Every E9 row at K'* = 500 equalled single-stage T2 exactly, and every row at
  K'* = 10000 equalled single-stage T4/T3 exactly (80 rows, 0 violations).
- Out-of-fold stage-1 training scores gave no shortlist-fit failures. An
  earlier, uncommitted run with in-sample scores failed every gboost
  K'* = 500 unit, because the training shortlist was 100/100 positive.

Logs are in `logs/`. The repository `.gitignore` excludes `*.log`, so the logs
are not committed.

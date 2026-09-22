# E9 two-stage screening and E10 identity-signal break-even: design note

**Tier:** exploratory code and design. Nothing here is a result. The governing
document is the draft protocol addendum
`cbdc-privacy/PROTOCOL_ADDENDUM_E9_E11_DRAFT_22SEP2026.md` (sections 0–2 and
amendment A0 part 1), which is **not yet approved**. Until it is approved and a
lock exists, the drivers run only on the DEV seed block 700001–700999, and
outputs from that block are never reportable (`_dev_smoke/README.md`).

Both arms reuse the confirmatory machinery:

- independent train/test worlds per replicate (`confirmatory/replicate.run_replicate`);
- the review budget k = k*·n/10,000 with k* = 500 (`endpoint.budget_for`);
- the lower-id deterministic tiebreak (`endpoint.select_alerts`);
- the confirmatory pair `logit` (M1) and `gboost` (M4) from `detection_experiment._models`;
- the degeneracy audit on each training world;
- failures counted, never redrawn.

Two code changes outside `exploratory/` are needed, and both are inert when off:

| change | when off | check |
|---|---|---|
| `run_replicate(cfg_factory=None, score_hook=None)` | record byte-identical to d28bef0 | compared against the original module at n = 2000; `test_hooks_are_inert` |
| `DGPConfig.identity_rng_stream = False` | `generate()` output and `to_dict()` byte-identical to d28bef0 | golden SHA-256 digests of the entities, wallets and transactions tables plus the config, for `default_config` at two seeds, `surveillance_strong_config`, and three surface worlds (`test_off_is_byte_identical_to_d28bef0`) |

`to_dict()` omits `identity_rng_stream` when it is off. Otherwise every config
serialized before the option existed (ladder records, the `run_diagnostics.py`
resume check, the ladder replay) would stop matching. Wherever the confirmatory
code is cited by commit, cite d28bef0 (addendum A0).

## E9: two-stage screening (`two_stage.py`, `run_two_stage.py`)

**Procedure, per test world and model.**

1. Stage 1 ranks all n entities by the T2 score and shortlists the top K'.
2. Stage 2 re-ranks only the shortlist by the T4 (or T3) score and reviews the top k.

Both stages call `select_alerts` with the entities' own tiebreak keys. The
scores are the ones `run_replicate` already computed, delivered through
`score_hook`; nothing is re-fitted.

**Exact identities (tested, including under score ties):**

- K' = k gives the single-stage T2 reviewed set.
- K' = N gives the single-stage T4 (or T3) reviewed set.

Both hold as set equality, not just equal counts, and on every applicable
dev-smoke row.

**Per-replicate fields** (one row per model × stage-2 tier × training variant × K'\*):

| field | definition |
|---|---|
| `MissedPer10k` | missed illicit entities per 10k under two-stage review |
| `DisclosedPer10k` | 10,000·K'/n |
| `gap_recovered` | (miss_T2 − miss_2s)/(miss_T2 − miss_hi) |

`gap_recovered` is NaN with `gap_zero` when \|denominator\| < 1e-9. It is
computed but flagged `gap_negative` when the denominator is negative.

**Summary statistics.**

- **Recovered share, A0 primary:** the ratio of replicate means,
  `gap_recovered_ratio_of_means`. The mean of per-replicate ratios is
  secondary, reported with a t-interval and with zero-gap replicates counted.
  In the n = 2000 dev smoke, 6 of 8 (model, tier, variant) combinations had one
  replicate out of three with a zero gap.
- **K'₅₀ and K'₉₀** (addendum §1), computed by `kprime_summary`. The
  point-curve version is the smallest grid K'\* whose ratio-of-means share is
  at least 0.5 or 0.9. The lower-bound version uses the lower limit of a 90%
  bootstrap band instead; the bootstrap resamples whole replicates across all
  K' values. Only replicates carrying every grid point enter. A zero-gap K'
  point never counts as reaching q. Implementation choice for the addendum to
  confirm: "the lower 90% bound" of a ratio of means has no t-form, so it is
  this bootstrap limit. Like λ\*\_LB, it depends on R.
- **Primary cell, A0:** gboost, stage 2 = T4, `full` training. It is recorded
  in the summary's `primary_cell` field.

**Stage-2 training assumption.**

`full` is the default and the A0 primary. The stage-2 model is fitted on the
whole training world, so identity attributes are observed for every training
entity. The headline disclosure count is test-time lookups only (= K'). The
full-population fit needs identity-resolved training labels, which in practice
come from previously investigated cases (A0).

`shortlist` is a sensitivity check. Stage-2 models are refitted on the
training entities that stage 1 would have shortlisted on the training world,
at the same K' rate. The stage-1 training scores are out-of-fold (entity
5-fold) by default. In-sample gboost scores memorize the training world: at
K'\* = 500 and n_train = 2000 the in-sample training shortlist was 100/100
positive in every dev replicate and could not be fitted. A single-class
training shortlist is logged in `two_stage_failures` and never imputed.

**Neither variant bounds the other.** The shortlist variant changes both the
training population and its size. Its gap from `full` shows how much the
training assumption matters in this generator, but it is not an upper or lower
bound. The addendum's A0 wording, "the shortlist-trained variant bounds how
much this matters", should become "indicates how much this matters".

## E10: identity-signal break-even (`identity_signal.py`, `break_even.py`, `extra_models.py`, `run_signal_scale.py`)

**Path.** λ ∈ [0, 2] is piecewise linear through the three resolved
`surface_configs` identity blocks. Tuple fields interpolate element-wise.

- λ ∈ [0, 1]: from s = low to s = mid.
- λ ∈ [1, 2]: from s = mid to s = high.

Interpolation uses (1−t)·a + t·b, which returns the endpoints bit-exactly.
Behaviour parameters, b and prevalence come unchanged from
`world_config(b, "low", p)`. The anchors λ = 0, 1, 2 reproduce s = low, mid and
high field for field, for every b and for two prevalences (tested). λ = 1 is
also field-identical to `default_config`. At λ = 0 the identity attributes
carry no class signal by construction.

`mid_position()` now confirms that the anchors are exact. For the record, it
also keeps the straight-line finding that ruled the straight line out: on a
low→high line, the λ reproducing s = mid would be 0 for the legit-side fields,
0.27–0.54 for the launderer rates and 1 for launderer account age (pinned by
test).

**The kink at λ = 1. Disclose it wherever λ\* is reported.** The path is
continuous but not differentiable at λ = 1, and the two segments move
different parameters (pinned by `test_kink_segments_move_different_parameters`).

| segment | what moves | what stays fixed |
|---|---|---|
| [0, 1] | launderer side only: `watchlist_tpr` 0.02→0.40, `sar_lambda_launderer` 0.10→0.40, `kyc_low_prob_launderer` 0.30→0.45, `juris_beta_launderer` (2,5)→(2.4,4), `acct_age_mu_launderer` 6.3→5.9 (its whole range) | legit side is equal at low and mid |
| [1, 2] | launderer rates continue (`watchlist_tpr`→0.72, `sar_lambda_launderer`→1.20, `kyc_low_prob_launderer`→0.70, `juris_beta_launderer`→(3.5,3)); legit side also moves: `sar_lambda_legit` 0.10→0.05, `kyc_low_prob_legit` 0.30→0.20, `juris_beta_legit[0]` 2.0→1.8 | launderer account age |

Equal steps in λ on the two segments are not equal steps in any common
signal measure. A λ\* is read on its own segment's scale. A crossing at, say,
λ = 0.6 means 60% of the way from s = low to the default world, not 30% of the
way to s = high.

**Paired identity stream.** The E10 factory `cfg_factory(λ)` sets
`identity_rng_stream=True`. Identity attributes are then drawn from
`np.random.default_rng(SeedSequence(seed).spawn(1)[0])`, a child stream that
depends on the seed alone, and the main stream skips those draws. Tests
establish three things:

- **(a) Option off is unchanged.** Output is byte-identical to d28bef0 (the
  golden digests above).
- **(b) Option on pairs the worlds.** Labels, wallets and transactions are
  identical across λ at a fixed seed, and all five identity columns change
  between λ = 0 and λ = 2. Without the option, the transactions do change
  (tested), which is why the option exists.
- **(c) Option on at λ = 1 gives a valid world.** The schema matches, values
  are in range, and both degeneracy gates pass at n = 2000. Watchlist rates
  are still class-conditional.

This world is a **different realisation** from the option-off default world at
the same seed. Labels match because they are drawn first, but the identity
columns and every transaction differ. A paired-stream λ = 1 estimate is
therefore not a re-run of the confirmatory default world; it is the same
generator with a different draw. In the dev smoke, T2 results were identical
across all 9 λ values in every (replicate, model): the pairing reaches through
features, fits and the endpoint.

**Models.**

- M1 = `logit` and M4 = `gboost` come from `run_replicate`.
- M2 = `additive` is fitted through `score_hook` (`extra_models.py`) and
  writes the same record fields. It is `HistGradientBoostingClassifier(
  interaction_cst="no_interactions", class_weight="balanced")` at sklearn
  defaults, the untuned no-interaction counterpart of the confirmatory gboost.
- M2 is **not** the ladder M2, which tunes a 4-point grid by nested CV. The
  addendum should state that E10's M2 is untuned, like its M1 and M4.

**Per-λ estimands.** For each (model, λ), the driver records
Δ(T2→T4) = MissedPer10k(T2) − MissedPer10k(T4), and the same for T3. A
positive Δ means identity helps. Each Δ gets its mean and a two-sided 90%
Student-t interval. Intervals are pointwise.

**Break-even estimator** (`break_even.py`: pure, unit-tested on synthetic inputs).

- **λ\*\_mean, primary (A0):** the first λ at which mean Δ exceeds τ,
  linearly interpolated between the bracketing grid points. Its band is a
  replicate-row bootstrap: whole replicates are resampled across all λ, which
  keeps the pairing (B = 2000, 90% percentile).
- **λ\*\_LB, secondary:** the first λ at which the lower 90% t-bound exceeds
  τ. It is reported but not headlined, because it moves toward λ\*\_mean as R
  grows (tested).
- The summary names `primary: "lambda_star_mean"` and lists it first. The
  top-level summary carries `primary_estimand`.
- Rules: strict inequality, so touching τ is not exceeding it; a NaN point is
  not evidence.
- Flags (reported, never filled in): `left_censored` (value = λ₀),
  `not_reached` (NaN), `recrossed`.
- τ ∈ {0, 1.0} missed per 10k. 1.0 is a reference level that matches D10, not
  a policy margin.

**Stop rule (A0).** The run halts, cancelling pending units, as soon as more
than 10% of the R units at any single λ fail. The summary records `halted` and
`failures_per_lambda`.

## Seed policy, lock, registry, scan

`seed_guard.check_seeds` applies these rules in order:

1. With no lock, the whole train block must sit inside DEV.
2. With a lock, the block may not touch DEV at all.
3. Every train and test seed (train + 10,000,003) must lie in
   [0, 2³²−1]. Seeds pass to sklearn unnarrowed.
4. No seed may fall in a hard-coded spent block: 20260707; 900001–900020 and
   their test images; 2026080501.. and their test images; 2026081901..;
   2026081951..; 2026082051..; 2026091201...
5. With a lock, `addendum_lock.validate` must pass (below).

A lock cannot override rules 3 and 4.

**Lock format** (`addendum_lock.py`). JSON with these fields: `arm` (E9, E10
or E11), `seed_base`, `R`, `grid`, `models`, `protocol_file` (relative to the
repository root), `protocol_sha256`, `freeze_commit`, `written`, and
`lock_sha256` (a hash of everything else). `validate` refuses unless every one
of these holds:

- the self-hash matches;
- the run's arm, seed base, R, grid and model set equal the lock's;
- R ≤ 10,000;
- the protocol file on disk hashes to `protocol_sha256`;
- the freeze commit exists and is an ancestor of HEAD;
- the run's train and test blocks overlap neither the registry nor any seed
  recorded in the repository's result files.

`write_lock` performs the same block check before writing. **No lock has been
written for the reported bases.** That happens after MF approves the addendum
and the protocol is committed.

**Spent-seed registry** (`SPENT_SEEDS.json`, created on the first locked
run). A locked run records its train and test blocks as `in_progress`
**before** generating anything, and marks them `completed` at the end. A
crashed run therefore still burns its seeds, and later runs refuse overlap
with either status. This is stricter than "append after a completed run", on
purpose. DEV runs are not registered, because the whole DEV block is burned.

**Recorded-seed scan** (`seed_scan.py`). There are two passes.

- The structured pass is the gate. It reads JSON, JSONL and CSV files. Any
  key or column whose name contains "seed" contributes its value(s). A
  `seed_base` or `conf_seed_base` with an `R` beside it contributes its whole
  block, plus the test image when an offset is recorded. Lock files and the
  registry are skipped as declarations, not records.
- The raw pass reports every 6–10 digit integer in the queried range found in
  any text file, as pointers to read.

Positive control: the scanner finds the known spent blocks, including 106
structured hits for 2026080501–2026080552 and 22 for the September permutation
seeds.

**Clearance of the reported ranges (22 Sep 2026).**

| range | this worktree (23 result files record seeds) | whole `cbdc-privacy/` tree, read-only (73 result files, including the main checkout and sibling worktrees) |
|---|---|---|
| 2026110001–2026139999 | structured **CLEAR**; one raw mention, in `addendum_lock.py`'s docstring example | structured **CLEAR**; raw mentions only in the addendum draft's planning text and that docstring |
| 2036110004–2036140002 (the test-seed image) | **CLEAR** | **CLEAR** |

## Outputs schema

`run_two_stage.py --out-dir D` writes two files.

- `two_stage_replicates.jsonl`: one `run_replicate` record per replicate, plus
  `wall_seconds`, `two_stage` rows, `two_stage_config`, and optionally
  `two_stage_failures`. Each row holds `model, stage2_tier, train_variant,
  kprime_star, kprime, k, n_fit_stage2, n_pos_fit_stage2, TP, MissedPer10k,
  DisclosedPer10k, miss_T2, miss_hi, gap_recovered, gap_flag`.
- `two_stage_summary.json` with these fields:
  - header: `arm="E9"`, `status="EXPLORATORY"`, `halted`, `primary_cell`,
    `R`, `seed_mode`, `seed_base`, `test_seed_offset`, `addendum_lock`, `args`,
    `provenance`, `stage2_training_default`, `R_planned`, `R_ok`, `failures`,
    `wall_seconds_*`;
  - `cells[]`: per-cell MissedPer10k t-interval, both recovered-share
    estimators, and zero-gap and negative-gap counts;
  - `kprime_q{"model|tier|variant"}`: the curve with bootstrap band, and
    `K_q["0.5"|"0.9"]["mean"|"LB"]`.

`run_signal_scale.py --out-dir D` writes two files.

- `signal_scale_replicates.jsonl`: one record per (λ, replicate) unit,
  including the `additive` model fields, plus `lambda`, `lambda_index` and
  `wall_seconds`.
- `signal_scale_summary.json` with these fields:
  - header: `arm="E10"`, `status`, `halted`, `primary_estimand`,
    `secondary_estimand`, `path`, `identity_rng_stream=true`, `models`, `R`,
    the seed fields, `provenance`, `lambda_grid`, `mid_position`, `n_units`,
    `n_ok`, `failures`, `failures_per_lambda`, `wall_seconds_*`;
  - `results["<model>|T2_minus_T4" / "…T2_minus_T3"]`: `per_lambda[]`, and
    `break_even[]` per τ (`primary`, `lambda_star_mean`, flags,
    `lambda_star_LB`, flags, `curve`, `bootstrap`).

Both drivers refuse to overwrite an existing summary. NaN is written as the
literal `NaN`.

## Command lines for the reported runs (after approval and locks)

Placeholders: `<E9_LOCK>` and `<E10_LOCK>` are written with `write_lock` after
the freeze commit. The seed bases are the addendum's. Run from the repository
root, on a clean checkout that descends from the freeze commit.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python detection/exploratory/run_two_stage.py \
  --seed-base 2026110001 --R 52 --addendum-lock <E9_LOCK> \
  --n-train 10000 --n-test 10000 --k-star 500 \
  --kprime-grid 500 750 1000 1500 2000 3000 5000 10000 \
  --stage2-tiers T4 T3 --train-variants full shortlist \
  --stage1-train-scores oof --alpha 0.10 --bootstrap-B 2000 --workers 8 \
  --out-dir revision-<date>/evidence/e9-two-stage

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python detection/exploratory/run_signal_scale.py \
  --seed-base 2026120001 --R 52 --addendum-lock <E10_LOCK> \
  --n-train 10000 --n-test 10000 --k-star 500 --b mid --prevalence 0.05 \
  --lambda-grid 0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 \
  --models gboost logit additive --thresholds 0 1.0 --alpha 0.10 \
  --bootstrap-B 2000 --workers 8 \
  --out-dir revision-<date>/evidence/e10-break-even
```

The E9 lock's `grid` is the K'\* list and its `models` are `["gboost",
"logit"]`. The E10 lock's `grid` is the λ list and its `models` are
`["gboost", "logit", "additive"]`.

## Wall-time estimates (from dev-seed probes, not full runs)

| arm | measured per unit | run size | estimate |
|---|---|---|---|
| E10 | 30.8 s mean (31.0 s max) per (λ, replicate), all three models, n_train = n_test = 10,000, 8 concurrent units; 29.3 s at 4 concurrent | R = 52 × 9 λ = 468 units, 8 workers | 468 × 30.8 s / 8 ≈ **30 min**; allow 35 |
| E9 | 29 s per replicate at 8,000/10,000 with both training variants (earlier probe) | R = 52, n_train 10,000 | ≈ 32 s per replicate, 52 × 32 s / 8 ≈ **3.5 min**; allow 5 |

Bootstrap time is under a minute per arm. These figures are extrapolated from
unit timings.

## Open items for the addendum

1. **E9 lower-bound K'\_q.** Confirm that "the lower 90% bound" for the ratio
   of means is the replicate-bootstrap lower limit, which is what is
   implemented.
2. **A0 wording.** Change "the shortlist-trained variant bounds how much this
   matters" to "indicates", since the variant is not a bound.
3. **E10 M2 specification.** State that it is untuned
   (HistGB, no interactions, sklearn defaults), not the nested-CV ladder M2.
4. **E10 at λ = 1 under the paired stream.** State that it is a different
   draw from the default world at the same seed, so E10's λ = 1 Δ is not
   expected to equal the confirmatory estimate except in distribution.
5. **The kink.** State the segment-scale reading of λ\* (above) wherever λ\*
   is reported.
6. **After approval.** Commit the protocol, run `write_lock` for E9 and E10,
   then run.

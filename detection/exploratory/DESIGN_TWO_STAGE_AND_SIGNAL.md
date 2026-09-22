# E9 two-stage screening and E10 identity-signal gain-threshold crossing: design note

**Tier:** exploratory code and design. Nothing here is a result.

The governing document is the draft protocol addendum
`cbdc-privacy/PROTOCOL_ADDENDUM_E9_E11_DRAFT_22SEP2026.md` (§§0–2 and
amendment A0, part 1). MF has **not yet approved** it. Until MF approves it
and a lock exists, the drivers run only on the DEV seed block 700001–700999.
Outputs from that block are never reportable (see `_dev_smoke/README.md`).

**Common machinery.** Both arms reuse the confirmatory setup:

- independent train/test worlds per replicate, via
  `confirmatory/replicate.run_replicate`;
- n_train = 8000 and n_test = 10,000, the confirmatory lock's D2 and D1;
- review budget k = k*·n/10,000 with k* = 500, via `endpoint.budget_for`;
- the lower-id deterministic tiebreak, via `endpoint.select_alerts`;
- the confirmatory model pair `logit` (M1) and `gboost` (M4), from
  `detection_experiment._models`;
- the degeneracy audit on each training world;
- failures counted, never redrawn.

**R = 52 matches the confirmatory replicate count, not its precision.** The
interval widths each run achieves are reported alongside every estimate. They
are not assumed to equal the confirmatory widths.

**Code outside `exploratory/`.** Two changes were needed. Both are inert when
switched off.

| change | behaviour when off | how that was checked |
|---|---|---|
| `run_replicate(cfg_factory=None, score_hook=None)` | record byte-identical to d28bef0 | compared against the original module at n = 2000; `test_hooks_are_inert` |
| `DGPConfig.identity_rng_stream = False` | `generate()` output and `to_dict()` byte-identical to d28bef0 | golden SHA-256 digests of the entities, wallets and transactions tables plus the config: `default_config` at two seeds, `surveillance_strong_config`, and three surface worlds (`test_off_is_byte_identical_to_d28bef0`) |

`to_dict()` omits `identity_rng_stream` when it is off. Configs serialized
before the option existed therefore still match: the ladder records, the
`run_diagnostics.py` resume check and the ladder replay. Wherever the
confirmatory code is cited by commit, cite d28bef0 (A0).

## E9: two-stage screening (`two_stage.py`, `run_two_stage.py`)

**Procedure, per test world and model.**

1. Stage 1 ranks all n entities by the T2 score and shortlists the top K'.
2. Stage 2 re-ranks only that shortlist by the T4 score (or T3) and reviews
   the top k.

Both stages call `select_alerts`, and each entity keeps its own tiebreak key.
The scores are the ones `run_replicate` already computed, delivered through
`score_hook`. Nothing is re-fitted.

**Exact identities.** These are tested, including under score ties. They hold
as set equality, not merely equal counts, and on every applicable dev-smoke
row.

- K' = k gives the single-stage T2 reviewed set.
- K' = N gives the single-stage T4 (or T3) reviewed set.

**Primary output: absolute.** The first key of the summary JSON is
`primary_absolute_missed_vs_identity_lookups`. For each model × stage-2 tier ×
training variant it reports:

- missed illicit entities per 10k against test-time identity lookups per 10k
  (K'), along the K' grid;
- replicate means with 90% t-intervals;
- the two single-stage reference points: T2, which uses no identity lookups,
  and the full tier, which uses identity for all n entities.

It involves no ratio, so it is defined whatever the sign of the identity gain.
The A0 primary cell (gboost, T4, `full` training) is flagged
`is_primary_cell`.

**Secondary annotation: recovered share and K'_q** (`recovery_summary`, under
`recovery_secondary`). Everything in this block is gated on the reference gain
G = miss_T2 − miss_hi. Its 90% t-interval is always reported.

| condition on G | what is reported |
|---|---|
| mean G ≤ 0 (full identity doesn't help, or increases misses) | every share, the mean of ratios and every K'_q are `"not_applicable"`, with `reason: "nonpositive_reference_gain"`. There is no gain to recover. Toy test: misses go from 10 to 20 under full identity. |
| 90% t-interval of G includes 0 | flag `"reference_gain_not_separated"`. The ratio and its bootstrap band are kept, with role `secondary_annotation`. |
| otherwise | the share is computed. The block is still a secondary annotation to the absolute table. |

Details of the secondary block:

- **Share at K'.** Computed as the ratio of replicate means (the A0 primary
  estimator): (mean miss_T2 − mean miss_2s) / mean G. The mean of
  per-replicate ratios is kept as a secondary estimator; it excludes and
  counts replicates whose own gain is ≤ 0.
- **Bootstrap.** Whole replicates are resampled across all K' values.
  - A draw whose resampled mean G ≤ 0 is **not applicable**. These draws are
    counted in `n_not_applicable` and `p_not_applicable`, never dropped
    silently.
  - `band_all_draws` is computed over all B draws, with not-applicable draws
    entering at −∞. A limit that falls in that mass is reported as `None`
    with the label "in not-applicable mass".
  - `band_conditional_on_positive_reference_gain` covers applicable draws
    only and says so in its name.
- **K'_q** is the smallest grid K' whose share is ≥ q, for q ∈ {0.5, 0.9}.
  - K'_q (point) uses the point curve.
  - K'_q_LB uses the lower limit of the all-draws band. It depends on R.
  - `None` means no grid point reaches q.

**Stage-2 training assumption.**

- **`full` (default and A0 primary).** The stage-2 model is fitted on the
  whole training world. The headline disclosure count is test-time lookups
  only (= K').
- **`shortlist` (sensitivity check).** Stage 2 is refitted on the training
  entities that stage 1 would shortlist, at the same K' rate. The stage-1
  training scores are out-of-fold by default. In-sample gboost scores
  memorize the training world: at K'* = 500 and n_train = 2000 the in-sample
  shortlist was 100/100 positive in every dev replicate and could not be
  fitted. A single-class training shortlist is logged in
  `two_stage_failures`, never imputed.

Neither variant bounds the other. The shortlist variant changes the training
population and its size together. Its difference from `full` indicates how
much the training assumption matters in this generator.

**Selective labels.** The `full` stage-2 model sees identity attributes *and*
true labels for every training entity. That includes the many customers a real
institution would never investigate, and whose true status it would therefore
never learn. E9 thus measures **test-time** disclosure, conditional on a
training population that is fully labelled and fully identity-attributed.

Real training labels exist only where someone investigated, and investigation
is itself selected by earlier scores and rules. That is the selective labels
problem (Lakkaraju, Kleinberg, Leskovec, Ludwig & Mullainathan, 2017, "The
Selective Labels Problem", *Proc. 23rd ACM SIGKDD*, pp. 275–284,
doi:10.1145/3097983.3098066; DOI verified via Crossref, 22 Sep 2026).

The `shortlist` variant only partly addresses this. It restricts identity
attributes and stage-2 labels to the shortlisted population, but its stage-1
model still learns from population-wide true labels. No arm here simulates
label selection. The manuscript should say so wherever E9 is interpreted.

## E10: identity-signal gain-threshold crossing (`identity_signal.py`, `break_even.py`, `extra_models.py`, `run_signal_scale.py`)

"Break-even" in the first build is renamed **gain-threshold crossing**: τ is a
reference gain level, not a cost-benefit balance point. The module file keeps
its old name for now.

**Path.** λ runs from 0 to 2, piecewise linear through the three resolved
`surface_configs` identity blocks:

- λ ∈ [0, 1] goes from s = low to s = mid;
- λ ∈ [1, 2] goes from s = mid to s = high.

Tuple fields interpolate element-wise. Interpolation uses (1−t)·a + t·b, which
returns the endpoints bit-exactly. Behaviour parameters, b and prevalence come
from `world_config(b, "low", p)`. What the tests establish about the path:

- λ = 0, 1 and 2 reproduce s = low, mid and high field for field, for every b
  and for two prevalences.
- λ = 1 is field-identical to `default_config`.
- At λ = 0 the identity attributes carry no class signal.
- `mid_position()` confirms the anchors.
- It also keeps the straight-line finding that ruled a straight line out. On
  a low→high line, the λ that reproduces s = mid differs by parameter: 0 for
  the legitimate-side fields, 0.27–0.54 for the launderer rates and 1 for
  launderer account age.

**The kink at λ = 1.** Disclose it wherever a crossing is reported. The path is
continuous but not differentiable at λ = 1, and the two segments move different
parameters. `test_kink_segments_move_different_parameters` pins this table.

| segment | what moves | what stays fixed |
|---|---|---|
| [0, 1] | launderer side only: `watchlist_tpr` 0.02→0.40, `sar_lambda_launderer` 0.10→0.40, `kyc_low_prob_launderer` 0.30→0.45, `juris_beta_launderer` (2,5)→(2.4,4), `acct_age_mu_launderer` 6.3→5.9 (its whole range) | legitimate side, which is equal at low and mid |
| [1, 2] | launderer rates continue (`watchlist_tpr`→0.72, `sar_lambda_launderer`→1.20, `kyc_low_prob_launderer`→0.70, `juris_beta_launderer`→(3.5,3)); legitimate side also moves: `sar_lambda_legit` 0.10→0.05, `kyc_low_prob_legit` 0.30→0.20, `juris_beta_legit[0]` 2.0→1.8 | launderer account age |

Equal steps in λ on the two segments are not equal steps in any common measure
of signal. A crossing is therefore read on its own segment's scale. For that
reason, every reported crossing now carries the actual parameter values at
that λ (see below).

**Paired identity stream.** `cfg_factory(λ)` sets `identity_rng_stream=True`.
Identity attributes are then drawn from
`np.random.default_rng(SeedSequence(seed).spawn(1)[0])`, and the main stream
skips those draws. What the tests establish:

- Option off: output is byte-identical to d28bef0.
- Option on: labels, wallets and transactions are identical across λ at a
  fixed seed, and all five identity columns change between λ = 0 and λ = 2.
- Option on at λ = 1: the world is valid. The schema matches, the values are
  in range, and both degeneracy gates pass at n = 2000.

That λ = 1 world is **a different draw** from the option-off default world at
the same seed. The labels match, but the identity columns and every
transaction differ. So E10's Δ at λ = 1 should equal the confirmatory estimate
only in distribution. In the dev smoke, T2 results were identical across all
9 λ values in every (replicate, model).

**Models.**

- M1 = `logit` and M4 = `gboost` come from `run_replicate`.
- M2 = `additive` comes through `score_hook`. It is
  `HistGradientBoostingClassifier(interaction_cst="no_interactions",
  class_weight="balanced")` at sklearn defaults. It is **untuned**, like E10's
  M1 and M4, and is not the ladder's nested-CV M2.

**Per-λ estimands.**

- Δ(T2→T4) is the effect of attributes plus watchlist.
- Δ(T2→T3) is the effect of attributes only.
- A positive Δ means identity helps.
- Each Δ is reported with its mean and a pointwise 90% t-interval, under
  `per_lambda_delta`.

**Crossings** (`break_even.py`; pure functions, unit-tested).

The two contrasts get **separate top-level tables of equal standing**:
`gain_threshold_crossing_T2_minus_T4` and `gain_threshold_crossing_T2_minus_T3`.
They are the first two keys of the summary, each organised by model × τ.

- **`crossing_mean` (primary, A0).** The first λ at which mean Δ exceeds τ,
  linearly interpolated between grid points.
- **`crossing_LB` (secondary).** The same, using the lower 90% t-bound. It
  depends on R and is not headlined.
- Touching τ does not count as exceeding it. A NaN curve point is not evidence
  of a crossing.
- Flags: `left_censored` (value = λ_min), `not_reached` (value NaN) and
  `recrossed`.
- **Parameter values at each crossing.** Every reported crossing carries
  `params_at_crossing_*`: the interpolated `watchlist_tpr/fpr`,
  `sar_lambda_*`, `kyc_low_prob_*`, `juris_beta_*`, `acct_age_mu_*` and
  `acct_age_sigma`, plus the path segment. The same values are attached to
  each finite bootstrap limit (`params`).
- τ ∈ {0, 1.0} missed per 10k. 1.0 is a reference level that matches D10. It
  is not a policy margin.

**Bootstrap band for a crossing (censoring-aware).** Whole replicate rows are
resampled, keeping the λ pairing, with B = 2000 and a 90% level.

- The percentile interval is computed over **all B draws**.
  - A draw that never crosses is right-censored at +∞.
  - A left-censored draw sits at λ_min.
  - `method="inverted_cdf"` ensures a limit is always an actual draw, never an
    interpolation towards ∞.
- A limit that falls in censored mass is reported as a labelled bound, not a
  number:
  - "`>lambda_max (not reached within tested range)`"
  - "`<=lambda_min (already above tau at lowest tested lambda)`"
- `p_no_crossing_in_range` = n_not_reached / B is reported, and so is
  `p_left_censored`.
- The finite-draws-only interval is kept as `conditional_on_crossing`, with an
  explicit note that it is not the band.

Why this changed: the first build took percentiles over finite draws only. On
a toy where 1,107 of 2,000 draws never crossed, it reported a finite band of
about 0.88–0.98. That was a conditional interval presented as the
unconditional one.

**Stop rule (A0).** The run halts, cancelling pending units, once more than
10% of the R units at any single λ have failed. The summary records `halted`
and `failures_per_lambda`.

## Seed policy, lock, registry, scan

`seed_guard.check_seeds` applies these rules in order:

1. With no lock, the whole train block must lie inside DEV.
2. With a lock, the block may not touch DEV at all.
3. Every train and test seed (test seed = train seed + 10,000,003) must lie
   in [0, 2³²−1].
4. No seed may fall in a hard-coded spent block. The protocol test-seed
   offset is required.
5. With a lock, `addendum_lock.validate` must pass.

A lock cannot override rules 3 or 4.

**Lock format** (`addendum_lock.py`). The lock is a JSON file with these keys:
`spec`, `protocol_file`, `protocol_sha256`, `freeze_commit`, `written` and
`lock_sha256`.

`spec` is the driver's full run spec; `--print-spec` emits it. It holds:

- `arm`, `seed_base`, `R`, `grid`, `models`;
- `n_train`, `n_test`, `k_star`;
- `test_seed_offset` and `identity_rng_stream`;
- `flags`, which differ by arm:
  - E9: stage-2 tiers, training variants, stage-1 training scores, alpha,
    bootstrap B;
  - E10: path, b, prevalence, thresholds, alpha, bootstrap B and seed;
- `environment`: Python, numpy, scipy, sklearn and pandas versions, plus the
  sha256 of `detection/requirements.txt`.

`validate` refuses unless **every** spec key matches exactly, including nested
flags and environment keys. Missing or unknown keys are refused. The fitted
model set is compared without regard to order, and grid ints and floats count
as equal. `validate` also checks:

- the lock's self-hash;
- that the protocol file's sha256 is unchanged;
- that the freeze commit exists and is an ancestor of HEAD;
- that R ≤ 10,000;
- that neither the registry nor any repository result file records a seed in
  the run's train or test block.

Tests cover a run that differs only in `n_train`, plus every other kind of
drift. **No lock has been written for the reported bases.**

**Spent-seed registry** (`SPENT_SEEDS.json`). A locked run writes its train
and test blocks as `in_progress` before generating anything, and marks them
`completed` at the end. A crashed run therefore still burns its seeds. DEV
runs are not registered, because the whole DEV block is burned anyway.

**Recorded-seed scan** (`seed_scan.py`). The structured pass is the gate. It
covers JSON, JSONL and CSV keys and columns whose names contain "seed", plus
`seed_base` + `R` blocks and their test images. Lock files and the registry
are skipped. The raw-text pass gives pointers only.

As a positive control, the scan finds the known spent blocks, for example 106
hits for the confirmatory seeds.

**Clearance, 22 Sep 2026:** the structured scan is **CLEAR** for
2026110001–2026139999 and for its test image 2036110004–2036140002. That holds
both in this worktree and across the whole `cbdc-privacy/` tree, read-only. The
only raw mentions are the addendum draft's planning text and one docstring.

## Summary schema (top-level keys, in order)

**E9, `two_stage_summary.json`:**

- `primary_absolute_missed_vs_identity_lookups`, containing `description`,
  `primary_cell` and `cells[]`. Each cell holds model, tier, variant,
  `is_primary_cell`, the single-stage T2 and hi-tier `MissedPer10k` values,
  and `curve[]` of K' points (`IdentityLookupsPer10k`, `MissedPer10k`
  mean_ci, `R_rows`, zero-gap and negative-gap counts).
- Header keys: `arm`, `status`, `halted`, `R`, `R_planned`, `R_ok`,
  `seed_mode`, `seed_base`, `test_seed_offset`, `addendum_lock`, `run_spec`,
  `args`, `provenance`, `stage2_training_default`, `failures`,
  `wall_seconds_total`, `wall_seconds_per_replicate`.
- `recovery_secondary`, keyed `"model|tier|variant"`. Each entry holds
  `n_replicates`, `role`, `estimator`, `reference_gain`, `status`,
  `reason`/`flags`, `bootstrap`, `curve`, `mean_of_ratios` and `K_q`.

**E10, `signal_scale_summary.json`:**

- `gain_threshold_crossing_T2_minus_T4` and
  `gain_threshold_crossing_T2_minus_T3`, each containing `contrast`, `primary`
  and `by_model{model: [per τ: crossing_mean, flag_mean, recrossed_mean,
  crossing_LB, …, params_at_crossing_mean, params_at_crossing_LB, curve,
  bootstrap{mean, LB: {lo, hi (value/censored/label/params),
  p_no_crossing_in_range, p_left_censored, n_*, conditional_on_crossing}}]}`.
- `per_lambda_delta`.
- Header keys: `arm`, `status`, `halted`, `primary_estimand`,
  `secondary_estimand`, `path`, `identity_rng_stream`, `models`, `R`,
  `seed_mode`, `seed_base`, `test_seed_offset`, `addendum_lock`, `run_spec`,
  `args`, `provenance`, `lambda_grid`, `mid_position`, `n_units`, `n_ok`,
  `failures`, `wall_seconds_total`, `wall_seconds_per_unit`,
  `failures_per_lambda`.

Both drivers refuse to overwrite an existing summary.

## Reported-run procedure (after approval)

1. Commit the approved protocol under `detection/protocols/`. That commit is
   the freeze commit.
2. Print each run spec from a clean checkout of the freeze commit or a
   descendant, in the reporting environment:
   `run_two_stage.py <reported args> --print-spec > e9_spec.json`, and the
   same for `run_signal_scale.py`.
3. Write the locks, which re-checks the seed blocks:
   `addendum_lock.write_lock(path, spec, protocol_file=..., freeze_commit=...)`.
4. Run the commands below. Every argument must equal the lock's spec, or the
   run is refused.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python detection/exploratory/run_two_stage.py \
  --seed-base 2026110001 --R 52 --addendum-lock <E9_LOCK> \
  --n-train 8000 --n-test 10000 --k-star 500 \
  --kprime-grid 500 750 1000 1500 2000 3000 5000 10000 \
  --stage2-tiers T4 T3 --train-variants full shortlist \
  --stage1-train-scores oof --alpha 0.10 --bootstrap-B 2000 --workers 8 \
  --out-dir revision-<date>/evidence/e9-two-stage

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python detection/exploratory/run_signal_scale.py \
  --seed-base 2026120001 --R 52 --addendum-lock <E10_LOCK> \
  --n-train 8000 --n-test 10000 --k-star 500 --b mid --prevalence 0.05 \
  --lambda-grid 0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 \
  --models gboost logit additive --thresholds 0 1.0 --alpha 0.10 \
  --bootstrap-B 2000 --workers 8 \
  --out-dir revision-<date>/evidence/e10-gain-threshold
```

## Wall-time estimates (from dev-seed probes, not full runs)

| arm | probe measurement | estimate for the reported run |
|---|---|---|
| E10 | 30.8 s mean per (λ, replicate) unit, all three models, n_train = n_test = 10,000, 8 concurrent units | R = 52 × 9 λ = 468 units, 8 workers: ≈ **30 min**, as an upper estimate (train is now 8000). Allow 35 min. |
| E9 | 29 s per replicate at 8000 / 10,000 with both training variants | R = 52, 8 workers: ≈ **3–4 min** |

The bootstrap adds under a minute per arm.

## Open items for the addendum

1. **A0 wording.** Replace "the shortlist variant *bounds* how much this
   matters" with "*indicates* how much this matters". The variant is not a
   bound.
2. **E10 M2.** State that it is untuned.
3. **E10 at λ = 1 under the paired stream.** State that it is a different
   draw from the default world, so it can match the confirmatory estimate
   only in distribution.
4. **The kink.** State the segment-scale reading of a crossing and report the
   parameter values at it.
5. **Selective labels.** State the disclosure-conditional reading of E9, as in
   the paragraph above.
6. **Terminology.** Replace "break-even" with "gain-threshold crossing"
   throughout the addendum.
7. **Headline outputs.** Make E9's headline absolute (missed vs lookups).
   Recovered share is a gated secondary annotation.

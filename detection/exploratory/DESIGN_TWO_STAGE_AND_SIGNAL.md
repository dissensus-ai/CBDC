# Two-stage screening and identity-signal scale: design note

**Tier:** exploratory code and design. Nothing here is a result. The seeds,
grids, primary estimands and any thresholds used in a reported run belong to a
protocol addendum that has not been written or approved. Until then the drivers
run only on the DEV seed block 700001–700999, and outputs from that block are
never reportable (`_dev_smoke/README.md`).

Both arms reuse the confirmatory machinery unchanged: independent train/test
worlds per replicate (`confirmatory/replicate.run_replicate`), the review
budget k = k*·n/10,000 with k* = 500 (`endpoint.budget_for`), the lower-id
deterministic tiebreak (`endpoint.select_alerts`), models `logit` and `gboost`
(`detection_experiment._models`), the training-world degeneracy audit, and
failures counted and not redrawn. `run_replicate` gained two optional
arguments, `cfg_factory` and `score_hook`. With both unset, the record is
byte-identical to d28bef0. This was checked against the original module at
n = 2000, and `test_hooks_are_inert` checks it on every test run.

## Arm A: two-stage screening (`two_stage.py`, `run_two_stage.py`)

**Question.** How much of the single-stage T2→T4 (or T2→T3) reduction in
missed illicit entities is kept if identity attributes are disclosed only for
a shortlist?

**Procedure, per test world and model.** Stage 1 ranks all n entities by the
T2 score and shortlists the top K'. Stage 2 re-ranks only the shortlist by the
T4 (or T3) score and reviews the top k. Both stages call `select_alerts` with
the entities' own tiebreak keys. The scores are the ones `run_replicate`
already computed: `scores[(model, tier)]`, delivered through `score_hook`,
with no re-fit.

**Exact identities (tested, including under score ties).**
K' = k gives the single-stage T2 reviewed set. K' = N gives the single-stage
T4 (or T3) reviewed set. Both hold as set equality, not only as equal counts.
They also held on all 80 applicable dev-smoke rows.

**Estimands per (model, stage-2 tier, training variant, K'\*):**

| field | definition |
|---|---|
| `MissedPer10k` | 10,000·(N_pos − TP_two-stage)/n |
| `DisclosedPer10k` | 10,000·K'/n (identity lookups at scoring time) |
| `gap_recovered` | (miss_T2 − miss_2s)/(miss_T2 − miss_hi). NaN with `gap_flag = "gap_zero"` when \|denominator\| < 1e-9. Computed but flagged `gap_negative` when the hi tier missed more than T2 |

Across replicates, the summary reports `MissedPer10k` with a 90% Student-t
interval (`inference.mean_ci`) and gap recovery computed two ways:
`gap_recovered_mean_of_ratios`, a t-interval over replicates with gap-zero
replicates dropped and counted, and `gap_recovered_ratio_of_means`. At k* = 500
the boosted T2→T4 gap is only a few entities per replicate (the confirmatory
estimate is 7.9 per 10k). Per-replicate ratios with small denominators are
therefore unstable. In the n = 2000 dev smoke, 6 of the 8 (model, tier,
variant) combinations, including gboost T4, had one of their three replicates
hit `gap_zero`. The ratio of means is the more stable choice. The addendum must
pick one.

K' is set by `budget_for(K'*, n)`, clipped at n, and must be ≥ k. The default
grid is K'\* ∈ {500, 750, 1000, 1500, 2000, 3000, 5000, 10000} per 10,000
(`--kprime-grid`).

### Stage-2 training assumption (read this before using any Arm A number)

**Default: `full`.** The stage-2 model is the T4 (or T3) model fitted on the
whole training world, which means identity attributes were observed for every
training entity. The disclosure count covers operational, test-time lookups
only. Historical identity access for training is assumed and not counted. This
is the default because it holds the model fixed between the single-stage and
two-stage arms, so gap recovery isolates the change in disclosure pattern at
scoring time. It is not an upper or lower bound on any deployable system. It
describes a regime in which a regulator lets the full identity-bearing history
be used for training and restricts disclosure only in operation.

**Variant: `shortlist`** (`--train-variants full shortlist`). For each K'\*,
stage-2 models are refit on only the training entities that stage 1 would have
shortlisted on the training world, at the same K' rate per 10k. This is the
regime where identity is never collected outside a shortlist, including
historically. The stage-1 training scores used for that shortlist are:

- `oof` (default): entity 5-fold out-of-fold T2 scores on the training world;
- `insample`: the T2 model's in-sample scores.

OOF is the default because in-sample gboost scores memorize the training world.
In a dev run with `insample` at n_train = 2000, the gboost training shortlist at
K'\* = 500 was 100/100 positives in all three replicates, which left no
negative class for the stage-2 fit (recorded as `two_stage_failures`). At
larger K' the in-sample shortlist is also much richer in positives than any
test-time shortlist. With `oof`, no fit failed in the smoke or the full-size
probe. When the training shortlist has a single class, the unit is logged in
`two_stage_failures` and its row is omitted, not imputed. At K'\* = 10,000 the
variant reuses the full-world scores. Refitting on identical rows is
bit-identical (`test_full_refit_matches_fitted_model`).

Neither training variant bounds the other. The variant moves stage-2 training
to the shortlisted population, which is closer to the test-time shortlist but
far smaller.

## Arm B: continuous identity-signal scale (`identity_signal.py`, `break_even.py`, `run_signal_scale.py`)

**Construction.** For λ ∈ [0, 1], every identity-attribute field of `DGPConfig`
(the 11 fields read by `dgp._identity_attrs`) is set to
(1−λ)·v(s=low) + λ·v(s=high). Tuple fields are handled element-wise. v(s) is
the resolved field value `surface_configs.world_config` produces, with defaults
included. Behaviour parameters, b and prevalence are taken from
`world_config(b, "low", p)` unchanged. The (1−λ)a + λb form returns a and b
bit-exactly, so λ = 0 reproduces s = low field for field and λ = 1 reproduces
s = high field for field (tested for every b and for two prevalences). At
λ = 0 every launderer parameter equals its legitimate counterpart, so the
identity attributes carry no class signal (tested).

**Where s = mid falls: it is not on the line.** Values from
`python3 identity_signal.py`:

| component | low | mid | high | λ that reproduces mid |
|---|---|---|---|---|
| watchlist_tpr | 0.02 | 0.40 | 0.72 | 0.5429 |
| watchlist_fpr | 0.02 | 0.02 | 0.02 | constant (on line) |
| sar_lambda_launderer | 0.10 | 0.40 | 1.20 | 0.2727 |
| sar_lambda_legit | 0.10 | 0.10 | 0.05 | 0 |
| kyc_low_prob_launderer | 0.30 | 0.45 | 0.70 | 0.3750 |
| kyc_low_prob_legit | 0.30 | 0.30 | 0.20 | 0 |
| juris_beta_launderer[0] | 2.0 | 2.4 | 3.5 | 0.2667 |
| juris_beta_launderer[1] | 5.0 | 4.0 | 3.0 | 0.5000 |
| juris_beta_legit[0] | 2.0 | 2.0 | 1.8 | 0 |
| juris_beta_legit[1] | 5.0 | 5.0 | 5.0 | constant (on line) |
| acct_age_mu_launderer | 6.3 | 5.9 | 5.9 | 1 |
| acct_age_mu_legit | 6.3 | 6.3 | 6.3 | constant (on line) |
| acct_age_sigma | 0.7 | 0.7 | 0.7 | constant (on line) |

The components disagree, from 0 (legit-side fields) through 0.27–0.54
(launderer rates) to 1 (account age), so no λ reproduces the default world.
Two structural reasons:

1. The s = high block also changes legitimate-side parameters that s = mid
   leaves at their defaults.
2. s = high does not override `acct_age_mu_launderer`, so the account-age gap
   is already fully open at s = mid.

The λ line is therefore a new one-parameter family through the two endpoint
worlds, not a refinement of the three-level surface. A λ curve cannot be
anchored on the confirmatory default-world estimate. `test_mid_is_not_collinear`
pins these values so that a silent change to the configuration fails the test.

**Common-random-numbers caveat.** `dgp.generate` draws identity attributes
before behaviour, and the Poisson and beta samplers consume a
parameter-dependent number of uniforms. So at a fixed seed, changing λ changes
the transaction stream. This was verified: s = low and s = high at the same
seed give different transaction counts but identical labels. Same-seed worlds
at different λ share labels and wallet counts, not behaviour. The λ curve is
therefore a sequence of nearly independent worlds, not a paired counterfactual.
This is the same issue ADJUDICATION.md item 6 records for the surface.

**Per-λ estimand.** For each λ and replicate r, one independent train/test
pair (train seed = base + r, test seed = base + r + 10,000,003, the same pair at
every λ) goes through `run_replicate` with `cfg_factory(λ)`. The drivers record
Δ = MissedPer10k(T2) − MissedPer10k(T4), and the same for T3, per model, where
positive means identity helps. The summary gives the mean and a two-sided 90%
Student-t interval per λ (`inference.mean_ci`).

**Break-even estimator (`break_even.py`, pure, unit-tested on synthetic inputs).**
Input is an R × G matrix of Δ, with NaN for failed units. Two estimands are
computed, and the addendum must choose one:

- **λ\*\_LB**: the smallest λ at which the lower 90% bound of mean Δ exceeds
  τ. This is the specification as written. It depends on R: more replicates
  tighten the bound and move λ\*\_LB down toward λ\*\_mean (tested). It is a
  detection threshold for a given design, not a property of the generator.
- **λ\*\_mean**: the smallest λ at which mean Δ exceeds τ. It estimates a
  generator property and makes no claim that the effect is demonstrated.

Both use linear interpolation between the bracketing grid points. Strict
inequality applies: touching τ does not count as exceeding it. A NaN curve
point counts as not exceeding. Censoring is reported, never filled in:
`left_censored` (already above τ at the first grid point, value = λ₀),
`not_reached` (value NaN), and `recrossed` (the curve falls back to ≤ τ after
the first crossing, so "smallest λ" and "λ beyond which" differ). τ defaults to
{0, 1.0} missed per 10k (`--thresholds`).

The uncertainty band comes from a bootstrap over replicate rows: whole
replicates are resampled across all λ, the estimand is recomputed, and the
percentile band is reported together with counts of censored draws. The default
is B = 2000 at a 90% level, with the bootstrap RNG seeded from the seed base.
This is not a DGP seed. For λ\*\_LB the interval is re-derived inside each
draw, so the band describes the estimator at this R.

## Outputs schema

`run_two_stage.py --out-dir D` writes:

- `D/two_stage_replicates.jsonl`: one `run_replicate` record per replicate
  (protocol 14.2 fields) plus `wall_seconds`, `two_stage` (list of rows:
  `model, stage2_tier, train_variant, kprime_star, kprime, k, n_fit_stage2,
  n_pos_fit_stage2, TP, MissedPer10k, DisclosedPer10k, miss_T2, miss_hi,
  gap_recovered, gap_flag`), `two_stage_config`, and optionally
  `two_stage_failures`.
- `D/two_stage_summary.json`: `arm, status="EXPLORATORY", seed_mode
  (DEV|ADDENDUM), seed_base, test_seed_offset, addendum_lock, args,
  provenance (git commit, dirty flag, library versions),
  stage2_training_default, R_planned, R_ok, failures, wall_seconds_*`, and
  `cells[]`. Each cell has `DisclosedPer10k_mean`, `MissedPer10k` (mean_ci),
  `miss_T2_mean`, `miss_hi_mean`, `gap_recovered_mean_of_ratios` (mean_ci),
  `gap_recovered_ratio_of_means`, `n_gap_zero`, `n_gap_negative`, `R_rows`.

`run_signal_scale.py --out-dir D` writes:

- `D/signal_scale_replicates.jsonl`: one `run_replicate` record per
  (λ, replicate) unit plus `lambda`, `lambda_index`, `wall_seconds`.
- `D/signal_scale_summary.json`: the same header fields, plus
  `lambda_grid`, `mid_position` (the table above), `n_units`, `n_ok`,
  `failures`, and `results["<model>|T2_minus_T4" / "…T2_minus_T3"]` with
  `per_lambda[]` (mean_ci per λ) and `break_even[]` per τ: `lambda_star_LB`,
  `lambda_star_mean`, flags, the `curve`, and `bootstrap`.

Both drivers refuse to overwrite an existing summary. JSON NaN is written as
the literal `NaN`, which Python's `json` reads back.

## Seed policy (`seed_guard.py`)

- Without `--addendum-lock`, the whole train block
  seed_base..seed_base+R−1 must lie inside 700001–700999.
- With `--addendum-lock <path>`, the file must exist and contain the seed base
  as a standalone numeric token. This is a tripwire, not a freeze format.
- Test seed = train seed + 10,000,003, the protocol v3 offset.
- Every train and test seed is refused if it falls outside [0, 2³²−1].
  Exploratory seeds are passed to sklearn unnarrowed, so `model_seed()` is the
  identity for them.
- Every train and test seed is refused if it collides with a spent block:
  20260707; 900001–900020 and their test images; 2026080501.. (confirmatory)
  and its test images; 2026081901.. (burned ladder pilot); 2026081951..;
  2026082051..; and 2026091201... Blocks written as "base.." in the repository
  have no stated end and are reserved for 1000 seeds.
- Dev seeds used so far: 700001–700003, 700101–700103, 700201–700202,
  700301–700302, plus test-only calls at 700001 and 700010 in the unit tests.

## Command lines for a reported run (after the addendum exists)

`<BASE_A>`, `<BASE_B>` and `<LOCK>` are placeholders the addendum fixes. They
must be distinct, non-DEV, and outside the spent blocks. Run from the
repository root on a clean checkout of the tagged commit:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python detection/exploratory/run_two_stage.py \
  --seed-base <BASE_A> --R 52 --addendum-lock <LOCK> \
  --n-train 8000 --n-test 10000 --k-star 500 \
  --kprime-grid 500 750 1000 1500 2000 3000 5000 10000 \
  --stage2-tiers T4 T3 --train-variants full shortlist \
  --stage1-train-scores oof --alpha 0.10 --workers 8 \
  --out-dir revision-<date>/evidence/two-stage

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python detection/exploratory/run_signal_scale.py \
  --seed-base <BASE_B> --R 52 --addendum-lock <LOCK> \
  --n-train 8000 --n-test 10000 --k-star 500 --b mid --prevalence 0.05 \
  --lambda-grid 0 0.05 0.1 0.15 0.2 0.25 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
  --thresholds 0 1.0 --alpha 0.10 --bootstrap-B 2000 --workers 8 \
  --out-dir revision-<date>/evidence/signal-scale
```

## Wall-time estimate

The smoke runs used n = 2000/2000 on the dev block. A single full-size unit
(n_train = 8000, n_test = 10,000) took about 29 s for Arm A (both training
variants, OOF included, both tiers, 8 K' values) and about 27 s per (λ,
replicate) unit for Arm B. The probes ran with 6 processes concurrently on the
12-core 9900X.

- **Arm A, R = 52:** 52 × 29 s / 8 workers ≈ 3–4 min. Dropping the shortlist
  variant roughly halves it.
- **Arm B, R = 52 × 14 λ = 728 units:** 728 × 27 s / 8 ≈ 41 min. Allow 45–60
  min for contention at 8 workers. The bootstrap (4 contrasts × 2 τ × B = 2000)
  adds under a minute.

These are extrapolations from single-unit timings, not measurements of a full
run. Memory per worker is modest; 8 workers stay well inside 128 GB.

## Decisions the protocol addendum must make

1. **Arm A primary cell.** Primary model (gboost, following protocol v3 L3?),
   stage-2 tier (T4 or T3), training variant (`full` default or `shortlist`),
   and whether the K' grid is fixed or only descriptive.
2. **Gap-recovery estimator.** Ratio of means (recommended, because the gboost
   gap is only a few entities per replicate) or mean of ratios. Also how
   `gap_zero` and `gap_negative` replicates are handled.
3. **Disclosure accounting.** Whether training-time identity access counts
   against the disclosure budget. Under `full` it does not; if it should, only
   the `shortlist` variant is consistent with that accounting.
4. **Arm B estimand.** λ\*\_LB, which depends on R, or λ\*\_mean. Which
   τ values, and why. τ = 1.0 matches D10's interval-width target
   (tau_delta), not a policy margin, and the two must not be conflated. The
   primary model and contrast (T2→T4 or T2→T3), the bootstrap B and level,
   and the λ grid also need fixing.
5. **The λ path does not pass through s = mid.** Either accept a family that
   excludes the default world and say so, or define a different path, for
   example a piecewise low→mid→high path. Such a path would need a new
   constructor; the tests would still pin the endpoints.
6. **Common random numbers.** Accept the non-CRN λ worlds, or add a generator
   option that draws identity attributes from a separate RNG substream. The
   option would be off by default so every existing reproduction is
   unchanged, and it would be a generator change, which is a scientific
   decision.
7. **Seeds and lock.** Two distinct non-DEV bases, a real lock format (the
   current check only confirms that the file names the base), and adding the
   spent blocks to `seed_guard.FORBIDDEN_RANGES` once they have been used.
8. **Failure rule.** Protocol 6.3's 10% stop threshold, applied per λ and per
   arm. The audit gate runs on every training world, including λ = 1.
9. **Multiplicity.** Arm A has 8 K' × 2 tiers × 2 models × 2 variants; Arm B
   has 14 λ × 4 contrasts × 2 τ. All intervals are pointwise. Designate a
   primary estimand or state that no simultaneous claim is made.
10. **Status of the confirmatory module edit.** `run_replicate` now has two
    optional hooks. They are inert when unset (verified), but the change
    should be recorded wherever the confirmatory code is cited by commit.

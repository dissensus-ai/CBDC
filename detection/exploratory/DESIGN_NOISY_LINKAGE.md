# Noisy-linkage arm: design

**Tier: exploratory scaffolding.** The code in this directory is built and tested.
No result exists. The only outputs so far are DEV-seed smoke runs
(`_dev_smoke_linkage/`, seeds 700501–700505, n = 2,000), and they are not
reportable. A protocol addendum has to freeze every choice listed in §12 before a
reported run. Until then this arm is sensitivity analysis under protocol v3 §7 and
erratum E5, not a confirmatory family member (v3 §5.3: "T2-resolved grids =
exploratory").

Files:

| file | role |
|---|---|
| `noisy_linkage.py` | corruption operator, cluster table (labels + attribute attachment), observed-cluster features, the two scoring rules |
| `run_noisy_linkage.py` | seed guard, grid, one replicate over all cells, t-interval summary, driver |
| `test_noisy_linkage.py` | 39 tests (see §11) |
| `_dev_smoke_linkage/` | DEV smoke output, not reportable |

## 1. Question and estimands

The confirmatory pipeline treats wallet→entity linkage as an oracle: T2 aggregates
over each entity's true wallet set. A deployed resolver makes errors. The question
this arm answers:

> As linkage degrades, does the identity increment grow? That is, does access to
> identity attributes (T3) and the watchlist bit (T4) substitute for broken
> linkage?

All quantities are missed illicit **true** entities per 10,000 true entities at a
review budget of k\* = 500 per 10,000, on independently generated train and test
worlds, for `gboost` (L3) and `logit` (L0), under each scoring rule r ∈
{coverage, conservative}. For cell ε = (ε_s, ε_m) and replicate j:

- `missed_T{2,3,4}(ε)`: absolute misses per tier.
- `delta_T2_minus_T4(ε) = missed_T2(ε) − missed_T4(ε)`. Positive means identity
  and watchlist reduce misses (the same sign convention as `confirmatory/endpoint.py`).
  `delta_T2_minus_T3(ε)` in the same way.
- **Key contrast**, paired within replicate:
  `growth_T2_minus_T4_vs_oracle(ε) = delta_T2_minus_T4(ε) − delta_T2_minus_T4(0,0)`.
  Positive means the identity increment is larger under corrupted linkage than
  under the oracle, which is identity substituting for linkage. `growth_T2_minus_T3_vs_oracle`
  is the attribute-only version.
- `T2_degradation_vs_oracle(ε) = missed_T2(ε) − missed_T2(0,0)`: what the linkage
  errors cost the pseudonymous tier.

Each quantity is summarised by the mean over successful replicates with a two-sided
90% Student-t interval (`inference.mean_ci`, alpha = 0.10), the interval the
confirmatory pipeline uses. Paired differences against the oracle cell use the same
worlds and the same corruption streams (§3.4), so between-world variation cancels.

## 2. Replicate structure

Each replicate uses the confirmatory seed scheme: train seed s, test seed
s + 10,000,003, both passed to `replicate._world` (default DGP config, `mid/mid`,
5% prevalence). The oracle features are built once per world. Then, for each cell:

1. corrupt the **train** and **test** linkage independently, each at that cell's
   (ε_s, ε_m), using a stream derived from that world's own seed;
2. build T2/T3/T4 features on the observed clusters of both worlds;
3. fit each model × tier on the training clusters with the training-cluster labels.
   Model seed = train seed, unchanged (`model_seed`; the guard ensures no wrap);
4. score the test clusters, then take the top k clusters under both rules.

The training world is corrupted too. A supervisor learns from case outcomes on the
clusters its own resolver produced. Training on oracle entities and testing on noisy
clusters would measure distribution shift, not operation under a noisy resolver. The
alternative (oracle train, noisy test) is a one-flag change if the addendum wants
it as a secondary contrast.

The degeneracy audit runs once per replicate on the **oracle** training world,
exactly as in `confirmatory/replicate.py`. The audit is a check on the generator.
Auditing the corrupted features would gate on the resolver. Failures (`FAIL_AUDIT`,
`FAIL_LABEL`, `FAIL_NUM`) are recorded and counted. They are never redrawn.

At (0, 0) the arm reproduces the confirmatory replicate exactly. The test
`test_oracle_cell_matches_confirmatory_replicate` asserts that for both models, all
three tiers and both rules, `MissedPer10k` equals `run_replicate`'s value.

## 3. Corruption operator (`corrupt_linkage`)

The operator acts on the wallet table only and never reads labels. Splits happen
first, on the true partition. Merges happen second, on the post-split partition.

### 3.1 Splits

- **`wallet` (default).** Each wallet is detached from its entity with probability
  ε_s, independently of other wallets, and becomes a singleton cluster. A one-wallet
  entity is unaffected. This is the common-input-heuristic failure: a wallet that
  never co-spends with its siblings stays unclustered. An entity with n_w ≥ 2
  wallets ends up split with probability 1 − (1 − ε_s)^{n_w}. Under the DGP's
  n_w = 1 + Bin(5, 0.3), the **entity-level** split rate is therefore well above
  ε_s. That is why realised rates are recorded (§3.3) and why tables must be
  indexed by realised rates as well as ε.
- **`bipartition`** (protocol v2 §6, as written there). Each multi-wallet entity,
  with probability ε_s, is cut into exactly two non-empty fragments: a random
  permutation of its wallets is cut at a uniformly random point.

**Deviation to resolve.** The team-lead brief specifies per-wallet detachment, which
is the default here. Protocol v2 §6 specifies per-entity bipartition. Both are
implemented. The addendum must choose which one is primary.

### 3.2 Merges

Every post-split cluster, with probability ε_m and independently of the others,
becomes an *initiator*. Each initiator is merged with one *partner* drawn from the
post-split partition, excluding itself. Merges compose transitively through
union-find, so an observed cluster can hold three or more true entities, and a merge
can rejoin two fragments of the same entity. The expected number of merge events is
ε_m × (post-split cluster count). That matches v2 §6's "expected extra merges ≈ m ·
n_entities" when there are no splits.

- **`uniform` (default).** The partner is uniform over the other clusters.
- **`counterparty`.** The partner is drawn with probability proportional to the
  number of external counterparties it shares with the initiator. An initiator that
  shares none falls back to uniform. The DGP has no direct entity-to-entity transfers
  (every flow runs to or from a shared pool of 3,000 external wallets), so a shared
  counterparty is the only graph proximity available. That is the closest analogue
  here to the observation in the common-input-heuristic literature that clustering
  errors are not uniform.

**Mechanism caveat (DEV diagnostic, not a result).** In one DEV world (seed 700505,
n = 2,000, ε_m = 0.2), the counterparty variant put illicit and legitimate entities
into merged clusters at almost the same rate: P(merged | illicit) = 0.357 and
P(merged | legit) = 0.360. Uniform merging gave 0.357 and 0.356. In this DGP, shared
external counterparties are driven by activity volume, not by class. The variant is
therefore a *different merge geometry*, not a stress test of label-correlated
resolver errors. If the addendum wants that stress test (for example launderers'
wallets preferentially merged with high-activity businesses), it needs a new mode.
It must not be read into this one.

### 3.3 Realised error rates

Every corruption records its realised rates for both worlds (`linkage_train`,
`linkage_test` in each cell record): `n_clusters`, `n_clusters_postsplit`,
`n_merge_initiators`, `frac_entities_split` (true entities spread over more than one
cluster), `frac_entities_in_merged_cluster` (true entities sharing a cluster with
another entity), `max_entities_per_cluster` and `mean_clusters_per_entity`. The
summary carries the test-world split and merge rates per cell. Report results against
the realised rates, because ε is a per-unit hazard, not an entity-level error rate.

### 3.4 Determinism and common random numbers

The corruption stream for a world is `SeedSequence([world_seed, 0x4C494E4B])`,
split into independent child streams for splits and merges. The stream is distinct
from the DGP's own stream for the same seed, and it is independent of the grid. Every
unit draws its uniforms whether or not it crosses the threshold. So, for a fixed
world:

- the set of split wallets is nested in ε_s (tested);
- for fixed ε_s, the initiator set is nested in ε_m, and each initiator keeps its
  partner variate.

Grid cells therefore differ only in which units cross the threshold. That is what
makes the paired `growth` contrasts precise. For a given seed, ε, split mode and
merge mode, the operator is deterministic (tested). At ε = 0 it returns exactly the
true partition, relabelled `C<i>` = `E<i>` (tested).

## 4. Features on observed clusters

`observed_entity_features` passes the observed partition under the column name
`entity_id` to the unchanged `features.build_entity_features`. Every T2 aggregate
(wallet count, internal flow, cross-wallet chain pass-through and the rest) thereby
becomes an aggregate over the observed cluster.

- A merge makes transfers between wallets of different true entities look
  *internal*. In this DGP there are none, but merged clusters still pool external
  flows, counterparties and wallet-level statistics.
- A split makes a true self-transfer look *external*, which removes it from
  `internal_vol` and `chain_fast_value`. That is the mechanism by which splits
  destroy the layering signal.

Per-wallet (T1) features do not depend on linkage and are reused as they are. At
ε = 0 the observed feature matrix equals the oracle matrix exactly, for every
attribute rule and mode (`test_eps_zero_is_oracle`, `check_exact=True`).

## 5. Identity attributes on a merged cluster

Attributes live on true entities. A cluster that spans several entities needs a rule
for what it carries. Every rule is **semi-oracle attachment**. No identity is
"discovered", and v2 §6 and v3 §7.1 require that this be said.

The institutional story matters here. If the PSP issues credentials per customer,
then every wallet carries its customer's credential, and credential-keyed linkage
would have no errors at all. Noisy linkage only arises when clusters come from a
heuristic resolver, *not* from credentials. In that case the scoring service holds a
cluster and has to look up attributes wallet by wallet, and a merged cluster maps to
several customer records. Three rules are implemented:

| rule | definition | reading |
|---|---|---|
| `max_risk` (**default**) | riskiest member value per attribute: min `kyc_tier`, min `account_age_days`, max `prior_sar_count`, max `jurisdiction_risk`, `on_watchlist` = any | v3 §7.1's freeze candidate. The service escalates on any linked customer's risk flags. This is what a risk-averse lookup does. |
| `majority` | attributes of the true entity contributing the most wallets (ties: lowest entity index) | The cluster is treated as "belonging to" its dominant customer. This was the team-lead brief's suggestion. |
| `first_wallet` | attributes of the entity owning the cluster's lowest-index wallet | The record inherited by a resolver keyed on its first-seen wallet, i.e. the earliest-issued credential. |

`max_risk` is the default because v3 §7.1 already names it as the freeze candidate,
and moving the default away from pre-written protocol text needs an explicit
amendment. **It has a direction, and that direction matters for the key question.**
Under merges, `on_watchlist = any` lights up every cluster that contains a
watchlisted entity. An illicit entity merged into a large cluster keeps its
watchlist signal intact, while its T2 behavioural signal is diluted. That mechanically
favours positive `growth_T2_minus_T4`. `majority` does the opposite: a small illicit
entity merged into a large legitimate one loses its attributes entirely. So the
answer to "does identity substitute for broken linkage?" may depend on this rule.
The addendum must either freeze one rule with this caveat attached, or pre-specify
both `max_risk` and `majority` as a bracketing pair.

## 6. Training labels

A cluster is labelled illicit iff **any** member wallet belongs to an illicit true
entity. That is erratum E5's frozen choice (any-member), and it is what a supervisor
learning from past cases on resolved clusters would see. Alternatives are majority
of wallets and the dominant entity's label. They were not implemented because E5
already froze any-member. Training-cluster prevalence per cell is recorded
(`train_cluster_prevalence`). Merges raise it; splits raise the count of positive
fragments.

## 7. Budget unit

The budget is **k = round(k\* × n_true / 10,000) cluster alerts**, the same k in
every cell of a replicate. This follows v3 §7.3 and §7.5.

The justification is that review capacity is analyst time, fixed by the institution,
and the PSP knows its customer count. A budget per observed cluster would grow as
splits multiply clusters and shrink as merges consolidate them. Δ would then mix
linkage quality with budget changes, and the paired oracle contrast would no longer
hold capacity constant. The alternative is kept only as a documented option for the
addendum (v3 §7.4 also lists "charge per newly covered entity", which is more
lenient and "must not be swapped post hoc").

If merges ever cut the cluster count below k, the budget is capped at the cluster
count, and `k_eff` is recorded with each score. At the default grid and k\* = 500
this never binds: the DEV smoke's smallest test partition was 1,396 clusters for
n = 2,000, against k = 100.

## 8. Scoring rules (per TRUE entity)

Both rules review the k highest-scoring clusters. Ties go to the lower cluster index,
using `endpoint.select_alerts`. At ε = 0 cluster index = entity index, so both rules
equal `endpoint.missed_per_10k` exactly (tested). `MissedPer10k = 10,000 × (N_pos −
TP) / n_true`.

**(a) Entity coverage** (`score_coverage`; E5 "optimistic", v3 §7.3). A true entity
counts as detected if the reviewed clusters hold at least one of its wallets, and at
least `min_frac` of them (default 0: any wallet is enough; set with
`--coverage-min-frac`). One alert can detect several entities when a merge has
happened. An entity split across reviewed fragments counts once. E5 requires this
rule to be labelled as optimistic under merges and never reported without (b).

**(b) Conservative** (`score_conservative`). The rule walks the reviewed clusters in
rank order. Each cluster credits **at most one** true illicit entity: the illicit
member that has not yet been credited and contributes the most wallets to that
cluster (ties go to the lower entity index). A cluster whose illicit members have all
been credited already credits nothing, but it has still used budget. The second
reviewed fragment of a split entity is the typical case.

- This extends E5's literal rule. E5 counts "a selected cluster ... as at most one
  TP if any member is illicit". Read literally, two reviewed fragments of one entity
  would score two TPs for one entity. That is not conservative, so fragments are
  deduplicated here, as the brief specifies. The addendum should record this as a
  clarification of E5.
- The rule is greedy in review order, so TP_conservative ≤ the maximum bipartite
  matching of reviewed clusters to illicit entities ≤ TP_coverage (the second
  inequality is tested). The operational reading is that an analyst reviewing a
  cluster files on its dominant new subject. Matching would be the generous version
  of "one per cluster". It is not implemented, and the addendum should say whether
  it wants that as a third column.
- `min_frac` does not apply to (b). Any reviewed fragment can credit its entity.

Hand-built tests pin these cases: a merged cluster holding two illicit entities
(coverage 2, conservative 1); fragments reviewed twice (counted once, with budget
wasted); order dependence of the dedup; `min_frac`; and the budget cap.

## 9. Seeds and guard

`run_noisy_linkage.py` requires `--seed-base` and `--R`, and checks every seed before
generating anything (`check_seeds`):

- Train seeds `seed_base … seed_base+R−1` must lie in DEV 700001–700999, unless
  `--addendum-lock <file>` names a file that exists and contains the seed base as a
  token.
- No train or test seed may fall in a spent range, even under a lock. The spent
  ranges are: pilot 900001–900020; development seed 20260707; confirmatory, ladder
  and September-diagnostic blocks 2026080501, 2026081951, 2026082051 and 2026091201
  (each + 0…999); and the test stream (+10,000,003) of every one of these.
- Every seed must be ≤ 2^32 − 1 with `model_seed(s) == s`, so the sklearn
  `random_state` is the protocol seed itself and never a wrapped value.

A refused run exits 2 before creating the output directory (tested). The driver also
refuses to write into an output directory that already contains `records.jsonl`.
Smoke seeds are 700501–700505 (upper DEV half). The lower half, 700001–700500, is
reserved for the concurrent two-stage arm.

## 10. Output schema

`--out-dir` receives:

- `config.json`: every argument, the grid, `dev_seeds`, and the addendum-lock path.
- `records.jsonl`: one line per replicate, written as it completes. Fields:
  `replicate_id, train_seed, test_seed, status, [error, traceback], audit{gate1_pass,
  gate2_pass, worst_feature, worst_auc}, N_positive_test, k, seconds, cells[]`. Each
  cell holds `eps_split, eps_merge, linkage_train{…}, linkage_test{…},
  train_cluster_prevalence, seconds, models{gboost|logit: {T2|T3|T4: {coverage|conservative:
  {TP, MissedPer10k, k_eff}}, delta_T2_minus_T4_{rule}, delta_T2_minus_T3_{rule}}}`.
- `summary.csv`: long format, one row per `(eps_split, eps_merge, model, rule,
  quantity)`, with columns `mean, sd, R_ok, ci_lo, ci_hi`. The quantities are
  `missed_T2, missed_T3, missed_T4, delta_T2_minus_T4, delta_T2_minus_T3,
  growth_T2_minus_T4_vs_oracle, growth_T2_minus_T3_vs_oracle,
  T2_degradation_vs_oracle, linkage_test_frac_entities_split,
  linkage_test_frac_entities_in_merged_cluster`. The key contrast is the two
  `growth_*` rows.
- `timing.json`: wall time, workers, seconds per replicate, mean seconds per cell.

## 11. Tests

```bash
cd detection/exploratory && python3 -m pytest test_noisy_linkage.py   # 39 tests
```

Coverage:

- ε = 0 identity: partition, feature matrix and labels, across 2 split modes × 2
  merge modes × 3 attribute rules.
- Scoring at ε = 0 equals `endpoint.missed_per_10k`, ties included.
- The oracle cell equals `confirmatory/replicate.run_replicate` exactly.
- Determinism.
- Nesting in ε.
- Errors occur and grow with ε. Split-only never merges; merge-only never splits.
- Bipartition makes exactly two fragments.
- Cluster labels and all three attribute rules on a hand-built merge.
- Both scoring rules on hand-built partitions.
- conservative ≤ coverage.
- Budget cap.
- The seed guard: DEV accepted; below and above DEV, pilot, confirmatory and
  20260707 refused; the lock accepted, a wrong base and a missing file refused;
  spent seeds refused even with a lock; the test stream checked; the sklearn ceiling
  enforced; a refused driver run creates nothing.
- Grid construction.

The existing suite still passes: `confirmatory/` 26 tests, `test_pipeline.py` 2
tests.

## 12. Choices the protocol addendum must freeze

1. **Split mechanism**: `wallet` (brief) or `bipartition` (v2 §6) as primary. Also
   whether the other is reported as secondary.
2. **Merge mechanism**: `uniform` as primary. Whether `counterparty` is reported,
   given that it is not class-selective in this DGP (§3.2). Whether a
   label-correlated merge mode should be built and pre-specified.
3. **Order of operations**: splits before merges, merges drawn from the post-split
   partition, transitive composition.
4. **Grid**: ε values and shape. The default is `axes`, which gives 13 cells: the
   oracle, the split axis, the merge axis, and the diagonal over {0.05, 0.10, 0.20,
   0.30}. The alternative is the `full` 5×5 factorial. v2 §6 named {0.05, 0.15}²
   plus the oracle, so the choice must be reconciled with that.
5. **Attribute attachment rule**: `max_risk` alone (v3 §7.1), or `max_risk` and
   `majority` as a pre-specified bracket (§5 explains why this can decide the
   answer).
6. **Training label**: any-member (E5; already frozen there, restate).
7. **Train-world corruption**: corrupted at the same ε (default), or oracle-train as
   a secondary contrast.
8. **Budget unit**: k cluster alerts per 10,000 *true* entities (v3 §7.3). Cap
   behaviour if C < k.
9. **Coverage rule threshold** `min_frac` (default 0 = any wallet).
10. **Conservative rule**: greedy in rank order, dominant *new* illicit member,
    fragment dedup (a clarification of E5's literal text). Whether maximum matching
    is added as a column.
11. **Estimand hierarchy**: which of `growth_T2_minus_T4_vs_oracle` /
    `growth_T2_minus_T3_vs_oracle`, for which model, rule and cell, is primary for
    this arm. Whether any directional claim is made at all, or all cells stay
    descriptive with pointwise 90% intervals and no multiplicity correction (v3
    §5.3 permits only exploratory status without an amendment and Holm across an
    enlarged family).
12. **Seeds**: the seed base, which must be outside every spent range and recorded
    in the lock file passed as `--addendum-lock`. Also R, n_train and n_test (the
    lock's D2 = 8,000 and D1 = 10,000 are the defaults here), k\* = 500, and the
    test offset of 10,000,003.
13. **Failure handling**: the failure threshold (v3 §6.3, 10%). Failures are
    counted, never redrawn.
14. **Environment**: the pinned `requirements.txt` (sklearn 1.8.0 etc.; HistGBM is
    not bit-stable across versions).

## 13. Commands for a reported run

After the addendum is written and records the seed base, say `S`:

```bash
cd detection/exploratory
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  ../../.venv/bin/python run_noisy_linkage.py \
    --seed-base S --R 52 --addendum-lock <path/to/addendum-lock> \
    --n-train 8000 --n-test 10000 --k-star 500 \
    --eps 0 0.05 0.10 0.20 0.30 --grid axes \
    --split-mode wallet --merge-mode uniform --attr-rule max_risk \
    --coverage-min-frac 0 --workers 8 \
    --out-dir results_noisy_linkage/<addendum-id>
```

Repeat with `--attr-rule majority` into a separate `--out-dir` if §12.5 brackets.
The same seed base is allowed there because it runs the same worlds under a different
rule. Always use a fresh output directory. The driver refuses to overwrite one.

## 14. Timing and full-run estimate

Measured with the pinned environment (Python 3.14.0, sklearn 1.8.0), one thread per
worker, on PurrPower (Ryzen 9 9900X, 24 threads):

| run | per replicate (13 cells) | per cell |
|---|---|---|
| DEV smoke, n_train = n_test = 2,000, R = 3, 3 workers | 21.0–21.3 s | 1.3 s |
| timing probe, n_train = 8,000, n_test = 10,000, R = 2, 2 workers (DEV 700990–700991; outputs discarded) | 93.1–93.2 s | ≈ 7.2 s |
| same size, `counterparty` merges, R = 1 (DEV 700992; discarded) | 95.5 s | ≈ 7.3 s |

Scaling from the smoke alone (≈ 4.5× for 4.5× the entities) predicts ≈ 95 s per
replicate, which agrees with the probe.

**Full reported run, R = 52, n_train = 8,000, n_test = 10,000, 13-cell grid, 8
workers:** 7 waves × ≈ 95 s ≈ **11–12 min** of wall time (about 80 CPU-minutes). The
`full` 25-cell factorial is ≈ 180 s per replicate, or ≈ 21–23 min. At n_train =
10,000, add about 10%. Each attribute-rule bracket doubles the total. Peak memory
measured at ≈ 950 MB RSS for one full-size replicate (DEV 700993, discarded), so
8 workers need ≈ 8 GB.

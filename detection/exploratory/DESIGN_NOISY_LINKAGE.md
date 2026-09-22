# Noisy-linkage arm (E11): design

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
| `test_noisy_linkage.py` | 49 tests (see §11) |
| `_dev_smoke_linkage/` | DEV smoke output, not reportable |

## 1. Scope, question and estimands

**Scope: semi-oracle attachment.** Identity attributes are attached through TRUE
wallet ownership, even when the behavioural clusters are corrupted. E11 therefore
tests noisy *aggregation* with stipulated access to customer attributes. It does not
test identity discovery or linkage repair: no rule here recovers a customer record
from a corrupted cluster, and no result from this arm may be described as if the
detector found or repaired identities.

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
- **Absolute deterioration, every tier**, paired within replicate:
  `T{2,3,4}_degradation_vs_oracle(ε) = missed_T{k}(ε) − missed_T{k}(0,0)`. These
  sit next to G in the primary summary block (`summary_primary.csv`, columns
  `G_T4, G_T3, deg_T2, deg_T3, deg_T4`), not in an appendix. G is a difference of
  differences, so G = 0 can coexist with T2 and T4 both getting much worse; the
  degradation curves are what show that.

**Interpretation.** A G contrast compatible with zero provides no clear evidence
that linkage corruption changes the incremental value of the auxiliary information
under this design; it says nothing about the stability of detection performance,
which the absolute degradation curves show. A positive G means the auxiliary block
is worth more under corrupted linkage than under the oracle; it does not by itself
say the auxiliary tiers are robust, since both can deteriorate.

Each quantity is summarised by the mean over successful replicates with a two-sided
90% Student-t interval (`inference.mean_ci`, alpha = 0.10), the interval the
confirmatory pipeline uses. Paired differences against the oracle cell use the same
worlds and the same corruption streams (§3.4), so between-world variation cancels.
Every interval is reported with its achieved half-width. R = 52 matches the
confirmatory replicate count, not its precision; achieved interval widths are
reported.

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
alternative (oracle train, noisy test) is not built; it would be a small driver
option if the addendum wants it as a secondary contrast.

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
several customer records. Four rules are implemented:

| rule | definition | reading |
|---|---|---|
| `attrwise_max_risk` (**default**; deprecated alias `max_risk`) | riskiest value *per attribute, taken separately*: min `kyc_tier`, min `account_age_days`, max `prior_sar_count`, max `jurisdiction_risk`, `on_watchlist` = any | v3 §7.1's freeze candidate. The service escalates on any linked customer's risk flags. Because each attribute takes its own extreme, a merged cluster can carry a profile **no single member has** (e.g. one member's watchlist bit with another's SAR count). |
| `riskiest_member` | the whole attribute vector of the ONE member ranking first under a transparent lexicographic score: `on_watchlist` (1 first), then `prior_sar_count` (higher first), `kyc_tier` (lower first), `jurisdiction_risk` (higher first), `account_age_days` (younger first), finally the lower entity-table row | Escalate on the single riskiest linked customer, but keep that customer's record intact. Always equals some real member's row (tested). Available; not in the pre-specified bracket unless the addendum adds it. |
| `majority` | attributes of the true entity contributing the most wallets (ties: lowest entity index) | The cluster is treated as "belonging to" its dominant customer. This was the team-lead brief's suggestion. |
| `first_wallet` | attributes of the entity owning the cluster's lowest-index wallet | The record inherited by a resolver keyed on its first-seen wallet, i.e. the earliest-issued credential. |

`attrwise_max_risk` is the default because v3 §7.1 already names it as the freeze candidate,
and moving the default away from pre-written protocol text needs an explicit
amendment. **It has a direction, and that direction matters for the key question.**
Under merges, `on_watchlist = any` lights up every cluster that contains a
watchlisted entity. An illicit entity merged into a large cluster keeps its
watchlist signal intact, while its T2 behavioural signal is diluted. That mechanically
favours positive `growth_T2_minus_T4`. `majority` does the opposite: a small illicit
entity merged into a large legitimate one loses its attributes entirely. So the
answer to "does identity substitute for broken linkage?" may depend on this rule.
The addendum must either freeze one rule with this caveat attached, or pre-specify
both `attrwise_max_risk` and `majority` as a bracketing pair. The driver now runs
several rules in one invocation (`--attr-rules`), on the SAME corrupted partitions,
and its default is that bracket. On a split-only cell there are no merged clusters,
so every rule gives identical numbers there by construction.

Toy case (test `test_attrwise_and_riskiest_member_differ_on_merge`): E0 is
watchlisted with a clean record (KYC 2, age 900, 0 SARs, jurisdiction 0.2); E1 is
unlisted with KYC 0, age 50, 5 SARs, jurisdiction 0.8; both are merged into one
cluster. `attrwise_max_risk` gives (KYC 0, age 50, SARs 5, jur 0.8, watchlist 1),
a profile neither has; `riskiest_member` gives E0's row, because the watchlist bit
ranks first; `majority` gives E1's row (two wallets to one).

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

- `run_spec.json`: the single run-spec dict, i.e. every field an addendum lock
  pins: `arm` ("E11"), `seed_base, R, n_train, n_test, k_star, test_offset,
  grid_name, eps, grid` (list of `[split_mode, eps_split, eps_merge]`),
  `split_mode, merge_mode, attr_rules, models, scoring_rules, coverage_min_frac,
  alpha, dev_seeds, addendum_lock, requirements_sha256` (of
  `detection/requirements.txt`), `commit`, `commit_dirty` (true if `detection/` had
  uncommitted changes, in which case `commit` does not identify the code). The key names follow the E9–E11 lock
  schema on `exp/two-stage-signal` (`addendum_lock.py`: `arm, seed_base, R, grid,
  models`), so that lock can be adopted after the branches merge; note that `grid`
  here is a list of cells, not a list of floats, and the lock's exact-match must
  take that into account. The seed guard in this driver stays in place until then.
- `records.jsonl`: one line per replicate, written as it completes. Fields:
  `replicate_id, train_seed, test_seed, status, [error, traceback], audit{gate1_pass,
  gate2_pass, worst_feature, worst_auc}, N_positive_test, k, seconds, cells[]`. Each
  cell holds `split_mode, eps_split, eps_merge, linkage_train{…}, linkage_test{…},
  seconds, attr{<attr_rule>: {train_cluster_prevalence, models{gboost|logit:
  {T2|T3|T4: {coverage|conservative: {TP, MissedPer10k, k_eff}},
  delta_T2_minus_T4_{rule}, delta_T2_minus_T3_{rule}}}}}`.
- **`summary_primary.csv` (the primary block)**: one row per `(split_mode,
  eps_split, eps_merge, attr_rule, model, rule)` with `R_ok` and, for each of
  `G_T4` (= growth_T2_minus_T4_vs_oracle), `G_T3`, `deg_T2`, `deg_T3`, `deg_T4`
  (= T{k}_degradation_vs_oracle), the columns `_mean, _ci_lo, _ci_hi,
  _half_width` (paired 90% t-intervals), plus `realised_frac_entities_split` and
  `realised_frac_entities_in_merged_cluster` (test world, mean over replicates).
  G is never shown without the three deterioration curves beside it. The driver
  also prints this block for gboost/conservative at the end of a run.
- `summary.csv`: long format, one row per `(split_mode, eps_split, eps_merge,
  attr_rule, model, rule, quantity)`, with columns `mean, sd, R_ok, ci_lo, ci_hi,
  half_width`. Quantities: `missed_T2, missed_T3, missed_T4, delta_T2_minus_T4,
  delta_T2_minus_T3, growth_T2_minus_T4_vs_oracle, growth_T2_minus_T3_vs_oracle,
  T2_degradation_vs_oracle, T3_degradation_vs_oracle, T4_degradation_vs_oracle,
  linkage_test_frac_entities_split, linkage_test_frac_entities_in_merged_cluster`.
- `timing.json`: wall time, workers, seconds per replicate, mean seconds per cell.

## 11. Tests

```bash
cd detection/exploratory && python3 -m pytest test_noisy_linkage.py   # 49 tests
```

Coverage:

- ε = 0 identity: partition, feature matrix and labels, across 2 split modes × 2
  merge modes × 4 attribute rules.
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
- Grid construction, including `axes+v2` (17 cells, one shared oracle).
- Attribute rules: `attrwise_max_risk` and `riskiest_member` differ on a toy merge;
  `riskiest_member` always equals a real member's row (property check over
  corrupted worlds); its tiebreak order; the `max_risk` alias.
- The oracle cell matches the confirmatory replicate under every attribute rule.
- End to end: the driver writes `run_spec.json` with every lock field, and the
  primary block with G and all three degradation curves (zero at the oracle).

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
4. **Grid**: proposed `axes+v2` (17 cells): the 13-cell per-wallet `axes` grid
   (oracle, split axis, merge axis, diagonal over {0.05, 0.10, 0.20, 0.30}) plus the
   carried-over v2 §6 subset (bipartition splits × uniform merges at
   {0.05, 0.15}²), sharing one oracle. The `full` 5×5 factorial stays available.
5. **Attribute attachment rule**: proposed bracket `attrwise_max_risk` +
   `majority` (§5 explains why this can decide the answer). `riskiest_member` is
   implemented but outside the bracket unless the addendum adds it.
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
    --eps 0 0.05 0.10 0.20 0.30 --grid axes+v2 \
    --split-mode wallet --merge-mode uniform \
    --attr-rules attrwise_max_risk majority --models gboost logit \
    --coverage-min-frac 0 --workers 8 \
    --out-dir results_noisy_linkage/<addendum-id>
```

One invocation runs both grids (the 13 per-wallet cells and the 5 v2 cells, one
shared oracle) and both bracket rules on the same corrupted partitions. Always use a
fresh output directory; the driver refuses to overwrite one. R = 52 matches the
confirmatory replicate count, not its precision; achieved interval widths are
reported (`*_half_width`).

## 14. Timing and full-run estimate

Measured with the pinned environment (Python 3.14.0, sklearn 1.8.0), one thread per
worker, on PurrPower (Ryzen 9 9900X, 24 threads):

| run | per replicate | per cell |
|---|---|---|
| DEV smoke, n_train = n_test = 2,000, R = 3, 3 workers | 21.0–21.3 s | 1.3 s |
| timing probe, n_train = 8,000, n_test = 10,000, R = 2, 2 workers (DEV 700990–700991; outputs discarded) | 93.1–93.2 s | ≈ 7.2 s |
| same size, `counterparty` merges, R = 1 (DEV 700992; discarded) | 95.5 s | ≈ 7.3 s |
| **proposed reported configuration**: same size, `axes+v2` (17 cells) × 2 attribute rules, R = 2, 2 workers (DEV 700990–700991; discarded; after the pre-freeze fixes) | 206.1–206.2 s | ≈ 10.9 s (both rules) |

The first three rows predate the pre-freeze fixes and ran one attribute rule on the
13-cell grid.

Scaling from the smoke alone (≈ 4.5× for 4.5× the entities) predicts ≈ 95 s per
replicate, which agrees with the probe.

**Full reported run as proposed in §13** (R = 52, n_train = 8,000, n_test = 10,000,
`axes+v2`, attribute bracket of two rules, 8 workers): 7 waves × ≈ 210 s ≈
**25 min** of wall time (about 180 CPU-minutes). One rule on the 13-cell grid alone
is ≈ 95 s per replicate, ≈ 11–12 min. The `full` 25-cell factorial with two rules
would be ≈ 360 s per replicate, ≈ 42 min. At n_train = 10,000, add about 10%. Peak memory
measured at ≈ 950 MB RSS for one full-size replicate (DEV 700993, discarded), so
8 workers need ≈ 8 GB.

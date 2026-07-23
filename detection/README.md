# CBDC Privacy — Honest Detection Pipeline

Replaces the manuscript's §4.5 "validation" (the AUC = 1.000 definitional
artifact) with a runnable, gated, reproducible detection experiment.
Built 2026-07-07 against `DESIGN_PACK_JUL2026.md`; fixes panel-review
blockers T0-1 (no code behind the empirical claims / false
Data-Availability statement) and T0-2 (identity-aware AUC = 1.000 was a
DGP artifact: launderers were *defined* to hold 3–6 wallets vs 1–2, so
`num_wallets` separated the classes by construction).

## What the old paper got wrong, structurally

The old toy DGP encoded the label into a single marginal feature. Any
classifier that saw `num_wallets` hit AUC 1.000, and "watchlist adds
zero marginal value (1.00 → 1.00)" was arithmetically forced. Here:

* **Wallet counts come from one shared distribution for both classes**
  (`dgp.py`, the `wallet_counts` line). Laundering is a latent
  multi-step behavior — placement → layering hops across own wallets →
  structured integration — never a single separable marginal.
* **A pre-registered degeneracy audit** (`degeneracy_audit.py`) runs
  before any headline number and hard-fails (exit ≠ 0) if ANY single
  feature reaches entity-level AUC > 0.95, or if any marginal is
  class-disjoint. Run it on the old DGP and it fails on `num_wallets`
  at 1.000; run it here and the worst feature sits ≈ 0.84–0.92 across
  seeds.

## Pipeline

| file | role |
|---|---|
| `dgp.py` | seeded synthetic CBDC transaction generator; `default_config()` + `surveillance_strong_config()` (the falsification world) |
| `features.py` | wallet-level (T1) and entity-level (T2–T4) features; tier column sets |
| `degeneracy_audit.py` | the gate: single-feature AUC cap 0.95 + marginal-overlap check; raises `DegeneracyError` / exits 1 |
| `detection_experiment.py` | four-tier experiment, entity-disjoint 5-fold CV, entity-clustered bootstrap CIs, label-permutation negative control |
| `equivalence_test.py` | TOST for T2 ≈ T4 (and T2 ≈ T3), paired entity-level bootstrap, margin δ = 0.03 default |
| `data_loaders.py` | the standard data interface + documented AMLworld / Elliptic adapter stubs |
| `run_all.py` | driver; regenerates every number in `results/` |
| `test_pipeline.py` | regression tests: the audit hard-fails on a reconstruction of the original num_wallets artifact, and passes (num_wallets AUC ≈ 0.5) on the shipped DGP |

**Tiers** (design pack §3): T1 structure-only on unlinked pseudonyms →
T2 + pseudonymous linkage (ZK-achievable) → T3 + identity attributes →
T4 + watchlist. The paper's thesis is the equivalence T2 ≈ T4.

**Inference honesty:** folds split on entities (never wallets/tx), the
effective N reported is the entity count, and every CI comes from an
entity-clustered bootstrap. T1's classifier scores wallets without
linkage; entity scores are max-aggregated only at evaluation.

**Model note:** design called for RandomForest/XGBoost/GNN. This build
uses `LogisticRegression` + `HistGradientBoostingClassifier` (sklearn's
gradient boosting — XGBoost is not installed here; same model family).
A GNN tier is deliberately deferred to the real-data run (PyTorch
Geometric on the ROCm box) — it changes nothing about the harness.

## Reproduce

```bash
cd detection/
python3 run_all.py                     # default: seed 20260707, 800 entities, δ=0.03
python3 run_all.py --seed 42           # any seed; all numbers regenerate
python3 degeneracy_audit.py [seed]     # gate only
```

Environment: Python 3.14, numpy 2.3.5, scipy 1.16.3, scikit-learn 1.8.0,
pandas 2.3.3. Runtime ≈ 30 s. Every random draw flows from the single
`--seed` (DGP, folds, models, bootstraps).

Outputs in `results/`: `results_{world}.csv` (AUC/AP + 95% clustered CIs
per tier per model), `oof_{world}.csv` (per-entity out-of-fold scores —
the TOST input), `degeneracy_audit_{world}.json`,
`tost_{world}_*.json`, `summary.json`.

## Reference results (seed 20260707, 800 entities, 29 launderers)

Default world, gboost, entity-level AUC [95% clustered CI]:
T1 0.936 [0.849, 0.994] → T2 0.988 [0.967, 0.998] → T3 0.992 → T4 0.992.
TOST T2 vs T4 (gboost): ΔAUC +0.0043, 90% CI [+0.0004, +0.0101] →
**EQUIVALENT** at δ = 0.03. The logit model's verdict is INCONCLUSIVE
(ΔAUC +0.041, CI crosses the margin): identity adds a little value that
the linear model cannot recover from behavior alone — reported, not
hidden. Label-permutation control ≈ 0.5. Nothing saturates at 1.000,
and the linkage step (T1 → T2) carries most of the marginal value — the
honest version of the paper's claim, on data where it *could* have come
out otherwise. (At seed 42 all four verdicts are EQUIVALENT.)

That it can come out otherwise is demonstrated, not asserted:
`surveillance_strong` (launderers behaviorally hidden, identity
attributes genuinely informative, every marginal still audit-clean)
yields logit T2 0.816 → T4 0.992, TOST verdict
**SURVEILLANCE_SUPERIOR** (gboost T2 0.925 → T4 0.997, INCONCLUSIVE
leaning superior). `run_all.py` exits non-zero if this world ever comes
back all-EQUIVALENT — a rigged-harness self-check.

## What this does and does not license

These are **synthetic** results demonstrating the harness. They make the
Data-Availability statement true for the *pipeline*, and they show the
old AUC = 1.000 is unreproducible under a non-rigged DGP. They do NOT
yet support the paper's headline claim on realistic laundering typologies
— that requires the real-data runs below. Until those are done, §4.5
must be framed as "illustrative synthetic validation of the harness",
not as evidence.

## Plugging in real data (MF / Maksakov)

1. **AMLworld (primary; Altman et al. 2023, NeurIPS).** Download
   HI-Small from Kaggle (link in `data_loaders.py`). Implement the
   column mapping in `load_amlworld()` — the docstring specifies it
   field by field. Identity/watchlist axes don't exist there: fill
   zeros and report T3/T4 as not-evaluable, or attach synthetic
   identity layers and label them as such.
2. **Elliptic (cross-check; Weber et al. 2019).** Download from Kaggle
   (link in `load_elliptic()`). Entity linkage must be proxied via
   co-spend clustering; use temporal (past→future) splits instead of
   entity folds; drop 'unknown' labels. Only the T1 → T2 step is
   evaluable.
3. Run the same three commands. The degeneracy audit runs on real data
   too — if a real dataset has a >0.95 single-feature separator, you
   want to know before quoting an AUC.
4. Pre-register δ before looking at real-data results (design pack §8
   Q2 — δ = 0.03 is the working default, MF to confirm).
5. The PET-AML latency sim (`../andrew-cbdc/pet_aml_sim.py`) is a
   separate artifact with two open bugs (escalation ordering, rejection
   calibration — design pack §4.5); it validates the performance
   envelope, not detection.

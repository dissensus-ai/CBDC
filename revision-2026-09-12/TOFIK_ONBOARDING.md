# CBDC onboarding for Tofik — draft handoff, not sent

Start with the September PDF, then `ADJUDICATION.md`, then `REPRODUCE.md`. The old root manuscript is an August artifact. The work is now a synthetic measurement paper about the incremental detection value of customer attributes and watchlists above an oracle linked-wallet representation. It does not implement a CBDC or prove privacy guarantees.

The current contribution is the replicated conditional map, the operational missed-case endpoint, and the separation of information blocks. Boosting gains 7.90 fewer missed illicit entities per 10,000 from the combined block; logistic regression gains 31.35. Both gains are positive. Customer attributes alone and watchlists alone are reported separately. Do not reuse the earlier 'identity adds nothing' wording or call H2 confirmatory: a code-level gatekeeping error has been corrected.

## First bounded task

Reproduce the fast numerical tables and a cold PDF build. Deliver a one-page claim-to-output check for the abstract: source filename, JSON key, calculation, displayed rounding, and scientific scope. Compare against the new source, not old `main-v4.tex` or generated corpus copies. Report mismatches directly and keep proposed edits separate from observed data. This is the same evidence discipline useful for the alpha paper.

## Next contribution, after reproduction

Design an oracle-versus-noisy-linkage experiment using the independently generated train/test pipeline. False merges/splits change both features and the alert unit. Pre-specify realistic parameter ranges, a fixed budget, explicit cluster labels, one entity-coverage metric, and a conservative at-most-one-positive-per-cluster metric. Separate pipeline-debug seeds from future estimation seeds. Do not claim oracle linkage is a formal upper bound: merges can make optimistic coverage look better. First deliver the protocol and a smoke experiment, not a manuscript headline.

A second useful sensitivity is tier-specific versus T4-selected tuning, because current shared tuning may disadvantage T2. Retain the existing tuning experiment as historical evidence; label any new sensitivity as retrospective and compute paired contrasts on the same populations. No claim of monotonic model capacity follows from comparing algorithms.

## Reading and code route

1. `manuscript/main.tex`: abstract, Methods, Results, limitations.
2. `detection/features.py`: T2/T3/T4 are 19/23/24 columns; T1 is a different wallet-level representation.
3. `detection/confirmatory/{endpoint,replicate,inference}.py`: independent populations, sign conventions, actual gate.
4. `detection/{ladder_config,ladder_experiment,surface_configs}.py`: model grids, nested tuning, generated regimes.
5. `detection/revision20260912/`: retrospective plan, diagnostics and table regenerator.
6. `evidence/SOURCE_VERIFICATION.md`: only retained references and exact supported claims.

Andrew's existing simulator/software and original writing contribution remains credited. This draft does not certify final author approval or add Tofik as an author before contribution. No recipient email is assumed; this handoff is prepared for Murad to send or authorize sending.

## Bridge to ASRI

Complete the CBDC claim-to-output exercise before ASRI implementation work. The transferable task is keeping target/label definitions, horizon, evaluation unit, native metric, uncertainty, and public claim synchronized. ASRI's current source of truth and live research branch must be read separately; this file does not infer its current status from the CBDC or alpha work.

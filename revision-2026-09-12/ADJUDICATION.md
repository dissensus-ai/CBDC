# CBDC scientific adjudication — 12 September 2026

**Decision:** build one compact synthetic measurement paper around the replicated T2/T3/T4 increments, model comparison, regime map, and operational budget diagnostic. Retain the institutional design as conditional discussion and the PET-AML simulator as a clearly separate illustrative appendix. No claims of cryptographic implementation, anonymity, universal privacy/detection equivalence, field validity, or journal readiness are supported.

This is an independently reviewable research manuscript draft, not a publication or submitted artifact. The September work resolves the scientific wording decisions formerly parked for author judgment. Authorship/declarations and external publication remain ordinary author actions, not numerical decisions silently taken here.

## New defects found by reading the code

1. **False confirmatory H2 label.** `run_confirmatory.py` assigned `gate = NON_INFERIOR` when `D4_delta_star` was null and H1 was `ESTIMATE_ONLY`. `inference.py` and the statistical erratum require H1 to actually establish non-inferiority. The new shared `summarize_hypotheses` implementation passes the real H1 status and respects a descriptive-only H2 lock. H2 is now `DESCRIPTIVE_ONLY`; the estimate 23.442 [21.832,25.052] does not change. Four regression tests cover unset tolerance, established H1, failed H1, and the explicit descriptive switch. No post-hoc margin was selected to rescue the label.
2. **Surface prevalence departure.** Protocol §5 names 3%; the driver and all 80 nondefault records use 5%. Report the executed 5% surface as a disclosed unamended deviation. Do not rewrite the historical protocol or pass it off as a 3% result.
3. **Claimed completion exceeded implementation.** The E2 factorial, default within-replicate bootstrap, and per-world ladder label-permutation diagnostics were absent. The first two are omitted from this narrower paper's contribution and reported as not done. The permutation diagnostics are supplied now as retrospective checks. 122/122 describes the executed world units, not completion of every item in E1–E6.
4. **Oracle linkage is not a formal upper bound.** A false merge may credit multiple entities to one alert under optimistic scoring. The new discussion treats oracle linkage as an idealized reference and requires explicit dual scoring for future noisy-linkage work; no nonexistent sensitivity is claimed.
5. **The model ladder is not a controlled causal intervention on capacity.** Model families, optimization, and regularization change. Additive boosting already exhibits the smaller increment; unconstrained boosting has a slightly larger increment. The paper reports that descriptive pattern, not a monotonic capacity law or identified mechanism.
6. **Surface 'pure recoverability' language overstated the code.** Obfuscation also changes cover activity, and changed attribute sampling can change downstream random draws. The surface is a generated-regime comparison. No causal purity or paired counterfactual interpretation is asserted.

## September review issues adjudicated

| Existing issue | Decision and implementation |
|---|---|
| CIT-1/2 invented legislative quotation/conditional; CIT-12 inaccurate Zatti reading | Remove the US political-legislative paragraph and its unsupported inference. It contributes nothing to the measurement. No replacement statutory assertion is invented. |
| LOG-1/2 identity versus full-surveillance conflation and unreported T2→T3 | Report separate customer-attribute, watchlist, and combined increments in abstract, methods, tables, and conclusion. Names are not model inputs. |
| LOG-3 AP headroom; NUM-5 14/15 flattened to equivalence | Replace the pilot equivalence headline with replicated continuous estimates and operational counts. Explicitly state nonlinear T2 AP is already high. No equivalence or non-inferiority is claimed without a justified margin. |
| LOG-4 mixed simulator configurations | Freshly rerun both complete configurations with --days 2 --seed 7; report 27 versus 592 escalations and 0.38 versus 0.55 modeled MPC hours alongside both rejection counts. Preserve all logs. |
| LOG-5 AUC 'cannot see' a planted effect | Delete this false assertion and use replicated positive-control AP values. Neither AUC nor AP is called universally insensitive. |
| LOG-14 unnamed study/gap claim | Remove the uncited study and universal absence claim. State the concrete contrast with Aurora/Hertha and existing privacy/payment constructions. |
| NUM-4 placement/integration probability conflation | Distinguish placement smurfing 0.60 from integration structuring 0.46 at obfuscation 0.6. |
| NUM-2/3 unarchived n=800 AP universal statement and nonexistent cross-reference | Drop that analysis from this paper. Use archived 20-replicate pilot resolution outputs and the complete 52-replicate result instead. |
| CIT-14 dead Zcash source; CIT-13 benchmark-to-maturity extrapolation | Remove unused deployment-maturity claims and those references. Seven retained references are checked to primary sources; three DOIs also checked via scite. |
| MET-1 title mismatch | Adopt the measurement title for this September candidate. Existing Zenodo metadata is not represented as updated. Root/author controls eventual public retitling. |
| MET-5 unpinned data/code | Cite August baseline commit 304a3fe; accompanying September archive contains exact inputs, scripts, results and hashes. A September GitHub publication is not asserted before the root agent pushes a review branch. |
| MET-11 affiliation | Use Dissensus alone, consistent with the work's independent-research setting. No current university appointment is implied. |
| MET-6 stale README/twin metadata | Put the September draft and corrected inference at the start of this branch README, with explicit historical labels for root main.tex/main.pdf. Preserve old files as evidence. |
| DLT draft's absolute no-bulk/no-unilateral claims | Reject them. Stable clusters enable bulk behavioral monitoring; issuers retain dossiers and can relink. Describe only intended access separation at the scoring service and threshold process; no implementation/security theorem. |
| Notification/abandonment and tipping-off | Exclude the primitive from the candidate; it is neither needed for the measurement nor supported as compliant or probing-resistant. |
| Coauthor pending research decisions | Scientific decisions above are made now. Existing Andrew contribution stays credited; this work does not certify his final endorsement. |

## What the result actually is

At 500 reviews per 10,000, fresh reproduction of all 52 original independent train/test pairs returns exactly identical raw records, zero failures. Adding attributes plus watchlist reduces missed illicit entities by 7.904 [7.132,8.675] for boosting and 31.346 [29.930,32.763] for logistic regression. For boosting, the attribute-only effect is 2.404 [1.881,2.927] and the watchlist effect 5.500 [4.715,6.285]. Exact nonrounded values are in `evidence/adjudicated-results.json`; manuscript display uses two decimals.

Default full-ladder AP gains are 0.0570 (linear), 0.0128 (additive), 0.0125 (forest), and 0.0160 (boosted), each positive. High auxiliary signal helps all models. The data contradict the old 'identity adds nothing' conclusion.

The excluded 20-replicate pilot found k=50 saturated in every replicate; boosting's difference was exactly zero at k=50,100,200. k=500 and interval-width targets were set post-pilot, pre-run. Crucially, interval-width targets are not a policy non-inferiority margin: that margin remains unset.

## Remaining limits, not parked decisions

- Synthetic, one generator family; no real customer dataset, deployment, sanctions compliance assessment, or adaptive evasion study.
- Oracle linkage; no false-merge/split experiment. The paper makes no corresponding performance claim.
- The ladder uses pooled OOF rankings and T4-selected shared hyperparameters; both limitations are disclosed. Independent-population results carry the operational headline.
- Pointwise 90% intervals; no simultaneous surface inference or confirmatory moderation claim. Two-replicate positive and negative control cells are diagnostics, not precise estimates.
- No new cryptography and no system timing validation. Centralized PET/role separation is a legitimate baseline.

These limits define the contribution. They do not justify leaving false scientific claims in the manuscript or inventing study completion. The corresponding next research tasks are concrete in `TOFIK_ONBOARDING.md`.

## Final executed outcome

All 22 retrospective permutation worlds finished with no failed units and no duplicate seeds. The original full-ladder replay matches all serialized scientific fields, including selected configurations. The minimal 15-file arXiv source archive cold-builds independently to identical rendered text. The final draft is 17 pages. Package hashes are in `evidence/package-validation.json`. No arXiv upload or journal submission occurred in this work.

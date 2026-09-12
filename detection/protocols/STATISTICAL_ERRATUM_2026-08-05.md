# Statistical erratum to protocol v3

**Date:** 2026-08-05  
**Applies to:** `01-prospective-protocol-v3.md`  
**Status:** Binding correction for freeze candidate. Prefer this erratum over conflicting v3 wording.  
**Scope:** Four statistical points only. Does not re-open construct validity, train/test design, or DLT selection.

---

## E1. Non-inferiority, not equivalence

### Error in v3

Rules of the form “declare equivalence if \(U \le \delta^\*\)” (v3 primary ops decision for \(\Delta_{\mathrm{miss}}\)) only establish that T2 is **not unacceptably worse** than T4 on missed illicit entities. That is **one-sided non-inferiority** of T2 relative to T4 (T2’s extra misses bounded above by \(\delta^\*\)), not two-sided equivalence.

### Correction

| Term | Definition for this paper |
|------|---------------------------|
| **Primary operational claim** | **Non-inferiority** of T2 to T4 at budget \(k^\*\): the mean extra misses \(\mathbb{E}[\Delta_{\mathrm{miss}}^{\mathrm{T2}-\mathrm{T4}}]\) satisfy \(\mathbb{E}[\Delta] \le \delta^\*\) |
| **Test orientation** | Upper confidence bound only: non-inferior if \(U_{\Delta} \le \delta^\*\) |
| **Surveillance-superior (auxiliary helps beyond tolerance)** | Lower bound \(L_{\Delta} > \delta^\*\) |
| **Inconclusive** | Interval straddles \(\delta^\*\) in the one-sided sense (\(L \le \delta^\* < U\)) |
| **Equivalence (two-sided)** | **Not** the primary claim. Would require the entire CI for \(\mathbb{E}[\Delta]\) inside \([-\delta^\*, +\delta^\*]\). Do not use “equivalent” / “TOST equivalence” language for the primary operational result unless a separate two-sided analysis is pre-specified and labeled exploratory or secondary |

**Sign reminder (unchanged):**  
\(\Delta_{\mathrm{miss}}^{\mathrm{T2}-\mathrm{T4}} = \mathrm{MissedPer10k}(\mathrm{T2}) - \mathrm{MissedPer10k}(\mathrm{T4})\).  
Positive \(\Delta\) means T2 misses **more** than T4 (withholding auxiliary information hurts detection).

**Manuscript language:** prefer “T2 is non-inferior to T4 within \(\delta^\*\) additional missed illicit entities per 10,000 at review budget \(k^\*\)” over “T2 is equivalent to T4.”

---

## E2. Hierarchical testing must actually gate

### Error in v3

Allowing H2 (DiD / model moderation) after H1 receives **any** verdict, including inconclusive, provides **no** multiplicity protection. That is sequential reporting, not gatekeeping.

### Correction — gatekeeping rule (freeze)

| Step | Claim | Proceed only if |
|------|--------|-----------------|
| **H1** | Non-inferiority of L3: \(U_{\Delta^{\mathrm{L3}}} \le \delta^\*\) | Always tested (primary) |
| **H2** | DiD moderation: L0 benefits more from auxiliary block than L3 | **Only if H1 establishes non-inferiority** (\(U_{\Delta^{\mathrm{L3}}} \le \delta^\*\)) |
| **H3** | L0 \(\Delta_{\mathrm{miss}}\) magnitude | Always tabulated as **descriptive**; no confirmatory claim |

**If H1 is inconclusive or surveillance-superior:**

- Report H1 verdict.  
- Report L0 and DiD **descriptively** only.  
- **Do not** claim confirmatory evidence that “model capacity conditions the value of identity-linked auxiliary information.”

**If authors later want H2 regardless of H1:** abandon gatekeeping and use a **joint** multiplicity adjustment (e.g. Holm on {H1, H2} two-parameter family) in a written protocol amendment **before** confirmatory runs — not post hoc.

**Default freeze:** gatekeeping as in the table above (H2 confirmatory only after H1 non-inferiority).

---

## E3. Confidence interval for the replicate mean (not raw percentiles of effects)

### Error in v3

“Percentile interval over replicates” is ambiguous. The empirical percentiles of \(\{\Delta_r\}\) describe **between-world dispersion** of the effect. They are **not** automatically a confidence interval for \(\mu = \mathbb{E}[\Delta_r]\).

### Correction — confirmatory interval for the mean

Let \(R_{\mathrm{ok}}\) successful confirmatory replicates, \(\Delta_1,\ldots,\Delta_{R_{\mathrm{ok}}}\) the within-replicate estimates (L3 \(\Delta_{\mathrm{miss}}\) for H1; \(\mathrm{DiD}_r\) for H2 when gated).

**Primary method (matches pilot precision formula):** Student-\(t\) CI for the mean

\[
\bar\Delta \pm t_{1-\alpha/2,\, R_{\mathrm{ok}}-1} \cdot \frac{s}{\sqrt{R_{\mathrm{ok}}}}
\]

where \(s\) is the sample SD of the replicate estimates. Report \(\bar\Delta\), \(s\), \(R_{\mathrm{ok}}\), and the interval. For **non-inferiority**, the decision uses the **upper** one-sided bound (or the upper endpoint of a two-sided interval at level chosen so the one-sided error is controlled — freeze: use upper endpoint of the two-sided \((1-\alpha)\) interval as \(U\), with \(\alpha\) from the author decision table).

**Allowed alternative (must pick one before pilot lock):** nonparametric bootstrap CI for the mean — resample replicates with replacement \(B\) times (e.g. \(B=5000\)), recompute \(\bar\Delta^*\), take percentile interval of \(\bar\Delta^*\).  
**Not allowed as primary:** reporting the 5th–95th percentiles of the raw \(\Delta_r\) as if they were a CI for \(\mu\).

**Always also report (descriptive):** histogram/box of \(\Delta_r\), median, IQR — as **between-world variation**, clearly labeled.

---

## E4. Pilot sample size must cover both H1 and H2 (when H2 is confirmatory)

### Error in v3

Sizing \(R\) only from \(\mathrm{SD}(\Delta_{\mathrm{miss}}^{\mathrm{L3}})\) under-sizes the study if H2 (DiD) remains confirmatory under gatekeeping.

### Correction

Excluded Monte Carlo pilot estimates:

- \(s_{\Delta} = \widehat{\mathrm{SD}}(\Delta_{\mathrm{miss},r}^{\mathrm{L3}})\)  
- \(s_{\mathrm{DiD}} = \widehat{\mathrm{SD}}(\mathrm{DiD}_r)\)

With precision targets \(\tau_{\Delta}\) and \(\tau_{\mathrm{DiD}}\) (author table):

\[
R_{\Delta} = \left\lceil \left(\frac{z\, s_{\Delta}}{\tau_{\Delta}}\right)^2 \right\rceil, \quad
R_{\mathrm{DiD}} = \left\lceil \left(\frac{z\, s_{\mathrm{DiD}}}{\tau_{\mathrm{DiD}}}\right)^2 \right\rceil, \quad
R = \max(R_{\Delta}, R_{\mathrm{DiD}}, R_{\min})
\]

Use the same \(z\) convention as the planned CI (e.g. \(z_{1-\alpha/2}\) as a pilot approximation to \(t\)).  
If H2 is demoted to descriptive, set \(R = \max(R_{\Delta}, R_{\min})\) only.

Pilot seeds remain **excluded forever** from confirmatory analysis (unchanged).

---

## E5. Noisy-linkage scoring: optimistic vs conservative (sensitivity caveat)

### Issue (not an error of silence — incomplete labeling)

Protocol v3 §7 credits **every** illicit true entity inside a selected merged cluster toward \(\mathrm{TP}\), while charging **one** alert. That is a clear rule, but **optimistic**: false merges can look beneficial by covering multiple illicit entities per alert.

### Correction — dual reporting for T2-resolved

| Estimator | Rule | Label in tables |
|-----------|------|-----------------|
| **Optimistic (entity-coverage)** | v3 §7 as written: one cluster alert adds all member true entities to reviewed set \(U\); \(\mathrm{TP}=\|\{e\in U:Y_e=1\}\|\) | **Oracle entity-coverage upper bound** (optimistic under merges) |
| **Conservative (cluster-level)** | Rank clusters; select \(k\) clusters; a selected cluster counts as **at most one** true positive if **any** member is illicit (or if the cluster’s majority/primary entity is illicit — freeze: **any-member**); splits: each fragment is a separate cluster competing for budget | **Conservative cluster-level** |

**Confirmatory primary H1/H2** remain on **T2-oracle** (one entity = one unit), unchanged.  
**T2-resolved** analyses report **both** optimistic and conservative columns; never only the optimistic merge-credit rule.  
Do not interpret optimistic > T4 gaps as evidence that noisy linkage “helps privacy designs” without the conservative companion.

---

## Summary table (v3 → erratum)

| Topic | v3 | Erratum |
|-------|----|---------|
| Ops claim | “Equivalence” if \(U\le\delta^\*\) | **Non-inferiority** if \(U\le\delta^\*\) |
| H2 gate | After any H1 verdict | **Only if H1 non-inferiority holds** |
| Interval | Ambiguous “percentile over replicates” | **\(t\)-CI for mean** (or bootstrap of means); raw percentiles = dispersion only |
| Pilot \(R\) | Sized for \(\Delta\) only | \(\max(R_{\Delta}, R_{\mathrm{DiD}})\) if H2 confirmatory |
| T2-resolved merges | Single optimistic credit rule | **Optimistic + conservative** dual report |

---

## What this erratum does not change

- Construct: identity-linked **auxiliary** information (T2/T3/T4 labels).  
- Train/test populations; \(n_{\mathrm{test}}=10{,}000\); retrain every model×tier.  
- Realised MissedPer10k formula and \(\Delta\) sign.  
- No silent seed redraw.  
- DLT mechanism pending Andrew.  
- Illustrative status of \(k^\*,\delta^\*\) until author decision table is signed.

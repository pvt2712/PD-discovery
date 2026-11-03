# Step 2 methods: what to apply when comparing phenotypes' causal mechanisms

> Investigation note, 3 November 2026, following the methods and novelty addendum (`methods_and_novelty_update.md`). It answers the investigator's question about the Step 2 comparison methods named there: difference graphs such as DDAG and LDiffPC, and shared-plus-deviation estimators such as multi-VAR and NetANOVA. Which should the study apply, and how? The note rests on two sources. The first is three parallel full-text reviews of about 40 methods papers and their software, covering difference graphs and shift detectors, shared-plus-deviation estimators, and the group-invariance tests used in clinical panel research. The second is a fourth synthetic simulation, which runs each family of methods next to its speed-adjusted counterpart on the same 600 synthetic cohorts. Citations were checked against Crossref, arXiv or the proceedings pages. No PPMI data were read.

## Summary

**Short answer.** Apply the machinery of these methods, not the methods themselves.
- All of them compare phenotypes on the yearly scale. A phenotype that only progresses faster is therefore read as a different pathway.
- Re-targeted to a speed-scaled continuous-time reference, the same machinery works.
- Run unmodified, the published methods become the named comparators of the methods test (H-M1).

**What to apply in Step 2.**
1. **One estimator: a speed-adjusted shared-plus-deviation model.**
   - Each phenotype's continuous-time links are the reference phenotype's links plus a few deviations. Each patient runs them at their own speed.
   - Deviations are found with an adaptive lasso. Its weights start from a robust speed estimate: the weighted median of link ratios. This is the template proven in Mendelian randomisation with invalid instruments (Windmeijer 2019).
2. **Per-link tests for the gated confirmatory links.**
   - Every link's deviation is tested within one joint fit, after the speed has been identified from all links at once.
   - The identifying rule is the minimal-L1 rule from anchor-free item-bias testing (Chen Y 2023). It does not assume that any particular link is unchanged.
   - Benjamini–Hochberg control is applied across the links.
   - Testing one link at a time against the speed-only fit leaks onto neighbouring links.
3. **A per-domain test.** It asks whether anything still drives a domain differently once speed is removed. This is the question of FANS and CD-NOD, asked against the speed-scaled reference.
4. **The global speed-only test, unchanged.** It is calibrated by a parametric bootstrap over patients, because likelihood-ratio invariance tests proved "prone to false discovery" in the newest panel method (IVPP; Du 2026).
5. **Reporting.**
   - The share of the phenotype difference that speed explains.
   - The speed-adjusted difference graph.
   - The change in total effect on loss of independence, split into a speed part and a mechanism part.

**What the fourth synthetic check showed.** The set-up: six domains; 800 TD and 250 PIGD patients with five yearly transitions each; speed varying between patients; a load difference; 600 cohorts.
- **Speed only (PIGD twice as fast).**
  - Every yearly-scale method declared a mechanism difference in 100% of cohorts and flagged 55–98% of the truly shared links. These were the link-wise invariance tests, the difference-graph search, the shared-plus-deviation lasso and the entry-wise continuous-time comparison.
  - Node-level shift detectors flagged every domain.
  - The speed-adjusted versions flagged 0.4–1.0% of shared links, and the global test rejected in 6.5% of cohorts.
- **One real link difference.** No yearly method ever isolated the true link (0–1% of cohorts). The speed-adjusted joint discovery found it, and only it, in 96–97% of cohorts, and the jointly fitted per-link tests in 89–94%.
- **No speed difference at all.** Problems appeared even here:
  - separate per-phenotype graphs still disagreed in 99.5% of cohorts;
  - the shared-plus-deviation lasso flagged 28–71% of links;
  - the omnibus yearly invariance test rejected in 15% of cohorts. Patient-level speed variation combined with a load difference shifts the population-average yearly dynamics.

**Named comparators for H-M1.**
- Clinical standards:
  - multigroup RI-CLPM;
  - multigroup panel GVAR with IVPP;
  - an NCT-style permutation test on temporal networks;
  - a naive multigroup continuous-time model (ctsem).
- Causal-discovery methods:
  - tsLDiffPC and tsDCI (Ruer 2026), and DCI;
  - CD-NOD and sparse mechanism shift;
  - the data-shared lasso or multi-VAR;
  - JCI context edges;
  - separate graphs.

**Not to use.**
- NetANOVA as published: it needs an instrument for every domain and targets same-time effects.
- iSCAN and FANS: their theory is for nonlinear models, and FANS needs at least 5,000 samples per group.
- The NCT package as it stands: it handles cross-sectional networks only.
- Cross-validated lasso selection as if it were a test.
- Yearly re-targeting with a group-level speed, which is anti-conservative (toy 3).

**Novelty, sharpened.**
- The estimator template is not new: a shared vector times a scalar, plus sparse deviations, with tests. It exists for:
  - Mendelian randomisation with invalid instruments (Kang 2016; Windmeijer 2019);
  - anchor-free detection of item bias (Chen Y 2023; Halpin 2024);
  - tests of equality up to scale (Swait & Louviere 1993; Allison 1999);
  - choice models with a person-level random scale (Fiebig 2010).
- What is new is applying it to continuous-time causal dynamics: speed as the scalar, a patient-level random speed, the mapping from yearly to continuous time, and the definition-matched null.
- None of about 30 group-comparison methods across four families has a time-scale parameter.
- Proportional drift appears in the literature only as a change of time unit.
- No PD study was found that formally tested whether lagged effects differ between patient subgroups.

## 1. What Step 2 must separate, and why yearly comparisons cannot

The report (§1) splits a pathway into four parts:
- **speed**, the time scale;
- **load**, a steady push on a domain;
- **noise**;
- **mechanism**, the links themselves.

In continuous time, a PIGD patient i runs the links A_PIGD = A_TD + Δ at a personal speed s_i, with log s_i = γ + σ·u_i. Speed multiplies every link and every domain's own rate of change together. Δ is what speed cannot explain, and only Δ is a pathway difference.

The three reviews and the earlier simulations show five ways in which speed corrupts any comparison of yearly coefficients:

1. **The change is dense.**
   - Self-effects (each domain's year-to-year persistence) change most, then direct links, then entries with no direct link.
   - In a sparse six-domain system at 1.5 times the speed, persistence changed by −0.09 to −0.13. Direct links changed by +0.02 to +0.04, and entries without a link by less than 0.01 (clinical-methods review).
   - At twice the speed, in the system of toy 1, axial → independence (no direct link) rose from 0.021 to 0.068.
2. **One real difference leaves a footprint.**
   - A single continuous-time difference moves several yearly entries.
   - In a three-domain chain, a difference on X2 → X3 also changes the yearly X1 → X3 entry (difference-graph review).
   - Yearly tests therefore cannot localise even a real difference.
3. **Noise changes.**
   - Yearly innovation variances grow with speed. In the difference-graph review's three-domain example they were 0.63–0.66 at the slower speed and 0.78–0.84 at 1.5 times the speed.
   - That breaks the equal-noise premises of DDAG, of LDiffPC (its Proposition 2) and of DCI's orientation step.
4. **The direction can flip.**
   - A yearly cross-lag first rises and then falls as the interval lengthens.
   - Once one year passes a link's peak (2.67 years for one link in the review's example), a faster phenotype shows a *weaker* yearly link (Kuiper & Ryan 2018).
5. **Speed varies between patients.**
   - Group-average yearly dynamics are then no longer a matrix power (toy 3).
   - Pooled regressions are distorted further when faster patients are also further along.

So Step 2 must estimate and compare in continuous time, with each patient's own speed. There the zero pattern does not depend on speed, speed is one scalar, and Δ is what remains.

## 2. The methods, family by family

**Table 1. The four families and what Step 2 takes from each.**

| Family | Methods read (software) | What counts as a "difference" | What a pure speed difference does | What Step 2 borrows | Named H-M1 comparator? |
|---|---|---|---|---|---|
| Difference graphs | DCI 2018 (causaldag); DDAG 2024 (no official code; a re-implementation in pyCIPHOD); LDiffPC 2026, preprint (pyCIPHOD); tsLDiffPC and tsDCI 2026 (pyCIPHOD); Assaad 2025 (theory only) | a regression coefficient, or precision entry, that differs between groups | every yearly coefficient differs; the equal-noise premises fail; label permutation is invalid | the per-link equality test, applied to Δ in continuous time with all other deviations free; stability selection over patients; penalising the difference itself | yes: tsLDiffPC, tsDCI, DCI (DDAG optional) |
| Node-level shift detectors | CD-NOD 2020 (causal-learn); sparse mechanism shift 2022 (sparse_shift); iSCAN 2023 (iscan-dag); FANS 2026 (research code, GPU); Huang & Zhang 2024; JCI and S-FCI context edges | a domain whose distribution given its parents differs | every domain is flagged; load shifts are read as mechanism changes, or as "noise" by FANS | the per-domain question, asked against the speed-scaled reference | yes: CD-NOD, sparse mechanism shift, JCI context edges. iSCAN and FANS are excluded (linear data, sample size) |
| Shared plus deviations | data-shared lasso 2016; stratified lasso 2017; joint Granger models 2015–2022; joint graphical lasso 2014; multi-VAR 2022–2024 and penalised subgrouping 2026 (multivar); NetANOVA 2023 | a deviation from a shared parameter that survives the penalty | dense deviations; the "shared" part becomes a hybrid of both groups; none gives calibrated tests | the parameterisation, adaptive weights, BIC-type tuning, stability selection, NetANOVA's factorial layout | yes: data-shared lasso, multi-VAR. NetANOVA cannot be run on PPMI |
| Group-invariance tests in clinical panel research | multigroup RI-CLPM (lavaan); multigroup panel GVAR, INIT and IVPP (psychonetrics, IVPP); NCT 2023; moderated networks and BGGM (cross-sectional only); group effects in ctsem and DynNet | rejection of equal lagged effects; edges freed one at a time | invariance is rejected, with self-effects flagged first; additive group effects on drift flag every non-zero drift entry | IVPP's two-step design (a global test against the constrained model, then per link), calibrated by bootstrap; ctsem as a second host for the speed model | yes: RI-CLPM, panel GVAR with IVPP, NCT-style permutation, naive ctsem |
| Model-light check | shared eigenvectors (Xu, Düker & Matteson 2024) | yearly matrices that do not share eigenvectors | none, if both phenotypes' yearly matrices come from one drift | a cheap sensitivity check | no (diagnostic) |

### 2.1 Difference graphs

**What they do.**
- **DCI** (Wang 2018) deletes a difference edge when a regression coefficient is equal across the two groups for some conditioning set. It uses F-tests, and orients edges by equality of residual variances; full orientation needs equal noise variances. The maintained code is in causaldag, with a stability-selection wrapper.
- **DDAG** (Malik 2024) estimates the difference of the two precision matrices directly, with an ℓ1 penalty, then peels off terminal vertices.
  - It assumes the two groups share their noise variances. The abstract's "unequal variances" means unequal across variables, not across groups.
  - It has no tests and no official code; pyCIPHOD holds a re-implementation.
- **LDiffPC** (Bystrova & Devijver 2026, preprint) is a PC-style search built on coefficient-equality tests.
  - It introduces diff-separation and diff-faithfulness, which explain how one change propagates into coefficients that did not themselves change.
  - Its theory assumes noise that does not change between groups.
  - The preprint reports no simulations. Its authors' package, pyCIPHOD, uses a two-sample Wald test instead of the paper's permutation test, and never tests the largest conditioning sets.
- **tsLDiffPC and tsDCI** (Ruer 2026) extend these to linear time series with lags, for one long series per regime, at α = 0.05. They are the closest published comparators in discrete time.
- **Assaad (2025)** gives conditions under which total and direct effects can be identified from a difference graph. It gives no rule for turning direct changes into total changes.

**Under speed.**
- The correct yearly difference graph is dense, so these algorithms would recover it faithfully. The failure lies in the target, not the algorithm.
- Speed also changes the yearly noise, so their premises fail.
- Label permutation is invalid because the groups are not exchangeable.

**What Step 2 borrows.**
1. The per-link equality test, applied to Δ in continuous time with every other deviation estimated at the same time. This plays the role of "conditioning on all other parents".
2. DCI's stability selection, resampling patients.
3. DDAG's principle of penalising the difference itself.

Diff-separation explains, for the paper's argument, why yearly comparisons smear one change across several coefficients.

### 2.2 Node-level shift detectors

**What they do.**
- **CD-NOD** (Huang 2020) adds the group as a node; its edges mark domains with "changing modules".
- **Sparse mechanism shift** (Perry 2022) prefers the graph with the fewest changed conditional distributions, testing at α = 0.05 divided by the number of domains.
- **iSCAN** (Chen T 2023) finds shifted nodes from the curvature of the log-density. Its theory covers nonlinear models only.
- **FANS** (Seo 2026) finds shifted nodes and splits each shift in two:
  - a function shift, where the residual depends on the parents;
  - a noise shift.

  It was tested with 50,000 samples per group, and its performance degrades below 5,000.
- **Huang & Zhang (2024)** give node scores for differential Bayesian networks.
- **JCI and S-FCI context edges** apply the same idea inside full discovery (Mooij 2020; Li 2023).

**Under speed.**
- Every domain is shifted.
- A group-level load shift is flagged as a mechanism change by tests on the conditional distribution, and labelled a "noise shift" by FANS.

**What Step 2 borrows.**
- **The per-domain question:** does a domain still depend on its parents differently once the speed-scaled reference is removed? This becomes the speed-adjusted per-domain test.
- **The framing:** sparse deviations from a speed-scaled reference are the continuous-time form of the sparse-mechanism-shift hypothesis.

### 2.3 Shared plus deviations

**What they do.**
- **The data-shared lasso** (Gross & Tibshirani 2016) and the **stratified lasso** (Ollier & Viallon 2017) write each group's coefficients as β_g = β + Δ_g, under one penalty with group weights.
- **Group or fused penalties** are used by:
  - joint Granger models (Basu 2015; Skripnikov & Michailidis 2019; Manomaisaowapak & Songsiri 2022);
  - the joint graphical lasso (Danaher 2014).
- **multi-VAR and penalised subgrouping** (Fisher 2022; Crawford 2026; R package multivar) split each person's transition matrix into common, subgroup and person-specific parts with adaptive weights.
- **NetANOVA** (Jiang & Zhang 2023) writes several causal networks as a baseline plus differential effects, with bootstrap p-values and false-discovery control. It needs a known instrument for every node and targets same-time effects, so it cannot run on PPMI.

**Facts that matter.**
- None of these gives calibrated tests of group differences. NetANOVA's bootstrap p-value is neither defined nor checked.
- multivar accepts known group labels. But all its published simulations use 30 or more occasions per person, it has no missing-data handling, and it standardises each person's series. At about five transitions per person only eBIC tuning is usable.

**Under speed.**
- The deviations are dense.
- With two groups and standard weights, the data-shared lasso's "shared" matrix becomes an entry-by-entry hybrid: the faster group's persistence and the slower group's links. Both groups therefore receive deviations.
- Reference coding loads every difference onto the non-reference group.
- Fused penalties split each difference.

**What Step 2 borrows.**
- **From these papers:**
  - the shared-plus-deviation parameterisation: reference coding for two groups, and NetANOVA's baseline-plus-differential layout for two phenotype axes;
  - adaptive weights;
  - eBIC or two-step BIC tuning;
  - stability selection.
- **Identification and inference come from outside time series.** Two literatures supply them:
  - Mendelian randomisation with invalid instruments: outcome effects = θ × exposure effects + sparse direct effects (Kang 2016; Bowden 2016; Guo 2018; Windmeijer 2019);
  - anchor-free detection of item bias (Chen Y 2023; Halpin 2024).

  Both identify the scalar by robust rules, and give valid tests for individual deviations. The rules are a weighted median or minimal-L1 fit of the entry ratios, a robust M-estimate started at the median, or a plurality vote.
- **The rule works only on the continuous-time scale.** Applied to yearly matrices, it returned a "speed" of 0.86 for a group running 1.5 times faster (shared-deviation review).

### 2.4 Group-invariance tests used in clinical panel research

**Multigroup RI-CLPM** (Mulder & Hamaker 2021).
- It compares, by χ² difference, a model with lagged effects free against one with them equal across groups.
- The authors warn that it tests raw coefficients and needs equal intervals.
- It is the form used in PD and PPMI work: PD versus controls in DeNoPa (Schroeders 2022), and PPMI with no phenotype comparison (Hodgson 2026). Neither tested formally whether lagged effects differ between groups.

**Multigroup panel GVAR** (Epskamp 2020; psychonetrics).
- It constrains the temporal network to be equal, then compares models by likelihood ratio, AIC or BIC, with score tests and partial pruning.
- **IVPP** (Du 2026) formalises a global test followed by per-edge pruning. Its simulations found:
  - the likelihood-ratio version "prone to false discovery";
  - comparison against the constrained model better than against the free one.

**Other network comparisons.**
- **NCT** (van Borkulo 2023) is for cross-sectional networks only. Its code reads the upper triangle, so it drops self-effects and half of the directed edges.
- Temporal comparisons use custom permutation code (Kenny 2026).
- Moderated network models and BGGM comparisons are cross-sectional.
- Cross-lagged panel networks (Wysocki 2025) have no group procedure.

**Continuous-time models.**
- ctsem (Driver 2017) adds group effects additively to each drift entry. DynNet (Rouanet 2026) does the same and cannot host a patient-level speed without code changes.
- Proportional drift appears in the literature only as a change of time unit (Kuiper & Ryan 2018).
- ctsem 3.11 has the building blocks for drift = speed × A: a speed parameter with a random effect and a phenotype effect, referenced in every drift formula. No worked example exists.

**Applied practice.** Formal tests of lag invariance between patient subgroups are rare, and none was found in PD.

**Under speed.**
- Every one of these rejects invariance, flagging self-effects first.
- Additive group effects in continuous time flag every non-zero drift entry and no zero entry.

**What Step 2 borrows.**
- IVPP's two-step design: a global test against the constrained (here, speed-only) model, then per-link tests, calibrated by parametric bootstrap.
- ctsem, as an independent second host for the speed model.

### 2.5 A model-light check: shared eigenvectors

**Why it works.**
- If one drift generates both phenotypes' yearly matrices, the matrices commute and share eigenvectors, whatever the speeds.
- Xu, Düker & Matteson (2024) test simultaneous diagonalisability of asymmetric matrices, including VAR coefficient matrices.
- A mechanism difference generally breaks the commutation.

**Limits.**
- It is a necessary condition only, and it can have low power.
- Pooled regressions are biased when speed correlates with disease state.

So it is a diagnostic, not the test of record.

## 3. Fourth synthetic check: each family next to its speed-adjusted counterpart

**Design** (`scripts/toy/toy_comparators.py`, with `toy_comparators_extra.py` and `toy_comparators_joint.py` on the same cohorts):
- **Structure.** Six domains and seven direct links, as in toy 1.
- **Dynamics.** A continuous-time model run at each patient's own speed. Log-speed varies with standard deviation 0.4 within each phenotype, as in toy 3. PIGD has an extra push on the axial domain (load).
- **Cohorts.** 800 TD and 250 PIGD patients, with five yearly transitions each.
- **Scenarios:**
  - S0: equal median speed (load difference only);
  - S1: PIGD twice as fast;
  - S2: S1 plus a PIGD-only link, axial → independence;
  - S3: S1 plus function → independence stronger in PIGD (0.25 → 0.40).
- **Replicates.** 200 cohorts each for S0 and S1, and 100 each for S2 and S3. Every procedure ran on the same cohorts.
- **Map.** The speed-adjusted procedures use the map's eight links: the seven links plus axial → independence, as Step 1 would supply.
- **Comparators.** Each naive procedure was written from the published definition of its family, not from the authors' code. §5 gives the authors' software for H-M1.

| ID | Procedure | Stands for |
|---|---|---|
| N1 | Link-wise phenotype-interaction tests, all lagged effects in one model; Benjamini–Hochberg (BH) at 5% | v0.3's link-wise tests; per-edge tests in multigroup RI-CLPM, panel GVAR and moderated VAR |
| N1s | Each interaction freed alone from the invariant model; BH | score tests of RI-CLPM and psychonetrics; IVPP's per-edge step |
| N1o | Omnibus test of all 36 lagged effects | the global invariance test of RI-CLPM, panel GVAR and IVPP |
| N2, N2b | Phenotype → domain given the lagged parents: intercept and slopes (N2), slopes only (N2b) | context edges of JCI and S-FCI; CD-NOD changing modules; sparse mechanism shift; iSCAN; FANS |
| N3 | Difference-graph search with time tiers: keep an edge unless some set of lagged domains makes its coefficient equal (α = 0.05 per test; one regression direction) | LDiffPC and DCI logic restricted to time order. The published tsLDiffPC/tsDCI search is more lenient and behaves differently (tsLDiffPC/tsDCI note) |
| N4, N4b | Shared effects plus group deviations, lasso; cross-validated (N4) or BIC (N4b) | data-shared lasso, multi-VAR, joint Granger models |
| N5 | Group-specific continuous-time drift, entry-wise tests; BH | naive multigroup ctsem or DynNet |
| N6 | Permutation test of the yearly network difference (overall size, largest entry) | NCT-style tests on temporal networks |
| N7 | Separate graphs: a link significant in one phenotype only | per-phenotype graphs read naively |
| O1 | Speed-only likelihood-ratio test with patient-level speed (13 df) | Step 2 global test |
| O2 | Adaptive lasso on Δ (weighted-median start), BIC | Step 2 joint discovery |
| O3 | Score test of each Δ at the speed-only fit; BH | per-link test against the constrained model |
| O3j | Wald tests with all eight cross-link deviations free and each domain's own rate shared; BH | per-link test, anchored on the domains' own rates |
| O3m | Minimal-L1 speed, Monte Carlo p-values; BH | per-link test, anchor-free (Chen Y 2023) |
| O4 | Score test of all deviations into one domain at the speed-only fit; BH over domains | per-domain test against the constrained model |
| O4j, O4m | The same question from the O3j and O3m fits: all deviations into one domain tested at once | per-domain test, anchored and anchor-free |
| O5 | Commutator of the two yearly matrices, centred bootstrap | shared-eigenvector check |

**Table 2. No real mechanism difference.** Share of the eight map links falsely flagged as different (in brackets, cohorts with at least one false flag), and rejection rates for the global tests.

| Procedure | S0: equal speed, load difference | S1: PIGD twice as fast |
|---|---|---|
| N1 link-wise interaction tests | 0.1% (3.5%) | 57% (100%) |
| N1s one interaction at a time | 0.2% (3.5%) | 60% (100%) |
| N3 difference-graph search | 1.6% (37%) | 55% (100%) |
| N4 shared plus deviations, cross-validated | 71% (100%) | 98% (100%) |
| N4b shared plus deviations, BIC | 28% (96%) | 92% (100%) |
| N5 naive continuous-time, entry-wise | 0.2% (6%) | 63% (100%) |
| N7 separate graphs | 8% (99.5%) | 4% (100%) |
| **O2 speed-adjusted joint discovery** | **0.4% (3%)** | **0.4% (3%)** |
| O3 per-link score tests at the speed-only fit | 0.4% (3%) | 0.6% (5%) |
| **O3j joint per-link tests, anchored** | 0.4% (3.5%) | 1.0% (7%) |
| **O3m per-link tests, anchor-free** | 0.25% (2%) | 0.4% (3%) |
| *Global tests (share of cohorts rejecting)* | | |
| N1o omnibus invariance test | 15% | 100% |
| N6 permutation test, overall / largest entry | 5.5% / 4.5% | 100% / 100% |
| O5 shared-eigenvector check | 4.0% | 3.5% |
| **O1 speed-only test** | **6.0%** | **6.5%** |
| *Domain-level tests (share of domains falsely flagged)* | | |
| N2 phenotype → domain, intercept and slopes | 17% (axial in 86% of cohorts, from load) | 100% |
| N2b phenotype → domain, slopes only | 1.7% | 99.6% |
| O4 per-domain score test at the speed-only fit | 1.7% | 0.9% |
| **O4j per-domain test, joint and anchored** | 1.3% | 1.0% |
| **O4m per-domain test, anchor-free** | 0.6% | 0.7% |

Monte Carlo standard errors are about 1.6 percentage points at 5% with 200 cohorts.

**Table 3. One real link difference.** Three figures for each link-level procedure:
- power for the true link;
- the share of cohorts in which exactly the true link, and no other map link, was flagged;
- the mean number of other map links flagged.

Domain tests show how often the true target domain (independence) was flagged, and how often it was flagged alone.

| Procedure | S2: PIGD-only axial → independence | S3: function → independence stronger in PIGD |
|---|---|---|
| N1 link-wise interaction tests | 100%; exactly 0%; 4.0 others | 100%; 0%; 3.8 |
| N1s one interaction at a time | 100%; 0%; 3.7 | 100%; 0%; 3.7 |
| N3 difference-graph search | 100%; 0%; 3.9 | 100%; 1%; 2.8 |
| N4 / N4b shared plus deviations | 100%; 0%; 6.8 / 6.2 | 100%; 0%; 6.9 / 6.3 |
| N5 naive continuous-time | 100%; 0%; 4.9 | 100%; 0%; 3.8 |
| N7 separate graphs | 36%; 0%; 0.0 | 0%; 0%; 0.3 |
| **O2 speed-adjusted joint discovery** | **100%; 96%; 0.04** | **100%; 97%; 0.03** |
| O3 per-link score tests at the speed-only fit | 100%; 35%; 0.8 | 100%; 4%; 1.2 |
| **O3j joint per-link tests, anchored** | 99%; 92%; 0.07 | 100%; 91%; 0.10 |
| **O3m per-link tests, anchor-free** | 97%; 94%; 0.03 | 97%; 89%; 0.08 |
| O1 speed-only test (rejects) | 98% | 100% |
| O5 shared-eigenvector check (rejects) | 28% | 78% |
| N2 / N2b domain tests | all six domains flagged | all six domains flagged |
| O4 per-domain score test at the speed-only fit | independence 99%; independence alone 73% | 100%; 51% |
| **O4j per-domain test, joint and anchored** | independence 100%; alone 95% | 100%; 94% |
| **O4m per-domain test, anchor-free** | independence 96%; alone 93% | 100%; 96% |

**What the check shows.**

1. **Speed defeats every yearly-scale method, including those built to find differences.**
   - With speed only, the link-level methods flagged on average 4.4–7.8 of the 8 map links, in every cohort (separate graphs aside).
   - The yearly link-wise tests flagged axial → independence, which has no direct link, in 39% of cohorts. That is a spurious "new pathway" in PIGD, created by within-year indirect effects.
   - The node-level detectors flagged every domain.
   - The global invariance and permutation tests rejected in every cohort.
   - A real difference moves several yearly entries (its footprint). Discounting flags inside the footprint barely helps: outside it, the yearly methods still flagged 46–98% of map links (S2, S3).
2. **Moving to continuous time is not enough on its own.**
   - Under speed only, the entry-wise continuous-time comparison flagged the true links 71% of the time, because their strength scales with speed. It flagged the absent link in only 4% of cohorts.
   - Continuous time protects the structure, not the strengths. Speed must still be estimated and removed.
3. **Joint estimation localises; one-at-a-time testing leaks.**
   - The speed-adjusted joint discovery isolated the true link in 96–97% of cohorts.
   - Score tests at the speed-only fit found the true link every time but also flagged others: exactly right in only 35% (S2) and 4% (S3). When one link truly differs, the speed-only fit is wrong in a way that spills onto its neighbours; for example, the speed estimate rose from 2.0 to 2.2.
   - Testing every link within one joint fit removes the leak. The anchored and anchor-free per-link tests isolated the true link in 89–94% of cohorts. Without a difference they flagged 0.25–1.0% of shared links.
   - The anchor-free speed estimate stayed at 2.0 even when a link truly differed (median 2.00–2.03), while the speed-only fit's estimate rose to 2.2.
   - The joint per-domain tests named the changed domain alone in 93–96% of cohorts. The score version at the speed-only fit did so in only 51–73%.
4. **Three problems appear even without a speed difference.**
   - Significance in one phenotype but not the other is not a difference: separate graphs disagreed in 99.5% of cohorts with no difference at all.
   - Penalised selection is not a test: the shared-plus-deviation lasso flagged 28% (BIC) to 71% (cross-validation) of links.
   - The omnibus yearly invariance test rejected in 15% of cohorts, three times its nominal rate. Speed varies within each phenotype, and the load difference moves PIGD patients through a different range of states, so the population-average yearly dynamics differ slightly even though every patient's mechanism is the same.
5. **The speed model recovered its own parameters.**
   - Speed ratio: median estimate 1.99 (true value 2).
   - Spread of log-speed within phenotypes: 0.40 (true value 0.4).
   - In the free fits, speed explained a median 95% of the phenotype difference in continuous-time links under speed only, and 83–85% when one link truly differed.

**The published tsLDiffPC/tsDCI search.** A second-round investigation re-implemented it faithfully (`tsldiffpc_tsdci_investigation.md`). It also conditions on current-time values of other domains and tests both regression directions, so it removes edges far more readily than N3. On the same cohorts it behaved differently:
- under speed only, it flagged 0.7% of shared cross-links, but named 48% of domains false root causes through persistence;
- its power for a real link was 49% (new link) and 10% (stronger link).

**Larger cohorts.** A separate check closes a gap left in the report (§2).
- With exact gradients, the patient-level speed-only test could now be run on the same larger speed-only cohorts as the group-level test (toy 3's three-domain system; `toy_random_speed_scale_person.py`).
- It stayed calibrated as cohorts grew; the group-level test did not.

| Cohort size (TD + PIGD), five transitions each | 600 | 2,400 | 9,600 |
|---|---|---|---|
| Group-level test rejects, slow / fast dynamics | 6.5% / 12.5% | 10.5% / 19% | 27% / 65% |
| Patient-level test rejects, slow / fast dynamics | 4.5% / 5.0% | 5.0% / 3.5% | 3.0% / 7.0% |
| Cohorts per cell | 200 | 200 | 100 |

- Across the six cells the patient-level test rejected 46 of 1,000 speed-only cohorts (4.6%).
- Its mean likelihood ratio was 7.9–8.3 (8 when calibrated), and its median speed estimate was 2.00–2.01 (true value 2).

**Limits of the check.**
- Each naive procedure is an analogue of its family, not the authors' code.
- The data are linear and Gaussian, with no hidden patient-level burden, no dropout, no ordinal scores and no artefacts from the phenotype definition.
- The map was supplied rather than learned.
- There was one structure and one speed ratio, at sizes near PPMI's.

The H-M1 benchmark will run the authors' software (§5) on cohorts fitted to PPMI, with these complications included.

## 4. What to apply in the study

### 4.1 The model

For patient i with phenotype g, the yearly transitions are the exact discretisation of

dX = s_i·[(A + Δ_g)·X + b_g]·dt + √s_i·G_g·dW, with log s_i = γ_g + β′z_i + σ·u_i,

where:
- the reference phenotype (TD) has Δ = 0 and γ = 0;
- A lives on the Step 1 map's links;
- the load b_g and the noise G_g differ by phenotype;
- the baseline characteristics z_i may change speed;
- patient-level burden enters as in Step 4.

**What separates speed from mechanism.**
- Speed moves every link and every domain's own rate together; a mechanism difference moves only some. The sparsity of Δ therefore identifies the speed.
- The identifying rule is minimal-L1: speed = the weighted median of the ratios of PIGD's to TD's continuous-time entries, weighted by the size of TD's entries.
- The rule is exact as long as the deviating entries carry less than half of the weight and are not all of one sign (Chen Y 2023; Bowden 2016).
- A uniform strengthening of every link *and* every domain's own rate is speed by definition.

### 4.2 Finding the deviations (exploratory localisation)

1. **Adaptive lasso on Δ.**
   - The domains' own rates are included in Δ; speed is left unpenalised.
   - The weights are 1/|Δ̃|, with Δ̃ taken from the free fits at the weighted-median speed. This is Windmeijer et al.'s median-initialised adaptive lasso, which selects consistently (the "oracle property").
2. **Tuning.** BIC with the number of transitions in the penalty, and eBIC as a sensitivity analysis. Refit on each selected set of links.
3. **Stability selection** over random halves of the patients (Shah & Samworth 2013), reporting how often each link is selected.
4. **Intervals and inference.** Patient-bootstrap intervals for the selected deviations. If selected links are also tested, use sample splitting.

In the check, this flagged 0.4% of truly shared links and isolated a real difference in 96–97% of cohorts.

Two things not to import from the shared-plus-deviation software:
- standardising each patient's series separately, which is multivar's default;
- p-values read naively off a bootstrap of penalised fits.

### 4.3 Tests and calibration

- **Global (H-P, primary).**
  - The speed-only likelihood ratio with patient-level speed, unchanged from the report.
  - It is calibrated by a parametric bootstrap from the fitted speed-only model, resampling patients, inside the extended definition-matched null.
  - The χ² reference held in the check (6.0–6.5%), but real data add ordinal scores, dropout and burden, and IVPP's likelihood-ratio test was over-liberal.
- **Per link (H-C, gated).** Each pre-specified link's deviation is tested within one joint fit, never one link at a time against the speed-only fit.
  - **Primary, anchor-free (O3m).**
    - Fit both phenotypes' continuous-time links freely, and identify the speed by the minimal-L1 rule.
    - Take each deviation's p-value from Monte Carlo draws of the free fit (the method of Chen Y 2023), with BH over the pre-specified family.
    - In the check it flagged 0.25–0.4% of shared links when no link differed, and isolated the true link in 89–94% of cohorts when one did.
  - **Sensitivity, anchored (O3j).**
    - Wald tests with every cross-link deviation free and each domain's own rate shared.
    - It flagged 0.4–1.0% of shared links and isolated the true link in 91–92% of cohorts.
    - It assumes that no domain's own rate deviates.
  - Both are calibrated in the study by the same parametric bootstrap as the global test.
- **Per domain (secondary).** The same joint fits give a test of all deviations into a domain at once, with the anchor-free version as primary.
  - In the check it flagged 0.6–0.7% of unchanged domains, and named the changed domain alone in 93–96% of cohorts.
  - The score version at the speed-only fit leaked: the changed domain alone in only 51–73% of cohorts.
- **Model-light check (sensitivity).**
  - The shared-eigenvector check needs no continuous-time model: it is a centred patient bootstrap of the commutator of the two yearly matrices.
  - In the check it rejected in 3.5–4.0% of cohorts without a mechanism difference. With one it rejected in 28% (new link) and 78% (stronger link).
  - Its power is modest, and pooled regressions bias it when speed tracks stage. It therefore supports the speed-only test but never replaces it. Xu, Düker & Matteson (2024) give the formal version.

### 4.4 Outputs for clinicians

1. **How much of the phenotype difference is speed.** The share of the difference in continuous-time links that a single speed explains, from the free fits.
2. **The speed-adjusted difference graph.** The deviations Δ, with intervals and selection frequencies, expressed per year at TD's speed.
3. **A card for each domain.** It shows:
   - the common speed;
   - the domain's load;
   - its noise;
   - which links into it deviate.
4. **The route to loss of independence.**
   - The total effect of each domain on loss of independence over h years is split exactly into two parts:
     - a speed part, exp(h·s·A) − exp(h·A);
     - a mechanism part, exp(h·s·(A+Δ)) − exp(h·s·A).
   - The two phenotypes are also compared at speed-matched horizons (h years for TD, h/s for PIGD), where the speed part vanishes.
   - A first-order (Fréchet-derivative) formula attributes the mechanism part to individual deviations. A perturbation travels to the changed link, crosses it, and travels on to independence.
   - If loss of independence is modelled as an absorbing event rather than a domain, this needs the joint state–event model.

### 4.5 Two phenotype axes and more than two groups

**Factorial layout,** following NetANOVA's baseline-plus-differential effects:
- speed: log s_i = γ_PIGD·PIGD_i + γ_RBD·RBD_i (+ interaction) + σ·u_i;
- links: A_i = A + PIGD_i·Δ_PIGD + RBD_i·Δ_RBD;
- load and noise by group;
- tests per factor, with BH across factors.

**Three or more unordered groups:** symmetric deviations with group weights r_g = √(n_g/N), as in the data-shared and stratified lasso.

### 4.6 Software and compute

**Our engine.**
- Built in PyTorch, with exact gradients and second derivatives through the matrix exponential. Its likelihood reproduces toy 3's to machine precision.
- The full Step 2 suite took a median of about 30 seconds per synthetic cohort on one processor core (six domains, 1,050 patients). That covers the speed-only and free fits, the per-link tests and 12 penalised fits with refits.
- At ten domains expect several times that. A parametric bootstrap with 200 replicates remains feasible.

**ctsem 3.11** (Driver, Voelkle & Oud 2026) as an independent second host.
- A speed parameter with a random effect and a phenotype effect, referenced in every drift formula.
- The syntax is drafted in the clinical-methods review but untested, so it is validated by simulation first.
- ctsem has no ordinal data type.

**DynNet** cannot host a patient-level speed without code changes. It stays as a naive continuous-time comparator.

## 5. The comparator set for the methods test (H-M1)

**Table 4. Named comparators, each run unmodified on cohorts simulated from PPMI-fitted models.**

| ID | Comparator | Family | Software | Primary settings | Expected under speed only |
|---|---|---|---|---|---|
| C1 | Multigroup RI-CLPM | clinical SEM standard | lavaan 0.7-3 | MLR, FIML, lagged effects equal over waves; global χ² difference (36 df for six domains); per-edge score tests with BH | invariance rejected; self-effects first |
| C2 | Multigroup panel GVAR, with IVPP | network-psychometric standard | psychonetrics 0.17.8; IVPP 1.1.2 | FIML; temporal network equal versus free; equality score tests (BH); IVPP partial pruning at its default α = 0.01 | as C1 |
| C3 | NCT-style permutation test on temporal networks | network comparison | custom code following NCT | 1,000 patient-level label permutations; overall strength, largest difference, edge-wise; self-effects reported separately | rejected in every cohort (N6) |
| C4 | Naive multigroup continuous-time model | continuous time | ctsem 3.11.1 | an additive phenotype effect on each drift entry; likelihood ratio and entry-wise intervals | every non-zero drift entry flagged; no zero entry |
| C5 | tsLDiffPC and tsDCI | difference graphs | pyCIPHOD (MIT, beta; pin the commit) | two-slice tables; α = 0.05; maximum conditioning set 3–4 at ten domains | few false cross-links, but persistence read as root causes in about half of the domains; low power (tsLDiffPC/tsDCI note) |
| C6 | DCI | difference graphs | causaldag | lag-to-current candidate pairs; package defaults (α = 0.1) and α = 0.05; stability-selection variant | as C5 |
| C7 | CD-NOD | node-level shift | causal-learn 0.1.4.8 | Fisher-z test, α = 0.05, time tiers | every domain a changing module |
| C8 | Sparse mechanism shift | node-level shift | sparse_shift (MIT; pin the commit) | Fisher-z test on the map; α = 0.05 divided by the number of domains | every domain shifted |
| C9 | Data-shared lasso and multi-VAR | shared plus deviations | glmnet with penalty factors; multivar 1.4.0 with known subgroups | eBIC (γ = 0.5) and two-step BIC; patient-blocked cross-validation as sensitivity | dense deviations; a hybrid shared part |
| C10 | JCI-FCI and J-PCMCI+ context edges | context nodes | causal-learn; tigramite | phenotype as a context node | phenotype → every domain |
| C11 | Separate graphs | naive split | per-phenotype tiered FCI | v0.3's Analysis B read naively | discordant links even without differences |
| C12 | DYNOTEARS (added in protocol v0.4.1) | gradient-based structure learning | re-implementation of the authors' causalnex code, validated against the paper; no installation | one graph per phenotype from the two-slice tables, compared link by link; λ = 0.05, thresholds 0.3 (same-time) and 0.1 (lagged), the paper's large-sample settings | discordant lagged links in 99.5% of speed-only and 42.5% of equal-speed cohorts; never isolates a real difference (protocol Appendix E, check 6) |

DDAG (through pyCIPHOD's re-implementation) is optional. iSCAN and FANS are excluded: their theory is for nonlinear models, and FANS needs at least 5,000 samples per group.

**A caution on C2.** The panel GVAR keeps the covariances stationary, with the first wave drawn from the stationary distribution. Cohorts followed from diagnosis are not stationary, so its random-intercept cross-lagged network variant is run as well.

**Scoring rules, fixed before any run.** Four outcomes:
- **False mechanism-difference rate per truly shared link,** reported separately for self-effects, direct links and links absent from the drift.
- **Any-flag rate.**
- **Power.**
- **Localisation,** in two versions:
  - exact: the true link and nothing else;
  - footprint-aware: flags on the yearly entries that the real difference itself moves are counted separately.

All are reported as functions of cohort size and speed ratio. Each method is run at its own default α and at a common 0.05.

**Prerequisites.**
- R with lavaan, psychonetrics, IVPP, glmnet, multivar and ctsem.
- Python with causaldag, pyCIPHOD and sparse_shift. causal-learn and tigramite are already installed.

Installing these needs the investigator's approval. Versions and commits will be pinned.

## 6. Proposed changes for protocol v0.4

- **Step 2 estimator.** Specify it as in §4.1–4.2. This replaces the general wording "adapts the data-shared lasso".
- **Per-link (H-C) tests.** Test each pre-specified link's deviation within one joint fit: the anchor-free version (minimal-L1 speed, Monte Carlo p-values) as primary, the anchored joint Wald as sensitivity. Apply BH over the H-C family, gated on H-P.
- **Per-domain test.** Add it as a secondary analysis.
- **Global test.** Calibrate it by parametric bootstrap from the fitted speed-only model, inside the extended definition-matched null.
- **Exploratory localisation.** Use stability selection and report selection frequencies.
- **H-M1.**
  - Adopt the comparator set and scoring rules of §5, fixed before running.
  - Add two scenarios: equal speed with a load difference, and differing load variance (burden) only.
  - Report results by type of link, and as functions of cohort size and speed.
- **Reporting.** Add the share explained by speed, and the total-effect decomposition at speed-matched horizons.
- **Sensitivity analysis.** Add the shared-eigenvector check.
- **Novelty wording.** Cite Mendelian randomisation with invalid instruments, anchor-free item-bias detection and scale-heterogeneity tests as precedents for the estimator template. The claim rests on the application: continuous-time causal dynamics, a patient-level speed, the definition-matched null.
- **Corrections to the report and the addendum.**
  - DDAG assumes the two groups share their noise variances.
  - DDAG estimates precision differences rather than testing coefficients.

## 7. Limits and open questions

- **The synthetic check** used analogues rather than the authors' code, with linear Gaussian scores, no burden, no dropout and no definition artefacts. The H-M1 benchmark removes these simplifications.
- **Calibration when speed is estimated.** A parametric bootstrap is the default. Its cost at ten domains is still to be measured.
- **When identification fails.** If deviations carry half of the weight, or all point the same way, speed and mechanism cannot be separated from these data. An example is PIGD strengthening every cross-link while each domain's own rate stays the same. The study then reports the share explained by speed, and a sensitivity analysis under a plurality rule.
- **Recovering continuous-time links from yearly data.** The matrix logarithm can be non-unique when a yearly matrix has complex or negative eigenvalues. This is to be checked on PPMI aggregates in WP0.
- **Speed correlated with stage.** The model conditions on each patient's current state, but the eigenvector check and every yearly comparator are biased by this correlation.
- **Loss of independence.** If it is treated as an absorbing event, the total-effect decomposition needs the joint state–event extension.
- **Not read in full:**
  - the data-shared lasso paper, so its group weights come from secondary sources;
  - the final version of Crawford 2026;
  - IVPP and the eigenvector test, read at abstract and documentation level;
  - the OpenReview pages of DDAG and FANS, which are behind a bot check.

## References

Each entry was checked on 2–3 November 2026 against the source marked in brackets: C = Crossref record, A = arXiv record, P = proceedings page, J = JMLR page. "(in the report)" marks entries already verified in the investigation report.

- Allison PD. Comparing logit and probit coefficients across groups. Sociol Methods Res 1999;28(2):186–208. doi:10.1177/0049124199028002003 [C]
- Assaad CK. Causal reasoning in difference graphs. Proc CLeaR 2025, PMLR 275:15–30. arXiv:2411.01292 [A]
- Basu S, Shojaie A, Michailidis G. Network Granger causality with inherent grouping structure. J Mach Learn Res 2015;16(13):417–453 [J]
- Bowden J, Davey Smith G, Haycock PC, Burgess S. Consistent estimation in Mendelian randomization with some invalid instruments using a weighted median estimator. Genet Epidemiol 2016;40(4):304–314. doi:10.1002/gepi.21965 [C]
- Bystrova D, Devijver E. Constraint-based difference graph discovery in a linear setting. arXiv:2606.13947 (2026) [A]
- Chen T, Bello K, Aragam B, Ravikumar P. iSCAN: identifying causal mechanism shifts among nonlinear additive noise models. Adv Neural Inf Process Syst 2023;36. arXiv:2306.17361 [A]
- Chen Y, Li C, Ouyang J, Xu G. DIF statistical inference without knowing anchoring items. Psychometrika 2023;88(4):1097–1122. doi:10.1007/s11336-023-09930-9 [C]
- Crawford CM, Park JJ, Chow S-M, Ernst AF, Pipiras V, Fisher ZF. Penalized subgrouping of heterogeneous time series. Multivariate Behav Res 2026;61(4):579–602. doi:10.1080/00273171.2026.2622120 [C]
- Danaher P, Wang P, Witten DM. The joint graphical lasso for inverse covariance estimation across multiple classes. J R Stat Soc B 2014;76(2):373–397. doi:10.1111/rssb.12033 [C]
- Driver C, Voelkle M, Oud H. ctsem: continuous time structural equation modelling. R package version 3.11.1; 2026. doi:10.32614/CRAN.package.ctsem (in the report)
- Driver CC, Oud JHL, Voelkle MC. Continuous time structural equation modeling with R package ctsem. J Stat Softw 2017;77(5). doi:10.18637/jss.v077.i05 [C]
- Du X, Urnes Johnson S, Epskamp S. The invariance partial pruning approach to the network comparison in time-series and panel data. Psychol Methods 2026;31(4):749–767. doi:10.1037/met0000824 [C]
- Epskamp S. Psychometric network models from time-series and panel data. Psychometrika 2020;85(1):206–231. doi:10.1007/s11336-020-09697-3 [C]
- Fiebig DG, Keane MP, Louviere J, Wasi N. The generalized multinomial logit model: accounting for scale and coefficient heterogeneity. Mark Sci 2010;29(3):393–421. doi:10.1287/mksc.1090.0508 [C]
- Fisher ZF, Kim Y, Fredrickson BL, Pipiras V. Penalized estimation and forecasting of multiple subject intensive longitudinal data. Psychometrika 2022;87(2). doi:10.1007/s11336-021-09825-7 [C]
- Gross SM, Tibshirani R. Data shared lasso: a novel tool to discover uplift. Comput Stat Data Anal 2016;101:226–235. doi:10.1016/j.csda.2016.02.015 [C]
- Guo Z, Kang H, Cai TT, Small DS. Confidence intervals for causal effects with invalid instruments by using two-stage hard thresholding with voting. J R Stat Soc B 2018;80(4):793–815. doi:10.1111/rssb.12275 [C]
- Halpin PF. Differential item functioning via robust scaling. Psychometrika 2024;89(3):796–821. doi:10.1007/s11336-024-09957-6 [C]
- Hodgson P, Jordan A, Sinani C, Charura D, Hodgson A, Glandorf HL. Longitudinal dynamics of physical function with anxiety and depression in Parkinson's disease: a cross-lagged panel analysis of the PPMI dataset. Brain Behav 2026;16(2):e71257. doi:10.1002/brb3.71257 [C]
- Huang B, Zhang K, Zhang J, Ramsey J, Sanchez-Romero R, Glymour C, Schölkopf B. Causal discovery from heterogeneous/nonstationary data. J Mach Learn Res 2020;21(89):1–53 (in the report)
- Huang X, Zhang H. Detecting responsible nodes in differential Bayesian networks. Stat Med 2024;43(17):3294–3312. doi:10.1002/sim.10125 [C]
- Jiang Z, Zhang D. Analysis of variance of multiple causal networks. Adv Neural Inf Process Syst 2023;36:11580–11591. doi:10.52202/075280-0510 [C]
- Kang H, Zhang A, Cai TT, Small DS. Instrumental variables estimation with some invalid instruments and its application to Mendelian randomization. J Am Stat Assoc 2016;111(513):132–144. doi:10.1080/01621459.2014.994705 [C]
- Kenny EA, Brand BA, Ebrahimi OV, McCutcheon RA. Dopamine antagonists and dynamic symptom networks in schizophrenia. Br J Psychiatry 2026:1–7. doi:10.1192/bjp.2026.10775 [C]
- Kuiper RM, Ryan O. Drawing conclusions from cross-lagged relationships: re-considering the role of the time-interval. Struct Equ Modeling 2018;25:809–823. doi:10.1080/10705511.2018.1431046 (in the report)
- Li A, Jaber A, Bareinboim E. Causal discovery from observational and interventional data across multiple environments. Adv Neural Inf Process Syst 2023;36. doi:10.52202/075280-0741 [C]
- Malik V, Bello K, Ghoshal A, Honorio J. Identifying causal changes between linear structural equation models. Proc UAI 2024, PMLR 244:2383–2398 [P]
- Manomaisaowapak P, Songsiri J. Joint learning of multiple Granger causal networks via non-convex regularizations: inference of group-level brain connectivity. Neural Netw 2022;149:157–171. doi:10.1016/j.neunet.2022.02.005 [C]
- Mooij JM, Magliacane S, Claassen T. Joint causal inference from multiple contexts. J Mach Learn Res 2020;21(99):1–108 (in the report)
- Mulder JD, Hamaker EL. Three extensions of the random intercept cross-lagged panel model. Struct Equ Modeling 2021;28(4):638–648. doi:10.1080/10705511.2020.1784738 [C]
- Ollier E, Viallon V. Regression modelling on stratified data with the lasso. Biometrika 2017. doi:10.1093/biomet/asw065 [C]
- Perry R, von Kügelgen J, Schölkopf B. Causal discovery in heterogeneous environments under the sparse mechanism shift hypothesis. Adv Neural Inf Process Syst 2022;35:10904–10917. arXiv:2206.02013 [A]
- Rouanet A, Philipps V, Taddé BO, Pérès K, Proust-Lima C. Modeling temporal relationships between multivariate repeated markers along with clinical endpoints: application to Alzheimer's disease and related dementias. Biom J 2026;68(4):e70157. doi:10.1002/bimj.70157 [C]
- Ruer A, Loranchet T, Bystrova D, Assaad CK. Root cause analysis via difference graph discovery from linear time-series data. arXiv:2608.21117 (2026) [A]
- Schroeders U, Zimmermann J, Wicke T, Schaumburg M, Lang E, Trenkwalder C, Mollenhauer B. Dynamic interplay of cognitive functioning and depressive symptoms in patients with Parkinson's disease. Neuropsychology 2022;36(4):266–278. doi:10.1037/neu0000795 [C]
- Seo G, Shim J, Kim M, Byun H, Jung Y, Song K. Dissecting causal mechanism shifts via FANS: function and noise separation. Proc ICML 2026, PMLR 306:109074–109104 [P]
- Shah RD, Samworth RJ. Variable selection with error control: another look at stability selection. J R Stat Soc B 2013;75(1):55–80. doi:10.1111/j.1467-9868.2011.01034.x [C]
- Skripnikov A, Michailidis G. Joint estimation of multiple network Granger causal models. Econom Stat 2019;10:120–133. doi:10.1016/j.ecosta.2018.08.001 (in the report)
- Swait J, Louviere J. The role of the scale parameter in the estimation and comparison of multinomial logit models. J Mark Res 1993;30(3):305–314. doi:10.1177/002224379303000303 [C]
- van Borkulo CD, van Bork R, Boschloo L, Kossakowski JJ, Tio P, Schoevers RA, Borsboom D, Waldorp LJ. Comparing network structures on three aspects: a permutation test. Psychol Methods 2023;28(6):1273–1285. doi:10.1037/met0000476 [C]
- Wang Y, Squires C, Belyaeva A, Uhler C. Direct estimation of differences in causal graphs. Adv Neural Inf Process Syst 2018;31. arXiv:1802.05631 (in the report)
- Windmeijer F, Farbmacher H, Davies N, Davey Smith G. On the use of the lasso for instrumental variables estimation with some invalid instruments. J Am Stat Assoc 2019;114(527):1339–1350. doi:10.1080/01621459.2018.1498346 [C]
- Wysocki A, McCarthy I, van Bork R, Cramer AOJ, Rhemtulla M. Cross-lagged panel networks. advances.in/psychology 2025;2(1):e739621. doi:10.56296/aip00037 [C]
- Xu Y, Düker M-C, Matteson DS. Testing simultaneous diagonalizability. J Am Stat Assoc 2024;119(546):1513–1525. doi:10.1080/01621459.2023.2202435 [C]

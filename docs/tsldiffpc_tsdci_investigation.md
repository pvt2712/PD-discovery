# tsLDiffPC and tsDCI: the framework, how it behaves on our design, and how to use it

> Investigation note, 3 November 2026: a second, deeper round on the difference-graph methods for time series of Ruer, Loranchet, Bystrova & Assaad (2026). It follows the Step 2 methods note (`step2_comparison_methods.md`) and feeds protocol v0.4. It rests on five sources: the paper, read in full; the authors' code (pyCIPHOD, MIT licence), read line by line but not run; the authors' benchmark, reconstructed from their generator; a re-implementation validated against the paper's published results; and a fifth synthetic check on the cohorts of our earlier checks. No PPMI data were read.

## Summary

**What it is.**
- tsLDiffPC and tsDCI find where two regimes' linear time-series mechanisms differ, without learning either regime's graph.
- **Difference edges.** A difference edge, X(t−1) → Y(t) or X(t) − Y(t), is kept only if the regression coefficient differs between the regimes for every conditioning set.
- **Root causes.** The series whose incoming coefficients change are the "effect-defying root causes".
- **Orientation of same-time edges.** tsDCI orients them by whether a residual variance is the same in both regimes; tsLDiffPC uses colliders and Meek's rules.
- **Guarantees.** They assume perfect tests, and there is no control of error rates.

**What the code does.**
- The conditioning sets range over *every* other node of the two-slice window, current-time nodes included.
- Both regression directions are tested, and an edge is removed at the first set with p > 0.05.
- There is no multiplicity correction, and rows are treated as independent.
- Because the allowed sets do not depend on the current graph, tsLDiffPC and tsDCI return the same skeleton. Only their orientation of same-time edges differs.

**A validated re-implementation.**
- Rebuilt from the authors' code, it reproduced the paper's main results (strict root-cause F1, Figure 4) within Monte Carlo error at every setting. The paper used only 10 datasets per setting.
- It runs in a fraction of a second per fit (median 0.2 seconds across settings up to nine series). The authors' code fits each test through data-frame operations.

**On our design** (the fifth check; 600 synthetic cohorts, the same as in the Step 2 note).
- **Speed only.** The search rarely invented cross-links (0.7% of shared links). But it flagged persistence, each domain's own year-to-year carry-over, in 47% of domains, so 48% of domains were named false root causes.
- **One real link difference.** Power was low: 49% for a new link (79% with conditioning sets capped at three nodes) and 10% for a stronger link.
- **Re-targeted against a speed-matched reference** (TD's mechanism run at each PIGD patient's own speed), false root causes fell to 0.2%, but power fell to 35% and 4%.
- **The paper's real data show the same signature.** Almost every change the methods found, in IT monitoring and in an intensive-care record, was a series' own persistence.

**How we use it.**
1. **As named comparator C5 in the methods test (H-M1),** with fixed settings, and a fidelity check of our re-implementation against the authors' code.
2. **Its vocabulary, extended, in the per-domain verdicts.** An effect-defying root cause is a domain with mechanism deviations; a noise-defying one is a domain whose noise differs beyond speed. Speed and load are added, which the framework lacks.
3. **Not as a test of record.** Its search is too lenient to have power at PPMI's sizes. It works on the yearly scale, so it reads speed as changed persistence. It assumes independent rows. It has no term for speed, load or patient-level burden.
4. **Not as an exploratory check either,** even re-targeted, because power stays low.

## 1. The framework

### 1.1 Setting and target

**Model.** The model is a linear discrete-time dynamic structural causal model (DT-DSCM). Each series Y at time t is a linear function of its parents up to a maximum lag (instantaneous parents allowed), plus independent noise.

**Regimes.** There are two regimes, "normal" and "anomalous", each with time-invariant coefficients and noise. The regimes are known in advance.

**Target.** The target is the full-time difference graph: an edge X(t−ℓ) → Y(t) exists if and only if that coefficient differs between the regimes. It is represented in a window of ℓmax + 1 time slices. By stationarity, one decision about a pair applies to all its copies in time.

**Root causes.** They are defined at the level of the whole series:
- **effect-defying:** an incoming coefficient changes, including the series' own lag;
- **noise-defying:** its noise distribution changes.

The methods target effect-defying root causes only. A pure noise change leaves the coefficients on the full parent set unchanged, so coefficient tests cannot see it.

**Outputs.**
- **Ĉ:** the heads of directed difference edges, the "certain" root causes.
- **Ĉ_P:** pairs from undirected same-time edges, each containing at least one root cause.

### 1.2 Assumptions

| # | Assumption | Needed by |
|---|---|---|
| — | Linearity | all |
| A1 | Causal sufficiency: independent noises, no hidden confounding | all |
| A2 | Stationarity within each regime | all |
| A3 | Every anomaly comes from a change in the model | all |
| A4 | One topological order shared by both regimes (no edge reversals) | all |
| A5 | Diff-adjacency-faithfulness: a changed coefficient differs for every conditioning set | all |
| A6 | Diff-orientation-faithfulness, for colliders and non-colliders | tsLDiffPC, tsLDiffPC² |
| A7 | Var-orientation-faithfulness: a changed coefficient changes residual variances for every set (the time-series analogue of DCI's assumption) | tsDCI, tsDCIPC |
| A8 | The anomalous graph is a subgraph of the normal one | tsLDiffPC², tsDCIPC |
| A9 | Faithfulness in the normal regime | tsLDiffPC², tsDCIPC |

A5 and A6 are strictly weaker than the diff-faithfulness of LDiffPC (Bystrova & Devijver 2026).

### 1.3 Algorithms and guarantees

**tsLDiffPC.**
- **Skeleton:** regression-coefficient equality tests over conditioning sets of increasing size.
- **Lagged edges:** oriented by time.
- **Same-time edges:** unshielded colliders, then Meek's rules restricted to orientations that respect time.

**tsDCI.**
- **Skeleton:** the same tests.
- **Same-time edges:** oriented by invariance of residual variances (F-tests).

**tsLDiffPC² and tsDCIPC.** They also copy orientations learned by a tiered PC on the normal regime.

**tPCUnion** is a naive baseline. It learns each regime's graph and tests coefficients on the union.

**Guarantees.**
- They hold only with perfect equality tests: a correct skeleton, correct returned orientations, and Ĉ ⊆ C.
- There are no finite-sample results, no error-rate control and no runtime figures.

### 1.4 The authors' benchmark and applications

**Benchmark.** Reconstructed from the generator; the paper states few of these details.
- **Series and edges.**
  - d = 3, 5, 7 or 9 series.
  - Lag-1 edges with probability 0.3, plus a self-lag for every series.
  - Same-time edges with probability 0.3 along a random order.
- **Coefficients.** ±U(0.2, 0.8), with each target's same-time and lagged coefficients each capped at a total of 0.95.
- **Noise and series.** N(0, 1) noise in both regimes; two independent series of 1,000 points after a burn-in of 200; no check of stationarity.
- **Change models.** Only same-time coefficients change, in three models:
  - one edge;
  - all same-time parents of a node;
  - all parents of a node with at least two.

  Each changed coefficient is redrawn as ±U(0.2, 1.0) with a change of at least 0.5.
- **Truth.** The truth is one series. Strict F1 = 2/(1 + |Ĉ|) when that series is in Ĉ.
- **Runs.** Ten datasets per setting; α = 0.05; conditioning sets unbounded.

**Reported results** (Figure 4).
- tsDCI had strict F1 0.60–0.90 throughout.
- tsLDiffPC had about 0 for single-edge changes, 0.17–0.50 for all-parents changes, and 0.70–0.83 for nodes with at least two parents.

Because the changes are same-time, the root cause is found only if the same-time edge gets oriented. tsLDiffPC cannot orient a lone changed edge.

**Real data.**
- **IT monitoring** (eight series; 1,000 normal and 100 anomalous points). All four methods returned:
  - two persistence changes (CMB, PMDB);
  - one undirected same-time edge (ESB − RTMB).
- **An intensive-care record** (nine signals; regimes split at a changepoint).
  - The changes found were the persistence of CVP, RR, HR and ABP, plus CVP → PAP.
  - There was no ground truth.
- In both, nearly every detected change is a series' own lag. The authors do not comment on this.

### 1.5 Limits

**Stated by the authors:**
- the assumptions in the table above;
- a cost that "increases rapidly" with the number of series and lags;
- known regimes;
- point anomalies out of scope.

**Not mentioned:**
- many short series or panels;
- hidden confounding beyond the assumption;
- time-scale or speed changes;
- noise-defying root causes in practice;
- control of false discoveries.

## 2. What the code does

pyCIPHOD 0.1.0 (beta), `ts_diff_constraint_based.py`, read line by line.

**Candidates.**
- Every pair of window nodes with at least one node at the current time: lag → current pairs, self-lags included, and current − current pairs.
- Lag–lag pairs are never candidates.

**Conditioning sets.**
- All other window nodes, not only current neighbours. A lagged link can therefore be conditioned on current-time values of other series.
- Sizes go up to 2d − 2 by default.

**The test.** A two-sample Wald z-test of the OLS coefficient: y on [1, x, S] in each regime, with classical standard errors. Rows are treated as independent.

**Removal rule.**
- An edge goes at the first set, in either regression direction (y on x, or x on y), with p > α = 0.05.
- With up to 2^(2d−2) sets per direction, a true difference must reach significance about 2,000 times in a row at six domains to be kept.

**One skeleton for both.**
- The allowed sets never depend on the current graph, so each candidate edge is decided on its own. tsLDiffPC and tsDCI return the same skeleton; only the recorded separating sets differ.
- tsDCI pairs the two regression directions set by set. tsLDiffPC goes level by level.

**Orientation.**
- Lagged edges are oriented by time.
- tsDCI tests residual-variance equality over the paired power sets of the two endpoints. tsLDiffPC applies colliders, then Meek's rules restricted to time.
- Membership of an undirected edge is checked as an ordered pair. Whether the graph class stores both orders could not be checked, because that file was not saved. Our re-implementation uses the paper's symmetric meaning.

**Other details.**
- Background knowledge is a stub.
- tsiSCAN is disabled.
- The README's reproduction command fails as written.
- Failed fits are skipped, and missing values can be handled test by test (`twd`).

## 3. A validated re-implementation

`scripts/toy/ts_ldiffpc.py` implements three parts of the published algorithms:
- the skeleton, in each algorithm's loop order;
- tsDCI's variance orientation;
- tsLDiffPC's collider and Meek orientation, with the root-cause sets.

Every test is computed from Gram matrices rather than by refitting through data frames. The algebra is identical, and a fit takes a fraction of a second (median 0.2 seconds across settings up to nine series).

`scripts/toy/toy_tsldiffpc_validation.py` rebuilds the authors' benchmark from their generator and scores strict root-cause F1 as they do, over 30 datasets per setting. The paper used 10.

| Algorithm, change model | d = 3: ours (SE) / reported | d = 5 | d = 7 | d = 9 |
|---|---|---|---|---|
| tsDCI, one edge | 0.78 (0.07) / 0.83 | 0.81 (0.07) / 0.60 | 0.63 (0.09) / 0.73 | 0.70 (0.09) / 0.80 |
| tsDCI, all parents | 0.76 (0.07) / 0.83 | 0.82 (0.07) / 0.67 | 0.96 (0.03) / 0.79 | 0.70 (0.09) / 0.90 |
| tsDCI, at least two parents | 0.77 (0.07) / 0.65 | 0.73 (0.08) / 0.87 | 0.63 (0.09) / 0.69 | 0.91 (0.05) / 0.77 |
| tsLDiffPC, one edge | 0.06 (0.04) / 0.17 | 0.01 (0.01) / 0.00 | 0.00 (0.00) / 0.03 | 0.00 (0.00) / 0.00 |
| tsLDiffPC, all parents | 0.14 (0.06) / 0.17 | 0.37 (0.09) / 0.17 | 0.30 (0.09) / 0.33 | 0.37 (0.09) / 0.50 |
| tsLDiffPC, at least two parents | 0.66 (0.09) / 0.72 | 0.66 (0.09) / 0.77 | 0.73 (0.08) / 0.83 | 0.83 (0.07) / 0.70 |

**Agreement.**
- With 10 datasets, the reported values carry standard errors of about 0.10–0.15. Every gap is within about 1.5 combined standard errors. Over the 24 settings, the mean absolute difference from the reported values was 0.10 (largest 0.21).
- The re-implementation reproduces the paper's pattern. tsDCI is high throughout. tsLDiffPC cannot name a root cause behind a single changed same-time edge, but does well when colliders form.

**Still to do.** A fidelity check against the authors' package itself (20 datasets per scenario) needs pyCIPHOD installed, and that waits for approval (protocol v0.4, D23).

## 4. How it behaves on our design

`scripts/toy/toy_tsldiffpc.py` runs on the same 600 synthetic cohorts as the Step 2 note's fourth check:
- six domains;
- 800 TD and 250 PIGD patients with five yearly transitions each;
- each patient with their own speed, plus a load difference.

The scenarios:
- S0: equal speed;
- S1: PIGD twice as fast;
- S2: S1 plus a PIGD-only link, axial → independence;
- S3: S1 plus function → independence stronger in PIGD.

Root causes are the targets of retained lagged edges, because in this design the phenotypes differ in lagged effects. The three uses:
- **T1:** the method as published, with TD and PIGD transitions as the two regimes and every conditioning set.
- **T1c:** the same, with conditioning sets capped at three nodes, as needed at ten domains.
- **T2:** PIGD's transitions against a reference sample, run with every conditioning set. The reference is TD's mechanism, fitted on TD patients only, run at each PIGD patient's own speed (drawn from its posterior given the whole trajectory), with PIGD's load and noise.

| | T1, as published | T1c, sets capped at 3 | T2, against a speed-matched reference |
|---|---|---|---|
| S0: cohorts with any cross-link flagged | 2.5% | 3.0% | 0% |
| S0: domains named root causes | 0.5% | 0.6% | 0% |
| S1: shared cross-links flagged | 0.7% | 1.3% | 0.1% |
| S1: domains whose persistence was flagged | 47% | 48% | 0% |
| S1: domains named root causes (false) | 48% | 49% | 0.2% |
| S2: power for the new link; it alone among cross-links | 49%; 49% | 79%; 76% | 35%; 35% |
| S3: power for the stronger link | 10% | 11% | 4% |
| S2 and S3: other domains named root causes | 52%; 50% | 52%; 51% | 0.2%; 0.2% |
| Tests per cohort (median, S1) | 6,745 | 1,674 | 52 |

**What this shows.**
1. **Speed is read as changed persistence.**
   - At twice the speed, each domain's yearly carry-over falls the most, so the search keeps self-lag differences.
   - It names about half of the domains root causes when no mechanism differs. This is the same signature as in the paper's real data.
2. **The search is lenient, so it seldom invents cross-links, and it seldom finds real ones.**
   - A difference must stay significant for every one of about 2,000 conditioning sets in both directions. Conditioning on current-time values of other domains inflates standard errors through collinearity.
   - Even the real new link was missed half the time, and a strength difference beyond speed nearly always.
   - Capping the sets at three nodes gives fewer chances to remove an edge. Power for a new link then rose to 79%, while false cross-links stayed near 1%.
3. **Re-targeting fixes the false root causes but not the power.**
   - Against TD's mechanism run at PIGD's own speeds, persistence no longer differed and false root causes vanished.
   - PIGD's speed estimate absorbed part of the real difference (median 2.2 against a true 2), and the lenient search did the rest: power was 35% and 4%.
4. **For comparison, on the same cohorts:**
   - the jointly fitted continuous-time per-link tests flagged 0.25–1.0% of shared links, and isolated the true link in 89–94% of cohorts;
   - the speed-adjusted per-domain test named the changed domain alone in 93–96% (Step 2 note).

## 5. The framework, element by element, against our design

| Element | In the paper | Under our design (many short panels; phenotypes as regimes) | What we do |
|---|---|---|---|
| Regimes known in advance | A changepoint or a label | Phenotypes are known at baseline (landmark) | Same, but phenotypes come from graph variables: the extended definition-matched null handles that |
| One long stationary series per regime | Yes | About 1,400 patients with 2–9 yearly visits; pooling patients' transitions breaks independence and stationarity | Participant-level likelihood with each patient's own speed and load |
| Causal sufficiency (A1) | Assumed | Violated by patient-level burden, speed and load | Random speed and load; latent-aware Step 1 |
| Equal noise (tsDCI orientation) | Holds in their benchmark | Speed changes yearly noise | Noise modelled per phenotype and scaled by speed |
| Effect-defying root cause | Any changed incoming coefficient, own lag included | Under speed, every domain qualifies | Per-domain test of deviations *beyond speed* |
| Noise-defying root cause | Defined, never detected | Under speed, every domain qualifies | Noise difference beyond speed, in the domain verdict |
| Load (intercept) | Absent from the model; a level shift is absorbed by intercepts | Phenotypes differ in load | Load is its own component of the domain verdict |
| Window graph, time orientation | Lagged edges oriented by time | Our two-slice tables are the same window | Step 1 uses tiers; Steps 2 to 5 need no orientation search |
| Coefficient-equality test | Every conditioning set, any non-rejection removes | Low power; yearly scale | One joint continuous-time fit; each deviation tested with all others free |
| Error control | None | A study needs calibrated tests | Parametric bootstrap inside the extended null; Benjamini–Hochberg; Holm for the primary tests |

## 6. How to apply it in the experiment

### 6.1 As named comparator C5 in H-M1

**Input.**
- Per phenotype, two-slice tables of yearly transitions: lag block X(t−1) and current block X(t), on the domains' copula scales.
- Complete transitions, or test-by-test deletion of missing values (`twd = True`).
- Rows pooled over patients and treated as independent, exactly as the method assumes.

**Settings.**
- α = 0.05 (the paper's value), with 0.01 and 0.1 as sensitivity.
- Conditioning sets unbounded at six domains, and capped at three at ten domains (D23).

**Outputs scored.**
- Retained lagged difference edges, mapped to links: self-lags, direct links, and links absent from the drift.
- Same-time difference edges, reported separately.
- Ĉ and Ĉ_P, scored as false root-cause domains under shared mechanisms, and as power and localisation with a real difference (protocol §8.5).

**Software.**
- pyCIPHOD at a pinned commit, installed with approval, for a fidelity check against our re-implementation on 20 datasets per scenario.
- The re-implementation then carries the full benchmark.

**Expectation, from this check:**
- few false cross-links;
- false root causes in about half of the domains under speed;
- low power.

### 6.2 What Step 2 adopts

- **The root-cause vocabulary, extended.** Each domain's verdict (protocol §7.7) says whether it is:
  - an effect-defying root cause *beyond speed* (mechanism deviations into the domain);
  - a noise-defying root cause *beyond speed*;
  - a domain that differs only in load, or only through the common speed.

  This is the clinicians' question, "which domain is driven differently?", without speed's false positives.
- **The assumption list as a checklist.** Sufficiency, stationarity, a shared order and the faithfulness variants name what the engine must handle or test: burden, speed–stage correlation, and identification.
- **The "re-target, then test" idea, done in continuous time.** The comparison is against TD's mechanism run at PIGD's speed. The test is a joint fit with every deviation estimated together, not a set-by-set search.

### 6.3 What Step 2 does not adopt

- **The search itself as a test of record.** It is too lenient for PPMI's sizes, uncontrolled for multiplicity, and its p-values treat rows as independent.
- **Yearly-scale testing.** Speed changes every yearly coefficient, persistence most. One real continuous-time difference moves several yearly entries.
- **Variance-based orientation.** Speed changes yearly noise by design. Orientation in our setting comes from time and from Step 1.

### 6.4 Options considered and set aside

1. **The re-targeted search as an exploratory check (T2).** It is calibrated but has too little power (35% and 4%). It adds nothing over the per-link and per-domain tests from the joint fit.
2. **A tiered version,** with conditioning sets limited to the lag block. This is the "difference-graph search with time tiers" of the fourth check. It has more power, but under speed it flagged 55% of shared links, so it is a comparator, not a tool.
3. **Regimes within patients,** for example yearly transitions before and after starting dopaminergic treatment. The framework was built for a known within-series change, so the question fits. But treatment start follows symptoms (confounding by indication), most patients have only one or two transitions on each side, and the treatment effect is pharmacological rather than a change in disease mechanism. It is not part of Part I. If wanted, it belongs with the treatment-effect study (protocol §12.6), with g-methods.

## 7. Changes made in protocol v0.4

- **§8.4, comparator C5** (tsLDiffPC and tsDCI):
  - settings as in §6.1;
  - expected behaviour as found here;
  - the fidelity check against pyCIPHOD before the re-implementation carries the benchmark.
- **§2.4, H-M1.** Counts false root-cause domains as well as false links.
- **§8.5.** Adds the rate of false root-cause domains, and performance by type of link (persistence, direct, absent).
- **§7.2, methods table.** Records why the search is not used as Step 2 machinery.
- **§7.3.4.** Maps the root-cause vocabulary onto the per-domain verdicts.
- **Appendix E, check 5.** Summarises the results.
- **Decision D23.** Covers installing pyCIPHOD at a pinned commit.

## References

- Bystrova D, Devijver E. Constraint-based difference graph discovery in a linear setting. arXiv:2606.13947 (2026).
- Ruer A, Loranchet T, Bystrova D, Assaad CK. Root cause analysis via difference graph discovery from linear time-series data. arXiv:2608.21117 (2026); ECML-PKDD CAESAR workshop. Code: pyCIPHOD (github.com/CIPHOD/pyCIPHOD), version 0.1.0, MIT licence.
- Wang Y, Squires C, Belyaeva A, Uhler C. Direct estimation of differences in causal graphs. Adv Neural Inf Process Syst 2018;31. arXiv:1802.05631.

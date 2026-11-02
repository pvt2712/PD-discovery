# Paper 2, experiment 1: causal discovery for levodopa escalation — protocol

*Pre-registration, drafted 29 October 2026 and approved by the investigator on 1 November 2026 (§12); the approved
text is kept unchanged in `docs/PROTOCOL_approved_2026-11-01.md`. Experiment 2 (causal inference: what escalation changes, why experience misleads,
hidden-reason bounds, subgroups) is `../escalation-study-v2`; this experiment estimates no effect. Changes after
approval go to §12 with a date. The decisions behind this draft (expert DAG, cohort, location, installs, and the scope
of §10.2) were taken on 29 October 2026; the inventory of earlier work it reuses is `docs/inventory.md`.*

## 1. Questions

- **D1 (Arm A, recoverability).** On data that look like the pooled PPMI PD cohorts (sample size, variable types,
  repeated annual visits, missing values, a latent non-motor trait), with the expert DAG as the data-generating truth:
  which causal-discovery learners recover the structure that matters for the escalation decision, and under which
  realistic stresses do they fail?
- **D2 (Arm B, agreement and plausibility).** On PPMI itself, where no ground truth exists: does the data contradict
  the expert DAG; which learned relations are stable under resampling; how often does each learner assert what is
  impossible (negative controls); and do clinicians, blinded, read the disagreements between data and experts as
  plausible hypotheses or as artifacts?
- **D3 (hand-over).** Which graph, and which parents of the dose decision, go to the inference experiment.

The decision rule (§8) is fixed here: the learner, or ensemble, whose graph is reported and handed over is chosen by
pre-specified Arm A and Arm B criteria, and a deep-learning or pretrained learner replaces the classical anchor only if
it beats it in Arm A.

## 2. The expert DAG

**v5.1 (`v51_escalation`, `docs/expert_dags.yaml`)**, the escalation-focused induced subgraph of the evidence-graded
v4.1 progression model (ppmi-causal, 15 October 2026), identical in ppmi-causal, escalation-study (v1) and the
manuscript repository. It is the Arm A truth and the first thing Arm B tests.

- 23 observed nodes: 17 time-varying (levodopa, agonist, MAO-B and amantadine doses; months on levodopa; cumulative
  levodopa; OFF non-axial motor; OFF axial; MDS-UPDRS II; Schwab & England; wearing-off; dyskinesia; Epworth;
  impulse-control domains; MoCA; GDS-15; hallucinations) and 6 baseline (age, sex, education, APOE ε4, baseline DaT
  binding, disease duration at enrolment); one latent person-level non-motor trait.
- 75 edges: 45 same-year, 7 lagged, 15 from baseline nodes, 8 from the trait. Grades are inherited from v4.1
  (15 guideline, 35 strong, 18 moderate, 5 structural, 1 weak, 1 exploratory); 12 edges carry an "unsupported"
  note from earlier tests.
- Earlier test (ppmi-causal, in-sample clustered GCM): 24.8 / 28.0 / 22.4% of 701 implied independences violated in
  PPMI 1.0 / 2.0 / genetic PD, against 38.6 / 36.0 / 33.4% for within-tier relabellings (p = 0.0099 in each).

**Two-slice layout** (as in ppmi-causal `expert.unroll`): baseline nodes → year *t* (suffix `_0`) → doses at *t*+1 →
the rest of year *t*+1 (`_1`). Every time-varying node depends on its own previous value; a same-year edge from a
non-dose node into a dose becomes "X at *t* → dose at *t*+1" (the dose recorded at *t*+1 is a decision taken after
visit *t*); years since enrolment (`t`) points into every time-varying node. The pooled analysis adds two cohort
context nodes, `era2` (PPMI 2.0 era) and `genetic` (genetic PD arm), which may point into any node and receive none
(joint causal inference, Mooij, Magliacane & Claassen, JMLR 2020). The table therefore has **43 observed variables**:
2 context, 5 exogenous (age, sex, education, APOE ε4, `t`), 2 baseline disease, 17 at *t*, 17 at *t*+1.

**v2's 13-node core map** (`docs/core_map.yaml`, copied from escalation-study-v2) is tested in Arm B only (§7.1): its
11 measured nodes, several of them two-column level-and-change nodes, and two latents without proxies make it too
coarse for a simulation truth.

## 3. Cohort and data

- **Pooled PD cohorts** (decision of 29 October 2026): PPMI 1.0 de novo sporadic PD untreated at enrolment (431),
  PPMI 2.0 sporadic PD (957), genetic PD (353): **1,741 participants**, 8,829 annual-grid rows (8.9 / 3.1 / 5.9 per
  participant). PD only: controls and prodromal participants are never pooled in, because PD enrolment required a
  dopamine-transporter deficit and pooling across enrolment criteria conditions on enrolment. Cohort differences
  enter through the context nodes.
- **Annual grid** (v2 `build.py`, copied unchanged): for each participant and year, the visit within ±4 months of the
  anniversary with the most complete core battery; MDS-UPDRS IV is 0 before any PD treatment. **Two-slice rows:**
  consecutive grid years of one participant (year *t* and *t*+1). Participants are the unit of every resampling.
- **Missing values** (share of grid rows; PPMI 1.0 / 2.0 / genetic, v2 `results/build_missingness.csv`): OFF motor
  exam 30 / 18 / 43%, axial 28 / 17 / 41%, MoCA 6 / 7 / 12%, Part IV items 4–8%, questionnaires 3–7%, doses and
  levodopa history 0%. In ppmi-causal's PPMI 1.0 two-slice table 64% of transitions had a missing value, so no
  learner may drop incomplete rows silently (§5.3).
- **Not available:** CSF and plasma biomarkers (the local PPMI download has no biospecimen category).
- **Sensitivity analyses (secondary, Arm B; `escd arm-b --subset ... --methods selected`, `--one-per-person`):** PPMI
  1.0 alone (the homogeneous de novo cohort); the pooled cohorts minus one cohort at a time; one transition per
  participant (an i.i.d. check for the learners that assume it). Each repeats the expert DAG's test (§7.1) and the
  learner selected in the main analysis (§8), with its knowledge-free variant, on 50 subsamples; a selected ensemble
  brings its members. A context node that a subset fixes (`genetic` without the genetic cohort; `era2` and `genetic`
  in PPMI 1.0 alone) is left out of that run, from the data and from the expert DAG: a fixed root node has no edges,
  so the subset's DAG is the DAG without it, not the DAG with it hidden. Any other variable without variation stops
  the run (in a subsample, that subsample's fits count as failed) instead of being dropped. The design's suggested
  de novo baseline cross-section is not used: it holds no dose nodes, and the dose decision is the subject of this
  experiment.

## 4. Variables

| Node | Two-slice columns | Type | Tier | Source (v2 build) |
|---|---|---|---|---|
| era2, genetic | `era2`, `genetic` | binary | context | enrolment year ≥ 2020; genetic-arm enrolment |
| age, male, educ, apoe4 | same | continuous, binary, count, binary | exogenous | `age_enroll`, `male`, `educ_years`, `apoe4` |
| t | `t` | count (years) | exogenous | grid year of the row |
| dat0, dur0 | same | continuous | baseline disease | putamen SBR at baseline; months since diagnosis |
| ldopa, da, maob, ama | `_0`, `_1` | mg/day LEDD, point mass at 0 | *t*; doses at *t*+1 | `ledd_levodopa`, `ledd_agonist`, `ledd_maob`, `ledd_amantadine` |
| t_ldopa, cum_ldopa | `_0`, `_1` | months; cumulative dose, mass at 0 | *t*; *t*+1 | `months_on_ldopa`, `cum_ldopa` |
| motor_na, axial | `_0`, `_1` | ordinal sums | *t*; *t*+1 | OFF MDS-UPDRS III without / with only axial items |
| adl, se | `_0`, `_1` | ordinal | *t*; *t*+1 | MDS-UPDRS II; Schwab & England |
| fluct, dysk | `_0`, `_1` | ordinal (0–8) | *t*; *t*+1 | MDS-UPDRS 4.3–4.4; 4.1–4.2 |
| ess, cog, dep | `_0`, `_1` | ordinal | *t*; *t*+1 | Epworth; MoCA; GDS-15 |
| icd_core, hall | `_0`, `_1` | count; ordinal (0–4) | *t*; *t*+1 | QUIP core domains; MDS-UPDRS 1.2 |
| trait | — | latent | — | person-level non-motor trait |

**Tiers given to every learner that accepts knowledge:** context | exogenous | baseline disease | year *t* | doses
at *t*+1 | the rest of *t*+1. An edge from a later tier into an earlier one is forbidden; edges among context nodes and
among exogenous nodes are forbidden; exogenous nodes may be caused by context nodes only. Within a tier the data
decide. FCI receives these as tiered-FCI semantics (an arrowhead at the later node, the earlier mark left open; Bang &
Didelez, CLeaR 2025), with tails at context and exogenous nodes (`knowledge.py`).

## 5. Methods

### 5.1 Panel

| Family | Learner (role) | Implementation | Output | Knowledge | Missing values |
|---|---|---|---|---|---|
| Classical | **FCI + clustered GCM** (anchor A1) | causal-learn 0.1.4.8; test `cgcm` = v2 `citest.ClusterGCM` (additive splines, cross-fitted over participants, CR1); separating sets capped at 3 in both search stages (§10.2) | PAG | tiers | test-wise |
| Classical | FCI, PC + Fisher-z (A2, A3: the textbook i.i.d. test) | causal-learn `mv_fisherz`; FCI capped as A1 | PAG, CPDAG | tiers | test-wise |
| Classical | BOSS, BIC (A4) | causal-learn | CPDAG | none (not supported) | imputed |
| Classical | BOSS, DG-BIC (A5) | Tetrad 7.6.10 algcomparison via JPype, OpenJDK 21 (degenerate-Gaussian BIC for mixed data) | CPDAG | tiers | imputed |
| Continuous optimisation | NOTEARS-linear (control C1; also tiered as zero bounds) | reference algorithm (Zheng et al. 2018), `methods/continuous.py` | DAG | none (tiered variant: bounds) | imputed |
| Continuous optimisation | DAGMA-linear (C2) | dagma 1.1.1 | DAG | none | imputed |
| Baselines | sortnregress, R²-sortnregress (B1, B2) | Reisach et al. 2021, 2023 | DAG | none | imputed |
| Neural | SDCD (N1); DiBS edge marginals (N2) | `sdcd`; `dibs-lib` (side environments) | DAG | none | imputed |
| Pretrained | AVICI `scm-v0` (P1), CDFM (P2), Arrow (P3) | `avici`; `cdfm-base`; `arrowfm` (side environments) | DAG | none | imputed |
| Ensemble | family-majority vote (M1) | `methods/ensemble.py` | PAG | — | — |

Every learner that accepts knowledge also runs without it (the `_noknow` variants): in Arm A's base settings and in
Arm B's negative controls. Every learner runs in Arm B; the deep-learning learners (N1, N2, P1–P3) run in Arm A's
realistic settings only (§6.3). NOTEARS, DAGMA and the sortability baselines are **controls**, never candidates for
selection: on standardised data the first two lose the variance cue that orients them, and the baselines show how
much of a learner's score the sortability of the data alone explains (with a varsortability report).

### 5.2 Environments

Main environment: Python 3.12 (`uv.lock`). Learners that pin other Python versions run as workers in their own uv
environments (`envs/`, `workers/`, `methods/external.py`): causica 0.4.6 on 3.10 (DECI: installed and checked, not in
the panel, §5.5), avici 1.0.7 and sdcd 0.1.4 on 3.11, dibs-lib 1.3.3, cdfm-base 0.1.0 and arrowfm 0.1.0 on 3.12;
pretrained weights (AVICI `scm-v0` 68 MB, CDFM 38.7 MB,
Arrow 147 MB, from Hugging Face) are stored in `envs/` and loaded offline. A worker receives the shared imputed table
and returns a weight or edge-probability matrix; it runs single-threaded by default (`ESCD_WORKER_THREADS`), because
several learners run at once; temporary files live under `data/derived/tmp/` and are deleted after each fit.

Worker check (`scripts/check_workers.py`, a linear-Gaussian graph with 10 nodes and 10 edges, 1,000 rows; AUROC of
the skeleton / of the directed edges): SDCD 1.00 / 0.96, AVICI 1.00 / 0.92, CDFM 1.00 / 0.96, Arrow 1.00 / 0.96,
DiBS 0.95 / 0.84, DECI 0.83 / 0.79. DECI's augmented Lagrangian had not converged after causica's 2,000 epochs
(14.5 min on that small graph; about 0.46 s per step at 7,000 rows x 43 variables); capped at 5,000 optimiser steps it
still took over an hour per fit at the pooled size (§10.1), so it is not in the panel (§5.5).

### 5.3 Preprocessing (the same for every learner of a kind)

- Conditional-independence learners: Gaussian-copula normal scores, test-wise deletion.
- Learners that need complete rows: one chained-equation imputation per dataset (scikit-learn `IterativeImputer`,
  Bayesian ridge, 10 rounds), shared by all of them; Tetrad's discrete variables (at most 5 levels) are rounded back
  to observed levels.
- Scale-sensitive learners (NOTEARS, DAGMA, the neural and pretrained learners) receive z-scores, except
  in the units stress setting (raw clinical units). Variance-sortnregress reads raw units by definition.
- Sensitivity (Arm A `listwise` setting): listwise deletion for the complete-row learners.

### 5.4 Tuning (Arm A only)

Each learner has a grid of at most **6 configurations** (α for tests; penalty discount for scores; λ and threshold for
continuous optimisation; thresholds, steps or particles for neural and pretrained learners; `methods/*.py`). Every
configuration runs on **5 tuning replicates of each generator's base setting** (seeds disjoint from the evaluation
seeds); the configuration with the highest mean adjacency F1 (§6.4) is frozen for all of Arm A and for Arm B. A
knowledge-free variant uses its learner's configuration. No configuration is ever chosen on PPMI: tuning on real data
against the expert DAG would teach the learners to copy the experts.

### 5.5 Not included

- **LLM prior:** excluded unless approved separately; its agreement with the expert DAG would be inflated (LLMs read
  the literature the experts used), and it could only ever see variable names and definitions.
- **VAMBN** as the second generator: it needs R/bnlearn and a TensorFlow 1.x HI-VAE, and its modules are latent
  embeddings, which blur a variable-level truth. G2 (§6.1) takes its role: the same DAG with a different function
  class.
- **BayesDAG** (Python 3.8 research code), **DCDI** (no maintained package), **CUTS+** (needs long series; the
  authors call 21 time points too short) and **CSF/plasma nodes** (not available).
- **Dropped for their cost at the pooled size (29 October 2026, §10.2; the adapters stay in the repository):** PC
  with the clustered GCM (≈ 42 min per fit); Tetrad's tiered BOSS-FCI, GFCI, GRaSP-FCI, FCI and PC (out of memory on
  16 GB); DECI (over an hour per fit with training capped at 5,000 steps, without converging; causica also has no
  missing-data mask and assumes no latent confounders); J-PCMCI+ (more than 29 min per fit, full cost unmeasured;
  tiered FCI on the two-slice table already orders the variables by visit).

## 6. Arm A: recoverability

### 6.1 Generators

The expert DAG run forward year by year for each simulated participant (`generator.py`). Every node is a
Gaussian-copula variable: its latent score is a function of its parents' **observed** values on the normal-score
scale, plus the trait, optional person effects and noise; the observed value is the pooled marginal's quantile at
Φ(latent). Parents act through their observed values, so the expert DAG over the observed variables is exactly true,
and the marginals reproduce every variable's type (binary, ordinal sums, counts, continuous, doses with a point mass
at zero). Latent scales are calibrated on a pilot cohort so that the pooled marginals match.

- **G1:** linear in the parents' scores, Gaussian noise.
- **G2:** gradient-boosted regressions (depth 3, 200 rounds, learning rate 0.05, 50 rows per leaf) with resampled
  out-of-fold residuals.
- **Fitting (after approval):** on the pooled PPMI cohorts (`escd fit-generator`): marginals (level frequencies or
  201 quantiles), equations per node for the enrolment year and for later years with the cohort context saturated,
  trait loadings from a one-factor model on the participants' mean residuals of the trait's children (and baseline
  DaT), the cohort mix, follow-up and the missingness pattern bank. Acceptance check: median absolute difference of
  the Spearman correlations of real and simulated tables on the scored pairs ≤ 0.05; otherwise the discrepancy is
  reported and the generator used as fitted (no tuning of the generator towards any learner).
- **Synthetic specification** (`generator.synthetic`, for building and testing the harness without PPMI): the same
  DAG with effects drawn in [0.08, 0.35] (70% positive), PPMI-like marginals, cohort mix and follow-up; G2 replaces
  the linear terms with non-monotone transforms and Gaussian noise with t(5) noise.

### 6.2 Missingness

- **Realistic (MAR):** masks resampled per participant-year from PPMI's own patterns, stratified by cohort, years
  since enrolment (0, 1–2, 3–5, 6+) and levodopa status (the OFF exam is skipped on treatment), so joint patterns
  (one exam, several nodes) are kept; follow-up length from the cohort's distribution.
- **MNAR:** MAR plus value-dependent missingness of the assessments a worse state plausibly prevents (OFF motor exam,
  axial score, MoCA, GDS-15, Epworth: extra probability 0.25 in the worse quartile) and informative dropout (after a
  year with Schwab & England below 80%, each later year is lost with probability 0.25).
- **Complete:** no missing values (diagnostic).

### 6.3 Settings

| Setting | Changed from base | Replicates | In the decision set R |
|---|---|---|---|
| base | 1,741 participants, realistic MAR, trait latent, z-scores | 20 | yes |
| hide1 | OFF non-axial motor hidden (both years) | 20 | yes |
| mnar | MNAR missingness | 20 | yes |
| n300, n431, n1000, n4500 | participants | 10 each | no |
| hide2 | OFF non-axial motor and wearing-off hidden | 10 | no |
| complete | no missing values | 10 | no |
| raw | raw clinical units for the scale-sensitive learners | 10 | no |
| person | per-node person effects (stable person heterogeneity) | 10 | no |
| listwise | the complete-row learners drop incomplete rows instead of imputing | 10 | no |

Each setting runs under G1 and G2: 24 settings, 300 datasets (120 of them in R). The hidden nodes follow a pre-specified rule
(`expert.hide_rule`): the measured non-dose node with the most children in the expert DAG (ties: most dose children),
which for v5.1 is OFF non-axial motor, then wearing-off. Knowledge-free variants run in the two base settings. The
deep-learning learners run in R only (Plan B, §10.2), and so the ensemble, of which they are members, is scored in R
only.

### 6.4 Truth and metrics (`truth.py`, `metrics.py`)

**Truth.** A two-slice table hides the trait, so its variables follow a maximal ancestral graph (MAG); with the tiers,
the best any learner can recover is the tiered PAG. Both come from the two-slice DAG with the latent trait, without a
search: the MAG skeleton from ancestral separating sets, its marks from ancestral relations, the PAG from causal-
learn's FCI orientation rules on that skeleton (checked against brute-force FCI with a d-separation oracle,
`tests/test_truth.py`).

**Scored pairs.** Pairs with a year-*t*+1 node and no context node (498 of 903 for v5.1). Excluded: pairs inside
year *t* (their dependence runs through hidden earlier years); the 46 pairs that only the hidden earlier years connect
(adjacent in the MAG of the DAG with one latent earlier year, more years add none); and, for the recoverable truth,
pairs whose generating edge is near zero (effect strength below 0.05: the square root of the share of the node's
latent variance the input explains, by permutation on a pilot cohort; the same definition for G1 and G2).

| Metric | Definition | Use |
|---|---|---|
| **Adjacency F1** | against the MAG skeleton on the scored pairs | **primary** |
| Arrowhead precision, recall, F1 | invariant marks of the tiered PAG, on pairs adjacent in both graphs; arrowheads where the PAG has a circle reported as over-claims | key secondary |
| Decision-parent recall and precision | possible parents of the next-year levodopa dose against its MAG parents | key secondary |
| Parent-set F1 | same, averaged over ldopa_1, adl_1, dysk_1, fluct_1, se_1 | secondary |
| SHD | pairs whose adjacency or marks differ from the PAG | secondary |
| SID (latent-aware) | for each dose at *t*+1 and each other *t*+1 node, is the learned parent set a valid adjustment set in the true generating DAG (generalised adjustment criterion; pairs with no valid observed set excluded); CPDAG/PAG outputs give an interval | secondary |
| Arrowhead F1 against the MAG | every mark definite | secondary |
| Expert-edge recall | the expert's edges, in their *t*+1 pair, by kind | descriptive |

### 6.5 Analysis

Per setting: mean and replicate standard deviation of each metric per learner. Paired comparisons use the same
datasets for all learners. **Generator dependence:** per setting, the Spearman correlation of the learners' mean
adjacency F1 under G1 and G2; below 0.5 the ranking is reported as generator-dependent and the decision (§8) uses
only verdicts that agree under both generators.

### 6.6 Stability threshold for Arm B

On 2 replicates of each base setting, anchor A1 runs on 25 participant subsamples (90%; Plan B); for τ in {0.5, …, 0.9} the
precision of the pairs selected at frequency ≥ τ is computed against the truth. The smallest τ with mean precision ≥
0.8 is used in Arm B (0.9 if none reaches it).

## 7. Arm B: PPMI

### 7.1 The expert maps are tested before any learner runs

- **v5.1** on the pooled two-slice table (`falsify.py`): for every pair the DAG separates even with the earlier years
  hidden, one implied independence given a minimal observed separating set (338 statements; separating sets of median
  size 8); the cohort context is saturated (cohort differences are not the expert's claims). On data simulated from
  v5.1 itself, 7% of a sample of these statements were rejected at α = 0.05, near the nominal 5%. Test: clustered GCM (splines, cross-fitted over participants, CR1),
  α = 0.05; untestable statements counted and left out. Statistic: share violated. Null: **200** relabellings that
  permute the time-varying labels within groups (doses among doses, the rest among the rest, the same permutation in
  both years); p = (1 + #null ≤ observed) / (1 + 200). Edge support: each expert edge, in its *t*+1 pair, tested
  given the child's other observed parents. The PPMI 1.0-only version is a sensitivity analysis.
- **Core map, decision node (v2's Q1, moved here):** each guideline reason and analysis block tested against the
  escalation decision on v2's emulated-trial rows, per cohort, exactly as v2 pre-specified it (`audit.py`, copied;
  test-wise deletion primary; reading rule of v2 DESIGN.md §4.1). v2's existing Q1 tables are not touched.
- **Core map, whole map (v2 DESIGN.md §6.2, moved here):** its implied independences among measured nodes, with the
  two latent nodes kept latent, tested on the same rows with block tests (all columns of a two-column node jointly;
  Bonferroni over the second node's columns), against within-role relabellings; the number of testable implications
  is reported with and without the latent nodes.

### 7.2 Stability

Every learner with its frozen configuration on the full table and on **50 participant subsamples** (Plan B; 90% of each
cohort, without replacement: subsampling frequencies are far better calibrated than bootstrap ones, Banerjee, Andrews
& Kummerfeld 2025). A pair is stable at frequency ≥ τ (§6.6); its marks are the full-data fit's (the modal marks if
the full fit lacks it). Reported: stable graph, median subsample frequency of the full-data pairs, failed fits. The
ensemble is rebuilt on every subsample from its members' graphs (its knowledge-free counterpart from the
knowledge-free variants, for §7.4), so it is assessed like any learner.

### 7.3 Agreement with the expert DAG

Each expert edge: found with compatible marks, found reversed or bidirected, or not found; each stable pair the
expert lacks: data only. This is agreement, not accuracy.

### 7.4 Negative controls

Every learner **without knowledge**, on the full table and the 50 subsamples. Errors: (i) an arrowhead into a
context node, or into an exogenous node (sex, APOE ε4, age, education, years since enrolment) from a non-context node;
(ii) a cross-tier edge oriented later → earlier (a bidirected cross-tier edge encodes confounding and is not counted;
later o→ earlier is reported as a possible violation). Rate: (ii) over the cross-tier edges oriented in either
direction, with the 2.5th–97.5th percentiles over subsamples.

### 7.5 Clinician review

A blinded sheet (`arm_b.clinician_sheet`) of every disagreement of the selected learner's stable graph with v5.1
(data-only pairs; expert edges not found), mixed with as many control items (half confirmed expert edges, half
tier-respecting pairs neither source proposes), shuffled, phrased "Could X (this year) plausibly influence Y (next
year)?", without source or statistics. At least two movement-disorder neurologists sort each item independently into
plausible new hypothesis / likely artifact / cannot judge; agreement by Cohen's κ; disagreements adjudicated by
discussion. The control items calibrate the reviewers: the share of confirmed expert edges they call plausible, and
the share of unproposed pairs they call artifacts.

## 8. Decision rule

1. **Arm A score:** mean adjacency F1 over the realistic settings R (base, hide1, mnar under G1 and G2: 120 datasets).
2. **Best anchor:** the classical anchor (A1–A5, primary variants) with the highest score.
3. **Beating the anchor:** a contender (neural, pretrained, ensemble) beats it if its
   paired mean difference over the 120 datasets exceeds 0.02 and the lower limit of its 95% replicate-bootstrap
   interval exceeds 0.
4. **Arm B eligibility:** median subsample frequency of its full-data pairs ≥ 0.6, and the 97.5th percentile of its
   knowledge-free backwards rate below 0.5 (orientation better than chance).
5. **Selection:** among the eligible best anchor and the eligible contenders that beat it, the highest Arm A score.
   If no deep-learning or pretrained learner beats the anchor, the anchor is selected. Controls and baselines are
   never selected.

Everything else (other metrics, settings, sensitivities) is reported beside the selection and never changes it.

## 9. Hand-over to the inference experiment

One aggregate file (`results/handover/ppmi/to_inference.yaml`, `escd handover`):
1. v5.1's test: violation rate, null, p, and the support of each edge into the next-year levodopa dose;
2. the selected learner's stable graph and the possible parents of the next-year levodopa dose, each mapped to v2's
   pre-decision covariate and flagged when v2's 43-covariate set lacks it (a data-revealed parent);
3. members of the set the graph places after the decision (none expected: all are measured before it);
4. the trait indicators the data support (for v2's Q4 benchmarks).
What experiment 2 does with these is decided there; v2's `validity.py` controls (positive control, MoCA negative-
control outcome, future-escalation negative-control exposure) and its planted-effect plasmode stay with experiment 2.

## 10. Run order and compute

```
uv sync
uv run escd build && uv run escd trials          # PPMI -> data/derived (gated)
uv run escd fit-generator                        # G1, G2, pattern bank (gated)
uv run escd arm-a --source fitted --tune         # tuning on simulated data
uv run escd arm-a --source fitted                # Arm A
uv run escd tau --fitted                         # stability threshold
uv run escd audit                                # Q1, moved from v2 (gated)
uv run escd core-map                             # the whole core map (§7.1; gated)
uv run escd arm-b --data ppmi --source fitted    # Arm B (gated)
uv run escd arm-b --data ppmi --source fitted --methods selected --subset ppmi1
                                                 # sensitivity runs (§3); likewise --subset no_ppmi1,
                                                 # no_ppmi2, no_genetic, and --one-per-person
uv run escd handover --data ppmi
uv run pytest -q
```

Compute (Apple M4, 10 cores, 16 GB): see §10.1, measured with `scripts/timing.py` on synthetic base-size data.

### 10.1 Measured cost (29 October 2026)

One synthetic dataset at the pooled base size (1,741 participants, 7,220 two-slice rows), one fit at a time with
other measurements running beside it (roughly the conditions of a parallel run), single-threaded side workers.
Files: `results/timing_*.csv`, `results/test_counts*.csv`, `results/worker_check.csv`.

| Learner (with tiers unless noted) | Wall time per fit | Note |
|---|---|---|
| FCI + clustered GCM, conditioning sets capped at 3 | 17.5 min | uncapped: not finished after 84 CPU-min |
| FCI + Fisher-z, capped at 3 | 4.1 min | uncapped: not finished after 67 CPU-min |
| PC + clustered GCM (uncapped) | ≈ 42 min (estimated) | 356,297 tests × 6–8 ms per cross-fitted test |
| PC + Fisher-z (uncapped) | 10.9 min | 356,297 tests; causal-learn's PC has no depth cap |
| BOSS (causal-learn); Tetrad BOSS | 3 s; 4 s | |
| Tetrad BOSS-FCI | out of memory | 8 GB and 12 GB heaps; untiered: 2.4 min, 6.7 GB |
| Tetrad GFCI | out of memory | after 47 min (8 GB); untiered also out of memory |
| Tetrad PC (DG-LRT) | out of memory after 63 min | untiered: 10 s |
| NOTEARS; tiered NOTEARS; DAGMA | 52 s; 7 s; 3 s | |
| sortnregress; R²-sortnregress | < 1 s | |
| SDCD; DiBS; AVICI; CDFM; Arrow | 11.4; 16.8; 15.7; 24.7; 2.6 min | single-threaded |
| DECI (5,000-step cap) | > 60 min (timed out) | 0.46 s per step uncontended |
| J-PCMCI+ | > 29 min | stopped before it finished (1,597 participant series, 15 years, 9 context nodes) |
| Clustered-GCM test (falsification) | 0.03 s per statement | 338 statements × 201 relabellings ≈ 35 min |

**What this means.** FCI's possible-d-separation stage tries every subset of each edge's possible-d-sep set; on the
dense two-slice graphs at the pooled sample size that never finishes, with any test. Capping the separating sets at
size 3 (in the adjacency search and in the possible-d-separation stage, as Tetrad's depth setting does) makes the
paper's own anchor run in 17.5 min. Tetrad's FCI-type learners exhaust memory with the tiers at this size on a 16 GB
machine. DECI cannot be trained to convergence at any affordable budget here.

Before tuning, a trivial dense baseline scored well: variance-sortnregress reached adjacency F1 0.70 (385 edges)
against 0.46–0.63 for the anchors at default settings. The baselines stay unselectable, but every contender's Arm A
score is to be read against them.

### 10.2 Scope (decided 29 October 2026, recorded in §12)

From the measurements of §10.1 the investigator chose Plan B with the recommended options:

1. **FCI's separating sets are capped at 3**, in the adjacency search and in the possible-d-separation stage, for both
   FCI anchors (A1, A2). Without a cap the latent-aware anchor cannot run at the pooled size.
2. **Tetrad: the tiered BOSS only** (A5). Its tiered BOSS-FCI, GFCI, GRaSP-FCI, FCI and PC exhaust memory at the
   pooled size on 16 GB (§5.5); the optional untiered BOSS-FCI sensitivity is not run.
3. **DECI is dropped** (§5.5): no missing-data mask and no convergence at an affordable budget.
4. **PC with the clustered GCM is dropped** (≈ 42 min per fit); FCI with the same test and PC and FCI with Fisher-z
   remain.
5. **J-PCMCI+ is dropped** (§5.5): more than 29 min per fit with its full cost unmeasured, and tiered FCI on the
   two-slice table is already ordered by visit. Keeping it in R and Arm B would have added at least 14 hours.
6. **Plan B.** The deep-learning learners run in Arm A's realistic settings R only, and so does the ensemble (§6.3);
   Arm B uses 50 subsamples (§7.2); τ is calibrated on 2 replicates of each base setting × 25 subsamples (§6.6). The
   Arm B sensitivity runs of §3, which the first estimate left out, repeat only the expert DAG's test and the selected
   learner (with every learner they would add about 31 h). Per base-size dataset the panel costs about 1.75 CPU-hours
   in R (1.2 of them for the five deep-learning learners) and 0.55 elsewhere; the knowledge-free variants add about
   0.55 in the base settings and in Arm B. Estimated machine time on this Mac with 8 parallel jobs:

   | Stage | Plan A: full design (not chosen) | **Plan B (chosen)** |
   |---|---|---|
   | Tuning (60 fits per learner) | ≈ 18 h | ≈ 13 h |
   | Arm A (300 datasets) | ≈ 100 h | ≈ 45 h |
   | τ calibration | ≈ 16 h (10 datasets × 50) | ≈ 3 h (4 × 25) |
   | Arm B, main analysis | ≈ 42 h (101 fits per variant) | ≈ 13 h (51) |
   | Arm B sensitivity runs (§3), selected learner | ≈ 15 h | ≈ 8 h |
   | Falsification, Q1, core map | ≈ 2 h | ≈ 2 h |
   | **Total** | **≈ 8 days** | **≈ 3.5 days** |

   Plan A kept PC with the clustered GCM, ran the deep-learning learners in every setting and used 100 Arm B
   subsamples; Plan C (not chosen) was Plan A on a compute cluster, parallel by dataset and subsample (about 18 h on
   100 cores). Estimates assume cost grows with the number of rows to the power 1.5 for the constraint-based learners
   and linearly for the rest; the sensitivity runs assume the costliest anchor, FCI with the clustered GCM, is
   selected (a selected ensemble brings its members: about 31 h).

## 11. Data governance and the approval gate

- PPMI data-use agreement: participant-level tables, fitted generators, pattern banks, simulated panels derived from
  them, worker files and every per-row array stay in `data/derived/` (gitignored); `results/` and `docs/` hold
  aggregates only (pair-level frequencies, counts, rates, metrics); nothing leaves the machine; no PPMI content goes
  to any external service; the pretrained models run locally.
- **Gate:** every command that reads PPMI participant data (`build`, `trials`, `audit`, `fit-generator`, `arm-a
  --source fitted`, `arm-b --data ppmi`) stops unless `ppmi_runs_approved = true` in `pyproject.toml`. It is false
  until this protocol is approved; the date of the change goes to §12.
- Nothing is committed to git without the investigator's instruction.

## 12. Amendments

- 29 October 2026: protocol drafted; harness built and tested on synthetic data only.
- 29 October 2026, before approval: the scope of §10.2, chosen by the investigator from the measured costs of
  §10.1: Plan B with the recommended options. FCI capped at 3 in both search stages; Tetrad reduced to the tiered
  BOSS; DECI, PC with the clustered GCM and J-PCMCI+ dropped (§5.5); the deep-learning learners, and so the
  ensemble, in Arm A's realistic settings only; 50 Arm B subsamples; τ calibrated on 2 replicates × 25 subsamples;
  the Arm B sensitivity runs restricted to the expert DAG's test and the selected learner (the first estimate had
  left them out). Anchors relabelled A1–A5. Expected machine time on this Mac: about 3.5 days. The approval gate
  stays closed (§11).
- 31 October 2026, before approval: a context node that a sensitivity subset fixes is left out of that run (§3).
  Found on synthetic data, when the test of the `no_genetic` run did not finish: with `genetic` constant, PC's
  Fisher-z tests are undefined, so the node is never separated, and PC, which has no depth cap, tries every subset of
  the other 41 variables as a separating set for each of its edges. The approval gate stays closed (§11).
- **1 November 2026: protocol approved by the investigator.** `ppmi_runs_approved` set to true; the approved text is
  kept in `docs/PROTOCOL_approved_2026-11-01.md` (SHA-256 7012bf5a…550ebb). Clerical, the same day: the §10 run
  order gains `escd core-map`, which §7.1 already requires; the Q1 audit and the core-map test run before Arm A
  rather than after it (they are independent of Arm A, so the order changes no result).
- 1 November 2026, after approval, an implementation fix: the core-map test (§7.1) stopped in its first cohort on a
  singular matrix in `citest.block_test`, inside the joint coefficients that function also returns for description
  and that the test does not use. When a block column has no residual variation, those coefficients are now left
  empty. The test's p-values are unchanged: such a block was already untestable (p = NaN) and is counted as §7.1
  says. The core-map test was rerun the same day.

## Appendix A. What the pasted design got wrong (checked 29 October 2026)

- J-PCMCI+ is Günther, Ninad & Runge, UAI 2023, arXiv 2306.12896 (arXiv 2310.01753 is CausalTime). AVICI is arXiv
  2205.12934. BOSS-FCI: the Tetrad manual says it tends to be more accurate than GFCI, not faster (Ramsey, Andrews &
  Spirtes, arXiv 2510.04263).
- The "1,693 enrolled — 798 PD, 721 at-risk, 174 controls" split has no source found; the 4,500 target is
  NCT04477785. This protocol uses its own counts.
- arXiv 2507.00173 (Pal, Ghosh & Yang) had 2,924 biomarkers but n = 199 and ran no FCI on PPMI.
- DECI's maintained package (causica 0.4.6) has no missing-data mask and assumes no latent confounders.
- CUTS+ targets irregular, not dense, series; it still needs long ones.
Full report: `docs/design_claims_check.md`.

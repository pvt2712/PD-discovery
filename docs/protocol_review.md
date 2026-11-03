# Review of the Option B/C protocol draft v0.2 and its critical review

> 2 November 2026. Reviewed: `Parkinsons_Option_B_C_Research_Protocol_Revised_v0_2.md`, the review-integrated Option B/C protocol (here "B/C v0.2"), and `Parkinsons_Protocol_Critical_Review_and_Improvement_Plan.md`, the critical review of v0.1 that it implements (here "the critique"). Both were compared against the phenotype-discovery protocol draft v0.2 and the pilot evidence from the escalation-discovery experiment. The revision is `PROTOCOL.md`, draft v0.3; section numbers below refer to it.

## 1. Verdict

- **Sound framing.** B/C v0.2 takes in most of the critique faithfully and is a sound programme-level framing. It separates discovery, prediction and treatment effects. It names the right null, shared structure with different strengths. It adds subtype-only and shared-graph controls to the prognostic comparison, and it keeps validation locked and silent.
- **Not yet executable.** The subgroup, the primary method, the inference rule, the simulation's success criterion and the clinical endpoint are all still open. Eleven method families are listed without costs or feasibility evidence.
- **A gap in both documents.** Neither addresses the problem that matters most for the chosen focus. A clinical phenotype is computed from the very symptom scores the graph models, so splitting patients on it creates differences even when the mechanisms are shared. Until that artefact is handled, a test of "same structure, different strength" against "different structure" cannot be interpreted.
- **Two further gaps.** Either would undermine the results if left:
  - "different structure" is defined as exact zeros, which data cannot confirm;
  - the phenotype could end up used as an external context, which it is not.
- **Option C.** As written, it cannot be delivered within Paper 2. It is probably short of events in PPMI, and its step from graph to prediction is undefined.
- **Recommendation, carried out in v0.3.** Merge B/C v0.2 with the phenotype-discovery design:
  - Part I is the phenotype study, with the critique's points made concrete;
  - Option C becomes a conditional Part II, with an event check and a fallback endpoint.

## 2. What B/C v0.2 does well (kept in v0.3)

- It separates three questions that are often blurred: graph discovery, prognostic prediction and treatment effects. A discovered edge is never read as a treatment effect.
- The M0 / M1 / M2 vocabulary, and the insistence that a stronger coefficient is not a different graph.
- It keeps the ambiguity of partial graphs (PAG marks), resamples by participant, and does not treat selection frequency as a probability.
- Option C's five-model comparison, with subtype-only (C2) and shared-graph (C3) controls and placebo graphs.
- Locked-model external validation; silent prospective validation; model updating treated as new development.
- Death treated as a failure, never as missing data; state and event endpoints kept apart.
- Stop/go gates, a bias matrix, reporting standards, novelty discipline and minimum case-report fields.

## 3. Major issues

### 3.1 Phenotypes are computed from the variables in the graph (missing from both documents)

**Problem.** TD/PIGD is a ratio of tremor items to gait and balance items; the Fereshtehnejad subtypes threshold motor, cognitive, RBD and autonomic scores. Stratifying on such a rule is selection on the system being modelled. It restricts the range of the defining variables, creates collider bias between them, and produces regression to the mean. B/C v0.2 mentions an "expert causal selection DAG" and "alternative strata analyses", but neither removes the artefact. The critique's primary contrast, M1 against M2, assumes the groups themselves are innocent.

**Illustration.** Suppose all patients share the same mechanisms and scores carry measurement error.
- Patients classed as TD were selected partly because their year-0 tremor was measured high and their gait score low.
- A year later both drift back: tremor seems to improve and gait to worsen in TD, with the reverse in PIGD.
- A within-group analysis then reports "faster axial progression in TD", and links estimated from year-0 values are biased differently in each group.
- All of this is an artefact of the rule.

**Fix in v0.3.**
- A landmark. The phenotype is set at year 0 and the analysed transitions start at year 1, so no year-0 value, and none of its measurement error, enters an analysed row (§4).
- A definition-matched null, simulated with shared mechanisms and the same rule applied (§7.3).
- A permuted-label reference beside it. The gap between the two measures the artefact itself.
- H-M1 as the primary methodological test (§2.4).

Label permutation alone, as in the Network Comparison Test, is not a fix: labels computed from the variables are not exchangeable under the null (§1.3).

### 3.2 The phenotype must not be used as an external context

**Problem.** B/C v0.2 keeps J-PCMCI+ as a high-priority context method.
- Context-based methods treat group membership as set from outside and unconfounded with the system. These are joint causal inference under its usual assumptions, CD-NOD and J-PCMCI+.
- A phenotype is produced by the disease. It shares hidden causes, such as overall pathology and baseline state, with every later symptom.
- These methods would therefore report those shared causes as effects of the phenotype.

**Fix in v0.3.**
- The phenotype is a baseline-tier node in a latent-aware search (§1.3, §7.2). This is joint causal inference with only its exogeneity assumption, which the landmark makes true.
- Its edges signal heterogeneity and are never read as effects.
- Era (PPMI 1.0 or 2.0) can be a context.
- J-PCMCI+ is dropped; it also took more than 29 minutes per fit in the pilot.

### 3.3 "Different structure" cannot be shown by significance, and it depends on the time step

**Problem.**
- M2 is defined as supp(A₁) ≠ supp(A₂), a statement about exact zeros. Finite data cannot confirm a zero.
- "Significant in one group and not in the other" is a known fallacy (Gelman 2006). With a minority group of a few hundred, it mainly reflects power.
- Structure also depends on the interval. A sparse continuous-time system gives a dense annual transition matrix, because effects pass through intermediate steps within the year (Voelkle 2012; Kuiper 2018). An annual-lag "absent link" therefore means something different from an 18-month one.

**Fix in v0.3.**
- Explicit estimands per link: strength β, difference δ, level γ.
- A smallest effect of interest Δ and five classes (§2.3). A structural difference needs a significant difference plus an equivalence test showing the link negligible in one phenotype (Lakens 2018).
- An honest expectation: at PPMI's sizes, many links will be unresolved. They are reported with upper bounds, never as "absent", and WP3 measures how often each class can be reached.
- "Structure" is defined at the annual lag, relative to the measured domains.

### 3.4 Eleven method families, no primary method, no costs

**Problem.**
- §7.3 of B/C v0.2 calls hierarchical continuous-time SEM the "primary modelling candidate". §15 instead lists "PCMCI+ benchmark, LPCMCI sensitivity, J-PCMCI+ context, one irregular-time method".
- No method is costed, though the design multiplies fits: subsamples × phenotypes × null datasets × simulation scenarios.
- The pilot measured what these costs look like:
  - FCI took 17.5 minutes per fit on 43 columns, and J-PCMCI+ more than 29 minutes;
  - Tetrad's latent-aware learners ran out of 16 GB, and DECI did not converge;
  - eight parallel deep-learning jobs crashed the machine;
  - no tuning job finished in 12 hours.
- The same applies to our own v0.2. Evaluating Analysis B in simulation is a nested simulation, because each simulated dataset needs its own null. At full size that means millions of fits, and v0.2 had not costed it.

**Fix in v0.3.**
- One primary learner (tiered, FCI type) in two configurations: a cluster-aware test for single pooled fits, and a fast test inside resampling, where decisions are calibrated by simulation.
- One robustness learner and a sortability control.
- A reduced nested design for Analysis B.
- A compute budget of about 7 days at the target costs, capped at 10 days, with a set order of cuts (§8.8).
- A table giving each B/C method's place and the reason (§7.1).

### 3.5 Stable differences between people create lagged links

**Problem.** In pooled cross-lagged models, a stable person-level burden produces year-to-year associations between domains without any causal link (Hamaker 2015). B/C v0.2 mentions random-intercept cross-lagged models only as non-causal references.

**Fix in v0.3.**
- A hidden burden in every simulated scenario, so its bias is measured.
- A latent-aware learner, whose PAG marks flag possible confounding.
- The level-versus-strength classification.
- A random-intercept sensitivity analysis in Analysis C (§7.4).

### 3.6 PPMI is not irregularly sampled

**Problem.** Both documents take irregular visits as the main premise, which is why continuous-time SEM is primary and ReTimeCausal a conditional high priority. But PPMI's assessments sit on a regular annual grid, and its irregularity comes mostly from missed years. Continuous-time modelling earns its place elsewhere: comparing effects with cohorts that use other intervals (for example 18 months), which is Part II's replication.

**Fix in v0.3.**
- A pre-specified interval rule, decided from WP0's interval distribution (§6.3).
- Continuous-time SEM becomes a sensitivity analysis and the tool for replication.
- ReTimeCausal is dropped.
- Continuous-time SEM forests are dropped too. They split on covariates, so splitting on phenotype-defining variables would recreate the artefact of §3.1.

### 3.7 Inference and simulation are under-specified

- **Size and power are mixed up.** The primary simulation outcome is "false declarations … calibrated at a prespecified sensitivity/recovery target". A procedure runs at a fixed decision rule; its false-difference rate and its power are then measured separately. v0.3 states H-M1 with a pass criterion: at or below 5%, within two Monte Carlo standard errors.
- **Monte Carlo precision.** 200 repetitions give a standard error of 1.5 percentage points at 5%, too coarse to tell 5% from 8%. v0.3 gives the formula, sets replicate numbers per procedure and reports the errors (§8.6).
- **No benchmark safeguards.** B/C v0.2 has no ADEMP structure (Morris 2019), no simulator fidelity criterion and no sortability control. In the pilot, a sortability baseline beat the classical learners (F1 0.70 against 0.46–0.63), so a benchmark without that control can flatter methods.
- **No primary metric for Option C.** It lists calibration, net benefit and AUC for two contrasts. v0.3 has one primary metric, out-of-sample log loss, in an intersection–union test of C4 against C2 and C3, with net benefit as the key secondary.

### 3.8 Option C: scope, events and logic

- **Scope.** A prospective multicentre cohort needs sites, funding, ethics approval and at least 24 months of follow-up after the last enrolment: years of work. B/C v0.2's timeline admits this (C2 set-up in months 12–18) but still lists the cohort as a deliverable.
- **Events.** The endpoint is "alive and independent at 24 months". In early, research-enrolled PD, both death and loss of independence within 24 months are probably uncommon. B/C v0.2 notes this for deaths only; its source left PPMI out of a mortality analysis because fewer than 20 deaths had been recorded (Tan 2024). The development sample may be too small for adequate precision (Riley 2020).
- **Logic.** Prediction is a weak test of a causal map. A well-tuned history model usually matches a graph-based one, and a map can be right without improving prediction. A null result for C4 against C2 says little about Option B.
- **Undefined step.** "Subgroup-specific graph features" are never defined.

**Fix in v0.3.**
- Part II is conditional on Part I's gate, and II-a, external replication of the mechanisms, comes before prognosis.
- WP0 counts the events, and a fallback order is pre-specified: the PPMI milestone composite (Brumm 2023), then a 48-month horizon, then continuous outcomes.
- C3 and C4 are defined as forecasts from the shared and the phenotype-specific transition models, added to the history model.
- The prospective validation and the target-trial emulation are outlines for separate protocols (§12).

### 3.9 PPMI realities not addressed

B/C v0.2 is written for a generic cohort. v0.3 handles the PPMI-specific issues that the pilot measured or that the design implies:
- an untreated baseline, so the phenotype is assessed off medication;
- treatment starting during follow-up, a break in the motor scores;
- OFF exams missing in 18–43% of annual rows;
- MDS-UPDRS II items shared with the phenotype rule (2.10, 2.12, 2.13), removed from the daily-function node;
- different follow-up in PPMI 1.0 and 2.0, so findings are replicated by era;
- PPMI access already in place (B/C v0.2 says "access is not assumed"), while external cohorts are not yet accessible;
- the Data Use Agreement rule that no participant-level data go to any external service, including AI tools. B/C v0.2 only says to avoid "unapproved" AI services.

## 4. Issues found in our own phenotype-discovery v0.2

The merge also exposed weaknesses in our own draft, all fixed in v0.3:
- **No explicit estimand.** "Strength difference" and "structure difference" were not defined. A structural verdict rested on Analysis B plus "a supporting interaction", which is close to the significance fallacy of §3.3.
- **An uncosted nested simulation.** Its cost for Analysis B was not estimated, and there was no compute budget, although compute was the pilot's main failure.
- **No check of visit intervals,** and no plan for comparing cohorts that use other intervals.
- **Label noise and definition artefacts not separated.** The permuted-label control appeared only as a negative control.
- **Loose wording on joint causal inference.** v0.2 said it "assumes exogenous contexts" without saying which assumption fails. Exogeneity holds under the landmark; unconfoundedness does not.

## 5. Response to the critique, point by point

| Critique's recommendation | In v0.3 | Where |
|---|---|---|
| Structural versus coefficient heterogeneity (M0 / M1 / M2) as the primary scientific contrast | Adopted, with changes: classes per link with an equivalence margin, and definition artefacts added to the null. The primary clinical tests are strength differences (M1 against M0), the question PPMI can answer | §2.3, §2.4, §7.3–7.5 |
| Hierarchical continuous-time SEM as the primary model candidate | Not adopted as primary: a sensitivity analysis and the replication tool | §6.3, §12.3 |
| Random-intercept cross-lagged model as a non-causal comparator | Adopted as a sensitivity analysis | §7.4 |
| One pre-specified baseline segmentation; dynamic states separate | Adopted: TD/PIGD primary and one secondary axis; dynamic states and latent subtypes out of scope | §5, §11 |
| A selection DAG for the segmentation variables | Replaced by the landmark and the definition-matched null, which deal with the selection directly | §4, §7.3 |
| A shorter algorithm list, with PCMCI+ and Bagged-PCMCI+ mandatory | List cut further. PCMCI+ reduces to the pooled two-slice search used here; bagging is done as the participant bootstrap | §7.1, §7.8 |
| Formal inference on edge differences, not subtraction of selection frequencies | Adopted: the definition-matched null for Analysis B; interaction tests with false-discovery control for Analysis C | §7.3, §7.4 |
| Observation process and missing data | Adopted, with the PPMI specifics: OFF exams, dropout weights, a missing-not-at-random shift, nothing imputed after death | §7.9 |
| Measurement quality and a measurement-fidelity substudy | Adopted: shared items removed and an ON/OFF policy; the substudy in II-c | §5.4, §6.1, §12.5 |
| Primary simulation outcome: false structure under M1 | Adopted, with changes. H-M1 tests false differences under shared mechanisms (S0, S1, S4); false structure verdicts under M1 (S2, S2b) are a key performance measure. Size and power are reported separately | §2.4, §8.5 |
| Five-model prognostic comparison with placebo graphs | Adopted, with the step from graph to predictors defined and one primary metric | §12.4 |
| 24-month alive-and-independent state as the primary endpoint | Adopted provisionally, with an event check and a pre-specified fallback order | §12.4 |
| One named clinical action | Adopted provisionally: referral for a multidisciplinary functional and fall-risk assessment | §12.2 |
| Locked external and silent prospective validation | Kept; prospective validation outlined for a separate protocol | §12.4, §12.5 |
| Treatment effects by a separate target-trial emulation | Kept, as an outline | §12.6 |
| A novelty matrix, with candidate papers to verify | Adopted; all candidate papers verified. The closest work is association graphs and cross-sectional causal discovery | §1.4 |
| MCD as a secondary latent-subgroup analysis | Not adopted: weakly identified with 3 to 9 visits per person, and the pilot's deep-learning jobs exhausted memory | §7.1, §11 |

## 6. Minor and editorial issues in B/C v0.2

- The executive summary says algorithm benchmarking is "conditional on clinical-panel feasibility", which is unclear. In v0.3 the benchmark is governed by the simulation decision rule (§8.7).
- §1.3 (item 6) promises prospective validation as a contribution, although no positive finding is assumed. v0.3 makes it conditional.
- §6.1 asks for one primary phenotype definition, while §15 says "1–2 prespecified groupings". v0.3 has one primary axis and one secondary axis.
- §5.1 proposes about 8–12 domains, while the expert questionnaire rates 13. v0.3 has about ten core domains and up to three optional ones.
- §16 promises reference entries R19 onwards that were never added. v0.3 adds them, verified.
- R7 (ReTimeCausal) is correct as cited; "Irregularly Time Series" is the published title. The placeholder "CT-SEM forests (2025)" is Cáncer and colleagues, Psychological Methods 2025.

## 7. References

- **Existence.** Every reference and candidate paper checked exists; B/C v0.2's R1–R18 were essentially correct.
- **Claims checked.**
  - R13 left PPMI out of its mortality analysis because fewer than 20 deaths had been recorded (Tan 2024).
  - R14 defined dependency as needing help with basic activities of daily living, measured as Schwab & England below 80% (Macleod 2016).
  - The PPMI milestone set includes functional dependence at the same threshold (Brumm 2023).
- **Corrections.**
  - Penalized FCI is now published (Pal, Ghosh and Yang, Annals of Applied Statistics 2026), and it analyses cross-sectional biomarkers, not progression.
  - The critique's candidate papers now have verified details in v0.3's list: the 2024 graph-prognosis paper is AdaMedGraph (Lian 2024), the association-graph paper is Pan 2024 (AMIA), and TCD-Arena is Stein 2026 (ICLR).
- **Novelty.** Five searches (Europe PMC, arXiv and the web) found no study that learns causal graphs per motor phenotype or clinical subtype, or uses a subtype as a context, on longitudinal PD clinical data. That does not prove there is none; the WP1 search records its databases, queries and dates.

## 8. Decisions for the investigator

1. **One protocol or two.** v0.3 merges both v0.2 drafts into one protocol, with Part I as the phenotype study and Part II as the conditional translation. Keep a separate Option B/C document only if a programme-level proposal is needed for a supervisor or funder.
2. **Scope of Part II (D15).** v0.3 keeps Option C as an outline. Should II-a and II-b be specified fully now, or after Part I?
3. **Smallest effect of interest Δ (D12).** 0.10 is lenient and gives more verdicts; 0.05 is strict and leaves more links unresolved. The panel chooses.
4. **WP0 approval.** It is still needed before any PPMI data are read. WP0 supplies the questionnaire's group sizes, the interval rule and Part II's event counts.

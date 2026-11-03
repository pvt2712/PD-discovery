# Phenotype-specific progression mechanisms in Parkinson's disease: a causal-discovery study in PPMI

> Protocol draft v0.2, 2 November 2026. Design draft for review by the investigator, movement-disorder neurologists and a statistician. Not approved; no phenotype analysis of PPMI data has been run. It supersedes the Option B/C proposal draft v0.1 (2 November 2026) and the escalation-discovery experiment (approved 1 November 2026, stopped 2 November 2026), which serve as pilot evidence (§2.4, Appendix B). Thresholds marked *provisional* are fixed before pre-registration (§14).

## Summary

**Clinical question.** Clinicians already sort Parkinson's disease (PD) into phenotypes, such as tremor-dominant (TD) versus postural instability and gait difficulty (PIGD), or mild-motor versus diffuse-malignant, and know that some phenotypes progress faster. They do not know whether phenotypes decline through *different pathways* (for example, loss of independence driven by gait and cognition in one phenotype and by bradykinesia in another) or through *the same pathways at a different speed*. Guidance on monitoring and care (NICE NG71, the MDS evidence-based reviews, AAN 2021) is largely phenotype-independent. If pathways differ, what to monitor and when to refer should differ by phenotype; if only speed differs, the same plan applies at different intervals.

**Design.** A causal-discovery study of year-to-year transitions in PPMI's sporadic PD cohorts untreated at baseline (about 1,400 participants), with the phenotype assigned at baseline by pre-specified clinical rules. Three analyses answer the question at different levels of confidence:

- **A. All participants, phenotype as a baseline variable.** Finds the progression mechanisms that differ by phenotype and classifies each difference as one of level ("speed") or of strength ("pathway").
- **B. Separate graphs per phenotype, against a definition-matched null.** The null is the set of differences that would appear if all phenotypes shared the same mechanisms, produced by simulating patients and applying the same phenotype rule.
- **C. Confirmatory hypotheses.** About ten phenotype differences chosen with clinicians before any phenotype analysis, tested with false-discovery control.

A simulation benchmark fitted to PPMI establishes beforehand which procedure keeps false phenotype differences under control at PPMI's sample sizes.

**Why it is new.** A quick search (2 November 2026) found PPMI subtyping studies and causal discovery on PPMI biomarkers, but no subtype-specific causal discovery of clinical progression; a full search is part of WP1. The methodological point applies to any clinical cohort: phenotypes defined from the very variables in the graph create false "phenotype-specific" links unless they are compared against a definition-matched null.

**What clinicians get.** A progression map with shared, phenotype-specific and unresolved links; for each phenotype, the routes to loss of independence; and, domain by domain, a verdict of "speed" or "pathway". These are hypotheses for monitoring and counselling, not treatment rules.

## 1. Questions and hypotheses

### 1.1 Clinical questions

- **Q1 (shared or phenotype-specific).** Which year-to-year links between motor, cognitive, non-motor, treatment and functional domains do phenotypes share, and which differ?
- **Q2 (speed or pathway).** Where phenotypes differ, is it a difference of level (the same mechanism, shifted: "speed") or of strength or structure (a different mechanism: "pathway")?
- **Q3 (routes to loss of independence).** For each phenotype, which domains feed next year's daily function and independence?
- **Q4 (fit of the expert map).** Does a clinician-built progression map fit the data equally well in each phenotype?

### 1.2 Methodological question

- **M1.** At PPMI's sample sizes, how often does fitting a graph separately within each phenotype produce false phenotype differences, and does comparison against a definition-matched null keep them at the nominal rate while still detecting real ones?

### 1.3 Hypotheses

- **H-M1 (primary methodological test, simulation).** Under shared mechanisms, naive split-and-fit declares phenotype-specific links well above the nominal 5% (range restriction, collider bias from the phenotype rule, unequal group sizes); the definition-matched procedure stays at or below 5%.
- **H-C1 to H-C10 (primary clinical tests, PPMI).** About ten pre-specified phenotype differences, each a phenotype × link interaction with an expected direction (candidates in Appendix A), frozen with the clinical panel in WP1; Benjamini–Hochberg control at 5% across the family.
- **H-S (key secondary).** Most phenotype differences in progression are differences of level ("speed") rather than of strength or structure ("pathway"). The share in each class is estimated, not assumed.

## 2. Background

### 2.1 Phenotypes in practice

The TD/PIGD classification (Jankovic 1990; MDS-UPDRS version, Stebbins 2013) is the most widely used. PIGD is associated with faster motor, cognitive and functional decline, but the classification is unstable in early PD (Simuni 2016). Fereshtehnejad's clinical subtypes (mild-motor predominant, intermediate, diffuse malignant), defined from motor severity, cognition, REM-sleep behaviour disorder (RBD) and dysautonomia, separate progression rates in PPMI (Fereshtehnejad 2015, 2017). An MDS task force asked for subtypes that are reproducible and clinically or biologically meaningful (Mestre 2021). Data-driven progression subtypes have been found across cohorts (PPMI with PDBP, Dadu 2022; PPMI, ICEBERG and LuxPARK, npj Parkinson's Disease 2024). Biological classifications (SynNeurGe, Höglinger 2024; NSD-ISS, Simuni 2024) use α-synuclein seed amplification, which PPMI has measured (Siderowf 2023) but our local download does not hold.

### 2.2 The gap

Different average trajectories do not show different mechanisms. A phenotype that carries more disease burden will progress faster along the same pathways. The clinically useful distinction, speed or pathway, needs a structural (causal) analysis of how domains drive each other: split finely enough to see differences, and carefully enough not to invent them.

### 2.3 Why it is hard

1. **Phenotypes are defined from graph variables.** TD/PIGD is a ratio of tremor items to gait and balance items; Fereshtehnejad's subtypes threshold motor, cognitive, RBD and autonomic scores. Splitting patients by such a rule restricts the range of the defining variables and makes their causes dependent within each group (collider bias; Hernán 2004), so graphs fitted within phenotypes can differ even when the mechanisms do not.
2. **Phenotypes are not exogenous.** The disease produces them, so they cannot be treated as an external context (joint causal inference, Mooij 2020, and CD-NOD, Huang 2020, assume exogenous contexts). Here the phenotype is a baseline variable whose links to later change signal heterogeneity; it is never read as a cause.
3. **Hidden disease burden.** A person-level burden that drives many domains makes phenotypes differ in level everywhere; the design must tell this apart from different pathways.
4. **Few annual visits.** PPMI gives 3 to 9 annual visits per person. Time-series methods built for long, regular series (PCMCI+, Runge 2020; LPCMCI, Gerhardus 2020; J-PCMCI+, Günther 2023; regime methods) do not fit, so the unit of analysis is the year-to-year transition, pooled over participants.
5. **Treatment feedback.** Dopaminergic treatment responds to symptoms and changes them, so it is a node in the graph.
6. **Small groups.** The minority phenotype (PIGD, or diffuse malignant) may hold a few hundred participants, where an absent link usually means low power.

### 2.4 Pilot evidence from the escalation-discovery experiment

The stopped experiment used the same PPMI download. No code is reused; these are its lessons (details in Appendix B).

| Observation | Consequence for this design |
|---|---|
| 1,741 PD participants pooled; 8.9 / 3.1 / 5.9 annual rows per participant in PPMI 1.0 / PPMI 2.0 / genetic PD | PPMI 2.0 adds people but little follow-up, and PPMI 1.0 carries long-run progression: both are reported, and findings are replicated by era |
| OFF-state motor exam missing in 30 / 18 / 43% of annual rows; MoCA in 6 / 7 / 12% | The ON/OFF policy is a clinical decision (§6.1); no method may drop incomplete rows silently |
| FCI with a cluster-robust test took 17.5 min per fit on 43 columns (separating sets capped at 3); Tetrad's latent-aware learners ran out of 16 GB; J-PCMCI+ took over 29 min per fit | A graph of about 27 columns; two learners and one control; cost measured before replicate numbers are fixed |
| Learner tuning finished no job in about 12 hours, because each job ran every learner and configuration before saving | One fit per job, saved as it finishes; at most 3 configurations per learner |
| Eight parallel jobs with deep-learning learners exhausted memory and crashed the session | At most 4 parallel jobs; no deep-learning learners in the primary panel |
| The PPMI-fitted simulator missed its fidelity target (median absolute correlation difference 0.062 and 0.059 against ≤ 0.05) | A smaller graph, and a fidelity criterion and its consequence fixed in advance (§8.1) |
| On synthetic data, a trivial sortability baseline beat the classical learners at default settings (adjacency F1 0.70 against 0.46–0.63) | A sortability control, standardised inputs and a varsortability report |
| A constant variable in a data subset made PC search every separating set for 34 hours | Constant and near-constant variables are found and handled within each phenotype before any fit |
| Earlier expert progression maps were partly supported by PPMI (22–28% of implied independences violated, against 33–39% for relabelled maps) | The new map is tested the same way, in each phenotype (Q4) |

## 3. Design overview

| Work package | Content | Output | Gate to continue |
|---|---|---|---|
| WP0 Feasibility audit | Aggregate counts only: phenotype group sizes, visits, transitions, missing data, phenotype stability, treatment timing | Feasibility tables for the clinical panel | The minority phenotype is large enough for the WP3 power target |
| WP1 Clinical consensus | Phenotype definitions, progression domains, expert map and confirmatory hypotheses, from a two-round questionnaire completed by at least two movement-disorder neurologists; literature search | Frozen definitions, map and hypothesis family | Panel decisions recorded |
| WP2 Protocol freeze | Final protocol and analysis plan | Registered protocol | Investigator approval |
| WP3 Simulation benchmark | PPMI-fitted simulator; scenarios with and without phenotype differences | Learner settings, difference procedure, power at real sizes | False differences controlled; power target met, or Analysis B downgraded to descriptive |
| WP4 PPMI analyses | Analyses A, B and C; expert-map test | Phenotype progression atlas | — |
| WP5 Review and robustness | Blinded clinician review; replication by era; sensitivity analyses | Graded findings | — |
| WP6 Reporting | Clinical paper; methods paper or section | Manuscripts for the PPMI Data and Publications Committee | — |

No phenotype-split analysis of PPMI is run before WP2: the clinicians build the map and choose the hypotheses without seeing data-driven graphs.

## 4. Population and data

- **Source.** PPMI (Marek 2018), under its Data Use Agreement, from the local download (clinical, imaging-summary and genetic files; no biospecimen category).
- **Primary population.** Sporadic PD untreated at the baseline visit: PPMI 1.0 de novo PD (431 in the pilot build) and PPMI 2.0 sporadic PD (957 in the pilot build; untreated status at baseline to be confirmed in WP0). An untreated baseline makes every phenotype assignment an OFF-medication assessment.
- **Secondary population.** Genetic PD, often treated at enrolment and later in the disease, is analysed separately as exploratory.
- **Annual grid.** For each participant and year, the visit closest to the enrolment anniversary within ±4 months with the most complete core battery.
- **Landmark.** The phenotype is assigned at year 0. The primary progression analyses use transitions from year 1 onwards (year *t* to *t*+1, *t* ≥ 1), so the year-0 values that define the phenotype are never in a row that is analysed, and the phenotype label is never a function of variables in the same row. Including the year 0 → 1 transition is a sensitivity analysis. WP0 reports how many transitions this leaves per phenotype and era.
- **Unit of resampling.** The participant, in every subsample, bootstrap and cross-fit.
- **Exclusions.** Participants without a usable baseline phenotype. A later change of diagnosis (for example to multiple system atrophy) stays in the primary analysis and is excluded in a sensitivity analysis, because excluding it uses future information.

## 5. Phenotypes

### 5.1 Requirements

A phenotype must be assignable at the baseline visit from information a clinician has then, defined by a published rule, measurable in PPMI, used in practice, and frequent enough for the WP3 power target.

### 5.2 Candidates

| System | Rule (operational details fixed in WP1) | PPMI inputs | For | Against |
|---|---|---|---|---|
| Motor phenotype: TD / PIGD / indeterminate | Ratio of mean tremor items (2.10, 3.15–3.18) to mean PIGD items (2.12, 2.13, 3.10–3.12); TD ≥ 1.15, PIGD ≤ 0.90 (Stebbins 2013) | MDS-UPDRS II and III items | Universal; used daily | Unstable in early PD; the ratio rule creates selection artefacts; indeterminate group |
| Clinical subtype: mild-motor / intermediate / diffuse malignant | Motor summary, cognition, RBD and dysautonomia scores against 75th-percentile cut-offs (Fereshtehnejad 2015, 2017) | MDS-UPDRS, MoCA and cognitive battery, RBD questionnaire, SCOPA-AUT | Prognostic; validated in PPMI | Built from many graph variables; small diffuse-malignant group; cohort-dependent cut-offs |
| RBD status | RBD screening questionnaire above a cut-off at baseline | RBDSQ | Recognisable; prognostic for cognition | A questionnaire, not polysomnography |
| Age at onset | Early, typical or late onset (cut-offs in WP1) | Diagnosis history | Exogenous; large groups | A demographic axis more than a phenotype |
| Genotype (exploratory) | GBA or LRRK2 carrier versus non-carrier | Genetic consensus file | Biological; exogenous | Small groups; carriers enrolled differently |

### 5.3 Provisional choice

- **Primary axis: motor phenotype, TD versus PIGD**, the clinicians' daily language. Participants with an indeterminate phenotype are left out of the primary analyses and included in a sensitivity analysis (merged with PIGD).
- **Secondary axis: Fereshtehnejad clinical subtype**, contrasting diffuse malignant with mild-motor predominant.
- RBD status, age at onset and genotype are sensitivity or exploratory axes.
- The clinical panel confirms or replaces these in WP1, before any phenotype analysis.

### 5.4 Definition rules

- The phenotype comes from the year-0 (baseline) visit; untreated, so OFF-state by construction.
- Published cut-offs are used as published. A cohort-derived cut-off (Fereshtehnejad's percentiles) is computed once on the pooled primary population and frozen.
- Nodes share no items: MDS-UPDRS II items 2.10 (tremor), 2.12 (walking and balance) and 2.13 (freezing) leave the daily-function node, because tremor and gait have their own nodes.
- Phenotype stability (year 0 against years 1 and 2) is reported in WP0. Reassignment at year 1 is a sensitivity analysis, flagged as using later information.

## 6. Variables and the expert map

### 6.1 Progression domains (provisional; final list in WP1)

| Domain | Measure | Notes |
|---|---|---|
| Tremor | MDS-UPDRS III tremor items | Needed for the phenotype contrast |
| Bradykinesia and rigidity | MDS-UPDRS III items 3.3–3.8 and 3.14 | Speech and facial expression (3.1, 3.2) assigned in WP1 |
| Axial and gait | MDS-UPDRS III items 3.9–3.13 | Includes freezing and postural stability |
| Daily function | MDS-UPDRS II without items 2.10, 2.12, 2.13 | |
| Independence | Modified Schwab & England ADL | 80% is the usual dependence threshold |
| Cognition | MoCA, or a cognitive composite (WP1) | |
| Mood | GDS-15 | |
| Autonomic | SCOPA-AUT | |
| Hallucinations | MDS-UPDRS item 1.2 | |
| Daytime sleepiness | Epworth | Optional |
| RBD symptoms | RBDSQ | Optional; not a node when RBD status is the phenotype axis |
| Motor complications | MDS-UPDRS IV (dyskinesia, fluctuations) | Optional; zero before treatment |
| Dopaminergic treatment | Total levodopa-equivalent daily dose | Time-varying; the dose recorded at year *t*+1 precedes the other year-*t*+1 measurements |

The motor state used after treatment starts (OFF where available, or ON) is decided in WP1; the other state is a sensitivity analysis.

**Baseline and context variables:** age, sex, education, disease duration at baseline, baseline dopamine-transporter binding (putamen), the phenotype label, and the enrolment era (PPMI 1.0 or 2.0) as an exogenous context. With about ten core domains in two years and seven baseline variables, the transition table has about 27 columns.

### 6.2 Temporal knowledge given to every learner

Tiers, in order: era | baseline variables, including the phenotype label | year *t* | treatment at *t*+1 | the rest of year *t*+1. No edge runs from a later tier into an earlier one, and nothing causes era, age or sex. Within a tier the data decide.

### 6.3 Expert phenotype-progression map (built in WP1)

- Each neurologist completes the expert questionnaire (`docs/clinical_expert_questionnaire.md`) independently. Its Section E rates every ordered pair of domains: could X this year directly change Y next year (2 strong, 1 weak, 0 none, ? unsure, with a minus sign for an effect that improves Y).
- The questionnaire also asks for impossible links, and for unmeasured factors that drive several domains (for example overall pathology burden).
- A link enters the map when every rater gives it 1 or 2, and stays out when every rater gives it 0. Every other link returns in round 2 with the group's anonymised answers. After round 2 the majority decides; a tie is settled by the investigator and recorded as disputed. The same two rounds settle the phenotype definitions, the domains and the confirmatory hypotheses (D1 to D7).
- The map is frozen before WP3. It is the skeleton of the simulator, the source of adjustment sets for Analysis C, and the target of the Q4 test.

## 7. Analyses

### 7.1 Learners

- **Primary learner.** Latent-aware and constraint-based, accepting tiers (FCI type; Spirtes 2000, Zhang 2008). It returns a partial ancestral graph (PAG), uses a conditional-independence test that accounts for repeated transitions per participant, and caps separating sets for cost. The test (Fisher-z on Gaussian-copula scores, or a cross-fitted nonparametric test) is chosen in WP3 by accuracy and measured cost.
- **Robustness learner.** Score-based with tiers, on multiply imputed data.
- **Control.** A sortability baseline that ignores conditional independence (Reisach 2021); its score shows how much of a result the data's variance ordering alone explains.
- At most three configurations per learner, chosen on simulated data only. One fit per job, saved as it finishes.

### 7.2 Analysis A: discovery with the phenotype as a baseline variable (all participants)

The primary learner runs on the pooled transition table with the phenotype label in the baseline tier. A link between the phenotype and a year-*t*+1 domain means that next year's value of that domain differs by phenotype among people with the same current state: a *phenotype-associated mechanism*. The phenotype is never used as an exogenous context, and the tiers forbid links into it.

For each phenotype-associated domain, a pre-specified follow-up model classifies the difference:
- next year's value on spline functions of its parents in the learned graph;
- plus the phenotype (a level difference);
- plus phenotype × parent interactions (strength differences).

Standard errors are clustered by participant, with false-discovery control across all follow-up tests (exploratory level).

### 7.3 Analysis B: separate graphs per phenotype, against a definition-matched null

- **Equal sizes.** Each phenotype group is subsampled S times (*provisional* S = 25 to 50) to the same size, 90% of the smaller group. The learner runs on every subsample, giving each link's selection frequency p(e, k) in phenotype k.
- **Statistic.** D(e) = p(e, TD) − p(e, PIGD) for every scored pair, with adjacency and arrow marks compared separately.
- **Definition-matched null.** R datasets (*provisional* R ≥ 100) are simulated from the PPMI-fitted simulator with **shared mechanisms and a shared hidden burden** (§8.1), at the real sizes and missingness. The same phenotype rule is applied to the simulated baselines, and the same subsampled procedure gives null values of D(e). These contain everything a phenotype split produces without a real difference: range restriction, collider bias from the rule, burden-driven level differences, unequal missingness and sample-size effects.
- **Decision.** A link is phenotype-specific when its observed |D(e)| exceeds its null (p-value by rank among the null values; Benjamini–Hochberg at 5% over the scored pairs) and |D(e)| ≥ 0.25 (*provisional*).

### 7.4 Analysis C: confirmatory phenotype-difference hypotheses

- **Hypotheses.** Each names a link X(*t*) → Y(*t*+1), two phenotypes and an expected direction.
- **Model.** Y(*t*+1) on a spline of Y(*t*), a spline of X(*t*), splines of the adjustment set from the expert map, the phenotype, and phenotype × X(*t*).
- **Shared range.** The analysis is restricted to the range of X(*t*) that both phenotypes cover, so a shared curved mechanism is not mistaken for a phenotype difference.
- **Missing values.** Multiple imputation (20 imputations) within each phenotype, so the imputation does not shrink the interaction.
- **Inference.** Participant-clustered standard errors; two-sided tests; Benjamini–Hochberg at 5% across the family.
- **Reporting.** Effects on clinical scales, for example points of Schwab & England per minimal clinically important difference of the axial score.
- **Controls.** Two or three hypotheses where the panel expects no difference are tested alongside as controls and reported separately.

### 7.5 Speed or pathway (Q2)

Every phenotype-associated mechanism found in A or C is classed by a pre-specified rule:

- **Speed:** a level difference with no strength difference, and no structural difference beyond the null in B.
- **Pathway, strength:** a phenotype × parent interaction (in C, or in A's follow-up), with no structural difference beyond the null.
- **Pathway, structure:** a structural difference beyond the null in B, with a supporting interaction.
- **Unresolved:** anything else, including findings that are unstable under subsampling or inconsistent between eras.

The simulation (scenarios S1 to S3) measures how accurately this rule classifies known differences.

### 7.6 Routes to loss of independence (Q3)

For each phenotype: the year-*t* domains that are possible ancestors of next year's independence and daily function in Analysis A's graph, with phenotype-specific strengths from the follow-up models and participant-bootstrap intervals. Descriptive.

### 7.7 Expert-map test per phenotype (Q4)

Every independence the expert map implies, given a minimal observed separating set, is tested in each phenotype with the participant-clustered test. Statistic: the share violated. Two nulls:
- maps relabelled within tiers: does the expert map beat a scrambled one?
- the definition-matched null: is the difference between phenotypes larger than a split alone creates?

### 7.8 Stability, replication and negative controls

- **Stability.** Selection frequencies over participant subsamples. The frequency threshold for a stable link is calibrated on simulated data: the smallest threshold with precision ≥ 0.8.
- **Replication.** Findings are checked in PPMI 1.0 and PPMI 2.0 separately (direction and frequency), and in random half-samples.
- **Negative controls.**
  - The learner run without tiers: arrowheads into era, age or sex, and later-to-earlier edges.
  - The whole pipeline on permuted phenotype labels: the floor of "phenotype differences" when the label carries no information.

### 7.9 Missing data

Constraint-based tests use test-wise deletion; models and the score-based learner use multiple imputation by chained equations. Missing-not-at-random sensitivity: missing OFF exams, MoCA and questionnaires are imputed worse by a pre-set shift. Informative dropout: inverse-probability-of-remaining weights as a sensitivity analysis.

### 7.10 Blinded clinician review

- Every phenotype-specific finding is mixed with an equal number of control items: shared links, and pairs no source proposes.
- Items are shuffled and phrased "In [phenotype], could X this year influence Y next year more than in [other phenotype]?", without statistics.
- At least two neurologists independently rate each item as plausible, likely artefact, or cannot judge. Agreement is summarised by Cohen's κ, and disagreements are resolved by discussion.
- Plausibility supports interpretation; it does not validate causality.

### 7.11 Sensitivity analyses (pre-specified)

1. Secondary phenotype axis (clinical subtype); RBD status; age at onset; genotype (exploratory).
2. Indeterminate motor phenotype merged with PIGD.
3. The year 0 → 1 transitions included, with the phenotype label left out of the graph (Analysis B only).
4. PPMI 1.0 only.
5. Participants whose diagnosis changed excluded.
6. The other motor state (ON instead of OFF, or the reverse).
7. Phenotype reassigned at year 1 (uses later information).
8. Missing-not-at-random shift; dropout weights.
9. Transitions in which treatment started excluded.

## 8. Simulation benchmark (WP3)

### 8.1 Simulator

- **Mechanisms.** Fitted to the pooled primary population over the frozen expert map. Each node's year-*t*+1 value is a flexible function of its parents plus noise, on Gaussian-copula scales that reproduce each variable's type.
- **Hidden burden.** A person-level hidden burden enters every domain.
- **Baselines.** Baseline values come from the fitted joint distribution, including every variable a phenotype rule needs, so the rule applied to simulated baselines reproduces realistic phenotype frequencies.
- **Missingness and dropout.** Missingness is resampled from PPMI's own patterns by era, year and treatment status; dropout follows the observed follow-up distribution.
- **Fidelity criterion (*provisional*).** Median absolute difference between real and simulated Spearman correlations ≤ 0.05 on the scored pairs, and phenotype frequencies within 3 percentage points. If missed, the discrepancy is reported and the benchmark runs on the simulator as fitted, with no tuning towards any method.

### 8.2 Scenarios

| Scenario | Truth | Tests |
|---|---|---|
| S0 Shared | Shared mechanisms, shared burden | False phenotype differences |
| S1 Speed | Shared mechanisms; burden much higher in PIGD | Level differences not read as pathways |
| S2 Strength | Selected links twice as strong in PIGD | Power for strength differences |
| S3 Structure | One link present only in PIGD, another only in TD | Power and localisation for structural differences |
| S4 Curvature | Shared non-linear mechanisms; phenotypes cover different ranges | Curvature not read as a strength difference |

Stresses crossed with S0 to S3 as cost allows: missing not at random, measurement noise in the phenotype-defining items, dropout tied to severity, and sample sizes halved and doubled.

### 8.3 Procedures and metrics

- **Procedures compared:**
  - naive split-and-fit (a link present in one group and absent in the other);
  - split with subsampling intervals;
  - split with the definition-matched null (Analysis B);
  - pooled discovery with the phenotype node (Analysis A);
  - interaction tests (Analysis C).
- **Metrics:**
  - false phenotype-difference rate (S0, S1, S4);
  - recall and precision of true differences (S2, S3);
  - adjacency and arrowhead F1 against the true PAG;
  - accuracy of the speed-or-pathway rule;
  - cost per fit.

### 8.4 Decision rule (pre-specified)

- **Choice of procedure.** WP4 uses the difference procedure whose false-difference rate stays at or below 5% in S0, S1 and S4 and whose recall is highest in S2 and S3.
- **Power gate.** If no procedure reaches recall ≥ 0.5 for a doubling of link strength at the real group sizes, phenotype-specific findings from Analysis B are reported as descriptive only.
- **Replicate numbers.** These follow from Monte Carlo precision and measured cost; a standard error of 2 percentage points on a 5% false-difference rate needs about 120 datasets per scenario.

## 9. Statistical considerations

- **Precision.** There is no general sample-size rule for graph recovery; WP3 gives power at the real group sizes. Analysis A uses all participants and has the most power; B and C depend on the size of the minority phenotype.
- **Multiplicity.** The confirmatory family is controlled by Benjamini–Hochberg at 5%; each discovery analysis has its own false-discovery control; everything else is labelled exploratory.
- **Clustering.** The participant is the unit of every resampling and the cluster of every standard error.
- **Primary tests.** H-M1 (simulation) and H-C1 to H-C10 (PPMI) are pre-specified. All other results are reported beside them and never replace them.

## 10. Bias and validity

| Threat | How it misleads | Mitigation |
|---|---|---|
| Selection by the phenotype rule | Within-phenotype graphs gain or lose links that do not differ in reality | Definition-matched null; pooled Analysis A |
| Range restriction | Links look weaker inside a phenotype whose defining scores vary less | Definition-matched null; equal-size comparisons |
| Shared items | A node partly encodes the phenotype rule | Landmark design; no shared items between nodes |
| Hidden burden ("speed") | Level differences everywhere read as different pathways | Burden in the null simulator; level or strength classification |
| Curvature | A shared curved mechanism looks like a strength difference across ranges | Spline main effects; shared-range restriction; scenario S4 |
| Unequal group sizes | A link missing in the small group read as biology | Equal-size subsampling; power from WP3 |
| Phenotype instability | The baseline label misclassifies the later course | Stability reported; reassignment sensitivity analysis |
| Treatment by indication | Treatment choice differs by phenotype and confounds progression links | Treatment as a node; transitions with treatment start excluded in sensitivity |
| ON/OFF state | Motor scores mean different things on and off medication | Panel decision; the other state as sensitivity |
| Informative dropout | Those who stay look slower | Dropout weights; dropout in the simulator |
| Era differences | PPMI 1.0 and 2.0 differ in enrolment and follow-up | Era as context; replication by era |
| Multiplicity | Attractive graphs selected after the fact | A small confirmatory family; every result reported |

## 11. Clinical outputs and intended use

- A shared progression map with phenotype-specific links marked and graded: confirmatory, replicated discovery, exploratory, unresolved.
- A one-page card per phenotype: routes to loss of independence, the speed-or-pathway verdict per domain, and effect sizes on clinical scales.
- **Intended use:** hypotheses for monitoring (which domains to reassess, and how often), referral (physiotherapy, falls, cognitive assessment) and counselling, and input to the design of phenotype-stratified trials.
- **Not for:** individual prediction, treatment choice, or replacing guideline-based care. Treatment effects need a separate target-trial emulation.

## 12. Out of scope

- **Option C of draft v0.1** (prognostic model, external cohorts, prospective validation): a later programme. The external cohorts (Oxford Discovery, Tracking Parkinson's, ICEBERG, LuxPARK) need applications.
- **Treatment effects by phenotype:** a separate target-trial protocol.
- **Data-driven latent subtypes** (mixtures of causal models, Varambally 2024): exploratory and later, since identifiability is weak with 3 to 9 visits per person.
- **Biological phenotypes** (seed amplification; SynNeurGe): these need PPMI's biospecimen data, which are not in the local download.

## 13. Data governance

- Under the PPMI Data Use Agreement, participant-level tables, the fitted simulator and every per-row array stay on this machine in a git-ignored folder.
- Results and documents hold aggregates only, and nothing participant-level goes to any external service.
- Manuscripts go to the PPMI Data and Publications Committee with the required acknowledgement.
- Every command that reads PPMI participant data is gated until the protocol is approved; WP0 runs with the investigator's written approval.
- Nothing is committed to git without the investigator's instruction.

## 14. Decisions needed

| # | Decision | Recommended default | Decided by | When |
|---|---|---|---|---|
| D1 | Primary phenotype axis | Motor phenotype, TD versus PIGD | Clinical panel | WP1 |
| D2 | Secondary axis | Fereshtehnejad clinical subtype | Clinical panel | WP1 |
| D3 | Indeterminate group | Sensitivity analysis, merged with PIGD | Panel and investigator | WP1 |
| D4 | Population | Sporadic PD untreated at baseline; genetic PD exploratory | Investigator | WP0 |
| D5 | Motor state after treatment | OFF where available; ON as sensitivity | Clinical panel | WP1 |
| D6 | Progression domains | About ten core, up to three optional | Clinical panel | WP1 |
| D7 | Expert map and confirmatory hypotheses | About ten hypotheses and two or three expected-null controls | Clinical panel | WP1 |
| D8 | Landmark | Phenotype at year 0; transitions from year 1 | Investigator | WP2 |
| D9 | Learner test and configurations | Chosen in WP3 by accuracy and cost | Investigator | WP3 |
| D10 | Provisional thresholds | \|D\| ≥ 0.25; Benjamini–Hochberg 5%; fidelity ≤ 0.05; S and R by cost | Investigator | WP2 to WP3 |
| D11 | Pre-registration | OSF, before WP4 | Investigator | WP2 |

## 15. Timeline (provisional)

| Weeks | Work |
|---|---|
| 1–2 | WP0 feasibility audit |
| 2–6 | WP1 two-round expert questionnaire, literature search |
| 7 | WP2 protocol freeze and registration |
| 6–11 | WP3 simulator and benchmark, built and tested on synthetic data first |
| 11–13 | WP4 PPMI analyses |
| 13–16 | WP5 blinded review, replication, sensitivity analyses |
| 16–20 | WP6 manuscripts |

## 16. Compute and reproducibility

The machine is an Apple M4 with 10 cores and 16 GB.
- At most 4 parallel jobs; one fit per job, saved as it finishes and skipped on rerun; a seed per fit.
- Long runs are started from the Terminal app, with lockfiles for every environment.
- The harness is built and tested on synthetic data before any PPMI run, and costs are measured on synthetic data of the real size before replicate numbers are fixed.

## References

- Dadu A, Satone V, Kaur R, et al. Identification and prediction of Parkinson's disease subtypes and progression using machine learning in two cohorts. npj Parkinsons Dis 2022;8:172.
- Fereshtehnejad SM, Romenets SR, Anang JBM, et al. New clinical subtypes of Parkinson disease and their longitudinal progression: a prospective cohort comparison with other phenotypes. JAMA Neurol 2015;72:863–873.
- Fereshtehnejad SM, Zeighami Y, Dagher A, Postuma RB. Clinical criteria for subtyping Parkinson's disease: biomarkers and longitudinal progression. Brain 2017;140:1959–1976.
- Gerhardus A, Runge J. High-recall causal discovery for autocorrelated time series with latent confounders. NeurIPS 2020.
- Günther W, Ninad U, Runge J. Causal discovery for time series from multiple datasets with latent contexts. UAI 2023, PMLR 216:766–776.
- Hernán MA, Hernández-Díaz S, Robins JM. A structural approach to selection bias. Epidemiology 2004;15:615–625.
- Höglinger GU, Adler CH, Berg D, et al. A biological classification of Parkinson's disease: the SynNeurGe research diagnostic criteria. Lancet Neurol 2024;23:191–204.
- Horváth K, Aschermann Z, Ács P, et al. Minimal clinically important difference on the motor examination part of MDS-UPDRS. Parkinsonism Relat Disord 2015;21:1421–1426.
- Horváth K, Aschermann Z, Kovács M, et al. Minimal clinically important differences for the experiences of daily living parts of the MDS-UPDRS. Mov Disord 2017;32:789–793.
- Huang B, Zhang K, Zhang J, et al. Causal discovery from heterogeneous/nonstationary data. J Mach Learn Res 2020;21.
- Jankovic J, McDermott M, Carter J, et al. Variable expression of Parkinson's disease: a base-line analysis of the DATATOP cohort. Neurology 1990;40:1529–1534.
- Marek K, Chowdhury S, Siderowf A, et al. The Parkinson's Progression Markers Initiative (PPMI): establishing a PD biomarker cohort. Ann Clin Transl Neurol 2018;5:1460–1477.
- Mestre TA, Fereshtehnejad SM, Berg D, et al. Parkinson's disease subtypes: critical appraisal and recommendations. J Parkinsons Dis 2021;11:395–404.
- Mooij JM, Magliacane S, Claassen T. Joint causal inference from multiple contexts. J Mach Learn Res 2020;21.
- Pal, Ghosh, Yang. Penalized FCI for causal discovery on PPMI biomarkers. arXiv 2507.00173 (2025).
- Progression subtypes in Parkinson's disease identified by a data-driven multi cohort analysis. npj Parkinsons Dis 2024; doi:10.1038/s41531-024-00712-3.
- Reisach AG, Seiler C, Weichwald S. Beware of the simulated DAG! Causal discovery benchmarks may be easy to game. NeurIPS 2021.
- Runge J. Discovering contemporaneous and lagged causal relations in autocorrelated nonlinear time series datasets. UAI 2020, PMLR 124.
- Siderowf A, Concha-Marambio L, Lafontant DE, et al. Assessment of heterogeneity among participants in the PPMI cohort using α-synuclein seed amplification: a cross-sectional study. Lancet Neurol 2023;22:407–417.
- Simuni T, Caspell-Garcia C, Coffey C, et al. How stable are Parkinson's disease subtypes in de novo patients: analysis of the PPMI cohort? Parkinsonism Relat Disord 2016;28:62–67.
- Simuni T, Chahine LM, Poston K, et al. A biological definition of neuronal α-synuclein disease: towards an integrated staging system for research. Lancet Neurol 2024;23:178–190.
- Spirtes P, Glymour C, Scheines R. Causation, Prediction, and Search, 2nd ed. MIT Press, 2000.
- Stebbins GT, Goetz CG, Burn DJ, et al. How to identify tremor dominant and postural instability/gait difficulty groups with the MDS-UPDRS: comparison with the UPDRS. Mov Disord 2013;28:668–670.
- Varambally S, Ma YA, Yu R. Discovering mixtures of structural causal models from time series data. ICML 2024, PMLR 235.
- Zhang J. On the completeness of orientation rules for causal discovery in the presence of latent confounders and selection bias. Artif Intell 2008;172:1873–1896.

The full literature search in WP1 checks these entries and extends them.

## Appendix A. Candidate confirmatory hypotheses (for the panel to rate)

Each candidate is a link from year *t* to year *t*+1, a phenotype contrast and an expected direction. The panel rates them in Section F of the expert questionnaire, which lists the links without these expected directions so as not to lead the answers. The panel keeps about ten, adds its own, and names two or three expected-null controls. The rationales are proposals to check against the literature in WP1.

| # | Link (year t → year t+1) | Expected | Proposed rationale |
|---|---|---|---|
| A1 | Axial and gait → independence | Stronger in PIGD | Gait and balance impairment as the main route to dependence |
| A2 | Axial and gait → cognition | Stronger in PIGD | Gait and cognitive decline linked by shared non-dopaminergic pathology |
| A3 | Cognition → independence | Stronger in PIGD | Cognitive impairment compounding gait impairment |
| A4 | Tremor → daily function | Stronger in TD | Tremor as the main motor burden in TD |
| A5 | Bradykinesia and rigidity → daily function | No difference (control) | A shared dopaminergic core |
| A6 | Treatment → bradykinesia and rigidity | No difference (control) | Appendicular signs respond to levodopa in both |
| A7 | Treatment → axial and gait | Weaker in PIGD | Axial signs respond less to dopaminergic treatment |
| A8 | Autonomic → independence | Stronger in PIGD | Orthostatic symptoms adding to falls and dependence |
| A9 | Hallucinations → independence | Stronger in PIGD | Psychosis predicting dependence and care needs |
| A10 | RBD symptoms → cognition | Stronger in PIGD (diffuse malignant on the secondary axis) | RBD marking a more diffuse disease |
| A11 | Mood → daily function | No difference (control) | Depression affecting function in both |
| A12 | Treatment → motor complications | Panel to judge | Age and dose may matter more than phenotype |

## Appendix B. Pilot evidence (escalation-discovery, 29 October to 2 November 2026)

Aggregates from `../escalation-discovery` (its PROTOCOL.md §3, §10.1 and §12, and `results/`):

- **Cohort.** 1,741 PD participants (PPMI 1.0 de novo 431, PPMI 2.0 sporadic 957, genetic PD 353); 8,829 annual-grid rows; 8.9 / 3.1 / 5.9 rows per participant.
- **Missing values on the annual grid** (PPMI 1.0 / 2.0 / genetic PD): OFF motor exam 30 / 18 / 43%, axial score 28 / 17 / 41%, MoCA 6 / 7 / 12%; Part IV items 4–8%; questionnaires 3–7%; doses 0%. In an earlier PPMI 1.0 two-year table, 64% of transitions had a missing value.
- **Cost** (Apple M4, 16 GB; 43 columns, 7,220 transition rows):
  - FCI with the cluster-robust test took 17.5 min per fit with separating sets capped at 3; uncapped, it did not finish.
  - PC with the same test took about 42 min.
  - Tetrad's tiered BOSS-FCI, GFCI and FCI ran out of memory.
  - J-PCMCI+ took over 29 min per fit.
  - DECI did not converge in over an hour.
  - The deep-learning and pretrained learners took 2.6 to 25 min per fit, single-threaded.
- **Tuning.** On the PPMI-fitted simulator, no tuning job finished in about 12 hours (8 hours with 8 workers on 1 November, then 4 hours with 4 workers on 2 November), because each job ran every learner and configuration in turn and saved only at the end.
- **Memory.** On 1 November, eight parallel jobs with deep-learning learners exhausted 16 GB. The operating system ended the session, and the run, started from an app that had quit, lost access to its files.
- **Simulator fidelity.** Over 544 scored pairs, the median absolute difference between real and simulated Spearman correlations was 0.062 (linear simulator) and 0.059 (boosted simulator), against a pre-set target of 0.05; the 90th percentile was 0.13.
- **Sortability.** On synthetic data, variance-sortnregress reached adjacency F1 0.70 against 0.46–0.63 for the classical learners at default settings.
- **Constant variables.** With a context variable constant in a subset, PC (no depth cap) searched every separating set and ran for 34 hours.
- **Expert-map test** (earlier work, ppmi-causal). 24.8 / 28.0 / 22.4% of 701 implied independences were violated in PPMI 1.0 / 2.0 / genetic PD, against 38.6 / 36.0 / 33.4% for maps relabelled within tiers (p = 0.0099 in each).
- **Not produced.** No learner was evaluated, and no causal discovery was run on PPMI. The levodopa-decision audit and the core-map test did run, but they concern the escalation decision and are not used here.

## Appendix C. Changes from draft v0.1

- **Kept from Option B:**
  - subgroups fixed at baseline before any subgroup graph is seen;
  - shared versus different structure versus different strength, now framed as speed or pathway;
  - control of false subgroup differences (H-B2, now H-M1);
  - resampling by participant, and metrics that respect partial graphs;
  - pre-registration;
  - blinded clinician review;
  - a small set of clinical domains.
- **Changed:**
  - the focus is now clinical phenotypes;
  - the time-series algorithm matrix is replaced by transition-based discovery, with the phenotype as a baseline variable;
  - external replication is replaced by replication across PPMI eras and half-samples;
  - "uncertainty-aware comparison" is made concrete as the definition-matched null;
  - the hypotheses are concrete and confirmatory.
- **Dropped:**
  - Option C (prognosis, external and prospective validation, the 24-month alive-and-independent endpoint), moved to a later programme;
  - irregular-time methods, because PPMI's annual grid is regular;
  - mixture and regime methods (MCD, SPACETIME, Regime-PCMCI), out of scope for now.

# Phenotype-specific progression mechanisms in Parkinson's disease: causal discovery in PPMI (Part I) and clinical translation (Part II)

> Protocol draft v0.3, 2 November 2026, for review by the investigator, movement-disorder neurologists and a statistician. Not approved; no phenotype analysis of PPMI data has been run. Two v0.2 drafts were written from the same Option B/C proposal (v0.1): the phenotype-discovery protocol and the review-integrated Option B/C protocol. This draft merges them and answers the critical review of v0.1. The review of both drafts is `docs/protocol_review.md`, and Appendix D lists the changes. Part I is Paper 2, experiment 1. Part II goes ahead only if Part I passes its gate (§13), and it needs its own data access, funding and ethics approval. Thresholds marked *provisional* are fixed before pre-registration (§17).

## Summary

**Clinical question.** Clinicians already sort Parkinson's disease (PD) into phenotypes, such as tremor-dominant (TD) versus postural instability and gait difficulty (PIGD), and know that some phenotypes progress faster. They do not know whether phenotypes decline through *different pathways* (for example, loss of independence driven by gait and cognition in one phenotype and by bradykinesia in another) or through *the same pathways at a different speed*. Guidance on monitoring and care (NICE NG71, the MDS evidence-based reviews, AAN 2021) is largely phenotype-independent. If pathways differ, what to monitor and when to refer should differ by phenotype; if only speed differs, the same plan applies at different intervals.

**Four ways phenotypes can differ.** Only the last two are differences of mechanism:
1. *an artefact of the split*: phenotypes are defined from the same symptom scores the analysis models, so splitting patients on them changes the relationships within each group even when the mechanisms are shared;
2. *speed*: the same links, but one phenotype worsens faster from the same current state (a level difference);
3. *strength*: the same links, of different strength;
4. *structure*: a link that matters in one phenotype and is negligible in the other.

**Design.** Part I studies year-to-year transitions in PPMI's sporadic PD cohorts untreated at baseline (about 1,400 participants). The phenotype is assigned at baseline by a pre-specified clinical rule, and the analysed transitions start a year later. Three analyses answer the question at increasing levels of confidence:
- **A. Shared map, all participants.** Causal discovery with the phenotype as a baseline variable finds which domains progress differently by phenotype.
- **B. Separate maps per phenotype, against a definition-matched null.** The null is the set of differences that would appear if all phenotypes shared the same mechanisms, produced by simulating patients and applying the same phenotype rule.
- **C. Confirmatory tests.** About ten phenotype differences, chosen with clinicians before any phenotype analysis, are tested with false-discovery control. Each is classed as similar, stronger in one phenotype, structurally different or unresolved.

A simulation benchmark fitted to PPMI, run before the PPMI analyses, establishes which procedures keep false phenotype differences under control at PPMI's group sizes. Part II follows only if Part I finds replicated phenotype-specific mechanisms. It tests whether they replicate in an independent cohort and whether they improve prognosis beyond patient history and the phenotype label, and it outlines a prospective validation and a treatment-effect study for separate protocols.

**What is new.** Phenotypes, progression subtypes, symptom networks and graph-based prognosis in PD all exist already (§1.4). The contributions sought are:
- a phenotype-resolved progression map that says, link by link, whether phenotypes differ in speed, strength or structure;
- a procedure, with its error rates measured at real cohort sizes, that keeps "phenotype-specific" findings from being artefacts of defining groups from the variables in the graph.

**What clinicians get.** A progression map with shared, phenotype-specific and unresolved links; for each phenotype, the routes to loss of independence; and, domain by domain, a verdict of speed or pathway. These are hypotheses for monitoring, referral and counselling, not treatment rules.

## 1. Background

### 1.1 Phenotypes in practice

The TD/PIGD classification (Jankovic 1990; MDS-UPDRS version, Stebbins 2013) is the most widely used. PIGD is associated with faster motor, cognitive and functional decline, but the classification is unstable in early PD (Simuni 2016). Fereshtehnejad's clinical subtypes (mild-motor predominant, intermediate, diffuse malignant), defined from motor severity, cognition, REM-sleep behaviour disorder (RBD) and dysautonomia, separate progression rates in PPMI (Fereshtehnejad 2015, 2017). An MDS task force asked for subtypes that are reproducible and clinically or biologically meaningful (Mestre 2021). Data-driven progression subtypes have been found across cohorts (PPMI with PDBP, Dadu 2022; PPMI, ICEBERG and LuxPARK, Hähnel 2024). Biological classifications (SynNeurGe, Höglinger 2024; NSD-ISS, Simuni 2024) use α-synuclein seed amplification, which PPMI has measured (Siderowf 2023) but our local download does not hold.

### 1.2 The gap

Different average trajectories do not show different mechanisms: a phenotype that carries more disease burden will progress faster along the same pathways. The clinically useful distinction, speed or pathway, needs a structural (causal) analysis of how domains drive each other, split finely enough to see differences and carefully enough not to invent them.

### 1.3 Why it is hard

1. **Phenotypes are defined from graph variables.** TD/PIGD is a ratio of tremor items to gait and balance items; Fereshtehnejad's subtypes threshold motor, cognitive, RBD and autonomic scores. Splitting patients by such a rule does three things. It restricts the range of the defining variables. It makes their causes dependent within each group (collider bias; Hernán 2004). And it selects people whose year-0 scores were extreme partly by chance, whose scores then drift back (regression to the mean). Graphs and coefficients fitted within phenotypes can therefore differ when the mechanisms do not. Permuting group labels, as the Network Comparison Test does (van Borkulo 2023), does not solve this, because labels computed from the variables are not exchangeable under the null.
2. **Phenotypes are not external contexts.** The disease produces them, and they share hidden causes (overall pathology, baseline state) with every later symptom. This holds even when, as here, they are fixed before the analysed years. Context-based discovery in its usual forms would read those shared causes as effects of the phenotype: joint causal inference with unconfounded contexts (Mooij 2020), CD-NOD (Huang 2020) and J-PCMCI+ (Günther 2023). Here the phenotype is a baseline variable in a latent-aware search, which amounts to joint causal inference with only its exogeneity assumption. Its links to later change signal heterogeneity and are never read as causal effects.
3. **Hidden person-level burden.** A stable burden that drives many domains creates year-to-year associations between domains without any causal link between them (the between-person confound of cross-lagged models; Hamaker 2015), and level differences between phenotypes everywhere.
4. **Absence cannot be shown by non-significance.** A link that is significant in one phenotype and not in the other is no evidence of a difference (Gelman 2006). Absence can only be supported as "smaller than a pre-set negligible size", by an equivalence test (Lakens 2018).
5. **Structure depends on the time step.** A link absent over a short interval can appear at an annual lag through intermediate steps, and effects estimated over one interval differ from effects over another (Voelkle 2012; Kuiper 2018). "Structure" in this protocol means the direct effect at an annual lag, relative to the measured domains.
6. **Few annual visits.** PPMI gives on average 3 (PPMI 2.0) to 9 (PPMI 1.0) annual visits per person, on a regular grid. Methods built for long series (PCMCI+, Runge 2020; LPCMCI, Gerhardus 2020; J-PCMCI+; regime methods) reduce here to a pooled search over two time slices. The unit of analysis is therefore the year-to-year transition, pooled over participants.
7. **Treatment feedback.** PPMI participants are untreated at enrolment and most start dopaminergic treatment during follow-up. Treatment responds to symptoms and changes them, so it is a node in the graph, and its start is a known break in the motor scores.
8. **Small groups.** The minority phenotype (PIGD, or diffuse malignant) may hold a few hundred participants. In groups that small, an absent link usually means low power.

### 1.4 Prior work and novelty

The following are established, and this study claims none of them:
- clinical phenotypes and their prognosis (§1.1);
- data-driven longitudinal progression subtypes across cohorts;
- longitudinal symptom-association networks and graph-informed prognosis in PD;
- causal discovery across contexts, mixtures and regimes, and benchmarks of time-series causal discovery under violated assumptions;
- causal discovery on cross-sectional PPMI biomarkers (Penalized FCI; Pal 2026).

| Prior work | Overlap | What it leaves open |
|---|---|---|
| Multi-cohort progression subtypes (Hähnel 2024) | Longitudinal subtyping in PPMI, ICEBERG and LuxPARK | Do subtypes differ in mechanism, or only in trajectory? |
| Phenotypic association graphs from longitudinal records (Pan 2024) | Symptom association graphs over time in PPMI, PDBP and BioFIND | Do the associations survive causal assumptions and a phenotype comparison? |
| Multi-modal graph prognosis, AdaMedGraph (Lian 2024) | Graph-informed progression prediction in PPMI and PDBP | Do phenotype-specific mechanisms add to the phenotype label? |
| Blockwise mixed-membership subtyping (Kang 2026) | Heterogeneity across patients, domains and time | Are the differences structural? |
| Progression milestones across clinical, pathological and data-driven subtypes (Negida 2026) | Outcomes compared between subtype systems over 10 years | Mechanism, once selection and severity are accounted for? |
| Measurement fidelity in cognitive trajectory prediction (Lin 2026) | The limits measurement sets on individual prediction | Does measurement quality also limit graph recovery? |
| TCD-Arena (Stein 2026) | Robustness of time-series causal discovery to violated assumptions | Short multi-patient panels with groups defined from graph variables |
| Network Comparison Test (van Borkulo 2023) | Testing network differences between groups | Valid only when group labels are exchangeable, which defined phenotypes are not |

**Open question.** Do clinical phenotypes differ in how symptom domains drive each other over time, and if so in speed, strength or structure, once the artefacts of defining phenotypes from the same variables, hidden burden, unequal group sizes and the observation process are accounted for?

Searches on 2 November 2026 (Europe PMC, arXiv and the web) found no phenotype- or subtype-specific causal discovery of clinical progression in PD. The closest work learns association graphs (Pan 2024) or applies causal discovery to cross-sectional biomarkers (Pal 2026). The WP1 literature search, with its databases, queries and dates recorded, confirms or corrects this before any claim of priority.

### 1.5 Pilot evidence from the escalation-discovery experiment

The stopped experiment used the same PPMI download. No code is reused; these are its lessons (details in Appendix B).

| Observation | Consequence for this design |
|---|---|
| 1,741 PD participants pooled; 8.9 / 3.1 / 5.9 annual rows per participant in PPMI 1.0 / PPMI 2.0 / genetic PD | PPMI 2.0 adds people but little follow-up, and PPMI 1.0 carries long-run progression: both are reported, and findings are replicated by era |
| OFF-state motor exam missing in 30 / 18 / 43% of annual rows; MoCA in 6 / 7 / 12% | The ON/OFF policy is a clinical decision (§6.1); no method may drop incomplete rows silently |
| FCI with a cluster-robust test took 17.5 min per fit on 43 columns (separating sets capped at 3); Tetrad's latent-aware learners ran out of 16 GB; J-PCMCI+ took over 29 min per fit | A graph of about 27 columns; a fast configuration for resampling; a compute budget fixed before the benchmark (§8.8) |
| Learner tuning finished no job in about 12 hours, because each job ran every learner and configuration before saving | One fit per job, saved as it finishes; at most 3 configurations per learner |
| Eight parallel jobs with deep-learning learners exhausted memory and crashed the session | At most 4 parallel jobs; no deep-learning learners |
| The PPMI-fitted simulator missed its fidelity target (median absolute correlation difference 0.062 and 0.059 against ≤ 0.05) | A smaller graph, and a fidelity criterion and its consequence fixed in advance (§8.2) |
| On synthetic data, a trivial sortability baseline beat the classical learners at default settings (adjacency F1 0.70 against 0.46–0.63) | A sortability control, standardised inputs and a sortability report |
| A constant variable in a data subset made PC search every separating set for 34 hours | Constant and near-constant variables are found and handled within each phenotype before any fit |
| Earlier expert progression maps were partly supported by PPMI (22–28% of implied independences violated, against 33–39% for relabelled maps) | The new map is tested the same way, in each phenotype (Q4) |

## 2. Questions, estimands and hypotheses

### 2.1 Clinical questions (Part I)

- **Q1 (shared or phenotype-specific).** Which year-to-year links between motor, cognitive, non-motor, treatment and functional domains do phenotypes share, and which differ?
- **Q2 (speed or pathway).** Where phenotypes differ, is it a difference of level (the same mechanism, shifted: "speed") or of strength or structure (a different mechanism: "pathway")?
- **Q3 (routes to loss of independence).** For each phenotype, which domains feed next year's daily function and independence?
- **Q4 (fit of the expert map).** Does a clinician-built progression map fit the data equally well in each phenotype?

### 2.2 Methodological question

- **M1.** At PPMI's sample sizes, how often does fitting a graph separately within each phenotype produce false phenotype differences? Do the definition-matched null and the interaction tests keep them at the nominal rate while still detecting real ones? And how often can a difference of strength be told apart from a difference of structure?

### 2.3 Estimands and classes

The unit is a year-to-year transition, year *t* to *t*+1 with *t* ≥ 1, in a participant of phenotype *k* (TD or PIGD on the primary axis). For a link from domain X to domain Y:

- **Link strength β:** the change in next year's Y per unit of this year's X, in phenotype *k*. It holds this year's Y and the link's adjustment set from the expert map fixed, and it is averaged over the range of X that both phenotypes cover.
- **Strength difference δ:** β in PIGD minus β in TD.
- **Level difference γ:** the difference between phenotypes in next year's Y from the same current state.
- **Smallest effect of interest Δ:** the size below which a link strength, or a difference in strength, is clinically negligible. *Provisional:* 0.10 standard deviations of next year's Y per standard deviation of this year's X. The panel sees it in clinical units, and it is fixed in WP2 (D12).
  - For comparison, psychological panel studies call standardised cross-lagged effects of 0.03, 0.07 and 0.12 small, medium and large (Orth 2024), so 0.10 is a lenient margin.
  - A smaller margin makes the classes "similar" and "structurally different" harder to reach, and so more conservative. Δ = 0.05 is a sensitivity analysis.

Effects are reported per minimal clinically important difference (MCID) of X where one exists, such as 3.25 points of MDS-UPDRS III (Horváth 2015) and 3.05 points of MDS-UPDRS II (Horváth 2017), and per standard deviation otherwise.

Each confirmatory link, and each phenotype-associated link from Analysis A, falls into one class. M0, M1 and M2 are the critical review's model classes, here applied to one link.

| Class | Model class | Rule |
|---|---|---|
| Similar | M0 | The 90% confidence interval of δ lies within ±Δ (an equivalence test at 5%) |
| Stronger in one phenotype | M1 | δ ≠ 0 under the family's false-discovery control, and β is non-zero with the same sign in both phenotypes |
| Structurally different | M2 | δ ≠ 0, and either β is non-zero in one phenotype with its 90% interval within ±Δ in the other, or β is non-zero with opposite signs |
| Different, type unresolved | M1 or M2 | δ ≠ 0, but neither of the two rules above holds |
| Unresolved | — | None of the above. Reported with the largest difference the data allow (the upper confidence bound of \|δ\|) |

A level difference into a domain whose tested links are all similar or unresolved is a **speed** difference. A difference of strength or structure is a **pathway** difference (§7.5).

At PPMI's sizes, intervals in the minority phenotype will often be too wide to show equivalence, so many links will be unresolved. They are reported as unresolved, never as "no difference" or "absent". WP3 measures how often each class can be reached at Δ = 0.10 and at Δ = 0.05.

### 2.4 Hypotheses

- **H-M1 (primary methodological test, simulation).** Under shared mechanisms (scenarios S0, S1 and S4), naive split-and-fit declares phenotype-specific links for more than 5% of truly shared links. The definition-matched procedure (Analysis B) and the interaction tests (Analysis C) stay at or below 5%, within two Monte Carlo standard errors.
- **H-C1 to H-C10 (primary clinical tests, PPMI).** About ten pre-specified strength differences (δ ≠ 0), each with an expected direction (candidates in Appendix A), frozen with the clinical panel in WP1. They are tested two-sided, with Benjamini–Hochberg control at 5% across the family.
- **H-S (key secondary).** Most phenotype differences in progression are differences of speed rather than of pathway. The share in each class is estimated, not assumed.
- Part II's questions (§12.1) are not tests of Part I.

## 3. Design overview and governance

| Work package | Content | Output | Gate to continue |
|---|---|---|---|
| WP0 Feasibility audit | Aggregate counts only: phenotype group sizes, visits, transitions, missing data, phenotype stability, treatment timing, visit intervals, and events for Part II's endpoint | Feasibility tables for the clinical panel | The minority phenotype is large enough for the WP3 power target; the interval rule (§6.3) is applied |
| WP1 Clinical consensus | Phenotype definitions, domains, expert map, confirmatory hypotheses and Δ, from a two-round questionnaire completed by at least two movement-disorder neurologists; literature search | Frozen definitions, map and hypothesis family | Panel decisions recorded |
| WP2 Protocol freeze | Final protocol and statistical analysis plan | Registered protocol | Investigator approval |
| WP3 Simulation benchmark | PPMI-fitted simulator; scenarios with and without phenotype differences; measured costs | Learner settings, difference procedure, power and class reachability at real sizes | False differences controlled; power target met, or Analysis B downgraded to descriptive |
| WP4 PPMI analyses | Analyses A, B and C; expert-map test | Phenotype progression atlas | — |
| WP5 Review and robustness | Blinded clinician review; replication by era; sensitivity analyses | Graded findings | Part II gate (§13) |
| WP6 Reporting | Clinical paper; methods paper or section | Manuscripts for the PPMI Data and Publications Committee | — |
| Part II (conditional) | External replication; prognostic value; outlines of a prospective validation and a treatment-effect study | Separate protocols and approvals | §13 |

No phenotype-split analysis of PPMI is run before WP2: the clinicians build the map and choose the hypotheses without seeing data-driven graphs.

**Oversight.**
- Part I: the investigator, the supervisors, a statistician and the clinical panel.
- Part II adds allied-health clinicians, site coordinators, and patient and public representatives. It also needs independent data-governance and clinical-safety review before any prospective recruitment.

## Part I. Phenotype causal discovery in PPMI

## 4. Population and data

- **Source.** PPMI (Marek 2018), under the investigator's Data Use Agreement, from the local download (clinical, imaging-summary and genetic files; no biospecimen category). No external cohort is accessible; Part II needs applications.
- **Primary population.** Sporadic PD untreated at the baseline visit, which makes every phenotype assignment an OFF-medication assessment:
  - PPMI 1.0 de novo PD (431 in the pilot build);
  - PPMI 2.0 sporadic PD (957 in the pilot build; untreated status at baseline to be confirmed in WP0).
- **Secondary population.** Genetic PD, often treated at enrolment and later in the disease, is analysed separately as exploratory.
- **Annual grid.** For each participant and year, the visit closest to the enrolment anniversary within ±4 months with the most complete core battery.
- **Landmark.** The phenotype is assigned at year 0. The primary progression analyses use transitions from year 1 onwards (year *t* to *t*+1, *t* ≥ 1).
  - Neither the year-0 values that define the phenotype nor their measurement error enter an analysed row, which removes regression to the mean from those rows.
  - The phenotype label is never a function of variables in the same row.
  - The range restriction and shared causes that remain are handled by the definition-matched null (§7.3).
  - Including the year 0 → 1 transition is a sensitivity analysis. WP0 reports how many transitions the landmark leaves per phenotype and era.
- **Unit of resampling.** The participant, in every subsample, bootstrap and cross-fit.
- **Exclusions.** Participants without a usable baseline phenotype are excluded. A later change of diagnosis (for example to multiple system atrophy) stays in the primary analysis and is excluded only in a sensitivity analysis, because excluding it uses future information.

## 5. Phenotypes

### 5.1 Requirements

A phenotype must be:
- assignable at the baseline visit from information a clinician has then;
- defined by a published rule;
- measurable in PPMI;
- used in practice;
- frequent enough for the WP3 power target.

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
- Published cut-offs are used as published. A cohort-derived cut-off (Fereshtehnejad's percentiles) is computed once on the pooled primary population and frozen; in Part II's cross-validation it is recomputed within each training fold.
- Nodes share no items: MDS-UPDRS II items 2.10 (tremor), 2.12 (walking and balance) and 2.13 (freezing) leave the daily-function node, because tremor and gait have their own nodes.
- Phenotype stability (year 0 against years 1 and 2) is reported in WP0. Reassignment at year 1 is a sensitivity analysis, flagged as using later information.

## 6. Variables, time and the expert map

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

### 6.3 Time step and visit intervals

- A transition spans one step of the annual grid. A step across a missed year (two years) is not used in the primary analyses.
- **Interval rule (pre-specified).** WP0 reports the distribution of actual intervals between analysed visits.
  - If at least 90% lie within 12 ± 3 months, the discrete annual model is primary, and the actual interval enters only a sensitivity analysis.
  - Otherwise, the interval enters the Analysis C models with its interactions, and the continuous-time model below becomes a co-primary sensitivity analysis.
- **Continuous-time model.** A hierarchical continuous-time structural equation model (Driver 2018) of the confirmatory links, with participant-level random effects.
  - In Part I it is a sensitivity analysis.
  - In Part II it is the main tool for comparing effects with cohorts that use other visit intervals.
  - Its drift coefficients are effects under the model, not identified causal arrows.

### 6.4 Expert phenotype-progression map (built in WP1)

- Each neurologist completes the expert questionnaire (`docs/clinical_expert_questionnaire.md`) independently. Its Section E rates every ordered pair of domains: could X this year directly change Y next year? Codes: 2 strong, 1 weak, 0 none, ? unsure, with a minus sign for an effect that improves Y.
- The questionnaire also asks for impossible links, and for unmeasured factors that drive several domains (for example overall pathology burden). Its answers on minimal clinically important differences (Section H) inform Δ.
- A link enters the map when every rater gives it 1 or 2, and stays out when every rater gives it 0. Every other link returns in round 2 with the group's anonymised answers. After round 2 the majority decides; a tie is settled by the investigator and recorded as disputed. The same two rounds settle the phenotype definitions, the domains and the confirmatory hypotheses (D1 to D7). Round 2 also shows the panel Δ in clinical units, computed from WP0's standard deviations, for confirmation (D12).
- The map is frozen before WP3. It is the skeleton of the simulator, the source of adjustment sets for Analysis C, and the target of the Q4 test.

## 7. Analyses

### 7.1 Learners

- **Primary learner.** Latent-aware and constraint-based, accepting tiers (FCI type; Spirtes 2000, Zhang 2008). It returns a partial ancestral graph (PAG) and caps separating sets for cost. In single pooled fits (Analysis A and the expert-map test) it uses a conditional-independence test that accounts for repeated transitions per participant. The test is chosen in WP3 by accuracy and measured cost: Fisher-z on Gaussian-copula scores with a cluster correction, or a cross-fitted nonparametric test.
- **Fast configuration for resampling.** Inside subsamples, nulls and the simulation (Analysis B, stability, WP3), the same learner runs with the uncorrected Fisher-z test on Gaussian-copula scores. Repeated transitions make that test admit extra links, but every resampling decision is calibrated against simulated data with the same participant structure, which absorbs them. WP3 checks that the simulator reproduces the within-participant dependence. Target cost: at most one minute per fit on a table of the real size, measured in WP3.
- **Robustness learner.** Score-based with tiers, on multiply imputed data.
- **Control.** A sortability baseline that ignores conditional independence (Reisach 2021). Variance- and R²-sortability are reported for real and simulated data (Reisach 2023): they show how much of a result the ordering of variances alone explains.
- At most three configurations per learner, chosen on simulated data only. One fit per job, saved as it finishes.

**Methods from the Option B/C draft and their place here:**

| Method | Decision | Reason |
|---|---|---|
| PCMCI+, LPCMCI | Not used as separate methods | With 1 to 8 transitions per person they reduce to the pooled two-slice search with tiers used here; LPCMCI's allowance for hidden confounders is kept through the FCI-type learner |
| Bagged-PCMCI+ (Debeire 2024) | Idea kept, tool not used | It bootstraps blocks of one long series; the participant bootstrap (§7.8) is the panel equivalent |
| J-PCMCI+, joint causal inference or CD-NOD with the phenotype as a context | Not used | The phenotype is not an external context (§1.3); J-PCMCI+ took over 29 minutes per fit in the pilot. Era may be a context |
| Hierarchical continuous-time SEM (Driver 2018) | Sensitivity analysis; the tool for Part II replication | PPMI's annual grid is regular (checked by the interval rule, §6.3); drift coefficients are not identified causal arrows |
| Continuous-time SEM forests (Cáncer 2025) | Not used | They split on covariates; splitting on phenotype-defining variables recreates the artefact of §1.3 |
| MCD (Varambally 2024) and other deep-learning learners | Not used | Mixtures are weakly identified with 3 to 9 visits per person; the pilot's deep-learning jobs exhausted memory |
| SPACETIME (Mameche 2025), Regime-PCMCI (Saggioro 2020) | Not used | Temporal regimes are not patient phenotypes; both need long series |
| ReTimeCausal (Li 2026) | Not used | Built for irregular sampling; PPMI's grid is regular |
| Mixed-effects and random-intercept cross-lagged models | Sensitivity analysis (§7.4) | They separate stable between-person differences from year-to-year links |

### 7.2 Analysis A: discovery with the phenotype as a baseline variable (all participants)

The primary learner runs on the pooled transition table with the phenotype label in the baseline tier. A link between the phenotype and a year-*t*+1 domain means that next year's value of that domain differs by phenotype among people with the same current state: a *phenotype-associated mechanism*. The phenotype is never used as an exogenous context, and the tiers forbid links into it.

For each phenotype-associated domain Y, a pre-specified follow-up model estimates the level difference and the strength difference of each parent of Y in the learned graph:
- next year's Y on spline functions of its parents in the learned graph;
- plus the phenotype (the level difference);
- plus phenotype × parent terms (the strength differences).

Standard errors are clustered by participant, with Benjamini–Hochberg control across all follow-up tests (exploratory level). The links are classed by the rules of §2.3.

### 7.3 Analysis B: separate graphs per phenotype, against a definition-matched null

- **Equal sizes.** Each phenotype group is subsampled S times (*provisional* S = 25 to 50) to the same size, 90% of the smaller group. The learner, in its fast configuration, runs on every subsample, giving each link's selection frequency p(e, k) in phenotype k.
- **Statistic.** D(e) = p(e, TD) − p(e, PIGD) for every scored pair, with adjacency and arrow marks compared separately.
- **Two reference distributions.**
  - *Permuted labels.* The same procedure on 50 random relabellings of participants at the observed group sizes, showing the differences that sampling noise alone produces.
  - *Definition-matched null.* R datasets (*provisional* R = 100) are simulated from the PPMI-fitted simulator with **shared mechanisms and a shared hidden burden** (§8.2), at the real sizes and missingness. The same phenotype rule is applied to the simulated baselines, and the same subsampled procedure gives null values of D(e). These contain everything a phenotype split produces without a real difference: range restriction, collider bias from the rule, regression to the mean, burden-driven level differences, unequal missingness and sample-size effects.
  - The gap between the two references measures the artefact created by defining phenotypes from graph variables, a result in its own right.
- **Decision.** A link is a candidate phenotype-specific link when its observed |D(e)| exceeds its definition-matched null (p-value by rank among the null values; Benjamini–Hochberg at 5% over the scored pairs) and |D(e)| ≥ 0.25 (*provisional*).
- **Role.** An exploratory screen for structural differences, and the real-data counterpart of H-M1. It is never a confirmatory test. A candidate counts as structurally different only when the class rules of §2.3 also say so.

### 7.4 Analysis C: confirmatory phenotype-difference hypotheses

- **Hypotheses.** Each names a link X(*t*) → Y(*t*+1), two phenotypes and an expected direction.
- **Model.** Y(*t*+1) is regressed on:
  - a spline of Y(*t*);
  - a spline of X(*t*);
  - splines of the adjustment set from the expert map;
  - era;
  - the phenotype;
  - phenotype × X(*t*), with X centred at the median of the shared range.

  The interaction estimates δ, and the phenotype term estimates γ at that median. β in each phenotype is the average slope of X over the shared range.
- **Shared range.** The analysis is restricted to the range of X(*t*) that both phenotypes cover, so a shared curved mechanism is not mistaken for a phenotype difference.
- **Missing values.** Multiple imputation (20 imputations) within each phenotype, so the imputation does not shrink the interaction.
- **Inference.**
  - Standard errors are clustered by participant.
  - δ is tested two-sided, with Benjamini–Hochberg at 5% across the family: these are the primary tests.
  - The tests of β and the equivalence tests for the classes (§2.3) use 5% and are not adjusted further, because they only describe the differences the primary tests confirm.
- **Reporting.** Effects on clinical scales, for example points of Schwab & England per minimal clinically important difference of the axial score.
- **Controls.** Two or three hypotheses where the panel expects no difference are tested alongside as controls and reported separately.
- **Sensitivity.**
  - A participant random intercept (a mixed model), for stable between-person differences.
  - Δ = 0.05.
  - The continuous-time model (§6.3).

### 7.5 Speed or pathway (Q2)

Every domain with a phenotype difference in Analysis A or C is classed by a pre-specified rule:

- **Speed:** a level difference (false-discovery controlled across domains), with every tested link into the domain similar or unresolved and no structural difference beyond the null in Analysis B.
- **Pathway, strength:** at least one link into the domain is stronger in one phenotype, and none is structurally different.
- **Pathway, structure:** at least one link into the domain is structurally different (§2.3) and is a candidate beyond the definition-matched null in Analysis B.
- **Pathway, type unresolved:** at least one link into the domain differs, but its type is unresolved or Analysis B does not support it.
- **Unresolved:** anything else, including findings that are unstable under subsampling or inconsistent between eras.

The simulation (scenarios S1, S2, S2b and S3) measures how accurately this rule classifies known differences.

### 7.6 Routes to loss of independence (Q3)

For each phenotype: the year-*t* domains that are possible ancestors of next year's independence and daily function in Analysis A's graph, with phenotype-specific strengths from the follow-up models and participant-bootstrap intervals. Descriptive.

### 7.7 Expert-map test per phenotype (Q4)

Every independence the expert map implies, given a minimal observed separating set, is tested in each phenotype with the participant-clustered test. The statistic is the share violated, against two nulls:
- maps relabelled within tiers: does the expert map beat a scrambled one?
- the definition-matched null: is the difference between phenotypes larger than a split alone creates?

### 7.8 Stability, replication and negative controls

- **Stability.** Selection frequencies over participant bootstrap samples, the panel version of bagging (Debeire 2024). The frequency threshold for a stable link is calibrated on simulated data: the smallest threshold with precision ≥ 0.8.
- **Replication.** Findings are checked in PPMI 1.0 and PPMI 2.0 separately (direction and frequency), and in random half-samples.
- **Negative controls.**
  - The learner run without tiers: arrowheads into era, age or sex, and later-to-earlier edges.
  - The whole pipeline on permuted phenotype labels: the floor of "phenotype differences" when the label carries no information.

### 7.9 Missing data, dropout and death

- Constraint-based tests use test-wise deletion; models and the score-based learner use multiple imputation by chained equations.
- Missing-not-at-random sensitivity: missing OFF exams, MoCA and questionnaires are imputed worse by a pre-set shift.
- Informative dropout: inverse-probability-of-remaining weights as a sensitivity analysis.
- Death and loss to follow-up are recorded separately, and nothing is imputed after death.

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
10. A participant random intercept in Analysis C.
11. Δ = 0.05 instead of 0.10.
12. The actual visit interval as a covariate, and the continuous-time model of the confirmatory links (§6.3).

## 8. Simulation benchmark (WP3)

The benchmark is reported in the ADEMP structure: aims, data-generating mechanisms, estimands, methods and performance measures (Morris 2019). The harness is built and tested on synthetic data before the simulator is fitted to PPMI.

### 8.1 Aims

1. Test H-M1.
2. Measure the false-difference rates, power and class accuracy of the procedures in §8.4 at PPMI's group sizes.
3. Choose the learner configurations (at most three per learner) and confirm the cost of the fast configuration.
4. Fix the numbers of subsamples, null datasets and replicates from measured cost (§8.8).

### 8.2 Data-generating mechanisms

- **Mechanisms.** Fitted to the pooled primary population over the frozen expert map. Each node's year-*t*+1 value is a flexible function of its parents plus noise, on Gaussian-copula scales that reproduce each variable's type.
- **Hidden burden.** A person-level hidden burden enters every domain.
- **Baselines.** Baseline values come from the fitted joint distribution, including every variable a phenotype rule needs, so the rule applied to simulated baselines reproduces realistic phenotype frequencies.
- **Missingness and dropout.** Missingness is resampled from PPMI's own patterns by era, year and treatment status; dropout follows the observed follow-up distribution.
- **Fidelity criterion (*provisional*).** Median absolute difference between real and simulated Spearman correlations ≤ 0.05 on the scored pairs, and phenotype frequencies within 3 percentage points. If missed, the discrepancy is reported and the benchmark runs on the simulator as fitted, with no tuning towards any method.

| Scenario | Truth | Tests |
|---|---|---|
| S0 Shared | Shared mechanisms, shared burden | False phenotype differences |
| S1 Speed | Shared mechanisms; burden much higher in PIGD | Level differences not read as pathways |
| S2 Strength | Selected links twice as strong in PIGD | Power for strength differences |
| S2b Weak versus strong | Selected links four times as strong in PIGD, with the TD strength near Δ | Strength differences not read as structure |
| S3 Structure | One link present only in PIGD, another only in TD | Power and localisation for structural differences |
| S4 Curvature | Shared non-linear mechanisms; phenotypes cover different ranges | Curvature not read as a strength difference |

Stresses crossed with S0 to S3 as cost allows: missing not at random, measurement noise in the phenotype-defining items, dropout tied to severity, hidden burden absent or strong, and sample sizes halved and doubled.

### 8.3 Estimands

The true class of every link (similar, strength or structure) and of every domain (speed or pathway), the true δ and γ of every link, and the true PAG over the measured variables.

### 8.4 Procedures compared

- **P1.** Naive split-and-fit: a link present in one group and absent in the other. In S0, S1 and S4 it runs with the cluster-aware test, so that H-M1 measures the split, not the test.
- **P2.** Split with subsampling intervals.
- **P3.** Split with the definition-matched null (Analysis B).
- **P4.** Pooled discovery with the phenotype node, with its follow-up models (Analysis A).
- **P5.** Interaction tests with the class rules (Analysis C).
- The sortability control.

### 8.5 Performance measures

Each is reported with its Monte Carlo standard error:
- the false phenotype-difference rate per truly shared link (S0, S1, S4), and the family-wise rate of P5's family;
- the rate of false "structure" verdicts (S2, S2b);
- recall and precision of true differences (S2, S2b, S3);
- accuracy of the class rules (§2.3) and of the speed-or-pathway rule (§7.5), and how often each class is reached at Δ = 0.10 and at Δ = 0.05;
- bias and 95% interval coverage of δ and γ (P5);
- adjacency and arrowhead F1 against the true PAG;
- cost per fit and failures.

### 8.6 Replicates and the nested evaluation of Analysis B

- The Monte Carlo standard error of a rate p estimated from R datasets is √(p(1 − p)/R). At p = 5%, 120 datasets give 2.0 percentage points and 475 give 1.0.
- **P1, P2 and P4:** 120 datasets per scenario (*provisional*), with 10 subsamples per group for P2.
- **P5:** cheap regressions, so 1,000 datasets per scenario, with 5 imputations each.
- **P3:** every analysed dataset needs its own definition-matched null, so evaluating P3 is a nested simulation. It is evaluated on a reduced design (*provisional*): 10 datasets in each of S0 to S4, each with its own simulator fit, 50 null datasets and 10 subsamples. Monte Carlo error is reported. The full design (25 to 50 subsamples, 100 null datasets) applies to the real data only.

### 8.7 Decision rule (pre-specified)

- **Choice of procedure.** WP4 uses the difference procedure whose false-difference rate stays at or below 5% in S0, S1 and S4 and whose recall is highest in S2 and S3.
- **Power gate.** If no procedure reaches recall ≥ 0.5 for a doubling of link strength at the real group sizes, phenotype-specific findings from Analysis B are reported as descriptive only.
- **Class reachability.** If equivalence is shown for fewer than half of the truly similar links at the chosen Δ, the classes "similar" and "structurally different" are reported as out of reach at PPMI's sizes. Links are then described by their estimated differences and confidence bounds.

### 8.8 Compute budget

Costs are measured on synthetic data of the real size before any number below is fixed. The table assumes the targets of 30 seconds per fast fit and 5 minutes per cluster-aware fit, with 4 parallel jobs.

| Component | Fits | Hours |
|---|---|---|
| WP3: P1 with the cluster-aware test (S0, S1, S4) | 3 scenarios × 120 datasets × 2 = 720 | 15 |
| WP3: P1, P2 and P4 with the fast test | 6 scenarios × 120 datasets × 23 = 16,560 | 35 |
| WP3: P3, nested | 5 scenarios × 10 datasets × 1,020 = 51,000 | 106 |
| WP4: Analysis B, both reference distributions | (1 + 100 + 50) × 25 subsamples × 2 phenotypes = 7,550 | 16 |
| WP4: Analysis A, expert-map test, stability | About 20 cluster-aware and 200 fast fits | 1 |
| Total | | About 173 (7.2 days) |

If measured costs push the total above 10 days, the design is cut in this order:
1. stress cells;
2. the nested design, to 5 datasets per scenario;
3. subsamples, from 25 to 15;
4. null datasets, from 100 to 50.

A fast fit costing more than 2 minutes means the fast configuration is redesigned (fewer columns or a lower separating-set cap) before the benchmark runs.

## 9. Statistical considerations

- **Precision.** There is no general sample-size rule for graph recovery; WP3 gives power and class reachability at the real group sizes. Analysis A uses all participants and has the most power; Analyses B and C depend on the size of the minority phenotype.
- **Multiplicity.**
  - The confirmatory family is controlled by Benjamini–Hochberg at 5%.
  - The class tests describe confirmed differences and are not adjusted further.
  - Each discovery analysis has its own false-discovery control.
  - Everything else is labelled exploratory.
- **Clustering.** The participant is the unit of every resampling and the cluster of every standard error.
- **Primary tests.** H-M1 (simulation) and H-C1 to H-C10 (PPMI) are pre-specified. All other results are reported beside them and never replace them.
- **Monte Carlo error** is reported with every simulated rate.

## 10. Bias and validity

| Threat | How it misleads | Mitigation |
|---|---|---|
| Selection by the phenotype rule | Within-phenotype graphs gain or lose links that do not differ in reality | Landmark; definition-matched null; pooled Analysis A |
| Range restriction | Links look weaker inside a phenotype whose defining scores vary less | Definition-matched null; equal-size comparisons; shared-range restriction |
| Regression to the mean | Extreme year-0 scores drift back, mimicking phenotype-specific change | Landmark: no year-0 values in analysed rows; definition-matched null |
| Shared items | A node partly encodes the phenotype rule | No shared items between nodes |
| Hidden burden | Level differences everywhere, and lagged links without a causal path, read as different pathways | Burden in the simulator; latent-aware learner; level or strength classes; random-intercept sensitivity |
| Non-significance read as absence | "Present in PIGD, absent in TD" produced by unequal power | Difference tests; equivalence margin Δ; equal-size comparisons |
| Curvature | A shared curved mechanism looks like a strength difference across ranges | Spline main effects; shared-range restriction; scenario S4 |
| Time step | Annual-lag "structure" includes effects passed through intermediate steps within the year | Estimand defined at the annual lag; interval rule; continuous-time sensitivity |
| Unequal group sizes | A link missing in the small group read as biology | Equal-size subsampling; power from WP3 |
| Phenotype instability | The baseline label misclassifies the later course | Stability reported; reassignment sensitivity analysis |
| Treatment by indication | Treatment choice differs by phenotype and confounds progression links | Treatment as a node; transitions with treatment start excluded in sensitivity |
| ON/OFF state | Motor scores mean different things on and off medication | Panel decision; the other state as sensitivity |
| Informative dropout | Those who stay look slower | Dropout weights; dropout in the simulator; nothing imputed after death |
| Era differences | PPMI 1.0 and 2.0 differ in enrolment and follow-up | Era as context; replication by era |
| Unstable orientation | Weak tests give confident arrows | PAG marks kept; participant bootstrap; tiers |
| Multiplicity and selective reporting | Attractive graphs selected after the fact | A small confirmatory family; pre-registration; every result reported |
| Leakage (Part II) | Outcome information reaches the predictions through graphs, cut-offs or tuning | Every step inside training folds; the package frozen before external outcomes are seen |

## 11. Clinical outputs and intended use

- A shared progression map with phenotype-specific links marked and graded (confirmatory, replicated discovery, exploratory, unresolved). Each phenotype difference is labelled with its class (§2.3).
- A one-page card per phenotype: routes to loss of independence, the speed-or-pathway verdict per domain, and effect sizes on clinical scales.
- **Intended use:** hypotheses for monitoring (which domains to reassess, and how often), referral (physiotherapy, falls, cognitive assessment) and counselling, and input to the design of phenotype-stratified trials.
- **Not for:** individual prediction, treatment choice, or replacing guideline-based care. Treatment effects need a separate target-trial emulation (§12.6).
- **Out of scope for both parts:**
  - data-driven latent subtypes (mixtures of causal models, Varambally 2024), because identifiability is weak with 3 to 9 visits per person;
  - biological phenotypes (seed amplification; SynNeurGe), which need PPMI's biospecimen data.

## Part II. Clinical translation (conditional)

Part II is an outline. Each component gets its own protocol, data access and approvals after Part I's gate (§13). Its value is clinical utility. Prediction accuracy does not validate the causal map, and the map can be right without improving prediction.

## 12. Part II components

### 12.1 Questions

- **C-RQ0 (replication).** Do Part I's replicated phenotype differences replicate, in direction and class, in an independent cohort?
- **C-RQ1 (prognosis).** Does phenotype-specific mechanism information improve the prediction of loss of independence beyond longitudinal history, the phenotype label and a shared map?
- **C-RQ2 (prospective validation).** Does a locked model generalise to newly recruited patients and give decision-analytically useful estimates for one pre-specified care action?
- **Treatment effects by phenotype:** a separate target-trial emulation (§12.6).

### 12.2 Intended clinical use

An assistive risk estimate at a neurology follow-up visit, to inform monitoring frequency, functional and falls assessment, rehabilitation and care planning. It is not for diagnosis, drug choice, treatment withdrawal or automatic scheduling. One action is named for evaluation: *provisionally*, referral for a multidisciplinary functional and fall-risk assessment. The clinical panel sets its burdens, the capacity for it and the risk threshold at which clinicians would act.

### 12.3 II-a. External replication of phenotype-specific mechanisms

- **Cohort.** One independent cohort never used in Part I, with MDS-UPDRS, a cognitive scale and Schwab & England. Candidates, each needing an application and a harmonisation audit: Oxford Discovery and Tracking Parkinson's (through the Oxford Parkinson's Disease Centre), ICEBERG and LuxPARK.
- **What is replicated.** Part I's confirmatory and replicated-discovery differences, with their direction and class. The phenotype rule, landmark and Analysis C models are fixed before the external data are analysed.
- **Visit intervals.** Where the cohort's interval differs from PPMI's (for example 18 months), effects are compared through the continuous-time model (§6.3), not as annual-lag coefficients.
- **Criterion.** A difference replicates when δ has the same sign and its one-sided test at 5% rejects zero in the external cohort. Its estimate is reported beside Part I's.

### 12.4 II-b. Prognostic incremental value

- **Landmark.** A follow-up visit (*provisional*: year 2), when the year-0 phenotype and two years of history are known.
- **Endpoint (provisional).** Alive and independent 24 months after the landmark, among participants alive and independent at it.
  - Independence means Schwab & England ≥ 80%; scores below 80% have been used to define dependency (Macleod 2016).
  - Death is a failure, not a missing value.
  - Secondary endpoint: first sustained dependence or death within 24 months. Recovery after dependence makes the two endpoints differ.
- **Events.** In early, research-enrolled PD, death and loss of independence within 24 months may be rare: a genome-wide mortality analysis left PPMI out because fewer than 20 deaths had been recorded (Tan 2024). WP0 counts the events, and the required development sample size is computed from the observed event rate and an anticipated model fit (Riley 2020). If PPMI falls short, the endpoint changes, in this order:
  1. the PPMI milestone composite (Brumm 2023): 25 milestones in six domains, whose functional-dependence milestone uses the same Schwab & England threshold;
  2. a 48-month horizon;
  3. next-year Schwab & England and MDS-UPDRS II as continuous outcomes.
- **Models.** All use the same information available at the landmark, with comparable tuning effort.

| Model | Information | Question |
|---|---|---|
| C0 Clinical | Age, sex, disease duration, MDS-UPDRS II and III, MoCA, Schwab & England, Hoehn and Yahr, levodopa-equivalent dose | Is the work clinically competitive? |
| C1 Longitudinal | C0 plus each domain's history (last value and yearly slope), in a flexible learner | Does anything beat good use of the history? |
| C2 Phenotype label | C1 plus the phenotype | Is the label enough? |
| C3 Shared map | C1 plus forecasts from a shared transition model (Analysis A's graph, follow-up models without phenotype terms), iterated over two annual steps | Does a shared mechanism map add value? |
| C4 Phenotype-specific map | C1 plus forecasts from the phenotype-specific transition model (level terms, and interaction terms on the links classed as different) | Do different mechanisms add value? |

- **Primary contrasts.** C4 against C2, and C4 against C3. "Adds value" requires both, as an intersection–union test with each contrast at 5%.
  - Primary metric: out-of-sample log loss.
  - Key secondary metric: net benefit at the clinician-chosen threshold (Vickers 2006).
  - Calibration (intercept and slope), discrimination and the Brier score are reported for every model.
- **Placebo controls.** C4 with permuted phenotype labels in its transition model, and with random maps of the same density. A gain that survives these comes from the model's form, not the mechanisms.
- **Internal validation.** Nested cross-validation by participant. Every step happens inside the training folds: imputation, cohort-derived cut-offs, graph discovery, transition models, tuning, calibration and thresholds.
- **External validation.**
  - The full package is frozen before the external outcomes are seen: code version, rules, models, calibration and thresholds.
  - Then calibration, discrimination, Brier score and net benefit are reported, overall and by phenotype, site, sex and age where precision allows.
  - Recalibration is model updating and needs a new validation.
- **Reporting.** TRIPOD+AI (Collins 2024); risk of bias by PROBAST+AI (Moons 2025).

### 12.5 II-c. Prospective silent validation (outline for a separate protocol)

- **Cohort.** A multicentre observational cohort at movement-disorder clinics outside the PPMI network. It enrols consecutive eligible patients across disease stages and records screening and reasons for non-enrolment.
- **Visits.**
  - Baseline, 6, 12, 18 and 24 months (36 if funded), with actual dates.
  - Structured capture of interim events: falls, admissions, treatment changes, loss of independence, institutionalisation and death.
  - Reasons for missed visits are recorded.
- **Silent use.** The locked model runs at each landmark. Predictions are time-stamped and stored without being shown to clinicians, care is unchanged, and outcomes are ascertained independently.
- **Measurement substudy.** Repeat and inter-rater assessments test how measurement quality affects graph stability and prediction.
- **Sample size.** By external-validation criteria (Riley 2024), from the observed event rate, case mix and site clustering. The minimum data fields are in Appendix C.
- **Timing.** It needs funding, sites, ethics approval and at least 24 months of follow-up after the last enrolment, so it lies outside Paper 2's timeline.
- **Later use.** Any clinician-facing use would be a further, separately approved impact study (DECIDE-AI; Vasey 2022).

### 12.6 II-d. Phenotype-specific treatment effects (outline for a separate protocol)

A registered target-trial emulation (TARGET; Cashin 2025) compares named, feasible strategies in patients eligible for both. Its design:
- time zero, eligibility, grace period, adherence, censoring and outcomes aligned between the strategies;
- the phenotype fixed at time zero, and positivity checked within each phenotype;
- g-methods for time-varying confounding by indication;
- negative controls and sensitivity analyses for unmeasured confounding.

Part I's map informs the choice of confounders; it gives no causal licence by itself.

## 13. Stop/go criteria

- **WP0.** Stop or redesign if the minority phenotype cannot meet WP3's power target even with the secondary axis or merged groups. Apply the interval rule (§6.3).
- **WP3.**
  - If no procedure controls false differences (H-M1), Part I reports only the shared map and the level differences, as associations.
  - If the power gate fails, Analysis B is descriptive.
- **Part I to Part II.** Part II starts only when three conditions hold:
  - at least one confirmatory or replicated-discovery phenotype difference lies on a route to daily function or independence;
  - for II-b, there are enough endpoint events, or an accepted fallback (§12.4);
  - data access and approvals for the external cohort are in place.
- **II-b.** If C4 does not beat both C2 and C3, the result is published as negative, and II-c is not pursued with this model.
- **II-c.** Too few events, incomplete follow-up or unrepresentative sites: extend follow-up, or report the validation as inconclusive.
- **Clinical deployment.** Deployment requires all three of the following:
  - prospective evidence of safety and utility;
  - identified treatment effects for any treatment claim;
  - resolved governance and regulatory requirements.

## 14. Data governance and ethics

- **PPMI.**
  - Under the Data Use Agreement, participant-level tables, the fitted simulator and every per-row array stay on this machine in a git-ignored folder.
  - Results and documents hold aggregates only.
  - No participant-level data go to any external service, including AI tools.
  - Manuscripts go to the PPMI Data and Publications Committee with the required acknowledgement.
- **Gating.** Every command that reads PPMI participant data waits for protocol approval; WP0 runs with the investigator's written approval.
- **Expert questionnaire.** The investigator checks whether the institution requires ethics review, or a formal exemption, for an expert survey. Answers are reported without names, and participation is voluntary.
- **Part II.**
  - Separate ethics approvals and data agreements.
  - For II-c: pseudonymous identifiers with keys held at each site, role-based access, encrypted storage and transfer, audit logs, a documented retention period and an incident process.
  - Consent that covers model validation and future reuse.
- **Provenance.** Each dataset freeze has a manifest: extraction date, participant counts, codebook and instrument versions, exclusions, transformations and file hashes.
- **Version control.** Nothing is committed to git without the investigator's instruction.

## 15. Reporting and dissemination

- **Part I.** STROBE for the observational analyses; the simulation in ADEMP structure (Morris 2019); the expert consensus by ACCORD (Gattrell 2024).
- **Part II.** TRIPOD+AI and PROBAST+AI for prognosis; TARGET for treatment effects; DECIDE-AI for any later live evaluation.
- **Outputs.**
  1. The registered protocol with the simulation specification (OSF).
  2. The clinical paper: the phenotype progression atlas.
  3. A methods paper or section: false phenotype differences and the definition-matched null.
  4. Part II papers, if its gates pass.
- Every result is reported, including null, unresolved and negative findings, and every protocol deviation.

## 16. Reproducibility, compute and quality assurance

- Pre-registration on OSF before WP4, with code and lockfiles for every environment.
- Unit tests for the annual grid, the landmark, the phenotype rules, participant-level splits and exclusions. Seeds, configurations, costs and failures are logged for every fit.
- **Compute.** The machine is an Apple M4 with 10 cores and 16 GB.
  - At most 4 parallel jobs.
  - One fit per job, saved as it finishes and skipped on rerun.
  - Long runs are started from the Terminal app.
- The harness is built and tested on synthetic data before any PPMI run, and costs are measured before replicate numbers are fixed (§8.8).
- An audit report for clinical review lists graph uncertainty, forbidden-edge violations, group sizes and completeness, and what the outputs must not be used for.

## 17. Decisions needed

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
| D12 | Smallest effect of interest Δ | 0.10 SD per SD, shown in clinical units; 0.05 as sensitivity | Panel, statistician and investigator | WP2 |
| D13 | Interval rule | Discrete annual model if at least 90% of intervals lie within 12 ± 3 months | Investigator | WP0 |
| D14 | Compute budget | At most 10 days of 4-job compute for WP3 and WP4; cuts in the order of §8.8 | Investigator | WP3 |
| D15 | Scope of Part II | Outline only in this protocol; II-a and II-b planned after Part I's gate; II-c and II-d as separate protocols | Investigator and supervisors | Now |
| D16 | Part II endpoint | Alive and independent at 24 months, with the fallbacks of §12.4 | Clinical panel and statistician | Before Part II |
| D17 | External cohort | One application, to a cohort with MDS-UPDRS, a cognitive scale and Schwab & England | Investigator | After WP5 |

## 18. Timeline (provisional)

| Weeks | Work |
|---|---|
| 1–2 | WP0 feasibility audit |
| 2–6 | WP1 two-round expert questionnaire, literature search |
| 7 | WP2 protocol freeze and registration |
| 6–11 | WP3 simulator and benchmark, built and tested on synthetic data first (about 7 days of compute) |
| 11–13 | WP4 PPMI analyses |
| 13–16 | WP5 blinded review, replication, sensitivity analyses |
| 16–20 | WP6 manuscripts |
| After week 16, if the gate passes | Part II: external-cohort application and the II-a and II-b protocols, over months; II-c and II-d as separately funded protocols, over years |

## References

- Brumm MC, Siderowf A, Simuni T, et al. Parkinson's Progression Markers Initiative: a milestone-based strategy to monitor Parkinson's disease progression. J Parkinsons Dis 2023;13:899–916. doi:10.3233/JPD-223433.
- Cáncer PF, Arnold M, Estrada E, Voelkle MC. Continuous-time structural equation model forests. Psychol Methods 2025, advance online. doi:10.1037/met0000766.
- Cashin AG, Hansford HJ, Hernán MA, et al. Transparent reporting of observational studies emulating a target trial: the TARGET statement. BMJ 2025;390:e087179.
- Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. BMJ 2024;385:e078378.
- Dadu A, Satone V, Kaur R, et al. Identification and prediction of Parkinson's disease subtypes and progression using machine learning in two cohorts. npj Parkinsons Dis 2022;8:172.
- Debeire K, Gerhardus A, Runge J, Eyring V. Bootstrap aggregation and confidence measures to improve time series causal discovery. CLeaR 2024, PMLR 236:979–1007.
- Driver CC, Voelkle MC. Hierarchical Bayesian continuous time dynamic modeling. Psychol Methods 2018;23:774–799.
- Fereshtehnejad SM, Romenets SR, Anang JBM, et al. New clinical subtypes of Parkinson disease and their longitudinal progression: a prospective cohort comparison with other phenotypes. JAMA Neurol 2015;72:863–873.
- Fereshtehnejad SM, Zeighami Y, Dagher A, Postuma RB. Clinical criteria for subtyping Parkinson's disease: biomarkers and longitudinal progression. Brain 2017;140:1959–1976.
- Gattrell WT, Logullo P, van Zuuren EJ, et al. ACCORD (ACcurate COnsensus Reporting Document): a reporting guideline for consensus methods in biomedicine developed via a modified Delphi. PLoS Med 2024;21:e1004326.
- Gelman A, Stern H. The difference between "significant" and "not significant" is not itself statistically significant. Am Stat 2006;60:328–331.
- Gerhardus A, Runge J. High-recall causal discovery for autocorrelated time series with latent confounders. NeurIPS 2020.
- Günther W, Ninad U, Runge J. Causal discovery for time series from multiple datasets with latent contexts. UAI 2023, PMLR 216:766–776.
- Hähnel T, Raschka T, Sapienza S, et al. Progression subtypes in Parkinson's disease identified by a data-driven multi cohort analysis. npj Parkinsons Dis 2024;10:95.
- Hamaker EL, Kuiper RM, Grasman RPPP. A critique of the cross-lagged panel model. Psychol Methods 2015;20:102–116.
- Hernán MA, Hernández-Díaz S, Robins JM. A structural approach to selection bias. Epidemiology 2004;15:615–625.
- Höglinger GU, Adler CH, Berg D, et al. A biological classification of Parkinson's disease: the SynNeurGe research diagnostic criteria. Lancet Neurol 2024;23:191–204.
- Horváth K, Aschermann Z, Ács P, et al. Minimal clinically important difference on the motor examination part of MDS-UPDRS. Parkinsonism Relat Disord 2015;21:1421–1426.
- Horváth K, Aschermann Z, Kovács M, et al. Minimal clinically important differences for the experiences of daily living parts of the MDS-UPDRS. Mov Disord 2017;32:789–793.
- Huang B, Zhang K, Zhang J, et al. Causal discovery from heterogeneous/nonstationary data. J Mach Learn Res 2020;21.
- Jankovic J, McDermott M, Carter J, et al. Variable expression of Parkinson's disease: a base-line analysis of the DATATOP cohort. Neurology 1990;40:1529–1534.
- Kang K, Gu Y. A blockwise mixed membership model for multivariate longitudinal data: discovering clinical heterogeneity and identifying Parkinson's disease subtypes. Ann Appl Stat 2026;20:452–475.
- Kuiper RM, Ryan O. Drawing conclusions from cross-lagged relationships: re-considering the role of the time-interval. Struct Equ Modeling 2018;25:809–823.
- Lakens D, Scheel AM, Isager PM. Equivalence testing for psychological research: a tutorial. Adv Methods Pract Psychol Sci 2018;1:259–269.
- Li W, Li B, Wu A, et al. Causal discovery for irregularly time series with consistency guarantees (ReTimeCausal). ICML 2026, PMLR 306:67973–67999.
- Lian J, Luo X, Shan C, et al. Personalized progression modelling and prediction in Parkinson's disease with a novel multi-modal graph approach. npj Parkinsons Dis 2024;10:229.
- Lin W, Grewal SS. Measurement fidelity, not model complexity, governs individual cognitive trajectory prediction in Parkinson's disease. npj Digit Med 2026;9:698.
- Macleod AD, Counsell CE. Predictors of functional dependency in Parkinson's disease. Mov Disord 2016;31:1482–1488.
- Mameche S, Cornanguer L, Ninad U, Vreeken J. SPACETIME: causal discovery from non-stationary time series. Proc AAAI Conf Artif Intell 2025;39:19405–19413.
- Marek K, Chowdhury S, Siderowf A, et al. The Parkinson's Progression Markers Initiative (PPMI): establishing a PD biomarker cohort. Ann Clin Transl Neurol 2018;5:1460–1477.
- Mestre TA, Fereshtehnejad SM, Berg D, et al. Parkinson's disease subtypes: critical appraisal and recommendations. J Parkinsons Dis 2021;11:395–404.
- Mooij JM, Magliacane S, Claassen T. Joint causal inference from multiple contexts. J Mach Learn Res 2020;21.
- Moons KGM, Damen JAA, Kaul T, et al. PROBAST+AI: an updated quality, risk of bias, and applicability assessment tool for prediction models using regression or artificial intelligence methods. BMJ 2025;388:e082505.
- Morris TP, White IR, Crowther MJ. Using simulation studies to evaluate statistical methods. Stat Med 2019;38:2074–2102.
- Negida A, Mukhopadhyay N, Berman BD, Barrett MJ. Comparative analysis of progression milestones across Parkinson's disease clinical, pathological, and data-driven subtypes: a 10-year follow-up. npj Parkinsons Dis 2026;12:239.
- Orth U, Meier LL, Bühler JL, et al. Effect size guidelines for cross-lagged effects. Psychol Methods 2024;29:421–433.
- Pal S, Ghosh D, Yang S. Penalized FCI for causal structure learning in a sparse DAG for biomarker discovery in Parkinson's disease. Ann Appl Stat 2026;20:1873–1893.
- Pan W, Su C, Maasch JRMA, et al. Learning phenotypic associations for Parkinson's disease with longitudinal clinical records. AMIA Jt Summits Transl Sci Proc 2024;2024:374–383.
- Reisach AG, Seiler C, Weichwald S. Beware of the simulated DAG! Causal discovery benchmarks may be easy to game. NeurIPS 2021.
- Reisach A, Tami M, Seiler C, et al. A scale-invariant sorting criterion to find a causal order in additive noise models. NeurIPS 2023 (Adv Neural Inf Process Syst 36:785–807).
- Riley RD, Ensor J, Snell KIE, et al. Calculating the sample size required for developing a clinical prediction model. BMJ 2020;368:m441.
- Riley RD, Snell KIE, Archer L, et al. Evaluation of clinical prediction models (part 3): calculating the sample size required for an external validation study. BMJ 2024;384:e074821.
- Runge J. Discovering contemporaneous and lagged causal relations in autocorrelated nonlinear time series datasets. UAI 2020, PMLR 124.
- Saggioro E, de Wiljes J, Kretschmer M, Runge J. Reconstructing regime-dependent causal relationships from observational time series. Chaos 2020;30:113115.
- Siderowf A, Concha-Marambio L, Lafontant DE, et al. Assessment of heterogeneity among participants in the PPMI cohort using α-synuclein seed amplification: a cross-sectional study. Lancet Neurol 2023;22:407–417.
- Simuni T, Caspell-Garcia C, Coffey C, et al. How stable are Parkinson's disease subtypes in de novo patients: analysis of the PPMI cohort? Parkinsonism Relat Disord 2016;28:62–67.
- Simuni T, Chahine LM, Poston K, et al. A biological definition of neuronal α-synuclein disease: towards an integrated staging system for research. Lancet Neurol 2024;23:178–190.
- Spirtes P, Glymour C, Scheines R. Causation, Prediction, and Search, 2nd ed. MIT Press, 2000.
- Stebbins GT, Goetz CG, Burn DJ, et al. How to identify tremor dominant and postural instability/gait difficulty groups with the MDS-UPDRS: comparison with the UPDRS. Mov Disord 2013;28:668–670.
- Stein G, Penzel N, Piater T, Denzler J. TCD-Arena: assessing robustness of time series causal discovery methods against assumption violations. ICLR 2026.
- Tan MMX, Lawton MA, Pollard MI, et al. Genome-wide determinants of mortality and motor progression in Parkinson's disease. npj Parkinsons Dis 2024;10:113.
- van Borkulo CD, van Bork R, Boschloo L, et al. Comparing network structures on three aspects: a permutation test. Psychol Methods 2023;28:1273–1285.
- Varambally S, Ma YA, Yu R. Discovering mixtures of structural causal models from time series data. ICML 2024, PMLR 235.
- Vasey B, Nagendran M, Campbell B, et al. Reporting guideline for the early-stage clinical evaluation of decision support systems driven by artificial intelligence: DECIDE-AI. Nat Med 2022;28:924–933.
- Vickers AJ, Elkin EB. Decision curve analysis: a novel method for evaluating prediction models. Med Decis Making 2006;26:565–574.
- Voelkle MC, Oud JHL, Davidov E, Schmidt P. An SEM approach to continuous time modeling of panel data: relating authoritarianism and anomia. Psychol Methods 2012;17:176–192.
- Zhang J. On the completeness of orientation rules for causal discovery in the presence of latent confounders and selection bias. Artif Intell 2008;172:1873–1896.

The entries added in this draft, and the candidate papers named in the Option B/C draft, were checked against Crossref, Europe PMC, arXiv, OpenReview and publisher pages on 2 November 2026. The WP1 literature search checks the rest and extends the list.

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

## Appendix C. Prospective case-report form: minimum fields (Part II-c)

| Form | Field group | Specification |
|---|---|---|
| Enrolment | Site, pseudonymous ID, screening status, eligibility, consent date | Site-held identifiers kept apart from analysis extracts |
| History | Diagnosis date, motor phenotype, comorbidities, past care | Source-verified; versioned case definitions |
| Visit | Planned date, actual date, visit type, setting, reason for a missed or moved visit | Actual dates mandatory |
| Symptoms | MDS-UPDRS II and III with ON/OFF state, cognition, selected non-motor scales | Instrument version and rater recorded |
| Function | Schwab & England, independence state, support needs, date of change if known | Pre-specified threshold and adjudication |
| Treatment | Medications with start, stop and dose; rehabilitation and other services | Indication recorded where possible |
| Events | Falls, admissions, move to a care home, death | Source, date certainty, adjudicator |
| Follow-up status | Active, withdrawn, lost, died, administrative end | Dropout kept distinct from death |
| Provenance | Entry date, correction history, instrument version | Audit trail and quality flags |

## Appendix D. Changes from the two v0.2 drafts

**From the phenotype-discovery draft v0.2:**
- **Added:**
  - explicit estimands, the smallest effect of interest Δ and five classes per link, so that a structural difference needs an equivalence test, not a non-significant result (§2.3);
  - scenario S2b (strength differences that look like structure);
  - the ADEMP structure for the simulation;
  - a nested, reduced evaluation of Analysis B, which the earlier draft had not costed;
  - a fast learner configuration for resampling, and a compute budget with an order of cuts (§8.8);
  - a permuted-label reference beside the definition-matched null;
  - the interval rule and a continuous-time sensitivity analysis (§6.3);
  - a random-intercept sensitivity analysis;
  - Part II as a conditional outline; reporting guidelines; decisions D12 to D17.
- **Changed:**
  - H-M1 now covers Analysis C as well as Analysis B;
  - the speed-or-pathway rule uses the classes of §2.3;
  - the "out of scope" section is replaced by Part II and §11.

**From the Option B/C draft v0.2:**
- **Kept:**
  - the M0, M1 and M2 vocabulary;
  - the five prognostic models, including the phenotype-label (subtype-only) and shared-map controls, and placebo maps;
  - locked-model external validation and the silent prospective design;
  - death treated as a failure, not a missing value; state and event endpoints kept apart;
  - the reporting standards and the minimum case-report fields.
- **Changed:**
  - "subgroup" is now a clinical phenotype assigned at a landmark;
  - the definition-matched null and equivalence-based classes replace the generic structural test;
  - continuous-time SEM moves from primary candidate to sensitivity analysis and replication tool;
  - Option C becomes a conditional Part II with an event check and a fallback endpoint.
- **Not adopted:**
  - J-PCMCI+ or joint causal inference with the phenotype as a context;
  - MCD, SPACETIME, Regime-PCMCI, ReTimeCausal and continuous-time SEM forests;
  - the prospective cohort as a deliverable of this protocol.

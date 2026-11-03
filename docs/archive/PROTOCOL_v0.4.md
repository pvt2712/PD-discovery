# Phenotype-specific progression mechanisms in Parkinson's disease: speed-adjusted causal discovery in PPMI (Part I) and clinical translation (Part II)

> Protocol draft v0.4, 3 November 2026, for review by the investigator, movement-disorder neurologists and a statistician. It is not approved, and no phenotype analysis of PPMI data has been run.
>
> v0.4 adopts the framework recommended in the investigation report (`docs/pathway_heterogeneity_framework.md`): speed-adjusted subgroup causal discovery. It also takes in three companion documents: the methods and novelty addendum (`docs/methods_and_novelty_update.md`), the Step 2 methods note (`docs/step2_comparison_methods.md`) and the tsLDiffPC/tsDCI note (`docs/tsldiffpc_tsdci_investigation.md`).
>
> Appendix D lists the changes from v0.3, which is archived in `docs/archive/`. Appendix E summarises the synthetic evidence behind the design.
>
> Part I is Paper 2, experiment 1. Part II goes ahead only if Part I passes its gate (§13), and it needs its own data access, funding and ethics approval. Thresholds marked *provisional* are fixed before pre-registration (§17).

## Summary

**Clinical question.**
- Clinicians already sort Parkinson's disease (PD) into phenotypes, such as tremor-dominant (TD) versus postural instability and gait difficulty (PIGD), or by REM-sleep behaviour disorder (RBD). They know that some phenotypes progress faster.
- They do not know whether phenotypes decline through *different pathways* or through *the same pathways at a different speed*. A different pathway would be, for example, loss of independence driven by gait and cognition in one phenotype and by bradykinesia in another.
- Beyond phenotypes, they do not know whether every patient progresses through the same pathway.
- Guidance on monitoring and care is largely phenotype-independent (NICE NG71, the MDS evidence-based reviews, AAN 2021). If pathways differ, what to monitor and when to refer should differ. If only speed differs, the same plan applies at different intervals.

**Five ways progression can differ between groups.** Only the last two are differences of mechanism:
1. *an artefact of the split.* Phenotypes are defined from the same symptom scores the analysis models, so splitting patients on them changes the relationships within each group even when the mechanisms are shared;
2. *speed.* The same mechanism runs faster: every link and every domain's own rate of change scale together (a time scale);
3. *load.* A steady push on some domains (a level difference);
4. *strength.* The same links, of different strength beyond speed;
5. *structure.* A link that matters in one group and is negligible in the other.

**Why yearly comparisons fail.**
- If a phenotype runs the same mechanism s times faster, its yearly effects are a matrix power of the other phenotype's: B_fast = B_slow^s. Three things follow:
  - every yearly coefficient changes;
  - effects that pass through an intermediate domain within the year appear as new yearly "links";
  - the yearly noise grows.
- In synthetic cohorts of PPMI's size where phenotypes differed only in speed, every procedure that compares yearly coefficients declared mechanism differences in every cohort (Appendix E). These included:
  - link-wise interaction tests;
  - a difference-graph search restricted to time order;
  - shared-plus-deviation estimators;
  - the invariance tests of clinical panel research.
- The published difference-graph method for time series, tsLDiffPC/tsDCI, has a more lenient search. It rarely invented cross-links, but it named false "root causes" in about half of the domains and had low power for real differences.
- Speed must therefore be estimated and removed in continuous time, with a speed for each patient.

**Design.**
- **Population.** Part I studies year-to-year transitions in PPMI's sporadic PD cohorts untreated at baseline, about 1,400 participants. The phenotype is assigned at baseline, and the analysed transitions start a year later.
- **One model for every step:** a continuous-time transition model on a shared causal map, with a speed and a load for each patient.
- **Step 1, shared map** (all participants). Latent-aware causal discovery with temporal tiers, with the phenotype as a baseline variable.
- **Step 2, phenotypes: speed, load or mechanism?**
  - A global speed-only test (H-P);
  - speed-adjusted joint discovery of the links that deviate;
  - per-link and per-domain tests;
  - separate graphs against a definition-matched null, as the transparent reference.
- **Step 3, hidden mechanism classes** (PPMI 1.0). Do data-driven progression classes differ only in speed and load, or in mechanism (H-L)?
- **Step 4, individuals** (PPMI 1.0). How much do patients vary in speed and load, and do key links vary beyond speed?
- **Step 5, confirmatory links.** About ten links, pre-specified with clinicians, are tested as speed-adjusted differences, gated on H-P.
- **Simulation benchmark.** It is fitted to PPMI and run before any PPMI analysis. It tests whether the speed-adjusted procedures keep false mechanism differences at the nominal rate where named current methods do not (H-M1).
- **Part II** follows only if Part I finds replicated mechanism differences.

**Primary tests.** H-M1 (simulation); H-P and H-L (PPMI), with Holm control; H-C1 to H-C10, gated on H-P.

**What is new.** The novelty claims are narrow (§1.4):
- the first calibrated test, within a disease, of whether causal mechanisms differ once speed and load are allowed to differ. In PD, it is also the first causal discovery on longitudinal clinical data that compares phenotypes;
- the first test of whether groups or patients share one causal mechanism up to a time-scale factor;
- the first cohort-fitted null that re-applies the phenotype definition rule.

The estimator template is borrowed: a shared vector times a scalar, plus sparse deviations, comes from Mendelian randomisation with invalid instruments and from anchor-free item-bias tests. The contribution is applying it to continuous-time causal dynamics.

**What clinicians get.** A pathway card for each phenotype and each class. It shows:
- the speed relative to the reference;
- the load profile;
- the links that differ beyond speed;
- the routes to loss of independence, split into the part speed explains and the part mechanism explains.

These are hypotheses for monitoring, referral and counselling, not treatment rules.

## 1. Background

### 1.1 Phenotypes in practice

**Motor phenotype.**
- The TD/PIGD classification (Jankovic 1990; MDS-UPDRS version, Stebbins 2013) is the most widely used.
- PIGD is associated with faster motor, cognitive and functional decline (Aleksovski 2018), but the classification is unstable in early PD (Simuni 2016).

**Clinical subtypes.** Fereshtehnejad's subtypes (mild-motor predominant, intermediate, diffuse malignant) are defined from:
- motor severity;
- cognition;
- RBD;
- dysautonomia.

They separate progression rates in PPMI (Fereshtehnejad 2015, 2017).

**RBD and body-first disease.** RBD at diagnosis marks a body-first course in the α-synuclein origin and connectome (SOC) model (Borghammer 2021). The model predicts earlier non-motor load and faster progression along shared propagation routes: mostly speed and load, not different mechanisms.

**Calls for better subtypes.** An MDS task force asked for subtypes that are reproducible and clinically or biologically meaningful (Mestre 2021).

**Data-driven subtypes.** Progression subtypes have been found across cohorts: PPMI with PDBP (Dadu 2022); PPMI, ICEBERG and LuxPARK (Hähnel 2024).

**Biological classifications.** SynNeurGe (Höglinger 2024) and NSD-ISS (Simuni 2024) use α-synuclein seed amplification. PPMI has measured it (Siderowf 2023), but our local download does not hold it.

### 1.2 The gap

Different average trajectories do not show different mechanisms. A phenotype that carries more burden, or runs the same disease faster, will progress faster along the same pathways. The clinically useful distinction between speed and pathway needs a structural (causal) analysis of how domains drive each other. That analysis has to be:
- split finely enough to see differences;
- calibrated so as not to invent them;
- able to tell a faster clock from a different mechanism.

### 1.3 Why it is hard

1. **Phenotypes are defined from graph variables.**
   - TD/PIGD is a ratio of tremor items to gait and balance items. Fereshtehnejad's subtypes threshold motor, cognitive, RBD and autonomic scores.
   - Splitting patients by such a rule does three things:
     - it restricts the range of the defining variables;
     - it makes their causes dependent within each group (collider bias; Hernán 2004);
     - it selects people whose year-0 scores were extreme partly by chance, so their scores then drift back (regression to the mean).
   - Graphs fitted within phenotypes can therefore differ when the mechanisms do not.
   - Permuting group labels does not solve this, because labels computed from the variables are not exchangeable under the null. The Network Comparison Test is one example (van Borkulo 2023).
   - These are "endogenous context variables", for which a formal theory now exists (Günther 2024; Popescu 2026).
2. **Phenotypes are not external contexts.**
   - The disease produces them, and they share hidden causes (overall pathology, baseline state) with every later symptom.
   - In Step 1 the phenotype is a baseline variable in a latent-aware search, which amounts to joint causal inference with only its exogeneity assumption (Mooij 2020).
   - In Steps 2 to 5 the phenotype acts on speed, load or links in a transition model. Its effects signal heterogeneity, not causal effects of the label.
3. **Hidden person-level burden.** A stable burden that drives many domains creates year-to-year associations between domains without any causal link between them (Hamaker 2015). It also creates level differences between phenotypes everywhere.
4. **Absence cannot be shown by non-significance.** A link that is significant in one phenotype and not in the other is no evidence of a difference (Gelman 2006). Absence can only be supported as "smaller than a pre-set negligible size", by an equivalence test (Lakens 2018).
5. **In yearly data, speed looks like a different pathway.**
   - Suppose two groups share one mechanism and one runs it s times faster. Their yearly effects are then related by a matrix power, B(fast) = B(slow)^s, which has four consequences:
     - every yearly coefficient changes, persistence most;
     - an effect passing through an intermediate domain within the year appears as a yearly "link";
     - the yearly noise grows;
     - a faster group can even show a *weaker* yearly link once one year passes the link's peak (Kuiper 2018).
   - When patients within a phenotype run at different speeds, group-average yearly dynamics are no longer a matrix power at all.
   - Two literatures describe the same problem: coarse sampling (Plis 2015; Hyttinen 2017; Gong 2015) and interval dependence in panel models (Voelkle 2012; Kuiper 2018; Driver 2025).
   - In continuous time, speed is a single factor, and which links exist does not depend on it. The comparison is therefore made there, with a speed for each patient (§7.1).
6. **Few annual visits.**
   - PPMI gives on average 3 (PPMI 2.0) to 9 (PPMI 1.0) annual visits per person.
   - Methods built for long series reduce here to a pooled search over two time slices. Examples are PCMCI+ (Runge 2020), LPCMCI (Gerhardus 2020), J-PCMCI+ and regime methods.
   - The unit of analysis is the year-to-year transition, pooled over participants.
7. **Treatment feedback.**
   - PPMI participants are untreated at enrolment, and most start dopaminergic treatment during follow-up.
   - Treatment responds to symptoms and changes them.
   - It improves motor scores on its own timetable (Holford 2006; Vu 2012), so it is not scaled by a patient's disease speed.
8. **Small groups.** The minority phenotype may hold a few hundred participants. In groups that small, an absent link usually means low power.

### 1.4 Prior work and novelty

The following are established, and this study claims none of them:
- clinical phenotypes and their prognosis (§1.1);
- data-driven longitudinal progression subtypes across cohorts;
- longitudinal symptom-association networks and graph-informed prognosis in PD;
- causal discovery across contexts, mixtures and regimes, and benchmarks of time-series causal discovery under violated assumptions;
- causal discovery on cross-sectional PPMI biomarkers (Penalized FCI; Pal 2026);
- group comparisons of causal or temporal networks in other diseases:
  - Alzheimer's stages;
  - psychiatric trajectory classes;
  - boys versus girls in a birth cohort, with a formal test (Larsen 2026);
  - many molecular subtypes;
- methods that learn where two groups' causal or temporal models differ:
  - difference graphs (Wang 2018; Malik 2024; Bystrova & Devijver 2026; Ruer 2026);
  - shared-plus-deviation estimators (Gross & Tibshirani 2016; Fisher 2022; Crawford 2026; Jiang & Zhang 2023);
  - group-invariance tests for panel networks (Mulder & Hamaker 2021; Epskamp 2020; Du 2026);
- the estimator template of a scalar times a shared vector plus sparse deviations, with tests:
  - Mendelian randomisation with invalid instruments (Kang 2016; Bowden 2016; Windmeijer 2019);
  - anchor-free item-bias tests (Chen Y 2023; Halpin 2024);
  - tests of equality up to scale (Swait & Louviere 1993; Allison 1999);
  - person-level random scale in choice models (Fiebig 2010).

| Prior work | Overlap | What it leaves open |
|---|---|---|
| Multi-cohort progression subtypes (Hähnel 2024) | Longitudinal subtyping in PPMI, ICEBERG and LuxPARK | Do subtypes differ in mechanism, or only in trajectory or speed? |
| Phenotypic association graphs from longitudinal records (Pan 2024) | Symptom association graphs over time in PPMI, PDBP and BioFIND | Do the associations survive causal assumptions and a speed-adjusted phenotype comparison? |
| Multi-modal graph prognosis, AdaMedGraph (Lian 2024) | Graph-informed progression prediction in PPMI and PDBP | Do phenotype-specific mechanisms add to the phenotype label and to a patient's speed? |
| Blockwise mixed-membership subtyping (Kang 2026) | Heterogeneity across patients, domains and time | Are the differences structural, or speed? |
| Progression milestones across clinical, pathological and data-driven subtypes (Negida 2026) | Outcomes compared between subtype systems over 10 years | Mechanism, once selection, severity and speed are accounted for? |
| Measurement fidelity in cognitive trajectory prediction (Lin 2026) | The limits measurement sets on individual prediction | Does measurement quality also limit graph recovery? |
| Motor subtype and later working memory (Johnson 2016) | A directed model of TD versus PIGD over two waves | Pre-specified paths; disease duration differed by subtype, so stage and speed are mixed |
| MRI propagation by RBD status (Pyatigorskaya 2021) | Directed propagation models compared by RBD | Imaging, cross-sectional; speed not separated |
| A shared causal manifold across AD, PD and ageing (Saadawy 2026, preprint) | "One graph, different rates" across diagnoses | No formal test; compares diagnoses, not phenotypes within PD |
| Causal discovery in a birth cohort by sex (Larsen 2026, preprint) | A formal test of group differences in a causal graph | Outside PD; speed not separated |
| Difference graphs for time series (Ruer 2026) | Where two regimes' lagged mechanisms differ | One long series per regime; reads speed as persistence changes (Appendix E) |
| Panel network comparison (IVPP; Du 2026) | A global invariance test, then per-edge pruning, in panel data | Yearly-scale coefficients; no speed |
| Network Comparison Test (van Borkulo 2023) | Testing network differences between groups | Cross-sectional only; group labels must be exchangeable |
| TCD-Arena (Stein 2026) | Robustness of time-series causal discovery to violated assumptions | Short multi-patient panels with groups defined from graph variables |

**Open question.** Do clinical phenotypes, data-driven classes or individual patients differ in how symptom domains drive each other over time? If so, is the difference one of speed, load, strength or structure? Each must be judged once four things are accounted for:
- the artefacts of defining phenotypes from the same variables;
- hidden burden;
- unequal group sizes;
- the observation process.

**What remains new** (searches of 2 November 2026, with citation tracking; the addendum's §2):
- No study has separated speed and load from mechanism when comparing causal or temporal networks between patient groups. Several define their groups by speed or outcome, which builds in the trap of §1.3, point 5.
- No study has tested whether groups or persons share one multivariate causal mechanism up to a time-scale factor. Proportional drift appears in the literature only as a change of time unit (Kuiper 2018).
- No study has used a cohort-fitted null that re-applies the group-definition rule.
- No PD study was found that formally tested whether lagged effects differ between patient subgroups.

The WP1 literature search, with its databases, queries and dates recorded, confirms or corrects this before any claim of priority. Four sources remain to be read in full: Larsen 2026, Saadawy 2026, Rocchi 2026, and the cited-by lists of four method papers.

### 1.5 Pilot evidence from the escalation-discovery experiment

The stopped experiment used the same PPMI download. No code is reused; these are its lessons (details in Appendix B).

| Observation | Consequence for this design |
|---|---|
| 1,741 PD participants pooled; 8.9 / 3.1 / 5.9 annual rows per participant in PPMI 1.0 / PPMI 2.0 / genetic PD | PPMI 2.0 adds people but little follow-up. PPMI 1.0 carries long-run progression and the class and individual analyses (Steps 3 and 4) |
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

- **Q1 (one pathway or several).** Does every patient progress through the same pathway? Asked for phenotypes (Step 2), data-driven classes (Step 3) and individuals (Step 4).
- **Q2 (speed, load or mechanism).** Where groups differ, is it speed, load, or mechanism (strength or structure)? Which links and which domains differ beyond speed?
- **Q3 (routes to loss of independence).** For each phenotype and class, which domains feed next year's daily function and independence? How much of any difference does speed explain?
- **Q4 (fit of the expert map).** Does a clinician-built progression map fit the data equally well in each phenotype?

### 2.2 Methodological question

- **M1.** At PPMI's sizes, do current methods for comparing groups' causal or temporal networks misread speed and load differences as mechanism differences, while the speed-adjusted procedures do not? Can the speed-adjusted procedures localise a real difference? The current methods are:
  - link-wise tests;
  - difference graphs;
  - shared-plus-deviation estimators;
  - group-invariance tests;
  - separate graphs.

### 2.3 Model, estimands and classes

**Model** (§7.1).
- A participant i with phenotype g follows

  dX = s_i·[(A + Δ_g)·X + b_g]·dt + √s_i·G_g·dW, with log s_i = γ_g + β′z_i + σ·u_i, u_i ~ N(0, 1).

- It is observed at the annual visits, using the actual intervals. Treatment enters as a symptomatic effect on the observed scores (§6.1).
- The reference phenotype (TD on the primary axis) has Δ = 0 and γ = 0.

**Estimands.**
- **Speed ratio** exp(γ_g): how much faster phenotype g runs the reference mechanism.
- **Speed spread** σ: the standard deviation of log-speed between patients within a phenotype.
- **Load** b_g: the steady push on each domain, per year at the reference speed. **Noise** G_g.
- **Link strength** A_jk: the direct effect of domain k on domain j, per year at the reference speed. For clinicians it is also reported as the yearly effect at each phenotype's typical speed.
- **Speed-adjusted deviation** Δ_jk: how much the link differs in phenotype g once both phenotypes run at the same speed. A link "differs beyond speed" when Δ_jk ≠ 0.
- **Share explained by speed:** 1 − ‖A_g − ŝ·A_TD‖² / ‖A_g − A_TD‖², from free fits of each phenotype's links.
- **Total effect** of domain X on independence over h years. It is split into a speed part and a mechanism part (§7.8).
- **Smallest effect of interest ε** (called Δ in v0.3).
  - Definition: the size below which a deviation is clinically negligible. *Provisional:* 0.10 standard deviations of the target domain per standard deviation of the source domain, per year at the reference speed.
  - The panel sees it in clinical units, and it is fixed in WP2 (D12). ε = 0.05 is a sensitivity analysis.
  - For comparison, psychological panel studies call standardised cross-lagged effects of 0.03, 0.07 and 0.12 small, medium and large (Orth 2024).

Effects are reported per minimal clinically important difference (MCID) of the source domain where one exists, and per standard deviation otherwise. Two examples:
- 3.25 points of MDS-UPDRS III (Horváth 2015);
- 3.05 points of MDS-UPDRS II for improvement, and 2.51 for worsening (Horváth 2017).

**Classes per link.** Each tested link falls into one class, judged on its speed-adjusted deviation. M0, M1 and M2 are the critical review's model classes, here applied to one link.

| Class | Model class | Rule |
|---|---|---|
| Similar | M0 | The 90% confidence interval of Δ_jk lies within ±ε (an equivalence test at 5%) |
| Stronger beyond speed | M1 | Δ_jk ≠ 0 under the family's false-discovery control, and the link is non-zero with the same sign in both phenotypes at a common speed |
| Structurally different | M2 | Δ_jk ≠ 0, and either the link is non-zero in one phenotype with its 90% interval within ±ε in the other, or it has opposite signs |
| Different, type unresolved | M1 or M2 | Δ_jk ≠ 0, but neither of the two rules above holds |
| Unresolved | — | None of the above. Reported with the largest deviation the data allow (the upper confidence bound of \|Δ_jk\|) |

**Domain verdicts** (§7.7) replace v0.3's speed-or-pathway rule. For every domain the decomposition reports four things:
- the phenotype's speed, which is common to all domains;
- its load difference;
- its noise difference;
- whether any link into it differs beyond speed.

A phenotype whose domains all show "no mechanism difference" differs by **speed and load only**.

At PPMI's sizes many links will be unresolved. They are reported as unresolved, never as "no difference" or "absent". WP3 measures how often each class can be reached at ε = 0.10 and at ε = 0.05.

### 2.4 Hypotheses

- **H-M1 (primary methodological test, simulation).** Consider shared mechanisms with speed and load differences, hidden burden and definition artefacts: scenarios S0, S0L, S1, S1V, S4, S6 and S7 (§8.2).
  - The named comparators (§8.4) declare mechanism differences for more than 5% of truly shared links, or name false root-cause domains.
  - The speed-adjusted procedures stay at or below 5%, within two Monte Carlo standard errors.
- **H-P (primary, PPMI).** The primary phenotypes differ in mechanism beyond speed and load.
  - Statistic: the likelihood ratio in the patient-level speed model, comparing the phenotype acting on speed and load only with it acting on the links as well.
  - Reference: the extended definition-matched null (R = 200; §7.3).
  - Reported with the share of the difference that speed explains.
- **H-L (primary, PPMI 1.0).** Data-driven progression classes differ in mechanism beyond speed and load.
  - Statistic: the likelihood ratio of speed classes against mechanism classes at the selected number of classes, with a random speed within each class.
  - Reference: a parametric bootstrap from the fitted speed-class model that repeats the choice of the number of classes (R = 200).
  - Descriptive only if the power gate fails (§8.7).
- **Multiplicity of the primary PPMI tests.**
  - Holm across the primary family: the smaller p-value is tested at 2.5% and the other at 5%.
  - If the panel makes RBD status a co-primary axis (D18), the family becomes H-P (TD/PIGD), H-P (RBD) and H-L, under Holm at an overall 5%.
- **H-C1 to H-C10 (gated confirmatory tests, PPMI).**
  - About ten pre-specified links, each with an expected direction beyond speed (candidates in Appendix A), frozen with the clinical panel in WP1.
  - Tested only if H-P rejects, as speed-adjusted per-link tests (§7.6), with Benjamini–Hochberg control at 5% across the family.
- **H-D (key secondary).** Which domains differ in mechanism beyond speed? Per-domain tests (§7.3), with Benjamini–Hochberg control across domains.
- **H-I (key secondary, PPMI 1.0).**
  - How much do individuals vary in speed, and how much of that do phenotype, RBD status and age at onset explain?
  - Do the two key links vary beyond speed? This part is exploratory.
- **H-V (key secondary).** Are mechanism classes, if found, reproducible and useful? Judged by:
  - stability;
  - replication by era;
  - biology not used in fitting;
  - prediction of class membership at baseline;
  - blinded clinician review;
  - prognostic value in Part II.
- v0.3's H-S, "most differences are speed", is retired. H-P and H-L answer it.
- Part II's questions (§12.1) are not tests of Part I.

## 3. Design overview and governance

| Work package | Content | Output | Gate to continue |
|---|---|---|---|
| WP0 Feasibility audit | Aggregate counts only: phenotype and RBD group sizes, visits and follow-up per participant by era, transitions, missing data, phenotype stability, treatment timing, actual visit intervals, and events for Part II's endpoint | Feasibility tables for the clinical panel | The minority phenotype is large enough for the WP3 power target; PPMI 1.0 follow-up is enough for Steps 3 and 4 |
| WP1 Clinical consensus | A two-round questionnaire completed by at least two movement-disorder neurologists, plus a literature search. It settles: phenotype definitions and axes (including RBD); domains; the expert map; confirmatory hypotheses and ε; the two key links; and the clinicians' forecasts of speed, load or mechanism | Frozen definitions, map and hypothesis family; recorded forecasts | Panel decisions recorded |
| WP2 Protocol freeze | Final protocol and statistical analysis plan | Registered protocol | Investigator approval |
| WP3 Simulation benchmark | PPMI-fitted simulator with patient-level speed and load; scenarios S0 to S7 with stresses; the named comparators; validation of the continuous-time engine and of its second host | Settings, calibration, power and class recovery at real sizes; the H-M1 result | H-M1 holds for the speed-adjusted procedures; the power gate decides the status of H-L |
| WP4 PPMI analyses | Steps 1 to 5; the expert-map test | Phenotype and class progression atlas; pathway cards | — |
| WP5 Review and robustness | Blinded clinician review; replication by era; sensitivity analyses; comparison with the recorded forecasts | Graded findings | Part II gate (§13) |
| WP6 Reporting | Clinical paper; methods paper | Manuscripts for the PPMI Data and Publications Committee | — |
| Part II (conditional) | External replication; prognostic value; outlines of a prospective validation and a treatment-effect study | Separate protocols and approvals | §13 |

No phenotype-split, class or individual-level analysis of PPMI is run before WP2: the clinicians build the map, choose the hypotheses and record their forecasts without seeing data-driven results.

**Oversight.**
- Part I: the investigator, the supervisors, a statistician and the clinical panel.
- Part II adds allied-health clinicians, site coordinators, and patient and public representatives. It also needs independent data-governance and clinical-safety review before any prospective recruitment.

## Part I. Phenotype causal discovery in PPMI

## 4. Population and data

- **Source.** PPMI (Marek 2018), under the investigator's Data Use Agreement, from the local download (clinical, imaging-summary and genetic files; no biospecimen category). No external cohort is accessible; Part II needs applications.
- **Primary population.** Sporadic PD untreated at the baseline visit, which makes every phenotype assignment an OFF-medication assessment:
  - PPMI 1.0 de novo PD (431 in the pilot build, about 9 annual visits each);
  - PPMI 2.0 sporadic PD (957 in the pilot build, about 3 annual visits each; untreated status at baseline to be confirmed in WP0).
- **Which data carry which step.**
  - Steps 1, 2 and 5 use both cohorts.
  - Steps 3 and 4 rest on PPMI 1.0. In a synthetic check, latent classes were recovered well with about seven transitions per person, and poorly with two (Appendix E).
  - PPMI 2.0 replicates class-level results only.
- **Secondary population.** Genetic PD, often treated at enrolment and later in the disease, is analysed separately as exploratory.
- **Annual grid.**
  - For each participant and year, the visit closest to the enrolment anniversary within ±4 months, with the most complete core battery.
  - The actual visit dates are kept, because the continuous-time engine uses the actual intervals (§6.3).
- **Landmark.** The phenotype is assigned at year 0. The primary progression analyses use transitions from year 1 onwards (year *t* to *t*+1, *t* ≥ 1). Four consequences:
  - Neither the year-0 values that define the phenotype nor their measurement error enter an analysed row, which removes regression to the mean from those rows.
  - The phenotype label is never a function of variables in the same row.
  - The range restriction and shared causes that remain are handled by the definition-matched null (§7.3).
  - Including the year 0 → 1 transition is a sensitivity analysis. WP0 reports how many transitions the landmark leaves per phenotype and era.
- **Unit of resampling.** The participant, in every subsample, bootstrap and cross-fit.
- **Exclusions.**
  - Participants without a usable baseline phenotype are excluded.
  - A later change of diagnosis (for example to multiple system atrophy) stays in the primary analysis, because excluding it uses future information. Excluding it is a sensitivity analysis.

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
| RBD status | RBD screening questionnaire above a cut-off at baseline | RBDSQ | Recognisable; prognostic for cognition; a proxy for body-first disease, for which the SOC model predicts speed and load differences along shared routes (Borghammer 2021) | A questionnaire, not polysomnography; misclassification pulls differences towards zero |
| Clinical subtype: mild-motor / intermediate / diffuse malignant | Motor summary, cognition, RBD and dysautonomia scores against 75th-percentile cut-offs (Fereshtehnejad 2015, 2017) | MDS-UPDRS, MoCA and cognitive battery, RBD questionnaire, SCOPA-AUT | Prognostic; validated in PPMI | Built from many graph variables; small diffuse-malignant group; cohort-dependent cut-offs |
| Age at onset | Early, typical or late onset (cut-offs in WP1) | Diagnosis history | Exogenous; large groups | A demographic axis more than a phenotype |
| Genotype (exploratory) | GBA or LRRK2 carrier versus non-carrier | Genetic consensus file | Biological; exogenous | Small groups; carriers enrolled differently |

### 5.3 Provisional choice

- **Primary axis: motor phenotype, TD versus PIGD**, the clinicians' daily language.
  - Participants with an indeterminate phenotype are left out of the primary analyses.
  - They are included in a sensitivity analysis, merged with PIGD.
- **Proposed co-primary axis: RBD status** (D18; panel, round 2).
  - The SOC model predicts the class "similar" for most links: speed and load, not mechanism.
  - A stricter RBD definition is a sensitivity analysis.
- **Secondary axis: Fereshtehnejad clinical subtype**, contrasting diffuse malignant with mild-motor predominant.
- **Supportive analysis.** Continuous baseline tremor and gait scores act on speed and load in the same model. This avoids thresholds.
- **Sensitivity or exploratory axes:** age at onset and genotype.
- The clinical panel confirms or replaces these in WP1, before any phenotype analysis.

### 5.4 Definition rules

- The phenotype comes from the year-0 (baseline) visit. Participants are untreated then, so it is an OFF-state assessment by construction.
- Published cut-offs are used as published.
  - A cohort-derived cut-off (Fereshtehnejad's percentiles) is computed once on the pooled primary population and frozen.
  - In Part II's cross-validation it is recomputed within each training fold.
- Nodes share no items. MDS-UPDRS II items 2.10 (tremor), 2.12 (walking and balance) and 2.13 (freezing) leave the daily-function node, because tremor and gait have their own nodes.
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
| RBD symptoms | RBDSQ | Optional; not a node when RBD status is a phenotype axis |
| Motor complications | MDS-UPDRS IV (dyskinesia, fluctuations) | Optional; zero before treatment |
| Dopaminergic treatment | Total levodopa-equivalent daily dose | Time-varying; the dose recorded at year *t*+1 precedes the other year-*t*+1 measurements |

**Motor state after treatment.** The state used after treatment starts (OFF where available, or ON) is decided in WP1. The other state is a sensitivity analysis.

**Treatment in the continuous-time engine (D24).**
- It enters as a symptomatic effect of the current dose on the observed motor scores at the visit, following pharmacological progression models (Holford 2006; Vu 2012).
- It is not scaled by the patient's speed, and its effect is shared across phenotypes in the primary model.
- An effect of treatment on the underlying drift is a sensitivity analysis.

**Baseline and context variables:**
- age, sex, education and disease duration at baseline;
- baseline dopamine-transporter binding (putamen);
- the phenotype labels;
- the enrolment era (PPMI 1.0 or 2.0), as an exogenous context.

With about ten core domains in two years and seven baseline variables, the Step 1 transition table has about 27 columns.

### 6.2 Temporal knowledge given to every learner

- Tiers, in order: era | baseline variables, including the phenotype labels | year *t* | treatment at *t*+1 | the rest of year *t*+1.
- No edge runs from a later tier into an earlier one, and nothing causes era, age or sex.
- Within a tier the data decide.

### 6.3 Time step and visit intervals

- **Step 1** uses two-slice tables on the annual grid. A step across a missed year (two years) is not used there.
- **Interval rule for Step 1 (pre-specified).** WP0 reports the distribution of actual intervals between analysed visits.
  - If at least 90% lie within 12 ± 3 months, Step 1 uses the annual tables as they are.
  - Otherwise, the interval enters Step 1's follow-up models as a covariate.
- **Steps 2 to 5 use the continuous-time engine** (§7.1) with the actual interval of every transition. Irregular spacing is therefore handled exactly. A transition across a missed visit enters as a two-year transition; excluding such transitions is a sensitivity analysis.
- **Interpretation.** The engine's drift entries are direct effects per year at the reference speed, under the model, relative to the measured domains. They are not identified causal arrows by themselves: Step 1's map supplies the causal reading.

### 6.4 Expert phenotype-progression map (built in WP1)

- **Rating links.** Each neurologist completes the expert questionnaire (`docs/clinical_expert_questionnaire.md`) independently.
  - Its Section E rates every ordered pair of domains: could X this year directly change Y next year?
  - Codes: 2 strong, 1 weak, 0 none, ? unsure, with a minus sign for an effect that improves Y.
- **Other questions.**
  - The questionnaire also asks for impossible links, and for unmeasured factors that drive several domains (for example overall pathology burden).
  - Its answers on minimal clinically important differences (Section H) inform ε.
- **Consensus rules.**
  - A link enters the map when every rater gives it 1 or 2, and stays out when every rater gives it 0.
  - Every other link returns in round 2 with the group's anonymised answers. After round 2 the majority decides; a tie is settled by the investigator and recorded as disputed.
  - The same two rounds settle the phenotype definitions and axes, the domains, the confirmatory hypotheses and the two key links (D1 to D7, D18, D20).
  - Round 2 also shows the panel ε in clinical units, computed from WP0's standard deviations, for confirmation (D12).
- **Forecasts (D21).** Round 2 records each neurologist's forecast for each phenotype axis: will its differences be speed, load or mechanism, and in which domains? The forecasts are compared with the results only after analysis.
- **Use of the map.** It is frozen before WP3 and serves as:
  - the skeleton of the simulator;
  - with Step 1's discovered links, the support of A and Δ in Steps 2 to 5 ("every link any group might use");
  - the source of adjustment sets in the reference analyses;
  - the target of the Q4 test.

## 7. Analyses

### 7.1 One engine for Steps 2 to 5

**Model.**
- **Continuous-time dynamics.** Each participant's domains follow the continuous-time model of §2.3. The links are A + Δ_g, the load is b_g, and the noise G_g; all three are run at the participant's own speed s_i.
- **Speed.** Log-speed depends on the phenotype (or class), on optional baseline characteristics z_i, and on a random part with standard deviation σ.
- **Load.** In Step 4, and in the random-load sensitivity analyses, a participant-level random load adds to b_g.
- **Yearly transitions.** They follow exactly from the continuous-time model over each actual interval:
  - the transition matrix is the matrix exponential of the links times the interval and the speed;
  - the intercept and the noise covariance follow from the same matrix exponential (Van Loan's method).
- **Measurement.** Scores enter on Gaussian-copula (normal-scores) scales. A version with measurement error is a sensitivity analysis (§7.12).

**Estimation.**
- Maximum likelihood, integrating each participant's log-speed by Gauss–Hermite quadrature (12 nodes).
- Exact gradients and second derivatives are taken through the matrix exponential by automatic differentiation (PyTorch).
- The likelihood was validated against an independent implementation to machine precision (Appendix E).
- In the synthetic checks, the full Step 2 suite took about 30 seconds per cohort of 1,050 patients with six domains. That suite is the speed-only and free fits, the per-link and per-domain tests, and 12 penalised fits. Costs at ten domains are measured in WP3.

**What separates speed from mechanism.**
- Speed moves every link and every domain's own rate together; a mechanism difference moves only some. The sparsity of Δ therefore identifies the speed.
- **The identifying rule.** The rule is minimal-L1: speed is the weighted median of the ratios of the two phenotypes' continuous-time entries, weighted by the size of the reference entries (Chen Y 2023; Bowden 2016).
- **When the rule is exact.** It is exact while the deviating entries carry less than half of the weight and are not all of one sign.
- **Definition of speed.** A uniform strengthening of every link *and* every domain's own rate is speed by definition.
- **Anchored version (sensitivity).** It assumes that each domain's own rate shares the speed.
- **When identification fails** (§7.6), the study reports the share explained by speed, with a sensitivity analysis under a plurality rule.

**Second host.**
- ctsem 3.11 can express the same model: a speed parameter with a random effect and a phenotype effect, referenced in every drift entry (Driver 2017; Driver, Voelkle & Oud 2026).
- It is validated by simulation in WP3, then used for the continuous-time check (§7.12).

### 7.2 Step 1: one shared causal map, all participants (Analysis A)

**Learner.**
- A latent-aware, constraint-based learner with tiers (FCI type; Spirtes 2000; Zhang 2008).
- Its theory with tiered background knowledge is now established (Bang & Didelez 2023, 2025).
- The phenotype labels sit in the baseline tier. This is joint causal inference with only the exogeneity assumption (Mooij 2020).

**Tests.**
- When a test conditions on the phenotype, it is stratified by phenotype, so that a link present in one phenotype only is not averaged away.
- WP3 chooses the conditional-independence test by accuracy and measured cost from three candidates:
  - Fisher-z on Gaussian-copula scores with a cluster correction;
  - fedCI's likelihood-ratio tests for mixed and ordinal data (Hahn 2026);
  - a test robust to unequal variance, because patient-level speed makes residual variance depend on the current state.
- **Missing data:** test-wise deletion, and multiple imputation inside the tests (Witte 2022).

**Fast configuration for resampling.**
- Inside subsamples, nulls and the simulation, the same learner runs with the uncorrected Fisher-z test on Gaussian-copula scores.
- Every resampling decision is calibrated against simulated data with the same participant structure.
- Target cost: at most one minute per fit on a table of the real size, measured in WP3.

**Cross-checks and controls.**
- **Cross-checks:**
  - S-FCI with phenotype or era as the domain (Li 2023);
  - tiered IOD if PPMI 1.0 and 2.0 do not share every variable (Bang & Didelez 2025);
  - BOSS-FCI, or FCI with targeted testing, as a sensitivity learner if memory allows (Ramsey 2025).
- **Control.** A sortability baseline that ignores conditional independence (Reisach 2021). Variance- and R²-sortability are reported for real and simulated data (Reisach 2023).
- At most three configurations per learner, chosen on simulated data only. One fit per job, saved as it finishes.

**Output.**
- The map's links: their union with the expert map's links is the support of Steps 2 to 5.
- Marks for possible hidden confounding.
- The domains whose next-year value differs by phenotype among people with the same current state.

Those links signal heterogeneity, and Step 2 says whether it is speed, load or mechanism.

**Methods considered and their place.**

| Method | Decision | Reason |
|---|---|---|
| PCMCI+, LPCMCI | Not used as separate methods | With 1 to 8 transitions per person they reduce to the pooled two-slice search with tiers used here |
| Bagged-PCMCI+ (Debeire 2024) | Idea kept, tool not used | The participant bootstrap (§7.10) is the panel equivalent |
| Joint causal inference | Used in Step 1, with exogeneity only | The phenotype is not an external context (§1.3); J-PCMCI+ took over 29 minutes per fit in the pilot |
| Continuous-time models (Driver 2018) | The core engine of Steps 2 to 5; ctsem is the second host | Speed is a single factor only in continuous time |
| DynNet (Rouanet 2026) | Named naive comparator (§8.4) | It cannot host a patient-level speed without code changes; group effects enter each transition entry additively |
| Difference graphs: DCI, DDAG, LDiffPC, tsLDiffPC and tsDCI | Named comparators; the idea of testing against a re-targeted reference is kept, but done in one joint continuous-time fit | Speed changes yearly coefficients densely and breaks their equal-noise premises. tsLDiffPC/tsDCI's lenient search rarely flags cross-links but names false root causes through persistence, with low power; re-targeting removes the false root causes but not the low power (Appendix E; tsLDiffPC/tsDCI note) |
| Shared plus deviations: data-shared lasso, multi-VAR, NetANOVA | Precedents for Step 2's layout; the first two are named comparators | None gives calibrated tests. multi-VAR needs long series. NetANOVA needs an instrument per node |
| Mixtures of causal models (for example MCD; Varambally 2024) | Used in Step 3, as mixtures of transition models with a random speed | PPMI 1.0's follow-up supports two to four classes; power gate (§8.7) |
| Continuous-time SEM forests (Cáncer 2025) | Not used | Splitting on phenotype-defining variables recreates the artefact of §1.3 |
| SPACETIME (Mameche 2025), Regime-PCMCI (Saggioro 2020), ReTimeCausal (Li 2026) | Not used | Temporal regimes and long series; PPMI's intervals are handled by the engine |

### 7.3 Step 2: phenotypes, speed, load or mechanism

**7.3.1 Global speed-only test (H-P).**
- **Null model.** The phenotype acts on log-speed, load and noise only, with a common mechanism A. Each participant has their own speed.
- **Alternative.** The phenotype also acts on the links (A_g free).
- **Statistic.** The likelihood ratio between the two.
- **Reference distribution.** The extended definition-matched null (§7.3.6), which includes a parametric bootstrap of the fitted null model, resampling participants (R = 200).
  - A χ² reference held in the synthetic checks (5–6.5%).
  - The bootstrap is used because the real data add ordinal scores, dropout and burden, and because likelihood-ratio invariance tests in panel networks have proved "prone to false discovery" (Du 2026).
- **Why speed is per participant.** A test that gives every participant in a phenotype the same speed rejected far too often when speeds varied within phenotypes: 10.5% at PPMI's size, and up to 65% in cohorts 16 times larger. The participant-level test stayed at 4.6% (Appendix E).
- **Reporting.** The test is reported with the share of the phenotype difference that speed explains (§2.3), with its cross-validated fit beside it.

**7.3.2 Speed-adjusted joint discovery (exploratory localisation).**
- **Model.** Both phenotypes' maps are estimated together: PIGD's links equal TD's links run at PIGD's speed, plus sparse deviations Δ.
- **Penalty.** An adaptive lasso on Δ, with the domains' own rates included and speed left unpenalised.
  - The weights come from free fits at the weighted-median speed.
  - This is the median-initialised adaptive lasso, which selects consistently (Windmeijer 2019).
- **Tuning.** BIC with the number of transitions in the penalty, with eBIC as a sensitivity analysis. Each selected support is refitted.
- **Stability selection.** Over random halves of the participants (Shah & Samworth 2013), reporting selection frequencies.
- **Intervals.** Participant-bootstrap intervals for the selected deviations. If selected links are also tested, sample splitting is used.
- **Synthetic check.** It flagged 0.4% of truly shared links, and isolated a real difference in 96–97% of cohorts (Appendix E).
- **Not imported from shared-plus-deviation software:**
  - per-person standardisation;
  - cross-validated lasso selection read as a test;
  - p-values read naively off a bootstrap of penalised fits.

**7.3.3 Per-link tests.** Every link's deviation is tested within one joint fit, never one link at a time against the speed-only fit. The one-at-a-time version spilled onto neighbouring links when one link truly differed (Appendix E).
- **Primary, anchor-free.**
  - Fit both phenotypes' continuous-time links freely and identify the speed by the minimal-L1 rule.
  - Each deviation's p-value comes from Monte Carlo draws of the free fit (the method of Chen Y 2023).
  - Synthetic check: 0.25–0.4% of shared links flagged, and the true link isolated in 89–94% of cohorts.
- **Sensitivity, anchored.**
  - Wald tests with every cross-link deviation free and each domain's own rate shared.
  - Synthetic check: 0.4–1.0% of shared links flagged, and the true link isolated in 91–92% of cohorts.
- **Families.**
  - Over all map links these tests are exploratory, with Benjamini–Hochberg control.
  - Over the pre-specified family they are Step 5's confirmatory tests.

**7.3.4 Per-domain tests (H-D).**
- From the same joint fits, all deviations into one domain are tested at once, with Benjamini–Hochberg control across domains.
- This is the question of FANS and CD-NOD (Seo 2026; Huang 2020): does anything still drive this domain differently once speed is removed?
- The tsLDiffPC vocabulary maps onto the decomposition:
  - its "effect-defying root causes" are domains with mechanism deviations;
  - its "noise-defying root causes" are domains whose noise differs beyond speed.
- Synthetic check: 0.6–1.3% of unchanged domains flagged, and the changed domain named alone in 93–96% of cohorts.

**7.3.5 Separate graphs, the transparent reference (v0.3's Analysis B).**
- **Equal sizes.** Each phenotype group is subsampled S times (*provisional* S = 25 to 50) to the same size, 90% of the smaller group. The Step 1 learner, in its fast configuration, runs on every subsample, giving each link's selection frequency p(e, k) in phenotype k.
- **Statistic.** D(e) = p(e, TD) − p(e, PIGD), with adjacency and arrow marks compared separately.
- **Two references.**
  - 50 permuted-label datasets.
  - The extended definition-matched null (§7.3.6).
  - The gap between the two measures the artefact created by defining phenotypes from graph variables.
- **Decision.** Candidate links beyond the null:
  - Benjamini–Hochberg at 5%, by rank among the null values;
  - |D(e)| ≥ 0.25 (*provisional*).
- **Role.**
  - It assumes nothing about dynamics, so it shows clinicians where yearly graphs differ.
  - It is never a mechanism verdict. A pure speed difference also produces yearly differences, which the extended null now contains.

**7.3.6 The extended definition-matched null.**
- **Simulation.** R datasets (R = 200 for H-P; 100 for the separate-graph reference) are simulated from the PPMI-fitted simulator.
  - The simulator has **one shared mechanism**, a shared hidden burden, and participant-level speed and load. Speed and load may depend on the baseline state, as fitted.
  - The datasets have the real sizes and missingness.
- **Re-applying the rule.** The same phenotype rule is applied to the simulated baselines. Simulated phenotypes therefore inherit speed and load differences, range restriction, collider bias and regression to the mean, but no mechanism difference.
- **Running the procedure.** Every Step 2 procedure is re-run on each null dataset. Only mechanism differences should exceed the null.
- **What it combines.** This null joins two things:
  - the parametric bootstrap of the speed-only model;
  - the artefacts of defining phenotypes from graph variables, which no permutation can reproduce.

**7.3.7 Shared-eigenvector check (sensitivity).**
- **Why it works.** If one mechanism generates both phenotypes' yearly matrices, they commute and share eigenvectors, whatever the speeds.
- **The test.** A centred participant bootstrap of the commutator of the two yearly matrices, with no continuous-time model. Xu, Düker & Matteson (2024) give a formal version.
- **Synthetic check.** It rejected in 3.5–4.0% of cohorts without a mechanism difference. With one, it rejected in 28% (new link) and 78% (stronger link).
- **Role.** It supports H-P but never replaces it. Pooled regressions bias it when speed tracks disease stage.

**7.3.8 Phenotype axes.**
- **Priors.**
  - TD versus PIGD: PIGD is expected to be faster, with more cognitive and psychotic load (Aleksovski 2018). Whether its mechanisms differ is open.
  - RBD status: the SOC model predicts earlier non-motor load and a faster speed with shared propagation.
- **Each axis is analysed in its own model.** A joint model is the supportive analysis of §7.13: two axes, factorial, following NetANOVA's baseline-plus-differential layout (Jiang & Zhang 2023).
  - Speed: log s_i = γ_PIGD·PIGD_i + γ_RBD·RBD_i (+ interaction) + σ·u_i.
  - Links: A + PIGD_i·Δ_PIGD + RBD_i·Δ_RBD.
- **More than two unordered groups.** Symmetric deviations with group weights √(n_g/N) (Gross & Tibshirani 2016; Ollier & Viallon 2017).

### 7.4 Step 3: data-driven mechanism classes (PPMI 1.0)

- **Model.** A mixture of continuous-time transition models.
  - Each patient belongs to one of K latent classes.
  - Each class has its own links on the shared map, with class-specific sparsity, plus its own load and noise.
  - Membership may depend on baseline characteristics.
- **Two versions.**
  - In *speed classes*, the classes share one mechanism and differ in speed, load and noise.
  - In *mechanism classes*, their links are free.
  - H-L asks whether the classes the data support differ only in speed and load.
- **Within-class burden and speed.** A random speed, and in the sensitivity analysis a random load, sit inside every class. Without them, stable burden and speed differences within a class would look like extra classes or like mechanism differences.
- **Number of classes.** K runs from 2 to 4, chosen by cross-validated likelihood on held-out patients. The bootstrap likelihood-ratio test is a check (Nylund 2007).
- **Null distribution.** The whole procedure, including the choice of K, is repeated on data simulated from the fitted speed-class model (R = 200).
- **Skewed scores.** Classes must survive monotone transformations of skewed scores (Bauer & Curran 2003).
- **Links within classes.** Class-specific links are estimated by sample splitting: classes are found in one random half and links estimated in the other.
- **If mechanism classes exist, report:**
  - their maps and their routes to loss of independence;
  - their overlap with TD/PIGD and RBD status (adjusted Rand index);
  - biology not used to define them: dopamine-transporter binding, genetics, and seed amplification if obtained later;
  - stability under resampling;
  - the limited replication possible in PPMI 2.0;
  - whether membership can be predicted at baseline.
- **Synthetic check.**
  - With 430 patients and 7 transitions each, the test held its 5% level (6.0%), had 69% power for one extra link, and recovered classes well (adjusted Rand index 0.76 and 0.82).
  - With 950 patients and 2 transitions, it did none of these (Appendix E).

### 7.5 Step 4: individual speed and load (PPMI 1.0)

- **Speed and load.** Participant-level speed and load are random effects. Speed is one parameter per person informed by every link, so it is far better identified than separate person-level slopes.
- **Key links.**
  - Random deviations on at most two pre-specified key links (D20) are exploratory.
  - Variances of person-level lagged effects need about 50 or more occasions per person to be estimated well (Schultzberg & Muthén 2018), and PPMI 1.0 offers about eight.
- **Outputs:**
  - how much patients vary in speed;
  - whether speed variation alone fits as well as models in which key links also vary;
  - the share of speed variation explained by phenotype, RBD status and age at onset;
  - the correlation between initial state and speed, modelled explicitly (Gistelinck 2021).

### 7.6 Step 5: confirmatory links (H-C1 to H-C10)

- **Hypotheses.** Each names a link X → Y, a phenotype contrast and an expected direction beyond speed.
- **Gate.** Tested only if H-P rejects.
- **Tests.**
  - The anchor-free per-link test (§7.3.3) on the pre-specified family, two-sided.
  - Benjamini–Hochberg at 5%, calibrated within the extended definition-matched null.
  - These are the primary clinical tests.
- **Classes.** Each tested link is classed by the rules of §2.3, on its speed-adjusted deviation and the margin ε.
  - The equivalence tests use 5% and are not adjusted further, because they only describe the differences the primary tests confirm.
- **Reporting.** Effects on clinical scales: for example, points of Schwab & England per MCID of the axial score, per year at the reference speed and at each phenotype's typical speed.
- **Controls.** Two or three hypotheses where the panel expects no difference are tested alongside as controls and reported separately.
- **Sensitivity analyses.**
  - The anchored per-link test.
  - A participant random load.
  - ε = 0.05.
  - v0.3's yearly interaction model, as a reading on the observed scale:
    - Y(*t*+1) on splines of Y(*t*), X(*t*) and the adjustment set;
    - plus era and phenotype;
    - plus phenotype × X(*t*), restricted to the shared range of X.
- **When identification fails** (deviations carrying half of the weight, or all of one sign):
  - the confirmatory verdicts are reported as "speed and mechanism not separable";
  - the share explained by speed is reported instead.

### 7.7 Domain verdicts and pathway cards (Q2)

**Domain verdicts.** For each phenotype (and each class), every domain gets a verdict with four parts:
- **Speed:** the phenotype's speed ratio, with its interval.
- **Load:** the load difference beyond speed, with Benjamini–Hochberg control across domains.
- **Noise:** the noise difference beyond speed.
- **Mechanism:** the per-domain test (§7.3.4), and the classes of the links into the domain (§2.3).

**Overall verdicts.**
- A phenotype differs by **speed and load only** when H-P does not reject and no domain shows a mechanism difference.
- Findings that are unstable under subsampling, or inconsistent between eras, are **unresolved**.

**The pathway card** (§11) collects these verdicts with the routes of §7.8.

### 7.8 Routes to loss of independence (Q3)

- **Total effects.** For each phenotype and class, the total effect of each domain on independence and daily function over h years (h = 1, 2 and 5). They are computed from the fitted continuous-time links, with participant-bootstrap intervals.
- **Exact split of the difference between phenotypes:**
  - a speed part, exp(h·s·A) − exp(h·A);
  - a mechanism part, exp(h·s·(A+Δ)) − exp(h·s·A).
- **Speed-matched horizons.** The phenotypes are also compared at speed-matched horizons (h years for TD, h/s for PIGD), where the speed part vanishes.
- **Attribution.** A first-order formula attributes the mechanism part to individual deviations. A perturbation travels to the changed link, crosses it, and travels on to independence.
- **Absorbing events.** If loss of independence is modelled as an absorbing event rather than a domain, this needs a joint state–event model (an extension, not in Part I).

### 7.9 Expert-map test per phenotype (Q4)

Every independence the expert map implies, given a minimal observed separating set, is tested in each phenotype with the participant-clustered test. The statistic is the share of independences violated, compared against two nulls:
- maps relabelled within tiers: does the expert map beat a scrambled one?
- the extended definition-matched null: is the difference between phenotypes larger than a split alone creates?

### 7.10 Stability, replication and negative controls

- **Stability.**
  - Step 1: selection frequencies over participant bootstrap samples (the panel version of bagging; Debeire 2024). The threshold for a stable link is calibrated on simulated data: the smallest threshold with precision ≥ 0.8.
  - Step 2: selection frequencies from stability selection.
- **Replication.** Findings are checked in PPMI 1.0 and PPMI 2.0 separately (direction and size), and in random half-samples.
- **Negative controls.**
  - The Step 1 learner run without tiers: arrowheads into era, age or sex, and later-to-earlier edges.
  - The whole Step 2 pipeline on permuted phenotype labels: the floor of "mechanism differences" when the label carries no information.

### 7.11 Missing data, dropout and death

- **Step 1.** Constraint-based tests use test-wise deletion and multiple imputation inside the tests (Witte 2022).
- **Steps 2 to 5.**
  - Missing domain values are multiply imputed (20 imputations) within each phenotype, so the imputation does not shrink differences.
  - The engine is fitted on each imputed dataset, and the results are combined by Rubin's rules.
  - A state-space version that handles missing values by filtering is a sensitivity analysis (§7.12).
- **Missing not at random.** Missing OFF exams, MoCA and questionnaires are imputed worse by a pre-set shift, as a sensitivity analysis.
- **Informative dropout.** Inverse-probability-of-remaining weights, as a sensitivity analysis.
- **Death and loss to follow-up** are recorded separately, and nothing is imputed after death.

### 7.12 Continuous-time check

- **Second host.** The final Step 2 and Step 5 models are refitted in ctsem with measurement error, in a hierarchical continuous-time state-space model (Driver & Voelkle 2018). Measurement error, floors and ceilings can distort the main model.
- **Measurement model.** It is held equal across groups (Schuurman & Hamaker 2019).
- **Validation.** The ctsem speed model is validated by simulation in WP3 before use. No worked example of a multiplicative speed exists.
- **Not a host.** DynNet (Rouanet 2026) cannot host a participant-level speed. It runs as a naive continuous-time comparator, and as a check of ordinal measurement models without speed.

### 7.13 Blinded clinician review

- Every mechanism finding is mixed with an equal number of control items: shared links, and pairs no source proposes.
- Items are shuffled, and phrased without statistics: "In [phenotype], once its faster or slower course is taken into account, could X this year influence Y next year more than in [other phenotype]?"
- At least two neurologists independently rate each item as plausible, likely artefact, or cannot judge.
- Agreement is summarised by Cohen's κ, and disagreements are resolved by discussion.
- Plausibility supports interpretation; it does not validate causality.
- The recorded forecasts (§6.4) are compared with the results.

### 7.14 Sensitivity analyses (pre-specified)

1. Other axes: secondary phenotype axis (clinical subtype); RBD status, if not co-primary; age at onset; genotype (exploratory); the two-axis factorial model (§7.3.8).
2. Indeterminate motor phenotype merged with PIGD.
3. The year 0 → 1 transitions included (flagged: the phenotype-defining values then enter the rows).
4. PPMI 1.0 only.
5. Participants whose diagnosis changed excluded.
6. The other motor state (ON instead of OFF, or the reverse).
7. Phenotype reassigned at year 1 (uses later information).
8. Missing-not-at-random shift; dropout weights.
9. Transitions in which treatment started excluded; treatment acting on the drift.
10. The anchored per-link test in place of the anchor-free one.
11. ε = 0.05 instead of 0.10.
12. Two-year transitions (across a missed visit) excluded.
13. A group-level speed in place of the participant-level speed, as a quick screen known to be over-liberal.
14. Measurement-error state-space model in ctsem (§7.12).
15. The correlation between initial state and speed modelled (Gistelinck 2021).
16. Terms at a two-year lag (a check of the first-order Markov assumption).
17. A participant random load in Steps 2 and 5.

## 8. Simulation benchmark (WP3)

The benchmark is reported in the ADEMP structure: aims, data-generating mechanisms, estimands, methods and performance measures (Morris 2019). The harness is built and tested on synthetic data before the simulator is fitted to PPMI.

### 8.1 Aims

1. Test H-M1 against the named comparators.
2. Measure, at PPMI's group sizes and follow-up, the following for every procedure in §8.4:
   - false mechanism-difference rates;
   - power;
   - localisation;
   - class recovery;
   - bias of speed and of the speed-adjusted deviations.
3. Validate the continuous-time engine and its ctsem host.
4. Choose the Step 1 learner configurations (at most three per learner), and confirm the cost of the fast configuration.
5. Fix the numbers of subsamples, null datasets and replicates from measured cost (§8.8).

### 8.2 Data-generating mechanisms

- **Mechanisms.**
  - Continuous-time transition models fitted to the pooled primary population over the frozen map.
  - Participant-level speed and load may depend on the baseline state, as fitted.
  - Treatment enters as in the engine.
  - Scores are on Gaussian-copula scales that reproduce each variable's type.
- **Hidden burden.** A person-level hidden burden enters every domain.
- **Baselines.** Baseline values come from the fitted joint distribution, including every variable a phenotype rule needs. The rule applied to simulated baselines therefore reproduces realistic phenotype frequencies.
- **Missingness and dropout.** Missingness is resampled from PPMI's own patterns by era, year and treatment status. Dropout follows the observed follow-up distribution.
- **Fidelity criterion (*provisional*).**
  - Median absolute difference between real and simulated Spearman correlations ≤ 0.05 on the scored pairs.
  - Phenotype frequencies within 3 percentage points.
  - If missed, the discrepancy is reported and the benchmark runs on the simulator as fitted, with no tuning towards any method.

| Scenario | Truth | Tests |
|---|---|---|
| S0 Shared | Shared mechanism, equal speed and load | False mechanism differences |
| S0L Load only | Shared mechanism, equal speed, load difference | Load not read as mechanism; the yearly invariance tests' inflation |
| S1 Speed | Shared mechanism; phenotypes differ in speed (time scale) and load | Speed and load not read as mechanism |
| S1V Burden variance | Shared mechanism; the variance of burden differs by phenotype | Burden not read as mechanism |
| S2 Strength | One or two links stronger beyond speed | Power and localisation |
| S2b Weak versus strong | Selected links four times as strong in PIGD, with the TD strength near ε | Strength not read as structure |
| S3 Structure | One link present only in PIGD, another only in TD | Power and localisation for structure |
| S4 Curvature | Shared non-linear mechanism; phenotypes cover different ranges | Curvature not read as mechanism |
| S5 Mechanism classes | Two latent mechanism classes that cut across the phenotypes | Class recovery; power of H-L |
| S6 Speed classes | Latent speed classes only | Speed classes not read as mechanism classes |
| S7 Individual speed | Continuous individual variation in speed | Participant-level speed handled |

**Stresses**, crossed with S0 to S3 as cost allows:
- missing not at random;
- measurement noise in the phenotype-defining items;
- dropout tied to severity;
- hidden burden absent or strong;
- sample sizes halved and doubled;
- speed ratios 1.25, 1.5 and 2;
- deviations carrying more than half of the weight, the identification limit.

### 8.3 Estimands

For every link, the true class on the speed-adjusted scale. For every domain, the true verdict. For each phenotype and class, the true Δ, speed and load. Also the true PAG over the measured variables (Step 1).

### 8.4 Procedures compared

**Speed-adjusted (this protocol):**
- the speed-only test (H-P);
- speed-adjusted joint discovery;
- per-link tests, anchor-free and anchored;
- per-domain tests;
- separate graphs against the extended null;
- the shared-eigenvector check;
- the mixture tests (H-L);
- the random-effects models of Step 4.

**Named comparators, run as published.** Each uses the authors' software, at its default settings and at a common α = 0.05. The software is installed only with the investigator's approval and pinned (D23).

| ID | Comparator | Family | Software | Expected under speed only |
|---|---|---|---|---|
| C1 | Multigroup RI-CLPM: χ² difference with lagged effects equal over groups, per-edge score tests | Clinical SEM standard (Mulder & Hamaker 2021) | lavaan | Invariance rejected; persistence first |
| C2 | Multigroup panel GVAR with equality constraints; IVPP global test and partial pruning | Network-psychometric standard (Epskamp 2020; Du 2026) | psychonetrics; IVPP | As C1 |
| C3 | NCT-style permutation test on temporal networks, labels permuted by participant | Network comparison (van Borkulo 2023) | Custom code following NCT | Rejected in every cohort |
| C4 | Naive multigroup continuous-time model: an additive phenotype effect on each drift entry | Continuous time | ctsem; DynNet | Every non-zero drift entry flagged; no zero entry |
| C5 | tsLDiffPC and tsDCI, two-slice tables, every conditioning set (a cap of 3 at ten domains) | Difference graphs for time series (Ruer 2026) | pyCIPHOD (beta; commit pinned) | Persistence differences read as root causes in about half of the domains; low power (Appendix E) |
| C6 | DCI, lag-to-current candidate pairs; stability-selection variant | Difference graphs (Wang 2018) | causaldag | Dense yearly difference graph |
| C7 | CD-NOD with time tiers | Node-level shift (Huang 2020) | causal-learn | Every domain a changing module |
| C8 | Sparse mechanism shift, Fisher-z test on the map | Node-level shift (Perry 2022) | sparse_shift | Every domain shifted |
| C9 | Data-shared lasso, and multi-VAR with known subgroups | Shared plus deviations (Gross & Tibshirani 2016; Fisher 2022; Crawford 2026) | glmnet; multivar | Dense deviations; a hybrid shared part |
| C10 | Joint causal inference and J-PCMCI+ context edges | Context nodes (Mooij 2020; Günther 2023) | causal-learn; tigramite | Phenotype → every domain |
| C11 | Separate graphs read naively, and v0.3's yearly link-wise interaction tests | Naive split; v0.3 | This study's code | Discordant links even without differences |

**Faithful re-implementation.**
- For C5, a re-implementation of the published algorithm, written from the authors' code, reproduced the paper's published results within Monte Carlo error (tsLDiffPC/tsDCI note, §3).
- It is also checked against pyCIPHOD itself on 20 datasets per scenario.
- The re-implementation then carries the full benchmark, because the authors' code tests each conditioning set through data-frame operations and is slow.

**Excluded comparators.**
- DDAG: optional, with no official code.
- iSCAN and FANS: excluded. Their theory is for nonlinear models, and FANS needs at least 5,000 samples per group.

**The sortability control** runs alongside.

### 8.5 Performance measures

Each is reported with its Monte Carlo standard error, as a function of cohort size and speed ratio:
- the false mechanism-difference rate per truly shared link, separately for self-effects (persistence), direct links and links absent from the drift;
- the rate of false root-cause domains;
- the any-flag rate;
- power for true differences;
- localisation, in two versions:
  - exact: the true link and nothing else;
  - footprint-aware: flags on yearly entries that the real difference itself moves are counted separately;
- the rate of false "structure" verdicts (S2, S2b);
- class recovery (adjusted Rand index), and the false and true rates of H-L (S5, S6);
- bias and 95% interval coverage of speed, load and Δ;
- adjacency and arrowhead F1 for Step 1, against the true PAG;
- cost per fit and failures.

### 8.6 Replicates and nested designs

- **Precision.** The Monte Carlo standard error of a rate p estimated from R datasets is √(p(1 − p)/R). At p = 5%, 200 datasets give 1.5 percentage points.
- **Speed-adjusted Step 2 procedures:** 200 datasets per scenario in S0 and S1 (the null scenarios at the heart of H-M1), and 100 in S0L, S1V, S2, S2b, S3, S4 and S7.
- **Named comparators:** 100 datasets per scenario in S0, S0L, S1, S1V, S2 and S3, and 50 in the others.
- **Step 1 learners:** 120 datasets per scenario.
- **Nested designs** (*provisional*):
  - Every analysed dataset needs its own extended null, so evaluating calibration against it is a nested simulation. It uses a reduced design: 10 datasets in each of S0 to S4, each with its own simulator fit and 50 null datasets.
  - The latent-class test uses S0, S5 and S6 × 5 datasets × 30 bootstrap replicates.
  - The full designs (R = 200; 100 for the separate-graph reference) apply to the real data only.

### 8.7 Decision rules (pre-specified)

- **H-M1** holds if:
  - the speed-adjusted per-link, per-domain and global procedures stay at or below 5% in the shared-mechanism scenarios, within two Monte Carlo standard errors;
  - at least one named comparator exceeds 5%.
- **Choice of procedure.** WP4 uses the speed-adjusted procedures that meet H-M1 with the highest power in S2 and S3. If none meets it, Part I reports only the shared map and the speed and load estimates, as associations.
- **Power gate for H-L.** If the latent-class test has power below 0.5 for a pre-specified mechanism difference at PPMI 1.0's follow-up, H-L is reported as descriptive and leaves the primary family.
- **Class reachability.** If equivalence is shown for fewer than half of the truly similar links at the chosen ε, two consequences follow:
  - the classes "similar" and "structurally different" are reported as out of reach at PPMI's sizes;
  - links are described by their estimated deviations and confidence bounds.
- **Identification.** The study reports how often the speed-adjusted procedures fail when deviations carry more than half of the weight. The real-data report then says whether the data sit near that limit.

### 8.8 Compute budget

Costs are measured on synthetic data of the real size before any number below is fixed. At six domains and 1,050 patients, the full Step 2 suite took about 30 seconds per cohort (Appendix E). The table assumes:
- about 1 minute per engine fit and 8 minutes per full Step 2 suite at ten domains;
- about 3 minutes per mixture fit;
- 30 seconds per fast Step 1 fit;
- 1 minute per comparator fit;
- 4 parallel jobs.

| Component | Size | Hours |
|---|---|---|
| Patient-level speed tests on the real data, with the extended null, including the per-link and per-domain tests | 201 datasets × 2 models × 2 phenotype axes, plus the simulator fits | About 10 |
| Joint discovery and stability selection on the real data | 50 half-samples × 13 penalised fits × 2 axes | About 8 |
| Speed-adjusted procedures in the benchmark | 200 datasets in S0 and S1, 100 in seven other scenarios | About 37 |
| Latent classes on the real data: bootstrap null repeating the choice of K | 200 datasets × 3 values of K × 2 versions | About 30 |
| Latent-class test in the benchmark, reduced nested design | S0, S5, S6 × 5 datasets × 30 replicates | About 30 |
| Individual speed and load (PPMI 1.0); continuous-time check (ctsem) | A few final fits, after validation | About 24 |
| Named comparators | 11 comparators × 100 datasets in six scenarios | About 30 |
| Step 1, separate-graph reference, stability | As in v0.3, without its nested evaluation | About 70 |
| Total | | About 239 (10 days) |

If measured costs push the total above 10 days, the design is cut in this order:
1. stress cells;
2. the comparators' replicates, to 50 per scenario;
3. the nested designs, to 5 datasets per scenario;
4. separate-graph subsamples, from 25 to 15;
5. null datasets for the separate-graph reference, from 100 to 50.

A fast Step 1 fit costing more than 2 minutes means the fast configuration is redesigned (fewer columns, or a lower separating-set cap) before the benchmark runs.

## 9. Statistical considerations

- **Precision.**
  - There is no general sample-size rule for graph recovery or for mixtures of dynamic models. WP3 gives power, class recovery and class reachability at the real group sizes and follow-up.
  - Step 1 uses all participants and has the most power.
  - Steps 2 and 5 depend on the size of the minority phenotype; Steps 3 and 4 depend on PPMI 1.0's follow-up.
- **Primary tests.** These are pre-specified, and all other results are reported beside them and never replace them:
  - H-M1 (simulation);
  - H-P and H-L (PPMI), under Holm;
  - H-C1 to H-C10 (PPMI), gated on H-P.
- **Multiplicity.**
  - Holm across the primary PPMI family (two tests, or three if RBD status is co-primary).
  - Benjamini–Hochberg at 5% within H-C, within H-D, and within each discovery analysis.
  - The class and equivalence tests describe confirmed differences and are not adjusted further.
  - Everything else is labelled exploratory.
- **Calibration.**
  - Step 2 and Step 5 tests are referred to the extended definition-matched null, which contains a parametric bootstrap of the fitted speed-only model.
  - H-L is referred to a parametric bootstrap that repeats the choice of the number of classes.
  - χ² references are reported as approximations only.
- **Clustering.** The participant is the unit of every resampling and the cluster of every standard error.
- **Monte Carlo error** is reported with every simulated rate.

## 10. Bias and validity

| Threat | How it misleads | Mitigation |
|---|---|---|
| A faster disease clock (speed) | Every yearly coefficient changes, within-year paths appear as links, and the noise grows: a pathway difference that is not there | Comparison in continuous time with a speed for each participant; the extended null; named comparators in WP3 |
| Speed varies within phenotypes | Group-averaged dynamics break the matrix-power relation; group-level speed tests become over-liberal as cohorts grow | Participant-level speed as a random effect |
| Persistence read as a root cause | Speed changes each domain's own year-to-year persistence most, so persistence differences look like changed mechanisms | Domains' own rates are part of the speed; anchor-free identification; per-domain tests |
| Selection by the phenotype rule | Within-phenotype graphs gain or lose links that do not differ in reality | Landmark; extended definition-matched null; pooled Step 1 |
| Range restriction | Links look weaker inside a phenotype whose defining scores vary less | Extended null; equal-size comparisons; shared-range restriction in the yearly reading |
| Regression to the mean | Extreme year-0 scores drift back, mimicking phenotype-specific change | Landmark: no year-0 values in analysed rows; extended null |
| Shared items | A node partly encodes the phenotype rule | No shared items between nodes |
| Hidden burden | Level differences everywhere, and lagged links without a causal path | Burden in the simulator and in every null; latent-aware Step 1; random load; scenario S1V |
| Non-significance read as absence | "Present in PIGD, absent in TD" produced by unequal power | Difference tests; equivalence margin ε; equal-size comparisons |
| Curvature | A shared curved mechanism looks like a mechanism difference across ranges | Copula scales; splines in the yearly reading; scenario S4 |
| Floors and ceilings, measurement error | Bounded scales flatten effects near their limits; error attenuates effects differently by range | Copula scales; measurement-error state-space check (§7.12) |
| Initial state linked to speed | Fast progressors are more advanced at entry, which biases pooled regressions | Speed per participant conditional on the current state; the correlation modelled (sensitivity 15) |
| Speed and mechanism not separable | Deviations that carry half of the weight, or all point the same way, can be absorbed into speed | Identification stresses in WP3; the share explained by speed reported; a plurality-rule sensitivity |
| Unequal group sizes | A link missing in the small group read as biology | Joint estimation; equal-size subsampling for the reference; power from WP3 |
| Phenotype instability | The baseline label misclassifies the later course | Stability reported; reassignment sensitivity analysis; data-driven classes (Step 3) |
| Treatment by indication | Treatment choice differs by phenotype and confounds progression links | Treatment as a node in Step 1 and as a symptomatic effect not scaled by speed in the engine; transitions with a treatment start excluded in sensitivity |
| ON/OFF state | Motor scores mean different things on and off medication | Panel decision; the other state as sensitivity |
| Informative dropout | Those who stay look slower | Dropout weights; dropout in the simulator; nothing imputed after death |
| Era differences | PPMI 1.0 and 2.0 differ in enrolment and follow-up | Era as context; replication by era |
| Unstable orientation | Weak tests give confident arrows | PAG marks kept; participant bootstrap; tiers |
| Multiplicity and selective reporting | Attractive graphs selected after the fact | A small primary family under Holm; gated confirmatory tests; pre-registration; every result reported |
| Leakage (Part II) | Outcome information reaches the predictions through graphs, cut-offs or tuning | Every step inside training folds; the package frozen before external outcomes are seen |

## 11. Clinical outputs and intended use

- **The map.** A shared progression map, with links that differ beyond speed marked and graded: confirmatory, replicated discovery, exploratory or unresolved. Each difference is labelled with its class (§2.3).
- **A one-page pathway card per phenotype, and per class if mechanism classes exist.** It shows:
  - the speed relative to the reference (for example, "PIGD runs the shared mechanism about 1.8 times faster");
  - the load profile;
  - the domain verdicts;
  - the links that differ beyond speed;
  - the routes to loss of independence, with the part speed explains and the part mechanism explains.
- **For individuals (PPMI 1.0):**
  - how widely speed varies within each phenotype;
  - how much of that variation phenotype, RBD status and age at onset explain.
- **Intended use:**
  - hypotheses for monitoring: which domains to reassess, and how often (speed sets the interval, mechanism sets the content);
  - referral: physiotherapy, falls, cognitive assessment;
  - counselling;
  - input to the design of phenotype-stratified trials.
- **Not for:** individual prediction, treatment choice, or replacing guideline-based care. Treatment effects need a separate target-trial emulation (§12.6).
- **Out of scope for both parts:** biological phenotypes (seed amplification; SynNeurGe), which need PPMI's biospecimen data.
  - Data-driven mechanism classes are now in scope (Step 3), for PPMI 1.0 only and subject to the power gate.

## Part II. Clinical translation (conditional)

Part II is an outline. Each component gets its own protocol, data access and approvals after Part I's gate (§13).
- Its value is clinical utility.
- Prediction accuracy does not validate the causal map, and the map can be right without improving prediction.

## 12. Part II components

### 12.1 Questions

- **C-RQ0 (replication).** Do Part I's verdicts replicate, in direction and class, in an independent cohort? These are the speed, load and mechanism verdicts and the confirmed link differences.
- **C-RQ1 (prognosis).** Does phenotype- or class-specific mechanism information improve the prediction of loss of independence beyond three things?
  - longitudinal history;
  - the phenotype label;
  - a shared map run at each patient's estimated speed.
- **C-RQ2 (prospective validation).** Does a locked model generalise to newly recruited patients, and give decision-analytically useful estimates for one pre-specified care action?
- **Treatment effects by phenotype:** a separate target-trial emulation (§12.6).

### 12.2 Intended clinical use

- **The use.** An assistive risk estimate at a neurology follow-up visit, to inform monitoring frequency, functional and falls assessment, rehabilitation and care planning.
- **Not for** diagnosis, drug choice, treatment withdrawal or automatic scheduling.
- **The action evaluated.** One action is named: *provisionally*, referral for a multidisciplinary functional and fall-risk assessment. The clinical panel sets:
  - its burdens;
  - the capacity for it;
  - the risk threshold at which clinicians would act.

### 12.3 II-a. External replication

- **Cohort.** One independent cohort never used in Part I, with MDS-UPDRS, a cognitive scale and Schwab & England. Each candidate needs an application and a harmonisation audit:
  - Oxford Discovery and Tracking Parkinson's (through the Oxford Parkinson's Disease Centre);
  - ICEBERG;
  - LuxPARK.
- **What is replicated:**
  - each phenotype's speed ratio;
  - its domain verdicts;
  - Part I's confirmed and replicated-discovery differences beyond speed, with their direction and class.

  The phenotype rule, landmark, engine and tests are fixed before the external data are analysed.
- **Visit intervals.** The continuous-time engine uses the external cohort's actual intervals (for example 18 months). Its effects are directly comparable with PPMI's.
- **Criterion.** A difference replicates when its speed-adjusted deviation has the same sign and its one-sided test at 5% rejects zero in the external cohort. Its estimate is reported beside Part I's.

### 12.4 II-b. Prognostic incremental value

- **Landmark.** A follow-up visit (*provisional*: year 2), when the year-0 phenotype and two years of history are known.
- **Endpoint (provisional).** Alive and independent 24 months after the landmark, among participants alive and independent at it.
  - Independence means Schwab & England ≥ 80%. Scores below 80% have been used to define dependency (Macleod 2016).
  - Death is a failure, not a missing value.
  - Secondary endpoint: first sustained dependence or death within 24 months. Recovery after dependence makes the two endpoints differ.
- **Events.**
  - In early, research-enrolled PD, death and loss of independence within 24 months may be rare. A genome-wide mortality analysis left PPMI out because fewer than 20 deaths had been recorded (Tan 2024).
  - WP0 counts the events. The required development sample size is computed from the observed event rate and an anticipated model fit (Riley 2020).
  - If PPMI falls short, the endpoint changes, in this order:
    1. the PPMI milestone composite (Brumm 2023): 25 milestones in six domains, whose functional-dependence milestone uses the same Schwab & England threshold;
    2. a 48-month horizon;
    3. next-year Schwab & England and MDS-UPDRS II as continuous outcomes.
- **Models.** All use the same information available at the landmark, with comparable tuning effort.

| Model | Information | Question |
|---|---|---|
| C0 Clinical | Age, sex, disease duration, MDS-UPDRS II and III, MoCA, Schwab & England, Hoehn and Yahr, levodopa-equivalent dose | Is the work clinically competitive? |
| C1 Longitudinal | C0 plus each domain's history (last value and yearly slope), in a flexible learner | Does anything beat good use of the history? |
| C2 Phenotype label | C1 plus the phenotype | Is the label enough? |
| C3 Shared map at the patient's speed | C1 plus two-year forecasts from the shared continuous-time model, run at each patient's estimated speed and load | Does a shared mechanism with a personal clock add value? |
| C4 Phenotype- or class-specific map | C3 plus the deviations classed as different beyond speed | Do different mechanisms add value beyond speed? |

- **Primary contrasts.** "Adds value" requires both C4 against C2 and C4 against C3: an intersection–union test, with each contrast at 5%.
  - Primary metric: out-of-sample log loss.
  - Key secondary metric: net benefit at the clinician-chosen threshold (Vickers 2006).
  - Calibration (intercept and slope), discrimination and the Brier score are reported for every model.
- **Placebo controls.** C4 is also run with permuted phenotype labels in its transition model, and with random deviations of the same number. A gain that survives these comes from the model's form, not the mechanisms.
- **Internal validation.** Nested cross-validation by participant. Every step happens inside the training folds:
  - imputation;
  - cohort-derived cut-offs;
  - graph discovery;
  - the engine;
  - tuning, calibration and thresholds.
- **External validation.**
  - The full package is frozen before the external outcomes are seen: code version, rules, models, calibration and thresholds.
  - Then calibration, discrimination, Brier score and net benefit are reported, overall and by phenotype, site, sex and age where precision allows.
  - Recalibration is model updating and needs a new validation.
- **Reporting.** TRIPOD+AI (Collins 2024); risk of bias by PROBAST+AI (Moons 2025).

### 12.5 II-c. Prospective silent validation (outline for a separate protocol)

- **Cohort.** A multicentre observational cohort at movement-disorder clinics outside the PPMI network. It enrols consecutive eligible patients across disease stages and records screening and reasons for non-enrolment.
- **Visits.**
  - Baseline, 6, 12, 18 and 24 months (36 if funded), with actual dates. The continuous-time engine uses them as they are.
  - Structured capture of interim events: falls, admissions, treatment changes, loss of independence, institutionalisation and death.
  - Reasons for missed visits are recorded.
- **Silent use.** The locked model runs at each landmark.
  - Predictions are time-stamped and stored without being shown to clinicians.
  - Care is unchanged, and outcomes are ascertained independently.
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

- **WP0.**
  - Stop or redesign if the minority phenotype cannot meet WP3's power target, even with the secondary axis or merged groups.
  - If PPMI 1.0's follow-up is too short for Steps 3 and 4, those steps become descriptive.
  - Apply the interval rule to Step 1 (§6.3).
- **WP3.**
  - If H-M1 fails for the speed-adjusted procedures, Part I reports only the shared map and the speed and load estimates, as associations.
  - If the power gate fails, H-L is descriptive.
  - If the identification stresses show that the data sit near the limit (§8.7), mechanism verdicts are reported as "speed and mechanism not separable" where that applies.
- **Part I to Part II.** Part II starts only when three conditions hold:
  - at least one confirmatory or replicated-discovery difference beyond speed lies on a route to daily function or independence;
  - for II-b, there are enough endpoint events, or an accepted fallback (§12.4);
  - data access and approvals for the external cohort are in place.
- **II-b.** If C4 does not beat both C2 and C3, the result is published as negative, and II-c is not pursued with this model.
- **II-c.** If there are too few events, follow-up is incomplete or sites are unrepresentative: extend follow-up, or report the validation as inconclusive.
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
- **Gating.** Every command that reads PPMI participant data waits for protocol approval. WP0 runs with the investigator's written approval.
- **Software.**
  - Third-party packages, such as the comparators' R and Python packages, are installed only with the investigator's approval (D23).
  - Their versions and commits are pinned.
  - Each is tested on synthetic data before it touches PPMI data.
- **Expert questionnaire.**
  - The investigator checks whether the institution requires ethics review, or a formal exemption, for an expert survey.
  - Answers, including the forecasts, are reported without names, and participation is voluntary.
- **Part II.**
  - Separate ethics approvals and data agreements.
  - For II-c:
    - pseudonymous identifiers, with keys held at each site;
    - role-based access;
    - encrypted storage and transfer;
    - audit logs;
    - a documented retention period;
    - an incident process.
  - Consent that covers model validation and future reuse.
- **Provenance.** Each dataset freeze has a manifest: extraction date, participant counts, codebook and instrument versions, exclusions, transformations and file hashes.
- **Version control.** Nothing is committed to git without the investigator's instruction.

## 15. Reporting and dissemination

- **Part I.** STROBE for the observational analyses; the simulation in ADEMP structure (Morris 2019); the expert consensus by ACCORD (Gattrell 2024).
- **Part II.** TRIPOD+AI and PROBAST+AI for prognosis; TARGET for treatment effects; DECIDE-AI for any later live evaluation.
- **Outputs.**
  1. The registered protocol with the simulation specification (OSF).
  2. The clinical paper: the phenotype and class progression atlas with the pathway cards.
  3. A methods paper. It covers speed-adjusted comparison of causal mechanisms between patient groups, the extended definition-matched null, and the named-comparator benchmark.
  4. Part II papers, if its gates pass.
- Every result is reported, including null, unresolved and negative findings, and every protocol deviation. The clinicians' forecasts are reported beside the results.

## 16. Reproducibility, compute and quality assurance

- Pre-registration on OSF before WP4, with code and lockfiles for every environment (Python and R).
- **Unit tests:**
  - the annual grid, the landmark, the phenotype rules, participant-level splits and exclusions;
  - for the engine: the likelihood against an independent implementation, recovery of known speed, load and deviations, and calibration under the null on synthetic data.
- Seeds, configurations, costs and failures are logged for every fit.
- **Compute.** The machine is an Apple M4 with 10 cores and 16 GB.
  - At most 4 parallel jobs.
  - One fit per job, saved as it finishes and skipped on rerun.
  - Long runs are started from the Terminal app.
- The harness is built and tested on synthetic data before any PPMI run, and costs are measured before replicate numbers are fixed (§8.8). The synthetic checks behind this design are in `scripts/toy/` (Appendix E).
- An audit report for clinical review lists:
  - graph uncertainty;
  - forbidden-edge violations;
  - group sizes and completeness;
  - identification diagnostics;
  - what the outputs must not be used for.

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
| D9 | Step 1 test and learner configurations | Chosen in WP3 by accuracy and cost | Investigator | WP3 |
| D10 | Provisional thresholds | Separate-graph reference \|D\| ≥ 0.25; Benjamini–Hochberg 5%; fidelity ≤ 0.05; S and R by cost | Investigator | WP2 to WP3 |
| D11 | Pre-registration | OSF, before WP4 | Investigator | WP2 |
| D12 | Smallest effect of interest ε (Δ in v0.3) | 0.10 SD per SD per year at the reference speed, shown in clinical units; 0.05 as sensitivity | Panel, statistician and investigator | WP2 |
| D13 | Interval rule for Step 1 | Annual tables if at least 90% of intervals lie within 12 ± 3 months; Steps 2 to 5 use actual intervals | Investigator | WP0 |
| D14 | Compute budget | At most 10 days of 4-job compute for WP3 and WP4; cuts in the order of §8.8 | Investigator | WP3 |
| D15 | Scope of Part II | Outline only in this protocol; II-a and II-b planned after Part I's gate; II-c and II-d as separate protocols | Investigator and supervisors | Now |
| D16 | Part II endpoint | Alive and independent at 24 months, with the fallbacks of §12.4 | Clinical panel and statistician | Before Part II |
| D17 | External cohort | One application, to a cohort with MDS-UPDRS, a cognitive scale and Schwab & England | Investigator | After WP5 |
| D18 | RBD status as a co-primary axis | Yes, with Holm across H-P (TD/PIGD), H-P (RBD) and H-L | Clinical panel, round 2 | WP1 |
| D19 | Status of H-L | Primary, in PPMI 1.0, with the power gate of §8.7 | Investigator and statistician | WP2 |
| D20 | The two key links for Step 4 | Chosen by the panel from the confirmatory list | Clinical panel, round 2 | WP1 |
| D21 | Clinicians' forecasts of speed, load or mechanism | Recorded in round 2, before any analysis | Investigator | WP1 |
| D22 | Per-link test of record | Anchor-free (minimal-L1 speed, Monte Carlo p-values); anchored as sensitivity | Investigator and statistician | WP2 |
| D23 | Software for the WP3 comparators | Install R (lavaan, psychonetrics, IVPP, glmnet, multivar, ctsem) and Python (causaldag, pyCIPHOD, sparse_shift); versions and commits pinned | Investigator | Before WP3 |
| D24 | Treatment in the engine | A symptomatic effect on the observed motor scores, not scaled by speed, shared across phenotypes; a drift effect as sensitivity | Investigator and clinical panel | WP1 |

## 18. Timeline (provisional)

| Weeks | Work |
|---|---|
| 1–2 | WP0 feasibility audit |
| 2–6 | WP1 two-round expert questionnaire (round 2 adds the RBD axis, the key links and the forecasts); literature search |
| 7 | WP2 protocol freeze and registration |
| 6–12 | WP3 simulator, engine validation and benchmark, built and tested on synthetic data first (about 10 days of compute) |
| 12–14 | WP4 PPMI analyses |
| 14–17 | WP5 blinded review, replication, sensitivity analyses, comparison with the forecasts |
| 17–21 | WP6 manuscripts |
| After week 17, if the gate passes | Part II: external-cohort application and the II-a and II-b protocols, over months; II-c and II-d as separately funded protocols, over years |

## References

- Aleksovski D, Miljkovic D, Bravi D, Antonini A. Disease progression in Parkinson subtypes: the PPMI dataset. Neurol Sci 2018;39:1971–1976. doi:10.1007/s10072-018-3522-z.
- Allison PD. Comparing logit and probit coefficients across groups. Sociol Methods Res 1999;28:186–208. doi:10.1177/0049124199028002003.
- Bang CW, Didelez V. Constraint-based causal discovery with tiered background knowledge and latent variables in single or overlapping datasets. CLeaR 2025, PMLR 275:1116–1146.
- Bang CW, Didelez V. Do we become wiser with time? On causal equivalence with tiered background knowledge. UAI 2023, PMLR 216:119–129.
- Bauer DJ, Curran PJ. Distributional assumptions of growth mixture models: implications for overextraction of latent trajectory classes. Psychol Methods 2003;8:338–363. doi:10.1037/1082-989X.8.3.338.
- Borghammer P. The α-synuclein origin and connectome model (SOC model) of Parkinson's disease: explaining motor asymmetry, non-motor phenotypes, and cognitive decline. J Parkinsons Dis 2021;11:455–474. doi:10.3233/JPD-202481.
- Bowden J, Davey Smith G, Haycock PC, Burgess S. Consistent estimation in Mendelian randomization with some invalid instruments using a weighted median estimator. Genet Epidemiol 2016;40:304–314. doi:10.1002/gepi.21965.
- Brumm MC, Siderowf A, Simuni T, et al. Parkinson's Progression Markers Initiative: a milestone-based strategy to monitor Parkinson's disease progression. J Parkinsons Dis 2023;13:899–916. doi:10.3233/JPD-223433.
- Bystrova D, Devijver E. Constraint-based difference graph discovery in a linear setting. arXiv:2606.13947 (2026).
- Cáncer PF, Arnold M, Estrada E, Voelkle MC. Continuous-time structural equation model forests. Psychol Methods 2025, advance online. doi:10.1037/met0000766.
- Cashin AG, Hansford HJ, Hernán MA, et al. Transparent reporting of observational studies emulating a target trial: the TARGET statement. BMJ 2025;390:e087179.
- Chen Y, Li C, Ouyang J, Xu G. DIF statistical inference without knowing anchoring items. Psychometrika 2023;88:1097–1122. doi:10.1007/s11336-023-09930-9.
- Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. BMJ 2024;385:e078378.
- Crawford CM, Park JJ, Chow S-M, Ernst AF, Pipiras V, Fisher ZF. Penalized subgrouping of heterogeneous time series. Multivariate Behav Res 2026;61:579–602. doi:10.1080/00273171.2026.2622120.
- Dadu A, Satone V, Kaur R, et al. Identification and prediction of Parkinson's disease subtypes and progression using machine learning in two cohorts. npj Parkinsons Dis 2022;8:172.
- Debeire K, Gerhardus A, Runge J, Eyring V. Bootstrap aggregation and confidence measures to improve time series causal discovery. CLeaR 2024, PMLR 236:979–1007.
- Driver C, Voelkle M, Oud H. ctsem: continuous time structural equation modelling. R package version 3.11.1; 2026. doi:10.32614/CRAN.package.ctsem.
- Driver CC, Oud JHL, Voelkle MC. Continuous time structural equation modeling with R package ctsem. J Stat Softw 2017;77(5). doi:10.18637/jss.v077.i05.
- Driver CC, Voelkle MC. Hierarchical Bayesian continuous time dynamic modeling. Psychol Methods 2018;23:774–799.
- Driver CC. Inference with cross-lagged effects: problems in time. Psychol Methods 2025;30:174–202. doi:10.1037/met0000665.
- Du X, Urnes Johnson S, Epskamp S. The invariance partial pruning approach to the network comparison in time-series and panel data. Psychol Methods 2026;31:749–767. doi:10.1037/met0000824.
- Epskamp S. Psychometric network models from time-series and panel data. Psychometrika 2020;85:206–231. doi:10.1007/s11336-020-09697-3.
- Fereshtehnejad SM, Romenets SR, Anang JBM, et al. New clinical subtypes of Parkinson disease and their longitudinal progression: a prospective cohort comparison with other phenotypes. JAMA Neurol 2015;72:863–873.
- Fereshtehnejad SM, Zeighami Y, Dagher A, Postuma RB. Clinical criteria for subtyping Parkinson's disease: biomarkers and longitudinal progression. Brain 2017;140:1959–1976.
- Fiebig DG, Keane MP, Louviere J, Wasi N. The generalized multinomial logit model: accounting for scale and coefficient heterogeneity. Mark Sci 2010;29:393–421. doi:10.1287/mksc.1090.0508.
- Fisher ZF, Kim Y, Fredrickson BL, Pipiras V. Penalized estimation and forecasting of multiple subject intensive longitudinal data. Psychometrika 2022;87(2). doi:10.1007/s11336-021-09825-7.
- Gattrell WT, Logullo P, van Zuuren EJ, et al. ACCORD (ACcurate COnsensus Reporting Document): a reporting guideline for consensus methods in biomedicine developed via a modified Delphi. PLoS Med 2024;21:e1004326.
- Gelman A, Stern H. The difference between "significant" and "not significant" is not itself statistically significant. Am Stat 2006;60:328–331.
- Gerhardus A, Runge J. High-recall causal discovery for autocorrelated time series with latent confounders. NeurIPS 2020.
- Gistelinck F, Loeys T, Flamant N. Multilevel autoregressive models when the number of time points is small. Struct Equ Modeling 2021;28:15–27. doi:10.1080/10705511.2020.1753517.
- Gong M, Zhang K, Schölkopf B, Tao D, Geiger P. Discovering temporal causal relations from subsampled data. ICML 2015, PMLR 37:1898–1906.
- Gross SM, Tibshirani R. Data shared lasso: a novel tool to discover uplift. Comput Stat Data Anal 2016;101:226–235. doi:10.1016/j.csda.2016.02.015.
- Günther W, Ninad U, Runge J. Causal discovery for time series from multiple datasets with latent contexts. UAI 2023, PMLR 216:766–776.
- Günther W, Popescu O-I, Rabel M, Ninad U, Gerhardus A, Runge J. Causal discovery with endogenous context variables. Adv Neural Inf Process Syst 2024;37. arXiv:2412.04981.
- Hahn M, Zajak A, Heider D, Ribeiro AH. Federated causal discovery across heterogeneous datasets under latent confounding. arXiv:2603.05149 (2026).
- Hähnel T, Raschka T, Sapienza S, et al. Progression subtypes in Parkinson's disease identified by a data-driven multi cohort analysis. npj Parkinsons Dis 2024;10:95.
- Halpin PF. Differential item functioning via robust scaling. Psychometrika 2024;89:796–821. doi:10.1007/s11336-024-09957-6.
- Hamaker EL, Kuiper RM, Grasman RPPP. A critique of the cross-lagged panel model. Psychol Methods 2015;20:102–116.
- Hernán MA, Hernández-Díaz S, Robins JM. A structural approach to selection bias. Epidemiology 2004;15:615–625.
- Höglinger GU, Adler CH, Berg D, et al. A biological classification of Parkinson's disease: the SynNeurGe research diagnostic criteria. Lancet Neurol 2024;23:191–204.
- Holford NHG, Chan PLS, Nutt JG, Kieburtz K, et al. Disease progression and pharmacodynamics in Parkinson disease: evidence for functional protection with levodopa and other treatments. J Pharmacokinet Pharmacodyn 2006;33:281–311. doi:10.1007/s10928-006-9012-6.
- Horváth K, Aschermann Z, Ács P, et al. Minimal clinically important difference on the motor examination part of MDS-UPDRS. Parkinsonism Relat Disord 2015;21:1421–1426.
- Horváth K, Aschermann Z, Kovács M, et al. Minimal clinically important differences for the experiences of daily living parts of the MDS-UPDRS. Mov Disord 2017;32:789–793.
- Huang B, Zhang K, Zhang J, et al. Causal discovery from heterogeneous/nonstationary data. J Mach Learn Res 2020;21.
- Hyttinen A, Plis S, Järvisalo M, Eberhardt F, Danks D. A constraint optimization approach to causal discovery from subsampled time series data. Int J Approx Reason 2017;90:208–225. doi:10.1016/j.ijar.2017.07.009.
- Jankovic J, McDermott M, Carter J, et al. Variable expression of Parkinson's disease: a base-line analysis of the DATATOP cohort. Neurology 1990;40:1529–1534.
- Jiang Z, Zhang D. Analysis of variance of multiple causal networks. Adv Neural Inf Process Syst 2023;36:11580–11591. doi:10.52202/075280-0510.
- Johnson AR, Bucks RS, Kane RT, Thomas MG, Gasson N, Loftus AM. Motor subtype as a predictor of future working memory performance in idiopathic Parkinson's disease. PLoS One 2016;11:e0152534. doi:10.1371/journal.pone.0152534.
- Kang H, Zhang A, Cai TT, Small DS. Instrumental variables estimation with some invalid instruments and its application to Mendelian randomization. J Am Stat Assoc 2016;111:132–144. doi:10.1080/01621459.2014.994705.
- Kang K, Gu Y. A blockwise mixed membership model for multivariate longitudinal data: discovering clinical heterogeneity and identifying Parkinson's disease subtypes. Ann Appl Stat 2026;20:452–475.
- Kuiper RM, Ryan O. Drawing conclusions from cross-lagged relationships: re-considering the role of the time-interval. Struct Equ Modeling 2018;25:809–823.
- Lakens D, Scheel AM, Isager PM. Equivalence testing for psychological research: a tutorial. Adv Methods Pract Psychol Sci 2018;1:259–269.
- Larsen TE, Hernández-Lorca M, Ekstrøm CT, Vinding R, Bønnelykke K, Strandberg-Larsen K, Petersen AH. Mapping the dynamic interplay of mental health and weight across childhood: data-driven explorations using causal discovery. medRxiv 2026. doi:10.64898/2026.04.16.26350943.
- Li A, Jaber A, Bareinboim E. Causal discovery from observational and interventional data across multiple environments. Adv Neural Inf Process Syst 2023;36. doi:10.52202/075280-0741.
- Li W, Li B, Wu A, et al. Causal discovery for irregularly time series with consistency guarantees (ReTimeCausal). ICML 2026, PMLR 306:67973–67999.
- Lian J, Luo X, Shan C, et al. Personalized progression modelling and prediction in Parkinson's disease with a novel multi-modal graph approach. npj Parkinsons Dis 2024;10:229.
- Lin W, Grewal SS. Measurement fidelity, not model complexity, governs individual cognitive trajectory prediction in Parkinson's disease. npj Digit Med 2026;9:698.
- Macleod AD, Counsell CE. Predictors of functional dependency in Parkinson's disease. Mov Disord 2016;31:1482–1488.
- Malik V, Bello K, Ghoshal A, Honorio J. Identifying causal changes between linear structural equation models. UAI 2024, PMLR 244:2383–2398.
- Mameche S, Cornanguer L, Ninad U, Vreeken J. SPACETIME: causal discovery from non-stationary time series. Proc AAAI Conf Artif Intell 2025;39:19405–19413.
- Marek K, Chowdhury S, Siderowf A, et al. The Parkinson's Progression Markers Initiative (PPMI): establishing a PD biomarker cohort. Ann Clin Transl Neurol 2018;5:1460–1477.
- Mestre TA, Fereshtehnejad SM, Berg D, et al. Parkinson's disease subtypes: critical appraisal and recommendations. J Parkinsons Dis 2021;11:395–404.
- Mooij JM, Magliacane S, Claassen T. Joint causal inference from multiple contexts. J Mach Learn Res 2020;21.
- Moons KGM, Damen JAA, Kaul T, et al. PROBAST+AI: an updated quality, risk of bias, and applicability assessment tool for prediction models using regression or artificial intelligence methods. BMJ 2025;388:e082505.
- Morris TP, White IR, Crowther MJ. Using simulation studies to evaluate statistical methods. Stat Med 2019;38:2074–2102.
- Mulder JD, Hamaker EL. Three extensions of the random intercept cross-lagged panel model. Struct Equ Modeling 2021;28:638–648. doi:10.1080/10705511.2020.1784738.
- Negida A, Mukhopadhyay N, Berman BD, Barrett MJ. Comparative analysis of progression milestones across Parkinson's disease clinical, pathological, and data-driven subtypes: a 10-year follow-up. npj Parkinsons Dis 2026;12:239.
- Nylund KL, Asparouhov T, Muthén BO. Deciding on the number of classes in latent class analysis and growth mixture modeling: a Monte Carlo simulation study. Struct Equ Modeling 2007;14:535–569. doi:10.1080/10705510701575396.
- Ollier E, Viallon V. Regression modelling on stratified data with the lasso. Biometrika 2017. doi:10.1093/biomet/asw065.
- Orth U, Meier LL, Bühler JL, et al. Effect size guidelines for cross-lagged effects. Psychol Methods 2024;29:421–433.
- Pal S, Ghosh D, Yang S. Penalized FCI for causal structure learning in a sparse DAG for biomarker discovery in Parkinson's disease. Ann Appl Stat 2026;20:1873–1893.
- Pan W, Su C, Maasch JRMA, et al. Learning phenotypic associations for Parkinson's disease with longitudinal clinical records. AMIA Jt Summits Transl Sci Proc 2024;2024:374–383.
- Perry R, von Kügelgen J, Schölkopf B. Causal discovery in heterogeneous environments under the sparse mechanism shift hypothesis. Adv Neural Inf Process Syst 2022;35:10904–10917.
- Plis S, Danks D, Freeman C, Calhoun V. Rate-agnostic (causal) structure learning. Adv Neural Inf Process Syst 2015;28:3303–3311.
- Popescu O-I, Günther W, Rabel M, Runge J. Causal discovery for time series with endogenous context variables. ICML 2026, PMLR 306.
- Pyatigorskaya N, Yahia-Cherif L, Valabregue R, et al. Parkinson disease propagation using MRI biomarkers and partial least squares path modeling. Neurology 2021;96:e460–e471. doi:10.1212/WNL.0000000000011155.
- Ramsey J, Andrews B, Spirtes P. Efficient latent variable causal discovery: combining score search and targeted testing. arXiv:2510.04263 (2025).
- Reisach A, Tami M, Seiler C, et al. A scale-invariant sorting criterion to find a causal order in additive noise models. NeurIPS 2023 (Adv Neural Inf Process Syst 36:785–807).
- Reisach AG, Seiler C, Weichwald S. Beware of the simulated DAG! Causal discovery benchmarks may be easy to game. NeurIPS 2021.
- Riley RD, Ensor J, Snell KIE, et al. Calculating the sample size required for developing a clinical prediction model. BMJ 2020;368:m441.
- Riley RD, Snell KIE, Archer L, et al. Evaluation of clinical prediction models (part 3): calculating the sample size required for an external validation study. BMJ 2024;384:e074821.
- Rocchi N, Zanga A, Bernasconi A, et al. A causal discovery workflow for rare diseases: experts-in-the-loop analysis of sparse longitudinal data. J Med Syst 2026;50:9. doi:10.1007/s10916-025-02327-4.
- Rouanet A, Philipps V, Taddé BO, Pérès K, Proust-Lima C. Modeling temporal relationships between multivariate repeated markers along with clinical endpoints: application to Alzheimer's disease and related dementias. Biom J 2026;68:e70157. doi:10.1002/bimj.70157.
- Ruer A, Loranchet T, Bystrova D, Assaad CK. Root cause analysis via difference graph discovery from linear time-series data. arXiv:2608.21117 (2026).
- Runge J. Discovering contemporaneous and lagged causal relations in autocorrelated nonlinear time series datasets. UAI 2020, PMLR 124.
- Saadawy M, Saadawy E. NeuroCausal prior-weighted structure learning (NCP-WSL): disentangling a shared latent causal manifold across Alzheimer's, Parkinson's and healthy aging. SSRN preprint 2026. doi:10.2139/ssrn.7119099.
- Saggioro E, de Wiljes J, Kretschmer M, Runge J. Reconstructing regime-dependent causal relationships from observational time series. Chaos 2020;30:113115.
- Schultzberg M, Muthén B. Number of subjects and time points needed for multilevel time-series analysis: a simulation study of dynamic structural equation modeling. Struct Equ Modeling 2018;25:495–515. doi:10.1080/10705511.2017.1392862.
- Schuurman NK, Hamaker EL. Measurement error and person-specific reliability in multilevel autoregressive modeling. Psychol Methods 2019;24:70–91. doi:10.1037/met0000188.
- Seo G, Shim J, Kim M, Byun H, Jung Y, Song K. Dissecting causal mechanism shifts via FANS: function and noise separation. ICML 2026, PMLR 306:109074–109104.
- Shah RD, Samworth RJ. Variable selection with error control: another look at stability selection. J R Stat Soc B 2013;75:55–80. doi:10.1111/j.1467-9868.2011.01034.x.
- Siderowf A, Concha-Marambio L, Lafontant DE, et al. Assessment of heterogeneity among participants in the PPMI cohort using α-synuclein seed amplification: a cross-sectional study. Lancet Neurol 2023;22:407–417.
- Simuni T, Caspell-Garcia C, Coffey C, et al. How stable are Parkinson's disease subtypes in de novo patients: analysis of the PPMI cohort? Parkinsonism Relat Disord 2016;28:62–67.
- Simuni T, Chahine LM, Poston K, et al. A biological definition of neuronal α-synuclein disease: towards an integrated staging system for research. Lancet Neurol 2024;23:178–190.
- Spirtes P, Glymour C, Scheines R. Causation, Prediction, and Search, 2nd ed. MIT Press, 2000.
- Stebbins GT, Goetz CG, Burn DJ, et al. How to identify tremor dominant and postural instability/gait difficulty groups with the MDS-UPDRS: comparison with the UPDRS. Mov Disord 2013;28:668–670.
- Stein G, Penzel N, Piater T, Denzler J. TCD-Arena: assessing robustness of time series causal discovery methods against assumption violations. ICLR 2026.
- Swait J, Louviere J. The role of the scale parameter in the estimation and comparison of multinomial logit models. J Mark Res 1993;30:305–314. doi:10.1177/002224379303000303.
- Tan MMX, Lawton MA, Pollard MI, et al. Genome-wide determinants of mortality and motor progression in Parkinson's disease. npj Parkinsons Dis 2024;10:113.
- van Borkulo CD, van Bork R, Boschloo L, et al. Comparing network structures on three aspects: a permutation test. Psychol Methods 2023;28:1273–1285.
- Varambally S, Ma YA, Yu R. Discovering mixtures of structural causal models from time series data. ICML 2024, PMLR 235.
- Vasey B, Nagendran M, Campbell B, et al. Reporting guideline for the early-stage clinical evaluation of decision support systems driven by artificial intelligence: DECIDE-AI. Nat Med 2022;28:924–933.
- Vickers AJ, Elkin EB. Decision curve analysis: a novel method for evaluating prediction models. Med Decis Making 2006;26:565–574.
- Voelkle MC, Oud JHL, Davidov E, Schmidt P. An SEM approach to continuous time modeling of panel data: relating authoritarianism and anomia. Psychol Methods 2012;17:176–192.
- Vu TC, Nutt JG, Holford NHG. Progression of motor and nonmotor features of Parkinson's disease and their response to treatment. Br J Clin Pharmacol 2012;74:267–283. doi:10.1111/j.1365-2125.2012.04192.x.
- Wang Y, Squires C, Belyaeva A, Uhler C. Direct estimation of differences in causal graphs. Adv Neural Inf Process Syst 2018;31. arXiv:1802.05631.
- Windmeijer F, Farbmacher H, Davies N, Davey Smith G. On the use of the lasso for instrumental variables estimation with some invalid instruments. J Am Stat Assoc 2019;114:1339–1350. doi:10.1080/01621459.2018.1498346.
- Witte J, Foraita R, Didelez V. Multiple imputation and test-wise deletion for causal discovery with incomplete cohort data. Stat Med 2022;41:4716–4743. doi:10.1002/sim.9535.
- Xu Y, Düker M-C, Matteson DS. Testing simultaneous diagonalizability. J Am Stat Assoc 2024;119:1513–1525. doi:10.1080/01621459.2023.2202435.
- Zhang J. On the completeness of orientation rules for causal discovery in the presence of latent confounders and selection bias. Artif Intell 2008;172:1873–1896.

New entries in this draft were checked against Crossref, arXiv or the proceedings pages on 2–3 November 2026 (the investigation report, the addendum and the Step 2 note give the source for each). The WP1 literature search checks the rest and extends the list.

## Appendix A. Candidate confirmatory hypotheses (for the panel to rate)

Each candidate is a link from year *t* to year *t*+1, a phenotype contrast and an expected direction *beyond speed*.
- The panel rates them in Section F of the expert questionnaire, which lists the links without these expected directions so as not to lead the answers.
- The panel keeps about ten, adds its own, and names two or three expected-null controls. It also chooses the two key links for Step 4 (D20).
- The rationales are proposals to check against the literature in WP1.

| # | Link (year t → year t+1) | Expected beyond speed | Proposed rationale |
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
| A10 | RBD symptoms → cognition | Stronger in PIGD (diffuse malignant on the secondary axis) | RBD marking a more diffuse disease; not a node when RBD status is an axis |
| A11 | Mood → daily function | No difference (control) | Depression affecting function in both |
| A12 | Treatment → motor complications | Panel to judge | Age and dose may matter more than phenotype |

On the RBD axis, the SOC model predicts "similar" for most links, with differences of speed and load (Borghammer 2021). The panel is asked whether any link should differ beyond speed.

## Appendix B. Pilot evidence (escalation-discovery, 29 October to 2 November 2026)

Aggregates from `../escalation-discovery` (its PROTOCOL.md §3, §10.1 and §12, and `results/`):

- **Cohort.**
  - 1,741 PD participants: PPMI 1.0 de novo 431, PPMI 2.0 sporadic 957, genetic PD 353.
  - 8,829 annual-grid rows; 8.9 / 3.1 / 5.9 rows per participant.
- **Missing values on the annual grid** (PPMI 1.0 / 2.0 / genetic PD):
  - OFF motor exam 30 / 18 / 43%;
  - axial score 28 / 17 / 41%;
  - MoCA 6 / 7 / 12%;
  - Part IV items 4–8%; questionnaires 3–7%; doses 0%.

  In an earlier PPMI 1.0 two-year table, 64% of transitions had a missing value.
- **Cost** (Apple M4, 16 GB; 43 columns, 7,220 transition rows):
  - FCI with the cluster-robust test took 17.5 min per fit with separating sets capped at 3; uncapped, it did not finish.
  - PC with the same test took about 42 min.
  - Tetrad's tiered BOSS-FCI, GFCI and FCI ran out of memory.
  - J-PCMCI+ took over 29 min per fit.
  - DECI did not converge in over an hour.
  - The deep-learning and pretrained learners took 2.6 to 25 min per fit, single-threaded.
- **Tuning.** On the PPMI-fitted simulator, no tuning job finished in about 12 hours: 8 hours with 8 workers on 1 November, then 4 hours with 4 workers on 2 November. Each job ran every learner and configuration in turn and saved only at the end.
- **Memory.** On 1 November, eight parallel jobs with deep-learning learners exhausted 16 GB. The operating system ended the session, and the run, started from an app that had quit, lost access to its files.
- **Simulator fidelity.**
  - Over 544 scored pairs, the median absolute difference between real and simulated Spearman correlations was 0.062 (linear simulator) and 0.059 (boosted simulator), against a pre-set target of 0.05.
  - The 90th percentile was 0.13.
- **Sortability.** On synthetic data, variance-sortnregress reached adjacency F1 0.70, against 0.46–0.63 for the classical learners at default settings.
- **Constant variables.** With a context variable constant in a subset, PC (no depth cap) searched every separating set and ran for 34 hours.
- **Expert-map test** (earlier work, ppmi-causal). Of 701 implied independences, 24.8 / 28.0 / 22.4% were violated in PPMI 1.0 / 2.0 / genetic PD. For maps relabelled within tiers the figures were 38.6 / 36.0 / 33.4% (p = 0.0099 in each).
- **Not produced.**
  - No learner was evaluated, and no causal discovery was run on PPMI.
  - The levodopa-decision audit and the core-map test did run, but they concern the escalation decision and are not used here.

## Appendix C. Prospective case-report form: minimum fields (Part II-c)

| Form | Field group | Specification |
|---|---|---|
| Enrolment | Site, pseudonymous ID, screening status, eligibility, consent date | Site-held identifiers kept apart from analysis extracts |
| History | Diagnosis date, motor phenotype, RBD status, comorbidities, past care | Source-verified; versioned case definitions |
| Visit | Planned date, actual date, visit type, setting, reason for a missed or moved visit | Actual dates mandatory |
| Symptoms | MDS-UPDRS II and III with ON/OFF state, cognition, selected non-motor scales | Instrument version and rater recorded |
| Function | Schwab & England, independence state, support needs, date of change if known | Pre-specified threshold and adjudication |
| Treatment | Medications with start, stop and dose; rehabilitation and other services | Indication recorded where possible |
| Events | Falls, admissions, move to a care home, death | Source, date certainty, adjudicator |
| Follow-up status | Active, withdrawn, lost, died, administrative end | Dropout kept distinct from death |
| Provenance | Entry date, correction history, instrument version | Audit trail and quality flags |

## Appendix D. Changes from v0.3

The changes from the two v0.2 drafts to v0.3 are listed in v0.3's Appendix D (`docs/archive/PROTOCOL_v0.3.md`).

- **Framework adopted.** Speed-adjusted subgroup causal discovery (investigation report §5).
  - One continuous-time engine with a speed and a load for each participant serves Steps 2 to 5.
  - The five Steps replace Analyses A to C:
    - Analysis A becomes Step 1;
    - Analysis B becomes the transparent reference of Step 2 (§7.3.5);
    - Analysis C becomes Step 5, gated on H-P and speed-adjusted.
- **Questions.**
  - Q1 now asks whether every patient progresses through the same pathway, for phenotypes, classes and individuals.
  - Q2 separates speed, load and mechanism. Q3 splits routes into speed and mechanism parts.
  - M1 now names current methods.
- **Estimands.**
  - Added: speed ratio, speed spread, load, noise, the speed-adjusted deviation Δ, the share explained by speed, and the total-effect split.
  - The smallest effect of interest is renamed ε (it was Δ).
  - The MCID of MDS-UPDRS II gains its worsening value (2.51).
  - The link classes apply to Δ. The speed-or-pathway rule (v0.3 §7.5) is replaced by domain verdicts (§7.7). In v0.3, speed was a level difference; it is now a time scale, and load is the level difference.
- **Hypotheses.**
  - H-P and H-L are primary, under Holm.
  - H-C is gated on H-P.
  - H-D, H-I and H-V are key secondary.
  - H-S is retired.
  - H-M1 now names the comparators and covers the speed scenarios.
- **Phenotypes.** RBD status is proposed as a co-primary axis (D18). Continuous tremor and gait scores enter a supportive analysis.
- **Time.**
  - The engine uses actual intervals, so two-year transitions can be used.
  - The interval rule now applies to Step 1 only.
  - Treatment is a symptomatic effect, not scaled by speed (D24).
- **Step 1.** Added:
  - tiered FCI theory;
  - tests stratified by phenotype;
  - fedCI and unequal-variance tests among the WP3 candidates;
  - test-wise deletion with multiple imputation;
  - S-FCI and tiered IOD cross-checks;
  - a BOSS-FCI sensitivity learner.
- **Step 2.** New components:
  - the participant-level speed-only test;
  - speed-adjusted joint discovery with stability selection;
  - per-link tests within one joint fit, anchor-free primary and anchored sensitivity (D22);
  - per-domain tests;
  - the extended definition-matched null;
  - the shared-eigenvector check;
  - the two-axis factorial model.
- **Steps 3 and 4 are new:** latent mechanism classes and individual speed and load, in PPMI 1.0. v0.3 had put data-driven subtypes out of scope because of weak identification with few visits. The synthetic check supports them with about seven transitions per person, under a power gate.
- **Simulation.**
  - S1 is redefined as time scale plus load. S0L, S1V and S5 to S7 are added, with identification stresses.
  - Named comparators C1 to C11 are added.
  - Performance is reported by type of link, including false root-cause domains and footprint-aware localisation.
  - The compute is re-estimated at about 239 hours, from the measured cost of the synthetic checks.
- **Missing data in the engine.** Multiple imputation within phenotype, combined by Rubin's rules, with a state-space sensitivity analysis.
- **Continuous-time check.** ctsem becomes the second host of the speed model, with measurement error. DynNet is a naive comparator.
- **Prior work and novelty.** The claims are narrowed (§1.4) and the prior-work table extended. The estimator template is credited to Mendelian randomisation and anchor-free item-bias testing.
- **Part II.** C3 becomes the shared map at each patient's estimated speed. II-a replicates speed and mechanism verdicts, using the engine across visit intervals.
- **Decisions.** D18 to D24 are added. D12 is renamed (ε), and D13 is limited to Step 1.

## Appendix E. Synthetic evidence behind the design

All checks use synthetic data only, from the scripts in `scripts/toy/`, run with the ppmi-causal Python environment. The investigation report (§2, §8) and the Step 2 note (§3) give full tables.

**Check 1. Speed versus pathway** (`toy_speed_vs_pathway.py`).
- *Setting:* six domains; 800 TD and 250 PIGD patients with five yearly transitions each; PIGD twice as fast; 500 cohorts per scenario.
- *Results:*
  - link-wise tests flagged a difference in 100% of speed-only cohorts;
  - the group-level speed-only test rejected 5.6% of speed-only cohorts, and 90.8% with an extra link;
  - the largest speed-adjusted deviation fell on the true link in 97% of cohorts.

**Check 2. Latent classes** (`toy_latent_classes.py`).
- *Setting:* two classes (40% and 60%), the minority twice as fast; 200 cohorts per cell.
- *Results:*
  - with 430 patients × 7 transitions: 6.0% false rejections, 69% power, adjusted Rand index 0.76 and 0.82;
  - with 950 × 2: 12% false rejections, 21% power, adjusted Rand index 0.34 and 0.47.

**Check 3. Speed varying within phenotypes** (`toy_random_speed.py`, `toy_random_speed_scale.py`, `toy_random_speed_scale_person.py`).
- *Setting:* three domains; 400 TD and 200 PIGD patients; log-speed spread 0.4; cohorts 1, 4 and 16 times larger.
- *Results:*
  - the group-level test rejected 10.5% of speed-only cohorts at the base size, rising to 27–65% at 16 times;
  - the participant-level test rejected 4.6% of 1,000 cohorts across all sizes, with at least as much power.

**Check 4. Method families against the speed-adjusted procedures** (`toy_comparators.py`, `toy_comparators_extra.py`, `toy_comparators_joint.py`).
- *Setting:*
  - six domains; 800 TD and 250 PIGD patients; speed spread 0.4; a load difference; 600 cohorts;
  - four scenarios: equal speed (S0), PIGD twice as fast (S1), plus a PIGD-only link (S2), plus a stronger link (S3).
- *Speed only:*
  - yearly methods flagged 55–98% of shared links, in 100% of cohorts;
  - naive continuous time flagged 63%;
  - joint discovery flagged 0.4%, and the joint per-link tests 0.25–1.0%;
  - the global test rejected 6.5% of cohorts.
- *One real difference:*
  - yearly methods isolated it in 0–1% of cohorts;
  - joint discovery isolated it in 96–97%, and the joint per-link tests in 89–94%;
  - one-at-a-time score tests isolated it in only 35% and 4%.
- *Equal speed:*
  - separate graphs disagreed in 99.5% of cohorts;
  - the shared-plus-deviation lasso flagged 28–71% of links;
  - the omnibus yearly invariance test rejected 15% of cohorts.

**Check 5. tsLDiffPC and tsDCI as published** (`ts_ldiffpc.py`, `toy_tsldiffpc.py`).
- *Setting:* the cohorts of check 4, with a faithful re-implementation of the authors' search (`toy_tsldiffpc_validation.py`). It reproduced the paper's published root-cause F1 within Monte Carlo error.
- *Speed only:*
  - cross-links were rarely flagged (0.7%);
  - persistence was flagged in 47% of domains, so 48% of domains were named false root causes.
- *One real difference:* power was 49% for a new link (79% with conditioning sets capped at 3) and 10% for a stronger link.
- *Against a speed-matched TD reference:* false root causes fell to 0.2%, but power fell to 35% and 4%.

**Limits.**
- The checks are linear and Gaussian, with no hidden burden, dropout, ordinal scores or definition artefacts.
- In check 4 the comparators are analogues written from each family's published definition. In check 5 the algorithm follows the authors' code.
- WP3 repeats every comparison on the PPMI-fitted simulator, with the authors' software.

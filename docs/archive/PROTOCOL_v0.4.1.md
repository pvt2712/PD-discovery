# Phenotype-specific progression mechanisms in Parkinson's disease: speed-adjusted causal discovery in PPMI (Part I) and clinical translation (Part II)

> Protocol draft v0.4.1, 3 November 2026, for review by the investigator, movement-disorder neurologists and a statistician. It is not approved, and no phenotype analysis of PPMI data has been run.
>
> Since v0.4, the protocol follows the framework recommended in the investigation report (`docs/pathway_heterogeneity_framework.md`): speed-adjusted subgroup causal discovery. It also takes in three companion documents: the methods and novelty addendum (`docs/methods_and_novelty_update.md`), the Step 2 methods note (`docs/step2_comparison_methods.md`) and the tsLDiffPC/tsDCI note (`docs/tsldiffpc_tsdci_investigation.md`).
>
> Appendix D lists the changes from v0.3 and from v0.4, both archived in `docs/archive/`. Appendix E summarises the synthetic evidence behind the design.
>
> Part I is Paper 2, experiment 1. Part II goes ahead only if Part I passes its gate (§13), and it needs its own data access, funding and ethics approval. Thresholds marked *provisional* are fixed at registration, or by the pre-registered WP3 algorithms (§3, §17).

## Summary

**Clinical question.**
- Clinicians already sort Parkinson's disease (PD) into phenotypes, such as tremor-dominant (TD) versus postural instability and gait difficulty (PIGD), or by REM-sleep behaviour disorder (RBD). They know that some phenotypes progress faster.
- They do not know whether phenotypes decline through *different pathways* or through *the same pathways at a different speed*. A different pathway would be, for example, loss of independence driven by gait and cognition in one phenotype and by bradykinesia in another.
- Beyond phenotypes, they do not know whether every patient progresses through the same pathway.
- Guidance on treatment, monitoring and care is largely phenotype-independent: the NICE guideline NG71 (NICE 2017), the International Parkinson and Movement Disorder Society (MDS) evidence-based reviews (Fox 2018; Seppi 2019) and the American Academy of Neurology (AAN) guideline (Pringsheim 2021). If pathways differ, what to monitor and when to refer should differ. If only speed differs, the same plan applies at different intervals.

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
- In synthetic cohorts close to PPMI's size (1,050 patients with five transitions each) where phenotypes differed only in speed, every procedure that compares yearly coefficients declared mechanism differences in every cohort (Appendix E). These included:
  - link-wise interaction tests;
  - a difference-graph search restricted to time order;
  - shared-plus-deviation estimators;
  - the invariance tests of clinical panel research.
- A gradient-based structure learner run on each phenotype separately (DYNOTEARS) did the same in 99.5% of speed-only cohorts, and in 42.5% of cohorts with equal speed.
- The published difference-graph method for time series, tsLDiffPC/tsDCI, has a more lenient search. It rarely invented cross-links, but it named false "root causes" in about half of the domains and had low power for real differences.
- Speed must therefore be estimated and removed in continuous time, with a speed and a load for each patient.

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
- **Simulation benchmark.** It is fitted to PPMI and run before any PPMI analysis. It tests whether the speed-adjusted procedures keep false mechanism claims near the nominal rate (at most 7.5% for a 5% test), where named current methods do not (H-M1).
- **Part II** follows only if Part I passes its gate (§13).

**Primary tests.** H-M1 (simulation); H-P and H-L (PPMI), with Holm control; H-C1 to H-C10, gated on H-P.

**What is new.** The novelty claims are narrow (§1.4), and the WP1 literature search confirms them before any claim of priority. We found no study that:
- tests whether directed progression mechanisms differ between patient groups within a disease while groups and persons may differ in speed and load, against a null calibrated for those differences. In PD, directed models fitted by phenotype have so far been confirmatory;
- tests whether groups or persons share one multivariate causal mechanism up to a common time-scale factor. Time-scale-only hypotheses are tested for growth, trait evolution and univariate progression, and disease-progression models give each patient a pace along one shared course of scores, but none compares directed links between domains;
- compares groups' longitudinal causal networks against a cohort-fitted null that re-applies the group-definition rule. The closest precedents re-apply a clustering rule (Liu 2008) or simulate a homogeneous null within severity strata (Hoekstra 2026).

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
- The TD/PIGD classification (Jankovic 1990) is the most widely used, in its version for the MDS-sponsored revision of the Unified Parkinson's Disease Rating Scale (MDS-UPDRS; Stebbins 2013).
- PIGD starts with more severe motor disease and shows faster cognitive decline and more hallucinations and psychosis than TD (Aleksovski 2018). The classification is unstable in early PD: within a year, 39% of PIGD and 18% of TD patients in PPMI change class (Simuni 2016).

**Clinical subtypes.** Fereshtehnejad's subtypes (mild-motor predominant, intermediate, diffuse malignant) are defined from:
- motor severity;
- cognition;
- RBD;
- dysautonomia.

They separate progression rates in PPMI (Fereshtehnejad 2017), refining clusters first found in a Montreal cohort (Fereshtehnejad 2015).

**RBD and body-first disease.** RBD at diagnosis marks a body-first course in the α-synuclein origin and connectome (SOC) model (Borghammer 2021). The model predicts earlier non-motor load and faster progression along shared propagation routes: mostly speed and load, not different mechanisms.

**Calls for better subtypes.** An MDS task force asked for subtypes that are reproducible and clinically or biologically meaningful (Mestre 2021).

**Data-driven subtypes.** Progression subtypes have been found across cohorts: PPMI with PDBP (Dadu 2022); PPMI, ICEBERG and LuxPARK (Hähnel 2024). In the multi-cohort analysis, baseline differences between subtypes were small and differences in speed large (Hähnel 2024). Disease-progression models, including machine-learning ones, instead place every patient on one shared course with their own onset and pace (§1.4).

**Biological classifications.** SynNeurGe (Höglinger 2024) and the neuronal α-synuclein disease integrated staging system (NSD-ISS; Simuni 2024) use α-synuclein seed amplification. PPMI has measured it (Siderowf 2023), but our local download does not hold it.

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
4. **Absence cannot be shown by non-significance.** A link that is significant in one phenotype and not in the other is no evidence of a difference (Gelman & Stern 2006). Absence can only be supported as "smaller than a pre-set negligible size", by an equivalence test (Lakens 2018).
5. **In yearly data, speed looks like a different pathway.**
   - Suppose two groups share one mechanism and one runs it s times faster. Their yearly effects are then related by a matrix power, B(fast) = B(slow)^s, which has four consequences:
     - every yearly coefficient changes, persistence most;
     - an effect passing through an intermediate domain within the year appears as a yearly "link";
     - the yearly noise grows;
     - a faster group can even show a *weaker* yearly link once one year passes the link's peak (Kuiper & Ryan 2018).
   - When patients within a phenotype run at different speeds, group-average yearly dynamics are no longer a matrix power at all.
   - Two literatures describe the same problem: coarse sampling (Plis 2015; Hyttinen 2017; Gong 2015) and interval dependence in panel models (Voelkle 2012; Kuiper & Ryan 2018; Driver 2025).
   - In continuous time, speed is a single factor, and which links exist does not depend on it. The comparison is therefore made there, with a speed for each patient (§7.1).
6. **Few annual visits.**
   - PPMI gives on average 3 (PPMI 2.0) to 9 (PPMI 1.0) annual visits per person.
   - Methods built for long series reduce here to a pooled search over two time slices. Examples are PCMCI+ (Runge 2020), LPCMCI (Gerhardus & Runge 2020), J-PCMCI+ and regime methods.
   - In Step 1 the unit of analysis is the year-to-year transition, pooled over participants. Steps 2 to 5 model each participant's sequence of transitions.
7. **Treatment feedback.**
   - PPMI participants are untreated at enrolment, and most start dopaminergic treatment during follow-up.
   - Treatment responds to symptoms and changes them.
   - It improves motor scores on its own timetable (Holford 2006; Vu 2012), so it is not scaled by a patient's disease speed.
8. **Small groups.** The minority phenotype may hold a few hundred participants. In groups that small, an absent link usually means low power.

### 1.4 Prior work and novelty

The following are established, and this study claims none of them:
- clinical phenotypes and their prognosis (§1.1);
- data-driven longitudinal progression subtypes across cohorts;
- disease-progression models, including machine-learning ones, that place every patient on one shared course with their own time shift and pace:
  - disease course mapping (Schiratti 2017), applied in PPMI (Poulet & Durrleman 2023) and from isolated RBD to PD (Di Folco 2024);
  - a latent-time model in PPMI (Jin 2024);
  - Subtype and Stage Inference, which gives each patient a subtype and a stage (Young 2018);
- longitudinal symptom-association networks and graph-informed prognosis in PD;
- causal discovery across contexts, mixtures and regimes, and benchmarks of time-series causal discovery under violated assumptions;
- causal discovery on cross-sectional PPMI biomarkers (penalised Fast Causal Inference, FCI; Pal 2026);
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
| Disease course mapping (Schiratti 2017; in PPMI, Poulet & Durrleman 2023) | Every patient follows one shared multivariate course, with their own time shift and pace and a "space shift" in the order of change. In PPMI, the space shifts recover TD-like and PIGD-like patterns | Trajectories of scores, not which domain drives which. A space shift can arise from different starting states under one mechanism |
| Latent-time progression model in PPMI (Jin 2024) | Progression 22% slower in women and 25% slower in LRRK2 G2019S carriers: in effect a speed-only model | Speed is assumed to be the only difference; mechanism is not tested |
| Subtype and Stage Inference (Young 2018) | Data-driven subtypes, each with its own order of abnormality, and a stage for each patient | Order of events from cross-sectional data, not which domain drives which; no speed |
| Phenotypic association graphs from longitudinal records (Pan 2024) | Symptom association graphs over time in PPMI, PDBP and BioFIND | Do the associations survive causal assumptions and a speed-adjusted phenotype comparison? |
| Multi-modal graph prognosis, AdaMedGraph (Lian 2024) | Graph-informed progression prediction in PPMI and PDBP | Do phenotype-specific mechanisms add to the phenotype label and to a patient's speed? |
| Blockwise mixed-membership subtyping (Kang & Gu 2026) | Heterogeneity across patients, domains and time | Are the differences structural, or speed? |
| Progression milestones across clinical, pathological and data-driven subtypes (Negida 2026) | Outcomes compared between subtype systems over 10 years | Mechanism, once selection, severity and speed are accounted for? |
| Measurement fidelity in cognitive trajectory prediction (Lin & Grewal 2026) | The limits measurement sets on individual prediction | Does measurement quality also limit graph recovery? |
| Motor subtype and later working memory (Johnson 2016) | A directed model of TD versus PIGD over two waves | Pre-specified paths; disease duration differed by subtype, so stage and speed are mixed |
| MRI propagation by RBD status (Pyatigorskaya 2021) | Directed propagation models compared by RBD | Imaging, cross-sectional; speed not separated |
| A shared causal manifold across AD, PD and ageing (Saadawy & Saadawy 2026, preprint) | "One graph, different rates" across diagnoses | No formal test; compares diagnoses, not phenotypes within PD |
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
- **Pathways beyond speed and load.** We found no study that tests whether directed progression mechanisms differ between patient groups within a disease while groups and persons may differ in speed and load, against a null calibrated for those differences. Group-specific models of clinical course compare dynamics directly, often between groups defined by stage, outcome or speed, which builds in the trap of §1.3, point 5. In PD, directed models fitted by phenotype are confirmatory (Johnson 2016; Pyatigorskaya 2021).
- **A shared time scale.** We found no test of whether groups or persons share one multivariate causal or lagged mechanism up to a common time-scale factor. Time-scale-only hypotheses are tested for trait evolution (Clavel 2015), growth curves (Cole 2010) and univariate progression (Raket 2022). A subject-level rate multiplies network diffusion on a fixed connectome (Raj 2015); a neural ordinary differential equation gives each PPMI patient a speed along one shared course (Wang 2025); disease-progression models give each patient a pace along one shared course of scores (Schiratti 2017; Poulet & Durrleman 2023). None of these learns or compares directed links between domains.
- **Definition artefacts.** We found no comparison of groups' longitudinal or causal networks against a cohort-fitted null that re-applies the group-definition rule. SigClust re-applies a clustering rule to data simulated from one fitted model (Liu 2008); person-specific dynamics have been compared within severity strata against a simulated homogeneous null (Hoekstra 2026); and selection on sum scores is analysed by Haslbeck 2021.
- No PD study was found that formally tested whether lagged effects differ between patient subgroups.

The WP1 literature search, with its databases, queries and dates recorded, confirms or corrects this before any claim of priority. Four sources remain to be read in full: Larsen 2026, Saadawy & Saadawy 2026, Rocchi 2026, and the cited-by lists of four method papers.

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

  dX = s_i·[(A + Δ_g)·X + b_g + ℓ_i]·dt + √s_i·G_g·dW, with log s_i = γ_g + β′z_i + σ_g·u_i, u_i ~ N(0, 1).

- ℓ_i is the participant's own load, a stable push on each domain. It is normally distributed around a mean that depends on z_i and on u_i, so speed and load may be correlated.
- z_i is pre-specified: era, age at onset, sex, disease duration, and the year-0 and year-1 states. Conditioning on the states handles the link between where a participant starts and how fast they progress (Wooldridge 2005). Era also shifts the noise.
- It is observed at the annual visits, using the actual intervals. Treatment enters as a symptomatic effect on the observed scores, which may differ by phenotype (§6.1).
- The reference phenotype has Δ = 0 and γ = 0: TD on the motor axis, RBD-negative on the RBD axis, and mild-motor predominant on the clinical-subtype axis.

**Estimands.**
- **Speed ratio** s_g = exp(γ_g): how much faster phenotype g's typical (median) patient runs the reference mechanism. It is reported averaged over each phenotype's z_i, because z_i includes year-0 states that partly define the phenotype. Its estimate by the minimal-L1 rule (§7.1) is ŝ.
- **Speed spread** σ_g: the standard deviation of log-speed between patients within phenotype g.
- **Load** b_g: the steady push on each domain, per year at the reference speed; participants' own loads ℓ_i vary around it. **Noise** G_g: the year-to-year variability the model does not explain.
- **Link strength** A_jk: the direct effect of domain k on domain j, per year at the reference speed. For clinicians it is also reported as the yearly effect at each phenotype's typical speed.
- **Speed-adjusted deviation** Δ_jk: how much the link differs in phenotype g once both phenotypes run at the same speed. A link "differs beyond speed" when Δ_jk ≠ 0.
- **Share explained by speed:** 1 − ‖A_g − ŝ·A_TD‖² / ‖A_g − A_TD‖², from free fits of each phenotype's links, with ŝ here the least-squares speed. It is bias-corrected and reported with a participant-bootstrap interval. It is unstable when the free fits barely differ, so it is reported only when ‖A_g − A_TD‖ clearly exceeds its bootstrap noise.
- **Total effect** of domain X on independence over h years. It is split into a speed part and a mechanism part, averaged over both orders of the split and over the phenotype's speed distribution (§7.8).
- **Smallest effect of interest ε** (called Δ in v0.3).
  - Definition: the size below which a deviation is clinically negligible. *Provisional:* 0.10 standard deviations of the target domain per standard deviation of the source domain, per year at the reference speed, using the standard deviations of the pooled baseline scores on the normal-scores scale.
  - ε is a drift rate. The yearly coefficient it implies at the observed persistence is shown beside it.
  - The panel sees it in clinical units, and it is fixed in WP2 (D12). ε = 0.05 is a sensitivity analysis.
  - For comparison, psychological panel studies call standardised yearly cross-lagged effects of 0.03, 0.07 and 0.12 small, medium and large (Orth 2024). ε is compared with them on the yearly scale.

Effects are reported per minimal clinically important difference (MCID) of the source domain where one exists, and per standard deviation otherwise. Two examples:
- 4.63 points of MDS-UPDRS III for worsening, and 3.25 for improvement (Horváth 2015);
- 3.05 points of MDS-UPDRS II for improvement, and 2.51 for worsening (Horváth 2017).

**Classes per link.** Each tested link falls into one class, judged on its speed-adjusted deviation.

| Class | Rule |
|---|---|
| Similar | The 90% confidence interval of Δ_jk lies within ±ε (an equivalence test at 5%) |
| Different strength beyond speed | Δ_jk ≠ 0 under the family's false-discovery control, and the link is non-zero with the same sign in both phenotypes at a common speed |
| Structurally different | Δ_jk ≠ 0, and either the link is non-zero in one phenotype with its 90% interval within ±ε in the other, or it has opposite signs |
| Different, type unresolved | Δ_jk ≠ 0, but neither of the two rules above holds |
| Unresolved | None of the above. Reported with the largest deviation the data allow (the upper confidence bound of \|Δ_jk\|) |

**Domain verdicts** (§7.7) replace v0.3's speed-or-pathway rule. For every domain the decomposition reports four things:
- the phenotype's speed, which is common to all domains;
- its load difference;
- its noise difference;
- whether any link into it differs beyond speed.

A phenotype differs by **speed and load only** (its noise and treatment response may also differ) when every link's 90% interval lies within ±ε. This is an intersection–union test, which needs no multiplicity adjustment (Berger 1982). When H-P does not reject but that is not shown, the verdict is **no mechanism difference detected**, which is not evidence of absence (§1.3, point 4).

At PPMI's sizes many links will be unresolved. They are reported as unresolved, never as "no difference" or "absent". WP3 measures how often each class can be reached at ε = 0.10 and at ε = 0.05.

### 2.4 Hypotheses

- **H-M1 (primary methodological test, simulation).** Consider shared mechanisms with speed and load differences, hidden burden and definition artefacts: scenarios S0, S0L, S1, S1S, S1T, S1V, S4, S6 and S7 (§8.2).
  - **Calibration (the test).** Every speed-adjusted procedure (global, per-link and per-domain) keeps its rate of false mechanism claims at or below 7.5%, Bradley's liberal criterion for a 5% test (Bradley 1978). The rate is the probability that a dataset produces any false claim. The rate per truly shared link is reported beside it, with a standard error clustered by dataset.
  - **Benchmark (descriptive).** The named comparators (§8.4) run on the same datasets. They estimate yearly-scale differences, which are truly non-zero under speed, so the benchmark measures how often their output would be read as a mechanism difference. The comparative claim holds if at least one comparator's rate of false links or false root-cause domains has a lower 95% bound above 7.5%.
- **H-P (primary, PPMI).** The primary phenotypes differ in mechanism beyond speed and load.
  - Statistic: the likelihood ratio in the patient-level model, comparing the phenotype acting on speed, load, noise and treatment response only with it acting on the links as well.
  - Reference: the extended definition-matched null (up to 1,000 null datasets, with sequential stopping; §7.3.6).
  - Reported with the share of the difference that speed explains.
- **H-L (primary, PPMI 1.0).** Data-driven progression classes differ in mechanism beyond speed and load.
  - Statistic: the likelihood ratio of speed classes against mechanism classes at the selected number of classes, with a random speed within each class.
  - Reference: a parametric bootstrap from the fitted speed-class model that repeats the choice of the number of classes (up to 100 replicates, with sequential stopping; §7.4).
  - Descriptive only if the power gate fails (§8.7).
- **Multiplicity of the primary PPMI tests.**
  - Holm across the primary family: the smaller p-value is tested at 2.5% and the other at 5%.
  - If the panel makes RBD status a co-primary axis (D18), the family becomes H-P (TD/PIGD), H-P (RBD) and H-L, under Holm at an overall 5%.
  - If H-L becomes descriptive (§8.7), it leaves the family, and Holm runs over the remaining tests.
- **H-C1 to H-C10 (gated confirmatory tests, PPMI).**
  - About ten pre-specified links, each with an expected direction beyond speed (candidates in Appendix A), frozen with the clinical panel in WP1. The number of links is fixed before registration.
  - The links on an axis are tested only if that axis's H-P is rejected within the Holm procedure. They are speed-adjusted per-link tests (§7.6), with Benjamini–Hochberg control at 5% across the family: a false-discovery rate, not a family-wise error rate.
  - The tests are two-sided. A rejection opposite to the expected direction counts as a discovery and is reported as such. The expected-null controls are reported outside the family.
- **H-T (secondary, PPMI).** Does the response to dopaminergic treatment differ by phenotype (candidates A6, A7 and A12)? Tested on the engine's phenotype-specific treatment effects (§6.1), with Benjamini–Hochberg control.
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
| WP0 Feasibility audit | Aggregate counts only: phenotype and RBD group sizes, visits and follow-up per participant by era, transitions, missing data, phenotype stability, treatment timing, actual visit intervals, and events for Part II's endpoint | Feasibility tables for the clinical panel | The power target: the synthetic engine, re-run with WP0's counts, follow-up and missingness, gives at least 80% power for H-P at α = 0.025 against one deviation of 2ε. Steps 3 and 4 need a median of at least six post-landmark transitions per PPMI 1.0 participant |
| WP1 Clinical consensus | A two-round questionnaire completed by movement-disorder neurologists (target five, at least three; D25), plus a literature search. It settles: phenotype definitions and axes (including RBD); domains; the expert map; confirmatory hypotheses and ε; the two key links; and the clinicians' forecasts of speed, load or mechanism | Frozen definitions, map and hypothesis family; recorded forecasts | Panel decisions recorded |
| WP2 Protocol freeze | Final protocol and statistical analysis plan | Registered protocol | Investigator approval |
| WP3 Simulation benchmark | PPMI-fitted simulator with patient-level speed and load; scenarios S0 to S7 with stresses; the named comparators; validation of the continuous-time engine and of its second host | Settings, calibration, power and class recovery at real sizes; the H-M1 result | H-M1 holds for the speed-adjusted procedures; the power gate decides the status of H-L |
| WP4 PPMI analyses | Steps 1 to 5; the expert-map test | Phenotype and class progression atlas; pathway cards | — |
| WP5 Review and robustness | Blinded clinician review; replication by era; sensitivity analyses; comparison with the recorded forecasts | Graded findings | Part II gate (§13) |
| WP6 Reporting | Clinical paper; methods paper | Manuscripts for the PPMI Data and Publications Committee | — |
| Part II (conditional) | External replication; prognostic value; outlines of a prospective validation and a treatment-effect study | Separate protocols and approvals | §13 |

No phenotype-split, class or individual-level analysis of PPMI is run before WP2: the clinicians build the map, choose the hypotheses and record their forecasts without seeing data-driven results.

**Registration in two stages.**
- WP2 registers the protocol and the statistical analysis plan.
- WP3 then makes some choices by pre-registered algorithms: the Step 1 test and configurations, the replicate numbers, a replacement for a procedure of record that fails H-M1, and H-L's status. Their outcomes are logged in a time-stamped amendment before WP4.
- Fitting the simulator to PPMI uses the pooled data and makes no phenotype comparison.

**Oversight.**
- Part I: the investigator, the supervisors, a statistician and the clinical panel.
- Part II adds allied-health clinicians, site coordinators, and patient and public representatives. It also needs independent data-governance and clinical-safety review before any prospective recruitment.

## Part I. Phenotype causal discovery in PPMI

## 4. Population and data

- **Source.** PPMI (Marek 2018), under the investigator's Data Use Agreement, from the local download (clinical, imaging-summary and genetic files; no biospecimen category). No external cohort is accessible; Part II needs applications.
- **Primary population.** Sporadic PD untreated at the baseline visit, which makes every phenotype assignment an OFF-medication assessment:
  - PPMI 1.0 de novo PD (431 in the pilot build, about 9 annual visits each);
  - PPMI 2.0 sporadic PD (957 in the pilot build, about 3 annual visits each). Participants treated at baseline are excluded; WP0 reports how many.
- **Which data carry which step.**
  - Steps 1, 2 and 5 use both cohorts.
  - Steps 3 and 4 rest on PPMI 1.0. In a synthetic check, latent classes were recovered well with about seven transitions per person, and poorly with two (Appendix E).
  - PPMI 2.0 replicates class-level results only.
- **Secondary population.** Genetic PD, often treated at enrolment and later in the disease, is analysed separately as exploratory.
- **Annual grid.**
  - For each participant and year, the visit closest to the enrolment anniversary within ±4 months, with the most complete core battery.
  - The actual visit dates are kept, because the continuous-time engine uses the actual intervals (§6.3).
- **Landmark.** The phenotype is assigned at year 0. The primary progression analyses use transitions from year 1 onwards (year *t* to *t*+1, *t* ≥ 1). Four consequences:
  - Neither the year-0 values that define the phenotype nor their measurement error enter an analysed row. This removes the direct regression to the mean of the defining values. An indirect part remains when later scores are measured with error, because the label then still predicts later values; the extended null carries it through simulated measurement error (§7.3.6).
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
- frequent enough for the power target (§3).

### 5.2 Candidates

| System | Rule (operational details fixed in WP1) | PPMI inputs | For | Against |
|---|---|---|---|---|
| Motor phenotype: TD / PIGD / indeterminate | Ratio of mean tremor items (2.10, 3.15–3.18) to mean PIGD items (2.12, 2.13, 3.10–3.12); TD ≥ 1.15, PIGD ≤ 0.90 (Stebbins 2013) | MDS-UPDRS II and III items | Universal; used daily | Unstable in early PD; the ratio rule creates selection artefacts; indeterminate group |
| RBD status | RBD screening questionnaire above a cut-off at baseline | RBDSQ | Recognisable; prognostic for cognition; a proxy for body-first disease, for which the SOC model predicts speed and load differences along shared routes (Borghammer 2021) | A questionnaire, not polysomnography: in de novo PD its sensitivity was 0.44 and its specificity 0.84 at a score of 6 or more (Halsband 2018). Misclassification pulls differences towards zero |
| Clinical subtype: mild-motor / intermediate / diffuse malignant | Motor summary, cognition, RBD and dysautonomia scores against 75th-percentile cut-offs (Fereshtehnejad 2017) | MDS-UPDRS, MoCA and cognitive battery, RBD questionnaire, SCOPA-AUT | Prognostic; validated in PPMI | Built from many graph variables; small diffuse-malignant group; cohort-dependent cut-offs |
| Age at onset | Early, typical or late onset (cut-offs in WP1) | Diagnosis history | Exogenous; large groups | A demographic axis more than a phenotype |
| Genotype (exploratory) | GBA or LRRK2 carrier versus non-carrier | Genetic consensus file | Biological; exogenous | Small groups; carriers enrolled differently |

### 5.3 Provisional choice

- **Primary axis: motor phenotype, TD versus PIGD**, the clinicians' daily language.
  - Participants with an indeterminate phenotype are left out of Steps 2 and 5 on this axis; Step 1 uses all participants.
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
- Nodes share no items. MDS-UPDRS II items 2.10 (tremor), 2.12 (walking and balance) and 2.13 (freezing) leave the daily-function node, because tremor and gait have their own nodes. The simulator still generates these items, because the TD/PIGD rule needs them (§7.3.6).
- Phenotype stability (year 0 against years 1 and 2) is reported in WP0. Reassignment at year 1 is a sensitivity analysis, flagged as using later information.

## 6. Variables, time and the expert map

### 6.1 Progression domains (provisional; final list in WP1)

| Domain | Measure | Notes |
|---|---|---|
| Tremor | MDS-UPDRS III tremor items | Needed for the phenotype contrast |
| Bradykinesia and rigidity | MDS-UPDRS III items 3.3–3.8 and 3.14 | Speech and facial expression (3.1, 3.2) assigned in WP1 |
| Axial and gait | MDS-UPDRS III items 3.9–3.13 | Includes freezing and postural stability |
| Daily function | MDS-UPDRS II without items 2.10, 2.12, 2.13 | |
| Independence | Modified Schwab & England activities of daily living (ADL) scale | 80% is the usual dependence threshold |
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
- It enters as a symptomatic effect of the current dose on the observed scores at the visit, following pharmacological progression models (Holford 2006; Vu 2012). The effect is an Emax function of the levodopa-equivalent dose for each motor domain and for daily function, with an indicator of the examination state (OFF or ON). A linear dose effect replaces the Emax form if the curve is not identified.
- It is not scaled by the patient's speed.
- Its size may differ by phenotype in both models of H-P, so a different treatment response is never read as a mechanism difference. Phenotype differences in response are tested separately (H-T; candidates A6, A7 and A12).
- The simulator includes a treatment-initiation model fitted to PPMI (the hazard of starting treatment, given the current state, era and phenotype), so confounding by indication is present in every null dataset.
- An effect of treatment on the underlying drift is a sensitivity analysis. It covers a long-duration response and the functional protection that Holford 2006 reports.

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
  - Otherwise, the interval is added to Step 1's tables as a baseline-tier variable and conditioned on in every test.
- **Steps 2 to 5 use the continuous-time engine** (§7.1) with the actual interval of every transition. Irregular spacing is therefore handled exactly. A transition across a missed visit enters as a two-year transition; excluding such transitions is a sensitivity analysis.
- **Interpretation.** The engine's drift entries are direct effects per year at the reference speed, under the model, relative to the measured domains. They are not identified causal arrows by themselves: Step 1's map supplies the causal reading.
- **From the yearly map to drift entries.** Step 1's two-slice graph describes the yearly transition, which includes paths within the year.
  - A drift entry X → Y enters the support when Step 1 finds an adjacency from X in year t to Y in year t+1, or when the expert map proposes it.
  - A same-year adjacency can come from a within-year path in either direction, so it adds both entries.
  - Entries whose adjacency may reflect hidden confounding are kept and flagged.
  - Where a fitted yearly matrix has complex or negative eigenvalues, other drift matrices can give the same yearly transitions (Culver 1966). The engine reports when this happens, and WP3 measures how often it does on the simulator.

### 6.4 Expert phenotype-progression map (built in WP1)

- **Rating links.** Each neurologist completes the expert questionnaire (`docs/clinical_expert_questionnaire.md`) independently.
  - Its Section E rates every ordered pair of domains: could X this year directly change Y next year?
  - Codes: 2 strong, 1 weak, 0 none, ? unsure, with a minus sign for an effect that improves Y.
- **Other questions.**
  - The questionnaire also asks for impossible links, and for unmeasured factors that drive several domains (for example overall pathology burden).
  - Its answers on minimal clinically important differences (Section H) inform ε.
- **Consensus rules.**
  - A link enters the map when every rater gives it 1 or 2, and stays out when every rater gives it 0.
  - Every other link returns in round 2 with the group's anonymised answers. After round 2 the majority decides; a tie is settled by the investigator and recorded as disputed. With fewer than five raters, the map is described as investigator-led, and the analyses that depend on it are repeated without the disputed links (D25).
  - The same two rounds settle the phenotype definitions and axes, the domains, the confirmatory hypotheses and the two key links (D1 to D3, D5 to D7, D18, D20).
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
- **Continuous-time dynamics.** Each participant's domains follow the continuous-time model of §2.3. The links are A + Δ_g, the load is b_g + ℓ_i, and the noise G_g; all three are run at the participant's own speed s_i.
- **Speed.** Log-speed depends on the phenotype (or class), on the pre-specified characteristics z_i (§2.3), and on a random part whose standard deviation σ_g may differ by phenotype.
- **Load.** A participant-level random load ℓ_i adds to b_g in every step. Given the speed, it enters the transitions linearly, so it is integrated out exactly and the one-dimensional quadrature over speed remains. Its mean shifts with z_i and with the speed's random part, which lets speed and load correlate.
- **Yearly transitions.** They follow exactly from the continuous-time model over each actual interval:
  - the transition matrix is the matrix exponential of the links times the interval and the speed;
  - the intercept and the noise covariance follow from the same matrix exponential (Van Loan's method; Van Loan 1978).
- **Measurement.** Every score enters on one normal-scores transform, computed once on the pooled primary population over all visits and groups, with mid-ranks for ties, and then frozen.
  - A domain with more than half of its scores at a floor or ceiling (for example hallucinations or motor complications in early PD) enters through a latent-threshold model, or leaves H-P's support with the panel's agreement (D26). A latent-threshold domain makes the likelihood need numerical integration, which WP3 validates.
  - A version with measurement error is a sensitivity analysis (§7.12).

**Estimation.**
- Maximum likelihood, integrating each participant's log-speed by Gauss–Hermite quadrature (12 nodes). Given the speed, the participant load and any missing domain values are integrated out exactly by Kalman filtering, so the likelihood is exact under missing at random (§7.11).
- The free model is started both at its default and at the speed-only fit, so the likelihood ratio is never negative.
- Exact gradients and second derivatives are taken through the matrix exponential by automatic differentiation (PyTorch).
- The likelihood was validated against an independent implementation to machine precision (Appendix E).
- In the synthetic checks, the speed-only and free fits, the one-at-a-time per-link score tests and 12 penalised fits took about 30 seconds per cohort of 1,050 patients with six domains, without a participant load. The joint per-link and per-domain tests add further fits. Costs at ten domains, with the participant load, are measured before WP2 (§8.8).

**What separates speed from mechanism.**
- Speed moves every link and every domain's own rate together; a mechanism difference moves only some. The sparsity of Δ therefore identifies the speed.
- **The identifying rule.** The rule is minimal-L1: speed is the weighted median of the ratios of the two phenotypes' continuous-time entries, weighted by the size of the reference entries (Chen Y 2023; Bowden 2016).
- **An assumption, not a finding.** The rule recovers the true speed as long as the entries that deviate in the same direction carry less than half of the total weight. The data cannot check this, because a dense pattern of deviations fits as well as a sparse one. Deviations that mostly point one way also bias the estimate in finite samples.
- **Diagnostics (pre-specified).** Speeds estimated from all entries, from the domains' own rates only, from the cross-links only, and as the mode of the ratios (the plurality rule; Guo 2018) are compared. Together they give an interval of speeds that the data cannot tell apart.
- **Reporting.** A mechanism verdict is reported only if it holds for every speed in that interval. Otherwise the link is "speed and mechanism not separable", and the share explained by speed is reported.
- **Definition of speed.** A uniform strengthening of every link *and* every domain's own rate is speed by definition.
- **Anchored version (sensitivity).** It assumes that each domain's own rate shares the speed.

**Second host.**
- ctsem 3.11 can express the same model: a speed parameter with a random effect and a phenotype effect, referenced in every drift entry (Driver 2017; Driver 2026).
- It is validated by simulation in WP3, then used for the continuous-time check (§7.12).

### 7.2 Step 1: one shared causal map, all participants (Analysis A)

**Learner.**
- A latent-aware, constraint-based learner with tiers (FCI type; Spirtes 2000; Zhang 2008).
- Its theory with tiered background knowledge is now established (Bang & Didelez 2023, 2025).
- The phenotype labels sit in the baseline tier. This is joint causal inference with only the exogeneity assumption (Mooij 2020).

**Tests.**
- When a test conditions on the phenotype, it is stratified by phenotype, so that a link present in one phenotype only is not averaged away.
- WP3 chooses the conditional-independence test by accuracy and measured cost from three candidates:
  - Fisher-z on the normal scores with a cluster correction;
  - fedCI's likelihood-ratio tests for mixed and ordinal data (Hahn 2026);
  - a test robust to unequal variance, because patient-level speed makes residual variance depend on the current state.
- **Missing data:** test-wise deletion, and multiple imputation inside the tests (Witte 2022).

**Fast configuration for resampling.**
- Inside subsamples, nulls and the simulation, the same learner runs with the uncorrected Fisher-z test on the normal scores, stratified by phenotype as in the full test.
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
- The reported map, from the full fit.
- The support of Steps 2 to 5: the expert map's links together with the links the fast configuration selects on the real data. The same rule is re-applied in every null dataset (§7.3.6), so the Step 2 tests are calibrated after this selection.
- Marks for possible hidden confounding.
- The domains whose next-year value differs by phenotype among people with the same current state.

Those links signal heterogeneity, and Step 2 says whether it is speed, load or mechanism.

**Methods considered and their place.**

| Method | Decision | Reason |
|---|---|---|
| PCMCI+, LPCMCI | Not used as separate methods | With 1 to 8 transitions per person they reduce to the pooled two-slice search with tiers used here |
| Bagged-PCMCI+ (Debeire 2024) | Idea kept, tool not used | The participant bootstrap (§7.10) is the panel equivalent |
| Joint causal inference | Used in Step 1, with exogeneity only | The phenotype is not an external context (§1.3); J-PCMCI+ took over 29 minutes per fit in the pilot |
| Continuous-time models (Driver & Voelkle 2018) | The core engine of Steps 2 to 5; ctsem is the second host | Speed is a single factor only in continuous time |
| DynNet (Rouanet 2026) | Named naive comparator (§8.4) | It cannot host a patient-level speed without code changes; group effects enter each transition entry additively |
| Difference graphs: DCI (C6), tsLDiffPC and tsDCI (C5); DDAG optional; LDiffPC through its time-series version | Named comparators; the idea of testing against a re-targeted reference is kept, but done in one joint continuous-time fit | Speed changes yearly coefficients densely and breaks their equal-noise premises. tsLDiffPC/tsDCI's lenient search rarely flags cross-links but names false root causes through persistence, with low power; re-targeting removes the false root causes but not the low power (Appendix E; tsLDiffPC/tsDCI note) |
| Gradient-based structure learning: NOTEARS and DYNOTEARS (Zheng 2018; Pamfil 2020) | Named comparator C12 (§8.4); not the Step 1 learner | It assumes no hidden confounding, and on raw simulated data its accuracy leans on differences in variance that standardised scores remove (Reisach 2021). Run as one graph per phenotype, it compares yearly coefficients, so it cannot separate speed from mechanism |
| Shared plus deviations: data-shared lasso, multi-VAR, NetANOVA | Precedents for Step 2's layout; the first two are named comparators | None gives calibrated tests. multi-VAR needs long series. NetANOVA needs an instrument per node |
| Mixtures of causal models (for example MCD; Varambally 2024) | Used in Step 3, as mixtures of transition models with a random speed | PPMI 1.0's follow-up supports two to four classes; power gate (§8.7) |
| Continuous-time structural-equation-model (SEM) forests (Cáncer 2025) | Not used | Splitting on phenotype-defining variables recreates the artefact of §1.3 |
| SPACETIME (Mameche 2025), Regime-PCMCI (Saggioro 2020), ReTimeCausal (Li 2026) | Not used | Temporal regimes and long series; PPMI's intervals are handled by the engine |

### 7.3 Step 2: phenotypes, speed, load or mechanism

**7.3.1 Global speed-only test (H-P).**
- **Null model.** The phenotype acts on log-speed (its mean and spread), load, noise and treatment response only, with a common mechanism A. Each participant has their own speed and load.
- **Alternative.** The phenotype also acts on the links (A_g free).
- **Statistic.** The likelihood ratio between the two.
- **Reference distribution.** The extended definition-matched null (§7.3.6): up to 1,000 null datasets, stopping once 20 null statistics exceed the observed one (Besag & Clifford 1991). The smallest attainable p-value is then about 0.001.
  - A χ² reference held in the synthetic checks (5–6.5%).
  - The bootstrap is used because the real data add ordinal scores, dropout and burden, and because likelihood-ratio invariance tests in panel networks have proved "prone to false discovery" (Du 2026).
- **Why speed is per participant.** A test that gives every participant in a phenotype the same speed rejected far too often when speeds varied within phenotypes: 10.5% at the base size of 600 patients, and up to 65% in cohorts 16 times larger. The participant-level test stayed at 4.6% (Appendix E).
- **Reporting.** The test is reported with the share of the phenotype difference that speed explains (§2.3), with its cross-validated fit beside it.

**7.3.2 Speed-adjusted joint discovery (exploratory localisation).**
- **Model.** Both phenotypes' maps are estimated together: PIGD's links equal TD's links run at PIGD's speed, plus sparse deviations Δ.
- **Penalty.** An adaptive lasso on Δ, with the domains' own rates included and speed left unpenalised.
  - The weights come from free fits at the weighted-median speed.
  - This is the median-initialised adaptive lasso, which selects consistently (Windmeijer 2019).
- **Tuning.** The Bayesian information criterion (BIC) with the number of transitions in the penalty, with eBIC as a sensitivity analysis. Each selected support is refitted.
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
  - Each deviation's p-value comes from Monte Carlo draws of the free fit (the method of Chen Y 2023). Its calibration is checked in the extended null: if a link's null p-values are not uniform, its observed p-value is recalibrated against them.
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
- **Equal sizes.** Each phenotype group is subsampled n_sub times (*provisional* n_sub = 25 to 50) to the same size, 90% of the smaller group. The Step 1 learner, in its fast configuration, runs on every subsample, giving each link's selection frequency p(e, k) in phenotype k.
- **Statistic.** D(e) = p(e, TD) − p(e, PIGD), with adjacency and arrow marks compared separately.
- **Two references.**
  - 50 permuted-label datasets.
  - The extended definition-matched null (§7.3.6).
  - The gap between the two measures the artefact created by defining phenotypes from graph variables.
- **Decision.** Candidate links beyond the null:
  - each link's D(e) is standardised by its spread in the null, and the maximum over links in each null dataset gives a family-wise threshold (Westfall & Young 1993);
  - |D(e)| ≥ 0.25 (*provisional*).
- **Role.**
  - It assumes nothing about dynamics, so it shows clinicians where yearly graphs differ.
  - It is never a mechanism verdict. A pure speed difference also produces yearly differences, which the extended null now contains.

**7.3.6 The extended definition-matched null.**
- **One generator.** Every null dataset comes from one simulator, fitted to the pooled primary population under the null hypothesis. It has:
  - **one shared mechanism**, including a shared curvature (splines in the state and in disease duration);
  - a shared hidden burden, and participant-level speed and load;
  - measurement error matching published test–retest reliabilities;
  - treatment as in the engine, with a fitted treatment-initiation model;
  - dropout and death that depend on the simulated trajectory;
  - the real sizes, visit intervals and missingness.
- **Defining items.** It generates every variable the phenotype rules need. That includes the MDS-UPDRS items that define TD/PIGD but sit in no node (2.10, 2.12 and 2.13), as indicators of the tremor, axial and daily-function domains.
- **Matching the fitted nuisance values.** Speed, load, noise and treatment response depend on the simulated defining items and burden. They are fitted so that the re-derived phenotypes reproduce the real data's phenotype-level speed ratio, speed spread, load, noise and treatment response. Check: each of these, estimated on simulated data, must fall inside its real-data 95% interval.
- **Re-applying the rule.** The same phenotype rule is applied to the simulated baselines. Simulated phenotypes therefore inherit speed and load differences, range restriction, collider bias and regression to the mean, but no mechanism difference. No permutation of labels can reproduce these artefacts.
- **Running the procedure.** Every Step 2 procedure is re-run on each null dataset, starting from the selection of the support (§7.2). Only mechanism differences should exceed the null.
- **Size.** Up to 1,000 datasets for H-P, with sequential stopping (§7.3.1); 100 for the separate-graph reference.
- **If the fidelity or nuisance check fails,** H-P's p-value is the larger of two: the definition-matched p-value, and the conventional parametric-bootstrap p-value, with phenotype labels fixed and data simulated from the fitted speed-only model.

**7.3.7 Shared-eigenvector check (sensitivity).**
- **Why it works.** If one mechanism generates both phenotypes' yearly matrices, they commute and share eigenvectors, whatever the speeds.
- **The test.** A centred participant bootstrap of the commutator of the two yearly matrices, with no continuous-time model. A formal version exists (Xu 2024).
- **Synthetic check.** It rejected in 3.5–4.0% of cohorts without a mechanism difference. With one, it rejected in 28% (new link) and 78% (stronger link).
- **Role.** It supports H-P but never replaces it. Pooled regressions bias it when speed tracks disease stage.

**7.3.8 Phenotype axes.**
- **Priors.**
  - TD versus PIGD: PIGD is expected to be faster, with more cognitive and psychotic load (Aleksovski 2018). Whether its mechanisms differ is open.
  - RBD status: the SOC model predicts earlier non-motor load and a faster speed with shared propagation.
- **Each axis is analysed in its own model.** A joint model is a sensitivity analysis (§7.14, item 1): two axes, factorial, following NetANOVA's baseline-plus-differential layout (Jiang & Zhang 2023).
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
- **Number of classes.** K runs from 2 to 4, chosen by BIC on the mechanism-class model, with the cross-validated likelihood on held-out patients reported beside it. Every bootstrap replicate repeats the same choice (Nylund 2007).
- **Null distribution.** The whole procedure, including the choice of K, is repeated on data simulated from the fitted speed-class model: up to 100 replicates, stopping once 10 exceed the observed statistic (Besag & Clifford 1991).
- **Fitting.** On the real data each mixture fit uses 50 random starts, of which the best 10 run to convergence. Each replicate uses 5 random starts plus starts at the null fit and at the real-data estimate, so the likelihood ratio is never negative, and it re-selects the class-specific sparsity. A replicate whose fits fail to converge counts as exceeding the observed statistic.
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
  - With 430 patients and 7 transitions each, the test held its 5% level (6.0%), had 69% power for one extra link, and recovered classes well (adjusted Rand index 0.76 and 0.82). That check used a χ² reference with 35 degrees of freedom, K fixed at 2, and no random speed within classes.
  - With 950 patients and 2 transitions, it did none of these (Appendix E).

### 7.5 Step 4: individual speed and load (PPMI 1.0)

- **Speed and load.** Participant-level speed and load are random effects. Speed is one parameter per person informed by every link, so it is far better identified than separate person-level slopes.
- **Key links.**
  - Random deviations on at most two pre-specified key links (D20) are exploratory.
  - Variances of person-level lagged effects need about 50 or more occasions per person to be estimated well (Schultzberg & Muthén 2018), and PPMI 1.0 offers about seven after the landmark.
- **Outputs:**
  - how much patients vary in speed;
  - whether speed variation alone fits as well as models in which key links also vary;
  - the share of speed variation explained by phenotype, RBD status and age at onset;
  - the correlation between initial state and speed, modelled explicitly (Gistelinck 2021).

### 7.6 Step 5: confirmatory links (H-C1 to H-C10)

- **Hypotheses.** Each names a link X → Y, a phenotype contrast and an expected direction beyond speed.
- **Gate.** Tested only if H-P on the same axis is rejected within the Holm procedure.
- **Tests.**
  - The anchor-free per-link test (§7.3.3) on the pre-specified family, two-sided.
  - Benjamini–Hochberg at 5% (a false-discovery rate), on per-link p-values whose calibration is checked in the extended null (§7.3.3). The family size is fixed before registration.
  - These are the primary clinical tests.
- **Classes.** Each tested link is classed by the rules of §2.3, on its speed-adjusted deviation and the margin ε.
  - Equivalence tests that class a link as "similar" form their own family, with Holm at 5%. Those inside the "structurally different" rule are not adjusted further.
- **Reporting.** Effects on clinical scales: for example, points of Schwab & England per MCID of the axial score, per year at the reference speed and at each phenotype's typical speed.
- **Controls.** Two or three hypotheses where the panel expects no difference are tested alongside as controls and reported separately, outside the Benjamini–Hochberg family.
- **Sensitivity analyses.**
  - The anchored per-link test.
  - A participant random load.
  - ε = 0.05.
  - v0.3's yearly interaction model, as a reading on the observed scale:
    - Y(*t*+1) on splines of Y(*t*), X(*t*) and the adjustment set;
    - plus era and phenotype;
    - plus phenotype × X(*t*), restricted to the shared range of X.
- **When identification is in doubt** (the speed interval of §7.1 is wide enough to change a verdict):
  - the confirmatory verdicts are reported as "speed and mechanism not separable";
  - the share explained by speed is reported instead.

### 7.7 Domain verdicts and pathway cards (Q2)

**Domain verdicts.** For each phenotype (and each class), every domain gets a verdict with four parts:
- **Speed:** the phenotype's speed ratio, with its interval.
- **Load:** the load difference beyond speed, with Benjamini–Hochberg control across domains.
- **Noise:** the difference beyond speed in year-to-year variability that the model does not explain.
- **Mechanism:** the per-domain test (§7.3.4), and the classes of the links into the domain (§2.3).

**Overall verdicts.**
- A phenotype differs by **speed and load only** when every link's 90% interval lies within ±ε (§2.3).
- When H-P does not reject but that is not shown, the verdict is **no mechanism difference detected**, reported with the largest deviation the data allow.
- Findings that are unstable under subsampling, or inconsistent between eras, are graded **unstable**.

**The pathway card** (§11) collects these verdicts with the routes of §7.8.

### 7.8 Routes to loss of independence (Q3)

- **Total effects.** For each phenotype and class, the total effect of each domain on independence and daily function over h years (h = 1, 2 and 5). They are computed from the fitted continuous-time links, with participant-bootstrap intervals.
- **Exact split of the difference between phenotypes:**
  - a speed part, exp(h·s·A) − exp(h·A);
  - a mechanism part, exp(h·s·(A+Δ)) − exp(h·s·A);
  - the split depends on its order, so the two orders are averaged (a Shapley split);
  - each part is averaged over the phenotype's speed distribution, because the average of exp(h·s·A) over patients is not exp(h·s_g·A).
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
- **Replication.** Findings are checked in PPMI 1.0 and PPMI 2.0 separately (direction and size), and in random half-samples. A finding replicates when it has the same sign in the other era, with a 90% interval that excludes zero.
- **Negative controls.**
  - The Step 1 learner run without tiers: arrowheads into era, age or sex, and later-to-earlier edges.
  - The whole Step 2 pipeline on permuted phenotype labels: the floor of "mechanism differences" when the label carries no information.

### 7.11 Missing data, dropout and death

- **Step 1.** Constraint-based tests use test-wise deletion and multiple imputation inside the tests (Witte 2022).
- **Steps 2 to 5.**
  - Missing domain values are handled inside the likelihood: the engine integrates over them exactly by Kalman filtering (§7.1), which is valid under missing at random. Each null dataset carries the real missingness and is analysed the same way.
  - Multiple imputation within each phenotype (20 imputations), with likelihood-ratio statistics pooled by the Meng–Rubin rule (Meng & Rubin 1992), is a sensitivity analysis.
- **Missing not at random.** Missing OFF exams, MoCA and questionnaires are imputed worse by a pre-set shift, as a sensitivity analysis.
- **Informative dropout.** Two sensitivity analyses: inverse-probability-of-remaining weights, estimated at each visit from the previous state, era and phenotype; and a shared-parameter model in which dropout and death depend on the participant's speed. The null datasets already carry dropout that depends on the simulated trajectory (§7.3.6).
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
8. Missing-not-at-random shift; dropout weights; the shared-parameter dropout model; multiple imputation (§7.11).
9. Transitions in which treatment started excluded; treatment acting on the drift; a treatment response shared across phenotypes (v0.4's model).
10. The anchored per-link test in place of the anchor-free one.
11. ε = 0.05 instead of 0.10.
12. Two-year transitions (across a missed visit) excluded.
13. A group-level speed in place of the participant-level speed, as a quick screen known to be over-liberal.
14. Measurement-error state-space model in ctsem (§7.12).
15. The correlation between initial state and speed modelled jointly (Gistelinck 2021), instead of by conditioning on the year-0 and year-1 states.
16. Terms at a two-year lag (a check of the first-order Markov assumption).
17. No participant load (v0.4's primary model).
18. Floor- and ceiling-bound domains kept on the normal-scores scale, instead of the latent-threshold model.

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
  - Continuous-time transition models fitted to the pooled primary population over the frozen map, with the shared curvature of §7.3.6.
  - Participant-level speed and load depend on the baseline state, as fitted. In S0 and S0L the phenotypes' speed distributions are set equal.
  - Treatment enters as in the engine, with the fitted treatment-initiation model.
  - Scores carry measurement error matching published test–retest reliabilities, and are mapped to each variable's type (ordinal, bounded or continuous).
- **Hidden burden.** A person-level hidden burden enters every domain.
- **Baselines.** Baseline values come from the fitted joint distribution, including every variable a phenotype rule needs. The rule applied to simulated baselines therefore reproduces realistic phenotype frequencies.
- **Missingness, dropout and intervals.** Missingness is resampled from PPMI's own patterns by era, year and treatment status. Dropout and death depend on the simulated trajectory, through a model fitted to PPMI. Visit intervals are resampled from PPMI's actual intervals.
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
| S1S Speed spread | Shared mechanism; the spread of speed is 1.5 times larger in PIGD | Unequal speed spread not read as mechanism |
| S1T Treatment response | Shared mechanism; the axial response to treatment is weaker in PIGD | A treatment-response difference not read as mechanism |
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
- dropout tied to severity, and dropout tied to speed;
- hidden burden absent or strong;
- sample sizes halved and doubled;
- speed ratios 1.25, 1.5 and 2;
- era differences in speed and load;
- an initial state strongly correlated with speed;
- a misspecified support: one true link missing from the map, and one spurious link added;
- deviations in one direction carrying more than half of the weight, the identification limit.

**Mechanisms outside the engine's class,** so that the benchmark does not only test the engine on its own model: a discrete-time yearly model, ordinal thresholds on the latent scores, and drift that changes with disease duration. Each runs in S0 and S1 for the calibration part of H-M1.

### 8.3 Estimands

For every link, the true class on the speed-adjusted scale. For every domain, the true verdict. For each phenotype and class, the true Δ, speed and load. Also the true partial ancestral graph (PAG) over the measured variables (Step 1).

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

**Named comparators, run as published.** Each uses the authors' software where it exists, at its published or default settings unless stated, with a common α = 0.05 where the method has a test. C3 and C11 use this study's code. C5 and C12 run through re-implementations validated against the authors' published results (below). Software is installed only with the investigator's approval and pinned (D23).

**A speed-matched arm** runs C5 and C11 against a TD reference simulated at each PIGD patient's estimated speed (as in Appendix E, check 5), in 20 datasets each of S0 and S1. It shows how much of each comparator's failure speed matching removes.

| ID | Comparator | Family | Software | Expected under speed only |
|---|---|---|---|---|
| C1 | Multigroup random-intercept cross-lagged panel model (RI-CLPM): χ² difference with lagged effects equal over groups, per-edge score tests | Clinical SEM standard (Mulder & Hamaker 2021) | lavaan | Invariance rejected; persistence first |
| C2 | Multigroup panel graphical vector autoregression (GVAR) with equality constraints; IVPP global test and partial pruning | Network-psychometric standard (Epskamp 2020; Du 2026) | psychonetrics; IVPP | As C1 |
| C3 | NCT-style permutation test on temporal networks, labels permuted by participant | Network comparison (van Borkulo 2023) | Custom code following NCT | Rejected in every cohort |
| C4 | Naive multigroup continuous-time model: an additive phenotype effect on each drift entry | Continuous time | ctsem; DynNet | Every non-zero drift entry flagged; no zero entry |
| C5 | tsLDiffPC and tsDCI, two-slice tables, every conditioning set (a cap of 3 at ten domains) | Difference graphs for time series (Ruer 2026) | pyCIPHOD (beta; commit pinned) | Persistence differences read as root causes in about half of the domains; low power (Appendix E) |
| C6 | DCI, lag-to-current candidate pairs; stability-selection variant | Difference graphs (Wang 2018) | causaldag | Dense yearly difference graph |
| C7 | CD-NOD with time tiers | Node-level shift (Huang 2020) | causal-learn | Every domain a changing module |
| C8 | Sparse mechanism shift, Fisher-z test on the map | Node-level shift (Perry 2022) | sparse_shift | Every domain shifted |
| C9 | Data-shared lasso, and multi-VAR with known subgroups | Shared plus deviations (Gross & Tibshirani 2016; Fisher 2022; Crawford 2026) | glmnet; multivar | Dense deviations; a hybrid shared part |
| C10 | Joint causal inference and J-PCMCI+ context edges | Context nodes (Mooij 2020; Günther 2023) | causal-learn; tigramite | Phenotype → every domain |
| C11 | Separate graphs read naively, and v0.3's yearly link-wise interaction tests | Naive split; v0.3 | This study's code | Discordant links even without differences |
| C12 | DYNOTEARS, one graph per phenotype from the two-slice tables, compared link by link; the paper's large-sample settings (λ = 0.05; thresholds 0.3 for same-time and 0.1 for lagged edges) | Gradient-based structure learning (Zheng 2018; Pamfil 2020) | Re-implementation of the authors' causalnex code, validated against the paper; no installation | Discordant lagged links in nearly every cohort, and in 42.5% of equal-speed cohorts; never isolates a real difference (Appendix E) |

**Faithful re-implementation.**
- For C5, a re-implementation of the published algorithm, written from the authors' code, reproduced the paper's published results within Monte Carlo error (tsLDiffPC/tsDCI note, §3).
- It is also checked against pyCIPHOD itself on 20 datasets per scenario.
- The re-implementation then carries the full benchmark, because the authors' code tests each conditioning set through data-frame operations and is slow.
- For C12, DYNOTEARS was re-implemented from the authors' causalnex code, with the same objective, gradient, bounds and optimisation schedule. On the paper's own benchmark it reproduced the published F1 scores to within 0.02 on average, with a largest difference of 0.07 (Appendix E, check 6). It needs no installation.

**Excluded comparators.**
- DDAG (Malik 2024): optional, with no official code.
- iSCAN (Chen T 2023) and FANS (Seo 2026): excluded. Their theory is for nonlinear models, and FANS needs at least 5,000 samples per group.

**The sortability control** runs alongside.

### 8.5 Performance measures

Each is reported with its Monte Carlo standard error, as a function of cohort size and speed ratio:
- the probability that a dataset produces any false mechanism claim, the common metric across methods;
- the false mechanism-difference rate per truly shared link, separately for self-effects (persistence), direct links and links absent from the drift, with standard errors clustered by dataset;
- the rate of false root-cause domains;
- for methods that also return same-time edges (C5, C12), discordant same-time edges, reported separately;
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

- **Precision.** The Monte Carlo standard error of a per-dataset rate p estimated from R datasets is √(p(1 − p)/R): at p = 5%, 200 datasets give 1.5 percentage points. Per-link rates use standard errors clustered by dataset.
- **Speed-adjusted Step 2 procedures:** 200 datasets per scenario in S0 and S1 (the null scenarios at the heart of H-M1), and 100 in S0L, S1S, S1T, S1V, S2, S2b, S3, S4, S6 and S7. With 200 datasets, a procedure whose true rate is 5% passes the 7.5% criterion with probability about 0.95, and one whose true rate is 10% with probability about 0.14.
- **Named comparators:** 100 datasets per scenario in S0 and S1; 50 in S0L, S1V, S2 and S3; 20 in the others. J-PCMCI+ (part of C10) runs on 20 datasets each of S0 and S1 only, because it took over 29 minutes per fit in the pilot.
- **Step 1 learners:** 120 datasets per scenario.
- **Calibration against the extended null.** Every analysed dataset needs its own extended null, so checking that calibration is a nested simulation. It uses the warp-speed method (Giacomini 2013): each of the 200 datasets in S0 and S1 gets its own simulator fit and a single null dataset, and the null statistics pooled across datasets give the reference.
- **H-L's power gate.** The null distribution of H-L's statistic at Holm's level comes from 200 S6 datasets, and its power from 50 S5 datasets, with K fixed at its true value. The pre-specified mechanism difference is one class-specific link of size 2ε.
- The full designs (up to 1,000 null datasets for H-P; 100 for H-L and for the separate-graph reference) apply to the real data only.

### 8.7 Decision rules (pre-specified)

- **H-M1** holds if, in every shared-mechanism scenario (S0, S0L, S1, S1S, S1T, S1V, S4, S6 and S7), the global, per-link and per-domain speed-adjusted procedures keep the probability of any false mechanism claim at or below 7.5% (§2.4). Its comparative part holds if at least one named comparator's rate of false links or false root-cause domains has a lower 95% bound above 7.5%.
- **Partial pass.** If the speed-adjusted procedures fail only in S4 (curvature), the analysis continues with the fitted curvature in the null (§7.3.6), and the verdicts are flagged as conditional on it. A failure only under the mechanisms outside the engine's class is reported as a limit of the engine.
- **Choice of procedure.** The anchor-free per-link test stays the test of record (D22) unless it fails H-M1. WP4 then uses the calibrated speed-adjusted procedure with the highest power in S2 and S3. If none is calibrated, Part I reports only the shared map and the speed and load estimates, as associations.
- **Power gate for H-L.** If the latent-class test has power below 0.5 against the pre-specified mechanism difference (§8.6) at PPMI 1.0's follow-up, H-L is reported as descriptive and leaves the primary family.
- **Class reachability.** If equivalence is shown for fewer than half of the truly similar links at the chosen ε, two consequences follow:
  - the classes "similar" and "structurally different" are reported as out of reach at PPMI's sizes;
  - links are described by their estimated deviations and confidence bounds.
- **Identification.** The study reports how often the speed-adjusted procedures fail when deviations in one direction carry more than half of the weight. The real-data report then says whether the data sit near that limit.

### 8.8 Compute budget

Every component is timed on 5 synthetic datasets of the real size before WP2, and the replicate numbers below are then confirmed or cut. At six domains and 1,050 patients, part of the Step 2 suite took about 30 seconds per cohort (Appendix E). The table assumes:
- about 1 minute per engine fit and 8 minutes per full Step 2 suite at ten domains, with the participant load (the investigation report's 5 minutes per fit is the pessimistic bound);
- 30 seconds per fast Step 1 fit;
- about 3 minutes per mixture fit, for each start;
- about 10 minutes per simulator fit for the extended null;
- 1 minute per comparator fit, except J-PCMCI+ at 30 minutes (DYNOTEARS took under 0.1 second per fit at six domains);
- 4 parallel jobs: hours = runs × minutes per run ÷ 60 ÷ 4.

| Component | Size | Hours |
|---|---|---|
| H-P on the real data, with the extended null, including the per-link and per-domain tests | Up to 1,000 null datasets × 2 axes × (one fast Step 1 fit + 2 engine fits), plus the simulator fits | Up to 22 |
| Joint discovery and stability selection on the real data | 50 half-samples × 12 penalised fits × 2 axes | About 5 |
| Speed-adjusted procedures in the benchmark | 200 datasets in S0 and S1, and 100 in ten other scenarios; one Step 2 suite and one fast Step 1 fit each | About 50 |
| Mechanisms outside the engine's class | 3 mechanisms × S0 and S1 × 50 datasets | About 11 |
| Calibration against the extended null (warp-speed) | 400 datasets × (one simulator fit + one null run) | About 31 |
| Latent classes on the real data | Up to 100 replicates × 3 values of K × 2 versions × 7 starts, plus the real-data fits | Up to 56 |
| H-L's power gate in the benchmark | 250 datasets × 2 versions × 7 starts | About 44 |
| Individual speed and load (PPMI 1.0); the ctsem check, with its simulation validation | A few final fits, after validation | About 30 |
| Named comparators, with the speed-matched arm | 12 comparators × 540 datasets; J-PCMCI+ on 40 | About 33 |
| Step 1, separate-graph reference, stability | As in v0.3, without its nested evaluation | About 70 |
| Total | | About 352 (15 days) |

**Above the budget.** At the assumed costs the total exceeds the 10-day budget of D14 by about 110 hours.
- Two items are upper bounds that sequential stopping usually shortens (up to 78 hours together).
- The real-data latent-class bootstrap is not run if H-L becomes descriptive (up to 56 hours).
- If the measured total still exceeds the budget, the investigator either raises it (D14) or the design is cut in this order:
  1. stress cells and the mechanisms outside the engine's class, to 20 datasets each;
  2. the comparators' replicates, to 50 in S0 and S1 and 20 elsewhere;
  3. the warp-speed calibration, to 100 datasets each in S0 and S1;
  4. separate-graph subsamples, from 25 to 15;
  5. null datasets for the separate-graph reference, from 100 to 50.

A fast Step 1 fit costing more than 1 minute, the target of §7.2, means the fast configuration is redesigned (fewer columns, or a lower separating-set cap) before the benchmark runs.

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
  - Benjamini–Hochberg at 5% within H-C (gated on the H-P of its axis), within H-D, within H-T, and within each discovery analysis.
  - Equivalence tests that class links as "similar" form their own Holm family. The overall "speed and load only" verdict is an intersection–union test and needs no adjustment.
  - Everything else is labelled exploratory.
- **Calibration.**
  - Step 2 and Step 5 tests are referred to the extended definition-matched null (§7.3.6), with sequential Monte Carlo p-values for H-P and per-link p-values recalibrated where the null shows they need it.
  - H-L is referred to a parametric bootstrap that repeats the choice of the number of classes.
  - χ² references are reported as approximations only.
- **Clustering.** The participant is the unit of every resampling and the cluster of every standard error.
- **Monte Carlo error** is reported with every simulated rate.

## 10. Bias and validity

| Threat | How it misleads | Mitigation |
|---|---|---|
| A faster disease clock (speed) | Every yearly coefficient changes, within-year paths appear as links, and the noise grows: a pathway difference that is not there | Comparison in continuous time with a speed for each participant; the extended null; named comparators in WP3 |
| Speed varies within phenotypes | Group-averaged dynamics break the matrix-power relation; group-level speed tests become over-liberal as cohorts grow | Participant-level speed as a random effect |
| Speed spread differs by phenotype | A free mechanism absorbs the misfit, and H-P rejects without a mechanism difference | A speed spread for each phenotype in both models; scenario S1S |
| Persistence read as a root cause | Speed changes each domain's own year-to-year persistence most, so persistence differences look like changed mechanisms | Domains' own rates are part of the speed; anchor-free identification; per-domain tests |
| Selection by the phenotype rule | Within-phenotype graphs gain or lose links that do not differ in reality | Landmark; extended definition-matched null; pooled Step 1 |
| Range restriction | Links look weaker inside a phenotype whose defining scores vary less | Extended null; equal-size comparisons; shared-range restriction in the yearly reading |
| Regression to the mean | Extreme year-0 scores drift back, mimicking phenotype-specific change | Landmark: no year-0 values in analysed rows; measurement error in the extended null for the indirect part |
| Shared items | A node partly encodes the phenotype rule | No shared items between nodes |
| Hidden burden | Level differences everywhere, and lagged links without a causal path | A participant load in every model; burden in the simulator and in every null; latent-aware Step 1; scenario S1V |
| Non-significance read as absence | "Present in PIGD, absent in TD" produced by unequal power | Difference tests; equivalence margin ε; equal-size comparisons |
| Curvature | A shared curved mechanism looks like a mechanism difference across ranges | Normal-scores scales; the fitted shared curvature in the null; splines in the yearly reading; scenario S4 |
| Floors and ceilings, measurement error | Bounded scales flatten effects near their limits; error attenuates effects differently by range | One frozen pooled transform; latent thresholds for bound-heavy domains; measurement error in the null; the state-space check (§7.12) |
| Initial state linked to speed | Fast progressors are more advanced at entry, which biases pooled regressions | Random speed and load conditioned on the year-0 and year-1 states (Wooldridge 2005); joint modelling as sensitivity 15 |
| Speed and mechanism not separable | Deviations in one direction that carry half of the weight or more are absorbed into speed, and the data cannot show it | Minimal-L1 stated as an assumption; pre-specified diagnostics and a speed interval (§7.1); identification stresses in WP3 |
| Unequal group sizes | A link missing in the small group read as biology | Joint estimation; equal-size subsampling for the reference; power from WP3 |
| Phenotype instability | The baseline label misclassifies the later course | Stability reported; reassignment sensitivity analysis; data-driven classes (Step 3) |
| Treatment by indication | Treatment choice differs by phenotype and confounds progression links | Treatment as a node in Step 1; in the engine, a symptomatic effect not scaled by speed that may differ by phenotype; a fitted treatment-initiation model in the null; scenario S1T; transitions with a treatment start excluded in sensitivity |
| ON/OFF state | Motor scores mean different things on and off medication | Panel decision; the other state as sensitivity |
| Informative dropout | Those who stay look slower | Trajectory-dependent dropout in the null; dropout weights and a shared-parameter model as sensitivity analyses; nothing imputed after death |
| Era differences | PPMI 1.0 and 2.0 differ in enrolment and follow-up | Era acts on log-speed, load and noise in both H-P models; era stresses in WP3; replication by era |
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
  - Independence means Schwab & England ≥ 80%. Scores below 80% have been used to define dependency (Macleod & Counsell 2016).
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
| PM0 Clinical | Age, sex, disease duration, MDS-UPDRS II and III, MoCA, Schwab & England, Hoehn and Yahr, levodopa-equivalent dose | Is the work clinically competitive? |
| PM1 Longitudinal | PM0 plus each domain's history (last value and yearly slope), in a flexible learner | Does anything beat good use of the history? |
| PM2 Phenotype label | PM1 plus the phenotype | Is the label enough? |
| PM3 Shared map at the patient's speed | PM1 plus two-year forecasts from the shared continuous-time model, run at each patient's estimated speed and load | Does a shared mechanism with a personal clock add value? |
| PM4 Phenotype- or class-specific map | PM3 plus the deviations classed as different beyond speed | Do different mechanisms add value beyond speed? |

- **Primary contrasts.** "Adds value" requires both PM4 against PM2 and PM4 against PM3: an intersection–union test, with each contrast at 5%.
  - Primary metric: out-of-sample log loss.
  - Key secondary metric: net benefit at the clinician-chosen threshold (Vickers & Elkin 2006).
  - Calibration (intercept and slope), discrimination and the Brier score are reported for every model.
- **Placebo controls.** PM4 is also run with permuted phenotype labels in its transition model, and with random deviations of the same number. A gain that survives these comes from the model's form, not the mechanisms.
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
  - Stop or redesign if the minority phenotype cannot meet the power target (§3), even with the secondary axis or merged groups.
  - If PPMI 1.0 participants have a median of fewer than six post-landmark transitions, Steps 3 and 4 become descriptive.
  - Apply the interval rule to Step 1 (§6.3).
- **WP3.**
  - If H-M1 fails for the speed-adjusted procedures, Part I reports only the shared map and the speed and load estimates, as associations. A failure only in S4 is a partial pass (§8.7).
  - If the power gate fails, H-L is descriptive.
  - If the identification stresses show that the data sit near the limit (§8.7), mechanism verdicts are reported as "speed and mechanism not separable" where that applies.
- **Part I to Part II.** Part II starts only when three conditions hold:
  - at least one confirmatory or replicated-discovery difference beyond speed lies on a route to daily function or independence;
  - for II-b, there are enough endpoint events, or an accepted fallback (§12.4);
  - data access and approvals for the external cohort are in place.
- **II-b.** If PM4 does not beat both PM2 and PM3, the result is published as negative, and II-c is not pursued with this model.
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

- **Part I.** STROBE (von Elm 2007) for the observational analyses; the simulation in ADEMP structure (Morris 2019); the expert consensus by ACCORD (Gattrell 2024).
- **Part II.** TRIPOD+AI and PROBAST+AI for prognosis; TARGET for treatment effects; DECIDE-AI for any later live evaluation.
- **Outputs.**
  1. The registered protocol with the simulation specification (OSF).
  2. The clinical paper: the phenotype and class progression atlas with the pathway cards.
  3. A methods paper. It covers speed-adjusted comparison of causal mechanisms between patient groups, the extended definition-matched null, and the named-comparator benchmark.
  4. Part II papers, if its gates pass.
- Every result is reported, including null, unresolved and negative findings, and every protocol deviation. The clinicians' forecasts are reported beside the results.

## 16. Reproducibility, compute and quality assurance

- Registration on OSF in two stages (§3), with code and lockfiles for every environment (Python and R).
- **Unit tests:**
  - the annual grid, the landmark, the phenotype rules, participant-level splits and exclusions;
  - for the engine: the likelihood against an independent implementation, recovery of known speed, load and deviations, and calibration under the null on synthetic data.
- Seeds, configurations, costs and failures are logged for every fit.
- **Compute.** The machine is an Apple M4 with 10 cores and 16 GB.
  - At most 4 parallel jobs.
  - One fit per job, saved as it finishes and skipped on rerun.
  - Long runs are started from the Terminal app.
- The harness is built and tested on synthetic data before any PPMI run, and every component is timed on synthetic data of the real size before WP2 (§8.8). The synthetic checks behind this design are in `scripts/toy/` (Appendix E).
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
| D10 | Provisional thresholds | Separate-graph reference \|D\| ≥ 0.25; Benjamini–Hochberg 5%; fidelity ≤ 0.05; n_sub and replicate numbers by measured cost | Investigator | WP2 to WP3 |
| D11 | Registration | OSF in two stages: protocol and analysis plan in WP2; WP3's algorithmic choices as a time-stamped amendment before WP4 | Investigator | WP2 and WP3 |
| D12 | Smallest effect of interest ε (Δ in v0.3) | 0.10 SD per SD per year at the reference speed, shown in clinical units; 0.05 as sensitivity | Panel, statistician and investigator | WP2 |
| D13 | Interval rule for Step 1 | Annual tables if at least 90% of intervals lie within 12 ± 3 months; Steps 2 to 5 use actual intervals | Investigator | WP0 |
| D14 | Compute budget | The revised design needs about 15 days of 4-job compute for WP3 and WP4 at the assumed costs (§8.8): raise the budget to 15 days, or keep 10 days and apply the cuts of §8.8 | Investigator | Before WP2 |
| D15 | Scope of Part II | Outline only in this protocol; II-a and II-b planned after Part I's gate; II-c and II-d as separate protocols | Investigator and supervisors | Now |
| D16 | Part II endpoint | Alive and independent at 24 months, with the fallbacks of §12.4 | Clinical panel and statistician | Before Part II |
| D17 | External cohort | One application, to a cohort with MDS-UPDRS, a cognitive scale and Schwab & England | Investigator | After WP5 |
| D18 | RBD status as a co-primary axis | Yes, with Holm across H-P (TD/PIGD), H-P (RBD) and H-L | Clinical panel, round 2 | WP1 |
| D19 | Status of H-L | Primary, in PPMI 1.0, with the power gate of §8.7 | Investigator and statistician | WP2 |
| D20 | The two key links for Step 4 | Chosen by the panel from the confirmatory list | Clinical panel, round 2 | WP1 |
| D21 | Clinicians' forecasts of speed, load or mechanism | Recorded in round 2, before any analysis | Investigator | WP1 |
| D22 | Per-link test of record | Anchor-free (minimal-L1 speed, Monte Carlo p-values), unless it fails H-M1 (§8.7); anchored as sensitivity | Investigator and statistician | WP2 |
| D23 | Software for the WP3 comparators | Install R (lavaan, psychonetrics, IVPP, glmnet, multivar, ctsem, DynNet) and Python (causaldag, pyCIPHOD, sparse_shift); versions and commits pinned. DYNOTEARS (C12) is re-implemented and needs no installation | Investigator | Before WP3 |
| D24 | Treatment in the engine | A symptomatic Emax effect of dose on the observed scores, with an examination-state indicator, not scaled by speed, and allowed to differ by phenotype in both H-P models; a drift effect as sensitivity | Investigator and clinical panel | WP1 |
| D25 | Size of the clinical panel | At least five movement-disorder neurologists; with fewer (at least three), the map is described as investigator-led and the map-dependent analyses are repeated without the disputed links | Investigator and supervisors | WP1 |
| D26 | Floor- and ceiling-bound domains | A latent-threshold model for a domain with more than half of its scores at a bound, or its removal from H-P's support | Clinical panel and statistician | WP1 |

## 18. Timeline (provisional)

| Weeks | Work |
|---|---|
| 1–2 | WP0 feasibility audit |
| 2–6 | WP1 two-round expert questionnaire (round 2 adds the RBD axis, the key links and the forecasts); literature search |
| 7 | WP2 protocol freeze and registration (stage 1) |
| 6–12 | WP3: the harness and engine validation on synthetic data from week 6; the PPMI-fitted simulator and the benchmark once the map is frozen (WP3 and WP4 together need about 15 days of compute, or 10 with the cuts of §8.8; D14) |
| 12 | Registration amendment with WP3's choices (stage 2) |
| 12–14 | WP4 PPMI analyses |
| 14–17 | WP5 blinded review, replication, sensitivity analyses, comparison with the forecasts |
| 17–21 | WP6 manuscripts |
| After week 17, if the gate passes | Part II: external-cohort application and the II-a and II-b protocols, over months; II-c and II-d as separately funded protocols, over years |

## References

- Aleksovski D, Miljkovic D, Bravi D, Antonini A. Disease progression in Parkinson subtypes: the PPMI dataset. Neurol Sci 2018;39:1971–1976. doi:10.1007/s10072-018-3522-z.
- Allison PD. Comparing logit and probit coefficients across groups. Sociol Methods Res 1999;28:186–208. doi:10.1177/0049124199028002003.
- Bang CW, Didelez V. Do we become wiser with time? On causal equivalence with tiered background knowledge. UAI 2023, PMLR 216:119–129.
- Bang CW, Didelez V. Constraint-based causal discovery with tiered background knowledge and latent variables in single or overlapping datasets. CLeaR 2025, PMLR 275:1116–1146.
- Bauer DJ, Curran PJ. Distributional assumptions of growth mixture models: implications for overextraction of latent trajectory classes. Psychol Methods 2003;8:338–363. doi:10.1037/1082-989X.8.3.338.
- Berger RL. Multiparameter hypothesis testing and acceptance sampling. Technometrics 1982;24(4):295–300. doi:10.1080/00401706.1982.10487790.
- Besag J, Clifford P. Sequential Monte Carlo p-values. Biometrika 1991;78(2):301–304. doi:10.1093/biomet/78.2.301.
- Borghammer P. The α-synuclein origin and connectome model (SOC model) of Parkinson's disease: explaining motor asymmetry, non-motor phenotypes, and cognitive decline. J Parkinsons Dis 2021;11:455–474. doi:10.3233/JPD-202481.
- Bowden J, Davey Smith G, Haycock PC, Burgess S. Consistent estimation in Mendelian randomization with some invalid instruments using a weighted median estimator. Genet Epidemiol 2016;40:304–314. doi:10.1002/gepi.21965.
- Bradley JV. Robustness? Br J Math Stat Psychol 1978;31(2):144–152. doi:10.1111/j.2044-8317.1978.tb00581.x.
- Brumm MC, Siderowf A, Simuni T, et al. Parkinson's Progression Markers Initiative: a milestone-based strategy to monitor Parkinson's disease progression. J Parkinsons Dis 2023;13:899–916. doi:10.3233/JPD-223433.
- Bystrova D, Devijver E. Constraint-based difference graph discovery in a linear setting. arXiv:2606.13947 (2026).
- Cáncer PF, Arnold M, Estrada E, Voelkle MC. Continuous-time structural equation model forests. Psychol Methods 2025, advance online. doi:10.1037/met0000766.
- Cashin AG, Hansford HJ, Hernán MA, et al. Transparent reporting of observational studies emulating a target trial: the TARGET statement. BMJ 2025;390:e087179. doi:10.1136/bmj-2025-087179.
- Chen T, Bello K, Aragam B, Ravikumar P. iSCAN: identifying causal mechanism shifts among nonlinear additive noise models. Adv Neural Inf Process Syst 2023;36:44671–44706. doi:10.52202/075280-1935.
- Chen Y, Li C, Ouyang J, Xu G. DIF statistical inference without knowing anchoring items. Psychometrika 2023;88:1097–1122. doi:10.1007/s11336-023-09930-9.
- Clavel J, Escarguel G, Merceron G. mvMORPH: an R package for fitting multivariate evolutionary models to morphometric data. Methods Ecol Evol 2015;6(11):1311–1319. doi:10.1111/2041-210X.12420.
- Cole TJ, Donaldson MDC, Ben-Shlomo Y. SITAR—a useful instrument for growth curve analysis. Int J Epidemiol 2010;39(6):1558–1566. doi:10.1093/ije/dyq115.
- Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. BMJ 2024;385:e078378. doi:10.1136/bmj-2023-078378.
- Crawford CM, Park JJ, Chow S-M, Ernst AF, Pipiras V, Fisher ZF. Penalized subgrouping of heterogeneous time series. Multivariate Behav Res 2026;61:579–602. doi:10.1080/00273171.2026.2622120.
- Culver WJ. On the existence and uniqueness of the real logarithm of a matrix. Proc Am Math Soc 1966;17(5):1146–1151. doi:10.1090/S0002-9939-1966-0202740-6.
- Dadu A, Satone V, Kaur R, et al. Identification and prediction of Parkinson's disease subtypes and progression using machine learning in two cohorts. NPJ Parkinsons Dis 2022;8:172. doi:10.1038/s41531-022-00439-z.
- Debeire K, Gerhardus A, Runge J, Eyring V. Bootstrap aggregation and confidence measures to improve time series causal discovery. CLeaR 2024, PMLR 236:979–1007.
- Di Folco C, Couronné R, Arnulf I, et al. Charting disease trajectories from isolated REM sleep behavior disorder to Parkinson's disease. Mov Disord 2024;39(1):64–75. doi:10.1002/mds.29662.
- Driver C, Voelkle M, Oud H. ctsem: continuous time structural equation modelling. R package version 3.11.1; 2026. doi:10.32614/CRAN.package.ctsem.
- Driver CC. Inference with cross-lagged effects: problems in time. Psychol Methods 2025;30:174–202. doi:10.1037/met0000665.
- Driver CC, Oud JHL, Voelkle MC. Continuous time structural equation modeling with R package ctsem. J Stat Softw 2017;77(5). doi:10.18637/jss.v077.i05.
- Driver CC, Voelkle MC. Hierarchical Bayesian continuous time dynamic modeling. Psychol Methods 2018;23:774–799. doi:10.1037/met0000168.
- Du X, Urnes Johnson S, Epskamp S. The invariance partial pruning approach to the network comparison in time-series and panel data. Psychol Methods 2026;31:749–767. doi:10.1037/met0000824.
- Epskamp S. Psychometric network models from time-series and panel data. Psychometrika 2020;85:206–231. doi:10.1007/s11336-020-09697-3.
- Fereshtehnejad SM, Romenets SR, Anang JBM, Latreille V, Gagnon JF, Postuma RB. New clinical subtypes of Parkinson disease and their longitudinal progression: a prospective cohort comparison with other phenotypes. JAMA Neurol 2015;72:863–873. doi:10.1001/jamaneurol.2015.0703.
- Fereshtehnejad SM, Zeighami Y, Dagher A, Postuma RB. Clinical criteria for subtyping Parkinson's disease: biomarkers and longitudinal progression. Brain 2017;140:1959–1976. doi:10.1093/brain/awx118.
- Fiebig DG, Keane MP, Louviere J, Wasi N. The generalized multinomial logit model: accounting for scale and coefficient heterogeneity. Mark Sci 2010;29:393–421. doi:10.1287/mksc.1090.0508.
- Fisher ZF, Kim Y, Fredrickson BL, Pipiras V. Penalized estimation and forecasting of multiple subject intensive longitudinal data. Psychometrika 2022;87:403–431. doi:10.1007/s11336-021-09825-7.
- Fox SH, Katzenschlager R, Lim SY, et al. International Parkinson and Movement Disorder Society evidence-based medicine review: update on treatments for the motor symptoms of Parkinson's disease. Mov Disord 2018;33(8):1248–1266. doi:10.1002/mds.27372.
- Gattrell WT, Logullo P, van Zuuren EJ, et al. ACCORD (ACcurate COnsensus Reporting Document): a reporting guideline for consensus methods in biomedicine developed via a modified Delphi. PLoS Med 2024;21:e1004326. doi:10.1371/journal.pmed.1004326.
- Gelman A, Stern H. The difference between "significant" and "not significant" is not itself statistically significant. Am Stat 2006;60:328–331. doi:10.1198/000313006X152649.
- Gerhardus A, Runge J. High-recall causal discovery for autocorrelated time series with latent confounders. Adv Neural Inf Process Syst 2020;33:12615–12625.
- Giacomini R, Politis DN, White H. A warp-speed method for conducting Monte Carlo experiments involving bootstrap estimators. Econom Theory 2013;29(3):567–589. doi:10.1017/S0266466612000655.
- Gistelinck F, Loeys T, Flamant N. Multilevel autoregressive models when the number of time points is small. Struct Equ Modeling 2021;28:15–27. doi:10.1080/10705511.2020.1753517.
- Gong M, Zhang K, Schölkopf B, Tao D, Geiger P. Discovering temporal causal relations from subsampled data. ICML 2015, PMLR 37:1898–1906.
- Gross SM, Tibshirani R. Data shared lasso: a novel tool to discover uplift. Comput Stat Data Anal 2016;101:226–235. doi:10.1016/j.csda.2016.02.015.
- Günther W, Ninad U, Runge J. Causal discovery for time series from multiple datasets with latent contexts. UAI 2023, PMLR 216:766–776.
- Günther W, Popescu O-I, Rabel M, Ninad U, Gerhardus A, Runge J. Causal discovery with endogenous context variables. Adv Neural Inf Process Syst 2024;37:36243–36284. doi:10.52202/079017-1143.
- Guo Z, Kang H, Cai TT, Small DS. Confidence intervals for causal effects with invalid instruments by using two-stage hard thresholding with voting. J R Stat Soc Series B Stat Methodol 2018;80(4):793–815. doi:10.1111/rssb.12275.
- Hahn M, Zajak A, Heider D, Ribeiro AH. Federated causal discovery across heterogeneous datasets under latent confounding. arXiv:2603.05149 (2026).
- Hähnel T, Raschka T, Sapienza S, et al. Progression subtypes in Parkinson's disease identified by a data-driven multi cohort analysis. NPJ Parkinsons Dis 2024;10:95. doi:10.1038/s41531-024-00712-3.
- Halpin PF. Differential item functioning via robust scaling. Psychometrika 2024;89:796–821. doi:10.1007/s11336-024-09957-6.
- Halsband C, Zapf A, Sixel-Döring F, Trenkwalder C, Mollenhauer B. The REM sleep behavior disorder screening questionnaire is not valid in de novo Parkinson's disease. Mov Disord Clin Pract 2018;5(2):171–176. doi:10.1002/mdc3.12591.
- Hamaker EL, Kuiper RM, Grasman RPPP. A critique of the cross-lagged panel model. Psychol Methods 2015;20:102–116. doi:10.1037/a0038889.
- Haslbeck JMB, Ryan O, Dablander F. The sum of all fears: comparing networks based on symptom sum-scores. Psychol Methods 2021 (advance online). doi:10.1037/met0000418.
- Hernán MA, Hernández-Díaz S, Robins JM. A structural approach to selection bias. Epidemiology 2004;15:615–625. doi:10.1097/01.ede.0000135174.63482.43.
- Hoekstra RHA, Epskamp S, Nierenberg AA, Borsboom D, McNally RJ. Testing similarity in longitudinal networks: the Individual Network Invariance Test. Psychol Methods 2026;31(5):844–870. doi:10.1037/met0000638.
- Höglinger GU, Adler CH, Berg D, et al. A biological classification of Parkinson's disease: the SynNeurGe research diagnostic criteria. Lancet Neurol 2024;23:191–204. doi:10.1016/S1474-4422(23)00404-0.
- Holford NHG, Chan PLS, Nutt JG, Kieburtz K, Shoulson I; Parkinson Study Group. Disease progression and pharmacodynamics in Parkinson disease: evidence for functional protection with levodopa and other treatments. J Pharmacokinet Pharmacodyn 2006;33:281–311. doi:10.1007/s10928-006-9012-6.
- Horváth K, Aschermann Z, Ács P, et al. Minimal clinically important difference on the motor examination part of MDS-UPDRS. Parkinsonism Relat Disord 2015;21:1421–1426. doi:10.1016/j.parkreldis.2015.10.006.
- Horváth K, Aschermann Z, Kovács M, et al. Minimal clinically important differences for the experiences of daily living parts of movement disorder society–sponsored unified Parkinson's disease rating scale. Mov Disord 2017;32:789–793. doi:10.1002/mds.26960.
- Huang B, Zhang K, Zhang J, et al. Causal discovery from heterogeneous/nonstationary data. J Mach Learn Res 2020;21(89):1–53.
- Hyttinen A, Plis S, Järvisalo M, Eberhardt F, Danks D. A constraint optimization approach to causal discovery from subsampled time series data. Int J Approx Reason 2017;90:208–225. doi:10.1016/j.ijar.2017.07.009.
- Jankovic J, McDermott M, Carter J, et al. Variable expression of Parkinson's disease: a base-line analysis of the DATATOP cohort. Neurology 1990;40:1529–1534. doi:10.1212/WNL.40.10.1529.
- Jiang Z, Zhang D. Analysis of variance of multiple causal networks. Adv Neural Inf Process Syst 2023;36:11580–11591. doi:10.52202/075280-0510.
- Jin R, Yoshioka H, Sato H, Hisaka A. Data-driven disease progression model of Parkinson's disease and effect of sex and genetic variants. CPT Pharmacometrics Syst Pharmacol 2024;13(4):649–659. doi:10.1002/psp4.13112.
- Johnson AR, Bucks RS, Kane RT, Thomas MG, Gasson N, Loftus AM. Motor subtype as a predictor of future working memory performance in idiopathic Parkinson's disease. PLoS One 2016;11:e0152534. doi:10.1371/journal.pone.0152534.
- Kang H, Zhang A, Cai TT, Small DS. Instrumental variables estimation with some invalid instruments and its application to Mendelian randomization. J Am Stat Assoc 2016;111:132–144. doi:10.1080/01621459.2014.994705.
- Kang K, Gu Y. A blockwise mixed membership model for multivariate longitudinal data: discovering clinical heterogeneity and identifying Parkinson's disease subtypes. Ann Appl Stat 2026;20:452–475. doi:10.1214/25-AOAS2131.
- Kuiper RM, Ryan O. Drawing conclusions from cross-lagged relationships: re-considering the role of the time-interval. Struct Equ Modeling 2018;25:809–823. doi:10.1080/10705511.2018.1431046.
- Lakens D, Scheel AM, Isager PM. Equivalence testing for psychological research: a tutorial. Adv Methods Pract Psychol Sci 2018;1:259–269. doi:10.1177/2515245918770963.
- Larsen TE, Hernández-Lorca M, Ekstrøm CT, et al. Mapping the dynamic interplay of mental health and weight across childhood: data-driven explorations using causal discovery. medRxiv 2026. doi:10.64898/2026.04.16.26350943.
- Li A, Jaber A, Bareinboim E. Causal discovery from observational and interventional data across multiple environments. Adv Neural Inf Process Syst 2023;36:16942–16956. doi:10.52202/075280-0741.
- Li W, Li B, Wu A, et al. Causal discovery for irregularly time series with consistency guarantees. ICML 2026, PMLR 306:67973–67999.
- Lian J, Luo X, Shan C, et al. Personalized progression modelling and prediction in Parkinson's disease with a novel multi-modal graph approach. NPJ Parkinsons Dis 2024;10:229. doi:10.1038/s41531-024-00832-w.
- Lin W, Grewal SS. Measurement fidelity, not model complexity, governs individual cognitive trajectory prediction in Parkinson's disease. NPJ Digit Med 2026;9:698. doi:10.1038/s41746-026-02681-8.
- Liu Y, Hayes DN, Nobel A, Marron JS. Statistical significance of clustering for high-dimension, low-sample size data. J Am Stat Assoc 2008;103(483):1281–1293. doi:10.1198/016214508000000454.
- Macleod AD, Counsell CE. Predictors of functional dependency in Parkinson's disease. Mov Disord 2016;31:1482–1488. doi:10.1002/mds.26751.
- Malik V, Bello K, Ghoshal A, Honorio J. Identifying causal changes between linear structural equation models. UAI 2024, PMLR 244:2383–2398.
- Mameche S, Cornanguer L, Ninad U, Vreeken J. SPACETIME: causal discovery from non-stationary time series. Proc AAAI Conf Artif Intell 2025;39:19405–19413. doi:10.1609/aaai.v39i18.34136.
- Marek K, Chowdhury S, Siderowf A, et al. The Parkinson's Progression Markers Initiative (PPMI): establishing a PD biomarker cohort. Ann Clin Transl Neurol 2018;5:1460–1477. doi:10.1002/acn3.644.
- Meng XL, Rubin DB. Performing likelihood ratio tests with multiply-imputed data sets. Biometrika 1992;79(1):103–111. doi:10.1093/biomet/79.1.103.
- Mestre TA, Fereshtehnejad SM, Berg D, et al. Parkinson's disease subtypes: critical appraisal and recommendations. J Parkinsons Dis 2021;11:395–404. doi:10.3233/JPD-202472.
- Mooij JM, Magliacane S, Claassen T. Joint causal inference from multiple contexts. J Mach Learn Res 2020;21(99):1–108.
- Moons KGM, Damen JAA, Kaul T, et al. PROBAST+AI: an updated quality, risk of bias, and applicability assessment tool for prediction models using regression or artificial intelligence methods. BMJ 2025;388:e082505. doi:10.1136/bmj-2024-082505.
- Morris TP, White IR, Crowther MJ. Using simulation studies to evaluate statistical methods. Stat Med 2019;38:2074–2102. doi:10.1002/sim.8086.
- Mulder JD, Hamaker EL. Three extensions of the random intercept cross-lagged panel model. Struct Equ Modeling 2021;28:638–648. doi:10.1080/10705511.2020.1784738.
- Negida A, Mukhopadhyay N, Berman BD, Barrett MJ. Comparative analysis of progression milestones across Parkinson's disease clinical, pathological, and data-driven subtypes: a 10-year follow-up. NPJ Parkinsons Dis 2026;12:239. doi:10.1038/s41531-026-01430-8.
- NICE. Parkinson's disease in adults. NICE guideline NG71. London: National Institute for Health and Care Excellence; 2017.
- Nylund KL, Asparouhov T, Muthén BO. Deciding on the number of classes in latent class analysis and growth mixture modeling: a Monte Carlo simulation study. Struct Equ Modeling 2007;14:535–569. doi:10.1080/10705510701575396.
- Ollier E, Viallon V. Regression modelling on stratified data with the lasso. Biometrika 2017. doi:10.1093/biomet/asw065.
- Orth U, Meier LL, Bühler JL, et al. Effect size guidelines for cross-lagged effects. Psychol Methods 2024;29:421–433. doi:10.1037/met0000499.
- Pal S, Ghosh D, Yang S. Penalized FCI for causal structure learning in a sparse DAG for biomarker discovery in Parkinson's disease. Ann Appl Stat 2026;20:1873–1893. doi:10.1214/26-AOAS2191.
- Pamfil R, Sriwattanaworachai N, Desai S, et al. DYNOTEARS: structure learning from time-series data. Proc AISTATS 2020, PMLR 108:1595–1605.
- Pan W, Su C, Maasch JRMA, Chen K, Henchcliffe C, Wang F. Learning phenotypic associations for Parkinson's disease with longitudinal clinical records. AMIA Jt Summits Transl Sci Proc 2024;2024:374–383.
- Perry R, von Kügelgen J, Schölkopf B. Causal discovery in heterogeneous environments under the sparse mechanism shift hypothesis. Adv Neural Inf Process Syst 2022;35:10904–10917. doi:10.52202/068431-0792.
- Plis S, Danks D, Freeman C, Calhoun V. Rate-agnostic (causal) structure learning. Adv Neural Inf Process Syst 2015;28:3303–3311.
- Popescu O-I, Günther W, Rabel M, Runge J. Causal discovery for time series with endogenous context variables. ICML 2026, PMLR 306:99138–99180.
- Poulet PE, Durrleman S. Multivariate disease progression modeling with longitudinal ordinal data. Stat Med 2023;42(18):3164–3183. doi:10.1002/sim.9770.
- Pringsheim T, Day GS, Smith DB, et al. Dopaminergic therapy for motor symptoms in early Parkinson disease practice guideline summary. Neurology 2021;97(20):942–957. doi:10.1212/WNL.0000000000012868.
- Pyatigorskaya N, Yahia-Cherif L, Valabregue R, et al. Parkinson disease propagation using MRI biomarkers and partial least squares path modeling. Neurology 2021;96:e460–e471. doi:10.1212/WNL.0000000000011155.
- Raj A, LoCastro E, Kuceyeski A, Tosun D, Relkin N, Weiner M. Network diffusion model of progression predicts longitudinal patterns of atrophy and metabolism in Alzheimer's disease. Cell Rep 2015;10(3):359–369. doi:10.1016/j.celrep.2014.12.034.
- Raket LL. Progression models for repeated measures: estimating novel treatment effects in progressive diseases. Stat Med 2022;41(28):5537–5557. doi:10.1002/sim.9581.
- Ramsey J, Andrews B, Spirtes P. Efficient latent variable causal discovery: combining score search and targeted testing. arXiv:2510.04263 (2025).
- Reisach AG, Seiler C, Weichwald S. Beware of the simulated DAG! Causal discovery benchmarks may be easy to game. Adv Neural Inf Process Syst 2021;34:27772–27784.
- Reisach AG, Tami M, Seiler C, Chambaz A, Weichwald S. A scale-invariant sorting criterion to find a causal order in additive noise models. Adv Neural Inf Process Syst 2023;36:785–807. doi:10.52202/075280-0036.
- Riley RD, Ensor J, Snell KIE, et al. Calculating the sample size required for developing a clinical prediction model. BMJ 2020;368:m441. doi:10.1136/bmj.m441.
- Riley RD, Snell KIE, Archer L, et al. Evaluation of clinical prediction models (part 3): calculating the sample size required for an external validation study. BMJ 2024;384:e074821. doi:10.1136/bmj-2023-074821.
- Rocchi N, Zanga A, Bernasconi A, et al. A causal discovery workflow for rare diseases: experts-in-the-loop analysis of sparse longitudinal data. J Med Syst 2026;50:9. doi:10.1007/s10916-025-02327-4.
- Rouanet A, Philipps V, Taddé BO, Pérès K, Proust-Lima C. Modeling temporal relationships between multivariate repeated markers along with clinical endpoints: application to Alzheimer's disease and related dementias. Biom J 2026;68:e70157. doi:10.1002/bimj.70157.
- Ruer A, Loranchet T, Bystrova D, Assaad CK. Root cause analysis via difference graph discovery from linear time-series data. arXiv:2608.21117 (2026).
- Runge J. Discovering contemporaneous and lagged causal relations in autocorrelated nonlinear time series datasets. UAI 2020, PMLR 124:1388–1397.
- Saadawy M, Saadawy E. NeuroCausal prior-weighted structure learning (NCP-WSL): disentangling a shared latent causal manifold across Alzheimer's, Parkinson's and healthy aging. SSRN preprint 2026. doi:10.2139/ssrn.7119099.
- Saggioro E, de Wiljes J, Kretschmer M, Runge J. Reconstructing regime-dependent causal relationships from observational time series. Chaos 2020;30:113115. doi:10.1063/5.0020538.
- Schiratti J-B, Allassonnière S, Colliot O, Durrleman S. A Bayesian mixed-effects model to learn trajectories of changes from repeated manifold-valued observations. J Mach Learn Res 2017;18(133):1–33.
- Schultzberg M, Muthén B. Number of subjects and time points needed for multilevel time-series analysis: a simulation study of dynamic structural equation modeling. Struct Equ Modeling 2018;25:495–515. doi:10.1080/10705511.2017.1392862.
- Schuurman NK, Hamaker EL. Measurement error and person-specific reliability in multilevel autoregressive modeling. Psychol Methods 2019;24:70–91. doi:10.1037/met0000188.
- Seo G, Shim J, Kim M, Byun H, Jung Y, Song K. Dissecting causal mechanism shifts via FANS: function and noise separation. ICML 2026, PMLR 306:109074–109104.
- Seppi K, Ray Chaudhuri K, Coelho M, et al. Update on treatments for nonmotor symptoms of Parkinson's disease—an evidence-based medicine review. Mov Disord 2019;34(2):180–198. doi:10.1002/mds.27602.
- Shah RD, Samworth RJ. Variable selection with error control: another look at stability selection. J R Stat Soc Series B Stat Methodol 2013;75:55–80. doi:10.1111/j.1467-9868.2011.01034.x.
- Siderowf A, Concha-Marambio L, Lafontant DE, et al. Assessment of heterogeneity among participants in the Parkinson's Progression Markers Initiative cohort using α-synuclein seed amplification: a cross-sectional study. Lancet Neurol 2023;22:407–417. doi:10.1016/S1474-4422(23)00109-6.
- Simuni T, Caspell-Garcia C, Coffey C, Lasch S, Tanner C, Marek K; PPMI Investigators. How stable are Parkinson's disease subtypes in de novo patients: analysis of the PPMI cohort? Parkinsonism Relat Disord 2016;28:62–67. doi:10.1016/j.parkreldis.2016.04.027.
- Simuni T, Chahine LM, Poston K, et al. A biological definition of neuronal α-synuclein disease: towards an integrated staging system for research. Lancet Neurol 2024;23:178–190. doi:10.1016/S1474-4422(23)00405-2.
- Spirtes P, Glymour C, Scheines R. Causation, prediction, and search. 2nd ed. Cambridge (MA): MIT Press; 2000. doi:10.7551/mitpress/1754.001.0001.
- Stebbins GT, Goetz CG, Burn DJ, Jankovic J, Khoo TK, Tilley BC. How to identify tremor dominant and postural instability/gait difficulty groups with the movement disorder society unified Parkinson's disease rating scale: comparison with the unified Parkinson's disease rating scale. Mov Disord 2013;28:668–670. doi:10.1002/mds.25383.
- Stein G, Penzel N, Piater T, Denzler J. TCD-Arena: assessing robustness of time series causal discovery methods against assumption violations. ICLR 2026. arXiv:2605.03045.
- Swait J, Louviere J. The role of the scale parameter in the estimation and comparison of multinomial logit models. J Mark Res 1993;30:305–314. doi:10.1177/002224379303000303.
- Tan MMX, Lawton MA, Pollard MI, et al. Genome-wide determinants of mortality and motor progression in Parkinson's disease. NPJ Parkinsons Dis 2024;10:113. doi:10.1038/s41531-024-00729-8.
- van Borkulo CD, van Bork R, Boschloo L, et al. Comparing network structures on three aspects: a permutation test. Psychol Methods 2023;28:1273–1285. doi:10.1037/met0000476.
- Van Loan CF. Computing integrals involving the matrix exponential. IEEE Trans Automat Contr 1978;23(3):395–404. doi:10.1109/TAC.1978.1101743.
- Varambally S, Ma YA, Yu R. Discovering mixtures of structural causal models from time series data. ICML 2024, PMLR 235:49171–49202.
- Vasey B, Nagendran M, Campbell B, et al. Reporting guideline for the early-stage clinical evaluation of decision support systems driven by artificial intelligence: DECIDE-AI. Nat Med 2022;28:924–933. doi:10.1038/s41591-022-01772-9.
- Vickers AJ, Elkin EB. Decision curve analysis: a novel method for evaluating prediction models. Med Decis Making 2006;26:565–574. doi:10.1177/0272989X06295361.
- Voelkle MC, Oud JHL, Davidov E, Schmidt P. An SEM approach to continuous time modeling of panel data: relating authoritarianism and anomia. Psychol Methods 2012;17:176–192. doi:10.1037/a0027543.
- von Elm E, Altman DG, Egger M, Pocock SJ, Gøtzsche PC, Vandenbroucke JP. The Strengthening the Reporting of Observational Studies in Epidemiology (STROBE) statement: guidelines for reporting observational studies. Lancet 2007;370(9596):1453–1457. doi:10.1016/S0140-6736(07)61602-X.
- Vu TC, Nutt JG, Holford NHG. Progression of motor and nonmotor features of Parkinson's disease and their response to treatment. Br J Clin Pharmacol 2012;74:267–283. doi:10.1111/j.1365-2125.2012.04192.x.
- Wang X, Zhao Y, Han K, et al. Conditional neural ODE for longitudinal Parkinson's disease progression forecasting. Proc IEEE BIBM 2025. doi:10.1109/BIBM66473.2025.11356813.
- Wang Y, Squires C, Belyaeva A, Uhler C. Direct estimation of differences in causal graphs. Adv Neural Inf Process Syst 2018;31. arXiv:1802.05631.
- Westfall PH, Young SS. Resampling-based multiple testing: examples and methods for p-value adjustment. New York: Wiley; 1993.
- Windmeijer F, Farbmacher H, Davies N, Davey Smith G. On the use of the lasso for instrumental variables estimation with some invalid instruments. J Am Stat Assoc 2019;114:1339–1350. doi:10.1080/01621459.2018.1498346.
- Witte J, Foraita R, Didelez V. Multiple imputation and test-wise deletion for causal discovery with incomplete cohort data. Stat Med 2022;41:4716–4743. doi:10.1002/sim.9535.
- Wooldridge JM. Simple solutions to the initial conditions problem in dynamic, nonlinear panel data models with unobserved heterogeneity. J Appl Econom 2005;20(1):39–54. doi:10.1002/jae.770.
- Xu Y, Düker M-C, Matteson DS. Testing simultaneous diagonalizability. J Am Stat Assoc 2024;119:1513–1525. doi:10.1080/01621459.2023.2202435.
- Young AL, Marinescu RV, Oxtoby NP, et al. Uncovering the heterogeneity and temporal complexity of neurodegenerative diseases with Subtype and Stage Inference. Nat Commun 2018;9:4273. doi:10.1038/s41467-018-05892-0.
- Zhang J. On the completeness of orientation rules for causal discovery in the presence of latent confounders and selection bias. Artif Intell 2008;172:1873–1896. doi:10.1016/j.artint.2008.08.001.
- Zheng X, Aragam B, Ravikumar P, Xing EP. DAGs with NO TEARS: continuous optimization for structure learning. Adv Neural Inf Process Syst 2018;31.

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

**Eligibility.** A6, A7 and A12 concern treatment response, which is not a drift link; they are tested under H-T (§2.4), not within H-C. A10 needs RBD symptoms as a node, so it is eligible only if RBD status is not a phenotype axis (D18).

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
- **Cost** (Apple M4, 16 GB; one synthetic dataset of the pooled size: 1,741 participants, 43 columns, 7,220 two-slice rows):
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

## Appendix D. Changes from v0.3 and v0.4

**v0.4.1 (3 November 2026): changes from v0.4,** which is archived as `docs/archive/PROTOCOL_v0.4.md`. Two additions requested by the investigator, then the fixes from three independent reviews (internal consistency; methods and statistics; references).
- **Additions.**
  - AI disease-progression models join the prior work (§1.1, §1.4): disease course mapping, a latent-time model in PPMI, and Subtype and Stage Inference.
  - Comparator C12, DYNOTEARS, re-implemented from the authors' code and validated against the paper (§7.2, §8.4; Appendix E, check 6).
- **Engine (§2.3, §6.1, §7.1).**
  - A participant load in every model, integrated out exactly given the speed.
  - Random effects conditioned on pre-specified characteristics, including the year-0 and year-1 states; era acts on speed, load and noise; a speed spread for each phenotype.
  - Treatment: an Emax effect of dose with an examination-state indicator, allowed to differ by phenotype. H-T tests the response differences (A6, A7 and A12).
  - One frozen normal-scores transform with mid-ranks, and latent thresholds for floor- or ceiling-bound domains (D26).
  - Missing values are handled exactly in the likelihood; multiple imputation becomes a sensitivity analysis.
- **Identification (§7.1).** The condition is corrected: the deviations in each direction must carry less than half of the weight. Minimal-L1 is stated as an assumption, with pre-specified diagnostics and a speed interval.
- **Null and tests (§7.3, §7.4).**
  - One nuisance-matched generator for the extended null, with measurement error, curvature, treatment initiation and trajectory-dependent dropout. The support is re-selected in every null dataset.
  - H-P uses up to 1,000 null datasets with sequential stopping, and a fallback p-value if the nuisance check fails.
  - Per-link p-values are recalibrated against the null where needed; separate graphs are judged by a family-wise maximum statistic.
  - H-L chooses K by BIC in every replicate, with multiple starts and up to 100 replicates.
- **Verdicts and multiplicity (§2.3, §2.4, §7.6, §7.7, §9).**
  - "Speed and load only" needs every link within ±ε; otherwise the verdict is "no mechanism difference detected".
  - H-C is gated on its own axis's H-P, controls the false-discovery rate, and has its family size fixed before registration. "Similar" claims form their own Holm family.
  - The link class "stronger beyond speed" is renamed "different strength beyond speed"; unstable findings are graded "unstable".
- **Benchmark (§8).**
  - H-M1 is split into calibration (the test, with Bradley's 7.5% criterion on the probability of any false claim) and a descriptive comparator benchmark.
  - New scenarios S1S (speed spread) and S1T (treatment response); stresses for era, speed-dependent dropout, the initial state and a misspecified support; mechanisms outside the engine's class.
  - Calibration against the extended null by the warp-speed method; H-L's power gate from 200 S6 and 50 S5 datasets.
  - The compute is re-estimated, with formulas, at about 352 hours (15 days), above the 10-day budget (D14).
- **Governance.** A numeric power target for WP0 (§3); registration in two stages; a target of five neurologists on the panel. D25 and D26 are added; D10, D11, D14, D22, D23 and D24 are revised.
- **Novelty (Summary, §1.4).** The claims follow the addendum's wording ("we found no …"), with the closest precedents cited: time-scale tests in other fields, subject-level rates and per-patient speeds in disease models, SigClust, and simulated nulls within strata.
- **Corrections.** Claims attributed to Aleksovski 2018, Fereshtehnejad 2015 and Horváth 2015 now match the sources. Part II's prognostic models are renamed PM0 to PM4. Cross-references, abbreviations and Appendix E's scenario labels are fixed. The reference list is corrected, completed with DOIs and extended to 142 entries.

**v0.4: changes from v0.3.** The changes from the two v0.2 drafts to v0.3 are listed in v0.3's Appendix D (`docs/archive/PROTOCOL_v0.3.md`).

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
- **Part II.** Prognostic model C3 (PM3 since v0.4.1) becomes the shared map at each patient's estimated speed. II-a replicates speed and mechanism verdicts, using the engine across visit intervals.
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
  - with 950 × 2: 12% false rejections, 21% power, adjusted Rand index 0.34 and 0.47;
  - simplifications: a χ² reference with 35 degrees of freedom, K fixed at 2, and no random speed within classes.

**Check 3. Speed varying within phenotypes** (`toy_random_speed.py`, `toy_random_speed_scale.py`, `toy_random_speed_scale_person.py`).
- *Setting:* three domains; 400 TD and 200 PIGD patients; log-speed spread 0.4; cohorts 1, 4 and 16 times larger.
- *Results:*
  - the group-level test rejected 10.5% of speed-only cohorts at the base size, rising to 27–65% at 16 times;
  - the participant-level test rejected 4.6% of 1,000 cohorts across all sizes, with at least as much power.

**Check 4. Method families against the speed-adjusted procedures** (`toy_comparators.py`, `toy_comparators_extra.py`, `toy_comparators_joint.py`).
- *Setting:*
  - six domains; 800 TD and 250 PIGD patients with five transitions each; speed spread 0.4; a load difference; no participant load in the analysis model;
  - four scenarios, labelled as in the scripts: equal speed with a load difference (S0, like §8.2's S0L; 200 cohorts), PIGD twice as fast (S1; 200), plus a PIGD-only link (S2, like §8.2's S3; 100), plus a stronger link (S3, like §8.2's S2; 100).
- *Engine:* the likelihood matched an independent implementation to machine precision. The speed-only and free fits, the one-at-a-time score tests and 12 penalised fits took about 30 seconds per cohort.
- *Speed only:*
  - yearly methods flagged 55–98% of shared links, in 100% of cohorts;
  - naive continuous time flagged 63%;
  - joint discovery flagged 0.4%, and the joint per-link tests 0.25–1.0%;
  - the per-domain tests flagged 0.6–1.3% of unchanged domains;
  - the global test rejected 6.5% of cohorts, and the shared-eigenvector check 3.5–4.0%.
- *One real difference:*
  - yearly methods isolated it in 0–1% of cohorts;
  - joint discovery isolated it in 96–97%, and the joint per-link tests in 89–94%;
  - the per-domain tests named the changed domain alone in 93–96%;
  - the shared-eigenvector check rejected in 28% (new link) and 78% (stronger link);
  - one-at-a-time score tests isolated it in only 35% and 4%.
- *Equal speed with a load difference:*
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

**Check 6. DYNOTEARS as published** (`dynotears.py`, `toy_dynotears_validation.py`, `toy_dynotears.py`).
- *Re-implementation:* written from the authors' causalnex code, with the same objective, gradient, bounds and optimisation schedule. The analytic gradient matched central differences to a relative error of 10⁻⁹.
- *Validation:* on the paper's own benchmark (Gaussian noise, Erdős–Rényi graphs, one lag, 5 to 50 variables, n = 50 and 500; 20 datasets per setting against the paper's 5), the F1 scores for same-time and lagged edges differed from the published values by 0.02 on average (largest 0.07). Every fit converged.
- *Setting:* the cohorts of check 4. One graph per phenotype on scores standardised over the whole cohort, at the paper's settings for large samples (λ = 0.05; thresholds 0.3 and 0.1); the graphs are compared link by link.
- *Equal speed with a load difference:* discordant lagged links in 42.5% of cohorts.
- *Speed only:* discordant lagged links in 99.5% of cohorts, on 32% of the shared links, and discordant same-time edges in 44.5%. The TD graph usually had no cross-links (median 0, against 3 in PIGD): the slower phenotype's yearly effects fall below the threshold.
- *One real difference:* flagged in 25% of cohorts for a new link and 84% for a stronger link, but never isolated: false flags came with it in every cohort.
- *At the package defaults* (λ = 0.1, no threshold): PIGD's same-time graph grew to a median of 18 edges, against 4 in TD.

**Limits.**
- The checks are linear and Gaussian, with no hidden burden, dropout, ordinal scores or definition artefacts.
- In check 4 the comparators are analogues written from each family's published definition. In checks 5 and 6 the algorithms follow the authors' code.
- WP3 repeats every comparison on the PPMI-fitted simulator, with the authors' software where it exists (C3 and C11 use this study's code; C5 and C12 the validated re-implementations).
- The revised engine and null of v0.4.1 have not been run in these checks: the participant load, conditioning on the initial states, phenotype-specific speed spread and treatment response, and one nuisance-matched generator with measurement error, curvature, treatment initiation and trajectory-dependent dropout. WP3 validates them before the benchmark.

# Newer causal discovery methods and a second novelty check

> Addendum to the investigation report (`pathway_heterogeneity_framework.md`), 2 November 2026. It answers two questions from the investigator: which causal discovery frameworks newer than PCMCI, J-PCMCI+ and JCI suit this study, and whether causal discovery has already been done for different phenotypes. It rests on four literature searches, run in parallel. Together they made about 530 database queries (Europe PMC, OpenAlex, Crossref, arXiv, OpenReview, PMLR) and 44 web searches, screened about 15,000 records by title or abstract, and checked which later papers cite about 35 key works. Every reference below was checked against Crossref, arXiv or the PMLR proceedings pages, as marked in the reference list. No PPMI data were read.

## Summary

**Newer methods.** No newer framework replaces the recommended one, but several strengthen it.
- *Step 1, the shared map.* Tiered FCI now has its own theory: it is sound with tiered knowledge and hidden variables, and a tiered variant combines datasets whose variables only partly overlap (Bang & Didelez 2023, 2025). Faster latent-variable searches (Ramsey 2025), missing-data handling for tiered discovery (Witte 2022) and mixed-data tests under hidden confounding (Hahn 2026) fill practical gaps. S-FCI (Li 2023) is a cross-check.
- *Step 2, phenotype comparison.* Difference-graph learners (Malik 2024; Bystrova & Devijver 2026; Ruer 2026) and "baseline plus deviations" estimators (Jiang & Zhang 2023; Fisher 2022; Crawford 2026) provide machinery to borrow. Run unmodified, they also become named naive comparators for the methods test (H-M1).
- *Groups defined from graph variables.* These now have a formal theory: endogenous context variables (Günther 2024; Popescu 2026; Rabel 2024).
- *Continuous time.* A latent-process network built for short clinical panels (DynNet; Rouanet 2026, extending Taddé 2020) is a second engine for the continuous-time check.

**Why none fits off the shelf.** Every multi-group, difference, mixture and context method found compares mechanisms on the yearly scale or cross-sectionally.
- In yearly data, speed changes almost every coefficient. That is a dense shift, which methods built on sparse mechanism shifts misread.
- Equilibrium data cannot show speed at all.
- Only the continuous-time zero pattern is both estimable from panels and unaffected by speed. "Sparse deviations from a speed-scaled reference", the core of Step 2, is the continuous-time form of the sparse-mechanism-shift idea (§1.1).

**Novelty: true only in a narrower form.**
- Causal and temporal networks *have* been compared between patient groups: Alzheimer's stages, psychiatric trajectory classes, boys and girls in a birth cohort, and many molecular subtypes. In PD, directed models have been fitted separately by phenotype, but they are confirmatory and mostly cross-sectional or based on brain imaging. No study has used causal discovery on longitudinal clinical data to compare PD phenotypes.
- None of these separates speed and load from mechanism. Several define groups by speed or outcome, which builds in the trap of §2 of the report.
- Tests of "same process, different speed" exist in biology, phylogenetics, growth and econometrics. None exists for multivariate causal or lagged mechanisms in panel data.
- Selection corrections and simulated homogeneous nulls exist. None re-applies the group-definition rule inside a cohort-fitted null.
- The report's three novelty claims therefore stand only in the narrower wording of §2.5.

## 1. Newer methods, by step of the framework

### 1.1 Three facts that rule out off-the-shelf use

1. **Speed is a dense shift in yearly data.** Running the same mechanism s times faster changes nearly every yearly coefficient (B(fast) = B(slow)^s). Methods that assume few mechanisms change between groups would flag almost every link (sparse mechanism shift: Perry 2022; difference graphs: Malik 2024, Bystrova & Devijver 2026). Methods that read patterns of dependent shifts as hidden confounding or selection would misread speed as confounding.
2. **Equilibrium data are blind to speed.** Replacing the drift A and the noise C by sA and sC leaves the stationary covariance unchanged, because AΣ + ΣAᵀ + C = 0 still holds. Cross-sectional methods and graphical continuous Lyapunov models therefore cannot see speed.
3. **The continuous-time zero pattern is speed-free.** In continuous time, speed multiplies the whole drift matrix, so which links exist does not change with speed. A structural difference found there cannot be a speed artefact. A strength difference is judged against the speed-scaled reference: A(PIGD) − ŝ·A(TD).

### 1.2 Shortlist

| Step | Method | What it adds | Fit for PPMI | Proposed use |
|---|---|---|---|---|
| 1 | Tiered FCI and tiered IOD (Bang & Didelez 2023, 2025) | Shows that FCI with tiered background knowledge is sound with hidden variables. The IOD variant combines datasets whose variable sets only partly overlap | Good | Theoretical basis of Step 1. Tiered IOD if PPMI 1.0 and 2.0 do not share every variable |
| 1 | Multiple imputation and test-wise deletion for tiered PC (Witte 2022; R packages tpc and micd) | Missing data in cohort causal discovery | Good | Missing-data handling in Step 1 |
| 1 | BOSS-FCI, GRaSP-FCI and FCI with targeted testing (Ramsey 2025; Tetrad) | Faster latent-variable searches that accept tiers | Possible to good | Sensitivity learner. Check memory first: the pilot's Tetrad latent learners ran out of memory |
| 1 | fedCI (Hahn 2026) | Likelihood-ratio independence tests for mixed and ordinal data, with site effects, under hidden confounding | Possible to good | Candidate test for WP3, alongside a test robust to unequal variance: random person speeds make pooled residual variance depend on the current state |
| 1 | S-FCI (Li, Jaber & Bareinboim 2023) | Multi-domain FCI with hidden confounders | Possible to good | Cross-check of JCI-FCI with phenotype or cohort as the domain. Its domain edges mean "differs", never "different mechanism" |
| 1–2 | Endogenous context variables (Günther 2024; Popescu 2026 for time series; Rabel 2024) | Formal theory for context variables caused by system variables, such as phenotype labels computed from symptoms, and for apparent mechanism changes that come from where the data have support | Conceptual | Formal basis for the landmark design and the definition-matched null. Check whether its algorithms add to Step 1 |
| 2 | Difference graphs: DDAG (Malik 2024), LDiffPC (Bystrova & Devijver 2026), difference graphs for linear time series (Ruer 2026); reasoning with difference graphs (Assaad 2025) | Learn where two groups' causal models differ, from tests of coefficient equality | Possible | Re-target the equality tests to the speed-scaled reference. Use unmodified versions as naive comparators in H-M1 |
| 2 | NetANOVA (Jiang & Zhang 2023) | A baseline network plus group differences, with bootstrap false-discovery control | Possible to poor: it needs an instrument for each node | Precedent for the joint parameterisation |
| 2 | multi-VAR and penalised subgrouping (Fisher 2022; Crawford 2026) | Common effects plus person or subgroup deviations, penalised, in time series | Possible | Nearest precedent for speed-adjusted joint discovery. It shrinks towards equality, not towards a speed-scaled reference. Naive comparator in H-M1 |
| 2 | FANS (Seo 2026) | Splits a mechanism shift into a change in the function and a change in the noise | Conceptual | Template for splitting a difference into speed, links and load |
| 2 | Sparse mechanism shift (Perry 2022) | Discovery across environments assuming few mechanisms change | Poor (speed is a dense shift) | Naive comparator in H-M1 |
| 3 | Mixtures of DAGs, covariate-dependent mixtures (Marchant 2025; Björkman 2026; Castelletti & Ferrini 2026) | Latent classes with their own graphs | Poor: rows assumed independent, no speed classes | Comparators only |
| 3 | Continuous-time subgrouping (Park 2025; Park 2026, preprint) | Subgroups that do not depend on the measurement interval | Poor: needs dense series, and would still split groups that differ only in speed | Cite as a foil |
| 4 | Structure learning on clustered data (Thompson, Wand & Baladandayuthapani 2026) | Random effects inside structure learning | Possible | Burden removal; no multiplicative person speed |
| CT check | DynNet (Rouanet 2026), extending CInLPN (Taddé 2020) | Latent-process network in fine time steps that approximates continuous time, for short panels with ordinal, binary and continuous markers, measurement error, and death or diagnosis | Good | Second continuous-time engine, closer to clinical data than ctsem. A person speed would need custom coding |
| CT check | Rate-agnostic discovery (Solovyeva 2023; Abavisani 2026, preprint) | One causal graph consistent with graphs sampled at several unknown whole-number rates | Poor: long series, whole-number rates | Cite for novelty claim 2 |
| CT check | Two-time-slice discovery from subsampled data (DHT-CIT; Wu 2024) | Orders variables from two cross-sections of a subsampled process | Possible | Cite; panel-shaped |

Not useful for PPMI:
- continuous-time neural learners (Rhino, CUTS/CUTS+, SCOTCH and similar), which need long, dense series;
- pretrained causal "foundation" models;
- graphical continuous Lyapunov models, which assume equilibrium and so cannot see speed;
- person-specific network methods.

## 2. Has causal discovery been done for different phenotypes?

### 2.1 Parkinson's disease

**No study was found that uses data-driven causal discovery on longitudinal clinical data to compare the directed relations between clinical domains across PD phenotypes, classes or individuals.** Directed analyses by phenotype do exist. The nearest are listed below.

| Study | Groups | Data and method | Why it is not this study |
|---|---|---|---|
| Johnson 2016 | TD vs PIGD | Two waves, 114 patients. A multiple-group structural equation model: postural severity predicted later working memory in PIGD only | Confirmatory, with one outcome and no baseline working memory. PIGD patients had shorter disease duration (49 vs 85 months), so stage and speed are confounded |
| Pyatigorskaya 2021 | PD with vs without RBD | Cross-sectional MRI, 54 patients. Two pre-set propagation directions fitted by partial least squares path modelling: brainstem-to-cortex fitted with RBD, cortex-to-brainstem without | Imaging and cross-sectional. Two hypotheses compared, not discovered. Load differences not separated. The nearest work on the RBD axis |
| Fereshtehnejad 2015; Chien 2026 | Onset age, progression rate, TD vs non-TD; TD vs PIGD | Cross-sectional structural equation and mediation models fitted per phenotype | Cross-sectional and confirmatory |
| Florin 2016; Zeng 2019; Shan 2025; Zhang 2025 | Akinetic-rigid vs TD; TD vs PIGD; RBD status; TD only | Granger-type directed brain connectivity from local field potentials or MRI. Zhang 2025 uses disease duration as a pseudo-time | Brain signals, not clinical progression; mostly cross-sectional |
| Zhou 2021; Semchin 2026, preprint | Latent subtypes | A mixture of linear dynamical systems on DaTscan images, and a connectome-constrained progression model with a disease time for each patient (both PPMI) | Imaging. Zhou's subtypes differ in both speed and trajectory, with no test separating them. Semchin fixes propagation to the connectome |
| Saadawy 2026, preprint | Alzheimer's vs PD vs healthy ageing | Bayesian DAG discovery on longitudinal ADNI, PPMI and OASIS-3 data. Hypothesis: the three share one causal graph and differ in initial conditions and progression rates | Compares diagnoses, not patients within PD. The abstract reports no test calibrated for speed, load or group definition. Not peer reviewed. The closest concept found anywhere |

Other causal discovery in PD covers the whole cohort:
- undirected PC-algorithm graphs per year in PPMI (Pan 2024);
- dynamic Bayesian networks of motor and cognitive outcomes (Hayete 2017);
- modular Bayesian networks for simulating patients (Gootjes-Dreesbach 2020);
- penalised FCI on cross-sectional biomarkers (Pal 2026).

Directed longitudinal PPMI studies never stratify by phenotype. These include cross-lagged and change-score models (Petkus 2019; Wang 2024) and those already in the report. No undirected symptom-network comparison of TD vs PIGD, RBD status, sex, genetic status or age at onset was found either.

Two corrections to the report:
- Al-Sharman 2026 is a directed, cross-sectional Bayesian network comparing multiple sclerosis with PD, not an undirected network.
- The DOI of Pal 2026 is 10.1214/26-AOAS2191.

### 2.2 Other diseases

Group-specific directed models of clinical course exist. All compare the groups' dynamics directly, so a faster group looks mechanistically different.

| Study | Groups | Method | Speed separated? | Calibrated group test? |
|---|---|---|---|---|
| Taddé 2020; Carles 2021 (Alzheimer's) | Clinical stage: normal, MCI, dementia | Latent-process dynamic model; stage-specific temporal influences | No | Wald tests only |
| Markova 2026; Krance 2019 (Alzheimer's) | Clinical status | Continuous-time or cross-lagged models fitted separately | No | No |
| Romano 2026 (Alzheimer's) | MCI converters vs non-converters | Cross-lagged panel networks | No. Conversion is itself a speed variable | No |
| Larsen 2026, preprint (birth cohort) | Boys vs girls | Temporal causal discovery over six life periods, with a formal test of graph differences (P = 0.079) | No (boys and girls differ in developmental tempo) | Yes; method not yet read |
| Ma 2023; Zandstra 2026 (psychiatry) | Trajectory classes; remitters vs non-remitters | Cross-lagged panel networks per group | No. Groups are defined by the rate of change of the modelled symptoms | No |
| Kenny 2026 (schizophrenia trials) | Randomised drug vs placebo arms | Panel graphical VAR per arm | No | Yes (permutation). Randomisation removes definition artefacts but not speed |
| Abavisani 2026, preprint (schizophrenia fMRI) | Patients vs controls | Rate-agnostic causal discovery | Treats sampling rate as a nuisance; no test of a shared time scale | No |
| Leão 2021; Gromicho 2022 (ALS; in the report) | Slow, average, fast progressors | Dynamic Bayesian networks per group | No. Groups are defined by speed | No |
| Rocchi 2026 (soft-tissue sarcoma) | Three disease mechanisms | Expert-in-the-loop causal discovery on sparse longitudinal data | Unknown | Unknown; full text not read |

Subtype-specific causal networks are routine in molecular data:
- AML refractory vs relapsed (Yajima 2015);
- heterogeneous reciprocal graphical models across cancers (Ni 2018);
- graphs varying with prognostic factors (Ni 2019);
- ER+ vs ER− breast cancer (Lovelace 2026).

The claim must therefore be limited to clinical progression.

### 2.3 Tests of a shared time scale (novelty claim 2)

"Same process, different speed" is a formally tested hypothesis in other fields:
- interventions that stretch the lifespan distribution of *C. elegans* by one factor (Stroustrup 2016);
- proportional evolutionary rate matrices on phylogenies, tested by likelihood ratio (mvMORPH; Clavel 2015);
- individual tempo on a shared growth curve (SITAR; Cole 2010);
- treatment as a slowing of disease progression (Raket 2022);
- multistate disease models in transformed time (Hubbard 2008);
- business-cycle time in economics (Stock 1987).

Closest to our setting:
- a network diffusion model in which a subject-level rate multiplies the dynamics on a fixed connectome (Raj 2015), and that rate varies with diagnosis;
- a neural ODE on PPMI imaging with a speed for each patient on one shared course (Wang 2025).

Neither learns or compares directed links between domains. Rate-agnostic causal discovery (Solovyeva 2023; Abavisani 2026) allows unknown sampling rates but does not test whether groups share one. No test was found of whether groups or persons share one multivariate causal mechanism up to a time-scale factor.

### 2.4 Handling groups defined from graph variables (novelty claim 3)

The problem is no longer handled only by critique:
- Haslbeck 2021 sets out when selecting on sum scores biases a network comparison and when it does not.
- Boot 2025 corrects Ising-model estimates for selection on a sum score.
- Endogenous-context theory treats graph-defined groups inside the discovery algorithm (Günther 2024).
- In depression, person-specific dynamics were compared within severity strata against a simulated null with homogeneous dynamics (Hoekstra 2026; Ebrahimi 2024).
- Floor effects can create apparent differences in dynamics between severity groups (Haqiqatkhah 2024, simulation).
- In clustering, SigClust re-applies the clustering rule to data simulated from one fitted model (Liu 2008). That is the closest precedent for re-applying a definition rule inside a null.

None of these re-applies a phenotype-definition rule inside a cohort-fitted null for comparing longitudinal causal networks with person-level speed and load.

### 2.5 Revised novelty claims

These replace the wording in the report's Summary and §7.

1. **Pathways beyond speed and load.** "We found no study that tests whether directed progression mechanisms differ between groups of patients within a disease while allowing groups and persons to differ in speed and load, against a null calibrated for those differences. Existing group-specific models of clinical course compare dynamics directly, often between groups defined by stage, outcome or speed (e.g. Taddé 2020; Gromicho 2022; Romano 2026; Larsen 2026). In PD, directed models fitted by phenotype are confirmatory: cross-sectional path or mediation models (Fereshtehnejad 2015; Chien 2026), a two-wave model without baseline adjustment (Johnson 2016), imaging path models by RBD status (Pyatigorskaya 2021) and brain-connectivity studies. A preprint proposes that Alzheimer's disease, PD and healthy ageing share one causal graph and differ in initial conditions and rates (Saadawy 2026). It compares diagnoses rather than patients within PD, and its abstract reports no such test."
2. **A shared time scale.** "We found no test of whether groups or persons share one multivariate causal or lagged mechanism up to a common time-scale factor on the drift or transition matrix. Time-scale-only hypotheses are tested for survival (Stroustrup 2016), trait evolution (Clavel 2015), growth curves (Cole 2010) and univariate progression (Raket 2022). Rate-agnostic causal discovery allows unknown sampling rates without testing a shared one (Solovyeva 2023; Abavisani 2026)." The decomposition adds time scale to load and mechanism; splitting shifts into function and noise already exists without time (Seo 2026).
3. **Definition artefacts.** "No test compares groups' longitudinal or causal networks against a cohort-fitted null that re-applies the group-definition rule to data with one shared mechanism and person-level speed and load." Cite Haslbeck 2021, Boot 2025, Günther 2024, Hoekstra 2026 and Haqiqatkhah 2024 as related work.

## 3. Proposed changes for protocol v0.4

- **Step 1.**
  - Rest the learner on tiered FCI theory (Bang & Didelez 2025).
  - Use tiered IOD if the two PPMI cohorts do not share every variable.
  - Handle missing data with tpc or micd (Witte 2022).
  - Add fedCI, plus a test robust to unequal variance, to the WP3 test comparison.
  - Cross-check with S-FCI.
  - Use BOSS-FCI or targeted testing as the sensitivity learner, if memory allows.
- **Step 2.** Keep speed-adjusted joint discovery. Borrow coefficient-equality tests from LDiffPC and re-target them to the speed-scaled reference. Cite NetANOVA and multi-VAR as precedents.
- **Methods test (H-M1).** Name the naive comparators, each expected to read speed as mechanism:
  - DDAG, LDiffPC and time-series difference graphs;
  - multi-VAR;
  - context edges from JCI or S-FCI;
  - sparse mechanism shift;
  - a mixture of DAGs.
  This turns the primary methodological claim into a concrete comparison with current methods.
- **Continuous-time check.** Add DynNet as a second engine.
- **Report §7.** Use the revised claims of §2.5 and add the closest prior work.

## 4. Limits of the search

- **Incomplete cited-by screens.** Screens for J-PCMCI+ (Günther 2023), MCD (Varambally 2024), Saeed 2020 and Strobl 2023 were not completed: the OpenAlex quota ran out and Semantic Scholar refused requests. Rerun them with an API key before any claim of priority.
- **Reading depth.** Most items were read at abstract level.
- **Full texts still to read:**
  - Larsen 2026, for its test of graph differences;
  - Rocchi 2026, for how its groups are defined;
  - Abavisani 2026, for its group comparison.
- **Proceedings coverage.** Machine-learning proceedings were reached through arXiv, PMLR pages and Crossref, which cover them less completely than journals.

## References

Each entry was checked on 2 November 2026 against the source marked in brackets: C = Crossref record, A = arXiv record, P = PMLR proceedings page, E = Europe PMC abstract.

- Abavisani M, Solovyeva K, Danks D, Pearlson G, Calhoun V, Plis S. Recovering directional brain networks under temporal undersampling: an application to schizophrenia. bioRxiv 2026. doi:10.64898/2026.08.09.743708 [C]
- Al-Sharman A, Khalil H, Malouche D, et al. Directional symptom dependencies in multiple sclerosis and Parkinson's disease: a comparative Bayesian network analysis. J Multidiscip Healthc 2026;19:1–19. doi:10.2147/JMDH.S607760 [E]
- Assaad CK. Causal reasoning in difference graphs. Proc CLeaR 2025, PMLR 275:15–30. arXiv:2411.01292 [A]
- Bang CW, Didelez V. Do we become wiser with time? On causal equivalence with tiered background knowledge. Proc UAI 2023, PMLR 216:119–129. arXiv:2306.01638 [A]
- Bang CW, Didelez V. Constraint-based causal discovery with tiered background knowledge and latent variables in single or overlapping datasets. Proc CLeaR 2025, PMLR 275:1116–1146. arXiv:2503.21526 [A]
- Björkman Z, Loría J, Wharrie S, Kaski S. Incorporating expert knowledge into Bayesian causal discovery of mixtures of directed acyclic graphs. Proc AISTATS 2026, PMLR 300:964–972. arXiv:2510.06735 [A]
- Boot J, de Ron J, Haslbeck JMB, Epskamp S. Correcting for selection bias after conditioning on a sum score in the Ising model. Behav Res Methods 2025;57(12). doi:10.3758/s13428-025-02820-1 [C]
- Bystrova D, Devijver E. Constraint-based difference graph discovery in a linear setting. arXiv:2606.13947 (2026) [A]
- Carles S, Taddé BO, Berr C, et al. Dynamic reciprocal relationships between cognitive and functional declines along the Alzheimer's disease continuum in the prospective COGICARE study. Alzheimers Res Ther 2021;13(1):148. doi:10.1186/s13195-021-00887-4 [C]
- Castelletti F, Ferrini L. Bayesian nonparametric mixtures of categorical directed graphs for personalized causal inference. Biostatistics 2026;27(1):kxag026. doi:10.1093/biostatistics/kxag026 [C]
- Chien CY, Tan CH, Hong WP, Chen KC, Yu RL. The role of motor subtypes in modulating neuropsychiatric and cognitive function in Parkinson's disease: a group-stratified mediation analysis. J Psychiatr Res 2026;192:352–359. doi:10.1016/j.jpsychires.2025.10.069 [C]
- Clavel J, Escarguel G, Merceron G. mvMORPH: an R package for fitting multivariate evolutionary models to morphometric data. Methods Ecol Evol 2015;6(11):1311–1319. doi:10.1111/2041-210X.12420 [C]
- Cole TJ, Donaldson MDC, Ben-Shlomo Y. SITAR—a useful instrument for growth curve analysis. Int J Epidemiol 2010;39(6):1558–1566. doi:10.1093/ije/dyq115 [C]
- Crawford CM, Park JJ, Chow S-M, Ernst AF, Pipiras V, Fisher ZF. Penalized subgrouping of heterogeneous time series. Multivariate Behav Res 2026;61(4):579–602. doi:10.1080/00273171.2026.2622120 [C]
- Ebrahimi OV, Borsboom D, Hoekstra RHA, et al. Towards precision in the diagnostic profiling of patients: leveraging symptom dynamics as a clinical characterisation dimension in the assessment of major depressive disorder. Br J Psychiatry 2024;224(5):157–163. doi:10.1192/bjp.2024.19 [C]
- Fereshtehnejad SM, Shafieesabet M, Farhadi F, et al. Heterogeneous determinants of quality of life in different phenotypes of Parkinson's disease. PLoS One 2015;10(9):e0137081. doi:10.1371/journal.pone.0137081 [C]
- Fisher ZF, Kim Y, Fredrickson BL, Pipiras V. Penalized estimation and forecasting of multiple subject intensive longitudinal data. Psychometrika 2022;87(2). doi:10.1007/s11336-021-09825-7 [C]
- Florin E, Pfeifer J, Visser-Vandewalle V, Schnitzler A, Timmermann L. Parkinson subtype-specific Granger-causal coupling and coherence frequency in the subthalamic area. Neuroscience 2016;332:170–180. doi:10.1016/j.neuroscience.2016.06.052 [C]
- Gootjes-Dreesbach L, Sood M, Sahay A, Hofmann-Apitius M, Fröhlich H. Variational autoencoder modular Bayesian networks for simulation of heterogeneous clinical study data. Front Big Data 2020;3:16. doi:10.3389/fdata.2020.00016 [C]
- Gromicho M, Leão T, Oliveira Santos M, et al. Dynamic Bayesian networks for stratification of disease progression in amyotrophic lateral sclerosis. Eur J Neurol 2022;29(8):2201–2210. doi:10.1111/ene.15357 (in the report)
- Günther W, Ninad U, Runge J. Causal discovery for time series from multiple datasets with latent contexts. Proc UAI 2023, PMLR 216:766–776. arXiv:2306.12896 (in the report)
- Günther W, Popescu O-I, Rabel M, Ninad U, Gerhardus A, Runge J. Causal discovery with endogenous context variables. Adv Neural Inf Process Syst 2024;37. arXiv:2412.04981 [A]
- Hahn M, Zajak A, Heider D, Ribeiro AH. Federated causal discovery across heterogeneous datasets under latent confounding. arXiv:2603.05149 (2026) [A]
- Haqiqatkhah MM, Ryan O, Hamaker EL. Skewness and staging: does the floor effect induce bias in multilevel AR(1) models? Multivariate Behav Res 2024;59(2):289–319. doi:10.1080/00273171.2023.2254769 [C]
- Haslbeck JMB, Ryan O, Dablander F. The sum of all fears: comparing networks based on symptom sum-scores. Psychol Methods 2021 (advance online). doi:10.1037/met0000418 [C]
- Hayete B, Wuest D, Laramie J, et al. A Bayesian mathematical model of motor and cognitive outcomes in Parkinson's disease. PLoS One 2017;12(6):e0178982. doi:10.1371/journal.pone.0178982 [C]
- Hoekstra RHA, Epskamp S, Nierenberg AA, Borsboom D, McNally RJ. Testing similarity in longitudinal networks: the Individual Network Invariance Test. Psychol Methods 2026;31(5):844–870. doi:10.1037/met0000638 [C]
- Hubbard RA, Inoue LYT, Fann JR. Modeling nonhomogeneous Markov processes via time transformation. Biometrics 2008;64(3):843–850. doi:10.1111/j.1541-0420.2007.00932.x [C]
- Jiang Z, Zhang D. Analysis of variance of multiple causal networks. Adv Neural Inf Process Syst 2023;36. doi:10.52202/075280-0510 [C]
- Johnson AR, Bucks RS, Kane RT, Thomas MG, Gasson N, Loftus AM. Motor subtype as a predictor of future working memory performance in idiopathic Parkinson's disease. PLoS One 2016;11(3):e0152534. doi:10.1371/journal.pone.0152534 [C]
- Kenny EA, Brand BA, Ebrahimi OV, McCutcheon RA. Dopamine antagonists and dynamic symptom networks in schizophrenia. Br J Psychiatry 2026. doi:10.1192/bjp.2026.10775 [C]
- Krance SH, Cogo-Moreira H, Rabin JS, et al. Reciprocal predictive relationships between amyloid and tau biomarkers in Alzheimer's disease progression: an empirical model. J Neurosci 2019;39(37):7428–7437. doi:10.1523/JNEUROSCI.1056-19.2019 [C]
- Larsen TE, Hernández-Lorca M, Ekstrøm CT, Vinding R, Bønnelykke K, Strandberg-Larsen K, Petersen AH. Mapping the dynamic interplay of mental health and weight across childhood: data-driven explorations using causal discovery. medRxiv 2026. doi:10.64898/2026.04.16.26350943 [C, E]
- Leão T, Madeira SC, Gromicho M, de Carvalho M, Carvalho AM. Learning dynamic Bayesian networks from time-dependent and time-independent data: unraveling disease progression in amyotrophic lateral sclerosis. J Biomed Inform 2021;117:103730. doi:10.1016/j.jbi.2021.103730 (in the report)
- Li A, Jaber A, Bareinboim E. Causal discovery from observational and interventional data across multiple environments. Adv Neural Inf Process Syst 2023;36. doi:10.52202/075280-0741 [C]
- Liu Y, Hayes DN, Nobel A, Marron JS. Statistical significance of clustering for high-dimension, low-sample size data. J Am Stat Assoc 2008;103(483):1281–1293. doi:10.1198/016214508000000454 [C]
- Lovelace TC, Benos PV. New causal discovery algorithm over censored variables identifies subtype-specific drivers of breast cancer progression. GigaScience 2026;15:giag060. doi:10.1093/gigascience/giag060 [C]
- Ma Z, Zhao J, Chen H, et al. Temporal network of depressive symptoms across college students with distinct depressive trajectories during the COVID-19 pandemic. Depress Anxiety 2023;2023:8469620. doi:10.1155/2023/8469620 [C]
- Malik V, Bello K, Ghoshal A, Honorio J. Identifying causal changes between linear structural equation models. Proc UAI 2024, PMLR 244:2383–2398 [P]
- Marchant R, Draca D, Francis G, et al. Covariate dependent mixture of Bayesian networks. arXiv:2501.05745 (2025) [A]
- Markova TZ, Heston MB, Berry AS, et al. Tracking the continuous time dynamics of tau and depressive symptoms across the Alzheimer's disease continuum. J Neurosci 2026;46(36):e2309252026. doi:10.1523/JNEUROSCI.2309-25.2026 [C]
- Ni Y, Müller P, Zhu Y, Ji Y. Heterogeneous reciprocal graphical models. Biometrics 2018;74(2):606–615. doi:10.1111/biom.12791 [C]
- Ni Y, Stingo FC, Baladandayuthapani V. Bayesian graphical regression. J Am Stat Assoc 2019;114(525):184–197. doi:10.1080/01621459.2017.1389739 [C]
- Pal S, Ghosh D, Yang S. Penalized FCI for causal structure learning in a sparse DAG for biomarker discovery in Parkinson's disease. Ann Appl Stat 2026;20(3):1873–1893. doi:10.1214/26-AOAS2191 [C]
- Pan W, Su C, Maasch JRMA, Chen K, Henchcliffe C, Wang F. Learning phenotypic associations for Parkinson's disease with longitudinal clinical records. AMIA Jt Summits Transl Sci Proc 2024;2024:374–383; preprint medRxiv 2020, doi:10.1101/2020.03.15.20036657 [C, preprint]
- Park JJ, Fisher ZF, Chow S-M, Molenaar PCM. Evaluating discrete time methods for subgrouping continuous processes. Multivariate Behav Res 2024;59(6):1240–1252. doi:10.1080/00273171.2023.2235685 [C]
- Park JJ, Fisher ZF, Hunter MD, et al. Unsupervised model construction in continuous-time. Struct Equ Modeling 2025;32(3):377–399. doi:10.1080/10705511.2024.2429544 [C]
- Park JJ, Mills NX, Riese H, Ebrahimi OV. Continuous-time subgrouping of heterogeneous time-series. PsyArXiv 2026. doi:10.31234/osf.io/za9b3_v1 [C]
- Perry R, von Kügelgen J, Schölkopf B. Causal discovery in heterogeneous environments under the sparse mechanism shift hypothesis. Adv Neural Inf Process Syst 2022;35:10904–10917. arXiv:2206.02013 [A]
- Petkus AJ, Filoteo JV, Schiehser DM, Gomez ME, Petzinger G. Worse cognitive performance predicts increased anxiety and depressive symptoms in patients with Parkinson's disease: a bidirectional analysis. Neuropsychology 2019;33(1):35–46. doi:10.1037/neu0000498 [C]
- Popescu O-I, Günther W, Rabel M, Runge J. Causal discovery for time series with endogenous context variables. Proc ICML 2026, PMLR 306 [P]
- Pyatigorskaya N, Yahia-Cherif L, Valabregue R, et al. Parkinson disease propagation using MRI biomarkers and partial least squares path modeling. Neurology 2021;96(3):e460–e471. doi:10.1212/WNL.0000000000011155 [C]
- Rabel M, Günther W, Runge J, Gerhardus A. Causal modeling in multi-context systems: distinguishing multiple context-specific causal graphs which account for observational support. arXiv:2410.20405 (2024) [A]
- Raj A, LoCastro E, Kuceyeski A, Tosun D, Relkin N, Weiner M. Network diffusion model of progression predicts longitudinal patterns of atrophy and metabolism in Alzheimer's disease. Cell Rep 2015;10(3):359–369. doi:10.1016/j.celrep.2014.12.034 [C]
- Raket LL. Progression models for repeated measures: estimating novel treatment effects in progressive diseases. Stat Med 2022;41(28):5537–5557. doi:10.1002/sim.9581 [C]
- Ramsey J, Andrews B, Spirtes P. Efficient latent variable causal discovery: combining score search and targeted testing. arXiv:2510.04263 (2025) [A]
- Rocchi N, Zanga A, Bernasconi A, et al. A causal discovery workflow for rare diseases: experts-in-the-loop analysis of sparse longitudinal data. J Med Syst 2026;50(1):9. doi:10.1007/s10916-025-02327-4 [C, E]
- Romano D, Tosi G; ADNI. Divergent network trajectories in mild cognitive impairment: converters versus non-converters to Alzheimer's disease. J Alzheimers Dis 2026;110(3):1414–1427. doi:10.1177/13872877261423962 [C]
- Rouanet A, Philipps V, Taddé BO, Pérès K, Proust-Lima C. Modeling temporal relationships between multivariate repeated markers along with clinical endpoints: application to Alzheimer's disease and related dementias. Biom J 2026;68(4):e70157. doi:10.1002/bimj.70157 [C]
- Ruer A, Loranchet T, Bystrova D, Assaad CK. Root cause analysis via difference graph discovery from linear time-series data. arXiv:2608.21117 (2026) [A]
- Saadawy M, Saadawy E. NeuroCausal prior-weighted structure learning (NCP-WSL): disentangling a shared latent causal manifold across Alzheimer's, Parkinson's and healthy aging. SSRN preprint 2026. doi:10.2139/ssrn.7119099 [C, abstract]
- Saeed B, Panigrahi S, Uhler C. Causal structure discovery from distributions arising from mixtures of DAGs. Proc ICML 2020, PMLR 119:8336–8345. arXiv:2001.11940 (in the report)
- Semchin D, d'Angremont E, Ding H, et al. Discovering subtypes of neurodegenerative progression with a scalable connectome-constrained dynamic model. arXiv:2609.10890 (2026) [A]
- Seo G, Shim J, Kim M, Byun H, Jung Y, Song K. Dissecting causal mechanism shifts via FANS: function and noise separation. Proc ICML 2026, PMLR 306 [P]
- Shan AD, Zhang H, Gao MX, et al. Altered effective connectivity in Parkinson's disease patients with rapid eye movement sleep behavior disorder. Quant Imaging Med Surg 2025;15(1):352–369. doi:10.21037/qims-24-1196 [C]
- Solovyeva K, Danks D, Abavisani M, Plis S. Causal learning through deliberate undersampling. Proc CLeaR 2023, PMLR 213:518–530 [P]
- Stock JH. Measuring business cycle time. J Polit Econ 1987;95(6):1240–1261. doi:10.1086/261513 [C]
- Strobl EV. Causal discovery with a mixture of DAGs. Mach Learn 2023;112(11):4201–4225. doi:10.1007/s10994-022-06159-y (in the report)
- Stroustrup N, Anthony WE, Nash ZM, et al. The temporal scaling of Caenorhabditis elegans ageing. Nature 2016;530(7588):103–107. doi:10.1038/nature16550 [C]
- Taddé BO, Jacqmin-Gadda H, Dartigues JF, Commenges D, Proust-Lima C. Dynamic modeling of multivariate dimensions and their temporal relationships using latent processes: application to Alzheimer's disease. Biometrics 2020;76(3):886–899. doi:10.1111/biom.13168 [C, E]
- Thompson R, Wand MP, Baladandayuthapani V. Structure learning on clustered data. arXiv:2607.08238 (2026) [A]
- Varambally S, Ma Y-A, Yu R. Discovering mixtures of structural causal models from time series data. Proc ICML 2024, PMLR 235:49171–49202. arXiv:2310.06312 (in the report)
- Wang Q, Zheng S, Jing B, et al. The activities of daily living partially mediate the relationship between rapid eye movement sleep behavior disorder and depressive symptoms in Parkinson's disease. Front Neurol 2024;15:1357721. doi:10.3389/fneur.2024.1357721 [C]
- Wang X, Zhao Y, Han K, et al. Conditional neural ODE for longitudinal Parkinson's disease progression forecasting. Proc IEEE BIBM 2025. doi:10.1109/BIBM66473.2025.11356813; arXiv:2511.04789 [C, A]
- Witte J, Foraita R, Didelez V. Multiple imputation and test-wise deletion for causal discovery with incomplete cohort data. Stat Med 2022;41(23):4716–4743. doi:10.1002/sim.9535 [C]
- Wu A, Li H, Kuang K, Zhang K, Wu F. Learning causal relations from subsampled time series with two time-slices. Proc ICML 2024, PMLR 235:53635–53655 [P]
- Yajima M, Telesca D, Ji Y, Müller P. Detecting differential patterns of interaction in molecular pathways. Biostatistics 2015;16(2):240–251. doi:10.1093/biostatistics/kxu054 [C]
- Zandstra MG, Scheepers FE, Lunansky G, et al. Symptom network dynamics during antipsychotic treatment in first-episode psychosis. Schizophr Bull 2026;52(2):sbag016. doi:10.1093/schbul/sbag016 [C]
- Zeng Q, Guan X, Guo T, et al. The ventral intermediate nucleus differently modulates subtype-related networks in Parkinson's disease. Front Neurosci 2019;13:202. doi:10.3389/fnins.2019.00202 [C]
- Zhang M, Zhou S, Yang P, et al. The precuneus region drives brain network changes in tremor-dominant Parkinson's disease: insights from a morphological causal analysis. MedComm 2025;6(11):e70441. doi:10.1002/mco2.70441 [C]
- Zhou Y, Tinaz S, Tagare HD. Robust Bayesian analysis of early-stage Parkinson's disease progression using DaTscan images. IEEE Trans Med Imaging 2021;40(2):549–561. doi:10.1109/TMI.2020.3031478 [C]

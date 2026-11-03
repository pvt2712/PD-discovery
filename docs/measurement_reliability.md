# Measurement reliability of the progression domains (decision D28)

Version 0.1, 3 November 2026. Based on the literature only; no PPMI data were accessed.

## Purpose

Decision D28 fixes each progression domain's measurement-error variance from published reliability instead of estimating it from PPMI. Here reliability (R) means the share of the between-person variance at a visit that is stable "true" difference rather than rater error, day-to-day fluctuation or other transient noise. On the standardised (normal-scores) scale, a domain with reliability R has true-score variance R and **error variance 1 − R**. For each domain this note gives:

- the best published test–retest reliability, ideally an intraclass correlation coefficient (ICC) in Parkinson's disease (PD) with a retest interval of days to weeks;
- a recommended value and a range for sensitivity analysis;
- the reasoning, and how values were derived where no study exists.

## How the values were chosen

1. **Published short-interval ICCs are treated as upper bounds.** They come from samples with a wider spread of severity than PPMI, retested 1–3 weeks apart under controlled conditions.
2. **PPMI-specific error for the MDS-UPDRS.** Evers et al. (2019) fitted a state-space model to PPMI itself (423 de novo patients, median follow-up 54 months). It splits the variance into three parts:
   - error plus short-term fluctuation (σ²E);
   - true-score variance at study entry (C0);
   - true change per year (σ²ΔT), which widens the true spread every year.

   Cross-sectional reliability at year *t* is then **R(t) = (C0 + t·σ²ΔT) / (C0 + t·σ²ΔT + σ²E)**. We computed R(t) from their Table 2 and supplementary Tables 7–8 (see the PPMI table below). Their headline figures (0.13–0.62) are the reliability of 1-year *change* scores, not of single visits.
3. **Discount for real-world conditions.** For MDS-UPDRS Parts II and III, the PPMI-derived values (baseline to year 2) are about 0.05–0.18 below the published short-interval ICCs. Questionnaires have no PPMI-specific estimate, so we discounted their best short-interval ICC by about the same amount.
4. **Target visit.** Recommended values target a typical PPMI follow-up visit (years 1–3). The low end of each range reflects baseline, where the true spread is smallest, or the weakest credible study.

## Recommended values

Legend: † value from an abstract or a secondary source (primary full text not checked); ‡ derived by us (see Notes). Error variance on the normal-scores scale = 1 − R.

| # | Domain | Instrument (PPMI scoring) | Best available coefficient (type) | Value [95% CI] | Population; retest interval | Recommended R (error variance) | Sensitivity range for R | Key references |
|---|---|---|---|---|---|---|---|---|
| 1 | Tremor | MDS-UPDRS III 3.15–3.18, sum (10 scores) | **None published for this subscore.** PPMI-derived reliability of tremor factors‡ | Rest tremor 0.68 → 0.84; postural/kinetic 0.52 → 0.63 (baseline → year 2) | 423 de novo PD (PPMI) | **0.70** (0.30) | 0.55–0.85 | Evers 2019; Siderowf 2002; Regnault 2025 |
| 2 | Bradykinesia and rigidity | III 3.3–3.8 + 3.14, sum (16 scores) | **None published.** PPMI-derived factors‡; internal consistency (Rasch person-separation index, PSI) | Composite 0.73 → 0.79‡; PSI 0.84 (15 limb items) | PPMI de novo PD; PSI from 1,906 untreated visits | **0.80** (0.20) | 0.70–0.90 | Evers 2019; Regnault 2025 |
| 3 | Axial and gait | III 3.9–3.13, sum (5 scores) | **None published.** PPMI-derived gait/posture factor‡ | 0.58 (baseline), 0.82 (year 1), 0.89 (year 2) | PPMI de novo PD | **0.75** (0.25) | 0.55–0.90 | Evers 2019 |
| 4 | Daily function | MDS-UPDRS II without 2.10, 2.12, 2.13 (10 items) | Test–retest ICC, full Part II (13 items) | 0.96† (abstract: > 0.85); PPMI-derived 0.78 → 0.88‡ | 435 Spanish-speaking PD, 1–2 weeks†; PPMI de novo | **0.80** (0.20) | 0.70–0.90 | Martinez-Martin 2013; Lavine 2024; Evers 2019 |
| 5 | Independence | Modified Schwab & England ADL | Test–retest ICC(3,1) | 0.70 (CI not reported) | 24 PD; within 2 weeks | **0.65** (0.35) | 0.50–0.80 | Dal Bello-Haas 2011 |
| 6 | Cognition | MoCA | Test–retest ICC | 0.79 (n, interval and CI not in abstract); 0.67 over 5–8 months | PD (Gill); 33 non-demented PD (Aiello) | **0.70** (0.30) | 0.60–0.80 | Gill 2008; Aiello 2024; Tran 2026 |
| 7 | Mood | GDS-15 | Test–retest ICC | GDS-15 in PD 0.92† (n and interval not in abstract); GDS-30 in PD 0.94 [0.89–0.97] | PD aged ≥ 65 (Persian version); 74 PD, 7 days (Italian GDS-30) | **0.80** (0.20) | 0.65–0.90 | Soltaninejad 2025; Massai 2018; Huang 2017; Nyunt 2009 |
| 8 | Autonomic | SCOPA-AUT total | Test–retest ICC | 0.87† (range 0.65–0.90, probably the domain ICCs) | 55 PD; interval not verified | **0.80** (0.20) | 0.65–0.90 | Visser 2004 (values via Evatt 2009, Matsushima 2014); Carod-Artal 2010 |
| 9 | Hallucinations | MDS-UPDRS item 1.2 | Item-level test–retest weighted kappa | ≥ 0.60† (true for all items except two item-3.17 scores) | 435 Spanish-speaking PD; 1–2 weeks† | **0.60** (0.40) | 0.40–0.75 | Martinez-Martin 2013; Evers 2019 |
| 10 | Daytime sleepiness | Epworth Sleepiness Scale | Test–retest Pearson r (**no PD study found**) | 0.82 | 87 healthy medical students; 5 months | **0.75** (0.25) | 0.65–0.85 | Johns 1992; Nguyen 2006; Hagell 2007 |
| 11 | RBD symptoms | RBDSQ | Test–retest ICC(3,1) | 0.86 [0.61–0.96] | 13 PD (Brazilian version); 21 days | **0.75** (0.25) | 0.55–0.90 | Pena-Pereira 2020; Röttgen 2025; Halsband 2018 |
| 12 | Motor complications | MDS-UPDRS IV | Test–retest ICC of part totals | > 0.85† (no separate Part IV value) | 435 Spanish-speaking PD; 1–2 weeks† | **0.70** (0.30) | 0.50–0.85 | Martinez-Martin 2013 |
| – | Anchor | MDS-UPDRS III total (33 scores) | Test–retest ICC | 0.93† (abstract: > 0.85); UPDRS motor 0.90; PPMI-derived OFF 0.79 → 0.89‡ | Spanish PD, 1–2 weeks†; 400 early untreated PD, 3–36 days | **0.85** (0.15) | 0.75–0.93 | Martinez-Martin 2013; Lavine 2024; Siderowf 2002; Metman 2004; Evers 2019 |

### Justification and strength of evidence

1. **Tremor (weak).** No direct test–retest study exists. In PPMI, rest tremor's reliability rises from 0.68 to 0.84 over two years, but postural and kinetic tremor is noisier (0.52–0.63). Tremor is also the most state-dependent motor sign, so 0.70 is a middle value.
2. **Bradykinesia and rigidity (moderate).** This is the largest and most internally consistent motor subscore (PSI 0.84 in untreated PPMI patients). Its PPMI-derived factor reliabilities are 0.65–0.86, a little below the whole of Part III.
3. **Axial and gait (weak).** Nearly everyone scores close to zero at baseline (reliability 0.58). The absolute error is small, though, and true differences widen fast (0.82–0.89 by years 1–2), so 0.75 is a compromise across follow-up.
4. **Daily function (moderate to strong).** Full Part II has a short-interval ICC of 0.96 and a PPMI-derived 0.78–0.88. Dropping 3 of 13 items lowers this by about 0.01–0.05 (Spearman–Brown).
5. **Independence (weak).** Only one small PD study exists (ICC 0.70), although the MDS Task Force rates the scale "recommended" for disability (Shulman 2016). In PPMI the scale is compressed near its ceiling (baseline mean 93.9%, SD 5.9; Evers 2019), which lowers reliability.
6. **Cognition (moderate).** PD test–retest ICCs range from 0.67 to 0.84. The lower value comes from non-demented PD over 6 months, which is closest to PPMI, where most participants score near the ceiling (baseline mean 27.1, SD 2.3; Evers 2019).
7. **Mood (moderate).** PD test–retest ICCs for the GDS are 0.89–0.94 at 1–2 weeks. The GDS-15 is half as long, however, and PPMI scores cluster near zero, so we discounted to 0.80.
8. **Autonomic (moderate).** The original PD validation gave a total ICC of 0.87, and independent versions gave 0.64–0.71. The total also mixes different organ systems: its Rasch reliability falls to 0.68 once the subscales are accounted for (Westergren 2022).
9. **Hallucinations (weak).** This is a single, rarely positive item, and the only evidence is that its test–retest kappa was at least 0.60. In PPMI the factor that carries it (items 1.1 and 1.2) has a reliability of only 0.33 at baseline, because hallucinations are rare.
10. **Daytime sleepiness (weak).** No PD test–retest data were found. Healthy adults show 0.82 over 5 months. In sleep-clinic patients, repeat scores about 2 months apart imply 0.72 (our calculation from Nguyen 2006).
11. **RBD symptoms (weak).** A very small PD study gives 0.86 at 3 weeks, and a mixed Chinese sample of RBD patients and controls gives 0.92 (Wang 2015). However, answers change with patients' awareness of RBD (ICC 0.54 before vs after diagnosis in isolated RBD), and the RBDSQ is not valid in de novo PD (Halsband 2018).
12. **Motor complications (weak).** The only evidence is a part-level threshold (ICC > 0.85). In PPMI almost all scores are zero in the early years, so reliability on a normal-scores scale is poorly defined (see Note 4).
13. **Part III total (strong).** Three PD test–retest studies give 0.90–0.93, and PPMI-derived values for OFF assessments are 0.79–0.89. The error SD is about 4 points across independent sources (Note 9).

## PPMI-derived reliabilities (supporting table)

Variance components are from Evers et al. 2019 (Table 2; supplementary Tables 7–8), in score units for the parts and factor-score units for the factors. R(t) is computed by us with the formula above. The factor scores are sums of all items in a part, weighted by their loadings from a principal-component analysis with varimax rotation. They therefore approximate, but do not equal, the protocol subscores.

| Measure | Main items | σ²E | C0 | σ²ΔT per year | R baseline | R year 1 | R year 2 | R year 4 |
|---|---|---|---|---|---|---|---|---|
| Part I | all | 4.87 | 11.82 | 4.22 | 0.71 | 0.77 | 0.81 | 0.85 |
| Part II | all | 3.66 | 12.85 | 7.22 | 0.78 | 0.85 | 0.88 | 0.92 |
| Part III, OFF | all | 15.52 | 57.44 | 31.13 | 0.79 | 0.85 | 0.89 | 0.92 |
| Part III, ON | all | 27.07 | 64.80 | 16.15 | n/a (no ON visits at baseline) | 0.75 | 0.78 | 0.83 |
| F3.4 rest tremor | 3.17, 3.18 | 0.19 | 0.41 | 0.28 | 0.68 | 0.78 | 0.84 | 0.89 |
| F3.7 postural/kinetic tremor | 3.15, 3.16 | 0.43 | 0.46 | 0.13 | 0.52 | 0.58 | 0.63 | 0.70 |
| F3.1 bradykinesia, left | 3.4–3.8 L | 0.14 | 0.76 | 0.10 | 0.84 | 0.86 | 0.87 | 0.89 |
| F3.2 bradykinesia, right | 3.4–3.8 R | 0.22 | 0.71 | 0.13 | 0.76 | 0.79 | 0.82 | 0.85 |
| F3.5 rigidity | 3.3 | 0.26 | 0.50 | 0.17 | 0.66 | 0.72 | 0.76 | 0.82 |
| F3.6 face, speech, global bradykinesia | 3.1, 3.2, 3.14 | 0.29 | 0.54 | 0.10 | 0.65 | 0.69 | 0.72 | 0.76 |
| F3.3 gait and posture | 3.9–3.13 | 0.11 | 0.15 | 0.35 | 0.58 | 0.82 | 0.89 | 0.93 |
| F1.3 cognition and hallucinations | 1.1, 1.2 | 0.30 | 0.15 | 0.26 | 0.33 | 0.58 | 0.69 | 0.80 |

These values assume the model's Gaussian random walk, so later-year values may be optimistic. Using Evers's change-score reliabilities (0.50 for Parts II and III) as 1 − error variance would overstate error variance two- to four-fold compared with these cross-sectional values.

## Notes on gaps and derivations

1. **Motor subscores (domains 1–3).** We found no published test–retest reliability for MDS-UPDRS tremor, bradykinesia–rigidity or axial subscores. Siderowf 2002 reported ICCs of 0.69–0.88 for "derived symptom-based" subscales of the original UPDRS, but which subscales these were, and their individual values, were not verified. Values were derived in three ways:
   - **Combining PPMI factors.** Where a domain spans several factors, we combined them assuming uncorrelated factors with unit weights: R = Σ(C0 + t·σ²ΔT) / Σ(C0 + t·σ²ΔT + σ²E). Tremor (F3.4 + F3.7) gives 0.58, 0.67 and 0.73 at baseline, year 1 and year 2. Bradykinesia–rigidity (F3.1 + F3.2 + F3.5 + F3.6) gives 0.73, 0.77 and 0.79.
   - **Spearman–Brown check from the Part III total**, assuming parallel items. Per-score reliability is ρ1 = R_T / (k − (k − 1)·R_T) with k = 33, and a subscore of m scores has R_m = m·ρ1 / (1 + (m − 1)·ρ1). We used R_T = 0.79 (PPMI baseline), 0.885 (PPMI year 2) and 0.93 (Spanish validation, 1–2 weeks):

     | Subscore | m | R_T = 0.79 | R_T = 0.885 | R_T = 0.93 |
     |---|---|---|---|---|
     | Tremor | 10 | 0.53 | 0.70 | 0.80 |
     | Bradykinesia–rigidity | 16 | 0.64 | 0.79 | 0.87 |
     | Axial | 5 | 0.36 | 0.54 | 0.67 |

     The parallel-item assumption understates homogeneous subscores (the axial items form their own factor), so treat these figures as a floor.
   - **Internal consistency is a poor guide for tremor.** In untreated PPMI patients the six rest-tremor items had a Rasch reliability (PSI) of only 0.36, because tremor at different body sites varies independently and most scores are zero (Regnault 2025). The sum can still be stable over time, as the PPMI-derived rest-tremor factor shows.
2. **Part II without items 2.10, 2.12 and 2.13.** Shortening from 13 to 10 items by Spearman–Brown (k = 13, m = 10) gives 0.73 (PPMI baseline), 0.85 (PPMI year 2) and 0.95 (Spanish, 1–2 weeks). The dropped tremor item sits in a noisier factor (PPMI-derived 0.59 at baseline), so the real loss is probably smaller.
3. **Hallucinations (item 1.2).** No item-specific coefficient was found. The Spanish validation reports test–retest kappa of at least 0.60 for every item except two rest-tremor scores. Quadratic-weighted kappa is equivalent to an ICC (Fleiss & Cohen 1973), so 0.60 is the central value. In PPMI the item is zero for most participants early on, and after a normal-scores transform it behaves almost like a yes/no indicator. Modelling it as an ordinal or binary indicator with misclassification may be better than a continuous score with fixed error variance.
4. **Part IV.** The only test–retest information is that every part total in the Spanish validation had ICC > 0.85, in a sample where 30% scored zero on Part IV. PPMI enrolled untreated patients, so Part IV is zero for nearly everyone in the first years. The value 0.70 is meant for visits after motor complications have appeared; before that, treating Part IV as an onset event is more defensible.
5. **GDS-15.** Most PD data are for the 30-item GDS. Halving the test length by Spearman–Brown (R15 = R30 / (2 − R30)) turns 0.89–0.94 into 0.80–0.89. This agrees with the 0.83 found for the GDS-15 in older community adults over 2 weeks (Nyunt 2009).
6. **Epworth Sleepiness Scale.** No PD test–retest study was found. The MDS Task Force recommends the scale for daytime sleepiness in PD (Högl 2010), but a systematic review found little test–retest evidence in any population (Kendzerska 2014). We derived r = 1 − SD²diff / (2·SD²) = 0.72 from Nguyen 2006: 142 sleep-clinic patients, mean interval 71 days, SD of score differences 3.9, score SD 5.2–5.3. PD internal consistency is α = 0.84 (Hagell 2007).
7. **ON versus OFF.** In PPMI the error variance for Part III ON assessments is 1.7 times that for OFF (27.07 vs 15.52). If treated participants contribute ON-state scores, lower the motor-domain reliabilities by about 0.05–0.10.
8. **Normal-scores scale.** These reliabilities carry over approximately after a monotone transform. The exception is a measure where many people share the floor or ceiling value: axial at baseline, hallucinations, Part IV, GDS-15, and Schwab & England at its ceiling. If normal scores are computed across all visits pooled, progression between visits adds true variance, so reliability will be somewhat higher than the per-visit values here.
9. **A more portable quantity, and an internal check.**
   - **Use the raw-point error SD.** Reliability depends on how spread out the sample is, but the error SD in raw points transfers between samples. For Part III, three independent sources agree on about 4 points:

     | Source | Error SD (points) | Basis |
     |---|---|---|
     | Evers 2019 (PPMI) | 3.94 | √15.52 |
     | Spanish validation, as cited by Lavine 2024 | 4.3 | reported SEM |
     | Steffen & Seney 2008 (original UPDRS motor score) | 3.97 | back-calculated from MDC95 = 11 |

     With PPMI access, D28 for Parts I–III can be applied as error variance = σ²E / observed variance at each visit (raw scale), with σ²E from Evers 2019.
   - **Check with PPMI's own repeat visits.** PPMI repeated the MDS-UPDRS (at least Parts II and III) at screening and baseline, about one month apart and before treatment (Evers 2019; Regnault 2019). This allows a direct PPMI test–retest estimate before the D28 values are frozen.

## Not verified

- **Martinez-Martin 2013.** The full text was not accessible (publisher login). Part II 0.96, Part III 0.93, the 1–2-week interval and the Part III SEM of 4.3 are taken from Lavine 2024's citation. The abstract states only ICC > 0.85 for each part and kappa ≥ 0.60 for all items but two; Evers 2019 describes the same results as ICC > 0.90. The number retested and the Part I and Part IV values are unknown.
- **Siderowf 2002.** Which derived subscales were tested, and their individual ICCs.
- **Gill 2008.** The number retested and the interval; the abstract gives only the ICC.
- **Visser 2004.** The total ICC of 0.87 and domain range of 0.65–0.90 come from the MDS Task Force review (Evatt 2009). Matsushima 2014 independently cites "ICCs 0.71 to 0.87" for Visser and Carod-Artal. The retest interval is unknown, and the review does not say whether the 0.65–0.90 range refers to domains or items.
- **Soltaninejad 2025** (GDS-15 ICC 0.92), **Tran 2026** (MoCA ICC 0.84 [0.69–0.92]) and **Carod-Artal 2010** (SCOPA-AUT ICC 0.71). The number retested and the interval for each.
- **Metman 2004.** Sample size and interval.
- **Forjaz 2015.** The reliability values of its Rasch tremor scale; the abstract says only "moderate".
- **Stiasny-Kolster 2007.** The abstract reports no test–retest result, and we could not confirm from the full text whether any retest was done.

## Corrections to how these sources are often cited

- **Goetz 2008 did not test test–retest reliability.** It reported internal consistency (Cronbach's α 0.79–0.93 across parts) and factor structure. The Spanish validation (Martinez-Martin 2013) is the only MDS-UPDRS validation with test–retest data, as Evers 2019 also notes.
- **Evers 2019's 0.13–0.62 are reliabilities of 1-year change scores.** They cannot be entered as 1 − error variance; cross-sectional values must be recomputed from the variance components, as done here.
- **Stiasny-Kolster 2007 studied isolated RBD and controls, not PD,** and its abstract gives no test–retest coefficient, so it cannot supply a D28 value. In de novo PD the RBDSQ performs poorly against polysomnography (AUC 0.68; Halsband 2018).
- **Visser 2004's abstract says only that test–retest reliability was "good"** (55 PD). The commonly quoted figures come from the full text via review articles.
- **Gill 2008 is cited correctly** (Mov Disord 2008;23(7):1043–1046). Its abstract gives test–retest ICC 0.79 and inter-rater ICC 0.81.

## References

All DOIs were checked against Crossref on 3 November 2026.

1. Aiello EN, Solca F, Torre S, et al. Reliable change indices for the Italian version of the Montreal Cognitive Assessment (MoCA) in non-demented Parkinson's disease patients. BMC Neurol. 2024;24(1):427. doi:10.1186/s12883-024-03920-9
2. Carod-Artal FJ, Ribeiro LS, Kummer W, Martinez-Martin P. Psychometric properties of the SCOPA-AUT Brazilian Portuguese version. Mov Disord. 2010;25(2):205-212. doi:10.1002/mds.22882
3. Dal Bello-Haas V, Klassen L, Sheppard MS, Metcalfe A. Psychometric properties of activity, self-efficacy, and quality-of-life measures in individuals with Parkinson disease. Physiother Can. 2011;63(1):47-57. doi:10.3138/ptc.2009-08
4. Evatt ML, Chaudhuri KR, Chou KL, et al. Dysautonomia rating scales in Parkinson's disease: sialorrhea, dysphagia, and constipation — critique and recommendations by Movement Disorders Task Force on Rating Scales for Parkinson's Disease. Mov Disord. 2009;24(5):635-646. doi:10.1002/mds.22260
5. Evers LJW, Krijthe JH, Meinders MJ, Bloem BR, Heskes TM. Measuring Parkinson's disease over time: the real-world within-subject reliability of the MDS-UPDRS. Mov Disord. 2019;34(10):1480-1487. doi:10.1002/mds.27790
6. Fleiss JL, Cohen J. The equivalence of weighted kappa and the intraclass correlation coefficient as measures of reliability. Educ Psychol Meas. 1973;33(3):613-619. doi:10.1177/001316447303300309
7. Forjaz MJ, Ayala A, Testa CM, et al. Proposing a Parkinson's disease-specific tremor scale from the MDS-UPDRS. Mov Disord. 2015;30(8):1139-1143. doi:10.1002/mds.26271
8. Gill DJ, Freshman A, Blender JA, Ravina B. The Montreal cognitive assessment as a screening tool for cognitive impairment in Parkinson's disease. Mov Disord. 2008;23(7):1043-1046. doi:10.1002/mds.22017
9. Goetz CG, Tilley BC, Shaftman SR, et al. Movement Disorder Society-sponsored revision of the Unified Parkinson's Disease Rating Scale (MDS-UPDRS): scale presentation and clinimetric testing results. Mov Disord. 2008;23(15):2129-2170. doi:10.1002/mds.22340
10. Hagell P, Broman JE. Measurement properties and hierarchical item structure of the Epworth Sleepiness Scale in Parkinson's disease. J Sleep Res. 2007;16(1):102-109. doi:10.1111/j.1365-2869.2007.00570.x
11. Halsband C, Zapf A, Sixel-Döring F, Trenkwalder C, Mollenhauer B. The REM Sleep Behavior Disorder Screening Questionnaire is not valid in de novo Parkinson's disease. Mov Disord Clin Pract. 2018;5(2):171-176. doi:10.1002/mdc3.12591
12. Högl B, Arnulf I, Comella C, et al. Scales to assess sleep impairment in Parkinson's disease: critique and recommendations. Mov Disord. 2010;25(16):2704-2716. doi:10.1002/mds.23190
13. Huang SL, Hsieh CL, Wu RM, Lu WS. Test-retest reliability and minimal detectable change of the Beck Depression Inventory and the Taiwan Geriatric Depression Scale in patients with Parkinson's disease. PLoS One. 2017;12(9):e0184823. doi:10.1371/journal.pone.0184823
14. Johns MW. Reliability and factor analysis of the Epworth Sleepiness Scale. Sleep. 1992;15(4):376-381. doi:10.1093/sleep/15.4.376
15. Kendzerska TB, Smith PM, Brignardello-Petersen R, Leung RS, Tomlinson GA. Evaluation of the measurement properties of the Epworth sleepiness scale: a systematic review. Sleep Med Rev. 2014;18(4):321-331. doi:10.1016/j.smrv.2013.08.002
16. Kim JY, Song IU, Koh SB, et al. Validation of the Korean version of the Scale for Outcomes in Parkinson's Disease-Autonomic. J Mov Disord. 2017;10(1):29-34. doi:10.14802/jmd.16057 (retest at 10–14 days; total Guttman split-half coefficient 0.86, not an ICC)
17. Lavine JS, Scotina AD, Haney S, Bakker JP, Izmailova ES, Omberg L. Impacts on study design when implementing digital measures in Parkinson's disease-modifying therapy trials. Front Digit Health. 2024;6:1430994. doi:10.3389/fdgth.2024.1430994
18. Martinez-Martin P, Rodriguez-Blazquez C, Alvarez-Sanchez M, et al. Expanded and independent validation of the Movement Disorder Society-Unified Parkinson's Disease Rating Scale (MDS-UPDRS). J Neurol. 2013;260(1):228-236. doi:10.1007/s00415-012-6624-1
19. Massai P, Colalelli F, Sansoni J, et al. Reliability and validity of the Geriatric Depression Scale in Italian subjects with Parkinson's disease. Parkinsons Dis. 2018;2018:7347859. doi:10.1155/2018/7347859
20. Matsushima M, Yabe I, Hirotani M, Kano T, Sasaki H. Reliability of the Japanese version of the Scales for Outcomes in Parkinson's Disease-Autonomic questionnaire. Clin Neurol Neurosurg. 2014;124:182-184. doi:10.1016/j.clineuro.2014.07.007 (31 patients with mixed diagnoses, 8 with PD; 1–4 weeks; total ICC 0.64)
21. Metman LV, Myre B, Verwey N, et al. Test-retest reliability of UPDRS-III, dyskinesia scales, and timed motor tests in patients with advanced Parkinson's disease: an argument against multiple baseline assessments. Mov Disord. 2004;19(9):1079-1084. doi:10.1002/mds.20101
22. Nguyen AT, Baltzan MA, Small D, Wolkove N, Guillon S, Palayew M. Clinical reproducibility of the Epworth Sleepiness Scale. J Clin Sleep Med. 2006;2(2):170-174. doi:10.5664/jcsm.26512
23. Nyunt MS, Fones C, Niti M, Ng TP. Criterion-based validity and reliability of the Geriatric Depression Screening Scale (GDS-15) in a large validation sample of community-living Asian older adults. Aging Ment Health. 2009;13(3):376-382. doi:10.1080/13607860902861027
24. Pena-Pereira MA, Sobreira-Neto MA, Sobreira E, et al. Validation of the Brazilian Portuguese version of the Rapid Eye Movement Sleep Behavior Disorder Screening Questionnaire (RBDSQ-BR). Arq Neuropsiquiatr. 2020;78(10):629-637. doi:10.1590/0004-282X20200125
25. Regnault A, Boroojerdi B, Meunier J, Bani M, Morel T, Cano S. Does the MDS-UPDRS provide the precision to assess progression in early Parkinson's disease? Learnings from the Parkinson's progression marker initiative cohort. J Neurol. 2019;266(8):1927-1936. doi:10.1007/s00415-019-09348-3 (PPMI Rasch PSI: Part II 0.72, Part III 0.86)
26. Regnault A, Prato MK, Quéré S, Benoit A, Massat NJ, Morel T. Optimizing the MDS-UPDRS Part III for early-stage Parkinson's: early supportive evidence for a limb-related bradykinesia/rigidity sub-score. NPJ Parkinsons Dis. 2025;11(1):297. doi:10.1038/s41531-025-01072-2
27. Röttgen S, Doppler CEJ, Seger A, et al. Context-dependency of the REM sleep behavior disorder screening questionnaire — clinical utility under scrutiny. Parkinsonism Relat Disord. 2025;130:107211. doi:10.1016/j.parkreldis.2024.107211
28. Shulman LM, Armstrong M, Ellis T, et al. Disability rating scales in Parkinson's disease: critique and recommendations. Mov Disord. 2016;31(10):1455-1465. doi:10.1002/mds.26649
29. Siderowf A, McDermott M, Kieburtz K, et al. Test-retest reliability of the Unified Parkinson's Disease Rating Scale in patients with early Parkinson's disease: results from a multicenter clinical trial. Mov Disord. 2002;17(4):758-763. doi:10.1002/mds.10011
30. Soltaninejad H, Taghizadeh G, Habibi SAH, et al. Validity and reliability of the Persian version of the Geriatric Depression Scale-15 in older adults with Parkinson's disease. Appl Neuropsychol Adult. 2025 (online ahead of print):1-8. doi:10.1080/23279095.2025.2547311
31. Steffen T, Seney M. Test-retest reliability and minimal detectable change on balance and ambulation tests, the 36-item short-form health survey, and the unified Parkinson disease rating scale in people with parkinsonism. Phys Ther. 2008;88(6):733-746. doi:10.2522/ptj.20070214
32. Stiasny-Kolster K, Mayer G, Schäfer S, Möller JC, Heinzel-Gutenbrunner M, Oertel WH. The REM sleep behavior disorder screening questionnaire — a new diagnostic instrument. Mov Disord. 2007;22(16):2386-2393. doi:10.1002/mds.21740
33. Tran HT, Huynh MBT. Reliability and validity of the Vietnamese version of the Montreal Cognitive Assessment in Parkinson's disease. Rev Neurol (Paris). 2026;182(3):175-185. doi:10.1016/j.neurol.2026.01.003
34. Visser M, Marinus J, Stiggelbout AM, Van Hilten JJ. Assessment of autonomic dysfunction in Parkinson's disease: the SCOPA-AUT. Mov Disord. 2004;19(11):1306-1312. doi:10.1002/mds.20153
35. Wang Y, Wang ZW, Yang YC, et al. Validation of the rapid eye movement sleep behavior disorder screening questionnaire in China. J Clin Neurosci. 2015;22(9):1420-1424. doi:10.1016/j.jocn.2015.03.008
36. Westergren A, Wictorin K, Hansson O, Hagell P. Novel insights regarding the measurement properties of the SCOPA-AUT. BMC Neurol. 2022;22(1):478. doi:10.1186/s12883-022-03008-2

## Addendum: PPMI's own screening–baseline retest (WP0, 3 November 2026)

PPMI gave MDS-UPDRS Parts II and III at screening and again at baseline, before any treatment. Among the primary population, 413 participants have both, a median of 31 days apart. On the pooled normal-scores scale (no phenotype split; aggregate results only, from `results/wp0/retest_reliability.json`):

| Domain | Pairs | ICC(2,1) | Implied error variance | Recommended above |
|---|---|---|---|---|
| Tremor | 413 | 0.80 | 0.20 | 0.30 |
| Bradykinesia and rigidity | 413 | 0.82 | 0.18 | 0.20 |
| Axial and gait | 413 | 0.75 | 0.25 | 0.25 |
| Daily function | 411 | 0.81 | 0.19 | 0.20 |

The PPMI values agree with the recommendations for three of the four domains. Tremor is more reliable in PPMI than the value derived from Evers 2019 (0.80 against 0.70); for D28, either use the PPMI value or keep 0.70 and treat 0.80 as the sensitivity analysis. These retest pairs are from untreated participants at entry, where the true spread is smallest, so they are, if anything, conservative for later visits.

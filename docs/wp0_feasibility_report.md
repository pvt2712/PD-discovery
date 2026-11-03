# WP0 feasibility audit: results

> 3 November 2026. Protocol draft v0.5, work package WP0. Aggregate counts only, from the PPMI download of 13 October 2026, built and audited on the investigator's machine with the investigator's written approval (given in the chat on 3 November 2026). Counts from 1 to 4 are suppressed, together with any cell that would reveal one; a few counts hidden that way are given rounded to the nearest 10. Nothing participant-level is in this report. The provenance manifest (file names, SHA-256 hashes, counts) is `results/wp0/manifest.json`.

## Summary

- **Population.** 1,127 people with sporadic Parkinson's disease, untreated when they joined PPMI: 430 in PPMI 1.0 and 697 in PPMI 2.0. By motor phenotype at baseline:
  - 804 tremor-dominant (TD);
  - 209 PIGD;
  - 114 indeterminate.
- **Landmark.** 986 were seen at the year-1 visit, the start of the analysed transitions: 706 TD, 177 PIGD and 103 indeterminate. That is fewer than the 1,388 the pilot build suggested before treated participants were excluded.
- **Transitions.** 3,727 year-to-year transitions after the landmark: 2,710 TD, 638 PIGD and 379 indeterminate. The PIGD group, with 177 people and 638 transitions, is what limits power.
- **Intervals.** 93% of intervals lie within 12 ± 3 months, so Step 1 can use the annual tables as they are (D13).
- **Decisions the data inform:**
  - **D5, motor state:** 21% of grid visits have only an ON examination.
  - **D26, floor-bound domains:** hallucinations and motor complications sit at their floor.
  - **D16, the Part II endpoint:** only 43 events in 906 participants at 24 months.
  - **D12, ε in clinical units:** the baseline standard deviations below give it.
- **Power (the WP0 gate).** At WP0's numbers, the synthetic pilot puts H-P's power against one deviation of 2ε above 99%, so the gate passes (§7).

## 1. Participant flow

PPMI's table of the Parkinson's disease cohort lists 2,262 people, including some who were never enrolled.

| Step | PPMI 1.0 | PPMI 2.0 |
|---|---|---|
| Listed in the PD cohort table | 715 | 1,547 |
| Not enrolled (screen failed, excluded, declined or pending) | 0 | 513 |
| Genetic enrolment (LRRK2, GBA, SNCA, PRKN, PINK1, other) | 280 | 81 |
| Treated at baseline | under 5 | about 250 |
| No baseline visit, or no usable baseline phenotype | under 5 | under 20 |
| **Primary population** | **430** | **697** |

## 2. Phenotypes

| | TD | PIGD | Indeterminate | Total |
|---|---|---|---|---|
| PPMI 1.0 | 305 | 77 | 48 | 430 |
| PPMI 2.0 | 499 | 132 | 66 | 697 |
| Total | 804 | 209 | 114 | 1,127 |

- **The rule.** Phenotypes come from the Stebbins 2013 rule, applied to the baseline (untreated) MDS-UPDRS: TD if the tremor/PIGD ratio is at least 1.15, PIGD if it is at most 0.90.
- **Overlap.** 105 participants have a ratio within 0.10 of a cut-off. The ratio's 10th, 50th and 90th percentiles are 0.28, 1.67 and 4.55.
- **Stability.** Among those with a classifiable examination later, the phenotype was unchanged:

| Baseline phenotype | Same at year 1 | Same at year 2 |
|---|---|---|
| TD | 84% (509 of 609) | 82% (431 of 523) |
| PIGD | 67% (97 of 145) | 61% (68 of 112) |
| Indeterminate | 23% (20 of 87) | 14% (10 of 74) |

- **Clinical subtypes** (provisional Fereshtehnejad 2017 rule, percentiles within this population; the secondary axis, D2):
  - mild motor-predominant about 500;
  - intermediate about 460;
  - diffuse malignant about 160.

## 3. The landmark and follow-up

| | Reached year 1 | Not reached | Total |
|---|---|---|---|
| TD | 706 | 98 | 804 |
| PIGD | 177 | 32 | 209 |
| Indeterminate | 103 | 11 | 114 |
| Total | 986 | 141 | 1,127 |

- **Transitions.** 3,727 consecutive-year transitions after the landmark: 2,608 in PPMI 1.0 and 1,119 in PPMI 2.0. Another 369 steps cross a missed year; the engine uses them with their actual interval (§6.3), but Step 1's annual tables do not.
- **Visits per participant from year 1 on.** PPMI 1.0 has a median of 8 (quartiles 5–12). PPMI 2.0 has a median of 3 (quartiles 2–4; 91% have 2 to 5).
- **Intervals.** Between consecutive grid visits the median is 1.00 year (10th–90th percentile 0.84–1.17); 93% lie within 12 ± 3 months.
- **Visit mode.** 85 of the 6,254 grid visits (1.4%) are remote; the grid takes an in-person visit whenever one exists.
- **Medication.** It changed in 55% of transitions.

## 4. Motor state and missing data

| Motor state at the grid visit | Visits |
|---|---|
| Untreated | 1,984 |
| OFF (practically defined) | 2,801 |
| ON only (no OFF examination) | 1,319 |
| OFF recorded, but no examination | 36 |
| No Part III examination | 114 |

| Domain | Missing, all grid visits |
|---|---|
| Tremor, bradykinesia and rigidity, axial and gait | 24%, almost all at ON-only visits |
| Daily function (MDS-UPDRS II) | 0.6% |
| Independence (Schwab & England) | 1.4% |
| Cognition (MoCA) | 6.6% |
| Mood, autonomic, sleepiness, RBD symptoms, motor complications | 4.6–4.9% |
| Hallucinations (MDS-UPDRS 1.2) | 1.2% |

**For D5.** With OFF examinations only (the recommended default), the three motor domains are missing at the 21% of visits that have only an ON examination. The engine's treatment term carries the examination state (§6.1), so those ON examinations could enter with their state recorded. That would cut the motor domains' missingness from 24% to about 3%, at the cost of leaning on the treatment model. For the panel and the investigator to decide.

## 5. Scores at baseline and year 1 (for D12 and D26)

| Domain (higher = worse) | Baseline mean (SD) | Year 1 mean (SD) | At the floor, baseline / year 1 | 0.1 SD at baseline |
|---|---|---|---|---|
| Tremor (3.15–3.18, 0–40) | 4.8 (3.5) | 5.7 (4.0) | 12% / 11% | 0.35 points |
| Bradykinesia and rigidity (3.3–3.8, 3.14, 0–64) | 14.0 (7.4) | 16.1 (8.4) | 0% / 0% | 0.74 points |
| Axial and gait (3.9–3.13, 0–20) | 1.6 (1.4) | 2.0 (1.6) | 22% / 17% | 0.14 points |
| Daily function (MDS-UPDRS II without 2.10, 2.12, 2.13) | 4.7 (3.9) | 5.6 (4.4) | 12% / 10% | 0.39 points |
| Independence (100 − Schwab & England) | 6.3 (6.3) | 8.2 (7.1) | 35% / 24% | 0.63 points |
| Cognition (30 − MoCA) | 3.1 (2.4) | 3.1 (2.7) | 13% / 15% | 0.24 points |
| Mood (GDS-15) | 2.3 (2.6) | 2.4 (2.6) | 27% / 24% | 0.26 points |
| Autonomic (SCOPA-AUT) | 9.2 (5.9) | 10.2 (6.2) | 2% / 1% | 0.59 points |
| Hallucinations (MDS-UPDRS 1.2) | 0.04 (0.23) | 0.07 (0.28) | **97% / 94%** | — |
| Daytime sleepiness (ESS) | 5.6 (3.7) | 5.8 (3.8) | 4% / 5% | 0.37 points |
| RBD symptoms (RBDSQ) | 4.1 (2.9) | 4.2 (3.0) | 3% / 3% | 0.29 points |
| Motor complications (MDS-UPDRS IV) | 0 (untreated) | 0.46 (1.42) | **100% / 88%** | — |

- **ε in clinical units (D12).** ε = 0.10 is a deviation of 0.1 SD of the target domain per SD of the source domain per year. For example, axial and gait → independence: 0.1 × 6.3 / 1.4 ≈ 0.45 Schwab & England points per year for each axial point.
- **Floor-bound domains (D26).** Hallucinations and motor complications have more than half their scores at the floor. Each needs a latent-threshold model or must leave H-P's support, as the panel decides. Independence is below the 50% rule.

## 5b. Measurement reliability from PPMI's own repeat (for D28)

PPMI gave MDS-UPDRS Parts II and III at screening and again at baseline, before treatment. In 413 participants of the primary population (median 31 days apart; pooled, no phenotype split), the test–retest ICCs on the normal-scores scale are:
- tremor 0.80;
- bradykinesia and rigidity 0.82;
- axial and gait 0.75;
- daily function 0.81.

These agree with the literature-based recommendations in `docs/measurement_reliability.md`, except tremor (0.70 recommended), which is more reliable in PPMI. The provisional error variances are in `configs/measurement_error.json`.

## 6. The Part II endpoint (D16)

43 of 906 participants with a known 24-month status died, or fell below 80% on Schwab & England, by 24 months (4.7%). That is too few events to develop a prognostic model for this endpoint (Riley 2020). The fallbacks of §12.4 apply: a longer horizon, or a composite.

## 7. Power for H-P (the WP0 gate)

> Added 4 November 2026 from the pilot's 8 datasets. Synthetic data only; the numbers are in `results/wp0/power/summary.json`.

**Setup.** The family-A simulator was rerun at WP0's numbers:
- 652 TD and 164 PIGD participants with at least one transition after the landmark;
- WP0's visit counts and missing shares, with 10 domains;
- scenario S2: PIGD progresses twice as fast, and one link (daily function → independence) is stronger in PIGD by 2ε beyond that speed.

Each dataset gets one speed-only fit and one free fit, without a bootstrap.

**Results.**
- **Every dataset rejected at 5%.** The likelihood ratios ran from 40 to 95, against chi-square critical values of 32.7 to 35.2 (21 to 23 df). The degrees of freedom vary because H-P re-selects its support in each dataset.
- **Power.** The estimated non-centrality is 54.1 (standard error 6.3; lower 95% bound 41.8), and 80% power needs only about 22. The noncentral power is 99.9% (range 99.1–99.99%). **The gate passes.**
- **Margin.** H-P's real critical value comes from its bootstrap, not from the chi-square distribution.
  - A critical value of 40 would leave the power at 99.5%, and one of 50 would leave it at 96%.
  - The weakest dataset (40.0 at 22 df) shows the spread: it would not have rejected at 50.
  - WP3's H-M1 checks the bootstrap's size.
- **For the analysis plan.** The speed ratio from the speed-only model (Sp-1) read 2.1 to 2.7 (median 2.4), against a true 2.0. In S2 that model is wrong by design, so part of PIGD's extra link shows up as faster speed. When H-P rejects, Sp-1 therefore mixes speed with the pathway difference, and the analysis plan should say how it is read then. Its bias when the speed-only model is right is judged in S1 (WP3).
- **Cost (for D14).** Each dataset took 2.0 to 5.4 hours (median 3.4) with 4 jobs running together, and the pilot took about 9 hours. So this Mac completes about one fit pair an hour at 10 domains. The 200-dataset Monte Carlo run cannot change the verdict and would take about a week. If a direct Monte Carlo check is wanted, 20 datasets that all reject show at least 80% power on their own (exact lower bound 83%).

## Files

| File | Contents |
|---|---|
| `results/wp0/wp0_audit.json` | Every aggregate table above, with suppression |
| `results/wp0/participant_flow.json`, `phenotype_axes.json` | Flow and phenotype axes, with suppression |
| `results/wp0/retest_reliability.json` | Screening–baseline retest reliability (§5b) |
| `results/wp0/power/summary.json` | The power pilot (§7; synthetic data) |
| `results/wp0/manifest.json` | Download stamp, file hashes, counts, approval reference |
| `data/derived/*.parquet` | Participant-level tables (git-ignored; never shared) |

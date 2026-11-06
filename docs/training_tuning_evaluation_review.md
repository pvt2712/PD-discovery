# Training, tuning and held-out evaluation: a review of the phenotype causal discovery

> 6 November 2026. Reviewed against the held-out evaluation note `causal-discovery-held-out-evaluation-methodology.md` (written outside this project, 6 November 2026). Protocol v0.5 with decisions up to D32; code at commit 779977c. No PPMI model estimates appear here; the only PPMI numbers are pipeline counts.

## Summary

**The note's main design does not fit PPMI.**
- The note judges a causal model by how well it predicts the results of interventions held out from training.
- PPMI has no randomised interventions. Our estimand is also not an interventional effect: it is the speed-adjusted dynamics of the domains under the care patients received (§2.3).
- What carries over:
  - the note's *structural* evaluation, which is possible only in simulation, where the truth is known;
  - its *observational-holdout* section (its §16);
  - its *environment* and *temporal* holdouts.

**Training and tuning are mostly sound for a hypothesis test.**
- Every statistical setting is chosen on simulated data with a known truth, or fixed from outside (published reliabilities, clinical guidelines, the panel), never on PPMI outcomes. This is the right counterpart of the note's validation set, and stronger, because the truth is known.
- Step 1's support is selected again inside every bootstrap replicate, so H-P is calibrated after the selection. The data are not used twice unaccounted.
- The participant is the unit of every resample, as the note asks (no pseudo-replication).

**Four weaknesses:**
1. **PPMI is no longer an untouched test set.**
   - The exploratory look (D30) computed H-P on all 883 participants of the landmark population.
   - Decisions D31 and D32 and the medication fixes followed it.
   - A later H-P on PPMI is pre-specified but not blind, and no PPMI participant remains unseen.
2. **The validation data lack the artefact that matters most.** Both simulator families draw speed and load *after* the phenotype label. So WP3 cannot yet see the route by which the phenotype rule itself could make H-P reject falsely: selection on traits.
3. **Observed and bootstrap fits are not started alike.**
   - Bootstrap replicates start at the model that generated them, while the observed fits start from default values.
   - On real data, whose likelihood surface is rougher, an under-optimised observed fit distorts the statistic, and the bootstrap cannot reproduce that error.
4. **Part I has no held-out check on real data.** Nothing checks, out of sample, whether:
   - the free model's extra links generalise to new participants;
   - the null model forecasts adequately;
   - Step 1's map holds on data it did not see.

**What to add.** H-P stays the only primary test; everything below is pre-specified secondary or diagnostic (§5):
- a record of every contact with PPMI, and a frozen pipeline;
- a prospective test on the next PPMI data release, the only PPMI data still unseen;
- participant-level cross-validated likelihood, and last-visit forecasts against simple baselines;
- a held-out test of Step 1's implied independences;
- an audit of the observed fits' starting values;
- separate WP3 datasets for choosing settings and for reporting performance;
- trait-first simulators.

**One decision for you:** whether external replication becomes unconditional, as the confirmatory test of the TD/PIGD question.

## 1. What the note asks for, and what fits this study

| The note's evaluation | What it needs | Here | Where it would sit |
|---|---|---|---|
| Interventional holdout (its primary design: whole intervention targets withheld) | Randomised or experimental interventions | Not possible: PPMI has none. Medication is chosen by indication; the note itself lists treating such an exposure as an intervention among its common errors (its §18, item 13) | Part II-d's target-trial emulation, which is a separate question |
| Structural evaluation | A known true graph | Possible only in simulation | WP2 engine gate and WP3 qualification (§8): done or planned |
| Observational holdout (its §16): held-out likelihood, prediction error, calibration, implied independences, stability | Participants or visits not used in fitting | Not done in Part I. Done only in the exploratory Step 3 and in Part II-b | §5 adds it |
| Environment holdout | A new site, era or cohort | PPMI 1.0 against 2.0 inside PPMI; an external cohort outside it | Era replication (§7.10) refits in each era, which is not a holdout. External replication (Part II-a) is a true holdout, but conditional |
| Temporal holdout | Later observations | Visits after the download of 13 October 2026 | §5 adds it |

The note's warning applies throughout: none of these evaluations shows that a direction is causally right. They test calibration, generalisation and consistency, and the protocol already says so (Part II: "prediction accuracy does not validate the causal map").

## 2. How the method is trained

### 2.1 What is fitted, on what

| Stage | What happens | Data | Code |
|---|---|---|---|
| Preparation | Scores on one normal-scores scale, computed on the pooled population over all visits and then frozen. Error variances = (1 − published reliability) × the pooled year-0 variance. Phenotype from year-0 items by published cut-offs. Year-0 values enter no model term (the landmark is year 1) | All 883 participants (706 TD, 177 PIGD), labels not used | [ppmi.py](../src/pdx/ppmi.py), [build_analysis_panel.py](../scripts/build_analysis_panel.py) |
| Step 1 | FCI with time tiers on the two-slice table, pooled and blind to the label. Fast configuration: Fisher-z, α = 0.05, separating sets of at most 3. The support is the expert map's links plus the lagged links found | All participants, labels not used | [step1.py](../src/pdx/step1.py) |
| Engine fits | Maximum likelihood (L-BFGS-B with exact gradients). Log-speed is integrated by 20-node Gauss–Hermite quadrature; load, measurement error and missing values by Kalman filtering. Null: one shared mechanism with phenotype-specific speed. Free: PIGD's links free on the support | All participants | [engine/fit.py](../src/pdx/engine/fit.py) |
| H-P | Likelihood ratio, referred to a conditional parametric bootstrap from the fitted null. The support is re-selected and both models refitted in every replicate, with sequential stopping | Simulated from the fitted null | [hp.py](../src/pdx/hp.py) |
| Speed estimates | Sp-1 to Sp-3 from the same fits; 200 participant-bootstrap samples, each with its support re-selected | Resampled participants | [speed.py](../src/pdx/speed.py) |

### 2.2 Assessment

**Sound:**
- **Selection inside the bootstrap.** Because Step 1 is re-run in every replicate, the bootstrap's reference includes the selection step. The note's rule that the whole pipeline is evaluated as a unit is met. Sample splitting would also be valid, but it would cost H-P power the PIGD group cannot spare.
- **The bootstrap calibrates the procedure as implemented,** including the optimiser and its stopping rule. A statistic computed by an imperfect optimiser is still calibrated, *provided the observed and simulated data are fitted the same way.* That proviso is the issue below.
- **A label-blind selection.** The phenotype never enters Step 1 or the preparation. The expert map comes from literature and guidelines, with PMIDs per link ([expert_map_provisional.json](../configs/expert_map_provisional.json)), not from PPMI results.
- **The landmark** keeps the phenotype-defining year-0 values out of every model term.

**Weakness: the observed and replicate fits are not started alike** ([fit_pair](../src/pdx/engine/fit.py), [_replicate](../src/pdx/hp.py)).
- **The observed null fit** starts at default values, plus one start with the speed spread at 0.6.
- **Each replicate's null fit** starts at the model that generated the data, near its optimum.
- **What goes wrong.** If the observed null stops short of its optimum, the statistic is inflated and the test becomes too liberal. The replicates cannot reproduce that error.
- **Evidence it happens on real data.** The PC run of 4–5 November is invalid as a test, but its fits are informative:
  - neither observed fit converged in 10.6 hours;
  - the free fit's two starts ended far apart in log-likelihood;
  - the null may have settled in the wrong speed basin.
- **What the revision does and does not fix.**
  - The alternating starts (the null restarted from the free fit's projection) and the stopping window reduce the problem.
  - They do not make the procedure symmetric.
  - WP3's warp-speed calibration copies the asymmetry, so H-M1 measures it, but on simulated data. Their surface is smoother than PPMI's, because the model is correct there.

**Disclosed, but chosen after looking:** D31 (λ = 0) was decided after the PPMI fit showed the speed–load correlation at its bound. It is a fix for a model the data could not identify, not a response to the statistic, and λ free is a sensitivity analysis (§7.14, item 21). It still belongs in the record of §5, item 1.

## 3. How the settings are chosen

| Setting | Value | How it is set | On what data | Uses PPMI outcomes? |
|---|---|---|---|---|
| Step 1 test and configurations | Fisher-z fast; clustered Fisher-z, depth 4 for the reported map | WP3 picks among 3 candidates, at most 3 configurations per learner | Simulated | No |
| Step 1 α, depth | 0.05; 3 | Fixed in advance (provisional) | — | No |
| Step 1 time step | Annual tables if ≥ 90% of intervals are within 12 ± 3 months | WP0 interval counts | PPMI visit dates only | No |
| Expert map | 14 links | Literature and guidelines; the panel's map replaces it | External | No |
| Quadrature nodes | 20 (D32) | The engine gate's rule, against 30 and 40 | Simulated | No |
| Stopping window, iterations | Gain < 0.01 over 100 iterations; at most 5,000 | Numerical, set in the revision | Simulated | No; set after the PPMI look |
| Load moving with speed | λ = 0 (D31) | After the PPMI fit failed to identify it | PPMI | Fit diagnostics, not the statistic |
| Measurement error | (1 − R) × year-0 variance | Published reliabilities; pooled variance | External + pooled PPMI | No |
| Treatment curve, half-life | Emax; ED50 = EC50 = 3; 28 days | Provisional (D24); the panel confirms | External | No |
| ε | 0.10 SD per SD per year | The panel (D12) | — | No |
| Localisation penalty | Adaptive lasso tuned by BIC with participants as units | WP3 compares BIC variants | Simulated (variants); PPMI (the penalty itself, in-sample) | Exploratory step only |
| Stable-link threshold | Smallest with precision ≥ 0.8 | Calibrated | Simulated | No |
| Step 3 classes K | 2–4 | BIC, with held-out cross-validated likelihood beside it | PPMI, held out | Exploratory step only |
| Part II-b models | — | Nested cross-validation by participant, everything inside the folds | PPMI, held out | Done correctly |

**Assessment:**
- The principle is right for a hypothesis test.
  - A test's settings must hold its error rates, not maximise fit, and error rates can only be measured where the truth is known.
  - Simulation with PPMI-fitted generators therefore plays the note's validation role, and PPMI outcomes never enter.
  - BIC instead of cross-validation for the localisation penalty is also right, because the aim is consistent selection, not prediction (the protocol keeps cross-validated lasso selection out of its tests).
- **Gap: choosing and reporting on the same simulated datasets.**
  - The protocol separates tuning datasets for the three tuned comparators, C1, C4 and C9 (§8.4).
  - It does not say the same for its own choices: the Step 1 configuration, the BIC variant, and the replacement localisation procedure (§8.7).
  - Choosing the best of several configurations and then reporting its performance on the same datasets flatters it.
- **Smaller caveat: the validation data are PPMI-shaped.**
  - The simulators are fitted to pooled PPMI, without phenotype comparisons, so the settings are tuned for PPMI-like data. That is intended.
  - Family A also shares the engine's model class. Family B, coded separately and outside that class, is the guard, which is why H-M1 requires both.

## 4. How the method is validated and tested

### 4.1 Validation: the engine gate and WP3

**What they measure.**
- The engine gate (WP2) checks, on synthetic data of the real size:
  - the likelihood against an independent implementation;
  - recovery, coverage, convergence and identification.
- WP3 measures, against the true structure:
  - H-P's false-rejection rate in eight shared-mechanism scenarios of family A and two of family B, with Bradley's 7.5% criterion;
  - H-P's power;
  - the speed estimates' bias and coverage;
  - localisation errors;
  - Step 1's adjacency and arrowhead F1 against the true partial ancestral graph;
  - a descriptive benchmark of twelve named comparators.
- This is the note's "structural evaluation", and it is the strongest evidence a test of this kind can have.

**The gap: selection on traits.**
- Both simulator families draw the year-0 state, apply the phenotype rule, and only then draw each participant's speed and load from the label ([family_a.py](../src/pdx/sim/family_a.py), [family_b.py](../src/pdx/sim/family_b.py)).
- In patients the order is reversed. A fast or heavily loaded course shapes the year-0 symptoms that the rule reads. So the label selects on speed, load and mechanism together: the collider route raised in the reviewer comments on TD/PIGD subgroups.
- The simulated phenotypes therefore inherit range restriction and regression to the mean, but not this route.
- §8.2 promises it ("the rule applied to simulated baselines reproduces ... the artefacts"). Until the simulators generate speed and load *before* the year-0 state, H-M1 cannot certify H-P against it.
- The definition-matched generator (§7.3.6) is meant to cover it, but so far it exists only as a harness.

### 4.2 Testing: the PPMI analysis

**For a test, the "test set" is the confirmatory dataset,** analysed once with a frozen pipeline. Here that is PPMI in WP4.

**What D30 changed.**
- The exploratory look ran the full procedure, including the observed statistic, on the whole landmark population.
- The protocol records the consequence: "a later confirmatory analysis is no longer blind" (D30).
- Changes then followed it:
  - the bug fixes, which were needed whatever the result;
  - the medication coding and the error variances, found by auditing the data pipeline;
  - D31 and D32.
- In the note's terms, PPMI has served as both validation and test set.
- The statistic's validity under the null does not depend on whether anyone saw it. What it does depend on is whether analysis choices responded to it. The defence is a complete record (§5, item 1) and an untouched confirmation (§5, items 2 and 7).

**Internal replication is not a holdout.**
- Era replication and random halves (§7.10) refit the analysis in each part.
- They show consistency, but both parts were in the exploratory look, and neither is trained on one part and tested on the other.

**The protocol's true holdouts are both conditional on Part I:**
- Part II-a, external replication with the pipeline frozen before the external data are opened;
- Part II-b, nested cross-validation and external validation.

So as written, a non-rejecting or speed-only result in PPMI is never tested on held-out data.

### 4.3 The note's common errors, checked

| The note's error (its §18) | Here |
|---|---|
| Observations of one unit split across training and test | The participant is the unit of every resample and fold |
| Model or threshold chosen on the test data | Partly: D31 followed the PPMI look (disclosed, with a sensitivity analysis) |
| Test data used for preprocessing | Acceptable while Part I has no holdout; any held-out analysis must refit Step 1 and the error variances inside its training part (§5) |
| Repeated measurements as independent units | Avoided: participant clusters throughout; site as a sensitivity analysis |
| Observational prediction read as causal accuracy | Avoided: the protocol says prediction does not validate the map |
| Large groups dominating averages | To watch: TD has four times PIGD's participants, so any predictive check should average within phenotype first |
| A non-significant response read as no effect | Avoided: "no mechanism difference detected" is not "speed and load only", which needs equivalence within ±ε |
| Abstention without coverage | Avoided: unresolved links and "not separable" verdicts are reported as such |
| Non-randomised exposure read as an intervention | Avoided: dose–score associations are descriptive; treatment effects go to a target-trial emulation |

## 5. Recommendations

| # | What | Why | Cost | When |
|---|---|---|---|---|
| 1 | **Data-contact record and freeze.** One table of every PPMI contact (WP0 aggregates; D30's look and run; what was seen), every later change with its trigger (bug, failed fit, or result) and commit. Freeze code and configuration by hash before the next full run; the checkpoint already stores both | It is the only defence of a non-blind test, and reviewers will ask | An hour | Before the PC full run |
| 2 | **A prospective test on the next PPMI release.** See below | The only PPMI data the exploratory look has not touched | Scoring only; no refit | Pre-register now; run at the next download |
| 3 | **Audit the observed fits' starts.** Up to 10 dispersed starts for each observed fit, run in parallel, keeping the best. Require the best null to be reached from at least two starts within 0.1 log-likelihood. Refit about 20 bootstrap replicates from cold starts as well as warm, and compare their statistics | Makes observed and replicate fits comparable (§2.2) | About one more observed-fit round, plus 20 replicates | Decide before the PC full run; it does not block the timing run |
| 4 | **Cross-validated likelihood by participant.** See below | Shows whether the free model's extra links generalise to new participants | About five times the observed fits | WP4, pre-specified as secondary |
| 5 | **Last-visit forecasts against baselines.** See below | Checks the null model's adequacy, and whether the free model forecasts better | One fit pair plus cheap baselines | WP4, secondary |
| 6 | **Step 1's implied independences on held-out participants.** Learn the map on a random half, or in PPMI 1.0; test the independences it implies in the other half, or in PPMI 2.0. Report the violation rate against random maps of the same density, and against the expert map | Real-data evidence that the support is adequate; a missing link is the WP3 stress that most threatens H-P | Minutes | WP4, secondary |
| 7 | **External replication, unconditional** (your decision). Run the frozen H-P and Sp-1 in one external cohort (Part II-a's candidates), whatever PPMI shows | After D30 it is the only clean confirmation; as written, a speed-only result is never replicated | Months: access and harmonisation | Decide before registration (WP2) |
| 8 | **Separate WP3 datasets for choosing and for reporting.** Reserve seed blocks: choose the Step 1 configuration, the BIC variant and any replacement procedure on one block; report H-M1, power and coverage on another | Removes the selection bias of §3 | None | Write into §8.6 before WP3 |
| 9 | **Trait-first simulators.** Draw speed and load first, generate the year-0 state from them, then apply the phenotype rule; add this as the null in both families | H-M1 cannot certify H-P against selection on traits without it (§4.1) | Code, about a day; reruns within WP3 | Before WP3 |
| 10 | **Transfer between eras (exploratory).** Fit on PPMI 1.0 and score the predictive likelihood of PPMI 2.0, and the reverse | The note's environment holdout | Two fit pairs | WP5 |

### The prospective test (item 2)

- **Freeze now:**
  - the commit and configuration;
  - the panel rules;
  - the fitted null and free models from the 13 October 2026 download, whose parameters the checkpoint already stores.
- **When the next download comes:**
  - take every visit dated after 13 October 2026 for participants of the landmark population, and score it by its predictive density given that participant's earlier visits under each frozen model;
  - the engine gives this directly: the participant's log-likelihood with the new visits, minus it without them, at fixed parameters.
- **Primary contrast:** mean log predictive density, free minus null, averaged within each phenotype and then across the two, with a participant-bootstrap interval.
- **Secondary checks:**
  - per-domain standardised forecast error;
  - coverage of 90% predictive intervals;
  - calibration by phenotype of the speed-only model, which tests Sp-1's forecast of faster PIGD progression on unseen visits.
- **Size.** WP0's code counts the new visits once the download exists; that number fixes the test's power.

### Cross-validated likelihood (item 4)

- **Folds.** Five, stratified by phenotype and era, with whole participants per fold.
- **Inside each training part:**
  - Step 1 is re-run;
  - the error variances are re-estimated;
  - both models are fitted.
- **Scoring.** Held-out participants are scored by their marginal likelihood, with speed and load integrated out.
- **The normal-scores scale** may stay frozen. It uses no labels or outcomes by phenotype, and it affects both models alike, so the paired contrast is unaffected. Refitting it per fold is a sensitivity analysis.
- **Reading.**
  - Cross-validated likelihood is roughly AIC (Stone 1977). Free beats null out of sample when the likelihood ratio exceeds about twice its degrees of freedom (24 with the provisional map), a stricter bar than H-P's 5% level.
  - It is therefore not a second test. It shows whether a mechanism difference is large enough to matter for new patients.
  - If H-P rejects but the free model loses out of sample, the finding should be graded unstable, an extension of §7.7's rule.

### Last-visit forecasts (item 5)

- **Design.** Hold out each participant's last visit, fit on the rest, and forecast it from the participant's history.
- **Compare against simple baselines:**
  - last value carried forward;
  - a linear mixed model with phenotype-by-time terms;
  - a flexible learner on lagged values, phenotype and characteristics, as the note's non-causal baseline.
- **Metrics:**
  - standardised error per domain;
  - log predictive density;
  - 90% interval coverage;
  - the worst domain;
  - averaged within phenotype first.
- **Reading.** If the speed-only model forecasts worse than the baselines, its fit, not the phenotypes, is the problem. That would cast doubt on any H-P result built on it.

## 6. What not to adopt from the note

- **A 60/20/20 split for H-P.**
  - H-P is a test on all participants, calibrated by the bootstrap and qualified by simulation.
  - A three-way split would leave about 106 PIGD participants in training and about 35 in each other part, where WP0's power check assumed all 177.
  - The held-out data should come from outside the analysed set (items 2 and 7), and inside it only for secondary checks (items 4–6).
- **Intervention metrics** (intervention MAE, intervention RMSE, Precision@k). There are no held-out interventions to score. Their observational counterparts are the predictive density and standardised forecast error of items 2, 4 and 5.
- **Tuning the test's settings on a validation loss.** The settings are tuned to hold error rates, which only simulation can measure.
- **Refitting on training plus validation data.** There is no PPMI validation part to fold back in.

# 8. What the readers found: a ranked list for the audit

*Collected on 4 November 2026 from the "Notes for the author" at the end of chapters 1–7. The line numbers are those of commit `a9b577f`. Each item was found by reading the code, and most were also reproduced with a small synthetic script; the chapters give the details and the scripts are in `docs/guide_examples/`. The revision of 5 November 2026 fixed group A and a few B items; see "Status after the revision" at the end. The chapters' line anchors still refer to `a9b577f`, so for the files the revision changed they are approximate.*

The items are ranked by how much they could change a result:
- **A** could change H-P's statistic or its validity, so these should be fixed before any result is reported;
- **B** affects the localisation, the speed intervals, the engine gate or the method qualification;
- **C** concerns data governance;
- **D** needs only the protocol or the documents to be corrected;
- **E** lists the gaps in the tests and minor items.

Two readers found item A1 independently.

## A. Could change H-P's result or its validity

| # | Finding | Where | Why it matters | Proposed fix | § |
|---|---|---|---|---|---|
| A1 | **A failed likelihood is reported as a converged fit.** On any failure `value_and_grad` returns 1e10 with a zero gradient. At a start point L-BFGS-B reads that as convergence, so `fit_model` reports `success=True` with log-likelihood −1e10, and neither `hp.py` nor `gate.py` checks. | `likelihood.py:216–230`, `fit.py:167` | If it happened to the speed-only fit, LR would be about 2e10, a certain false rejection. | Return an infinite objective, or raise, and mark the fit failed. Make `run_hp` and the gate treat a failed observed fit as an error. Add a test. | 2, 3 |
| A2 | **A domain never observed at the year-1 landmark in one phenotype makes the landmark prior NaN.** Only a warning is shown, every evaluation then fails, and A1 follows. | `likelihood.py:84, 87–88` | Silent failure. It is unlikely with 10 domains in PPMI, but possible in small strata. | Check in `prepare` and stop with a clear message. | 2 |
| A3 | **The optimiser can miss the optimum.** One speed-only start got stuck at a speed spread of zero (log σ = −14.5), 5 log-likelihood units worse. One dispersed free start beat both of `fit_pair`'s starts by 0.35, which moved LR from 3.36 to 4.07. The speed-only fit has a single start. The observed fit starts at the default values while the bootstrap replicates start at the observed estimate. | `gate.py:101`, `fit.py:193`, `hp.py:74` | LR can be biased in either direction, and the asymmetry between observed and replicate fits can change the test's size. | Give both fits several starts, including one with σ away from zero. Use the same starting protocol for the observed and the replicate fits. Let the engine gate's dispersed-start check decide how many. | 3, 5, 6 |
| A4 | **The dose is double-counted in switch months.** A change within a visit's month sums the two doses: 300 → 400 mg gives 700. | `ppmi.py:180` | The treatment terms (dose, exposure, treated, dose change) are wrong at those visits, and so is the share of transitions with a medication change. | Take the dose in force on the visit date, from the last record that starts on or before it. | 1 |
| A5 | **The measurement-error variances may be too large.** Error variance = 1 − ICC (the test–retest reliability) assumes the reliability sample has unit variance on the pooled scale; an early-disease sample varies less (synthetic case: 0.222 against a true 0.087). `configs/measurement_error.json` uses the same conversion. The landmark prior subtracts these variances, with a floor of 1e-3, which can pin the true landmark state. | `ppmi.py:480`, `likelihood.py:86` | Overstated error attenuates the drift estimates and can distort the year-1 prior. | Use error variance = (1 − ICC) × the reliability sample's variance on the pooled scale. Report the margin between each landmark variance and its error variance. Revisit D28 with the panel. | 1, 2 |
| A6 | **The treatment-state coding differs from D24 and §6.1.** "Untreated" comes from PDTRTMNT alone. `on` is 0 at every visit, so the short-duration dose term never enters, not even for daily function, which patients rate while on medication. Exposure is the current levodopa dose, not the effect-compartment exposure. The levodopa name pattern misses L-dopa, Duopa, co-careldopa, co-beneldopa and Dhivy, and matches carbidopa alone. | `ppmi.py:136–139` and the treatment code | The treatment terms partly carry the dose's symptomatic effect, so a wrong specification can leak into the drift. | Implement D24 as written. Code treatment from the medication log. Decide with the panel how daily function is treated. Complete the drug-name list. | 1, 2 |
| A7 | **Quadrature at 12 nodes misses the gate's own rule.** The spread was 0.022 at 113 participants and 0.24 at the real size (816), against 0.01. The pass flag ignores the LR spread, which the docstring says is what must agree; at the real size H-P's LR moved by 0.06. | `gate.py:75, 78, 394` | Pre-registration needs a rule that the engine passes. | Decide on 20 nodes (about 1.7× the compute), or keep 12 and make the LR spread the criterion. Record the choice in Appendix D. | 6 |
| A8 | **Reproducibility.** Step 1's FCI depends on Python's hash seed: causal-learn iterates over sets. In the example, the year-to-year links (H-P's support) were unchanged, but same-year edges and marks changed. A restart keeps the checkpoint only if LR matches to 1e-6, while a change of threads or machine can move it by up to 1e-3. A lower `--n-max` on resume gives a wrong p-value (1/6 where 1/11 is right). The reports record no seed, Step 1 options, node count or code version. | `step1.py` (causal-learn `FCI.py:199`), `run_hp_real.py:187, 219, 236` | A restart can throw away the bootstrap, and the result cannot be traced exactly. | Set `PYTHONHASHSEED=0`, or patch the iteration order. Store the observed fits and the settings in the checkpoint and match on them. Refuse a lower `--n-max`. Record the code version and settings in every report. | 4, 7 |

## B. Localisation, speed intervals, the gate and the qualification

| # | Finding | Where | Proposed fix | § |
|---|---|---|---|---|
| B1 | **H-M1 usually fails even if H-P is calibrated.** With the protocol's replicate numbers, all ten cells pass with probability about 0.41, or about 0.10 once the warp-speed critical value's noise is included. The reported Monte Carlo standard error is about 1.4× too small. | `qualification.py:143, 166`; protocol §8.7 | Revise the decision rule: an upper confidence bound per cell with multiplicity, or a pooled criterion, or more replicates. Correct the standard error. | 6 |
| B2 | **Localisation can never name a link once the family has 26 or more links.** With 2,000 Monte Carlo draws, Benjamini–Yekutieli can never reject a lone link at that family size. Step 1 adds links to the provisional map's 14. | `localise.py` | Use many more draws (for example 20,000), or analytic p-values where valid. | 5 |
| B3 | **The link classes use mixed intervals.** "Structurally different" uses marginal intervals, while the protocol asks for simultaneous ones. The reported `ci90` is marginal while "similar" is judged on simultaneous intervals. `speed_and_load_only` also tests the own rates, while the protocol says "every link". | `localise.py` | Use the protocol's intervals throughout, report the ones used, and settle whether own rates are included. | 5 |
| B4 | **Recalibration (§7.3.3) is not wired.** `recalibrate` has no caller, and `boot_replicate` drops the per-replicate output. | `localise.py`, `hp.py` | Wire it in, or record it as deferred. | 5 |
| B5 | **Crashes.** `per_link_tests` fails when the point speed is identified but no draw is. `bootstrap_speed` fails if every sample fails. | `localise.py:236–243`, `speed.py:134` | Handle both cases and report them as "not identified" or "failed". | 5 |
| B6 | **The gate's speed recovery uses the wrong truth.** It compares γ̂ with log 2, but given the landmark state the engine's γ is about 0.65–0.66, so a perfect engine shows a bias of about −0.03 to −0.04. | `gate.py:201, 206` | Compare with the conditional truth, or judge Sp-1, which targets log 2. | 6 |
| B7 | **The gate's other checks are incomplete.** Only one coordinate per block is profiled; κ_LDR and ω are not, and spreads at zero are not set aside. `time_replicate` drops the replicate's success flag and times one dataset, where §8.8 says 5. Recovery seeds collide beyond 100 datasets. There is no per-entry bias for $A$, loads, noise or κ. | `gate.py:235–248, 263, 374, 385–393` | Complete these before the gate counts as a pass. | 6 |
| B8 | **The qualification is incomplete.** It computes no coverage or standard error for Sp-1 to Sp-3. Several scenarios differ from §8.2's table: S3's TD-only link is 0.91ε, S1T halves only the short-duration κ, and the speed ratio is fixed at 2. Family B silently ignores five options. The simulator's missingness is completely at random, its intervals are jittered, and it has no splines and no ordinal score mapping. | `qualification.py:203`, `scenarios.py:102–110`, `family_*` | Align the scenarios with §8.2, or amend §8.2. Add the performance measures. Build the listed simulator features. | 6 |
| B9 | **Step 1's reported marks can mislead.** causal-learn still blocks arrowheads at the earlier tier, so `<->` never appears between years and a hidden cause can show as `A_t --> C_t1`. A test that cannot be computed returns p = 1, and its edge is removed. The clustered test is asymmetric and too liberal under strong shared heteroscedasticity (9.8% at 5%). The exogenous rule lets era cause sex and age at onset, against §6.2. `depth` does not bound the cost. | `step1.py:43, 81–99, 150–162, 181–186, 210`; causal-learn `FCI.py:294` | Patch the arrowhead rule. Return "untestable" instead of p = 1. Review the clustered test. Align the exogenous rule with §6.2. | 4 |
| B10 | **The WP1 consensus rules differ from §6.4.** Round 2 needs an absolute majority of all raters, so "unsure" counts against. "1–2" is read as strength 1 and improving. A sign tie defaults to worsening. D25's minimum of three raters is not checked. | `wp1.py` | Align with §6.4 before the panel's answers are compiled. | 1 |
| B11 | **Speed intervals.** Basic bootstrap intervals are reflected against the skew of the own-rate draws. The share-explained and speed–load correlation intervals are not clipped to their range. The 90/10 ratio is conditional on the covariates. | `speed.py` | Decide which interval type to use, clip to the range, and state the 90/10 ratio's definition. | 5 |

## C. Data governance

| # | Finding | Where | Proposed action | § |
|---|---|---|---|---|
| C1 | **Suppressed cells can sometimes be recovered.** The "<5" labels plus a grand total can pin a hidden cell, and a lone small cell beside a total can be recovered. Some outputs are published without suppression: `across_a_missed_year`, `at_risk`, the retest `pairs` and `median_gap_days`. | `wp0.py:72–77, 103, 163` | Review the WP0 report, which is now in the public repository. For example, in the PPMI 1.0 participant-flow column the two "under 5" cells must sum to 5. Neither cell is revealed, but this should be judged against PPMI's rules. Then harden `protect_2way`. | 1 |

## D. Protocol and document text to correct

| # | Item | § |
|---|---|---|
| D1 | Protocol §7.1 describes Van Loan blocks and second derivatives by automatic differentiation. The code composes the transitions over binary powers, uses the stationary covariance only when $A$ is clearly stable, and takes the Hessian by central differences of the exact gradient. | 2, 3 |
| D2 | Protocol §2.3 conditions speed and load on the true year-1 state $x_i(1)$. The code conditions on the observed, centred year-1 scores. $\Gamma^y_g$ is diagonal. ED50 and EC50 are fixed at 3.0. | 2 |
| D3 | Named in the protocol but not yet in the code: latent-threshold domains (D26); the treatment-on-drift sensitivity; §7.7 load and noise differences; §7.8 propagation; Step 5; §7.3.2 bootstrap intervals, sample splitting and eBIC; multiple imputation inside the tests; the S-FCI, tiered IOD and BOSS-FCI cross-checks; wrappers for comparators C7 and C10; the speed-matched arm of C5 and C11. | 2, 4, 5 |
| D4 | The "35× slower" claim for unblocked 10×10 products (`linalg.py:3–6`, `code_map.md`) was not reproduced: 2.6× per product and 1.3–1.6× for a whole evaluation. | 2 |
| D5 | The protocol's "10.5% at the base size" is the average of `toy_random_speed.py`'s slow and fast cells, not of `toy_random_speed_scale.py`'s base cells (6.5% and 12.5%). Two toy procedures behind §7.3.2–7.3.3's figures differ from the protocol: joint discovery's BIC counts transitions, not participants, and the anchor-free rule lacks the near-zero and sign filters. | 7 |
| D6 | `event_24m` counts from enrolment to year 2. §12.4 asks for 24 months after a year-2 landmark. | 1 |

## E. Test gaps and minor items

- **Untested branches.** Each chapter's notes list the untested branches. The most important are:
  - `simulate_from_fit`'s draws: the score check, mean z² = 0.97, would make a cheap test;
  - `bootstrap_speed` with the support re-selected, which is the real run's path;
  - `sequential_p`'s resume, and stopping the parallel bootstrap early;
  - the WP1 parsing edge cases;
  - the Step 1 options (`add_interval`, `include_dose`, `reported_map`, `sortability`) and `separate.py`;
  - the localisation extras (`anchored_tests`, `joint_discovery`, `stability_selection`, `commutator_test`, `embeddability_check`);
  - the likelihood without treatment or era, with more than two groups, and with an unstable drift.
- **The reference likelihood shares code with the engine:** the parameter mapping and the year-1 prior. The reference test cannot catch errors in those parts.
- **`test_no_edge_runs_backwards_in_time` cannot fail.**
- **Parallel settings.** `run_tasks` silently caps at 4 jobs and never restores the thread variables; `parallel.ordered` has no cap.
- **Minor settings bugs.** `--wp0 --d 4` gives two domains the wrong missing share; `--d 10`, the planned size, is unaffected. `write_sim_settings` hard-codes a date. The quick gate's output names lack `d`.
- **Small edge cases** are listed in chapters 1 and 3:
  - month-resolution ties;
  - grid windows counted in days;
  - unprorated sums;
  - `NormalScores.transform` on a domain with no observed value;
  - `Fit`'s bookkeeping;
  - interface edges.

## What this means for the exploratory run

The first look and the full PC run use the code as it stands. Items A3 to A6 can move H-P's statistic, and A1 and A2 can make a fit fail silently. **Treat the current PC result as a preliminary pass.** After the fixes, the panel is rebuilt (A4, A6) and H-P is rerun.

## What the PC run showed (5 November 2026)

The full exploratory run on the investigator's Windows PC (`--n-max 199 --speed-bootstrap 100 --jobs 6`; 883 participants, 706 TD and 177 PIGD) produced a report, but no usable result.

**A0, a new item.** Every bootstrap fit stopped at its start.
- **The cause.** The likelihood is summed in batches of participants sorted by number of visits. At least 32 participants are seen only at the landmark. In the single-threaded workers, whose batches hold 32, one batch therefore held only such participants. `participant_loglik` returns a constant for a batch with no transition, and the gradient call then raises an error. `value_and_grad` turned that into 1e10 with a zero gradient (A1), and L-BFGS-B read the zero gradient as convergence. The observed fits ran in the main process, whose batches hold 128, so they were not affected.
- **What it did.** All 199 replicates were exactly 0, so the reported p = 0.005 was empty. In the speed bootstrap, 98 of the 100 samples returned the default start values and 2 the observed estimates. Its "intervals" were those values reflected about the estimates: Sp-1's interval excluded the estimate, and the correlation intervals reached −2.
- **Why the tests missed it.** The simulators keep only participants with two or more visits.
- **Reproduced** on the Mac's copy of the panel: batches of 128 work, and batches of 32 fail with "element 0 of tensors does not require grad".

**What else the observed fits showed** (exploratory, and partly an artefact of the items above; the estimates themselves stay in the local `results/` folder, outside the public repository, until PPMI's publication rules have been followed):
- **Neither fit met the convergence test.** The observed fits ran for 10.6 hours, most likely to the iteration cap, and the free fit's two starts ended far apart.
- **The two fits put PIGD on very different time scales.** The speed-only fit made PIGD faster than TD. The free fit's PIGD drift was a small fraction of TD's, and one scale factor explained most of the difference (Sp-3). So the speed-only fit may have been stuck in the wrong speed basin (A3), which would inflate the statistic.
- **The load collapsed onto the speed.** The speed–load correlation was close to −1 in about half of the domains, with the load's own spread at zero (hence D31).
- **The localisation was unusable.** Its largest deltas were own rates, led by tremor's. It named no link or domain, as B2 predicts, and its speed contradicted Sp-1.
- **Step 1 was unstable.** It re-selected a different support in 98 of the 100 speed-bootstrap samples, so the warm start, which was used only when the support was identical, almost never applied.

## Status after the revision (5 November 2026)

| # | Status |
|---|---|
| A0 | Fixed. `chunks` leaves out participants seen only at the landmark, who add exactly 0, and `value_and_grad` skips a batch with nothing to differentiate. Tests use batches of 32 and 128, and run a two-worker bootstrap on a panel with more single-visit participants than a batch. |
| A1 | Fixed. A start that cannot be evaluated is skipped and recorded. A fit with no evaluable start is failed: log-likelihood −inf, never converged, with the reason given. `lr_stat` is NaN for a failed fit, so H-P counts the replicate as at least as large, and a failed observed fit stops the run. The first five replicates are checked by `hp.degenerate`. The gate counts failed fits. |
| A2 | Fixed. `prepare` stops with the phenotype and domain named. |
| A3 | Fixed. The speed-only fit is restarted from the free fit projected onto it (`free_to_null`), alternating until neither improves by 0.01. The observed speed-only fit also starts with the speed spread at 0.6. Each start stops when it gains less than 0.01 in log-likelihood over 100 iterations, with a cap of 5,000. Replicates follow the same rules from a warm start, written for the support they re-select (`remap_support`). Every fit records its starts and how each stopped. |
| A4 | Fixed. A record that stops in the visit's month gives way to any record starting that month. On the October download, total LEDD was overstated at 325 of 6,254 grid visits, and the share of transitions with a medication change falls from 0.578 to 0.554. |
| A5 | Fixed, provisionally. Error variance = (1 − reliability) × the year-0 variance on the frozen scale, which stands in for the reliability samples' variance (D28 to be confirmed in WP1). Each landmark variance's margin over its error variance is reported, and none is below 0.1. PPMI's own retest now reports the sample variance and the scaled error variance. |
| A6 | Fixed, with daily function provisional. An examination with a dose in force and no recorded state is "treated, state unrecorded" and not used (276 visits on the October download). The levodopa names are completed. The exposure is the levodopa dose (times one plus the COMT multiplier) through a slow effect compartment with a four-week half-life (provisional, D24). Daily function's short-duration term applies whenever the dose is positive (`on_when_treated`, provisional until WP1). |
| A7 | Decided: 20 nodes (D32), checked at the gate against 30 and 40. |
| A8 | Fixed. The run script requires `PYTHONHASHSEED=0`. The checkpoint stores the settings (input hashes, map, seed, protocol constants), the observed fits with their parameters, and every replicate's details, and reuses them only for the same settings. A lower `--n-max` is refused. Reports record the code versions. |
| B5 | Partly. `bootstrap_speed` reports "every sample failed" instead of crashing; `per_link_tests` is unchanged. |
| B7 | Partly. `time_replicate` keeps the replicate's outcome. |
| B11 | Partly. The share-explained and speed–load correlation intervals are clipped to their ranges. |

Also decided on 5 November: **D31**, the load does not move with speed in the primary model (λ = 0), and λ free is a sensitivity analysis.


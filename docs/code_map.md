# Code map: protocol v0.5 to `src/pdx`

Which module implements each part of the protocol, how it has been checked, and what is still to come. Everything here has been run on synthetic data only. No PPMI data have been read.

The code was reviewed on 3 November 2026 by two independent reviewers: one for the statistics of the engine and inference, one for Step 1, the simulators, the harness and data-use safety. Their findings were fixed and are covered by tests (see "Review fixes" below).

Checks are run with the ppmi-causal environment (pytest is not installed, so there is a plain runner):

```
PY="../ppmi-causal/.venv/bin/python"
$PY tests/run_tests.py                 # unit tests (about 2 minutes)
$PY scripts/smoke_test.py --n 300      # the whole chain on one small synthetic dataset
$PY scripts/run_engine_gate.py --quick # the engine-gate harness at small size
```

## Engine (§2.3, §7.1)

| Protocol | Code | Checked by |
|---|---|---|
| Model: dX = s_i[(A + Δ_g)X + b_g + ℓ_i]dt + √s_i G_g dW; log-speed on phenotype, z, year-1 state and a random part with σ_g; the load ℓ_i given u_i | `engine/model.py` (`ModelSpec`, `Layout`; kinds `null`, `free`, `joint`) | `tests/test_engine.py` |
| The load's mean (its dependence on z, the year-1 state and the speed's random part) and its spread are phenotype-specific, so the speed-only model is exactly nested in the free model | `engine/model.py`, `engine/fit.py::null_to` | `test_null_is_nested_in_free_and_joint` (to 1e-8) |
| Era also shifts the noise | `ModelSpec.noise_era_col`; parameter `omega` | reference test |
| Treatment on the observed scores: short-duration Emax (or linear) dose effect with the examination state, long-duration effect of exposure, by phenotype, not scaled by speed (D24) | `engine/model.py::treat_effect` | reference test, both forms |
| Exact transitions over each interval: E = exp(sΔtA), M = ∫exp(Aw)dw, S = ∫exp(Aw)Q exp(A′w)dw | `engine/transitions.py`: E and M are composed over binary powers of a step, with Taylor remainders, so the cost grows with log(interval). S comes from the stationary covariance when A is clearly stable; otherwise it is composed from positive pieces, which needs no stability of A. | `test_grouped_transitions_match_matrix_exponential` (against Van Loan for moderate intervals, against scipy's stationary covariance up to 60), `test_transitions_for_marginal_drift_and_long_intervals`, `test_both_noise_routes_agree` (to 1e-11 up to 80) |
| Likelihood: Gauss–Hermite over log-speed (12 nodes), Kalman filter on [X; ℓ], masked missing values, conditional on the year-1 scores | `engine/likelihood.py` | `test_likelihood_matches_reference`, `test_ten_domain_likelihood_matches_reference`. Both compare against `engine/reference.py`, an independent implementation with Van Loan blocks and scipy's Lyapunov solver, to 1e-8. |
| Exact gradients by automatic differentiation | `likelihood.value_and_grad` | `test_gradient_matches_finite_differences` |
| Maximum likelihood; the free model started at the speed-only fit (so LR ≥ 0) and at its own default (§7.1); the final likelihood from each start is kept | `engine/fit.py` (`fit_model`, `fit_pair`, `hessian`, `covariance_checked`) | `tests/test_fit.py` (recovery of the speed ratio; LR ≥ 0) |
| Conditional parametric bootstrap simulation | `engine/simulate.py::simulate_from_fit` | smoke test; the reviewer's score check (mean score zero over 300 datasets) |

Speed notes:
- Batched products of matrices at or above 8 × 8 go through `engine/linalg.py::bmm`. PyTorch's CPU batched matmul switches to one BLAS call per matrix above a size threshold, which is about 35× slower at ten domains.
- Participants are processed in chunks of similar visit counts, so padding costs nothing.
- The log-speed at the quadrature nodes is bounded at ±15, so a wild trial point in a line search stays cheap.
- Measured at ten domains and about 1,450 synthetic participants on the M4:
  - one likelihood-and-gradient evaluation takes about 2.5 s;
  - the speed-only fit took 223 evaluations: 23 minutes while other jobs shared the CPU, roughly 9 minutes alone;
  - the free fit took 375 evaluations from the null start, roughly 16 minutes alone, before its second start.
- The engine gate times whole fits and replicates (§8.8). This is well above the protocol's assumption of 1 minute per fit; see the README.

Known approximations:
- The plug-in prior for the year-1 state uses the year-1 scores as recorded, so it includes any treatment effect at year 1. Fit and bootstrap use the same prior, so H-P's calibration is unaffected, but κ and the starting state can be biased where many participants are treated by year 1.
- Each replicate's null fit starts at the fitted null (a warm start), while the observed null fit starts at the default. Null fits from different starts agreed in the review; the engine gate's dispersed-start check covers it.

## H-P and the speed estimates (§2.4, §2.5, §7.3.1)

| Protocol | Code | Checked by |
|---|---|---|
| H-P: likelihood ratio of the speed-only and free models, df = (G − 1)(n_sup − 1) | `hp.py::run_hp`, `observed_fits`, `lr_stat`, `h_p_df` | smoke test |
| Conditional bootstrap with support re-selection in every replicate | `hp.py::_replicate` (simulation, selection and refits all inside the failure guard) | smoke test |
| Besag–Clifford sequential stopping (h = 20, up to 999; ties and failed replicates count as at least as large; non-converged replicates are counted and used) | `hp.py::sequential_p`, with checkpoint and resume | `tests/test_hp.py` (stopping rules; uniform null p-values) |
| Warp-speed calibration (one bootstrap replicate per dataset) | `hp.py::warp_replicate`, `qualification.py` | `tests/test_qualification.py` |
| Definition-matched robustness analysis (§7.3.6) | `hp.py::reference_distribution` (takes a generator) | — (the generator fitted to PPMI comes in WP3) |
| Sp-1, Sp-2 (σ, 90/10 ratio, predictors, share explained, speed–load correlation) and Sp-3, with participant-bootstrap intervals | `speed.py::bootstrap_speed`. Intervals are basic, on the log scale for ratios and spreads. The support is re-selected in every sample. Sp-3 is bias-corrected and clipped to [0, 1], with its lower bound. Sp-3 is reported only if ‖A1 − A0‖ exceeds the 95th percentile of its centred bootstrap distribution. | smoke test; `tests/test_fit.py` |

## Localisation (§7.1, §7.3.2–§7.3.4, §7.3.7)

| Protocol | Code | Checked by |
|---|---|---|
| Minimal-L1 speed: weighted median of the entry ratios on the other phenotype's time scale; near-zero reference entries are left out and sign changes never vote. If no entry can vote, the speed is not identified and every verdict says so. | `localise.py::speed_rule`, `speed_rule_batch` | `tests/test_localise.py` |
| Diagnostics: all entries, own rates, cross-links, mode, least squares | `localise.py::speed_diagnostics`, `speed_mode` | `tests/test_localise.py` |
| Monte Carlo draws of the free fit; the set of speeds (draws plus diagnostics); every verdict re-derived at probe speeds across the set | `localise.py::free_draws`, `per_link_tests` | `test_disagreeing_diagnostics_make_verdicts_not_separable` |
| Anchor-free per-link tests (Monte Carlo p-values, Chen Y 2023) | `localise.py::per_link_tests` | smoke test; the reviewer measured 0.5–3.5% rejections at 5% under a pure speed difference |
| Link classes, with simultaneous 90% intervals for "similar" (for whichever family is tested), 90% intervals for the equivalence part of "structurally different", and "speed and mechanism not separable" when a class changes across the speed set | `localise.py::classify_links` | `test_link_classes`, `test_wide_interval_is_unresolved` |
| Intersection–union verdict "speed and load only", at every probe speed | `localise.py::speed_and_load_only` | `test_link_classes` |
| Recalibration of per-link p-values against the H-P bootstrap replicates | `localise.py::recalibrate`; per-replicate p-values via `run_hp(on_replicate=...)` | — |
| Per-domain tests (H-D) with Benjamini–Yekutieli, required at every probe speed | `localise.py::per_domain_tests` | `test_link_classes` |
| Anchored per-link tests (sensitivity) | `localise.py::anchored_tests` | smoke test |
| Speed-adjusted joint discovery: adaptive lasso, BIC with participants. The refit starts from the selected path point and runs only if a deviation fixed at zero sits on a clearly non-zero reference entry (otherwise the speed is not identified). | `localise.py::joint_discovery` | smoke test |
| Stability selection over complementary halves | `localise.py::stability_selection` | — |
| Shared-eigenvector (commutator) check | `localise.py::commutator_test` | smoke test. Caveat from the review: with phenotype-specific state covariances, measurement error alone breaks the commutation of yearly least-squares matrices, so the check may reject too often at realistic reliabilities; WP3 measures it. |
| Complex or negative eigenvalues of yearly matrices (§6.3) | `localise.py::embeddability_check` | — |
| Multiplicity: Holm, BH, BY, simultaneous intervals, IUT | `multiplicity.py` | `tests/test_multiplicity.py` (against statsmodels) |

## Step 1 and the support (§6.2, §6.3, §7.2)

| Protocol | Code | Checked by |
|---|---|---|
| Two-slice table on the annual grid. Consecutive grid years only, using the panel's grid years; steps across a missed year are left out. Rows with missing values are kept for test-wise deletion. Dose tiers optional. | `data.py::two_slice_table`, `step1.py::step1_table` | `test_testwise_deletion_keeps_incomplete_rows` |
| Interval rule: the interval is partialled out of every column, which conditions every test on it | `step1.py::step1_table(add_interval)` | — |
| FCI with tiers; only era may cause age at onset and sex. Tier knowledge puts an arrowhead at the later node and leaves the other mark to the data: causal-learn 0.1.4.8's own step, which sets a tail and would hide hidden confounding, is replaced during each call. | `step1.py::fci_graph`, `_orient_tiers` | `test_hidden_confounding_keeps_a_circle_at_the_earlier_node`, `test_no_edge_runs_backwards_in_time` |
| Tests with test-wise deletion: fast uncorrected Fisher-z, and a participant-clustered, heteroscedasticity-robust test | `step1.py::_register_tests` | `tests/test_step1.py` |
| Support = expert map ∪ lagged adjacencies ∪ own rates, re-applied in every replicate (selected on the H-P panel, so each replicate repeats the same selection) | `step1.py::select_support`, `make_support_fn` | `test_tiers_and_lagged_support` |
| Reported map with marks (clustered test, separating-set cap 4 by default) | `step1.py::reported_map` | smoke test |
| Sortability control | `step1.py::sortability` | — |
| Separate graphs with permutation and generator references, centred and floored standardisation, Westfall–Young threshold | `separate.py` | smoke test |

## Simulation and qualification (§8)

| Protocol | Code | Status |
|---|---|---|
| Family A (engine's class): phenotype from simulated year-0 items; treatment by indication through the engine's Emax forms; dropout; grid years | `sim/family_a.py` | Preset parameters, with 4-, 6- and 10-domain presets. Fitting to PPMI's pooled population waits for data access (WP3). |
| Family B (discrete time, non-linear, drifting burden, saturating treatment, site effects; measurement differing by phenotype and era through a common scale) | `sim/family_b.py` | S0, S1 and S3 analogues. Domain-specific non-invariant loadings are the `noninvariant` stress, because they change the measured mechanism. |
| Scenarios S0, S0L, S1, S1S, S1T, S1V, S4, S7, S2, S2b, S3 | `sim/scenarios.py` | Deviations are sized in ε on the standardised scale: S2 is 2ε; S2b is ε in TD and 4ε in PIGD; S3 has a PIGD-only and a TD-only link. Every scenario gets the same expert map, so df match. S5 and S6 (exploratory Step 3) are not yet written. |
| Stresses | `scenarios.generate` kwargs, through `sim_kw` in the harness | Available: spurious link, missing true link, family B non-invariance, sample size, visits, missingness, dropout. Still to add: MNAR, noisy defining items, era effects, initial state–speed correlation, the identification limit. |
| Warp-speed calibration, H-M1 with Bradley's 7.5%, power against the S1 critical value of the same configuration, Sp-1 bias | `qualification.py`, `scripts/run_qualification.py` | One JSON per dataset, named by a hash of the whole configuration and skipped on rerun; results pooled only within a configuration; at most 4 jobs |
| Fidelity dashboard summaries | `fidelity.py` | Part of the §8.2 list |
| Engine gate: reference agreement, gradients, quadrature nodes, dispersed starts for both models, Hessian conditioning, profiles over ±4 standard errors (speed, spread, load, noise, load spread, speed–load link, treatment), recovery under S1 (speed-only fit) and S2 (free fit), timing | `gate.py`, `scripts/run_engine_gate.py` | Quick mode for the harness. `--wp0` uses WP0's group sizes, visits and missingness (`configs/wp0_sim_settings.json`). Finished sections, dispersed-start fits, recovery datasets and profiles are checkpointed, with one seed per section, so a restart resumes with identical results (`tests/test_gate.py`). Progress goes to `<name>.log`. The report records the commit and a hash of uncommitted changes. The real-size run is a WP2 task. |
| Compute settings: participants per likelihood batch (`CHUNK`) and the layout of jobs and threads | `engine/likelihood.py::CHUNK`, `scripts/bench_threads.py` | Measured on 4 November 2026 (`results/bench/threads.json`). Batches of 128 ran 1.7 times faster than 512, with identical values; 4 single-threaded jobs beat every other layout. |

## Comparators (§8.4)

`comparators/__init__.py::REGISTRY` lists C1 to C12 with their status:
- **Implemented here:** wrappers are in `comparators/run.py`, and `tests/test_comparators.py` checks them.
  - C3: NCT-style permutation test, with per-edge decisions from the maximum standardised difference;
  - C5: the tsLDiffPC/tsDCI re-implementation;
  - C11: separate graphs read naively, and v0.3's interaction tests;
  - C12: the DYNOTEARS re-implementation.
- **Available in the existing environment, wrappers still to write:** C7 and C10.
- **Need software installed with the investigator's approval (D23):** C1, C2, C4, C6, C8 and C9, plus the pyCIPHOD cross-check of C5.

## Data, WP0 and real-data runs (§3, §4, §14)

| Protocol | Code | Status |
|---|---|---|
| PPMI tables (§4, §5, §6.1): the PD cohort, enrolment and genetic status, visits with domain scores (item codes outside range set missing; screening values standing in for baseline), the OFF or untreated examination, LEDD, the annual grid's fixed visit order, Stebbins phenotypes, the provisional clinical subtype, the participant flow, the Part II endpoint | `ppmi.py` (`build`, `annual_grid`, `stebbins`, `clinical_subtype`, `flow_table`), `scripts/build_ppmi_tables.py` | `tests/test_ppmi.py` (a mock download with made-up participants); ran on PPMI on 3 November 2026 |
| The H-P analysis panel: TD and PIGD seen at year 1, the pooled frozen normal scores, centred covariates, dose, exposure, examination state | `ppmi.py::analysis_panel`, `scripts/build_analysis_panel.py` | `test_analysis_panel_from_the_mock` |
| WP0's power gate (synthetic, at WP0's counts, visits and missingness) | `scripts/run_power_check.py`; `family_a.simulate(visit_probs, target_counts, p_missing per domain)`; the settings in `configs/wp0_sim_settings.json` (`--write-settings` refreshes them from the audit) | — |
| Orientation and the frozen pooled normal-scores transform (mid-ranks) | `transforms.py` | `tests/test_transforms.py` |
| WP0 audit: aggregate counts only, with disclosure control | `wp0.py`, `scripts/run_wp0_audit.py` | `tests/test_wp0.py`. Cross-tables are published with their own margins and complementary suppression, shares need denominators of at least `min_cell`, site sizes come in bands, and quantiles need at least 50 values. The interval share counts consecutive grid years only. The script refuses to run without `--approval`. Confirm the threshold of 5 against PPMI guidance. |
| The expert map from the panel's ratings (§6.4): round 1 (unanimity), round 2 (majority; ties to the investigator, recorded as disputed), fewer than five raters marked investigator-led | `wp1.py`, `scripts/wp1_consensus.py` | `tests/test_wp1.py`; an end-to-end check on made-up Word questionnaires. Returned questionnaires stay in `data/wp1/`; outputs are anonymised counts. |
| H-P, the speed estimates and localisation on the prepared PPMI panel | `scripts/run_hp_real.py` | Refuses to run without `--approval`. Reads only `data/` and writes only aggregates to `results/`. The bootstrap is checkpointed after every replicate. Map links outside the panel's domains (treatment, domains left out under D26) are set aside and reported; an unknown domain name stops the run. |
| Panel files (no pickling) | `data.py::save_panel`, `load_panel` | — |
| Pre-specified constants | `configs/protocol_v0.5.toml`, `config.py` | — |

`.gitignore` excludes `data/`, `results/` and participant-level file types wherever they appear (`*.npz`, `*.npy`, `*.parquet`, `*.feather`, `*.csv`, `*.pkl`).

## Review fixes (3 November 2026)

Engine and inference:
- Sp-3's reportability rule now uses a centred bootstrap test. The old rule passed almost always, even without a difference.
- The transitions are composed over binary powers with S built from positive pieces, which fixed two faults:
  - an unbounded cost at large speeds;
  - silent failure of S for drifts with eigenvalues summing to zero.
- Overflow errors are caught, and log-speed is bounded at the nodes.
- An unidentifiable speed now yields "speed not identified" instead of the smallest p-value for every link.
- "Similar", the class conditions and H-D are now checked across the speed set, with 90% intervals for equivalence.
- Speed intervals are on the log scale, every Sp-2 component has an interval, and the support is re-selected in the speed bootstrap.
- Hessian failures raise, and covariance conditioning is reported.
- A replicate's simulation and support selection are inside the failure guard.

Harness and data safety:
- WP0 disclosure control: complementary suppression and derived margins.
- Qualification results are keyed by configuration.
- Family A now simulates the engine's Emax treatment.
- Family B's null is invariant on the measured scale.
- Step 1: hidden-confounding marks are kept, and missing values are handled by test-wise deletion.
- The gate's changed-link recovery is checked against the free fit.
- Scenarios: S2, S2b and S3 match the table, and every scenario gets the same expert map.
- Separate-graph standardisation; C3's per-edge test; grid-year transitions; interval partialling; checkpointing; valid JSON for failed replicates; version pins; `.gitignore` patterns.

## Not yet in code

- **Fitting the simulators to PPMI's pooled population** (§8.2) and the definition-matched generator (§7.3.6). Both need approval. The plan is in `docs/wp3_simulator_fitting_plan.md`.
- **The second host (ctsem) and the continuous-time check (§7.12).** They need R packages (D23).
- **Latent-threshold domains (D26)**, if the panel keeps a floor-bound domain.
- **Exploratory Steps 3 and 4** (mixtures, random links), and S5/S6.
- **Part II.**

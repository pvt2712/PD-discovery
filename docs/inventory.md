# Inventory: the pasted design against earlier work

*29 October 2026. Read-only inventory of `ppmi-causal/`, `escalation-study/` (v1), `escalation-study-v2/` and
`levodopa-escalation/` against the pasted design. Aggregates only; no participant-level file was opened. Every
number quoted from the background notes was re-checked against the result files named here.*

## 1. Inventory: design item → what exists

Status: **reuse** (run as is), **port** (copy and extend), **new** (nothing exists).

### Arm A (recoverability, expert DAG as truth)

| Design item | What exists | Status |
|---|---|---|
| A1 fit each node from its expert parents on real PPMI | `ppmi-causal/src/ppmi_causal/plasmode.py:simulate` is hand-set linear-Gaussian (coefficients "chosen from" within-person estimates, a v2-style 13-node graph, all continuous). v1 `plasmode.py` plants *effects* in real records for the estimators, not structure. Nothing fits the v5.1 parents. | **new** |
| A2 PPMI n, types, real missingness | `plasmode.simulate`: N × T panel, person random intercepts, latent trait, ~8%/yr dropout tied to disability, OFF-motor missing 40% on high dose. No real marginals, no ordinal/binary types, no copied patterns. | **port** the panel logic; types and missingness **new** |
| A3 second generator (VAMBN) | Only cited (`paper2_baselines.md` §4: Gootjes-Dreesbach 2020, PPMI 362 de novo). VAMBN needs R (bnlearn) + an HI-VAE in TensorFlow 1.x. | **new** (see D4) |
| A4 flag near-zero expert edges | Real-data edge support for v5.1 (`expert_dags.yaml` notes "unsupported" on 12 edges, e.g. ldopa → adl, apoe4 → cog, dur0 → motor_na "no variance in a de novo cohort"). No generator-side strength. | **new** |
| A5 stress: hidden vars, n 300–4,500, raw vs standardised, MNAR | Latent-trait recovery and a no-latent control (`plasmode.run` (b)/(c)); test-size sweep 50–400 patients (`results/validation/ci_size_sweep.csv`, tests only); units on *real* data (`ml_checks.py`: raw / z / copula). No MNAR, no n axis for whole learners. | mostly **new** |
| A6 metrics | `plasmode.discovery_recall`: pattern-level adjacency recall 0.44 (lagged 0.20, same-year 0.64, baseline 0.44), precision 0.78, 3.4 false/rep (10 reps, N 400 × T 8) — **verified** in `results/plasmode/discovery_recall.json`. No arrowhead-vs-PAG, SHD, SID or parent-set metric; no oracle PAG. | adjacency **port**; rest **new** |

### Arm B (real PPMI: agreement and plausibility)

| Design item | What exists | Status |
|---|---|---|
| B1 test the expert DAG first (dagitty `localTests`) | `ppmi-causal/src/ppmi_causal/expert.py`: unrolls a summary DAG to the two-slice layout, local Markov statements with the latent trait handled by d-separation, clustered GCM, falsification vs within-tier relabellings, out-of-sample cohorts, DoWhy reproduction. v5.1: 24.8 / 28.0 / 22.4% violated vs 38.6 / 36.0 / 33.4% relabelled, p 0.0099, 701 statements. v4.1 cross-fitted 23.3% vs 36.3%, p 0.020 (**verified**, `results/expert/v41_cf_core.json`). v1 `dag.py`: exact null over all 431 time-respecting relabellings. R/dagitty not needed. | **reuse/port**; rerun with the cross-fitted test as primary |
| B2 ≥ 100 resamples, keep stable edges | `discovery.bootstrap` (participant bootstrap, B 100), `discovery.cpss` (complementary pairs + noise bound), v1 `dt_discover` (50 half-samples). `plasmode.bootstrap_calibration`: frequency ≥ 0.8 true only 76%. Only for FCI/PC. | **port** to every method |
| B3 agreement, not accuracy | `audit.py` (confirmed / latent-consistent / data-only / supported-not-discovered / weak / contradicted; v4.1: 9 / 2 / 2 / 24 / 7 / 5). | **port** to v5.1 |
| B4 negative controls without knowledge | `gradient_baselines.violation_stats` (backwards-in-time share, arrows into age/sex/APOE/time). Real-data rates on the 13-node core set (`paper2_baselines.md` §2): PC 73.1%, NOTEARS 60.4%, FCI 52.9%, GOLEM 49.6%, BOSS 43.5%, DAG-GNN 30.8%. | **reuse** metric; apply to the whole panel on the v5.1 nodes |
| B5 blinded clinician sort | `docs/clinician_review_v41.md` (an unblinded review of v4.1, never sent). | **new** (blinded sheet) |

### Methods panel

| Method | Status here |
|---|---|
| PC, FCI, tiered and untiered | **reuse**: `discovery.py` (`tiered_fci` with the tier-semantics patch, Bang & Didelez 2025; `tiered_pc`; `untiered`), causal-learn 0.1.4.8; tests `fisherz`, `cluster_fisherz`, `cluster_gcm(_lin)(_cf)` (`citests.py`) |
| BOSS | **reuse** untiered (causal-learn, BIC, complete cases); causal-learn's BOSS takes no tiers |
| BOSS-FCI, GFCI, FCIT, GRaSP-FCI; degenerate-Gaussian test/score | **absent**: need Java (Tetrad). Flagged as "your decision" since 14 Oct (`advanced_methods_review.md` §10, `ml_causal_discovery_review.md` §4.3) |
| NOTEARS, DAGMA (+ GOLEM, DAG-GNN) | **reuse**: `ml_checks.py`, `gradient_baselines.py` (gCastle 1.0.4, dagma 1.1.1, torch 2.14) |
| sortnregress; varsortability | varsortability and R²-sortability **reuse** (`ml_checks.sortability`: 0.62 / 0.72 raw); sortnregress **new** (a few lines) |
| SDCD / DCDI; DiBS / BayesDAG; DECI; AVICI, Arrow, CDFM | **none** (reviewed only; `advanced_methods_review.md` notes Rhino and N-ADMG were removed from causica) |
| Longitudinal: tiered FCI by visit; J-PCMCI+ | **reuse**: both run (`results/jpcmci_core`, `lpcmci_core`; tigramite 5.2.10.1); J-PCMCI+'s participant dummies had no power |
| Ensemble (majority vote) | **none** ("the ensemble row of the learner grid" is still open, `project_summary.md` §11) |
| LLM prior | **none**; `ml_causal_discovery_review.md` §2.3 argues against |

### Data and rules

- **Data build**: `escalation-study-v2/src/escalation_v2/build.py` (newest: ST visits kept) writes `subjects / visits / medlog / grid.csv`; its grid carries every v5.1 node. PPMI 1.0 de novo 431 participants, 3,825 grid rows (8.9 each); PPMI 2.0 sporadic 957, 2,934 (3.1); genetic PD 353, 2,070 (5.9) (`results/build_counts.csv`). OFF motor missing 30% of PPMI 1.0 grid rows, axial 28%, ON motor 33%.
- **No CSF/plasma**: the local PPMI download has no biospecimen category, so the design's CSF nodes are impossible without a new data request.
- **Pre-registration**: v2 `DESIGN.md` §4.1 pre-specifies reading rules for inference only; nothing pre-specified for discovery. Tuning on synthetic data only, equal budgets and a decision rule are **new**.
- **Variables**: v5.1 = 17 time-varying + 6 baseline nodes + latent trait, 75 edges (45 same-year, 7 lagged, 15 baseline, 8 trait); the two-slice table has 41 observed nodes + time.

## 2. Decisions taken on 29 October 2026

| Decision | Recommended | Taken |
|---|---|---|
| Expert DAG | v5.1 (`v51_escalation`); v2's core map tested in Arm B only | as recommended |
| Cohort and visits | PPMI 1.0 de novo primary, PPMI 2.0 and genetic PD as replication; annual grid | **pooled PD cohorts** (PPMI 1.0 + 2.0 + genetic, 1,741 participants) with two cohort context nodes; annual grid; PPMI 1.0 alone and leave-one-cohort-out as sensitivities |
| Location and boundary | new sibling repo; Q1 audit and the core-map falsification move to discovery; `validity.py` stays in v2 | as recommended (`paper 2/escalation-discovery/`; v2 untouched) |
| Installs | Java (OpenJDK 21 + Tetrad 7.6.10); Python deep-learning side environments; no R | as recommended |

## 3. What changed while building

- **Truth with hidden history.** A latent person effect on every node plus hidden earlier years makes 755 of 903
  pairs adjacent in the true MAG: dependences that are real but weak, so a complete graph would be a useless truth.
  The scored truth is the two-slice DAG with the latent trait; the 46 scored pairs that only hidden earlier years
  connect are left out, and per-node person effects became a stress setting (PROTOCOL.md §6.4).
- **One clustered test.** Discovery and falsification both use v2's `citest.ClusterGCM` (the 28 October corrections:
  quantile knots, an untestability guard); ppmi-causal's older copy is not used.
- **Negative controls.** A bidirected cross-tier edge encodes confounding, not reversed time, so only later -> earlier
  orientations count as backwards (PROTOCOL.md §7.4).
- **Tetrad memory.** BOSS-FCI and GRaSP-FCI need a JVM heap above 2 GB on these graphs (6 GB default) and run in a
  separate low-parallelism pass.

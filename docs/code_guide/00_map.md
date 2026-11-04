# The phenotype-discovery study: a guide to the code and its mathematics

*A companion to `PROTOCOL.md` (what the study does) and `docs/RUNBOOK.md` (how to run it). This guide explains how the code does it. It covers every module in the order the pipeline runs, every function or block, the mathematics behind each step, and a worked example for each. It was written on 4 November 2026 against commit `a9b577f`; the line numbers in the links are that commit's.*

**How to read it.** This chapter (§0) is the map: the question, the pipeline, the data flow, the shared notation and the four ideas everything else rests on. Chapters 1–7 follow the pipeline. Each function has the same five parts:
- *what it does*;
- *inputs → outputs*;
- *the mathematics*, derived and then mapped onto the code line by line;
- a *worked example*;
- *design choices and pitfalls*.

The worked examples use small synthetic numbers. Where a function can run on synthetic input, the example was actually run with the code in this repository and its output pasted as printed. **No example uses PPMI data.** Under the data-use agreement, participant-level tables stay in `data/derived/`, and this guide never opens them. Each chapter ends with *Notes for the author*: things noticed while reading the code that are worth checking.

| If you want to understand… | read |
|---|---|
| how the raw PPMI tables become the analysis panel, and the WP0 audit and WP1 consensus rules | [§1 The data layer](01_data.md) |
| the continuous-time model and its likelihood: the matrix exponential, the Kalman filter, the quadrature over speed | [§2 The engine I](02_engine_likelihood.md) |
| how the models are fitted, their standard errors, and simulation from a fit | [§3 The engine II](03_engine_fit.md) |
| Step 1's shared causal map and the named comparison methods | [§4 Step 1 and the comparators](04_step1_comparators.md) |
| the primary test H-P, the speed estimands and the localisation of differences | [§5 H-P, speed and localisation](05_hp_speed_localisation.md) |
| the simulators, the method qualification (H-M1) and the engine gate | [§6 Simulators, qualification and the gate](06_simulation_gate.md) |
| the run scripts, what each test proves, and the toy simulations behind the design | [§7 Scripts, tests and toys](07_scripts_tests.md) |
| **everything the readers flagged, ranked by how much it could change a result, with proposed fixes** | [§8 What the readers found](08_audit_findings.md) |

**The example scripts.** Every script that produced a printed output here is in [`docs/guide_examples/`](../guide_examples/), one folder per chapter (`s00` … `s07`), and uses synthetic data only. Run one from the repository folder, for example:

```
OMP_NUM_THREADS=1 ../ppmi-causal/.venv/bin/python docs/guide_examples/s00/ex_speed_vs_pathway.py
```

## 0. The map

### 0.1 The question as an estimand

The study asks whether patients with tremor-dominant (TD) and with postural instability and gait difficulty (PIGD) Parkinson's disease progress through **one shared mechanism at different speeds**, or through **different mechanisms**.

- **The mechanism** is the matrix $A$ of directed temporal effects between ten symptom domains. $A_{jk}$ is the effect of domain $k$'s level on the rate at which domain $j$ worsens, per year at the reference speed. The diagonal holds each domain's own rate (its persistence), which is kept negative.
- **H-P, the only primary test** (protocol §2.4). It compares two nested models fitted to the same data:
  - **the speed-only model (H-P's null):** both phenotypes run one mechanism $A$, PIGD at its own speed $e^{\gamma}$;
  - **the free model (the alternative):** each phenotype has its own drift on the support, $A_0$ for TD and $A_1$ for PIGD.

  The statistic is the likelihood ratio $\mathrm{LR}=2(\hat\ell_{free}-\hat\ell_{null})$, on $(G-1)(n_{sup}-1)$ degrees of freedom. Its p-value comes from a conditional parametric bootstrap that repeats the support selection in every replicate (§5).
- **The speed estimands** (key secondary, protocol §2.5):
  - Sp-1, the speed ratio from the speed-only model;
  - Sp-2, the spread of speed within each phenotype and its predictors;
  - Sp-3, the share of the difference between the phenotypes' drifts that one speed explains.
- **Localisation** says which links differ beyond speed (§7.3.3 of the protocol). It needs one identifying assumption, the minimal-L1 speed (§0.5, idea 3).

### 0.2 The pipeline at a glance

| Work package | Script | Main modules | Output |
|---|---|---|---|
| WP0 tables | `scripts/build_ppmi_tables.py` | `ppmi.build`, `wp0.protect_2way` | `data/derived/*.parquet` (participant level); `results/wp0/participant_flow.json`, `manifest.json` (aggregates) |
| WP0 audit | `scripts/run_wp0_audit.py` | `wp0.audit` | `results/wp0/wp0_audit.json` |
| WP0 power gate | `scripts/run_power_check.py` | `qualification`, `sim` | `results/wp0/power/summary.json` (synthetic) |
| Analysis panel | `scripts/build_analysis_panel.py` | `ppmi.analysis_panel`, `transforms` | `data/derived/panel_tdpigd.npz`, `panel_spec.json` |
| WP1 consensus | `scripts/wp1_consensus.py` | `wp1` | `results/wp1/*.json`, `configs/expert_map_draft.json` |
| WP2 engine gate | `scripts/run_engine_gate.py` | `gate`, `engine`, `sim` | `results/engine_gate/*.json` (synthetic) |
| WP3 qualification | `scripts/run_qualification.py` | `qualification`, `sim`, `hp`, `comparators` | `results/qualification/` (synthetic) |
| WP4 H-P on PPMI | `scripts/run_hp_real.py` | `step1`, `hp`, `speed`, `localise`, `parallel` | `results/wp4/hp_first_look.json` or `hp_tdpigd.json` (aggregates) |

Every script that reads PPMI participant data refuses to start without `--approval "<who, date, document>"` (`scripts/_common.py`, protocol §14).

### 0.3 The data flow in one picture

```
PPMI download (raw CSVs, 15 tables in 5 folders)
   │  ppmi.build: eligibility, phenotype (Stebbins 2013), domains oriented higher = worse,
   │              the annual grid, LEDD, treatment state
   ▼
data/derived/participants.parquet, grid.parquet, wp0_visits.parquet      (participant level)
   │  ppmi.analysis_panel + transforms: TD and PIGD from the year-1 landmark,
   │  frozen normal-scores transform, treatment terms, covariates
   ▼
data/derived/panel_tdpigd.npz + panel_spec.json                           (participant level)
   │  step1.select_support → support S = expert map ∪ Step 1's lagged links ∪ own rates
   │  engine: speed-only and free fits (likelihood: matrix exponential, Kalman filter,
   │          Gauss–Hermite quadrature over speed)
   │  hp: LR, conditional bootstrap (Besag–Clifford), speed: Sp-1..3, localise: links
   ▼
results/wp4/*.json                                                        (aggregates only)
```

### 0.4 Shared notation

| symbol | meaning | code |
|---|---|---|
| $i$; $g\in\{0,1\}$ | participant; phenotype group (0 = TD, the reference; 1 = PIGD) | a row of `Panel`; `panel.group` |
| $d$; $j,k$ | number of domains (10 in the analysis); domain indices | `spec.d` |
| $t_{i1}<\dots<t_{iT_i}$; $\Delta t$ | visit times in years since enrolment, from the year-1 landmark on (so they start near 1; only differences enter the model); an interval between visits | `panel.t[i]`; `prep.dt` |
| $Y_{ik}\in\mathbb R^d$ | observed domain scores at visit $k$, on the frozen normal-scores scale, oriented so that higher = worse; NaN = missing | `panel.y[i]` |
| $X_i(t)$ | true (latent) domain levels | |
| $A$; $A_g$; $A+D_g$ | drift: one mechanism (speed-only model); one per phenotype (free model); reference plus deviation (joint model, for localisation). Own rates are $A_{jj}=-e^{\text{raw}}<0$ in $A$ and $A_g$; the joint model's deviations $D_g$ are unconstrained, so an own rate of $A+D_g$ can turn positive | blocks `A`, `A{g}`, `D{g}`; `drifts()` |
| $\mathcal S$; $n_{sup}$ | the support (allowed drift entries; the diagonal always) and its size | `spec.support` (stored as [to, from]); `spec.n_sup`; `spec.sup_idx` |
| $s_i$ | participant speed, $\log s_i=\gamma_g+\beta^\top z_i+\theta^\top\tilde y_{i1}+\sigma_g u_i$ | `gamma`, `beta`, `theta`, `log_sigma`; `params["log_speed"]` |
| $z_i$; $\tilde y_{i1}$ | centred baseline covariates (era, age at onset, sex, disease duration); the centred *observed* year-1 scores, with missing values filled (the protocol writes the true state $x_i(1)$; see §2's notes) | `panel.z`; `y1c` |
| $u_i\sim N(0,1)$; $u_m,w_m$ | random part of log-speed; Gauss–Hermite nodes and weights, $m=1,\dots,K$ ($K=12$) | `gh_nodes(K)`; `spec.n_quad` |
| $b_g$ | phenotype load: a steady push on each domain | `b` |
| $\ell_i$ | participant load, $\ell_i\mid u_i\sim N(\gamma^y_g\odot\tilde y_{i1}+\Gamma^z_g z_i+\lambda_g u_i,\ \mathrm{diag}(\tau_g^2))$; $\gamma^y_g$ is a vector applied element by element ($\odot$), $\Gamma^z_g$ a $d\times p$ matrix, $\tau_g$ the load spread | `gam_y` (G, d), `gam_z` (G, d, p), `lam`, `log_tau` |
| $Q_g=G_gG_g^\top$ | noise covariance, multiplied by $e^{\omega\,\mathrm{era}_i}$; era is centred in $z$, so $Q_g$ is the noise at the mean era | `L` (Cholesky factor), `omega` |
| $h_{ik}$ | treatment terms in the measurement equation (short-duration Emax of dose with the examination state; long-duration effect of recent exposure) | `kappa`, `kappa_ldr`, `treat_mask`, `ed50`, `ec50_ldr` |
| $\Sigma_e=\mathrm{diag}(\sigma^2_{e,j})$ | measurement-error variances, fixed from published reliabilities (D28) | `spec.meas_var` |
| $E,\ M,\ S$ | over one interval: transition matrix, intercept, noise covariance | `transitions_grouped` |
| $\hat\ell_{null},\hat\ell_{free}$ | maximised log-likelihoods | `Fit.loglik` |
| LR; df | $2(\hat\ell_{free}-\hat\ell_{null})$; $(G-1)(n_{sup}-1)$ | `hp.lr_stat`; `hp.h_p_df` |
| $h$; $n_{max}$ | Besag–Clifford stopping count (20); cap on replicates (999) | `configs/protocol_v0.5.toml` `[hp]` |
| $\hat s_0,\ \hat s_{LS},\ \hat s_{L1}$ | speed-only, least-squares and minimal-L1 speeds. Sp-1 is $\hat s_0=\exp(\hat\gamma_1+\hat\theta^\top\Delta\bar y)$, the speed-only model's ratio with each phenotype at its own average landmark scores ($\Delta\bar y$ = PIGD's mean minus TD's); the free-fit speeds $\hat s_{LS}$ and $\hat s_{L1}$ estimate $e^{\gamma_1}$, so they differ from Sp-1 even under the null (§5) | `speed.sp1`; `speed.sp3`; `localise.speed_rule` |
| $\Delta_{jk}$ | speed-adjusted deviation of a link, $a_1/s-a_0$ | `localise` |
| $\varepsilon$ | smallest effect of interest: 0.10 SD of the target per SD of the source per year (provisional, D12) | `[localisation] eps` |

**The model.** A participant $i$ in phenotype $g$ has true levels that follow the linear stochastic differential equation

$$
dX_i=s_i\big[A_gX_i+b_g+\ell_i\big]\,dt+\sqrt{s_i}\,G_g\,dW_i,
$$

and is observed at the visits as $Y_{ik}=X_i(t_{ik})+h_{ik}+e_{ik}$ with $e_{ik}\sim N(0,\Sigma_e)$. Speed multiplies everything that moves: the drift, the load and the noise variance. A phenotype that runs the reference mechanism faster therefore moves along the same paths in less time.

### 0.5 The four ideas everything rests on

**1. Speed masquerades as mechanism in yearly data.** If PIGD runs TD's mechanism twice as fast, its continuous-time drift is exactly $2A$. Every entry doubles, so in continuous time the difference is one number. Its yearly transition matrix, however, is $e^{2A}$, not $e^{A}$ scaled, and different entries change by different factors. The example, [`ex_speed_vs_pathway.py`](../guide_examples/s00/ex_speed_vs_pathway.py), uses a motor sign (domain 1) that drives daily function (domain 2):

```python
A = np.array([[-0.50, 0.00],
              [0.30, -0.40]])
yearly_td, yearly_pigd = expm(A), expm(2.0 * A)
```

```text
yearly transition, TD:
 [[0.6065 0.    ]
 [0.1914 0.6703]]
yearly transition, PIGD (same A, speed 2):
 [[0.3679 0.    ]
 [0.2443 0.4493]]
entry-by-entry ratio PIGD / TD:
 [[0.6065    nan]
 [1.2769 0.6703]]
continuous-time drift ratio (what the engine compares): 2.0
```

A comparison of yearly coefficients would report two things:
- a 28% stronger motor → function link in PIGD;
- a weaker persistence of both domains (×0.61 and ×0.67).

That looks like a different mechanism, but it is one mechanism run faster. This is why the engine models speed explicitly. It works in continuous time and compares "one $A$ at two speeds" against "two $A$'s" (protocol §1.3; `docs/pathway_heterogeneity_framework.md`).

**2. One continuous-time engine, exact over the actual intervals** (§2, §3).
- The transition over an interval of length $\Delta t$ at speed $s$ is $E=e^{sA\Delta t}$. The intercept and the noise covariance follow from the same exponential: the code composes them over binary powers, and takes the noise from the stationary covariance when $A$ is clearly stable.
- Given the speed, everything is linear and Gaussian. The participant load, the measurement error and any missing domain are integrated out exactly by a Kalman filter.
- What remains is a one-dimensional integral over the random part of log-speed, done by Gauss–Hermite quadrature: $\int f(u)\phi(u)\,du\approx\sum_m w_m f(u_m)$.
- Maximum likelihood uses exact gradients from PyTorch's automatic differentiation.

**3. H-P does not need to split speed from deviations; localisation does** (§5).
- In the free model, multiplying PIGD's speed by a constant and dividing its drift by the same constant leaves the fit unchanged.
- The likelihood ratio compares the best proportional fit with the best free fit, whatever the split, so H-P needs no assumption.
- Saying *which* link differs does. The code takes the speed as the weighted median of the entry ratios $a_1/a_0$, weighted by $|a_0|$: the minimal-L1 speed $\arg\min_s\sum|a_1-s\,a_0|$.
- That speed is right as long as the entries that deviate in one direction carry less than half of the weight. A verdict is reported only if it holds across the whole set of speeds the data cannot tell apart.

**4. The bootstrap repeats everything that used the data** (§5).
- The support $\mathcal S$ is selected on the data: the expert map's links, plus the lagged links Step 1 finds, plus the own rates.
- The bootstrap therefore simulates from the fitted speed-only model, re-selects the support, and refits both models in every replicate. Otherwise the selection would make the test anti-conservative.
- The p-value is Besag–Clifford's sequential one. It stops at the $h$-th replicate whose statistic reaches the observed one, $p=h/n$, or else runs to the cap, $p=(e+1)/(n_{max}+1)$.
- With `--jobs N` the replicates run in parallel, each with its own seed, and the rule is applied in replicate order (§7, `parallel.py`).

### 0.6 Conventions used throughout the code

- **Aggregates only outside `data/`.** Participant-level tables live in `data/derived/` (git-ignored). Everything written to `results/` is counts, estimates and diagnostics.
- **One scale.** Every score is put on one normal-scores scale, computed once on the pooled primary population with mid-ranks for ties and then frozen. Simulated and bootstrap data are generated on that scale.
- **Orientation.** Every domain is oriented so that higher means worse; independence is 100 minus Schwab & England, cognition is 30 minus the MoCA.
- **Seeds.** Every section of the engine gate, every bootstrap replicate, speed sample and recovery dataset has its own seed. So a resumed or parallel run gives the same result as an uninterrupted serial one: bit for bit across any number of parallel workers, and to the fits' numerical tolerance against a single in-process job.
- **Checkpoints.** Long runs save each finished piece (`results/engine_gate/*.partial.json`, `results/wp4/hp_tdpigd_checkpoint.json`) and skip it when restarted.
- **Batch size.** The likelihood is evaluated over batches of participants: 32 in a single-threaded worker, 128 otherwise (`likelihood.default_chunk`). The batch size changes only the order of summation.

### 0.7 A suggested order for auditing

1. **§2, the engine I.** Everything rests on the likelihood. Check its tests first: the reference likelihood agrees to 1e-8, and the gradient check passes.
2. **§3, the engine II.** Check that the speed-only model is nested in the free model, so that LR ≥ 0.
3. **§5, H-P, speed and localisation.** The primary test and the estimands.
4. **§4, Step 1.** The support that H-P tests on.
5. **§6, the simulators and the gate.** How the method is qualified before it is used.
6. **§1, the data layer.** How PPMI's tables become the panel.
7. **§7, the scripts and tests.** How it is all run, and what the tests do and do not prove.

The *Notes for the author* at the end of each chapter collect what the readers flagged: possible bugs, untested branches and differences between the code and the protocol. [§8](08_audit_findings.md) ranks them all in one list.

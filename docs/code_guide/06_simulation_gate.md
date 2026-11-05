## 6. Simulators, method qualification and the engine gate

> **Changed on 5 November 2026** (D31, D32; [§8](08_audit_findings.md)). This chapter describes commit `a9b577f`. Since then:
> - the gate and the qualification take the analysis model's settings from `configs/protocol_v0.5.toml`: 20 nodes, λ = 0, the `[fit]` rules;
> - the gate checks quadrature at 20, 30 and 40 nodes, counts failed fits, and keeps the timed replicate's outcome.

*This chapter covers `src/pdx/sim/` (`family_a.py`, `scenarios.py`, `family_b.py`), `qualification.py`, `fidelity.py`, `parallel.py` and `gate.py`. They implement protocol §8 (the simulation study of WP3, the third work package) and §16 (the engine gate, run before registration). Line numbers are those of commit `a9b577f`. Every example below runs on synthetic data only; none reads `data/` or `results/`.*

### 6.0 Overview

**Why these modules exist.** Before H-P is run on PPMI, two questions have to be answered on data whose truth is known.
1. **Does the engine compute what it claims, and can its parameters be recovered?** This is the *engine gate* (protocol §16). It compares the likelihood with an independent implementation, checks the gradient, the quadrature and the optimiser, profiles the key parameters, and fits datasets simulated under a shared mechanism (S1) and under one changed link (S2).
2. **Does H-P keep its level when the mechanism is shared, and what is its power?** This is the *method qualification*, whose decision rule is H-M1 (protocol §8.7): H-P qualifies if its false-rejection probability is at most 7.5% (Bradley's liberal criterion, 1.5 times the nominal 5%) in every shared-mechanism scenario of both simulator families.

**The study in ADEMP form.** Protocol §8 follows Morris et al. (2019): aims, data-generating mechanisms, estimands, methods and performance measures.

| ADEMP | protocol | what the code does | where |
|---|---|---|---|
| **A**ims | §8.1 aims 1, 2 and 5; §16 | H-P's false-rejection probability under shared mechanisms and its power; the bias of the speed estimates; the cost of a fit and of a bootstrap replicate; the engine's correctness and identifiability | `qualification.py`, `gate.py` |
| **D**ata-generating mechanisms | §8.2 | two independently coded simulator families; eleven named scenarios; a few stresses | `sim/family_a.py`, `sim/family_b.py`, `sim/scenarios.py` |
| **E**stimands | §8.3 | the true speed ratio $s$, the true drifts $A$ and $A_{\text{PIGD}}$, the speed spreads $\sigma_g$; whether H-P's null holds (it does in the eight shared-mechanism scenarios, not in S2, S2b and S3) | `Truth` fields, `info["speed_ratio"]` |
| **M**ethods | §8.4 | H-P with Step 1's support re-selection and one bootstrap replicate per dataset (the warp-speed method); Sp-1 and Sp-3; the engine's speed-only and free fits | `qualification.dataset_task`; `gate.recovery_dataset` |
| **P**erformance measures | §8.5, §8.6 | rejection rate with its Monte Carlo standard error (MCSE) $\sqrt{p(1-p)/R}$; Bradley's 7.5%; power; bias and root-mean-square error (RMSE) of log Sp-1; in the gate, bias, RMSE and coverage of the log speed ratio and the coverage of the log speed spreads | `calibration_summary`, `h_m1`, `power_summary`, `speed_summary`; `gate.recovery` |

**How the pieces fit.**

```
scenarios.generate(scenario, n, rng, family, d, **stresses)
   ├─ family A: truth_for(scenario) ─► family_a.simulate(truth, ...)      ─┐  Panel (from the year-1 landmark)
   └─ family B: family_b.simulate(...)                                     ─┘  + info: support, truth, speed ratio
        │
        ├─ qualification.dataset_task   Step 1 support ─► fit_pair ─► LR; one bootstrap LR*; Sp-1; Sp-3
        │     make_tasks ─► run_tasks (one JSON per dataset, ≤ 4 spawned workers)
        │     ─► calibration_summary ─► h_m1 (the verdict), power_summary, speed_summary
        │
        ├─ fidelity.panel_summaries / compare      the same aggregates on real and simulated panels
        │
        └─ gate.run_gate     reference · gradient · quadrature · dispersed starts · identification ·
                             profiles · recovery (S1, S2) · timing
                             Checkpoint after every piece; parallel.ordered for independent pieces
```

**Family A in one place.** Family A is "the engine's class": its dynamics are the engine's linear stochastic differential equation, with a few deliberate departures (§6.6–§6.8). Participant $i$ with phenotype $g_i\in\{0,1\}$ (0 = TD, the reference; 1 = PIGD) has true domain levels $X_i(t)$ with

$$
dX_i=s_i\big[A_{g_i}X_i+b_{g_i}+\ell_i+c\tanh(X_i)\big]dt+\sqrt{s_i\,q}\;dW_i,\qquad A_0=A,\ A_1=A_{\text{PIGD}},
$$

$$
\log s_i=\log m_{g_i}+\sigma_{g_i}u_i+\beta_{\text{age}}\,\text{age}_i,\qquad \ell_i=\lambda u_i\mathbf 1+\tau_{g_i}\epsilon_i,\qquad u_i\sim N(0,1),\ \epsilon_i\sim N(0,I_d),
$$

observed at the visits as $Y_{ik}=X_i(t_{ik})+h_{ik}+e_{ik}$ with $e_{ik}\sim N(0,\Sigma_e)$, where $\Sigma_e$ is the diagonal matrix of `meas_var`. Here $m_g$ is phenotype $g$'s median speed, so the **true speed ratio** is $s=m_1/m_0$ (`truth.speed_ratio`); $c=0$ except in S4; $\beta_{\text{age}}=\lambda=0$ in every preset. H-P's null holds exactly when $A_{\text{PIGD}}=A$: PIGD then runs TD's mechanism, only faster. The phenotype is derived from simulated year-0 items by the Stebbins rule, treatment starts with a hazard that rises with the latent motor level, and dropout rises with the latent independence level.

**Family B in one place.** Family B is deliberately *outside* the engine's class (§6.17–§6.19): discrete Euler steps of non-linear dynamics $-0.25X+W\tanh(X)+0.1+\ell_i+B_i(t)$ with a calendar-time random-walk burden $B_i$, site effects, a treatment effect that saturates with time on treatment, and a measurement scale that differs by phenotype and era.

**The scenario truths of family A** (derived in §6.14; $A^\circ$ is the preset of §6.2, $E_{jk}$ the matrix with a single 1 in row $j$, column $k$, and $\varepsilon_{jk}=0.10\,\mathrm{sd}_j/\mathrm{sd}_k$ one smallest effect of interest in raw drift units):

| scenario | median speeds $(m_0,m_1)$ | speed spreads $\sigma$ | load spreads $\tau$ | $b_1-b_0$ | drifts | H-P's null |
|---|---|---|---|---|---|---|
| S0 | (1, 1) | (0.4, 0.4) | (0.10, 0.10) | 0 | $A=A_{\text{PIGD}}=A^\circ$ | true |
| S0L | (1, 1) | (0.4, 0.4) | (0.10, 0.10) | $0.05\,e_{\text{axial}}$ | same | true |
| S1 | (1, 2) | (0.4, 0.4) | (0.10, 0.10) | $0.05\,e_{\text{axial}}$ | same | true |
| S1S | (1, 2) | (0.4, 0.6) | (0.10, 0.10) | $0.05\,e_{\text{axial}}$ | same | true |
| S1T | (1, 2) | (0.4, 0.4) | (0.10, 0.10) | $0.05\,e_{\text{axial}}$ | same; PIGD's ON-state axial effect $\kappa$ halved | true |
| S1V | (1, 2) | (0.4, 0.4) | (0.10, 0.15) | $0.05\,e_{\text{axial}}$ | same | true |
| S4 | (1, 2) | (0.4, 0.4) | (0.10, 0.10) | $0.05\,e_{\text{axial}}$ | same, plus $0.08\tanh(X)$ in both | true (non-linear) |
| S7 | (1, 2) | (0.6, 0.6) | (0.10, 0.10) | $0.05\,e_{\text{axial}}$ | same | true |
| S2 | (1, 2) | (0.4, 0.4) | (0.10, 0.10) | $0.05\,e_{\text{axial}}$ | $A_{\text{PIGD}}=A^\circ+2\varepsilon_{\text{ind,fn}}E_{\text{ind,fn}}$ | false |
| S2b | (1, 2) | (0.4, 0.4) | (0.10, 0.10) | $0.05\,e_{\text{axial}}$ | $A=A^\circ+\varepsilon_{\text{ind,ax}}E_{\text{ind,ax}}$, $A_{\text{PIGD}}=A^\circ+4\varepsilon_{\text{ind,ax}}E_{\text{ind,ax}}$ | false |
| S3 | (1, 2) | (0.4, 0.4) | (0.10, 0.10) | $0.05\,e_{\text{axial}}$ | $A_{\text{PIGD}}=A^\circ+2\varepsilon_{\text{ind,ax}}E_{\text{ind,ax}}-A^\circ_{\text{fn,tr}}E_{\text{fn,tr}}$ | false |

(ind = independence, fn = daily function, ax = axial, tr = tremor.)

**Reading order.** The subsections follow the dependencies rather than the order of the protocol: family A first (it defines `Truth`, which the scenarios modify), then the scenarios, family B, the qualification harness, the fidelity summaries, the parallel runner (which the gate uses), and finally the gate.

| file | subsections |
|---|---|
| `sim/family_a.py` | 6.1 `Truth` · 6.2 `default_truth` · 6.3 `phenotype_rule` · 6.4 `_items` · 6.5 `_step` · 6.6–6.8 `simulate` · 6.9 `true_support` |
| `sim/scenarios.py` | 6.10 constants · 6.11 `_load_truth` · 6.12 `pooled_sd` · 6.13 `_std` · 6.14 `truth_for` · 6.15 `expert_map` · 6.16 `generate` |
| `sim/family_b.py` | 6.17–6.19 `simulate` |
| `qualification.py` | 6.20 constants · 6.21 `spec_for` · 6.22 `dataset_task` · 6.23 `config_key` · 6.24 `_path`, `_run_one` · 6.25 `run_tasks` · 6.26 `make_tasks` · 6.27 `_mcse` · 6.28 `_rank_quantile` · 6.29 `calibration_summary` · 6.30 `h_m1` · 6.31 `power_summary` · 6.32 `speed_summary` |
| `fidelity.py` | 6.33 `panel_summaries` · 6.34 `compare` |
| `parallel.py` | 6.35 the worker side · 6.36 `ordered` · 6.37 `check_setup`, `check_square` |
| `gate.py` | 6.38 `check_reference` · 6.39 `check_gradient` · 6.40 `check_quadrature` · 6.41 `check_quadrature_lr` · 6.42 `start_setup`, `start_fit` · 6.43 `check_dispersed_starts` · 6.44 `profile_setup`, `profile_job` · 6.45 `profile` · 6.46 `identification` · 6.47 `recovery_setup`, `recovery_dataset` · 6.48 `recovery` · 6.49 `time_replicate` · 6.50 `plain` · 6.51 `Checkpoint` · 6.52–6.54 `run_gate` |

**How the examples were run.** Every printed output below came from a script in [`docs/guide_examples/s06/`](../guide_examples/s06/), run from the repository folder with the ppmi-causal environment (Python 3.13), for example `OMP_NUM_THREADS=1 ../ppmi-causal/.venv/bin/python docs/guide_examples/s06/ex_simulate.py`. Each script puts `src/` on the import path itself. The fits use small panels (35–113 participants at four domains), so each script finishes in under about two minutes; the timings printed are those of a shared M4 laptop. Small panels make some checks fail that may pass at the real size; this is pointed out where it happens.

---

### 6.1 `TD`, `PIGD`, `INDETERMINATE` and `Truth` — the true model of family A

[`family_a.py:20`](../../src/pdx/sim/family_a.py#L20), [`family_a.py:23`](../../src/pdx/sim/family_a.py#L23)

**What it does.** The three constants are the phenotype codes: `TD = 0` (the reference group of the engine), `PIGD = 1`, and `INDETERMINATE = -1` for participants the Stebbins rule cannot classify. `Truth` is a dataclass that holds every parameter of family A's data-generating model. Two properties are derived from it: `d`, the number of domains, and `speed_ratio`, the ratio of the median speeds.

**Inputs → outputs.** A record; nothing is computed except the two properties.

| field | symbol | default | meaning | what the engine estimates for it |
|---|---|---|---|---|
| `domains` | | — | domain names; their order is the column order | `panel.domains` |
| `A` | $A$ | — | TD's drift at the reference speed, `A[to, from]` | `A` of the speed-only fit; `A` (group 0) of the free fit |
| `A_pigd` | $A_{\text{PIGD}}$ | — | PIGD's drift at the reference speed; equal to `A` when the mechanism is shared | the free fit's `A1` estimates approximately $s\,A_{\text{PIGD}}$ (§6.47) |
| `speed` | $(m_0,m_1)$ | (1.0, 2.0) | median speed of TD and PIGD | $\gamma_1=\log(m_1/m_0)$ |
| `sigma` | $(\sigma_0,\sigma_1)$ | (0.4, 0.4) | standard deviation of log-speed within each phenotype | `log_sigma` |
| `b` | $b_g$ | — | (2, d) loads: a steady push on each domain | `b` |
| `q` | $q$ | 0.10 | diffusion variance per year at the reference speed; $Q=qI$ in both phenotypes | `L`, the Cholesky factor of $Q_g$ |
| `tau` | $(\tau_0,\tau_1)$ | (0.10, 0.10) | standard deviation of the participant load, per domain | `log_tau` |
| `lam` | $\lambda$ | 0.0 | how the load moves with the speed's random part $u_i$ | `lam` |
| `meas_var` | $\Sigma_e$ | — | measurement-error variances | fixed in the analysis (`spec.meas_var`) |
| `treat_mask` | | — | domains the treatment acts on | `spec.treat_mask` |
| `kappa` | $\kappa_g$ | — | (2, d) maximal short-duration effect in the ON state (Emax) | `kappa` |
| `kappa_ldr` | $\kappa^{\text{LDR}}_g$ | — | (2, d) maximal long-duration effect of exposure | `kappa_ldr` |
| `ed50`, `ec50_ldr` | | 3.0, 3.0 | half-maximal dose and exposure, in units of 100 mg levodopa-equivalent daily dose (LEDD) | fixed (`spec.ed50`, `spec.ec50_ldr`) |
| `curvature` | $c$ | 0.0 | a shared non-linear term $c\tanh(X)$ (scenario S4) | nothing: outside the engine's class |
| `beta_age` | $\beta_{\text{age}}$ | 0.0 | effect of age at onset on log-speed | `beta` |
| `meta` | | {} | free-form notes | |

**The mathematics.** `Truth` parametrises the model of §6.0. Because $u_i$ has mean 0 and median 0, $m_g$ is both the median and the geometric mean of the speeds in phenotype $g$:

$$
\operatorname{median}(s_i\mid g)=\exp\big(E[\log s_i\mid g]\big)=m_g\quad(\beta_{\text{age}}=0),
$$

and the property `speed_ratio` returns $s=m_1/m_0$.

H-P's null is a statement about the drifts at the reference speed: $A_{\text{PIGD}}=A$. The speed enters only through $s_i$, which multiplies the drift, the loads and the diffusion variance alike. That is why `A_pigd` is stored at the reference speed: "no mechanism difference" is then `A_pigd == A`, whatever the speeds.

**Worked example** ([`ex_truth.py`](../guide_examples/s06/ex_truth.py), first lines):

```python
t = default_truth(4)
print("A_pigd equals A:", np.array_equal(t.A, t.A_pigd), "| speed (TD, PIGD) =", t.speed, "-> speed_ratio =", t.speed_ratio)
print("q =", t.q, "| tau =", t.tau, "| sigma =", t.sigma, "| meas_var =", t.meas_var, "| lam =", t.lam)
```

```text
A_pigd equals A: True | speed (TD, PIGD) = (1.0, 2.0) -> speed_ratio = 2.0
q = 0.1 | tau = (0.1, 0.1) | sigma = (0.4, 0.4) | meas_var = [0.02 0.02 0.02 0.02] | lam = 0.0
```

**Design choices and pitfalls.**
- `A[to, from]` is the engine's convention too (row = the domain whose rate changes), so `true_support` and the expert map can be handed to `ModelSpec` unchanged.
- The analysis is given the true measurement-error variances and the true Emax constants: `spec_for` copies `truth.meas_var` (§6.21), and `tests/test_sim.py::test_family_a_treatment_uses_the_engine_form` checks that `ed50` and `ec50_ldr` equal the engine's defaults. The qualification therefore does not test misspecified reliabilities (D28 treats that as a sensitivity analysis).
- `q` is one number: both phenotypes have the same isotropic noise, and era does not shift it ($\omega=0$). The engine still estimates a full $Q_g$ per phenotype and $\omega$, whose true values are $qI$ and 0.
- `b`, `meas_var`, `treat_mask`, `kappa` and `kappa_ldr` default to `None`. A `Truth` built by hand without them fails inside `simulate`; build it with `default_truth` and modify it with `dataclasses.replace`, as `scenarios.truth_for` does.

### 6.2 `default_truth` — the preset shared mechanism

[`family_a.py:53`](../../src/pdx/sim/family_a.py#L53)

**What it does.** It returns the preset truth for 4, 6 or 10 domains: a small shared mechanism in which motor domains feed daily function, which feeds independence (at 10 domains, cognition, autonomic, sleepiness, mood and hallucinations join). Any other `d` raises `ValueError`.

**Inputs → outputs.** `d` → a `Truth` with `A = A_pigd`, speeds (1, 2) and the defaults of §6.1.

| d | domains | links (to ← from: value) |
|---|---|---|
| 4 | tremor, axial, function, independence | function ← tremor 0.10; function ← axial 0.20; independence ← function 0.25 |
| 6 | + brady(kinesia), cognition | function ← tremor 0.10, ← axial 0.15, ← brady 0.20; independence ← function 0.25, ← cognition 0.15; cognition ← axial 0.08 |
| 10 | + mood, autonomic, hallucinations, sleepiness | the six-domain links, plus independence ← autonomic 0.08; mood ← cognition 0.10, ← sleepiness 0.08; hallucinations ← cognition 0.12; sleepiness ← autonomic 0.10 |

**The mathematics.**
- Drift: $A^\circ=-0.25\,I+\sum_{(j,k)}a_{jk}E_{jk}$ (lines 68–70). Every domain relaxes at rate 0.25 per year at the reference speed.
- Load: $b_0=b_1=-\tfrac12A^\circ\mathbf 1$ (line 71). The equilibrium of the noise-free dynamics is then the same in every domain:

$$
A^\circ x^\ast+b=0\ \Longrightarrow\ x^\ast=-(A^\circ)^{-1}b=\tfrac12\mathbf 1 .
$$

- Stability: the links form a directed acyclic graph in every preset. Ordered topologically, $A^\circ$ is triangular, so its eigenvalues are its diagonal, all $-0.25$. Each domain's own deviation from $x^\ast$ decays at rate 0.25 (a half-life of $\ln 2/0.25\approx 2.8$ years at speed 1, 1.4 years at speed 2), while the links pass deviations downstream.
- Treatment (lines 72–76): tremor and axial (and bradykinesia when present) are treated; $\kappa_{gj}=0.5$ and $\kappa^{\text{LDR}}_{gj}=0.25$ on them, 0 elsewhere, the same in both phenotypes.
- Measurement error: $\Sigma_e=0.02\,I$.

**Worked example** ([`ex_truth.py`](../guide_examples/s06/ex_truth.py)):

```python
t = default_truth(4)
print("A (row = to, column = from):\n", t.A + 0.0)
print("equilibrium -A^-1 b, per phenotype:", np.linalg.solve(-t.A, t.b[0]), np.linalg.solve(-t.A, t.b[1]))
print("true_support:\n", true_support(t).astype(int))
```

```text
domains: ['tremor', 'axial', 'function', 'independence'] | d = 4
A (row = to, column = from):
 [[-0.25  0.    0.    0.  ]
 [ 0.   -0.25  0.    0.  ]
 [ 0.1   0.2  -0.25  0.  ]
 [ 0.    0.    0.25 -0.25]]
A_pigd equals A: True | speed (TD, PIGD) = (1.0, 2.0) -> speed_ratio = 2.0
b (row 0 = TD, row 1 = PIGD):
 [[ 0.125  0.125 -0.025  0.   ]
 [ 0.125  0.125 -0.025  0.   ]]
equilibrium -A^-1 b, per phenotype: [0.5 0.5 0.5 0.5] [0.5 0.5 0.5 0.5]
eigenvalues of A: [-0.25 -0.25 -0.25 -0.25]
treat_mask: [1 1 0 0] | kappa: [0.5 0.5 0.  0. ] | kappa_ldr: [0.25 0.25 0.   0.  ]
q = 0.1 | tau = (0.1, 0.1) | sigma = (0.4, 0.4) | meas_var = [0.02 0.02 0.02 0.02] | lam = 0.0
true_support:
 [[1 0 0 0]
 [0 1 0 0]
 [1 1 1 0]
 [0 0 1 1]]
d = 6: 6 links; largest real eigenvalue -0.250; treated: ['tremor', 'axial', 'brady']
    function <- tremor 0.10; function <- axial 0.15; function <- brady 0.20; independence <- function 0.25; independence <- cognition 0.15; cognition <- axial 0.08
d = 10: 11 links; largest real eigenvalue -0.250; treated: ['tremor', 'axial', 'brady']
    function <- tremor 0.10; function <- axial 0.15; function <- brady 0.20; independence <- function 0.25; independence <- cognition 0.15; independence <- autonomic 0.08; cognition <- axial 0.08; mood <- cognition 0.10; mood <- sleepiness 0.08; hallucinations <- cognition 0.12; sleepiness <- autonomic 0.10
```

By hand for $d=4$: $A^\circ\mathbf 1=(-0.25,-0.25,0.05,0)$ (the function row is $0.10+0.20-0.25$), so $b=-\tfrac12A^\circ\mathbf 1=(0.125,0.125,-0.025,0)$, as printed. (`+ 0.0` only turns the $-0.0$ entries of `-0.25 * np.eye(d)` into $0.0$ for printing.)

**Design choices and pitfalls.**
- The presets are acyclic, so the true mechanism never has feedback loops. A real map with a cycle (for example mood ↔ sleep) is not exercised by any preset.
- The scenarios look domains up by name (`names.index("axial")`, line 55 of `scenarios.py`), so they work only with these presets.
- The 10-domain names (`brady`) differ from WP0's (`brady_rigidity`), but the order is the same, which is what `p_missing[:d]` relies on (see §6.N for $d=4$).

### 6.3 `phenotype_rule` — the TD/PIGD rule of Stebbins (2013)

[`family_a.py:81`](../../src/pdx/sim/family_a.py#L81)

**What it does.** It classifies each row of simulated items as TD, PIGD or indeterminate from the ratio of the mean tremor item score to the mean PIGD item score.

**Inputs → outputs.** `tremor_items` (n, 4 in the simulators) and `pigd_items` (n, 5) → an integer array of codes 0, 1 or −1.

**The mathematics.** With $\bar T_i$ and $\bar P_i$ the mean item scores and $r_i=\bar T_i/\bar P_i$ (taken as $+\infty$ when $\bar P_i=0$),

$$
g_i=\begin{cases}\text{TD} & r_i\ge 1.15\ \text{and not}\ (\bar T_i=\bar P_i=0)\\ \text{PIGD} & \bar P_i>0\ \text{and}\ r_i\le 0.90\\ \text{indeterminate} & \text{otherwise.}\end{cases}
$$

Line by line: line 84 takes the two means; line 85 starts every row as indeterminate; lines 86–87 compute $r_i$ without dividing by zero (where $\bar P_i=0$ the denominator is replaced by 1 and the result by $\infty$); line 88 assigns TD, which includes "PIGD items all 0 with tremor present" through $r_i=\infty$; line 89 assigns PIGD. The two conditions cannot both hold, so the order of lines 88 and 89 does not matter.

**Worked example** ([`ex_phenotype_rule.py`](../guide_examples/s06/ex_phenotype_rule.py), part 1):

```python
rows = [([2, 2, 2, 2], [1, 1, 1, 1, 1]), ([0, 0, 0, 0], [2, 2, 2, 2, 2]), ([1, 1, 1, 1], [1, 1, 1, 1, 1]),
        ([0, 0, 0, 0], [0, 0, 0, 0, 0]), ([1, 0, 0, 0], [0, 0, 0, 0, 0]),
        ([2, 2, 2, 1], [2, 2, 2, 2, 2]), ([2, 2, 2, 2], [2, 2, 2, 2, 1]), ([3, 2, 2, 2], [2, 2, 2, 2, 1])]
out = phenotype_rule(trem, pigd)
```

```text
tremor mean 2.00  PIGD mean 1.00  ratio  2.000  -> TD
tremor mean 0.00  PIGD mean 2.00  ratio  0.000  -> PIGD
tremor mean 1.00  PIGD mean 1.00  ratio  1.000  -> indeterminate
tremor mean 0.00  PIGD mean 0.00  ratio    inf  -> indeterminate
tremor mean 0.25  PIGD mean 0.00  ratio    inf  -> TD
tremor mean 1.75  PIGD mean 2.00  ratio  0.875  -> PIGD
tremor mean 2.00  PIGD mean 1.80  ratio  1.111  -> indeterminate
tremor mean 2.25  PIGD mean 1.80  ratio  1.250  -> TD
```

The first four rows are the cases of `tests/test_sim.py::test_phenotype_rule`. The last three straddle the band $0.90<r<1.15$.

**Design choices and pitfalls.**
- Stebbins' rule uses 11 tremor items; the simulators use 4. Only the ratio matters, but fewer items make the means coarser, so the simulated misclassification is not calibrated to the real rule.
- With 4 and 5 integer items, ratios of exactly 1.15 or 0.90 cannot occur ($5\sum T/(4\sum P)$ would need $\sum T>16$), so the closed boundaries are never tested in simulation.

### 6.4 `_items` — simulated MDS-UPDRS items

[`family_a.py:93`](../../src/pdx/sim/family_a.py#L93)

**What it does.** It turns a latent year-0 level into `k` item scores on the 0–4 scale of the MDS-UPDRS (the Movement Disorder Society's Unified Parkinson's Disease Rating Scale).

**Inputs → outputs.** `level` (n,), `k`, `rng` → (n, k) integer-valued floats in {0, …, 4}.

**The mathematics.** For item $m$ of participant $i$,

$$
T_{im}=\min\!\big(4,\max(0,\operatorname{round}(1.2+x_i+\eta_{im}))\big),\qquad \eta_{im}\sim N(0,0.6^2)\ \text{independent}.
$$

`np.round` rounds halves to the even integer. Family A calls it with $x_i=X_{i,\text{tremor}}(0)$ for 4 tremor items and $X_{i,\text{axial}}(0)$ for 5 PIGD items (line 130).

**Worked example** ([`ex_phenotype_rule.py`](../guide_examples/s06/ex_phenotype_rule.py), parts 2 and 3):

```python
for level in (-1.5, 0.0, 1.0, 3.0):
    it = _items(np.full(20000, level), 5, rng)
panel, info = family_a.simulate(default_truth(4), 5000, np.random.default_rng(1))
p2, i2 = family_a.simulate(default_truth(4), 5000, np.random.default_rng(1), keep_indeterminate=True)
```

```text
level -1.5: mean item 0.092 (1.2 + level = -0.3); share at 0: 0.910, at 4: 0.000
level +0.0: mean item 1.202 (1.2 + level = 1.2); share at 0: 0.123, at 4: 0.000
level +1.0: mean item 2.201 (1.2 + level = 2.2); share at 0: 0.002, at 4: 0.015
level +3.0: mean item 3.875 (1.2 + level = 4.2); share at 0: 0.000, at 4: 0.877
simulated 5000: TD 2228, PIGD 2283, indeterminate (dropped) 489 (9.8%)
  landmark mean score of TD  : tremor +0.436, axial -0.218
  landmark mean score of PIGD: tremor -0.042, axial +0.446
keep_indeterminate=True: TD 2228 PIGD 2772 excluded 0
```

Reading the output:
- Between the floor and the ceiling the mean item equals $1.2+x$; at the floor (level −1.5) 91% of items are 0, at the ceiling 88% are 4.
- About 10% of simulated participants are indeterminate and dropped. TD participants carry more tremor and less axial burden at the landmark, PIGD the reverse: the phenotype is a noisy function of the year-0 tremor–axial contrast.
- `keep_indeterminate=True` keeps the 489 and **relabels them PIGD** (2283 + 489 = 2772); see §6.6.

**Design choices and pitfalls.**
- The items exist only to define the phenotype. They never enter the panel, as in the protocol ("year-0 values define the phenotype and enter no model term", §2.3).
- Because the rule is applied to noisy items, phenotype groups overlap in their latent states: the simulated groups carry "the artefacts of defining phenotypes from graph variables" (§8.2).

### 6.5 `_step` — the Euler–Maruyama step of the curved scenario

[`family_a.py:98`](../../src/pdx/sim/family_a.py#L98)

**What it does.** It advances the latent state over one visit interval with ten Euler–Maruyama sub-steps. It is used only when `truth.curvature != 0` (scenario S4), because the curved model has no closed-form transition.

**Inputs → outputs.** State `X` (d,), drift `A`, load `b`, participant load `l`, speed `s`, diffusion variance `q`, interval `dt`, curvature `c`, `rng` → the state at the end of the interval.

**The mathematics.** For $dX=s\,[AX+b+\ell+c\tanh(X)]\,dt+\sqrt{s\,q}\,dW$ in calendar time, with $h=\delta/10$,

$$
X\leftarrow X+s\,\big(AX+b+\ell+c\tanh(X)\big)\,h+\sqrt{s\,q\,h}\;\xi,\qquad \xi\sim N(0,I_d),\quad\text{ten times}.
$$

When $c=0$, `simulate` uses the exact transition instead (§6.7): with $\delta^\ast=s\,\delta$ the interval in disease time,

$$
X(t+\delta)=E\,X(t)+M\,(b+\ell)+\zeta,\quad E=e^{\delta^\ast A},\ M=\int_0^{\delta^\ast}e^{Aw}dw,\ \zeta\sim N\Big(0,\ S=\int_0^{\delta^\ast}e^{Aw}Q\,e^{A^\top w}dw\Big).
$$

Euler's scheme is first-order accurate. For the decay alone, ten steps multiply a deviation by $(1-0.25\,s h)^{10}$ instead of $e^{-0.25\,s\delta}$.

**Worked example** ([`ex_step_vs_exact.py`](../guide_examples/s06/ex_step_vs_exact.py)): one PIGD participant at speed 2 over one year, 20,000 draws of `_step` with $c=0$ against the exact moments from `transition_np`.

```python
E, M, S = transition_np(A, q * np.eye(4), s, dt)
mean_exact = E @ x0 + M @ (b + load)
draws = np.array([_step(x0.copy(), A, b, load, s, q, dt, 0.0, rng) for _ in range(R)])
```

```text
max |E - expm|, |M - block|, |S - Van Loan|: 0.0e+00, 0.0e+00, 5.6e-17
exact mean       : [0.7607 0.7672 0.5633 0.2203]
Euler (10 steps) : [0.7642 0.7584 0.5743 0.2243]  (Monte Carlo SE about 0.0026 )
exact sd         : [0.3556 0.3556 0.3645 0.367 ]
Euler sd         : [0.362  0.3643 0.3693 0.3744]
with curvature c = 0.08, Euler mean: [0.8414 0.854  0.6495 0.2434] (the c tanh(X) push)
```

Reading the output:
- `transition_np` agrees with one block matrix exponential and with Van Loan's block to rounding.
- Euler's mean misses the exact one by up to 0.011, about four Monte Carlo standard errors. That is its discretisation bias. For the axial domain (start 1.2, no inputs into it) the exact factor is $e^{-0.5}=0.6065$ and Euler's is $0.95^{10}=0.5987$; $1.2\times(0.6065-0.5987)=0.0094$, the size of the gap printed (0.7672 against 0.7584).
- Euler's standard deviations are about 2% too large.
- $c=0.08$ pushes every domain up by $0.08\tanh(X)$ per unit of disease time.

**Design choices and pitfalls.**
- S4's truth is therefore a slightly biased discretisation of the curved model. This does not matter for its purpose (a shared non-linear mechanism outside the engine's class), but S4 is not "the curved model solved exactly".
- `_step` draws $10d$ normal numbers per interval, the exact branch $d$, so S4 and S1 do not share random numbers even with one seed.

### 6.6 `simulate`, lines 106–145 — options, year 0, phenotype, covariates, speed and load

[`family_a.py:106`](../../src/pdx/sim/family_a.py#L106)

**What it does.** This first block of `simulate` reads the options, draws every participant's year-0 state, derives the phenotype from simulated items, and draws the covariates, the speed and the participant load.

**Inputs → outputs.** The options:

| argument | default | meaning |
|---|---|---|
| `truth`, `n`, `rng` | | the model; participants simulated *before* the phenotype rule; the generator |
| `visits` | (4, 7) | number of visits from the landmark on (landmark included), uniform on this range |
| `jitter` | 0.15 | each interval is $1+U(-0.15,0.15)$ years |
| `p_missing` | 0.04 | missing share, one number or one per domain |
| `start_hazard` | (−2.2, 1.0) | logistic intercept and slope of treatment start on the latent motor level |
| `dropout` | (−4.0, 0.8) | logistic intercept and slope of dropout on the latent independence level |
| `keep_indeterminate` | False | keep the indeterminate participants (as PIGD) instead of dropping them |
| `visit_probs` | None | {number of visits: probability}, replacing the uniform `visits` |
| `target_counts` | None | (n_TD, n_PIGD) to subsample to |

After line 145 the block has produced, for the $m$ classified participants: `X0` (m, d), `ph` (m,), `z` (m, 4), `u` (m,), `s` (m,), `load` (m, d) and `Q`.

**The mathematics.**
- **Year 0** (lines 127–129). $X_i(0)\sim N(0,C_0)$ with $C_0=0.3I+0.1\,\mathbf 1\mathbf 1^\top$ (variance 0.4, covariance 0.1) except $C_0[\text{tr},\text{ax}]=C_0[\text{ax},\text{tr}]=-0.15$. The negative tremor–axial covariance is what separates the phenotypes. $C_0$ stays positive definite in every preset: $0.3I+0.1\mathbf 1\mathbf 1^\top$ has smallest eigenvalue 0.3, and the change of one symmetric pair by $-0.25$ moves eigenvalues by at most 0.25.
- **Phenotype** (lines 130–134). $g_i$ from §6.3 applied to `_items(X0[:, tremor], 4)` and `_items(X0[:, axial], 5)`. Indeterminate participants are dropped (`keep`), or, with `keep_indeterminate=True`, kept and recoded PIGD (line 133).
- **Covariates** (lines 135–140). Era $\sim$ Bernoulli(0.5), age $\sim N(0,1)$, sex $\sim$ Bernoulli(0.4), duration $\sim N(0,1)$, independent of everything; then centred, $z_i\leftarrow z_i-\bar z$, over the $m$ classified participants. Era is column 0, which `spec_for` names as the column that shifts the noise.
- **Speed** (lines 141–142).

$$
s_i=m_{g_i}\exp\!\big(\sigma_{g_i}u_i+\beta_{\text{age}}\,\text{age}_i\big),\qquad u_i\sim N(0,1),
$$

  where `age` is the draw before centring.
- **Load** (line 143). $\ell_i=\lambda u_i\mathbf 1+\tau_{g_i}\epsilon_i$, $\epsilon_i\sim N(0,I_d)$; so $\operatorname{Cov}(\ell_i\mid g)=\lambda^2\mathbf 1\mathbf 1^\top+\tau_g^2 I$.
- **Noise** (line 144). $Q=qI$.

**How the code was formed from the maths.** Everything that does not depend on the visit loop is drawn first and vectorised (one call per quantity for all participants). The order of the draws is fixed: year 0, items, covariates, $u$, the load's $\epsilon$. One seed therefore gives one set of participants, whatever happens later in the loop.

**A consequence worth knowing: the landmark carries the speed.** The speed is independent of the year-0 state given the phenotype, but the panel starts at the year-1 landmark, which is $X_i(0)$ propagated over the first gap *at speed $s_i$*. A fast participant has moved further towards the equilibrium by the landmark. The engine conditions log-speed on the centred landmark scores ($\theta^\top\tilde y_{i1}$), so its $\gamma_1$ is a contrast between the phenotypes *at equal landmark scores*, which is not $\log(m_1/m_0)$ here.

**Worked example** ([`ex_speed_landmark.py`](../guide_examples/s06/ex_speed_landmark.py)). The speeds are not stored in the panel, so the script records them by wrapping `family_a.transition_np`, whose calls for one participant all carry that participant's speed. It then regresses log-speed on phenotype and the centred landmark scores: the engine's speed equation without its random part.

```python
family_a.transition_np = lambda A, Q, s, dt: (calls.append(s), orig(A, Q, s, dt))[1]
panel, info = scenarios.generate("S1", 20000, np.random.default_rng(7), d=d)
X = np.column_stack([np.ones(panel.n), g, y1c])
coef, *_ = np.linalg.lstsq(X, np.log(speeds), rcond=None)
```

```text
d = 4: 18034 participants (9 s)
   log(truth.speed_ratio) = 0.6931; sample difference of mean log-speeds = 0.6799
   regression on [1, PIGD, centred landmark scores]: PIGD coefficient 0.6518, TD intercept 0.0211, landmark coefficients [ 0.093  0.102 -0.01  -0.016]
   PIGD coefficient + theta'(mean landmark PIGD - TD) = 0.6799 (the Sp-1 construction)
d = 10: 18039 participants (10 s)
   log(truth.speed_ratio) = 0.6931; sample difference of mean log-speeds = 0.6991
   regression on [1, PIGD, centred landmark scores]: PIGD coefficient 0.6602, TD intercept 0.0191, landmark coefficients [ 0.089  0.085  0.039 -0.023 -0.086  0.029  0.007  0.044  0.021  0.024]
   PIGD coefficient + theta'(mean landmark PIGD - TD) = 0.6991 (the Sp-1 construction)
```

Reading the output:
- The marginal difference of mean log-speeds is $\log 2=0.693$ up to sampling error (0.680 and 0.699; the standard error is about 0.006).
- At equal landmark scores the difference is about 0.65–0.66: the landmark coefficients are positive for the motor domains, because a participant who is further along at year 1 is more likely to be fast.
- Adding $\theta^\top(\bar{\tilde y}_{\text{PIGD}}-\bar{\tilde y}_{\text{TD}})$ back recovers the marginal difference exactly (an identity of least squares with an intercept and a group dummy). This is how Sp-1 is built (`speed.sp1`, §5), so Sp-1 targets $\log s$, while $\gamma_1$ targets about 0.65. §6.47 and §6.N return to this.

**Design choices and pitfalls.**
- `z` is centred over the $m$ classified participants, before the dropout filter and the subsampling of §6.8, so the final panel's covariates are centred only approximately (§6.8's example).
- `keep_indeterminate=True` merges the indeterminate into PIGD without a flag in `info` (its `n_excluded_indeterminate` becomes 0). No caller uses the option.
- `beta_age` uses the uncentred age; with $\beta_{\text{age}}\ne0$ the median speed of a phenotype would still be $m_g$ only because age has mean 0.

### 6.7 `simulate`, lines 146–188 — visits, latent paths, treatment, measurement, missingness and dropout

[`family_a.py:146`](../../src/pdx/sim/family_a.py#L146)

**What it does.** For each participant it draws the number of visits and their times, moves the latent state from year 0 to each visit, starts and escalates treatment, produces the observed scores with the treatment effect, measurement error and missing values, and stops at dropout.

**Inputs → outputs.** The arrays of §6.6 → per participant: scores `y_rows` (one row per visit, NaN = missing), times, dose, exposure and examination state (ON = 1). The first row is the landmark.

**The mathematics.** Visit $k=0$ is the landmark; the code's `k > 0` means "after the landmark".
- **Visits** (lines 151–153). $T_i\sim U\{4,\dots,7\}$, or $P(T_i=t)\propto$ `visit_probs[t]`. Intervals $\delta_{ik}=1+U(-0.15,0.15)$, $k=0,\dots,T_i-1$, and $t_{ik}=\sum_{j\le k}\delta_{ij}$. The first interval runs from year 0 to the landmark, so $t_{i0}\in[0.85,1.15]$.
- **Latent path** (lines 158–166). Starting from $X_i(t_{i,-1})=X_i(0)$ at year 0, over each interval with $\delta^\ast_{ik}=s_i\,\delta_{ik}$ (the interval in disease time):

$$
X_i(t_{ik})=E(\delta^\ast_{ik})\,X_i(t_{i,k-1})+M(\delta^\ast_{ik})\,(b_{g_i}+\ell_i)+V\Lambda^{1/2}\xi_{ik},\qquad S(\delta^\ast_{ik})=V\Lambda V^\top,\ \xi_{ik}\sim N(0,I_d),
$$

  with $E$, $M$ and $S$ as in §6.5 and the drift $A_{g_i}$ (`truth.A` for TD, `truth.A_pigd` for PIGD). The noise is drawn through the eigen-decomposition of $S$, with negative rounding errors in $\Lambda$ set to 0. With curvature, `_step` replaces the exact transition.
- **Treatment start** (lines 167–169). At each visit after the landmark, an untreated participant starts with probability

$$
P(\text{start at }k)=\operatorname{expit}\Big(\alpha_0+\alpha_1\,\tfrac{1}{\lvert\mathcal T\rvert}\textstyle\sum_{j\in\mathcal T}X_{ij}(t_{ik})\Big),\qquad(\alpha_0,\alpha_1)=(-2.2,1.0),
$$

  where $\mathcal T$ are the treated domains and $\operatorname{expit}(x)=1/(1+e^{-x})$. At the equilibrium level 0.5 this is $\operatorname{expit}(-1.7)=0.15$ per visit.
- **Dose and exposure** (lines 170–172), in units of 100 mg LEDD. $D_{ik}=3$ at the first treated visit, then $D_{ik}=\min(9,D_{i,k-1}+1.5)$; exposure $R_{ik}=\tfrac12R_{i,k-1}+\tfrac12D_{ik}$, a moving average of the dose.
- **Examination state** (line 173). $O_{ik}\sim$ Bernoulli(0.5) if treated, else 0.
- **Treatment term** (lines 174–175), the engine's Emax forms (`engine/model.py::treat_effect`):

$$
h_{ikj}=-\Big[\kappa_{g_ij}\frac{D_{ik}}{\text{ED}_{50}+D_{ik}}\,O_{ik}+\kappa^{\text{LDR}}_{g_ij}\frac{R_{ik}}{\text{EC}_{50}+R_{ik}}\Big],\qquad \text{ED}_{50}=\text{EC}_{50}=3 .
$$

  It is 0 outside the treated domains, because $\kappa$ and $\kappa^{\text{LDR}}$ are 0 there. It acts on the observed scores only, never on $X$ (decision D24).
- **Measurement and missingness** (lines 176–177). $Y_{ik}=X_i(t_{ik})+h_{ik}+e_{ik}$, $e_{ik}\sim N(0,\Sigma_e)$; then each $Y_{ikj}$ is set missing independently with probability $p_j$: missing completely at random (MCAR), the landmark included.
- **Dropout** (lines 182–183). After recording visit $k\ge1$, the participant leaves with probability $\operatorname{expit}(-4+0.8\,X_{i,\text{ind}}(t_{ik}))$: about 3% per visit at the equilibrium level 0.5, 8% at level 2.

**How the code was formed from the maths.** The transitions are exact, because the panel's intervals are irregular and the engine's likelihood is exact over the actual intervals: simulating with the same $E$, $M$, $S$ the likelihood uses (`transition_np` computes them in numpy, by a block exponential and a scaled-and-squared series, §2) means family A's linear scenarios lie exactly in the engine's class, apart from the departures listed below and in §6.6. Everything that depends on the path (treatment, dropout) is decided inside the loop, visit by visit, after the state at that visit is known. The order of random draws within a visit is: transition noise, treatment start (only if untreated, after the landmark), examination state (only if treated; Python's `and` skips the draw otherwise), measurement error, missingness, dropout.

**Worked example** ([`ex_simulate.py`](../guide_examples/s06/ex_simulate.py), parts 1 and 2):

```python
truth = scenarios.truth_for("S1", 4)
panel, info = family_a.simulate(truth, 400, np.random.default_rng(11))
i = next(k for k in range(panel.n) if panel.dose[k][-1] > 0 and len(panel.t[k]) >= 6)
h = -(np.outer(d / (3 + d) * panel.on[i], kap) + np.outer(panel.exposure[i] / (3 + panel.exposure[i]), kl))
```

```text
1. info: {'speed_ratio': 2.0, 'n_TD': 175, 'n_PIGD': 172, 'n_excluded_indeterminate': 53} | panel.n = 347 | p = 4
   visits per participant (landmark included): {2: 6, 3: 4, 4: 96, 5: 78, 6: 82, 7: 81}
   first visit time: 0.850 to 1.146; intervals: 0.850 to 1.150
   share missing per domain: [0.04  0.043 0.043 0.043]
   treated by the last visit: 0.464 | treated at the landmark: 0.0
   z column means (era, age, sex, duration): [0. 0. 0. 0.]
2. participant 0 (TD):
   t       : [0.976 1.826 2.783 3.748 4.68  5.58 ]
   dose    : [0.  0.  0.  3.  4.5 6. ]
   exposure: [0.  0.  0.  1.5 3.  4.5]
   on      : [0. 0. 0. 0. 0. 1.]
   year    : [1 2 3 4 5 6]
   exposure by the recursion 0.5 e[k-1] + 0.5 dose[k]: [0.  0.  0.  1.5 3.  4.5]
   treatment term h on (tremor, axial) at each visit:
 [[ 0.     0.     0.    -0.083 -0.125 -0.483]
 [ 0.     0.     0.    -0.083 -0.125 -0.483]]
```

By hand, at the last visit: $D=6$, $O=1$, $R=4.5$, so $h=-(0.5\cdot\tfrac{6}{9}+0.25\cdot\tfrac{4.5}{7.5})=-(0.333+0.150)=-0.483$. At the first treated visit ($D=3$, OFF, $R=1.5$) only the long-duration term acts: $-0.25\cdot\tfrac{1.5}{4.5}=-0.083$. Ten participants have fewer than four visits: they dropped out after visit 1 or 2.

**Design choices and pitfalls.**
- **Confounding by indication is built in.** Treatment starts on the *latent* motor level at the same visit, so the dose path carries information about $X$ that the observed past does not. The engine treats the dose path as fixed covariates, so family A tests H-P under this departure, as the protocol intends.
- **No one is treated at the landmark** (the start is checked only after it), so the year-1 scores are free of treatment in family A; in PPMI some participants are treated by year 1.
- **The grid is always complete.** `year` is $1,\dots,T_i$ without gaps (line 201), so Step 1's two-slice tables use every transition. Real panels miss visits; `visit_probs` changes how many visits there are, not whether years are skipped.
- **Missingness is MCAR** at a fixed share per domain, and the intervals are a uniform jitter around one year. The protocol's "missingness resampled from PPMI's own patterns" and "intervals resampled from PPMI's actual intervals" (§8.2) are not implemented yet (§6.N).
- **Dropout is tied to severity** (the latent independence level), one of §8.2's stresses, and it is always on; "dropout tied to speed" is not available.
- `cache = {}` (line 146) is never used.

### 6.8 `simulate`, lines 189–204 — the panel, target counts and `info`

[`family_a.py:189`](../../src/pdx/sim/family_a.py#L189)

**What it does.** It keeps participants with at least two visits, optionally subsamples each phenotype to a target count, and packs the result into a `Panel` with a small `info` dictionary.

**Inputs → outputs.** The per-participant lists → `(panel, info)`. `info` holds `speed_ratio` ($m_1/m_0$), `n_TD` and `n_PIGD` (after subsampling) and `n_excluded_indeterminate` (of the $n$ simulated).

**The mathematics.** With $\mathcal K=\{i: T_i^{\text{obs}}\ge 2\}$ and targets $(n_0,n_1)$, the code draws, for each phenotype $g$, a simple random sample without replacement of size $n_g$ from $\mathcal K\cap\{g_i=g\}$ (line 196), and keeps the union in the original order (line 197). If a phenotype has fewer than $n_g$ participants it raises `ValueError`.

Because the dropout check comes only after the second visit has been recorded, everyone with at least two planned visits is kept: $\mathcal K$ is everyone unless `visit_probs` allows a single visit.

**Worked example** ([`ex_simulate.py`](../guide_examples/s06/ex_simulate.py), parts 3 and 4; the visit shares and counts are invented):

```python
vp = {2: 0.2, 3: 0.3, 6: 0.5}
p3, i3 = family_a.simulate(truth, 1500, np.random.default_rng(12), visit_probs=vp,
                           p_missing=(0.01, 0.02, 0.10, 0.20), target_counts=(120, 90))
family_a.simulate(truth, 200, np.random.default_rng(12), target_counts=(120, 90))
```

```text
3. target_counts (120, 90) -> 120 90 | visits: {2: 49, 3: 65, 4: 2, 5: 1, 6: 93}
   share missing per domain: [0.014 0.017 0.095 0.207]
   z column means after subsampling: [ 0.025 -0.041  0.031 -0.032]
4. n too small: only 88 participants of phenotype 0 for a target of 120; raise n
```

Reading the output:
- The counts are exact. The visit counts follow the invented shares, with a few 4s and 5s produced by dropout among those who were to have 6.
- The per-domain missing shares follow `p_missing`.
- After subsampling, the covariate means are no longer exactly 0 (they were centred over the pool of §6.6).

**Design choices and pitfalls.**
- The subsample is drawn after all simulation, from the same generator, so with one seed the pool is the same whatever the targets.
- The engine expects centred covariates (`data.Panel` docstring); the departure here is of the order of $1/\sqrt{n}$ and harmless, but the panel is not centred as built.
- `scripts/run_engine_gate.py --wp0` and `scripts/run_power_check.py` pass WP0's target counts with a pool of five times their sum. They read `configs/wp0_sim_settings.json`, which holds three aggregates of the WP0 audit: `target_counts` (TD, PIGD), `visit_probs` (participants by number of visits, with suppressed small cells entered as 2) and `p_missing` (one share per domain, in the ten-domain order), plus a note. This guide does not quote their values.

### 6.9 `true_support` — the true links

[`family_a.py:207`](../../src/pdx/sim/family_a.py#L207)

**What it does.** It returns the boolean matrix of entries that are non-zero in either phenotype's true drift.

**Inputs → outputs.** A `Truth` → (d, d) booleans, `[to, from]`.

**The mathematics.** $\mathcal S^{\text{true}}=\{(j,k): A_{jk}\ne0\ \text{or}\ (A_{\text{PIGD}})_{jk}\ne0\}$. The diagonal is always included because the own rates are $-0.25$.

**Worked example** ([`ex_truth.py`](../guide_examples/s06/ex_truth.py); the full output is in §6.2):

```python
print("true_support:\n", true_support(default_truth(4)).astype(int))
```

```text
true_support:
 [[1 0 0 0]
 [0 1 0 0]
 [1 1 1 0]
 [0 0 1 1]]
```

The four own rates and the three links of the four-domain preset.

**Design choices and pitfalls.** It is the union over phenotypes, so a link present in only one phenotype (S3) is in it. `expert_map` (§6.15) builds the map handed to the analysis from it.

### 6.10 `NULL_SCENARIOS`, `POWER_SCENARIOS`, `EPS` — the scenario lists and the smallest effect of interest

[`scenarios.py:27`](../../src/pdx/sim/scenarios.py#L27)

**What it does.** It names the eight shared-mechanism scenarios on which H-M1 is judged (S0, S0L, S1, S1S, S1T, S1V, S4, S7), the three mechanism-difference scenarios used for power (S2, S2b, S3), and $\varepsilon=0.10$.

**Inputs → outputs.** Constants. `qualification.NULL_A` is `NULL_SCENARIOS`; `make_tasks` uses the position of a scenario in `NULL_SCENARIOS + POWER_SCENARIOS` to build its seeds (§6.26).

**The mathematics.** $\varepsilon$ is the protocol's smallest effect of interest (§2.3, provisional, D12): 0.10 standard deviations of the target domain per standard deviation of the source domain, per year at the reference speed. §6.13 turns it into raw drift units.

**Worked example** ([`ex_scenarios.py`](../guide_examples/s06/ex_scenarios.py), first line):

```python
print("NULL_SCENARIOS:", scenarios.NULL_SCENARIOS, "| POWER_SCENARIOS:", scenarios.POWER_SCENARIOS, "| EPS =", scenarios.EPS)
```

```text
NULL_SCENARIOS: ('S0', 'S0L', 'S1', 'S1S', 'S1T', 'S1V', 'S4', 'S7') | POWER_SCENARIOS: ('S2', 'S2b', 'S3') | EPS = 0.1
```

**Design choices and pitfalls.**
- `EPS` repeats `[localisation] eps = 0.10` of `configs/protocol_v0.5.toml` instead of reading it. Running the qualification at the sensitivity value $\varepsilon=0.05$ (§2.3) would need a code change, and a change in the configuration file would not reach the simulators.
- S5 and S6 (latent mechanism and speed classes, for the exploratory Step 3) are not defined: `truth_for("S5")` raises `ValueError` (§6.14's example).

### 6.11 `_load_truth` — the load difference of every scenario except S0

[`scenarios.py:32`](../../src/pdx/sim/scenarios.py#L32)

**What it does.** It returns a copy of a truth in which PIGD's load on the axial domain is 0.05 higher.

**Inputs → outputs.** `Truth` → `Truth` with `b[1, axial] += 0.05`; the input is not modified (`b.copy()` and `dataclasses.replace`).

**The mathematics.** $b_1=b_0+0.05\,e_{\text{axial}}$. Because the load enters as $s_i(\dots+b_g)$, this is a speed-scaled steady push: PIGD's axial domain heads towards a higher equilibrium. H-P's null allows any load difference, so this never makes the null false.

**Worked example** ([`ex_scenarios.py`](../guide_examples/s06/ex_scenarios.py)):

```python
print(scenarios._load_truth(default_truth(4)).b[1] - default_truth(4).b[0])
```

```text
_load_truth: b[PIGD] - b[TD] = [0.   0.05 0.   0.  ]
```

**Design choices and pitfalls.** The size 0.05 is in raw units per year at the reference speed. With the own rate $-0.25$ it moves PIGD's axial equilibrium from 0.5 to $0.5+0.05/0.25=0.7$, about a third of the axial landmark standard deviation (0.64, §6.12).

### 6.12 `pooled_sd` — the standard deviations that size $\varepsilon$

[`scenarios.py:38`](../../src/pdx/sim/scenarios.py#L38)

**What it does.** It simulates 3,000 participants from the S1 truth with a fixed seed (12345) and returns the standard deviation of each domain's landmark score, pooled over TD and PIGD. The result is cached for each `d` (`functools.lru_cache`).

**Inputs → outputs.** `d` → a tuple of $d$ standard deviations.

**The mathematics.** With $Y_{i0j}$ the landmark score of domain $j$ and $N_j$ the participants observed in it,

$$
\mathrm{sd}_j=\Big(\frac{1}{N_j}\sum_i\big(Y_{i0j}-\bar Y_{\cdot0j}\big)^2\Big)^{1/2}
$$

(`np.nanstd`, divisor $N_j$). The scores include measurement error. Protocol §2.3 says "the standard deviations of the pooled baseline scores"; the code uses the landmark (year-1) scores, the first analysed state, as the module docstring says. In PPMI "baseline" usually means year 0, so the two readings differ (§6.N).

**Worked example** ([`ex_scenarios.py`](../guide_examples/s06/ex_scenarios.py)):

```python
sd = scenarios.pooled_sd(4)
scenarios.pooled_sd(4)              # cached
```

```text
pooled_sd(4) = [0.5812 0.6358 0.6374 0.6365]  (first call 0.9 s)
second call 0.0000 s; CacheInfo(hits=1, misses=1, maxsize=None, currsize=1)
```

**Design choices and pitfalls.**
- The S1 truth sizes every scenario, S0 included. The deviations are therefore the same raw numbers in all scenarios, which keeps the expert map and the degrees of freedom identical.
- The sizing uses `simulate`'s default options, not a scenario's stresses (`sim_kw`, for example WP0's missingness), so a stressed dataset gets the same raw deviation.
- The cache lives in each process, and only `_std` calls it (S2, S2b and S3). Every spawned worker of `run_tasks` or `parallel.ordered` that simulates one of these recomputes it once (about a second at $d=4$).

### 6.13 `_std` — one $\varepsilon$ in raw drift units

[`scenarios.py:46`](../../src/pdx/sim/scenarios.py#L46)

**What it does.** It converts $\varepsilon$ for the link "from → to" into raw drift units.

**Inputs → outputs.** `d`, `to`, `frm` → a float.

**The mathematics.** A drift entry $A_{jk}$ in raw units moves domain $j$ by $A_{jk}$ per year per unit of domain $k$. On the standardised scale (standard deviations of $j$ per standard deviation of $k$) it is $\tilde A_{jk}=A_{jk}\,\mathrm{sd}_k/\mathrm{sd}_j$. Setting $\tilde A_{jk}=\varepsilon$:

$$
\varepsilon_{jk}=\varepsilon\,\frac{\mathrm{sd}_j}{\mathrm{sd}_k},
$$

which is line 49, `EPS * sd[to] / sd[frm]`.

**Worked example** ([`ex_scenarios.py`](../guide_examples/s06/ex_scenarios.py)):

```python
scenarios._std(4, ind, fn), scenarios._std(4, ind, ax), scenarios._std(4, fn, tr)
```

```text
one epsilon in raw drift units: independence<-function 0.09986, independence<-axial 0.10011, function<-tremor 0.10968
```

By hand: $0.10\times0.6365/0.6374=0.09986$ for independence ← function, and $0.10\times0.6374/0.5812=0.10968$ for function ← tremor.

**Design choices and pitfalls.** Because the landmark standard deviations are all close to 0.6, one $\varepsilon$ is close to 0.10 in raw units for every link here. With real-data standard deviations (WP3 fits the simulators to PPMI) the raw sizes would differ more.

### 6.14 `truth_for` — the truth of each named scenario

[`scenarios.py:52`](../../src/pdx/sim/scenarios.py#L52)

**What it does.** It builds the family A truth of one scenario from the preset of §6.2, the load difference of §6.11 and the deviations of §6.13.

**Inputs → outputs.** `scenario`, `d` → `Truth`. Unknown names raise `ValueError`.

**The mathematics.** With $t$ the preset, $t_L$ = `_load_truth(t)` (`tl` in the code) and $E_{jk}$ the matrix unit:

| scenario | code (line) | truth |
|---|---|---|
| S0 | `replace(t, speed=(1, 1))` (58) | $m_0=m_1=1$; no load difference |
| S0L | `replace(tl, speed=(1, 1))` (60) | equal speeds; $b_1=b_0+0.05e_{\text{ax}}$ |
| S1 | `tl` (62) | $m=(1,2)$; load difference |
| S1S | `replace(tl, sigma=(0.4, 0.6))` (64) | as S1; $\sigma_1=1.5\,\sigma_0$ |
| S1T | `kappa[1, ax] *= 0.5` (66–68) | as S1; PIGD's short-duration axial effect 0.25 instead of 0.5 |
| S1V | `replace(tl, tau=(0.10, 0.15))` (70) | as S1; PIGD's load spread 1.5 times TD's |
| S4 | `replace(tl, curvature=0.08)` (72) | as S1; drift $+\,0.08\tanh(X)$ in both phenotypes |
| S7 | `replace(tl, sigma=(0.6, 0.6))` (74) | as S1; wider individual speeds in both |
| S2 | `Ap[ind, fn] += 2 * _std(...)` (77) | as S1; $A_{\text{PIGD}}=A^\circ+2\varepsilon_{\text{ind,fn}}E_{\text{ind,fn}}$ |
| S2b | `A[ind, ax] = _std(...)`, `Ap[ind, ax] = 4 * _std(...)` (79–80) | as S1; $A=A^\circ+\varepsilon_{\text{ind,ax}}E_{\text{ind,ax}}$, $A_{\text{PIGD}}=A^\circ+4\varepsilon_{\text{ind,ax}}E_{\text{ind,ax}}$ |
| S3 | `Ap[ind, ax] = 2 * _std(...)`, `Ap[fn, tr] = 0` (82–83) | as S1; $A_{\text{PIGD}}=A^\circ+2\varepsilon_{\text{ind,ax}}E_{\text{ind,ax}}-A^\circ_{\text{fn,tr}}E_{\text{fn,tr}}$ |

In S2, PIGD runs $A_{\text{PIGD}}$ at twice the speed, so on the calendar scale its drift is $2(A^\circ+2\varepsilon_{\text{ind,fn}}E_{\text{ind,fn}})$: the link is stronger *beyond* speed by $2\varepsilon$ at the reference speed. No positive number $a$ makes $A_{\text{PIGD}}=aA$, so H-P's null is false. In S2b the deviation is $3\varepsilon$, with the TD strength exactly $\varepsilon$. In S3 PIGD gains independence ← axial ($2\varepsilon$) and loses function ← tremor.

**Worked example** ([`ex_scenarios.py`](../guide_examples/s06/ex_scenarios.py)), every scenario at $d=4$, deviations expressed in $\varepsilon$:

```python
for sc in scenarios.NULL_SCENARIOS + scenarios.POWER_SCENARIOS:
    t = scenarios.truth_for(sc, 4)
    D = t.A_pigd - t.A
    changes = [f"{names[j]}<-{names[k]}: {D[j, k] / scenarios._std(4, j, k):+.2f}" for j, k in zip(*np.nonzero(np.abs(D) > 1e-12))]
```

```text
scenario  speed       sigma        tau          kappa[PIGD,axial]  curvature  b[PIGD,ax]-b[TD,ax]  changed drift entries (PIGD minus TD, in epsilon)
S0        (1.0, 1.0)  (0.4, 0.4)   (0.1, 0.1)                0.50       0.00                 0.00  -
S0L       (1.0, 1.0)  (0.4, 0.4)   (0.1, 0.1)                0.50       0.00                 0.05  -
S1        (1.0, 2.0)  (0.4, 0.4)   (0.1, 0.1)                0.50       0.00                 0.05  -
S1S       (1.0, 2.0)  (0.4, 0.6)   (0.1, 0.1)                0.50       0.00                 0.05  -
S1T       (1.0, 2.0)  (0.4, 0.4)   (0.1, 0.1)                0.25       0.00                 0.05  -
S1V       (1.0, 2.0)  (0.4, 0.4)   (0.1, 0.15)               0.50       0.00                 0.05  -
S4        (1.0, 2.0)  (0.4, 0.4)   (0.1, 0.1)                0.50       0.08                 0.05  -
S7        (1.0, 2.0)  (0.6, 0.6)   (0.1, 0.1)                0.50       0.00                 0.05  -
S2        (1.0, 2.0)  (0.4, 0.4)   (0.1, 0.1)                0.50       0.00                 0.05  independence<-function: +2.00
S2b       (1.0, 2.0)  (0.4, 0.4)   (0.1, 0.1)                0.50       0.00                 0.05  independence<-axial: +3.00; TD independence<-axial = 1.00 eps
S3        (1.0, 2.0)  (0.4, 0.4)   (0.1, 0.1)                0.50       0.00                 0.05  function<-tremor: -0.91; independence<-axial: +2.00

S3 raw values: A[function, tremor] TD 0.100 vs PIGD 0.000; A[independence, axial] TD 0.000 vs PIGD 0.200
truth_for('S5'): unknown scenario S5
```

`tests/test_sim.py::test_mechanism_scenarios_follow_the_table` checks the S2 deviation ($2\varepsilon$ exactly), the S2b ratio (4) and S3's two one-phenotype links.

**Design choices and pitfalls** (compared with the table of protocol §8.2; also in §6.N):
- **S3's TD-only link is smaller than $\varepsilon$.** Function ← tremor is 0.10 in raw units, which is $0.91\varepsilon$ on the standardised scale. A structural difference below the smallest effect of interest is hard to detect and arguably "clinically negligible" by the protocol's own definition.
- **S2 and S2b change one link**; the protocol says "one or two links" (S2) and "selected links" (S2b).
- **S1T halves only the short-duration (ON-state) response** $\kappa$; the long-duration response $\kappa^{\text{LDR}}$ is unchanged.
- **S1V varies the participant load spread** $\tau$ ("burden variance" in the protocol); family A has no time-varying burden.
- **S7 widens $\sigma$ in both phenotypes** (0.6 against S1's 0.4). S1 already has continuous individual speed variation, so S7 differs from S1 only in its size.
- **The speed ratio is fixed at 2.** The stress "speed ratios 1.25, 1.5 and 2" (§8.2) cannot be run through `generate`, whose keyword arguments go to `simulate`, not to the truth.

### 6.15 `expert_map` — the map every scenario hands the analysis

[`scenarios.py:89`](../../src/pdx/sim/scenarios.py#L89)

**What it does.** It returns the support the analysis is given: the preset's true links, plus any link of the scenario's own truth, plus the candidate link independence ← axial (zero in every scenario except S2b and S3).

**Inputs → outputs.** `Truth` → (d, d) booleans, `[to, from]`, diagonal included.

**The mathematics.** $\mathcal S=\mathcal S^{\text{true}}(t^\circ)\cup\mathcal S^{\text{true}}(t)\cup\{(\text{ind},\text{ax})\}$. Every scenario's changed entries are either in the preset (function ← tremor, independence ← function) or the candidate (independence ← axial), so $\mathcal S$ is the same in all eleven scenarios. H-P's degrees of freedom $(G-1)(n_{\text{sup}}-1)$ are therefore the same under the null and under the alternatives, so a power scenario uses its null's critical value fairly.

**Worked example** ([`ex_generate.py`](../guide_examples/s06/ex_generate.py)):

```python
maps = {sc: scenarios.expert_map(scenarios.truth_for(sc)) for sc in scenarios.NULL_SCENARIOS + scenarios.POWER_SCENARIOS}
print(h_p_df(spec_for(panel, info)))
```

```text
expert map (row = to, column = from), identical in all 11 scenarios: True
[[1 0 0 0]
 [0 1 0 0]
 [1 1 1 0]
 [0 1 1 1]]
true links outside the map in S3: 0 | map entries that are zero in S1's truth: 1
generate('S1') info keys: ['n_PIGD', 'n_TD', 'n_excluded_indeterminate', 'scenario', 'speed_ratio', 'support', 'truth']
   H-P df with this map: 7
```

With 4 own rates and 4 links, $n_{\text{sup}}=8$ and df $=7$.

**Design choices and pitfalls.**
- The map contains one zero link under the null, as a real expert map would. Under the null its estimate should be near 0 in both phenotypes.
- Step 1 may still add links when the qualification re-selects the support (`make_support_fn`, §6.22), so the df of a dataset can exceed 7.

### 6.16 `generate` — one dataset of a named scenario, with stresses

[`scenarios.py:98`](../../src/pdx/sim/scenarios.py#L98)

**What it does.** It is the single entry point of the simulators. For family A it builds the truth, simulates, attaches the expert map and applies the support stresses. For family B it maps S0, S1 and S3 to `family_b.simulate`'s arguments.

**Inputs → outputs.** `scenario`, `n`, `rng`, `family` ("A" or "B"), `d`, and keyword arguments → `(panel, info)`. For family A, `info` adds `scenario`, `support` and `truth` to `simulate`'s `info`; for family B, `info` holds `speed_ratio`, `support`, `n_TD`, `n_PIGD` and `scenario`.

**The mathematics.**
- Family A: `truth_for(scenario, d)`, then `family_a.simulate(truth, n, rng, **kw)`: every remaining keyword (`visits`, `p_missing`, `visit_probs`, `target_counts`, …) is a simulation option.
- Support stresses (§8.2, "a misspecified support"): `spurious=True` adds axial ← tremor, a zero link; `missing_link=True` removes function ← axial, a true link (0.20 at $d=4$).
- Family B (lines 102–110): S0 → speed ratio 1, no load shift; S1 → speed ratio 2, load shift 0.05; S3 → as S1 plus `mechanism_shift = 0.12`. `noninvariant` is passed through. Anything else raises `ValueError`.

**Worked example** ([`ex_generate.py`](../guide_examples/s06/ex_generate.py)):

```python
_, i_sp = scenarios.generate("S1", 200, np.random.default_rng(4), spurious=True)
_, i_ml = scenarios.generate("S1", 200, np.random.default_rng(4), missing_link=True)
p_kw, i_kw = scenarios.generate("S1", 200, np.random.default_rng(4), p_missing=0.25, visits=(2, 3))
_, ib_sp = scenarios.generate("S1", 200, np.random.default_rng(4), family="B", spurious=True, target_counts=(10, 10))
```

```text
spurious adds: [(1, 0)] | missing_link removes: [(2, 1)]
simulate() options pass through: share missing 0.255 | max visits 3
family B info keys: ['n_PIGD', 'n_TD', 'scenario', 'speed_ratio', 'support'] | support:
 [[1 0 0 0]
 [0 1 0 0]
 [1 1 1 0]
 [0 1 1 1]]
family B ignores spurious and target_counts: same support True | n_TD, n_PIGD = 83 89
generate('S2', family='B'): family B has S0, S1 and S3 analogues
```

**Design choices and pitfalls.**
- **Family B silently ignores** `spurious`, `missing_link` and every simulation option except `noninvariant` (they are popped or never passed). With WP0's settings in `sim_kw`, family A and family B datasets would have different sizes, visits and missingness, and no error says so.
- `spurious` and `missing_link` change only the support handed to the analysis, not the data. With Step 1 on, the data may restore a missing link (§6.22).
- `generate` uses one generator for everything, so one seed fixes the dataset completely.

### 6.17 `family_b.simulate`, lines 21–43 — set-up, phenotype, covariates, sites, speed and load

[`family_b.py:21`](../../src/pdx/sim/family_b.py#L21)

**What it does.** Family B is "outside the engine's class, coded separately" (§8.2). This block names the domains, sets the non-linear mechanism $W$, draws year 0 and the phenotype as family A does, and draws era, sites, covariates, speed and load.

**Inputs → outputs.** `n`, `rng`, `d` (the first four domains as in family A; `extra0`, … beyond), `speed_ratio` (2.0), `sigma` (0.4, both phenotypes), `mechanism_shift` (0), `visits` (4, 7), `n_sites` (20), `p_missing` (0.04, one number), `load_shift` (0.05), `noninvariant` (0).

**The mathematics.**
- $W_{\text{fn,tr}}=0.10$, $W_{\text{fn,ax}}=0.20$, $W_{\text{ind,fn}}=0.25$ (the pattern of family A's four-domain links, acting through $\tanh$), decay 0.25.
- Year 0 and phenotype: as family A, with $C_0$ built the same way and the same rule; indeterminate participants are always dropped.
- Era $\sim$ Bernoulli(0.5); site $\sim U\{0,\dots,19\}$; site effects $\zeta_{\text{site}}\sim N(0,0.08^2I_d)$, one draw per site and domain.
- Covariates $z_i=$ (era, $N(0,1)$, Bernoulli(0.4), $N(0,1)$), centred.
- Speed: $s_i=r^{g_i}\exp(0.4\,v_i)$, $v_i\sim N(0,1)$, with $r$ = `speed_ratio`.
- Load: $\ell_i=0.1\,\epsilon_i+0.05\,g_i\,e_{\text{axial}}$ (the second term with `load_shift`).
- Treated domains: tremor and axial.

**Worked example.** See §6.19.

**Design choices and pitfalls.** The year-0 block repeats family A's code (`C0`, `_items`, `phenotype_rule` are imported or rewritten), so the phenotype artefacts are the same in both families. Only the dynamics, treatment and measurement are coded independently.

### 6.18 `family_b.simulate`, lines 44–81 — non-linear dynamics, burden, treatment and measurement

[`family_b.py:44`](../../src/pdx/sim/family_b.py#L44)

**What it does.** For each participant it simulates the latent state with four Euler sub-steps per interval, a drifting burden, a treatment that saturates with time on treatment, and measurement with a phenotype- and era-specific scale and site effects.

**Inputs → outputs.** Per participant: scores, times, dose, exposure (equal to the dose) and examination state.

**The mathematics.** Visits and intervals as in family A. Over each interval $\delta$, four sub-steps with $\Delta=s_i\,\delta/4$ (disease time):

$$
X\leftarrow X+\Delta\big(-0.25\,X+W_i\tanh(X)+0.1\,\mathbf 1+\ell_i+B_i\mathbf 1\big)+\sqrt{0.1\,\Delta}\;\xi,\qquad B_i\leftarrow B_i+0.03\sqrt{\delta/4}\;\eta,
$$

with $\xi\sim N(0,I_d)$, $\eta\sim N(0,1)$, $B_i=0$ at year 0, and $W_i=W$ except $(W_i)_{\text{ind,ax}}$ = `mechanism_shift` for PIGD in the S3 analogue. The burden is one number added to every domain; its increments run in calendar time (not scaled by speed), while its effect on $X$ is scaled by speed, like a load.

Treatment (lines 63–68): at each visit after the landmark an untreated participant starts with probability $\operatorname{expit}(-2.2+\tfrac12(X_{\text{tr}}+X_{\text{ax}}))$; with $a_{ik}=t_{ik}-t^{\text{on}}_i$ the time on treatment,

$$
D_{ik}=\min(9,\,3+1.5\,a_{ik}),\qquad h_{ikj}=-0.3\,\big(1-e^{-a_{ik}/0.5}\big)\,\big(0.5+0.5\,O_{ik}\big)\quad(j\in\{\text{tr},\text{ax}\}),
$$

so the effect is 0 at the visit of initiation and saturates at $-0.15$ (OFF) or $-0.3$ (ON). The exposure recorded is the dose.

Measurement (lines 51–52, 69):

$$
Y_{ik}=L_iX_i(t_{ik})+\zeta_{\text{site}(i)}+h_{ik}+0.15\,\varrho_{ik},\qquad L_i=\operatorname{diag}(L_{ij}),\quad L_{ij}=1+0.04\,g_i+0.05\,\text{era}_i+\nu\,(g_i-\tfrac12)\,(-1)^j,
$$

with $\varrho_{ik}\sim N(0,I_d)$, $\nu$ = `noninvariant` and domains numbered from $j=0$, then MCAR missingness with one share, and dropout with probability $\operatorname{expit}(-4+0.8\,X_{\text{ind}})$ after each visit from the second on.

**Why a common scale keeps the null and domain-specific loadings break it.** For linear dynamics $\dot X=AX$, measuring $Y=LX$ with a diagonal $L$ gives $\dot Y=LAL^{-1}Y$. If $L=cI$ (one scale for all domains), $LAL^{-1}=A$: the measured mechanism is unchanged, so a phenotype- or era-specific *common* scale leaves H-P's null true. If $L$ differs between domains and between phenotypes (`noninvariant`), $L_1AL_1^{-1}\ne L_0AL_0^{-1}$: the measured mechanisms differ, and the null is false on the scale the engine estimates. That is why `noninvariant` is a named stress, not a null scenario.

**Worked example** ([`ex_family_b.py`](../guide_examples/s06/ex_family_b.py)): the treatment curve, the loadings, and the linear part of the dynamics on the measured scale.

```python
tau = np.array([0.0, 0.25, 0.5, 1.0, 2.0])
load = 1.0 + 0.04 * ph + 0.05 * era + nonInv * (ph - 0.5) * np.where(np.arange(4) % 2 == 0, 1, -1)
M0, M1 = L0 @ A @ np.linalg.inv(L0), L1 @ A @ np.linalg.inv(L1)
```

```text
treatment term at time on treatment [0.   0.25 0.5  1.   2.  ] :
   OFF [ 0.    -0.059 -0.095 -0.13  -0.147] 
   ON  [ 0.    -0.118 -0.19  -0.259 -0.295]
noninvariant = 0.0:
   phenotype 0, era 0: loading per domain [1. 1. 1. 1.]
   phenotype 0, era 1: loading per domain [1.05 1.05 1.05 1.05]
   phenotype 1, era 0: loading per domain [1.04 1.04 1.04 1.04]
   phenotype 1, era 1: loading per domain [1.09 1.09 1.09 1.09]
noninvariant = 0.1:
   phenotype 0, era 0: loading per domain [0.95 1.05 0.95 1.05]
   phenotype 0, era 1: loading per domain [1.  1.1 1.  1.1]
   phenotype 1, era 0: loading per domain [1.09 0.99 1.09 0.99]
   phenotype 1, era 1: loading per domain [1.14 1.04 1.14 1.04]
noninvariant = 0.0: measured linear drift, PIGD minus TD (era 0), largest |entry| 0.0000
noninvariant = 0.1: measured linear drift, PIGD minus TD (era 0), largest |entry| 0.0493
```

With `noninvariant = 0.1`, the measured linear drifts differ by up to 0.049, about half an $\varepsilon$.

**Design choices and pitfalls.**
- **The common scale keeps only the linear part invariant.** The non-linear term becomes $c\,W\tanh(Y/c)$ on the measured scale, which differs between scales $c$ at third order ($c\tanh(y/c)\approx y-y^3/(3c^2)$). The engine fits a linear model, so its pseudo-true drifts can differ slightly between phenotypes in family B's null. Probably small; noted in §6.N.
- The analysis fixes the measurement-error variance at 0.02 (`spec_for`), while family B's error variance is $0.15^2=0.0225$, and the site effects add a further participant-constant error. Whether this misspecification is intended is asked in §6.N.
- The site effects are constant within a participant, so the engine sees them as part of the state; through the drift they act like a participant load.

### 6.19 `family_b.simulate`, lines 82–90 — the panel and the map

[`family_b.py:82`](../../src/pdx/sim/family_b.py#L82)

**What it does.** It keeps participants with at least two visits, builds the `Panel` and returns the support the analysis is given.

**Inputs → outputs.** → `(panel, info)` with `info = {"speed_ratio", "support", "n_TD", "n_PIGD"}`.

**The mathematics.** The support is $\{(j,j)\}\cup\{W_{jk}\ne0\}\cup\{(\text{ind},\text{ax})\}$: the true links plus the candidate link, in every scenario (the same map as family A's at $d=4$).

**Worked example** ([`ex_family_b.py`](../guide_examples/s06/ex_family_b.py)):

```python
for label, kw in (("S0", dict(speed_ratio=1.0, load_shift=0.0)), ("S1", dict(speed_ratio=2.0, load_shift=0.05)),
                  ("S3", dict(speed_ratio=2.0, load_shift=0.05, mechanism_shift=0.12))):
    panel, info = family_b.simulate(600, np.random.default_rng(21), **kw)
```

```text
S0: n_TD 250, n_PIGD 287, speed_ratio 1.0, median visits 5, missing 0.041, treated by last visit 0.421
S1: n_TD 250, n_PIGD 287, speed_ratio 2.0, median visits 5, missing 0.041, treated by last visit 0.479
S3: n_TD 250, n_PIGD 287, speed_ratio 2.0, median visits 5, missing 0.038, treated by last visit 0.480
support handed to the analysis:
 [[1 0 0 0]
 [0 1 0 0]
 [1 1 1 0]
 [0 1 1 1]]
a treated participant: t [0.859 1.839 2.843 3.896 4.939 5.801] 
   dose [0.    0.    0.    0.    3.    4.293] 
   exposure [0.    0.    0.    0.    3.    4.293] (exposure is the dose) 
   on [0. 0. 0. 0. 0. 0.]
```

The same seed gives the same participants and phenotypes in the three analogues (the year-0 draws come first); only the dynamics differ. The participant shown started treatment at 4.939 years: dose 3 at initiation, then $3+1.5\times0.862=4.293$.

**Design choices and pitfalls.**
- Family B's S3 analogue has only a PIGD-only link (independence ← axial through $\tanh$, 0.12 in raw units, not sized in $\varepsilon$); family A's S3 also removes a TD link.
- There is no `truth` in `info`, so `spec_for` falls back to treated domains 0 and 1 and measurement-error variance 0.02 (§6.21), and the gate's recovery cannot run on family B.

### 6.20 `NULL_A`, `NULL_B` — the ten cells of H-M1

[`qualification.py:30`](../../src/pdx/qualification.py#L30)

**What it does.** It names the shared-mechanism scenarios on which H-M1 is judged: the eight of family A (`NULL_A = scenarios.NULL_SCENARIOS`) and S0 and S1 of family B (`NULL_B`).

**Inputs → outputs.** Constants, used as the defaults of `h_m1` (§6.30) and by `scripts/run_qualification.py` to split null rows from power rows.

**The mathematics.** H-M1 passes when $\hat p_c\le0.075$ in each cell $c$ of $\{A{:}S0,\dots,A{:}S7\}\cup\{B{:}S0,B{:}S1\}$ (protocol §8.7), with the exception of S4 described in §6.30.

**Worked example.** The list is printed by [`ex_h_m1.py`](../guide_examples/s06/ex_h_m1.py) in §6.30.

**Design choices and pitfalls.** The module docstring describes the harness: one dataset per job, saved as it finishes and skipped on rerun, at most four parallel jobs (§16), and preset simulator parameters (fitting them to PPMI's pooled population is a WP3 step that needs data access).

### 6.21 `spec_for` — the analysis model of a simulated dataset

[`qualification.py:34`](../../src/pdx/qualification.py#L34)

**What it does.** It builds the engine's `ModelSpec` for a simulated dataset: the support the analysis is handed, the treated domains, the treatment form, the fixed measurement-error variances, the number of quadrature nodes and the era column of the noise. `gate.py` uses it too.

**Inputs → outputs.** `panel`, `info`, `treatment` ("emax"), `n_quad` (12), `era` (True) → `ModelSpec(d, support=info["support"], p=panel.p, treat_mask, treatment, meas_var, n_quad, noise_era_col)`. With a family A `info` (which carries `truth`), `treat_mask` and `meas_var` are the truth's; with family B, domains 0 and 1 and $0.02$ for every domain. `noise_era_col` is 0 (era is the first covariate) when `era` is true and the panel has covariates.

**The mathematics.** The specification fixes the parameter layout (§2). At $d=4$ with $p=4$ covariates, 2 treated domains and $n_{\text{sup}}=8$:

$$
\underbrace{8}_{A}+\underbrace{1}_{\gamma}+\underbrace{4}_{\beta}+\underbrace{4}_{\theta}+\underbrace{2}_{\log\sigma}+\underbrace{8}_{b}+\underbrace{20}_{L}+\underbrace{8}_{\Gamma^y}+\underbrace{32}_{\Gamma^z}+\underbrace{8}_{\lambda}+\underbrace{8}_{\log\tau}+\underbrace{1}_{\omega}+\underbrace{4}_{\kappa}+\underbrace{4}_{\kappa^{\text{LDR}}}=112
$$

parameters in the speed-only model. The free model drops $\gamma$ and adds PIGD's 8 drift entries (119), so H-P's degrees of freedom are $119-112=7=(G-1)(n_{\text{sup}}-1)$.

**Worked example** ([`ex_spec_for.py`](../guide_examples/s06/ex_spec_for.py)):

```python
panel, info = scenarios.generate("S1", 150, np.random.default_rng(2), family=fam)
spec = spec_for(panel, info)
```

```text
family A: d=4 p=4 treatment=emax n_quad=12 noise_era_col=0 treat_mask=[1 1 0 0] meas_var=[0.02 0.02 0.02 0.02]
   n_sup=8, H-P df=7, parameters: null 112, free 119, joint 120
family B: d=4 p=4 treatment=emax n_quad=12 noise_era_col=0 treat_mask=[1 1 0 0] meas_var=[0.02 0.02 0.02 0.02]
   n_sup=8, H-P df=7, parameters: null 112, free 119, joint 120
options: treatment linear | n_quad 20 | noise_era_col None
```

**Design choices and pitfalls.**
- In family A the analysis is handed the **true** measurement-error variances. Real reliabilities will be estimated with error (D28), so this flatters the engine slightly.
- In family B the fallback 0.02 differs from the simulator's $0.15^2=0.0225$ (§6.18).
- The Emax constants are not passed; `ModelSpec`'s defaults (3.0) equal the truth's, which `tests/test_sim.py` checks.

### 6.22 `dataset_task` — one dataset of the warp-speed calibration

[`qualification.py:45`](../../src/pdx/qualification.py#L45)

**What it does.** It runs everything for one simulated dataset: simulate, select the support with Step 1, fit the speed-only and free models, draw **one** bootstrap replicate (the warp-speed method), and compute Sp-1 and Sp-3 from the same fits. It returns a plain dictionary, which `_run_one` saves as JSON.

**Inputs → outputs.** A task dictionary: `scenario`, `seed`, `n` (participants simulated before the phenotype rule), and optionally `family` ("A"), `d` (4), `sim_kw` (simulation options and stresses), `treatment`, `step1` (True), `step1_cfg` (a `Step1Config` as a dictionary), `replicate` (True) and `fit_kw` (passed to the fits). The row it returns:

| key | meaning |
|---|---|
| `task`, `config`, `scenario`, `family`, `seed` | the task, its configuration key (§6.23) and identifiers |
| `n` | participants **analysed** (after the phenotype rule) |
| `lr`, `df` | H-P's statistic $T=\max\{2(\hat\ell_{\text{free}}-\hat\ell_{\text{null}}),0\}$ and its df on the selected support |
| `lr_star` | the replicate's statistic $T^\ast$, or `None` if the replicate failed |
| `converged`, `replicate_ok`, `replicate_converged` | the observed fits' and the replicate's status |
| `sp1`, `sp3_share`, `true_speed_ratio` | Sp-1, Sp-3's share and $s$ |
| `seconds`, `counts` | time from support selection to the end of the replicate; aggregate counts by phenotype |

**The mathematics.** With $\hat x_0$ the speed-only fit on the support $\hat{\mathcal S}$ selected from the data,
- the observed statistic is $T=2\{\hat\ell_{\text{free}}(\hat{\mathcal S})-\hat\ell_{\text{null}}(\hat{\mathcal S})\}$, clamped at 0 (`hp.lr_stat`);
- the replicate simulates $Y^\ast$ from $\hat x_0$ conditionally on each participant's phenotype, covariates, landmark scores, visit times, dose path, examination states and missingness pattern (`engine/simulate.py`, §3); re-selects the support $\hat{\mathcal S}^\ast$ on $Y^\ast$; refits both models; and returns $T^\ast$;
- Sp-1 $=\exp\{\hat\gamma_1+\hat\theta^\top(\bar{\tilde y}_{\text{PIGD}}-\bar{\tilde y}_{\text{TD}})\}$ from the speed-only fit, and Sp-3's share $=1-\lVert\hat A_1-\hat s_{\text{LS}}\hat A_0\rVert^2/\lVert\hat A_1-\hat A_0\rVert^2$ from the free fit (`speed.py`, §5).

The warp-speed method (Giacomini, Politis and White 2013) pools the $T^\ast_r$ of all datasets $r$ of a cell into one reference distribution, so each dataset needs one replicate instead of hundreds (§6.29).

Line by line: line 48 seeds one generator from the task; 49–50 simulate; 51 builds the specification; 52–54 build Step 1's selection rule around the expert map (`make_support_fn`, §4); 56–57 call `hp.warp_replicate` (observed fits + one replicate); 58–62 are the power-only branch (`replicate=False`: observed fits only, `lr_star` NaN); 64–70 assemble the row.

**Worked example** ([`ex_dataset_task.py`](../guide_examples/s06/ex_dataset_task.py)), one S1 dataset of 80 simulated participants with the protocol's defaults (Step 1 on, one replicate):

```python
task = Q.make_tasks("S1", 1, 80, "A", d=4)[0]
res = Q.dataset_task(task)
```

```text
task: {'scenario': 'S1', 'family': 'A', 'n': 80, 'seed': 1020000, 'd': 4}
wall time 102 s
   config               7c7d27c335
   scenario             S1
   family               A
   seed                 1020000
   n                    76
   lr                   6.0158
   lr_star              10.3800
   df                   7
   converged            True
   replicate_ok         True
   replicate_converged  True
   sp1                  1.5864
   sp3_share            0.0623
   true_speed_ratio     2.0000
   counts: {'0': {'participants': 40, 'visits': 223, 'transitions': 183, 'median_visits': 5.0, 'share_missing': 0.040625}, '1': {'participants': 36, 'visits': 189, 'transitions': 153, 'median_visits': 5.0, 'share_missing': 0.04717261904761905}}
```

Step 1 added no link (df = 7, the map's). One dataset says nothing about calibration; the pooled replicates of hundreds of such datasets do (§6.29). Sp-1 is 1.59 against a truth of 2 with 76 participants.

**Design choices and pitfalls.**
- One generator serves the simulation and the replicate, so the task's seed fixes both.
- `n` means different things in a successful row (participants analysed) and in a failed row (participants simulated, copied from the task by `_run_one`).
- `replicate=False` is meant for power-only runs, but it changes the configuration key, so such rows never meet an S1 critical value in `power_summary` (§6.31). `scripts/run_power_check.py` uses its own summary and is not affected.

### 6.23 `config_key` — which datasets may be pooled

[`qualification.py:73`](../../src/pdx/qualification.py#L73)

**What it does.** It hashes everything in a task except its seed and scenario. Datasets are pooled only within one configuration, and a power scenario finds its null reference by the same key.

**Inputs → outputs.** A task dictionary → 10 hexadecimal characters.

**The mathematics.** The key is the first 10 hexadecimal characters of the SHA-1 hash of `json.dumps(cfg, sort_keys=True, default=str)`, where `cfg` is the task without `seed` and `scenario`: 40 bits, so a collision between two of a handful of configurations has probability of order $k^2/2^{41}$, negligible.

**Worked example** ([`ex_tasks.py`](../guide_examples/s06/ex_tasks.py)):

```python
print(Q.config_key(a[0]), Q.config_key(c[0]), Q.config_key(b[0]))           # A:S1, A:S2, B:S1
print(Q.config_key({**c[0], "replicate": False}))
k_int = Q.config_key({**a[0], "sim_kw": {"visit_probs": {2: 0.2, 10: 0.8}}})
k_str = Q.config_key({**a[0], "sim_kw": {"visit_probs": {"2": 0.2, "10": 0.8}}})
```

```text
config keys  A:S1 47d16bc32c  A:S2 47d16bc32c  B:S1 ed897078f8
   replicate=False changes the key: d97e15b68a
   visit_probs with int keys -> 87a20c57fa, with str keys -> 2fb4f6cd61
```

**Design choices and pitfalls.**
- The family is part of the key, so families A and B never pool.
- **The key depends on how values are written, not only on what they mean.** Integer and string keys of `visit_probs` sort differently and give different keys; so would `3000` and `3000.0` in `fit_kw`. The scripts convert WP0's keys to integers consistently, so this is harmless now, but a task built by hand from the JSON settings file would not pool with, or skip, earlier results.
- `default=str` hashes any non-JSON value by its printed form; a large numpy array prints with "…", so two different arrays could share a key. Tasks hold only small JSON-able values today.

### 6.24 `_path` and `_run_one` — one JSON per dataset, skipped on rerun

[`qualification.py:81`](../../src/pdx/qualification.py#L81), [`qualification.py:85`](../../src/pdx/qualification.py#L85)

**What it does.** `_path` names a dataset's result file `{family}_{scenario}_{key}_seed{seed}.json`. `_run_one` returns the saved result if the file exists; otherwise it runs `dataset_task`, catches any exception into an error row (so a failed dataset is logged, not dropped), writes the result to a temporary file and renames it over the final name.

**Inputs → outputs.** `(task, out_dir)` → the row (also on disk).

**The mathematics.** None. The rename is atomic on one file system, so an interrupted run leaves either no file or a complete one, never half a JSON.

**Worked example** ([`ex_tasks.py`](../guide_examples/s06/ex_tasks.py)): one file written by hand, one task that fails at once (an unknown scenario); nothing is fitted.

```python
p = Q._path(out, done)
p.write_text(json.dumps({..., "note": "written by hand"}))
bad = {"scenario": "S9", "family": "A", "n": 50, "seed": 7}
rows = Q.run_tasks([done, bad], out, jobs=1)
```

```text
file name: A_S1_47d16bc32c_seed1020000.json
row 1 (existing file, skipped): written by hand
row 2 (failed, logged): {'scenario': 'S9', 'seed': 7, 'n': 50, 'family': 'A', 'error': "ValueError('unknown scenario S9')"}
files now: ['A_S1_47d16bc32c_seed1020000.json', 'A_S9_f24bfd6b5f_seed7.json']
```

**Design choices and pitfalls.**
- A failed dataset's file is written too, so a rerun does **not** retry it. That is deliberate (a failure counts as a rejection in the calibration, §6.29), but after fixing a bug the error files must be deleted by hand.
- The existence check uses only the file name, so a changed code version with the same task settings silently reuses old results. The code version is not part of the key.
- `json.dumps` writes a non-finite Sp-3 share as `NaN`, which Python reads back but strict JSON readers do not.

### 6.25 `run_tasks` — the datasets, at most four at a time

[`qualification.py:101`](../../src/pdx/qualification.py#L101)

**What it does.** It creates the output folder and runs the tasks: in this process if `jobs <= 1`, otherwise in a pool of `min(jobs, 4)` worker processes started with "spawn", one task at a time per worker (`chunksize=1`). Before starting the pool it sets the four thread variables (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`) to 1 unless they are already set.

**Inputs → outputs.** `tasks`, `out_dir`, `jobs` (4) → the rows, in task order.

**The mathematics.** With $W\le4$ workers and $m$ minutes per dataset, $R$ datasets take about $R\,m/(60W)$ hours (protocol §8.8's formula).

**Worked example.** The serial branch runs in [`ex_tasks.py`](../guide_examples/s06/ex_tasks.py) above (`jobs=1`). The parallel branch is the same `_run_one` in spawned workers.

**Design choices and pitfalls.**
- **The cap of 4 is silent**: `jobs=8` runs four workers. It implements §16 ("at most 4 parallel jobs"), but `parallel.ordered` (used by the gate) has no cap (§6.36).
- `setdefault` keeps a thread setting the user already made: with `OMP_NUM_THREADS=4` in the shell, four workers run four threads each. The variables are not restored afterwards, and the workers do not call `torch.set_num_threads(1)` (`parallel._start` does); PyTorch reads `OMP_NUM_THREADS` when it is imported, so the default setting still gives one thread.
- `pool.map` returns only when every task has finished, but each result is on disk as soon as it is done, so an interrupted run loses at most the tasks in progress.

### 6.26 `make_tasks` — tasks with unique seeds

[`qualification.py:114`](../../src/pdx/qualification.py#L114)

**What it does.** It builds `n_datasets` tasks for one scenario, family and size, each with its own seed; any extra keyword becomes part of every task (and of its configuration key).

**Inputs → outputs.** `scenario`, `n_datasets`, `n`, `family`, `seed0`, `**kw` → a list of task dictionaries.

**The mathematics.** With $j$ the scenario's position in the eleven-scenario list (S0 = 0, …, S3 = 10),

$$
\text{seed}_r=\underbrace{10^6\ (\text{A})\ \text{or}\ 2\cdot10^6\ (\text{B})}_{\text{family}}+10^4\,j+o+r,\qquad r=0,\dots,R-1,
$$

where $o$ is `seed0`. Seeds are unique across families, scenarios and datasets as long as $o+R\le10^4$.

**Worked example** ([`ex_tasks.py`](../guide_examples/s06/ex_tasks.py)):

```python
a = Q.make_tasks("S1", 3, 1400, "A", d=10, fit_kw={"maxiter": 3000})
c = Q.make_tasks("S2", 2, 1400, "A", d=10, fit_kw={"maxiter": 3000})
b = Q.make_tasks("S1", 2, 1400, "B", d=10, fit_kw={"maxiter": 3000})
print([t["seed"] for t in Q.make_tasks("S1", 3, 1400, "A", seed0=2)])
```

```text
seeds  A:S1 [1020000, 1020001, 1020002]  A:S2 [1080000, 1080001]  B:S1 [2020000, 2020001]
seed0=2 shifts them: [1020002, 1020003, 1020004]
```

**Design choices and pitfalls.** Overlapping ranges of `seed0` produce the same task, hence the same file name, so extending a run with an overlapping `seed0` skips the datasets already done instead of duplicating them. To add new datasets, use `seed0` equal to the number already run.

### 6.27 `_mcse` — the Monte Carlo standard error of a rate

[`qualification.py:123`](../../src/pdx/qualification.py#L123)

**What it does.** It returns $\sqrt{p(1-p)/R}$, or NaN when $R=0$.

**Inputs → outputs.** A rate $p$ and a count $R$ → a float.

**The mathematics.** If $R$ datasets reject independently with probability $p$, the number of rejections is Binomial$(R,p)$, and $\hat p$ has variance $p(1-p)/R$ (protocol §8.6). At $p=0.05$: 0.0098 with $R=500$, 0.0218 with $R=100$.

**Worked example** ([`ex_calibration.py`](../guide_examples/s06/ex_calibration.py)):

```python
Q._mcse(0.05, 500), Q._mcse(0.05, 100)
```

```text
_mcse(0.05, 500) = 0.00975 | _mcse(0.05, 100) = 0.02179
```

**Design choices and pitfalls.**
- At $\hat p=0$ or 1 it returns 0, which hides the uncertainty; a Wilson or Clopper–Pearson interval would not.
- For the warp-speed rejection rate the binomial formula **understates** the Monte Carlo error, because the critical value is itself estimated from the same number of replicates (§6.29).

### 6.28 `_rank_quantile` — the bootstrap critical value, with failures as $+\infty$

[`qualification.py:127`](../../src/pdx/qualification.py#L127)

**What it does.** It returns the $\lceil qR\rceil$-th smallest of $R$ values, with missing, NaN or infinite values counted as $+\infty$, and $+\infty$ for an empty list.

**Inputs → outputs.** `values`, `q` → a float.

**The mathematics.** With the values sorted, $v_{(1)}\le\dots\le v_{(R)}$, and failed replicates set to $+\infty$,

$$
\hat c_q=v_{(\lceil qR\rceil)} .
$$

A failed replicate is "at least as large as anything", as in the sequential procedure of the real analysis (§5): failures can only raise the critical value, which makes the calibration conservative rather than liberal.

**Worked example** ([`ex_calibration.py`](../guide_examples/s06/ex_calibration.py)):

```python
Q._rank_quantile([3.0, 1.0, 2.0, None], 0.5); Q._rank_quantile(list(range(1, 21)), 0.95)
Q._rank_quantile(list(range(1, 20)) + [float("inf")], 0.95)
```

```text
_rank_quantile([3, 1, 2, None], 0.5) = 2.0 | ([1..20], 0.95) = 19.0 | ([1..19, inf], 0.95) = 19.0
```

`tests/test_qualification.py` adds the case $[1,\infty]$ at 0.95, which gives $+\infty$.

**Design choices and pitfalls.**
- No interpolation, so the critical value is always one of the replicates.
- `min(len(v) - 1, ...)` guards the top end, but at `q = 0` the index is −1 and the function returns the *largest* value. It is only ever called with $q=0.95$.

### 6.29 `_groups` and `calibration_summary` — the warp-speed rejection rate and Bradley's criterion

[`qualification.py:136`](../../src/pdx/qualification.py#L136), [`qualification.py:143`](../../src/pdx/qualification.py#L143)

**What it does.** `_groups` sorts the rows into cells (family, scenario, configuration). For each cell `calibration_summary` takes the $(1-\alpha)$ quantile of the pooled bootstrap statistics as the critical value, counts how many datasets' observed statistics exceed it, and reports the rejection rate with its MCSE and whether it passes Bradley's 7.5%. It also reports the rejection rate against the $\chi^2$ critical value and counts of failures and non-convergence.

**Inputs → outputs.** Rows from `dataset_task` (or error rows), `alpha` (0.05), `bradley` (0.075) → a dictionary keyed `"family:scenario:config"` with `datasets`, `failed`, `critical_value`, `rejection_rate`, `mcse`, `passes_bradley`, `rejection_rate_chi2`, `non_converged`, `replicates_failed`, `replicates_not_converged`, `median_seconds` and one example task.

**The mathematics.** In a cell with $R$ datasets, of which $\mathcal O$ ran without error, the warp-speed critical value and rejection rate are

$$
\hat c=T^\ast_{(\lceil(1-\alpha)\lvert\mathcal O\rvert\rceil)}\quad(\text{failed replicates}=+\infty),\qquad
\hat p=\frac1R\sum_{r=1}^R\Big(\mathbb 1[r\notin\mathcal O]+\mathbb 1[r\in\mathcal O]\,\mathbb 1[T_r>\hat c]\Big),
$$

so a dataset that failed counts as a rejection. Then $\text{MCSE}=\sqrt{\hat p(1-\hat p)/R}$ and the cell passes when $\hat p\le0.075=1.5\alpha$, the upper end of Bradley's liberal interval $[0.5\alpha,1.5\alpha]$ (protocol §8.7 uses only the upper end). The $\chi^2$ rate uses each dataset's own degrees of freedom, $\mathbb 1[T_r>\chi^2_{\mathrm{df}_r,0.95}]$, because the support, and so the df, is re-selected in every dataset.

*Why the warp-speed method works.* Under the null, each $T^\ast_r$ is a draw from the bootstrap distribution of dataset $r$. If that distribution varies little between datasets of one cell, the pooled $T^\ast$ estimate it, and $\hat p$ estimates the size of the bootstrap test with one replicate per dataset instead of $B$ (Giacomini, Politis and White 2013).

*Its Monte Carlo behaviour.* Take an exactly calibrated procedure: $T_r$ and $T^\ast_r$ independent with the same continuous distribution $F$. Then $U=F(\hat c)$ is the $k$-th order statistic of $R$ uniforms, $k=\lceil0.95R\rceil$, so

$$
E[\hat p]=1-\frac{k}{R+1},\qquad \operatorname{Var}(\hat p)=\underbrace{\operatorname{Var}(U)}_{\text{noise of }\hat c}+\underbrace{\frac{E[U(1-U)]}{R}}_{\text{binomial}}\approx\frac{k(R-k+1)}{(R+1)^2(R+2)}+\frac{\bar p(1-\bar p)}{R},\quad \bar p=E[\hat p] .
$$

At $R=100$ the mean is $1-95/101=5.94\%$, not 5%, and the variance is about twice the binomial one. The MCSE that `_mcse` reports covers only the second term.

Line by line: line 148 loops over the cells; 149 keeps the rows without an error; 150 computes $\hat c$ from their `lr_star`; 151 marks a rejection for an error row or $T_r>\hat c$; 152 does the same against $\chi^2$; 153 averages over all $R$ rows; 154–162 assemble the report.

**Worked example** ([`ex_calibration.py`](../guide_examples/s06/ex_calibration.py)). Invented rows whose statistics are $\chi^2_7$ draws (nothing is fitted): 500 datasets, 3 failed replicates and 1 failed dataset. The second part repeats a cell 2,000 times for an exactly calibrated procedure (true rate 5%) and a clearly liberal one (10%).

```python
example = rows(500, rng, fail=3) + [{"family": "A", "scenario": "S1", "config": "k1", "seed": 1, "n": 1400,
                                     "error": "RuntimeError('...')"}]
cal = Q.calibration_summary(example)
cells = [Q.calibration_summary(rows(R, rng, inflate))["A:S1:k1"] for _ in range(2000)]
```

```text
{'datasets': 501, 'failed': 1, 'critical_value': 14.7382, 'rejection_rate': 0.0359, 'mcse': 0.0083, 'passes_bradley': True, 'rejection_rate_chi2': 0.0439, 'replicates_failed': 3}
chi-square(7) 95% point for comparison: 14.0671
R = 100, true rate 5%: mean rejection rate 0.0589, sd 0.0328 (reported mcse about 0.0235); passes in 0.728 of 2000 cells (binomial with a known critical value: 0.872)
R = 100, true rate 10%: mean rejection rate 0.1160, sd 0.0480 (reported mcse about 0.0320); passes in 0.194 of 2000 cells (binomial with a known critical value: 0.206)
   calibrated case in theory: mean 1 - ceil(0.95 R)/(R + 1) = 0.0594, sd 0.0333
R = 300, true rate 5%: mean rejection rate 0.0531, sd 0.0181 (reported mcse about 0.0129); passes in 0.888 of 2000 cells (binomial with a known critical value: 0.971)
R = 300, true rate 10%: mean rejection rate 0.1039, sd 0.0268 (reported mcse about 0.0176); passes in 0.136 of 2000 cells (binomial with a known critical value: 0.070)
   calibrated case in theory: mean 1 - ceil(0.95 R)/(R + 1) = 0.0532, sd 0.0183
R = 500, true rate 5%: mean rejection rate 0.0515, sd 0.0141 (reported mcse about 0.0099); passes in 0.938 of 2000 cells (binomial with a known critical value: 0.992)
R = 500, true rate 10%: mean rejection rate 0.1027, sd 0.0210 (reported mcse about 0.0136); passes in 0.087 of 2000 cells (binomial with a known critical value: 0.027)
   calibrated case in theory: mean 1 - ceil(0.95 R)/(R + 1) = 0.0519, sd 0.0140
(78 s)
```

Reading the output:
- The three failed replicates enter the pool as $+\infty$ and raise $\hat c$ slightly; the failed dataset counts as one of the 18 rejections ($18/501=0.0359$).
- The simulated spread of $\hat p$ matches the formula above (0.0328 against 0.0333 at $R=100$) and is about 1.4 times the reported MCSE.
- A calibrated procedure passes a 100-dataset cell in only 73% of repetitions (87% if the critical value were known), a 500-dataset cell in 94%; the protocol's "0.99 with 500 datasets" (§8.6) is the known-critical-value figure.

**Design choices and pitfalls.**
- **A high replicate failure rate makes the calibration look good.** If more than 5% of replicates fail, $\hat c=+\infty$ and only failed datasets reject. `replicates_failed` makes this visible; it should be read beside every rejection rate.
- Non-converged fits are counted and used, as in the real analysis.
- Each cell's critical value comes from its own replicates; nothing is shared across cells or configurations.
- Bradley's lower bound ($\hat p\ge2.5\%$, an over-conservative procedure) is not checked, which follows the protocol.

### 6.30 `h_m1` — the qualification verdict

[`qualification.py:166`](../../src/pdx/qualification.py#L166)

**What it does.** It turns the calibration table into the verdict of protocol §8.7: pass, partial pass (only S4 fails), fail, or not yet decidable (a required cell is missing, or a cell has results from several configurations and none was chosen).

**Inputs → outputs.** The output of `calibration_summary`, the required scenarios (`NULL_A`, `NULL_B`) and optionally `config = {family: key}` → `{"verdict", "failed", "missing", "ambiguous"}`.

**The mathematics.** With $\mathcal C$ the ten required cells and $P_c$ the event "$\hat p_c\le0.075$",

$$
\text{verdict}=\begin{cases}\text{incomplete} & \text{some }c\in\mathcal C\text{ has no result}\\ \text{ambiguous} & \text{some }c\text{ has several configurations}\\ \text{pass} & \bigcap_{c}P_c\\ \text{partial pass} & \text{the failed set is exactly }\{A{:}S4\}\\ \text{fail} & \text{otherwise.}\end{cases}
$$

If the cells are independent, $P(\text{pass})=\prod_cP(P_c)$.

**Worked example** ([`ex_h_m1.py`](../guide_examples/s06/ex_h_m1.py)), invented tables (nothing is fitted), and the chance that a perfectly calibrated H-P passes all ten cells with the protocol's replicate numbers (500 in A:S0 and A:S1, 100 in the six other family A cells, 300 in each family B cell):

```python
print(Q.h_m1(table(fail={"A:S4"})))
print(Q.h_m1(two, config={"A": "kA"}))
print(b[500] ** 2 * b[300] ** 2 * b[100] ** 6)          # per-cell binomial pass probabilities
print(warp[500] ** 2 * warp[300] ** 2 * warp[100] ** 6)  # per-cell pass rates measured in §6.29
```

```text
cells required: ['A:S0', 'A:S0L', 'A:S1', 'A:S1S', 'A:S1T', 'A:S1V', 'A:S4', 'A:S7', 'B:S0', 'B:S1']
all pass        -> {'verdict': 'pass', 'failed': [], 'missing': [], 'ambiguous': []}
only A:S4 fails -> {'verdict': 'partial pass (conditional on linear dynamics)', 'failed': ['A:S4'], 'missing': [], 'ambiguous': []}
A:S4 and A:S1S  -> {'verdict': 'fail', 'failed': ['A:S1S', 'A:S4'], 'missing': [], 'ambiguous': []}
B:S1 missing    -> {'verdict': 'incomplete', 'failed': [], 'missing': ['B:S1'], 'ambiguous': []}
two configs     -> {'verdict': 'ambiguous: several configurations; pass `config`', 'failed': [], 'missing': [], 'ambiguous': ['A:S1']}
   with config  -> {'verdict': 'pass', 'failed': [], 'missing': [], 'ambiguous': []}
P(all ten cells pass | every true rate 5%), binomial only: 0.408
   with the warp-speed critical value's noise:          0.103
```

**Design choices and pitfalls.**
- **A perfectly calibrated procedure fails H-M1 most of the time** with the protocol's replicate numbers: it passes all ten cells with probability 0.41 even with known critical values, and about 0.10 with warp-speed critical values. The six 100-dataset cells dominate (each passes with probability 0.73). This is a property of the decision rule (§8.6–§8.7) rather than of the code, but the code implements it as written (§6.N).
- "incomplete" takes precedence over failures: a table with a missing cell and a failed cell reports "incomplete".
- With several configurations in one cell, the caller must name the protocol run's key; otherwise the verdict is "ambiguous", never a silent choice.

### 6.31 `power_summary` — power against the matching null's critical value

[`qualification.py:189`](../../src/pdx/qualification.py#L189)

**What it does.** For each cell it takes the critical value of S1 of the same family and configuration (the matching null: S2, S2b and S3 are built on S1's truth) and reports the share of datasets whose statistic exceeds it.

**Inputs → outputs.** Rows and the calibration table → per cell `datasets` (without error), `failed`, `critical_value`, `power`, `mcse`.

**The mathematics.** $\hat\pi=\frac{1}{\lvert\mathcal O\rvert}\sum_{r\in\mathcal O}\mathbb 1[T_r>\hat c_{\text{S1}}]$, with $\hat c_{\text{S1}}=+\infty$ when no S1 cell has the same key, and MCSE $\sqrt{\hat\pi(1-\hat\pi)/\lvert\mathcal O\rvert}$. Protocol §8.7 asks for power against one deviation of $2\varepsilon$, which is S2.

**Worked example** ([`ex_power_speed.py`](../guide_examples/s06/ex_power_speed.py)), invented rows (nothing is fitted):

```python
power_rows = [{"family": "A", "scenario": "S2", "config": "k1", "lr": float(rng.noncentral_chisquare(7, 12.0))} for _ in range(100)]
power_rows += [{"family": "A", "scenario": "S2", "config": "k1", "error": "RuntimeError('...')"}] * 10
power_rows += [{"family": "A", "scenario": "S2", "config": "k2", "lr": 50.0}] * 5   # no S1 run with key k2
```

```text
S1 critical value (k1): 14.521
A:S2:k1 {'datasets': 100, 'failed': 10, 'critical_value': 14.521, 'power': 0.71, 'mcse': 0.0454}
A:S2:k2 {'datasets': 5, 'failed': 0, 'critical_value': inf, 'power': 0.0, 'mcse': 0.0}
power if the 10 failed datasets counted as non-rejections: 0.6455
```

**Design choices and pitfalls.**
- **Failed datasets leave the denominator here**, while in the calibration they count as rejections. Both choices err on the side of the test (conservative size, optimistic power). In the example the power would be 0.65 instead of 0.71 if failures counted against it.
- A power cell whose configuration has no S1 run gets power 0 against $+\infty$, without a warning (the `k2` row). This happens with `replicate=False` tasks (§6.22).
- The critical value inherits the noise of the S1 cell's own warp-speed estimate (§6.29), which the MCSE of power ignores.

### 6.32 `speed_summary` — bias and RMSE of log Sp-1

[`qualification.py:203`](../../src/pdx/qualification.py#L203)

**What it does.** For each cell it compares log Sp-1 with the log of the true speed ratio and reports the bias, the RMSE and the median of Sp-3's share.

**Inputs → outputs.** Rows → per cell `datasets`, `bias_log_sp1`, `rmse_log_sp1`, `median_sp3_share`.

**The mathematics.** With $e_r=\log\widehat{\text{Sp-1}}_r-\log s$,

$$
\text{bias}=\frac1R\sum_re_r,\qquad \text{RMSE}=\Big(\frac1R\sum_re_r^2\Big)^{1/2},
$$

and $\text{RMSE}^2=\text{bias}^2+\frac1R\sum_r(e_r-\bar e)^2$. The truth is the ratio of median (equivalently geometric-mean) speeds, which is what Sp-1 targets (§6.6).

**Worked example** ([`ex_power_speed.py`](../guide_examples/s06/ex_power_speed.py)): 200 invented S1 rows with $\log\text{Sp-1}=\log2+0.03+0.1\,\xi$:

```python
null = [{..., "sp1": float(np.exp(np.log(2) + 0.1 * rng.normal() + 0.03)), "true_speed_ratio": 2.0,
         "sp3_share": float(rng.uniform())} for _ in range(200)]
print(Q.speed_summary(null))
```

```text
A:S1:k1 {'datasets': 200, 'bias_log_sp1': 0.0228, 'rmse_log_sp1': 0.1041, 'median_sp3_share': 0.4673}
```

The bias recovers the planted 0.03 within its Monte Carlo error ($0.1/\sqrt{200}=0.007$).

**Design choices and pitfalls.**
- **No coverage is computed.** The module docstring promises "the bias and coverage of the speed estimates", and protocol §8.7 decides how Sp-1 to Sp-3 are reported from their coverage in S0 and S1 (at least 90%), but neither model-based nor bootstrap intervals are evaluated here (§6.N).
- No MCSE is given for the bias or the RMSE, although §8.5 asks for one with every measure; $\mathrm{sd}(e)/\sqrt R$ would do for the bias.
- Sp-2 is not evaluated, and Sp-3 only by its median share.

### 6.33 `panel_summaries` — the fidelity statistics of one panel

[`fidelity.py:11`](../../src/pdx/fidelity.py#L11)

**What it does.** It computes aggregate statistics of a panel, so that the same numbers can be compared between the real panel and simulated ones (protocol §8.2's fidelity dashboard). Every output is an aggregate: quantiles, correlations and shares.

**Inputs → outputs.** `panel`, `quantiles` → a dictionary of nine statistics (lists, so that they can be saved as JSON).

**The mathematics.** Let $\mathcal P=\{(i,k): t_{i,k+1}-t_{ik}\le1.5,\ Y_{ik}\text{ and }Y_{i,k+1}\text{ complete}\}$ be the complete one-step transitions.

| statistic | formula |
|---|---|
| `marginal_quantiles` | $q_p(Y_{\cdot\cdot j})$ over all observed scores, $p\in\{0.01,0.1,0.5,0.9,0.99\}$ (linear interpolation) |
| `lag_correlations` | $\operatorname{corr}(Y_{ikj},Y_{i,k+1,l})$ over $\mathcal P$: row $j$ at $t$, column $l$ at $t+1$ |
| `same_year_correlations` | $\operatorname{corr}(Y_{i,k+1,j},Y_{i,k+1,l})$ over $\mathcal P$ |
| `residual_sd` | standard deviation (divisor $N$) of the residuals of the least-squares fit $Y_{i,k+1}=c+BY_{ik}+r_{ik}$ over $\mathcal P$, all phenotypes pooled |
| `within_person_variance` | $\frac{1}{n'}\sum_i\widehat{\operatorname{Var}}_k(Y_{ikj})$ over participants with at least two visits (divisor: their observed values) |
| `share_missing` | share of missing scores per domain, all visits |
| `phenotype_share` | share of PIGD |
| `visits_quantiles` | 10%, 50% and 90% quantiles of the number of visits |
| `share_treated_last_visit` | share with a positive dose at their last visit |

**Worked example** ([`ex_fidelity.py`](../guide_examples/s06/ex_fidelity.py), first part; a synthetic S1 panel stands in for the real one):

```python
real_panel, _ = scenarios.generate("S1", 600, np.random.default_rng(100))
real = fidelity.panel_summaries(real_panel)
```

```text
/Users/thaipham/Documents/PhD/parkinson/paper 2/phenotype-discovery/src/pdx/fidelity.py:28: RuntimeWarning: Degrees of freedom <= 0 for slice.
  within = np.nanmean([np.nanvar(yi, axis=0) for yi in panel.y if len(yi) > 1], axis=0)
statistics: ['marginal_quantiles', 'lag_correlations', 'same_year_correlations', 'residual_sd', 'within_person_variance', 'share_missing', 'phenotype_share', 'visits_quantiles', 'share_treated_last_visit']
lag correlations (row = domain at t, column = domain at t+1):
 [[ 0.692 -0.025  0.203  0.081]
 [ 0.006  0.778  0.487  0.333]
 [ 0.113  0.339  0.809  0.578]
 [ 0.062  0.195  0.41   0.82 ]]
residual sd: [0.415 0.407 0.386 0.407] | within-person variance: [0.12  0.112 0.13  0.146]
phenotype share: 0.509 | visits quantiles: [4.0, 5.0, 7.0] | treated at last visit: 0.509
```

The lag correlations show the mechanism's direction: axial at $t$ predicts function at $t+1$ (0.487) more than the reverse (0.339). The warning comes from participants with a single observed value in some domain (two visits, one missing), whose variance is NaN and is then skipped by `nanmean`.

**Design choices and pitfalls.**
- The function is not called by any script or test yet.
- Only part of §8.2's dashboard is covered. Missing: autocorrelations beyond lag 1, phenotype stability, treatment initiation and dose beyond the last-visit share, dropout as such, and distributions by era and site.
- Transitions are chosen by a time window (at most 1.5 years), not by grid years, and must be complete in every domain, so one missing domain removes the whole transition.

### 6.34 `compare` — where the real panel sits among simulated panels

[`fidelity.py:39`](../../src/pdx/fidelity.py#L39)

**What it does.** For every statistic and every entry it computes the 2.5% and 97.5% quantiles across simulated panels and says whether the real value lies between them.

**Inputs → outputs.** `real` (one dictionary from `panel_summaries`) and `simulated` (a list of them) → per statistic `real`, `sim_2.5`, `sim_97.5` and `inside` (booleans).

**The mathematics.** With $x^{(1)},\dots,x^{(R)}$ an entry's simulated values and $\hat q_p$ their quantile (linear interpolation), inside $=\mathbb 1[\hat q_{0.025}\le x^{\text{real}}\le\hat q_{0.975}]$. Even for a perfect simulator a real value lands inside with probability about $E[F(\hat q_{0.975})-F(\hat q_{0.025})]$. With $R=20$ the interpolated quantiles sit at order positions 1.475 and 19.525 of 20, so this is about $(19.525-1.475)/21\approx0.86$ per entry.

**Worked example** ([`ex_fidelity.py`](../guide_examples/s06/ex_fidelity.py), second part): 20 simulated S1 panels (the stand-in's own scenario) and 20 S0 panels (no speed difference):

```python
sims = [fidelity.panel_summaries(scenarios.generate(sc, 600, np.random.default_rng(200 + r))[0]) for r in range(20)]
cmp = fidelity.compare(real, sims)
```

```text
20 simulated S1 panels against the stand-in: share of entries inside the 2.5-97.5% envelope
   {'marginal_quantiles': '1.00', 'lag_correlations': '0.62', 'same_year_correlations': '0.75', 'residual_sd': '0.50', 'within_person_variance': '1.00', 'share_missing': '1.00', 'phenotype_share': '1.00', 'visits_quantiles': '1.00', 'share_treated_last_visit': '1.00'}
20 simulated S0 panels against the stand-in: share of entries inside the 2.5-97.5% envelope
   {'marginal_quantiles': '0.35', 'lag_correlations': '0.56', 'same_year_correlations': '0.50', 'residual_sd': '0.00', 'within_person_variance': '0.00', 'share_missing': '1.00', 'phenotype_share': '1.00', 'visits_quantiles': '1.00', 'share_treated_last_visit': '0.00'}
S0 envelope for the 90% quantile of each domain: [0.876 0.893 0.83  0.806] to [0.977 0.984 0.945 0.94 ] | stand-in: [0.965 1.096 1.029 1.001]
```

Reading the output:
- The wrong simulator (S0) is caught: its scores are too low at the top (the stand-in's 90% quantiles lie above the S0 envelope in three domains), its residual and within-person variability are too small (slower PIGD participants move and fluctuate less per year), and fewer participants are treated.
- **The right simulator also "fails" many entries.** Against panels from its own scenario, only 62% of the lag correlations and half the residual standard deviations of the stand-in lie inside, partly by chance (the entries are correlated) and partly because 20 panels give a narrow envelope (0.86 per entry at best).

**Design choices and pitfalls.**
- Many entries are judged at once ($d^2$ correlations, $5d$ quantiles, …), so some will lie outside even for a faithful simulator. WP2's thresholds need to allow for this multiplicity and for the number of simulated panels.
- A NaN real value counts as outside.

### 6.35 `THREAD_VARS`, `_shared`, `_start` and `_run` — the worker side of the parallel runner

[`parallel.py:16`](../../src/pdx/parallel.py#L16), [`parallel.py:20`](../../src/pdx/parallel.py#L20), [`parallel.py:27`](../../src/pdx/parallel.py#L27)

**What it does.** `THREAD_VARS` names the environment variables that set the thread counts of OpenMP, OpenBLAS, Intel's MKL and Apple's vecLib. `_shared` is a dictionary that lives in each worker process. `_start(setup, args)` runs once when a worker starts: it limits PyTorch to one thread and fills `_shared` with `setup(*args)`. `_run((fn, item))` computes one piece, `fn(_shared, item)`.

**Inputs → outputs.** `_start`: a module-level `setup` function and its arguments → nothing (fills `_shared`). `_run`: a pair `(fn, item)` → `fn`'s result.

**The mathematics.** None; this is a cost argument. The shared state of a gate piece (a panel, a fit, prepared tensors) is large, and the items (a start, a parameter to profile, a dataset index) are small. Building the state once per worker costs $W$ setups for $W$ workers, instead of shipping it with each of $N$ items.

**Worked example** ([`ex_parallel.py`](../guide_examples/s06/ex_parallel.py)): what a worker sees.

```python
def threads(shared, item):
    import torch
    return item, os.environ.get("OMP_NUM_THREADS"), torch.get_num_threads(), os.getpid()
print("workers see:", list(parallel.ordered(threads, ["a", "b"], 2, parallel.check_setup, (1.0,))))
```

```text
workers see: [('a', '1', 1, 3931), ('b', '1', 1, 3932)]
```

Each worker is single-threaded, both through the environment variable and through PyTorch's own setting.

**Design choices and pitfalls.**
- `fn` and `setup` must be module-level functions, because "spawn" pickles them by name and the worker imports them afresh. A function defined in a script works only if the script has an `if __name__ == "__main__":` guard (the worker imports the script as `__mp_main__`).
- `args` are pickled once per worker, so memory grows with the number of workers.
- If `setup` raises inside a worker, CPython's pool keeps replacing the dying workers and the results never arrive. Running the same call with `jobs=1` first shows the error directly.

### 6.36 `ordered` — independent pieces in single-threaded workers, results in order

[`parallel.py:32`](../../src/pdx/parallel.py#L32)

**What it does.** It is a generator that yields `fn(shared, item)` for each item, in the order of the items. With `jobs <= 1` (or a single item) it runs everything in the calling process. Otherwise it sets the four thread variables to 1, starts a pool of `min(jobs, len(items))` workers with "spawn", restores the variables at once, and yields the workers' results in item order. Closing the generator early ends the workers immediately.

**Inputs → outputs.** `fn`, `items`, `jobs`, `setup`, `args` → a generator of results.

**The mathematics.** Let $r_j=f(\text{shared},x_j)$. If $f$ is a deterministic function of its arguments, which is the case when every piece carries its own seed (the gate's recovery datasets use `default_rng(seed + r)`, the bootstrap `default_rng([seed, 3, i])`), the sequence $(r_1,\dots,r_N)$ does not depend on how many workers there are or on which worker ran which piece. `pool.imap` returns results in input order, so a consumer that stops at a data-dependent point (Besag–Clifford's sequential rule, a checkpoint saved after each piece) sees the same sequence as a serial run.

A single job runs in the calling process with whatever threads that process has (by default several, with likelihood batches of 128 participants); a worker runs one thread with batches of 32 (`likelihood.default_chunk`, §2). Floating-point sums in a different order differ in the last digits, so the optimiser's path and the fitted values agree between the two to the fits' tolerance (the module docstring says below $10^{-4}$ in a test statistic), not bit for bit. Across any number of workers ($\ge2$) the results are identical.

**How the code was formed.**
1. *Why processes, and single-threaded ones.* The likelihood is limited by memory traffic, so several single-threaded jobs finish more fits per hour than one process with many threads (`scripts/bench_threads.py`, measured on 4 November 2026).
2. *Why "spawn".* It is the only start method that works on Windows and the safe one on macOS; a forked child of a process that already runs PyTorch threads can deadlock.
3. *Why set the variables before the pool and restore them after.* Numerical libraries read their thread variables when they are first imported, in the child. Setting them in `os.environ` just before the pool starts (lines 45–46) makes the children start single-threaded; restoring them right after (lines 49–54) leaves the parent unchanged.
4. *Why a generator.* Results can be checkpointed one at a time as they arrive, and the consumer can stop early: `close()` raises `GeneratorExit` at the `yield`, and the `finally` clause terminates the pool (lines 57–59).

**Worked example** ([`ex_parallel.py`](../guide_examples/s06/ex_parallel.py), run without `OMP_NUM_THREADS` in the shell):

```python
print("jobs=2:", list(parallel.ordered(parallel.check_square, [1, 2, 3, 4, 5], 2, parallel.check_setup, (10.0,))))
runs = {j: list(parallel.ordered(draw, range(6), j, seed_setup, (100,))) for j in (1, 2, 3)}   # draw: default_rng(seed + r)
gen = parallel.ordered(parallel.check_square, list(range(50)), 3, parallel.check_setup, (1.0,))
print("first result:", next(gen)); gen.close()
```

```text
parent pid 3927 | OMP_NUM_THREADS in the parent: None
jobs=2: [(10.0, 3929), (40.0, 3929), (90.0, 3929), (160.0, 3929), (250.0, 3929)] (0.6 s)
jobs=1: [(10.0, 3927), (40.0, 3927), (90.0, 3927)]
workers see: [('a', '1', 1, 3931), ('b', '1', 1, 3932)]
parent afterwards: None
per-piece seeds, jobs 1/2/3 identical: True [-1.1575 -0.7902  0.6257  1.1033  0.5621 -0.0635]
first result: (0.0, 3940)
generator closed; the pool was terminated
empty items: []
```

Reading the output:
- With two workers the results come back in item order. Here one worker happened to take all five tiny items before the other was ready; which worker runs which item is a matter of timing.
- With `jobs=1` everything runs in the parent (its process id).
- The parent's unset `OMP_NUM_THREADS` is unset again afterwards.
- Pieces with their own seeds give identical results with 1, 2 or 3 jobs.

**Design choices and pitfalls.**
- **No cap on `jobs`.** `run_tasks` caps the qualification at four workers (§16), but `ordered` does not, and `scripts/run_engine_gate.py`'s docstring shows `--jobs 5`.
- Results are delivered in order, so one slow piece holds back the delivery (and the checkpointing) of later pieces that have already finished.
- A worker that dies is replaced by the pool with the parent's restored environment, possibly without the thread variables (`_start` still sets PyTorch to one thread).

### 6.37 `check_setup` and `check_square` — the test pieces

[`parallel.py:62`](../../src/pdx/parallel.py#L62), [`parallel.py:67`](../../src/pdx/parallel.py#L67)

**What it does.** `check_setup(k)` returns `{"k": k, "pid": os.getpid()}`, recording which process built the shared state. `check_square(shared, x)` returns $(kx^2,\ \text{pid})$.

**Inputs → outputs.** See above.

**The mathematics.** $r_j=k\,x_j^2$, so the expected results are known exactly, and the process id shows where each piece ran.

**Worked example.** §6.36's first two lines. `tests/test_hp.py::test_parallel_runs_give_the_results_in_order_from_several_workers` checks the order, that workers are other processes and that `jobs=1` stays in the parent.

**Design choices and pitfalls.** They live in `parallel.py` (not in the tests) because spawned workers must import them from an installed module.

### 6.38 `check_reference` — the likelihood against an independent implementation

[`gate.py:39`](../../src/pdx/gate.py#L39)

**What it does.** It evaluates the engine's log-likelihood and the slow reference implementation (`engine/reference.py`) at three random points for each kind of model (speed-only, free, joint) and reports the worst relative difference. It passes below $10^{-8}$.

**Inputs → outputs.** A panel, a specification and a generator → `{"max_relative_difference", "pass"}`.

**The mathematics.** With $x_m=x^{\text{start}}_{\text{kind}}+0.05\,\xi_m$, $\xi_m\sim N(0,I)$ (start values moved so that every parameter is active),

$$
\delta_{\text{ref}}=\max_{\text{kind}}\max_{m\le3}\frac{\lvert\ell_{\text{engine}}(x_m)-\ell_{\text{ref}}(x_m)\rvert}{\max\{1,\lvert\ell_{\text{ref}}(x_m)\rvert\}}<10^{-8}.
$$

The reference loops over participants, quadrature nodes and visits; it uses scipy's matrix exponential with Van Loan's blocks (not the engine's binary-power composition), scipy's Lyapunov solver, and selects the observed components explicitly instead of masking (§2).

**Worked example** ([`ex_gate_reference_gradient.py`](../guide_examples/s06/ex_gate_reference_gradient.py)), on the gate's small dataset (S1, 40 simulated participants, `default_rng([2026, 0])`):

```python
rng = lambda k: np.random.default_rng([2026, k])          # as in run_gate
small, info = scenarios.generate("S1", 40, rng(0), d=4)
print("check_reference:", gate.check_reference(small, spec, rng(1)))
```

```text
small dataset: 35 participants, 151 transitions
check_reference: {'max_relative_difference': 3.160129892869025e-15, 'pass': True} (1.3 s)
   one point, null model: engine -353.8558383272, reference -353.8558383272, difference -6.3e-13
```

The two implementations agree to rounding ($3\times10^{-15}$ relative).

**Design choices and pitfalls.**
- **What the reference shares with the engine is not checked by it.** Both use `prepare` (the plug-in prior of the landmark state and the centred landmark scores) and `unpack_numpy` (the parameter transforms: $-e^{\text{raw}}$ for own rates, the Cholesky factor of $Q$). An error there would be reproduced by both.
- The check runs at 35 participants only; it tests the formulas, not the behaviour at the real size.

### 6.39 `check_gradient` — exact gradients against central differences

[`gate.py:52`](../../src/pdx/gate.py#L52)

**What it does.** For the speed-only and free models it compares the gradient from automatic differentiation with central finite differences on 20 coordinates chosen at random, and passes below a worst relative difference of $10^{-4}$.

**Inputs → outputs.** A panel, a specification, a generator, `n_coords` (20), `h` ($10^{-6}$) → `{"max_relative_difference", "pass"}`.

**The mathematics.** `value_and_grad` returns $f=-\ell$ and $g=\nabla f$. For a coordinate $j$,

$$
\widehat{\partial_jf}=\frac{f(x+he_j)-f(x-he_j)}{2h},\qquad \delta_{\text{grad}}=\max_j\frac{\lvert\widehat{\partial_jf}-g_j\rvert}{\max\{1,\lvert g_j\rvert\}}<10^{-4}.
$$

The central difference has truncation error $\tfrac{h^2}{6}\partial_j^3f$ and rounding error of order $\epsilon_{\text{mach}}\lvert f\rvert/h$, so the error falls like $h^2$ for large $h$ and rises like $1/h$ for small $h$.

**Worked example** ([`ex_gate_reference_gradient.py`](../guide_examples/s06/ex_gate_reference_gradient.py)), the gate's check and the same check at other step sizes:

```python
print("check_gradient (h = 1e-6):", gate.check_gradient(small, spec, rng(2)))
for h in (1e-2, 1e-4, 1e-8, 1e-10):
    r = gate.check_gradient(small, spec, rng(2), h=h)
```

```text
check_gradient (h = 1e-6): {'max_relative_difference': np.float64(6.675959472307983e-08), 'pass': np.True_} (1.6 s)
   h = 1e-02: worst relative difference 6.1e-03, pass False
   h = 1e-04: worst relative difference 6.1e-07, pass True
   h = 1e-08: worst relative difference 9.2e-06, pass True
   h = 1e-10: worst relative difference 2.5e-04, pass False
```

Reading the output: from $h=10^{-2}$ to $10^{-4}$ the error falls by $10^4$, the $h^2$ of truncation; below $10^{-6}$ rounding takes over. The default $h=10^{-6}$ sits near the bottom of the V.

**Design choices and pitfalls.**
- 20 of 112 (null) and 119 (free) coordinates are checked, chosen at random, and the joint model is not checked. A wrong gradient in a block that is rarely drawn could pass.
- The returned values are numpy types (`np.float64`, `np.True_`); `plain` converts them for the checkpoint (§6.50).

### 6.40 `check_quadrature` — the log-likelihood at 12, 20 and 30 nodes

[`gate.py:67`](../../src/pdx/gate.py#L67)

**What it does.** It re-evaluates a fit's log-likelihood, at its fitted parameters, with 12, 20 and 30 Gauss–Hermite nodes, and passes if the spread is below 0.01.

**Inputs → outputs.** A `Fit` and its panel → `{"loglik": {"12": …, "20": …, "30": …}, "spread", "pass"}`.

**The mathematics.** The engine integrates each participant's likelihood over the random part of log-speed (§2):

$$
\ell_K(\hat x)=\sum_i\log\sum_{m=1}^Kw_m\,p\big(y_i\mid u_m;\hat x\big),\qquad \text{spread}=\max_K\ell_K(\hat x)-\min_K\ell_K(\hat x)<0.01,
$$

with $u_m=\sqrt2\,\xi_m$ and $w_m=\omega_m/\sqrt\pi$ from the Hermite rule $(\xi_m,\omega_m)$. The parameters are not refitted at 20 or 30 nodes: the check measures the integration error at the fitted point.

**Worked example.** See §6.41, which calls this function for both models.

**Design choices and pitfalls.**
- The integration error is a sum over participants, so a fixed absolute threshold (0.01) tends to become harder to meet as the panel grows and as participants have more visits (each visit sharpens a participant's integrand in $u$). §6.41's example fails it at 113 participants.

### 6.41 `check_quadrature_lr` — the quadrature check with H-P's statistic

[`gate.py:78`](../../src/pdx/gate.py#L78)

**What it does.** From the speed-only fit it fits the free model, started at the speed-only solution, and evaluates both log-likelihoods and H-P's statistic at 12, 20 and 30 nodes. The pass flag is that of the speed-only log-likelihood's spread; the free model's spread and the statistic's spread are reported.

**Inputs → outputs.** `f0` (a speed-only fit), the panel, `fit_kw`, `nodes` → the keys of §6.40 for the speed-only fit, plus `free_loglik`, `free_spread`, `lr` (by node count) and `lr_spread`.

**The mathematics.**

$$
\mathrm{LR}_K=2\big\{\ell^{\text{free}}_K(\hat x_1)-\ell^{\text{null}}_K(\hat x_0)\big\},\qquad \text{LR spread}=\max_K\mathrm{LR}_K-\min_K\mathrm{LR}_K .
$$

If the integration error were the same for both models it would cancel in $\mathrm{LR}_K$; the docstring calls the statistic "the result that has to agree" (§16). Since $\lvert\Delta\mathrm{LR}\rvert\le2(\text{spread}_{\text{null}}+\text{spread}_{\text{free}})$, the statistic is also bounded by the two spreads.

**Worked example** ([`ex_gate_fit_checks.py`](../guide_examples/s06/ex_gate_fit_checks.py)), the S1 dataset of `tests/test_gate.py` (120 simulated, 113 analysed participants):

```python
f0 = fit_model(panel, spec, "null")
q = gate.check_quadrature_lr(f0, panel)
```

```text
speed-only fit: 113 participants, loglik -792.5895, 437 evaluations (21 s)
null loglik by nodes: {'12': -792.5895, '20': -792.5677, '30': -792.5688} spread 0.0218 pass False
free loglik by nodes: {'12': -790.9113, '20': -790.8727, '30': -790.8749} spread 0.0386
LR by nodes: {'12': 3.3565, '20': 3.3902, '30': 3.3877} spread 0.0337 (9 s)
```

Reading the output:
- **The check fails at 113 participants**: the 12-node log-likelihood is 0.022 below the 20- and 30-node values, which agree with each other to 0.001.
- The errors do not cancel in the statistic here (spread 0.034), but 0.034 is tiny against the statistic's scale (a $\chi^2_7$ critical value of 14).

**Design choices and pitfalls.**
- **The pass flag ignores the statistic.** The automatic verdict (§6.54) uses the speed-only spread only; neither `free_spread` nor `lr_spread` has a threshold, although the docstring names the statistic as what must agree.
- **The 0.01 threshold is likely to fail at the real size**, since the 12-node error is a sum over participants and tends to grow with their number and their visits. A criterion on the statistic relative to its critical value, or more nodes, would need to be decided (§6.N).
- The free fit here starts only at the speed-only solution (no second start), unlike `fit_pair`.

### 6.42 `start_setup` and `start_fit` — one fit from one start, for the workers

[`gate.py:89`](../../src/pdx/gate.py#L89), [`gate.py:95`](../../src/pdx/gate.py#L95)

**What it does.** `start_setup` builds the shared state of the dispersed-start workers: the panel, the specification, the fit options and the prepared likelihood tensors. `start_fit(shared, (kind, x0))` fits one model from one start and returns its final log-likelihood (`x0 = None` means the default start).

**Inputs → outputs.** As above → a dictionary; a float.

**The mathematics.** $\hat\ell(\text{kind},x_0)=\max\ell$ reached by L-BFGS-B from $x_0$ (`fit_model`, §3).

**Worked example** ([`ex_gate_dispersed_starts.py`](../guide_examples/s06/ex_gate_dispersed_starts.py), last line):

```python
shared = gate.start_setup(panel, spec, None)
gate.start_fit(shared, ("null", None))
```

```text
start_setup keys: ['fit_kw', 'panel', 'prep', 'spec'] | start_fit(null, default start): -522.587591
```

It equals the first speed-only log-likelihood of §6.43.

**Design choices and pitfalls.** Preparing the tensors once per worker saves repeating `prepare` for every start.

### 6.43 `check_dispersed_starts` — do fits converge from dispersed starts?

[`gate.py:101`](../../src/pdx/gate.py#L101)

**What it does.** It fits the speed-only and free models from their default start and from `n_starts` random starts around it, and passes if, for each model, the final log-likelihoods are within 0.1 of each other. It can resume from fits already done and report progress after each fit.

**Inputs → outputs.** Panel, specification, generator, `n_starts` (5), `scale` (0.3), `fit_kw`, `done` (log-likelihoods already found), `on_fit` (called after each new fit), `jobs` → `{"null": {"logliks", "spread"}, "free": {…}, "pass"}`.

**The mathematics.** For each kind, $x^{(0)}$ is the default start and $x^{(m)}=x_{\text{def}}+0.3\,\xi_m$, $\xi_m\sim N(0,I)$, $m=1,\dots,n_{\text{starts}}$, in raw parameter units (so log-scale parameters such as $\log\sigma$, $\log\tau$ and the log own rates are perturbed by a factor of about $e^{\pm0.3}$). Then

$$
\text{spread}_{\text{kind}}=\max_m\hat\ell\big(x^{(m)}\big)-\min_m\hat\ell\big(x^{(m)}\big),\qquad \text{pass}\iff\text{spread}_{\text{null}}<0.1\ \text{and}\ \text{spread}_{\text{free}}<0.1 .
$$

**How the resume works.** Lines 108–112 draw *all* the random starts (null first, then free) before any fit, and then drop the ones already done (`starts[len(done[kind]):]`). The generator is consumed in the same way whether or not the run is resumed, so a resumed run fits exactly the starts the uninterrupted run would have fitted. The fits then run through `parallel.ordered` in queue order, and `on_fit(done)` is called after each one, which is where `run_gate` saves the checkpoint.

**Worked example** ([`ex_gate_dispersed_starts.py`](../guide_examples/s06/ex_gate_dispersed_starts.py)), the dataset of `tests/test_gate.py` (80 simulated, 75 analysed participants), one dispersed start per model, then a run resumed from the second save:

```python
full = gate.check_dispersed_starts(panel, spec, np.random.default_rng(1), n_starts=1,
                                   on_fit=lambda done: seen.append({k: list(v) for k, v in done.items()}))
resumed = gate.check_dispersed_starts(panel, spec, np.random.default_rng(1), n_starts=1, done=seen[1])
```

```text
full run (38 s):
   null: logliks [-522.587591 -527.55062 ], spread 4.96e+00
   free: logliks [-520.90589  -520.554593], spread 3.51e-01
   pass: False
progress saved after each fit: [{'null': 1, 'free': 0}, {'null': 2, 'free': 0}, {'null': 2, 'free': 1}, {'null': 2, 'free': 2}]
resumed from the second save (19 s): identical to the full run: True
start_setup keys: ['fit_kw', 'panel', 'prep', 'spec'] | start_fit(null, default start): -522.587591
```

Reading the output:
- The resume works: from the second save only the two free fits are run, and the result is identical (this is what `tests/test_gate.py::test_dispersed_starts_resumed_midway_give_the_same_fits` checks).
- **The check itself fails on this small dataset**, in two different ways:
  - the speed-only fit from the dispersed start stops 5 log-likelihood units lower. [`ex_note_starts.py`](../guide_examples/s06/ex_note_starts.py) shows why: it ends with TD's speed spread at $\log\sigma_0=-14.5$, i.e. $\sigma_0\approx0$, where the likelihood is flat in $\log\sigma_0$, and a restart from there stays put;
  - the free fit from the dispersed start is 0.35 *higher* than from the default start, and higher than both of `fit_pair`'s starts, so the statistic on this dataset would be 4.07 instead of 3.36.

```text
null, default start:   loglik -522.5876, log_sigma[0] = -0.660
null, dispersed start: loglik -527.5506, log_sigma[0] = -14.509 (CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH)
   restarted where it stopped: loglik -527.5506 after 3 evaluations
free, fit_pair's two starts: [-520.9059 -520.9059]; a dispersed start: -520.5546
LR from fit_pair 3.363; with the dispersed free optimum 4.066
```

**Design choices and pitfalls.**
- **Spreads on the log scale can trap the optimiser.** As $\sigma\to0$ (or $\tau\to0$), $\partial\ell/\partial\log\sigma\to0$, so a start that pushes a spread towards zero can end on a flat edge that L-BFGS-B never leaves. A failed check may therefore reflect this edge rather than genuinely separate optima. A lower bound on the log-spreads during fitting, or restarting a spread found at zero, would avoid it (§6.N).
- Seventy-five participants for 112–119 parameters is far below the real size; the gate runs this check on the real-size dataset, where the surface should be better behaved. The point of the example is what the check can reveal.
- The free model's dispersed starts are drawn around its *own* default start, not around the speed-only solution.

### 6.44 `profile_setup` and `profile_job` — one profile, for the workers

[`gate.py:125`](../../src/pdx/gate.py#L125), [`gate.py:130`](../../src/pdx/gate.py#L130)

**What it does.** `profile_setup` holds the fit, the panel, the grid size and the fit options. `profile_job(shared, (name, index, se))` runs `profile` for one parameter and returns it as plain JSON values.

**Inputs → outputs.** As above → a dictionary; the profile of §6.45.

**The mathematics.** See §6.45.

**Worked example** ([`ex_gate_profile.py`](../guide_examples/s06/ex_gate_profile.py)): the profile of $\gamma_1$ below goes through `profile_job`.

```python
shared = gate.profile_setup(f0, panel, 5, None)             # what each parallel worker holds
p = gate.profile_job(shared, ("gamma", 0, float(se[j])))
```

```text
profile_setup keys: ['fit', 'fit_kw', 'n_grid', 'panel']
```

**Design choices and pitfalls.** The likelihood tensors are prepared inside `profile` for each parameter, not once per worker as for the dispersed starts; this costs little next to the $n_{\text{grid}}$ constrained fits.

### 6.45 `profile` — the profile log-likelihood of one parameter

[`gate.py:137`](../../src/pdx/gate.py#L137)

**What it does.** It fixes one raw parameter at each point of a grid of ±4 standard errors around its estimate, refits all the others, and calls the parameter identified if the profile falls by at least 1.92 on both sides.

**Inputs → outputs.** A fit, its panel, the block `name` and `index` of the parameter, its standard error `se`, `half_width_se` (4), `n_grid` (9), `fit_kw` → `{"parameter", "grid", "loglik", "se", "identified"}`.

**The mathematics.** For coordinate $j$ with estimate $\hat x_j$,

$$
\ell_p(v)=\max_{x:\,x_j=v}\ell(x),\qquad v\in\hat x_j+\operatorname{linspace}(-4\,\widehat{\mathrm{SE}}_j,\ 4\,\widehat{\mathrm{SE}}_j,\ n_{\text{grid}}),
$$

each maximum found by L-BFGS-B with the bounds $x_j\in[v,v]$, started at $\hat x$ with $x_j$ replaced by $v$ (lines 147–153). If $\widehat{\mathrm{SE}}_j$ is not finite or not positive, 1 is used instead. With $\ell^{\text{top}}=\max\{\max_v\ell_p(v),\hat\ell\}$ and the middle grid point excluded,

$$
\text{identified}\iff\ell^{\text{top}}-\min_{v<\hat x_j}\ell_p(v)\ge1.92\ \text{and}\ \ell^{\text{top}}-\min_{v>\hat x_j}\ell_p(v)\ge1.92 .
$$

$1.92=\chi^2_{1,0.95}/2=3.841/2$: a drop of 1.92 is where the 95% profile-likelihood interval $\{v:2(\hat\ell-\ell_p(v))\le3.84\}$ ends, so the rule says that interval is closed on both sides within ±4 standard errors. For a quadratic profile the drop at $\pm k$ standard errors is $k^2/2$: 2 at ±2, 8 at ±4.

**Worked example** ([`ex_gate_profile.py`](../guide_examples/s06/ex_gate_profile.py)), the speed ratio and TD's load spread on tremor, on the S1 dataset of §6.41:

```python
p = gate.profile_job(shared, ("gamma", 0, float(se[j])))                      # n_grid 5
p = gate.profile(f0, panel, "log_tau", 0, float(se[j]), n_grid=3)              # coarser, to save time
```

```text
profile_setup keys: ['fit', 'fit_kw', 'n_grid', 'panel']
profile of gamma[0] (estimate 0.980, se 0.227): grid [0.071 0.525 0.98  1.434 1.889]
   drop from the top [8.224 2.023 0.    1.888 6.98 ] -> identified True
profile of log_tau[0] (estimate -2.215, se 0.400): grid [-3.813 -2.215 -0.617]
   drop from the top [ 0.71   0.    21.436] -> identified False
(55 s for the two profiles)
```

Reading the output:
- $\gamma_1$ is identified, and nearly quadratic: the drops at ±2 standard errors are 2.02 and 1.89, at ±4 they are 8.2 and 7.0.
- TD's load spread on tremor is not: towards $\tau\to0$ the profile falls by only 0.71 at $-4$ standard errors (from $\tau=0.11$ to 0.022), because the likelihood flattens as a variance goes to zero. Upwards it is sharply determined. At the real size the drop to $\tau=0$ grows with the number of participants, so this may pass there.

**Design choices and pitfalls.**
- The grid is in the raw parameter (log of a spread, log of a Cholesky diagonal, …), so "±4 standard errors" is a multiplicative range for spreads.
- With an even `n_grid` the middle index is not the estimate and the two sides are unequal; the gate uses 9 (or 5 in quick mode).
- A spread estimated at zero has a huge standard error from the pseudo-inverse (§6.46), so its grid would be very wide. `run_gate` does not set such spreads aside before profiling (§6.53).

### 6.46 `identification` — the conditioning of the observed information

[`gate.py:161`](../../src/pdx/gate.py#L161)

**What it does.** It computes the observed information (the Hessian of minus the log-likelihood) at a fit, reports its smallest eigenvalue, condition number, positive definiteness and nearly flat directions, lists the spreads estimated at zero, recomputes the condition number without them, and returns the largest correlation between estimates and the standard errors.

**Inputs → outputs.** A fit, its panel, optionally a precomputed `H`, `boundary` (0.01) → the report of `covariance_checked` (§3), plus `spreads_at_zero`, `condition_number_without_them`, `well_conditioned_without_them`, `max_abs_correlation` and `se`.

**The mathematics.** $H=-\nabla^2\ell(\hat x)$ by central differences of the exact gradient (`engine.fit.hessian`, §3). With $H=V\Lambda V^\top$,

$$
\kappa(H)=\frac{\lambda_{\max}}{\min\{\lambda_k:\lambda_k>0\}},\qquad \text{well conditioned}\iff\lambda_{\min}>0\ \text{and}\ \kappa<10^{10},
$$

and the covariance is the pseudo-inverse $\hat V=\sum_{\lambda_k>0}\lambda_k^{-1}v_kv_k^\top$, with standard errors $\sqrt{\hat V_{jj}}$ and correlations $\hat V_{jk}/\sqrt{\hat V_{jj}\hat V_{kk}}$. A spread is "at zero" when its raw value is below $\log0.01$ (`log_tau` or `log_sigma` coordinates, lines 168–169). Why they are set aside: $\partial\ell/\partial\log\tau=2\tau^2\,\partial\ell/\partial\tau^2\to0$ as $\tau\to0$, and so does the second derivative, so such a coordinate has almost no information and inflates $\kappa$ although the other parameters can be well determined. $\kappa$ is then recomputed for $H$ restricted to the other coordinates (lines 170–171).

**Worked example** ([`ex_gate_fit_checks.py`](../guide_examples/s06/ex_gate_fit_checks.py)), the speed-only fit of §6.41:

```python
ident = gate.identification(f0, panel)
se = ident.pop("se")
```

```text
identification (12 s): condition number 9.81e+13, positive definite True, well conditioned False
   spreads at zero: ['log_tau[1]', 'log_tau[2]', 'log_tau[3]', 'log_tau[4]', 'log_tau[5]'] | without them: 9.51e+03 True | max |correlation| 0.83
   gamma[0]: estimate 0.9799, se 0.2273
   log_tau[0]: estimate -2.2149, se 0.3996
   flat directions (eigenvalue below max/1e10): 4 | first: {'eigenvalue': 8.333805457625569e-11, 'coordinates': [('log_tau[1]', -1.0), ('log_tau[3]', -0.0), ('log_tau[2]', -0.0)]}
```

Reading the output:
- Five load spreads are at zero on this small dataset (TD's on axial, function and independence, PIGD's on tremor and axial; `log_tau` is stored phenotype by phenotype). With them the condition number is $10^{14}$; without them it is $9.5\times10^3$, well conditioned.
- The first flat direction is exactly the coordinate `log_tau[1]`.
- The standard error of $\gamma_1$, 0.23, is the one the profile of §6.45 uses.

**Design choices and pitfalls.**
- The threshold $10^{10}$ (from `covariance_checked`) is lenient; a condition number of $10^9$ still passes.
- The spreads at zero are only listed; the standard errors of the remaining parameters still come from the pseudo-inverse of the full $H$.

### 6.47 `recovery_setup` and `recovery_dataset` — one dataset of the recovery check

[`gate.py:181`](../../src/pdx/gate.py#L181), [`gate.py:188`](../../src/pdx/gate.py#L188)

**What it does.** `recovery_setup` holds the settings of a recovery check. `recovery_dataset(shared, r)` simulates dataset $r$ with its own seed, fits both models with `fit_pair`, and compares the estimates with the truth: under a shared mechanism the speed-only fit (log speed ratio, reference drift, and optionally whether 95% intervals cover the true log speed ratio and log speed spreads); under a changed link the free fit (both phenotypes' drifts).

**Inputs → outputs.** `(scenario, n, seed, d, fit_kw, with_intervals, sim_kw)`; `r` → a row: `lr`, `converged`, `default_start_gain`, and `error_log_speed`, `error_A`, `covered_log_speed`, `covered_log_sigma` (null scenarios) or `error_A`, `error_A_other` (otherwise).

**The mathematics.** Dataset $r$ uses `default_rng(seed + r)`. With $\mathcal S$ the support handed to the analysis:
- statistic: $\mathrm{LR}=2(\hat\ell_{\text{free}}-\hat\ell_{\text{null}})$, not clamped, so a negative value would show;
- the free fit's two starts give $\hat\ell^{(\text{null start})}$ and $\hat\ell^{(\text{own start})}$; `default_start_gain` $=\hat\ell^{(\text{own start})}-\hat\ell^{(\text{null start})}$ measures what the second start adds (positive: the null-written start alone would have understated LR);
- null scenarios (lines 199–209): $e_\gamma=\hat\gamma_1-\log s$; $e_A=\hat A_{\mathcal S}-A_{\mathcal S}$ (the speed-only fit's reference drift against TD's); with intervals, $\hat V$ from the Hessian of the speed-only fit and

$$
\text{covered}_\gamma=\mathbb 1\big[\lvert e_\gamma\rvert\le1.96\sqrt{\hat V_{\gamma\gamma}}\big],\qquad \text{covered}_{\sigma_g}=\mathbb 1\big[\lvert\log\hat\sigma_g-\log\sigma_g\rvert\le1.96\sqrt{\hat V_{\log\sigma_g}}\big];
$$

- changed-link scenarios (lines 211–213): $e_A=\hat A^{\text{free}}_{0,\mathcal S}-A_{\mathcal S}$ and $e_{A,\text{other}}=\hat A^{\text{free}}_{1,\mathcal S}-s\,(A_{\text{PIGD}})_{\mathcal S}$. The free model has no speed parameter, so its PIGD drift absorbs the speed ratio.

**What is the truth for $\gamma_1$?** §6.6 showed that in family A the landmark carries information about the speed. The engine's $\gamma_1$ is a contrast at equal centred landmark scores, whose value in the simulator (by a large-sample regression of the simulated log-speeds) is about 0.65 at $d=4$ and 0.66 at $d=10$, not $\log2=0.693$. The same regression puts TD's speed at the pooled-mean landmark at $e^{0.02}$, which the reference drift absorbs. So `error_log_speed` has an expected offset of about $-0.03$ to $-0.04$ even for a perfect estimator, and `covered_log_speed` is computed around the wrong centre. Sp-1, which the qualification uses (§6.32), adds $\hat\theta^\top(\bar{\tilde y}_{\text{PIGD}}-\bar{\tilde y}_{\text{TD}})$ back and targets $\log s$.

**Worked example** ([`ex_gate_recovery.py`](../guide_examples/s06/ex_gate_recovery.py)), one S1 dataset with intervals and one S2 dataset, 60 simulated participants each:

```python
shared = gate.recovery_setup(scenario, 60, 3, 4, None, intervals, None)
row = gate.recovery_dataset(shared, 0)                 # dataset 0: seed 3 + 0
```

```text
S1 row (27 s):
   lr                     10.764
   converged              True
   default_start_gain     2.4244
   error_log_speed        -0.2748
   error_A                [-0.0537 -0.1205  0.189   0.2909 -0.1833  0.0887  0.1643 -0.2422]
   covered_log_speed      True
   covered_log_sigma      [True, True]
S2 row (39 s):
   lr                     14.9397
   converged              True
   default_start_gain     3.771
   error_A                [ 0.0938  0.1406  0.0616  0.1897 -0.2555 -0.1146 -0.0489 -0.1855]
   error_A_other          [-0.2629 -0.2995  0.3547  0.0669  0.047   0.3786 -0.0294 -0.3843]
```

(The summary lines of the same run are in §6.48.) With 55 participants the errors are large; the point is the bookkeeping. Note `default_start_gain`: on these tiny datasets the free model's own default start ended 2.4 and 3.8 log-likelihood units above the start at the speed-only solution, which is why `fit_pair` tries both.

**Design choices and pitfalls.**
- `error_A` lists the support entries row by row (`spec.support` order), own rates included.
- Bias is judged on $\gamma_1$ against $\log s$; see the estimand caveat above and §6.N.
- The intervals are model-based (Wald), from the pseudo-inverse of the Hessian.

### 6.48 `recovery` — the recovery check over datasets

[`gate.py:217`](../../src/pdx/gate.py#L217)

**What it does.** It runs `recovery_dataset` for datasets not yet done (in parallel if `jobs > 1`), calls `on_row(rows)` after each one (for the checkpoint), and summarises: mean absolute drift error, smallest LR, non-converged fits, largest gain from the free model's own start, bias and RMSE of the log speed ratio, coverages, and whether any LR is negative.

**Inputs → outputs.** `scenario`, `n`, `n_datasets`, `seed`, `d`, `fit_kw`, `with_intervals`, `sim_kw`, `rows` (done earlier), `on_row`, `jobs` → a summary dictionary.

**The mathematics.** Over the $R$ datasets,

$$
\overline{\lvert e_A\rvert}=\frac{1}{R\,n_{\mathcal S}}\sum_r\sum_{(j,k)\in\mathcal S}\lvert e_{A,r,jk}\rvert,\quad \text{bias}_\gamma=\bar e_\gamma,\quad \text{RMSE}_\gamma=\Big(\frac1R\sum_re_{\gamma,r}^2\Big)^{1/2},\quad \text{coverage}=\frac1R\sum_r\text{covered}_r,
$$

the speed-spread coverage being averaged over datasets and both phenotypes, and `pass_no_negative_lr` $=\min_r\mathrm{LR}_r>-10^{-6}$. With $R=20$, a coverage estimate has MCSE $\sqrt{0.95\times0.05/20}\approx0.05$, and the bias of $\gamma_1$ has MCSE $\mathrm{sd}(e_\gamma)/\sqrt{20}$.

**Worked example** ([`ex_gate_recovery.py`](../guide_examples/s06/ex_gate_recovery.py)): the rows of §6.47 summarised (passing `rows=[row]`, so nothing is refitted):

```python
out = gate.recovery(scenario, 60, 1, seed=3, d=4, with_intervals=intervals, rows=[row])   # nothing left to fit
```

```text
S1 summary: {'scenario': 'S1', 'datasets': 1, 'n': 60, 'mean_abs_error_A': 0.1666, 'min_lr': 10.764, 'non_converged': 0, 'max_gain_from_free_default_start': 2.4244, 'bias_log_speed': -0.2748, 'rmse_log_speed': 0.2748, 'coverage_log_speed': 1.0, 'coverage_log_sigma': 1.0, 'pass_no_negative_lr': True}
S2 summary: {'scenario': 'S2', 'datasets': 1, 'n': 60, 'mean_abs_error_A': 0.1363, 'min_lr': 14.9397, 'non_converged': 0, 'max_gain_from_free_default_start': 3.771, 'mean_abs_error_A_other': 0.2279, 'pass_no_negative_lr': True}
```

**Design choices and pitfalls.**
- **The drift has no bias summary**, only the mean absolute error, which mixes bias and noise; "recovered without material bias" (§16) needs the mean error of each entry, which is in the saved rows but not summarised.
- No summary exists for the loads, the noise, the load spreads, $\lambda$ or the treatment terms; coverage is computed only under the null and only for $\gamma_1$ and $\log\sigma_g$.
- The automatic part of the verdict uses only "no negative LR" from this check; bias and coverage are left to the investigator and statistician (§6.54).
- `rows` from a checkpoint must come from the same settings, which `Checkpoint` enforces.

### 6.49 `time_replicate` — the cost of one whole bootstrap replicate

[`gate.py:251`](../../src/pdx/gate.py#L251)

**What it does.** It simulates one S1 dataset of the real size and times three things: Step 1's support selection, the two observed fits, and one whole bootstrap replicate (simulation from the fitted null, support re-selection, both refits).

**Inputs → outputs.** `n`, `d`, `seed`, `fit_kw`, `step1` (True), `sim_kw` → `{"n", "d", "support_selection_s", "observed_fits_s", "replicate_s", "null_evaluations", "free_evaluations"}`.

**The mathematics.** Wall-clock differences $t_1-t_0$, $t_2-t_1$ and $t_3-t_2$. They feed §8.8's budget, hours $=$ runs × minutes ÷ 60 ÷ 4.

**Worked example** ([`ex_gate_timing.py`](../guide_examples/s06/ex_gate_timing.py)): at 60 simulated participants, then the same call with the replicate's simulation made to fail.

```python
r = gate.time_replicate(60, 4, seed=0)
hp.simulate_from_fit = broken             # _replicate catches the error and returns (inf, False, False, None)
r2 = gate.time_replicate(60, 4, seed=0)
```

```text
{'n': 52, 'd': 4, 'support_selection_s': 0.85, 'observed_fits_s': 28.47, 'replicate_s': 37.23, 'null_evaluations': 282, 'free_evaluations': 740}
{'n': 52, 'd': 4, 'support_selection_s': 0.02, 'observed_fits_s': 24.77, 'replicate_s': 0.0, 'null_evaluations': 282, 'free_evaluations': 740}
```

Reading the output:
- The free model's evaluations (740) count both of its starts.
- The first support selection includes a one-off import and registration of the causal-learn tests (0.85 s against 0.02 s the second time).
- **A failed replicate is timed as 0 seconds without any flag**: `_replicate`'s status is discarded (line 263).

**Design choices and pitfalls.**
- One replicate on one dataset is timed; §8.8 says "every component is timed on 5 synthetic datasets of the real size".
- The replicate's success should be returned with its time (§6.N).

### 6.50 `plain` — JSON-ready values for the checkpoint

[`gate.py:269`](../../src/pdx/gate.py#L269)

**What it does.** It converts a result into plain Python values (numpy arrays to lists, numpy scalars to numbers) by a round trip through JSON, so that what is checkpointed and what is kept in memory are the same.

**Inputs → outputs.** Any JSON-able structure with numpy parts → the same structure with lists, floats, ints, bools and string keys. Anything else raises `TypeError`.

**The mathematics.** None. The round trip also turns tuples into lists and integer dictionary keys into strings, and keeps NaN and infinity (Python's JSON writes them as `NaN` and `Infinity`).

**Worked example** ([`ex_gate_checkpoint.py`](../guide_examples/s06/ex_gate_checkpoint.py), part 1):

```python
gate.plain({"a": np.arange(3), "b": np.float64(1.5), "c": np.bool_(True), "d": (1, 2),
            "e": {2: 0.5}, "f": np.int64(7), "g": float("nan"), "h": np.inf})
```

```text
1. plain: {'a': [0, 1, 2], 'b': 1.5, 'c': True, 'd': [1, 2], 'e': {'2': 0.5}, 'f': 7, 'g': nan, 'h': inf}
   plain of an object that is not JSON: TypeError('object')
```

**Design choices and pitfalls.** Because a fresh section returns `plain(fn())` (§6.51), the report built in an uninterrupted run has the same types as one rebuilt from a checkpoint (lists, not arrays; `'2'`, not `2`).

### 6.51 `Checkpoint` — finished work kept, other settings refused

[`gate.py:280`](../../src/pdx/gate.py#L280)

**What it does.** It holds the gate's finished work in a JSON file. At creation it loads the file if it exists and refuses to continue if the file was written with other settings. `section(key, fn)` returns the saved result of `key` if there is one, and otherwise runs `fn`, stores its plain value, saves the file and logs the time taken. `save` writes to a temporary file and renames it over the checkpoint, retrying when Windows holds the file open.

**Inputs → outputs.** `path` (or `None` for no file), `settings`, `log` → an object with `data`, `section` and `save`.

**The mathematics.** None. The guarantees are:
1. *Atomic saves*: a crash leaves the old or the new checkpoint, never a half-written one.
2. *Same settings or nothing*: `plain(settings)` must equal the saved settings, so results from different runs are never mixed.
3. *Same results after a restart*: guaranteed by giving each section its own random stream (see below).

**Why one random stream per section.** If all sections drew from one generator, a section skipped on restart would not consume its random numbers, and every later section would receive different ones. With a stream per section (`default_rng([seed, k])` in `run_gate`), skipping a section changes nothing downstream. The example builds a two-section toy gate both ways and interrupts it after the first section:

**Worked example** ([`ex_gate_checkpoint.py`](../guide_examples/s06/ex_gate_checkpoint.py), parts 2 and 3):

```python
ck = gate.Checkpoint(path, {"n": 10, "sim_kw": {"target_counts": (3, 2)}}, said.append)
ck.section("a", lambda: calls.append(1) or {"x": np.float64(2.0)})
again = gate.Checkpoint(path, {"n": 10, "sim_kw": {"target_counts": [3, 2]}}, said.append)   # a tuple saved as a list
gate.Checkpoint(path, {"n": 11, "sim_kw": {"target_counts": (3, 2)}})                            # refused
```

```text
2. file exists before the first section: False
   section a: {'x': 2.0} | work done 1 time(s)
   after a restart: {'x': 2.0} | work done 1 time(s)
   log: ['a: done (0 min)', 'a: done earlier, skipped']
   other settings: <tmp>/gate.partial.json holds a gate run with other settings; move it away to start afresh
   saved file: {"settings": {"n": 10, "sim_kw": {"target_counts": [3, 2]}}, "a": {"x": 2.0}}
3. one shared stream     : uninterrupted ([-0.7931, 0.2406], [-1.8963, 1.3958]), resumed ([-0.7931, 0.2406], [-0.7931, 0.2406]) -> equal: False
3. one stream per section: uninterrupted ([-2.1682, 0.3459], [0.4283, -0.2376]), resumed ([-2.1682, 0.3459], [0.4283, -0.2376]) -> equal: True
```

Reading the output:
- The restart skips the finished section and returns the saved value; settings that differ only in form (a tuple against a list) are accepted, because both are compared after `plain`.
- With one shared stream, the resumed second section receives the numbers the first section had drawn in the uninterrupted run; with a stream per section it receives the same numbers as before.

**Design choices and pitfalls.**
- Nothing is written until the first section finishes, so the settings guard protects only runs that saved something.
- Pieces inside a section (dispersed-start fits, profiles, recovery datasets) are saved under their own keys by `run_gate` (§6.53–6.54), using `ck.data` and `ck.save()` directly.
- The settings do not include the code version; `scripts/run_engine_gate.py` records the commit and a hash of uncommitted changes in the final report, but a checkpoint written by older code is resumed silently.

### 6.52 `run_gate`, lines 316–348 — sizes, settings, seeds and the small dataset

[`gate.py:316`](../../src/pdx/gate.py#L316)

**What it does.** It fixes the sizes (shrinking them in quick mode), opens the checkpoint with the run's settings, defines one random stream per section, runs the reference and gradient checks on a small dataset, and simulates the real-size gate dataset.

**Inputs → outputs.** `n` (1400: the simulated pool), `d` (4), `n_datasets` (20), `seed` (2026), `quick`, `fit_kw`, `sim_kw` (for example WP0's target counts, visits and missingness), `checkpoint`, `log`, `jobs` → the report (built over §6.52–6.54).

**The mathematics.**
- Quick mode (lines 329–336): `n_datasets` $\le2$; with target counts $(n_0,n_1)$, they become $\big(\max\{2,\operatorname{round}(200\,n_0/(n_0+n_1))\}, \dots\big)$ and the pool $n=5\sum n_g$; otherwise $n=\min(n,200)$.
- Streams: `rng(k) = default_rng([seed, k])`; numpy's seed sequence makes the streams for different $k$ independent. $k=0$ the small dataset, 1 the reference check, 2 the gradient check, 3 the gate dataset, 4 the dispersed starts. The recovery datasets use `default_rng(seed + 100 + r)` (S1) and `default_rng(seed + 200 + r)` (S2), the timing `seed + 300`.
- The small dataset is S1 with 40 simulated participants and the default options, whatever `sim_kw` says; the gate dataset is S1 with $n$ and `sim_kw`.

**Worked example** ([`ex_gate_checkpoint.py`](../guide_examples/s06/ex_gate_checkpoint.py), parts 4 and 5). `Checkpoint` is replaced by a class that prints the settings `run_gate` would save and stops, so nothing is fitted; the target counts are invented.

```python
gate.Checkpoint = Peek                          # prints the settings, then raises Stop (caught)
for quick in (False, True):
    gate.run_gate(n=1400, d=10, n_datasets=20, quick=quick,
                  sim_kw={"target_counts": (700, 500), "visit_probs": {2: 0.4, 6: 0.6}})
gate.Checkpoint = real_checkpoint
gate.run_gate(n=200, d=4, checkpoint=p2)        # p2 holds a checkpoint with other settings
```

```text
4. settings run_gate would save: {'n': 1400, 'd': 10, 'n_datasets': 20, 'seed': 2026, 'quick': False, 'sim_kw': {'target_counts': (700, 500), 'visit_probs': {2: 0.4, 6: 0.6}}, 'fit_kw': {}}
4. settings run_gate would save: {'n': 1000, 'd': 10, 'n_datasets': 2, 'seed': 2026, 'quick': True, 'sim_kw': {'target_counts': (117, 83), 'visit_probs': {2: 0.4, 6: 0.6}}, 'fit_kw': {}}
   a checkpoint from another run: <tmp>/g.partial.json holds a gate run with other settings; move it away to start afresh
5. section streams: rng(k) = default_rng([seed, k]) for k = 0 small dataset, 1 reference, 2 gradient, 3 gate dataset, 4 dispersed starts
   recovery seeds: S1 2126 to 2145 | S2 2226 to 2245 | timing 2326
```

By hand, quick mode: $200\times700/1200=116.7\to117$, $200\times500/1200=83.3\to83$, $n=5\times200=1000$.

**Design choices and pitfalls.**
- The recovery seeds are integers offset by 100, 200 and 300, so with more than 100 datasets the S1 datasets from $r=100$ on reuse the S2 seeds, and the S2 datasets from $r=100$ on reuse the timing seed (common random numbers rather than independent datasets).
- `n` is the pool, not the panel size: with WP0's target counts the panel has exactly $\sum n_g$ participants.
- The reference and gradient checks run on 35 participants with default visits, not on the WP0-shaped data.

### 6.53 `run_gate`, lines 349–384 — the speed-only fit, quadrature, starts, identification and profiles

[`gate.py:349`](../../src/pdx/gate.py#L349)

**What it does.** It fits the speed-only model to the gate dataset if any section that needs it is unfinished (restarting from the saved solution if there is one), then runs the quadrature section, the dispersed starts (saving after each fit), the identification section (saving the standard errors separately), and the profiles of seven parameters (saving after each).

**Inputs → outputs.** The checkpoint and the gate dataset → the report's `quadrature`, `dispersed_starts`, `identification` and `profiles`; in the checkpoint also `null_fit_x`, `dispersed_starts_done`, `identification_se` and `profiles_done`.

**The mathematics.** The profiled coordinates (lines 374–375), with $\widehat{\mathrm{SE}}$ from the identification section:

| coordinate | meaning |
|---|---|
| `gamma[0]` | PIGD's log speed ratio $\gamma_1$ |
| `log_sigma[1]` | PIGD's log speed spread |
| `b[d+1]` | PIGD's load on domain 1 (axial), the load the S1 truth changes |
| `L[0]` | TD's first Cholesky diagonal of $Q$ (tremor), on the log scale |
| `log_tau[0]` | TD's log load spread on tremor |
| `lam[0]` | TD's speed–load link on tremor |
| `kappa[0]` | TD's short-duration treatment effect on tremor (if a treatment is modelled) |

Each runs `profile` with 9 grid points (5 in quick mode), so the section costs 63 constrained fits at most.

**Worked example** ([`ex_gate_checkpoint.py`](../guide_examples/s06/ex_gate_checkpoint.py), part 6): the coordinates resolved for $d=4$ and $d=10$.

```text
6. d = 4, 112 parameters; profiled: gamma[0] = log speed ratio of PIGD; log_sigma[1] = log speed spread of PIGD; b[5] = load of group 1 on axial; L[0] = group 0, Cholesky entry (0,0) on the log scale; log_tau[0] = log load spread of group 0 on tremor; lam[0] = speed-load link of group 0 on tremor; kappa[0] = short-duration effect of group 0 on tremor
6. d = 10, 322 parameters; profiled: gamma[0] = log speed ratio of PIGD; log_sigma[1] = log speed spread of PIGD; b[11] = load of group 1 on axial; L[0] = group 0, Cholesky entry (0,0) on the log scale; log_tau[0] = log load spread of group 0 on tremor; lam[0] = speed-load link of group 0 on tremor; kappa[0] = short-duration effect of group 0 on tremor
```

The quadrature, dispersed-start, identification and profile computations themselves are shown in §6.41, §6.43, §6.46 and §6.45.

**How the code was formed.** Each long piece is checkpointed at the finest grain that can be resumed exactly: one dispersed-start fit (the starts are drawn up front, §6.43), one profile (independent of the others), one identification (whose array of standard errors is stored apart, because the report keeps only plain summaries). The speed-only fit is refitted on restart from its saved parameters, which converges in a few evaluations instead of hundreds (line 351).

**Design choices and pitfalls.**
- **A resumed run equals an uninterrupted one only to the optimiser's tolerance here.** The restart refits the speed-only model from the saved parameters; [`ex_note_resume_fit.py`](../guide_examples/s06/ex_note_resume_fit.py) shows the refit moving by $2.2\times10^{-6}$ in the parameters and $3.5\times10^{-8}$ in the log-likelihood (3 evaluations). Sections computed after a restart (quadrature, identification, profiles) therefore start from a slightly different fit than they would have in one go; the module docstring promises the same results.

```text
uninterrupted: loglik -792.5895271012 after 437 evaluations (CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH)
resumed:       loglik -792.5895270664 after 3 evaluations (CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH)
largest parameter change 2.2e-06; log-likelihood change 3.5e-08
```

- Only one coordinate of each block is profiled: not $\kappa^{\text{LDR}}$ (protocol §16: "the treatment terms"), not $\omega$, $\theta$ or $\beta$.
- Spreads found at zero by the identification section are not removed from the profile list, so `log_tau[0]` can fail as in §6.45 (`all(p["identified"] ...)` then fails the gate).

### 6.54 `run_gate`, lines 385–401 — recovery, timing and the verdict

[`gate.py:385`](../../src/pdx/gate.py#L385)

**What it does.** It runs the recovery check under S1 (speed-only fit, with intervals except in quick mode) and under S2 (free fit), saving after each dataset, times one whole replicate, and combines eight automatic checks into `all_automatic_checks_pass`, with a note that bias and coverage are for the investigator and statistician to judge.

**Inputs → outputs.** → the report's `recovery_null`, `recovery_changed_link`, `timing`, `all_automatic_checks_pass` and `note`.

**The mathematics.**

$$
\text{pass}=\underbrace{\delta_{\text{ref}}<10^{-8}}_{6.38}\wedge\underbrace{\delta_{\text{grad}}<10^{-4}}_{6.39}\wedge\underbrace{\text{spread}^{\text{null}}_{12/20/30}<0.01}_{6.41}\wedge\underbrace{\text{spreads}_{\text{starts}}<0.1}_{6.43}\wedge\underbrace{\kappa_{\text{rest}}<10^{10}}_{6.46}\wedge\underbrace{\text{all profiles identified}}_{6.45}\wedge\underbrace{\min\mathrm{LR}_{\text{S1}},\min\mathrm{LR}_{\text{S2}}>-10^{-6}}_{6.48}.
$$

Not in the verdict: the statistic's and the free model's quadrature spreads, the bias, RMSE and coverage, the gain from the free model's own start, and the timing.

**Worked example.** The gate as a whole is not run here: even quick mode takes many minutes (`scripts/run_engine_gate.py --quick`; §7). Its pieces are run in §6.38–§6.51 and the settings in §6.52.

**Design choices and pitfalls.**
- On the small datasets of this chapter three of the eight automatic checks fail (quadrature, dispersed starts, one profile). The real-size run decides; the failures shown here say where to look first.
- §8.8 budgets the gate as "20 datasets × 14 fits, plus profile likelihoods on 5"; the code fits each recovery dataset with `fit_pair` (one speed-only fit and the free fit from two starts), adds Hessians for the S1 intervals, and profiles seven parameters on the single gate dataset.
- The note is right that bias and coverage need judgement, but with the estimand offset of §6.47 the bias of $\gamma_1$ will not centre on 0 even if the engine is perfect.

### 6.N Notes for the author

Ordered by likely impact. Where a note was reproduced on synthetic data, the script is named; the others come from reading the code against the protocol (marked "a check" where the reading should be confirmed). Nothing in `src/` was changed.

1. **H-M1 is likely to fail a well-calibrated H-P for Monte Carlo reasons alone** ([`qualification.py:143`](../../src/pdx/qualification.py#L143), [`qualification.py:166`](../../src/pdx/qualification.py#L166); protocol §8.6–§8.7).
   - *What:* with the protocol's numbers (500 datasets in A:S0 and A:S1, 100 in each of the six other family A cells, 300 in each family B cell) and a procedure whose true false-rejection probability is exactly 5% in every cell, all ten cells pass Bradley's 7.5% with probability 0.41 if the critical values were known, and about 0.10 with warp-speed critical values. The six 100-dataset cells dominate: each passes with probability 0.87 (known critical value) or 0.73 (warp-speed).
   - *Why:* the warp-speed critical value is the $\lceil0.95R\rceil$-th of $R$ replicates, so for a calibrated procedure $E[\hat p]=1-\lceil0.95R\rceil/(R+1)$ (5.94% at $R=100$) and $\operatorname{Var}(\hat p)$ is about twice the binomial variance. The MCSE reported by `calibration_summary` covers only the binomial part (about 1.4 times too small). The protocol's "passes with probability 0.99 with 500 datasets" is the known-critical-value figure; the warp-speed figure is about 0.94.
   - *Reproduction:* [`ex_calibration.py`](../guide_examples/s06/ex_calibration.py), [`ex_h_m1.py`](../guide_examples/s06/ex_h_m1.py) (§6.29–§6.30).
   - *Why it matters:* a "fail" would remove H-P from Part I (§8.7). Options to consider before WP3 runs: more datasets in the 100-dataset cells, a rule that allows for Monte Carlo error (for example an upper confidence bound with the correct variance), or a reference distribution built with more than one replicate per dataset.
2. **The quadrature check fails already at 113 participants, and its pass flag ignores H-P's statistic** ([`gate.py:75`](../../src/pdx/gate.py#L75), [`gate.py:78`](../../src/pdx/gate.py#L78), [`gate.py:394`](../../src/pdx/gate.py#L394)).
   - *What:* on the S1 dataset of `tests/test_gate.py` the speed-only log-likelihood moves by 0.022 between 12 and 20 nodes (threshold 0.01), the free one by 0.039, and the statistic by 0.034. The verdict uses only the speed-only spread, although the docstring says the statistic is what has to agree.
   - *Reproduction:* [`ex_gate_fit_checks.py`](../guide_examples/s06/ex_gate_fit_checks.py) (§6.41).
   - *Why it matters:* the integration error is a sum over participants and tends to grow with their number and visits, so the real-size gate (with WP0's group sizes and visit counts) will probably report a failure on this item. A criterion on the statistic relative to its critical value, a per-participant tolerance, or more nodes needs to be chosen.
3. **Dispersed starts: a speed spread trapped at zero, and a better free optimum that `fit_pair` misses** ([`gate.py:101`](../../src/pdx/gate.py#L101); `engine/fit.py`, §3).
   - *What:* on the dataset of `tests/test_gate.py` (75 participants) the check fails. A dispersed start of the speed-only model ends 5.0 log-likelihood units low with $\log\sigma_{\text{TD}}=-14.5$, where the likelihood is flat in $\log\sigma$, and a restart stays there. A dispersed start of the free model ends 0.35 above both of `fit_pair`'s starts, so the statistic there would be 4.07 instead of 3.36. The test checks only that the resume reproduces the run, not the pass flag.
   - *Reproduction:* [`ex_gate_dispersed_starts.py`](../guide_examples/s06/ex_gate_dispersed_starts.py), [`ex_note_starts.py`](../guide_examples/s06/ex_note_starts.py) (§6.43).
   - *Why it matters:* at the real size the surface may be better behaved, but a failure of this check may come from a log-spread drifting to $-\infty$ rather than from genuinely separate optima; and a missed free optimum understates the statistic, in the data and in the bootstrap replicates alike, with no obvious net effect on the level. A lower bound on log-spreads during fitting, or a restart for spreads found at zero, would separate the two.
4. **The gate's truth for the speed ratio is the marginal ratio; the engine's $\gamma_1$ is conditional on the landmark** ([`gate.py:201`](../../src/pdx/gate.py#L201), [`gate.py:206`](../../src/pdx/gate.py#L206); [`family_a.py:142`](../../src/pdx/sim/family_a.py#L142)).
   - *What:* in family A the landmark is year 0 propagated at each participant's speed, so the landmark scores carry information about the speed. Regressing the simulated log-speeds on phenotype and the centred landmark scores gives a PIGD coefficient of 0.652 ($d=4$) and 0.660 ($d=10$), not $\log2=0.693$; TD's speed at the pooled-mean landmark is $e^{0.02}$, which the reference drift absorbs.
   - *Reproduction:* [`ex_speed_landmark.py`](../guide_examples/s06/ex_speed_landmark.py) (§6.6, §6.47).
   - *Why it matters:* `bias_log_speed` will sit near $-0.03$ to $-0.04$ for a perfect engine, and `coverage_log_speed` is centred on the wrong value, which could be read as estimator bias at the real size (20 datasets). Comparing log Sp-1 with $\log s$ (as the qualification does), or defining $\gamma_1$'s truth by a large-sample fit, would remove the offset. The same consideration applies to the free fit's comparison with $s\,A_{\text{PIGD}}$.
5. **The qualification computes no coverage and no MCSE for the speed estimates** ([`qualification.py:1`](../../src/pdx/qualification.py#L1), [`qualification.py:203`](../../src/pdx/qualification.py#L203); protocol §8.5–§8.7).
   - *What:* `speed_summary` gives the bias and RMSE of log Sp-1 and the median Sp-3 share. There is no coverage of model-based intervals on the calibration datasets and no coverage of bootstrap intervals (30 datasets × 100 samples, §8.6), although the module docstring mentions coverage. Sp-2 is not evaluated, and no measure carries its MCSE.
   - *Why it matters:* §8.7's rule ("Sp-1 to Sp-3 are reported as planned if their 95% intervals cover at least 90% of the time in S0 and S1") cannot be evaluated with the current code.
6. **Scenario definitions that differ from protocol §8.2's table** ([`scenarios.py:57`](../../src/pdx/sim/scenarios.py#L57)–86; §6.14).
   - S3's TD-only link (function ← tremor, 0.10 raw) is $0.91\varepsilon$, below the smallest effect of interest; the PIGD-only link is $2\varepsilon$.
   - S2 and S2b change one link; the table says "one or two links" and "selected links".
   - S1T halves only the short-duration (ON-state) axial response; the long-duration response is unchanged.
   - S1V varies the participant load spread $\tau$ (0.10 against 0.15); family A has no time-varying burden.
   - S7 widens the speed spread to 0.6 in both phenotypes; S1 already has continuous individual speed (0.4).
   - The speed ratio is fixed at 2; the stress "speed ratios 1.25, 1.5 and 2" cannot be run through `generate`.
   - Family B's S3 analogue has only a PIGD-only link (0.12 raw, through $\tanh$, not sized in $\varepsilon$).
   - S5 and S6 are not written (documented in `docs/code_map.md`).
7. **Simulator features of §8.2 not yet implemented** (a check against the protocol; [`family_a.py:106`](../../src/pdx/sim/family_a.py#L106), [`scenarios.py:12`](../../src/pdx/sim/scenarios.py#L12)–15). Missingness is MCAR at fixed shares, not resampled from PPMI's patterns by era, year and treatment; intervals are a uniform jitter, not resampled; there are no splines in the state or in disease duration, no mapping of scores to ordinal or bounded types, no fitted initiation model, and dropout is always tied to severity. Of the stresses, MNAR, noisy defining items, era differences, an initial state correlated with speed, dropout tied to speed, hidden burden and the identification limit are still to add. The modules say that fitting to PPMI is a WP3 step; this list is for the record of what WP3 still has to build.
8. **Family B silently ignores simulation options and support stresses** ([`scenarios.py:102`](../../src/pdx/sim/scenarios.py#L102)–110).
   - *What:* only `noninvariant` reaches `family_b.simulate`; `spurious`, `missing_link`, `target_counts`, `visit_probs` and per-domain `p_missing` are dropped without an error.
   - *Reproduction:* [`ex_generate.py`](../guide_examples/s06/ex_generate.py) (§6.16).
   - *Why it matters:* a qualification run with WP0's settings would qualify family A at WP0's sizes and family B at the default ones.
9. **Family B's null and its analysis model** (a check; [`family_b.py:51`](../../src/pdx/sim/family_b.py#L51), [`family_b.py:69`](../../src/pdx/sim/family_b.py#L69); [`qualification.py:40`](../../src/pdx/qualification.py#L40)).
   - The common measurement scale (1.04 for PIGD, plus 0.05 per era) leaves the *linear* dynamics invariant, but the non-linear term becomes $cW\tanh(Y/c)$ on the measured scale, which differs between scales at third order. The null is therefore exactly true only to first order on the measured scale; probably negligible, worth a sentence in the methods.
   - The analysis fixes the measurement-error variance at 0.02, while family B uses $0.15^2=0.0225$ and adds site effects. Intended as part of "outside the engine's class", or an oversight?
10. **Profiles: one coordinate per block, spreads at zero not set aside** ([`gate.py:374`](../../src/pdx/gate.py#L374)–378; §6.45, §6.53).
    - $\kappa^{\text{LDR}}$ (protocol §16: "the treatment terms") and $\omega$ are not profiled.
    - A spread found at zero by the identification section stays in the profile list; TD's load spread on tremor fails the profile on the small dataset because the likelihood flattens towards $\tau=0$ ([`ex_gate_profile.py`](../guide_examples/s06/ex_gate_profile.py)). One such failure fails the whole gate.
11. **Recovery summaries do not support the bias judgement fully** ([`gate.py:235`](../../src/pdx/gate.py#L235)–248).
    - The drift has only a mean absolute error (bias and noise mixed); the loads, noise, load spreads, $\lambda$ and treatment terms have no summary; coverage is computed only under the null and only for $\gamma_1$ and $\log\sigma_g$. With 20 datasets a coverage estimate has MCSE of about 0.05.
    - The recovery seeds are `seed + 100 + r`, `seed + 200 + r` and `seed + 300`, so with more than 100 datasets the S1 and S2 datasets, and the timing dataset, share seeds ([`gate.py:385`](../../src/pdx/gate.py#L385)–393).
12. **A resumed gate equals an uninterrupted one only to the optimiser's tolerance** ([`gate.py:351`](../../src/pdx/gate.py#L351)–353; module docstring lines 15–16).
    - *What:* the speed-only fit is refitted from the saved parameters on restart; it moved by $2.2\times10^{-6}$ in the parameters and $3.5\times10^{-8}$ in the log-likelihood.
    - *Reproduction:* [`ex_note_resume_fit.py`](../guide_examples/s06/ex_note_resume_fit.py) (§6.53).
    - *Why it matters:* harmless for the verdicts, but the docstring's "the same results" is true bit for bit only for the sections finished before the interruption.
13. **`time_replicate` discards the replicate's status** ([`gate.py:263`](../../src/pdx/gate.py#L263)). A replicate that fails is timed as about 0 seconds with no flag ([`ex_gate_timing.py`](../guide_examples/s06/ex_gate_timing.py), §6.49). §8.8 also asks for timing on 5 datasets of the real size; the gate times one replicate on one dataset.
14. **Parallel settings differ between the harnesses** ([`qualification.py:105`](../../src/pdx/qualification.py#L105)–111, [`parallel.py:48`](../../src/pdx/parallel.py#L48), `scripts/run_engine_gate.py:5`). `run_tasks` caps workers at four silently, uses `setdefault` for the thread variables (a user's `OMP_NUM_THREADS=4` gives four threads per worker) and does not restore them; `parallel.ordered` has no cap, and the gate script's docstring suggests `--jobs 5`, above §16's four.
15. **Configuration keys and power** ([`qualification.py:73`](../../src/pdx/qualification.py#L73)–78, [`qualification.py:58`](../../src/pdx/qualification.py#L58), [`qualification.py:196`](../../src/pdx/qualification.py#L196)).
    - The key depends on how values are written (integer against string keys of `visit_probs`, `3000` against `3000.0`); [`ex_tasks.py`](../guide_examples/s06/ex_tasks.py) shows two keys for one setting. The scripts are consistent today.
    - `replicate=False` changes the key, so power-only rows never find an S1 critical value in `power_summary`, which then reports power 0 without a warning.
    - `power_summary` drops failed datasets from the denominator while `calibration_summary` counts them as rejections ([`ex_power_speed.py`](../guide_examples/s06/ex_power_speed.py): 0.71 against 0.65).
16. **Fidelity** ([`fidelity.py:11`](../../src/pdx/fidelity.py#L11)). Not called by any script or test; covers part of the §8.2 dashboard (no autocorrelations beyond lag 1, phenotype stability, treatment initiation and dose, dropout, or breakdowns by era and site); with 20 simulated panels even the right simulator puts many entries outside the envelope ([`ex_fidelity.py`](../guide_examples/s06/ex_fidelity.py): 62% of lag correlations and 50% of residual standard deviations inside). WP2's thresholds need to allow for multiplicity and the number of panels. `nanvar` warns for participants with one observed value in a domain.
17. **WP0's missing shares are misaligned at $d=4$** (`scripts/run_engine_gate.py:62`, `scripts/run_power_check.py:117`; a check of the scripts that feed the simulator). `p_missing[:d]` is in the ten-domain order (tremor, axial, brady_rigidity, function, …), but the four-domain simulator's domains are tremor, axial, function, independence. With `--wp0 --d 4`, function gets bradykinesia–rigidity's share and independence gets function's. $d=6$ and $d=10$ align.
18. **Smaller points.**
    - $\varepsilon$ ([`scenarios.py:29`](../../src/pdx/sim/scenarios.py#L29)), Bradley's 0.075 ([`qualification.py:143`](../../src/pdx/qualification.py#L143)) and the four-job cap repeat values of `configs/protocol_v0.5.toml` instead of reading them, so the $\varepsilon=0.05$ sensitivity needs a code change.
    - `pooled_sd` uses the landmark (year-1) scores; protocol §2.3 says "the pooled baseline scores" ([`scenarios.py:41`](../../src/pdx/sim/scenarios.py#L41)–43).
    - `keep_indeterminate=True` recodes the indeterminate as PIGD without a flag ([`family_a.py:133`](../../src/pdx/sim/family_a.py#L133)); unused.
    - `z` is centred before the subsampling to target counts, so the panel's covariates are centred only approximately ([`family_a.py:140`](../../src/pdx/sim/family_a.py#L140)).
    - `cache = {}` is unused ([`family_a.py:146`](../../src/pdx/sim/family_a.py#L146)).
    - `_rank_quantile(values, 0)` returns the largest value ([`qualification.py:133`](../../src/pdx/qualification.py#L133)); only $q=0.95$ is used.
    - The simulators use 4 tremor items, Stebbins' rule 11 ([`family_a.py:130`](../../src/pdx/sim/family_a.py#L130)).
    - In the result rows, `n` is the number analysed for a successful dataset and the number simulated for a failed one ([`qualification.py:66`](../../src/pdx/qualification.py#L66), [`qualification.py:93`](../../src/pdx/qualification.py#L93)).
    - Untested branches: `h_m1`, `power_summary`, `speed_summary`, `fidelity`, `profile`, `identification`, family B's `noninvariant`, `missing_link`, `keep_indeterminate`. The parallel-equivalence test of the gate pieces is marked slow and skipped by default (`tests/test_gate.py:57`), and `run_gate` as a whole is exercised only by `scripts/run_engine_gate.py --quick`.

## 3. The engine II: fitting, standard errors and simulation

*Files: [`src/pdx/engine/fit.py`](../../src/pdx/engine/fit.py) (250 lines) and [`src/pdx/engine/simulate.py`](../../src/pdx/engine/simulate.py) (73 lines), at commit `a9b577f`. Protocol: §2.3, §7.1, §7.3.1 and §16.*

### 3.0 Overview

**What this chapter covers.** Chapter 2 built the engine's log-likelihood $\ell(x)$ and its exact gradient. This chapter turns them into the numbers the study reports:
- **fitting** (`fit.py`): start values, maximum likelihood from one or more starts, and the pair of fits that H-P, the primary test, compares;
- **standard errors** (`fit.py`): the observed information, its inverse, and checks that say when the inverse can be trusted;
- **simulation** (`simulate.py`): new datasets drawn from a fitted model, for H-P's conditional parametric bootstrap.

It establishes three facts the rest of the study leans on:
1. the speed-only model is exactly nested in the free model, so H-P's likelihood ratio cannot be negative (§3.6, §3.8);
2. the information matrix is computed accurately, and its report says which directions are flat (§3.9, §3.12);
3. the simulator draws from exactly the model that the likelihood describes (§3.14–§3.16, the score check).

**Where it sits.**

```text
panel + ModelSpec (support, treatment form, measurement-error variances)
  │
  │  start_values("null"): yearly least squares per phenotype → drift, speed, load, noise
  ▼
fit_model(kind="null") ──────────────► Fit f0  (speed-only model, H-P's null)
  │                                       │
  │  null_to(f0.x, "free")                │  simulate_from_fit(f0, panel, rng)        (hp._replicate)
  ▼                                       ▼
fit_model(kind="free",                  simulated panel ──► fit_pair again ──► LR*  (one bootstrap replicate)
          starts = [null solution, start_values("free")]) ──► Fit f1  (free model, H-P's alternative)
  │
  ▼
LR = 2 (ℓ̂_free − ℓ̂_null) ≥ 0                                        (hp.lr_stat)

any Fit ──► hessian ──► covariance_checked ──► standard errors, Monte Carlo draws, identification report
```

| function | called by |
|---|---|
| `fit_pair` | `hp.observed_fits`, `hp._replicate`, `hp.warp_replicate`, `hp.reference_distribution`, `speed.bootstrap_sample`, `qualification`, `gate.recovery_dataset`, `gate.time_replicate` |
| `fit_model` | `fit_pair`; `localise` (anchored tests, the refit of joint discovery, stability selection); `gate` (dispersed starts, profiles, the quadrature check, the gate's null fit) |
| `null_to` | `fit_pair`; `localise.joint_discovery`; `gate.check_quadrature_lr`; `tests/test_engine.py` |
| `start_values` | `fit_model`, `fit_pair`, `gate`, `tests/helpers.py`, `scripts/bench_threads.py` |
| `hessian`, `covariance_checked`, `param_names` | `localise.free_draws`, `localise.anchored_tests`, `gate.identification`, `gate.recovery_dataset`, `scripts/run_hp_real.py` |
| `simulate_from_fit` | `hp._replicate` only |

**The model in brief** (Chapter 2 has it in full). A participant $i$ in phenotype $g$ (0 = tremor-dominant, TD, the reference; 1 = postural instability and gait difficulty, PIGD) has true domain levels that follow a linear stochastic differential equation, and is observed with error at the visits:

$$
dX_i=s_i\big[A_gX_i+b_g+\ell_i\big]dt+\sqrt{s_i}\,G_g\,dW_i,\qquad Y_{ik}=X_i(t_{ik})+h_{ik}+e_{ik},\quad e_{ik}\sim N(0,\Sigma_e),
$$

$$
\log s_i=\gamma_g+\beta^\top z_i+\theta^\top\tilde y_{i1}+\sigma_gu_i,\quad u_i\sim N(0,1),\qquad
\ell_i\mid u_i\sim N\big(\Gamma^y_g\tilde y_{i1}+\Gamma^z_gz_i+\lambda_gu_i,\ \mathrm{diag}(\tau_g^2)\big).
$$

The noise covariance $Q_g=G_gG_g^\top$ is multiplied by $e^{\omega\,\mathrm{era}_i}$ when an era column is named. In the code $\Gamma^y_g$ is **diagonal**: `gam_y` has shape $(G,d)$ and multiplies $\tilde y_{i1}$ element by element ([`likelihood.py:143`](../../src/pdx/engine/likelihood.py#L143)), so each domain's load mean depends on that domain's own landmark score. $\Gamma^z_g$ is a full $d\times p$ matrix.

The three kinds of model differ only in the drift and the speed offset ([`model.py:139`](../../src/pdx/engine/model.py#L139)):

| kind | drift of phenotype $g$ (at the reference speed) | speed offset | extra blocks |
|---|---|---|---|
| `null` (speed-only; H-P's null) | $A$ for every $g$ | $\gamma_g$, with $\gamma_0=0$ | `gamma` |
| `free` (H-P's alternative) | $A_0=A$; $A_g$ free for $g\ge1$ | 0 for every $g$ (the speed is absorbed into $A_g$) | `A1`, … |
| `joint` (localisation) | $A+D_g$, with $D_0=0$ | $\gamma_g$ | `D1`, …, `gamma` |

**The parameter vector.** Every fit works on one flat vector $x$ (`Fit.x`), cut into named blocks by `Layout` ([`model.py:82`](../../src/pdx/engine/model.py#L82)). The blocks hold *raw* values, transformed so that the optimiser can move freely:

| block | shape | natural quantity | raw → natural |
|---|---|---|---|
| `A`, `A{g}` | $n_{sup}$ | support entries of a drift, in `spec.sup_idx` order ((to, from) pairs, row by row) | own rates $A_{jj}=-e^{r}$; links unchanged |
| `D{g}` | $n_{sup}$ | deviations (joint model) | unchanged (own rates not constrained) |
| `gamma` | $G-1$ | log-speed of phenotype $g\ge1$ | unchanged |
| `beta`, `theta` | $p$, $d$ | effects on log-speed | unchanged |
| `log_sigma` | $G$ | $\sigma_g$ | $e^{r}$ |
| `b` | $G\times d$ | $b_g$ | unchanged |
| `L` | $G\times d(d+1)/2$ | $Q_g=L_gL_g^\top$ | lower triangle row by row; the diagonal of $L_g$ is $e^{r}$ |
| `gam_y`, `gam_z`, `lam` | $G\times d$, $G\times d\times p$, $G\times d$ | load-mean coefficients | unchanged |
| `log_tau` | $G\times d$ | $\tau_g$ | $e^{r}$ |
| `omega` | 1 if era shifts the noise, else 0 | $\omega$ | unchanged |
| `kappa`, `kappa_ldr` | $G\times n_{treat}$ | treatment maxima | unchanged |

With the 4-domain support of the examples ($n_{sup}=8$, 2 treated domains, era noise) the null model has 112 coordinates, the free model 119 and the joint model 120. With the 10-domain synthetic support ($n_{sup}=22$) they are 322, 343 and 344.

**The objective.** Chapter 2's `value_and_grad(x, lay, prep)` returns

$$
f(x)=-\ell(x),\qquad\nabla f(x)=-\nabla\ell(x),\qquad
\ell(x)=\sum_i\log\sum_{m=1}^{K}w_m\,p\big(y_{i,2:T_i}\mid y_{i1},u_m;x\big),
$$

with the gradient from PyTorch's automatic differentiation, batch by batch of participants. If the model cannot be evaluated at $x$ (a non-finite value or gradient, or a failed Cholesky factorisation), it returns $(10^{10},\mathbf 0)$ instead ([`likelihood.py:205`](../../src/pdx/engine/likelihood.py#L205)). This sentinel matters in §3.7 and in Notes 1.

**New notation in this chapter.**
- $x$ is the raw parameter vector and $\hat x$ the fitted one; $f(x)=-\ell(x)$; $I(x)=-\nabla^2\ell(x)$ is the observed information.
- As in the shared notation, $\ell$ with a parameter argument, $\ell(x)$ or $\hat\ell$, is a log-likelihood, while $\ell_i$, with a participant index, is that participant's load.
- $\rho_g=e^{\gamma_g}$ is phenotype $g$'s speed factor in the speed-only model.
- $\Delta t$ is the interval between two visits, and $c=s\,\Delta t$ the same interval on the reference clock. (The map, §0.4, writes $\tau$ for an interval; this chapter keeps $\tau_g$ for the load spread.)
- $E(c)=e^{cA}$, $M(c)=\int_0^ce^{Aw}\,dw$ and $S(c)=\int_0^ce^{Aw}Qe^{A^\top w}\,dw$ are the transition matrix, the intercept multiplier and the noise covariance over $c$.
- $\mathcal L(X)=AX+XA^\top$ is the Lyapunov operator.

**How the examples were run.** Unless it says otherwise, every example uses one synthetic dataset: scenario S1 of simulator family A, made by `tests/helpers.small_panel("S1", n=80, seed=1, d=4)`. In S1, PIGD runs TD's mechanism at twice the median speed, so H-P's null is true.
- 80 participants are simulated from year 0. After the phenotype rule and the two-visit minimum, 75 remain (39 TD, 36 PIGD), with 422 visits.
- The four domains are tremor, axial, function and independence.
- The truth: own rates −0.25; links function ← tremor 0.10, function ← axial 0.20 and independence ← function 0.25; median speeds 1 and 2 with log-speed spread $\sigma=0.4$; load spread $\tau=0.1$; diagonal noise 0.10 per year; measurement-error variance 0.02; Emax treatment on tremor and axial.
- The model comes from `tests/helpers.spec_for`: Emax treatment, era noise and 12 quadrature nodes. Its support is the 4 own rates plus function ← tremor, function ← axial, independence ← function, and the candidate independence ← axial, which is zero in S1.

The scripts are in [`docs/guide_examples/s03/`](../guide_examples/s03/), and each puts `src/` and `tests/` on the path itself. Run one from the repository folder:

```
OMP_NUM_THREADS=1 ../ppmi-causal/.venv/bin/python docs/guide_examples/s03/ex_fit_pair.py
```

The environment was Python 3.13.12, numpy 2.5.4, scipy 1.18.1 and torch 2.14.1, on one thread of the Apple M4. Timings in the outputs come from that machine. On 4 November 2026 the unit tests passed: `tests/run_tests.py test_engine` (8 passed, 4.4 s) and `tests/run_tests.py test_fit` (1 passed, 127 s).

---

### 3.1 `Fit` — the record of one fit

[`fit.py:20`](../../src/pdx/engine/fit.py#L20)

**What it does.** A dataclass that carries everything a later step needs from one maximum-likelihood fit: which model was fitted, its specification and layout, the raw estimate, the maximised log-likelihood, the optimiser's verdict and the named quantities. Only `fit_model` creates `Fit` objects.

**Inputs → outputs.** The fields:

| field | content |
|---|---|
| `kind` | `"null"`, `"free"` or `"joint"` |
| `spec`, `layout` | the `ModelSpec`, and the `Layout(spec, kind)` used |
| `x` | $\hat x$: the raw vector of the start that reached the highest likelihood |
| `loglik` | $\hat\ell=\ell(\hat x)$: minus scipy's final objective value for that start |
| `success`, `message` | scipy's convergence flag and message, for the kept start only |
| `n_eval` | objective evaluations, summed over **all** starts |
| `params` | `unpack_numpy(x, layout)`: the natural quantities (below) |
| `start_logliks` | the final log-likelihood of every start, in the order they were tried |
| `n_params` (property) | `layout.size` |

`params` holds:
- `A` $(G,d,d)$: each phenotype's drift at the reference speed (for the free model, on the phenotype's own clock, with the speed absorbed);
- `log_speed` $(G,)$: $[0,\gamma_1,\dots]$, all zeros for the free model;
- `Q` $(G,d,d)$: $L_gL_g^\top$, without the era factor;
- `b`, `beta`, `theta`, `gam_y`, `gam_z`, `lam`, `omega`, `kappa`, `kappa_ldr` (zeros where a block is empty), and `sigma` and `tau`, the exponentials of `log_sigma` and `log_tau`;
- `D` $=A_g-A_0$ for the joint model.

**The mathematics.** The record stores the solution of $\hat x=\arg\max_x\ell(x)$ and its value $\hat\ell=\ell(\hat x)$. Two facts are used downstream:
- the free and null vectors differ in length by $(G-1)n_{sup}-(G-1)=(G-1)(n_{sup}-1)$, H-P's degrees of freedom (`hp.h_p_df`);
- `params` is a fixed function of `x`, computed once at line 180, so a `Fit` can report its estimates without the likelihood code.

*From the maths to the code.* The fields are exactly what the consumers read:
- `hp` reads `loglik` and `success`: the statistic and the convergence flags;
- `simulate_from_fit` reads `params` and `spec`;
- `localise` and `gate` read `x`, `layout` and `params`;
- the gate reads `start_logliks`, to measure the gain from the free model's own start.

Keeping `x` raw and `params` natural avoids converting back and forth.

**Worked example** ([`ex_fit_record.py`](../guide_examples/s03/ex_fit_record.py)). A null fit is stopped after 40 iterations on purpose, so that every field shows, including a failed convergence.

```python
f = fit_model(panel, spec, "null", maxiter=40)          # stopped early on purpose
print("kind:", f.kind, "| n_params:", f.n_params, "| len(x):", len(f.x))
print("loglik: %.4f" % f.loglik, "| success:", f.success)
...
```

```text
kind: null | n_params: 112 | len(x): 112
loglik: -537.9900 | success: False
message: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
n_eval: 43 | start_logliks: [-537.99]
params keys: ['A', 'Q', 'b', 'beta', 'gam_y', 'gam_z', 'kappa', 'kappa_ldr', 'lam', 'log_speed', 'omega', 'sigma', 'tau', 'theta']
params['A'] shape: (2, 4, 4) | log_speed: [0.     0.1774]
```

The 40 iterations took 43 evaluations: one per iteration, plus a few extra line-search trials. The converged fit of §3.7 reaches −535.952, so stopping early cost 2.04 log-likelihood units here.

**Design choices and pitfalls.**
- `success` and `message` describe the kept start only. A fit can report success while another start failed, and the reverse.
- `n_eval` is the total over all starts (the counter in `fit_model` is shared), not the cost of the kept start.
- If `fit_model` is given a `penalty`, `loglik` is minus the *penalised* objective. No caller does this today: the penalised localisation fits have their own loop (`localise._penalised_fit`).
- `params["A"]` means different things by kind: at the reference speed for the null and joint models, on each phenotype's own clock for the free model. Code that compares drifts across kinds has to account for $e^{\gamma_g}$.
- `success = True` does not prove that the fit moved (Notes 1).

### 3.2 `_yearly_ols` — a yearly regression per phenotype

[`fit.py:39`](../../src/pdx/engine/fit.py#L39)

**What it does.** For one phenotype, it pools every pair of consecutive visits that are about a year apart and fully observed, and regresses each visit's scores on the previous visit's scores by ordinary least squares (OLS). It returns the yearly matrix, the intercept and the residual covariance, which seed every start value (§3.5).

**Inputs → outputs.** `panel`, phenotype `g`, number of domains `d` → `(B, a, Sigma)`:
- $\hat B$ $(d\times d)$, with $y_{k+1}\approx\hat a+\hat By_k$;
- $\hat a$ $(d,)$;
- $\hat\Sigma$ $(d\times d)$.

With fewer than $d+5$ usable pairs it returns the fixed fallback $(0.8I,\ 0,\ 0.3I)$.

**The mathematics.** Over an interval of exactly one year at speed $s$, the model's exact transition (§3.16) is

$$
X_{k+1}=E\,X_k+M\,(b_g+\ell_i)+\eta_k,\qquad E=e^{sA},\quad M=\int_0^se^{Aw}\,dw,\quad\eta_k\sim N\big(0,S(s)\big).
$$

This is a vector autoregression of order one, VAR(1), in the true levels. The code fits its analogue in the observed scores,

$$
y_{k+1}=a+B\,y_k+r_k,
$$

by OLS. Stack the $n$ usable pairs as rows: $X\in\mathbb R^{n\times d}$ holds the earlier visits, $Y\in\mathbb R^{n\times d}$ the later ones, and $Z=[\mathbf 1,X]\in\mathbb R^{n\times(d+1)}$. Then

$$
\hat C=\arg\min_C\|Y-ZC\|_F^2=(Z^\top Z)^{-1}Z^\top Y\in\mathbb R^{(d+1)\times d},\qquad\hat a=\hat C_{0,\cdot},\qquad\hat B=\big(\hat C_{1:,\cdot}\big)^{\top}.
$$

Each column of $\hat C$ is one domain's regression. The transpose turns "coefficients per outcome column" into "row = target domain, column = source domain", the convention of $A$. The residual covariance is

$$
\hat\Sigma=\frac1{n-1}\sum_k(r_k-\bar r)(r_k-\bar r)^\top,\qquad r_k=y_{k+1}-\hat a-\hat By_k,
$$

where $\bar r=0$ because the regression has an intercept.

$\hat B$ estimates $e^{sA}$ only roughly. Three things pull it away:
- **measurement error** in $y_k$ attenuates it towards zero (errors in variables). With one speed and no load, $\operatorname{plim}\hat B=E\,V_X(V_X+\Sigma_e)^{-1}$, where $V_X$ is the variance of the true levels;
- **speed varies** between participants, so $\hat B$ averages $e^{sA}$ over $s$;
- the **treatment terms** $h$ sit inside $y$ and are ignored.

All three are acceptable in a start value, and the fit corrects them.

*From the maths to the code.*

| line | step |
|---|---|
| 41 | the participants of phenotype $g$ |
| 43–46 | consecutive pairs $(y_k,y_{k+1})$ with $0.5\le t_{k+1}-t_k\le1.6$ years and no missing value in either row |
| 47–48 | fewer than $d+5$ pairs: the fallback $(0.8I,0,0.3I)$ |
| 49–51 | $Z=[\mathbf 1,X]$, and $\hat C$ by `lstsq`, a stable least-squares solve that never forms $(Z^\top Z)^{-1}$ |
| 52–53 | residuals; return $\hat C_{1:}^\top$, $\hat C_0$ and `np.cov(res.T)` (the rows of `res.T` are domains; the `reshape` keeps $d=1$ a matrix) |

A start value needs a cheap, closed-form summary of each phenotype's yearly dynamics. A pooled VAR(1) on roughly annual pairs is the simplest such summary. It maps onto the continuous-time quantities through the matrix logarithm (§3.3), the intercept formula (§3.4) and the residual decomposition (§3.5). The interval window admits roughly annual visits without demanding exact anniversaries.

**Worked example** ([`ex_start_values.py`](../guide_examples/s03/ex_start_values.py), part 3.2):

```python
B, a, Sig = _yearly_ols(panel, g, d)
print(" true yearly matrix at the median speed, expm(%.0f A):\n" % truth.speed[g], linalg.expm(truth.speed[g] * truth.A))
```

```text
== 3.2 _yearly_ols ==
group 0: 122 of 185 consecutive pairs used
 B (yearly matrix):
 [[ 0.542 -0.066 -0.049 -0.095]
 [-0.084  0.699  0.084  0.025]
 [ 0.022  0.133  0.712  0.054]
 [-0.073  0.091  0.197  0.826]]
 true yearly matrix at the median speed, expm(1 A):
 [[0.779 0.    0.    0.   ]
 [0.    0.779 0.    0.   ]
 [0.078 0.156 0.779 0.   ]
 [0.01  0.019 0.195 0.779]]
 a: [ 0.2    0.079  0.06  -0.004] | diag(Sigma): [0.17  0.124 0.138 0.158]
group 1: 124 of 162 consecutive pairs used
 B (yearly matrix):
 [[ 0.797 -0.023  0.082 -0.042]
 [-0.012  0.615  0.002  0.055]
 [ 0.005 -0.027  0.775 -0.074]
 [ 0.068  0.013  0.3    0.573]]
 true yearly matrix at the median speed, expm(2 A):
 [[0.607 0.    0.    0.   ]
 [0.    0.607 0.    0.   ]
 [0.121 0.243 0.607 0.   ]
 [0.03  0.061 0.303 0.607]]
 a: [ 0.042  0.197  0.086 -0.039] | diag(Sigma): [0.173 0.155 0.168 0.17 ]
```

Reading it:
- 63 of TD's 185 pairs were dropped, by the interval window or a missing score.
- With about 120 pairs per phenotype the estimates are rough. TD's persistence of tremor and axial (0.542 and 0.699, against 0.779) is pulled down most. Those are the two treated domains: the simulated treatment starts with a hazard that rises with severity, and its effect sits inside $y$.
- The two largest links are roughly right in TD: function ← axial (0.133 against 0.156) and independence ← function (0.197 against 0.195).

**Design choices and pitfalls.**
- The window ignores the actual interval within $[0.5,1.6]$: a 0.6-year step and a 1.5-year step count alike. That is harmless for a start value; the likelihood uses the exact intervals.
- Only complete pairs are used. At 10 domains with, say, 4% of values missing independently, only $0.96^{20}\approx44\%$ of pairs are complete. The fallback is unlikely at the study's size. It matters only for very small or sparse datasets, where it gives the phenotype a neutral start.
- `np.cov` divides by $n-1$, not $n-d-1$, so $\hat\Sigma$ is slightly too small. That is irrelevant for a start.

### 3.3 `_drift_from_yearly` — from a yearly matrix to a drift

[`fit.py:56`](../../src/pdx/engine/fit.py#L56)

**What it does.** Turns a yearly matrix into a continuous-time drift on that phenotype's own clock: matrix logarithm, then zeros outside the support, then own rates capped at −0.05.

**Inputs → outputs.** $B$ $(d\times d)$ and the `spec` (for the support) → $A$ $(d\times d)$, with $A_{jk}=0$ off the support and $A_{jj}\le-0.05$.

**The mathematics.** If one year on the phenotype's clock gives $B=e^{A^{obs}}$, with the speed absorbed ($A^{obs}=sA$), then

$$
A^{obs}=\log B,
$$

the principal matrix logarithm. It is defined when $B$ has no eigenvalue on the closed negative real axis. The code takes $\mathrm{Re}\,\log B$, because scipy's `logm` returns a complex matrix when $B$ has negative eigenvalues. If `logm` fails or returns non-finite values, the code falls back to the first-order inverse of $e^{A}\approx I+A$, that is $A=B-I$. Then

$$
A_{jk}\leftarrow0\ \text{ for }(j,k)\notin\mathcal S,\qquad A_{jj}\leftarrow\min(A_{jj},-0.05).
$$

The cap keeps every own rate strictly negative, which the raw parametrisation needs: `pack_drift` stores $\log(-A_{jj})$. A rate of −0.05 per year is a persistence of $e^{-0.05}=0.951$ per year.

*From the maths to the code.* Lines 57–62: the logarithm, its real part, the finiteness check and the fallback. Line 63: the support mask (`spec.support` always includes the diagonal). Lines 64–65: the cap on the diagonal. The logarithm inverts the exponential that links continuous time to yearly steps, which is the whole content. The mask and the cap make the result a legal point of the model's parameter space, so that `pack_drift` can encode it.

**Worked example** ([`ex_start_values.py`](../guide_examples/s03/ex_start_values.py), part 3.3):

```python
print("exact case: logm(expm(A)) recovers A on the support, max error %.1e"
      % np.abs(_drift_from_yearly(linalg.expm(A_true), spec) - A_true).max())
Bneg = np.diag([0.9, -0.2, 0.7, 0.8])                   # a yearly matrix with a negative eigenvalue
```

```text
== 3.3 _drift_from_yearly ==
exact case: logm(expm(A)) recovers A on the support, max error 2.8e-17
group 0 start drift (own time scale):
 [[-0.628  0.     0.     0.   ]
 [ 0.    -0.376  0.     0.   ]
 [ 0.053  0.189 -0.359  0.   ]
 [ 0.     0.093  0.251 -0.208]]
group 1 start drift (own time scale):
 [[-0.226  0.     0.     0.   ]
 [ 0.    -0.487  0.     0.   ]
 [ 0.01  -0.037 -0.234  0.   ]
 [ 0.     0.032  0.435 -0.53 ]]
negative eigenvalue: diag of the start drift = [-0.105 -1.609 -0.357 -0.223] | real part of log(-0.2) = -1.609
```

- From the exact yearly matrix, the true drift comes back to rounding.
- From the panel's rough $\hat B$ the start drifts are rough too. The truth is own rates of −0.25 for TD and −0.50 for PIGD at its median speed of 2.
- The last line feeds a diagonal $B$ with an entry of −0.2. Its logarithm is $\log0.2+i\pi$, and the code keeps the real part, −1.609. Since $e^{-1.609}=+0.2$, the result no longer reproduces $B$: the sign is lost silently.

**Design choices and pitfalls.**
- Zeroing the off-support entries after the logarithm is a projection, not the best support-restricted fit. It is good enough for a start.
- A yearly matrix with a negative eigenvalue gives a real part that is not a logarithm of $B$ (above). Only the start is affected.
- Any domain more persistent than 0.951 per year starts at 0.951.

### 3.4 `_load_from_intercept` — from a yearly intercept to a load

[`fit.py:69`](../../src/pdx/engine/fit.py#L69)

**What it does.** Solves for the load $b$ that produces a given yearly intercept $a$, under a drift $A$ at speed $s$.

**Inputs → outputs.** $A$ $(d\times d)$, $a$ $(d,)$ and the speed $s$ → $b$ $(d,)$, with $M(s)\,b=a$.

**The mathematics.** Over one year at speed $s$, the constant input $b$ (with the participant load at its start-value mean of zero) adds

$$
a=M(s)\,b,\qquad M(s)=\int_0^se^{Aw}\,dw,
$$

so $b=M(s)^{-1}a$. $M(s)$ comes from one block matrix exponential (Van Loan 1978). For $\mathcal B=\begin{pmatrix}F&G\\0&0\end{pmatrix}$, every power is $\mathcal B^k=\begin{pmatrix}F^k&F^{k-1}G\\0&0\end{pmatrix}$ for $k\ge1$, so

$$
e^{\mathcal B}=\begin{pmatrix}e^F&\sum_{k\ge1}\frac{F^{k-1}}{k!}G\\0&I\end{pmatrix}=\begin{pmatrix}e^F&\Big(\int_0^1e^{Fv}dv\Big)G\\0&I\end{pmatrix}.
$$

With $F=sA$ and $G=sI$, the upper-right block is $\int_0^1e^{sAv}s\,dv=\int_0^se^{Aw}dw=M(s)$, substituting $w=sv$. When $A$ is invertible, $M(s)=A^{-1}(e^{sA}-I)$; the block form needs no inverse of $A$ and also works for singular drifts.

*From the maths to the code.* Lines 71–72 build the $2d\times2d$ block $\begin{pmatrix}sA&sI\\0&0\end{pmatrix}$. Line 73 takes the upper-right block of its exponential. Line 74 solves $Mb=a$ without an explicit inverse. It is the same block-exponential identity that `transition_np` uses for $M$ (§3.16), applied once and inverted.

**Worked example** ([`ex_start_values.py`](../guide_examples/s03/ex_start_values.py), part 3.4), with the true drift, PIGD's true load and speed 2:

```python
M_block = linalg.expm(np.block([[s * A_true, s * np.eye(d)], [np.zeros((d, d)), np.zeros((d, d))]]))[:d, d:]
M_quad, _ = integrate.quad_vec(lambda w: linalg.expm(A_true * w), 0.0, s)
a = M_block @ b_true
print("true b:", b_true, "| recovered from a = M b:", _load_from_intercept(A_true, a, s))
```

```text
== 3.4 _load_from_intercept ==
M from the block exponential vs numerical integral: max difference 2.2e-16
true b: [ 0.125  0.175 -0.025  0.   ] | recovered from a = M b: [ 0.125  0.175 -0.025  0.   ]
```

**Design choices and pitfalls.**
- $M(s)$ is singular only if $A$ has an eigenvalue $\lambda\ne0$ with $s\lambda=2\pi ik$, $k\ne0$: a purely imaginary pair. The capped start drifts do not produce that in practice.
- In use, $a$ also contains the average treatment effect and the effect of centring. The start load absorbs them, and the fit separates them.

### 3.5 `start_values` — the default start of each kind of model

[`fit.py:77`](../../src/pdx/engine/fit.py#L77)

**What it does.** Builds the default starting vector of a `null`, `free` or `joint` fit from the yearly regressions: drifts, the speed (null and joint), loads and noise covariances, plus fixed values for the speed and load spreads. Everything else starts at zero.

**Inputs → outputs.** `panel`, `spec`, `kind` → $x^{(0)}$, of length `Layout(spec, kind).size`.

**The mathematics.** For each phenotype the yearly fit gives $(B_g,a_g,\Sigma_g)$, and §3.3 gives the drift on its own clock, $A^{obs}_g$.

1. **Speed (null and joint).** Under H-P's null, $A^{obs}_g=s_gA^{obs}_0$. The least-squares speed

   $$
   \hat s_g=\arg\min_s\big\|A^{obs}_g-sA^{obs}_0\big\|_F^2=\frac{\langle A^{obs}_g,A^{obs}_0\rangle_F}{\|A^{obs}_0\|_F^2},\qquad\langle P,R\rangle_F=\sum_{jk}P_{jk}R_{jk},
   $$

   is clipped to $[0.3,5]$ and stored as $\gamma_g=\log\hat s_g$. (Sp-3's $\hat s_{LS}$ is the same projection, applied to the fitted free drifts; Chapter 5.)
2. **Drift.** Null and joint: $A=A^{obs}_0$, TD's drift, since TD is the reference clock. Free: $A_g=A^{obs}_g$ for every $g$. Joint: $D_g=0$, so the joint start is the null start.
3. **Load.** $b_g=M(s_g;A^*_g)^{-1}a_g$ (§3.4). For the free model $A^*_g=A^{obs}_g$ and $s_g=1$; otherwise $A^*_g=A^{obs}_0$ and $s_g=\hat s_g$.
4. **Noise.** Write the yearly residual in terms of the model. With $y=X+e$ (ignoring $h$) and $B_g\approx E$,

   $$
   r_k=y_{k+1}-a_g-B_gy_k\approx\eta_k+e_{k+1}-B_ge_k\qquad\Longrightarrow\qquad\Sigma_g\approx S_g+\Sigma_e+B_g\Sigma_eB_g^\top .
   $$

   Over one year at speed $s$, $S(s)=\sum_{j\ge0}\frac{s^{j+1}}{(j+1)!}\mathcal L^j(Q)=sQ+O(s^2)$ (§3.16), so to first order

   $$
   Q_g\approx\frac{\Sigma_g-\Sigma_e-B_g\Sigma_eB_g^\top}{s_g}.
   $$

   `pack_chol` ([`model.py:227`](../../src/pdx/engine/model.py#L227)) then raises every eigenvalue to at least $10^{-6}$, takes the Cholesky factor and stores the log of its diagonal.
5. **Spreads.** $\sigma_g=0.3$ and $\tau_{g,j}=0.1$ for every phenotype and domain.
6. **Zeros.** $\beta$, $\theta$, $\Gamma^y$, $\Gamma^z$, $\lambda$, $\omega$, $\kappa$ and $\kappa^{ldr}$: no covariate effects, no speed–load link, no era effect on the noise and no treatment effect at the start.

*From the maths to the code.*

| lines | step |
|---|---|
| 78–81 | the layout; yearly fits and drifts for every phenotype |
| 82–85 | $\Sigma_e$, a zero vector, $A^{obs}_0$, speeds of 1 |
| 86–89 | null and joint: $\hat s_g$, clipped; $\gamma=\log\hat s$ |
| 90–96 | `A` ← TD's drift; free: `A{g}`; joint: `D{g}` = 0 |
| 97–104 | per phenotype: $b_g$ and $Q_g$ as above, packed |
| 105–108 | store `b` and `L`; $\log\sigma=\log0.3$; $\log\tau=\log0.1$ |

Each continuous-time block has a yearly counterpart that OLS estimates directly: drift ↔ $B$ (logarithm), load ↔ intercept (Van Loan), noise ↔ residual covariance (minus the measurement error, divided by the speed). The start is these inverses, made legal by the cap and the eigenvalue floor. The spreads have no yearly counterpart, so they get fixed moderate values.

**Worked example** ([`ex_start_values.py`](../guide_examples/s03/ex_start_values.py), part 3.5):

```python
for kind in ("null", "free", "joint"):
    lay = Layout(spec, kind)
    x = start_values(panel, spec, kind)
    par = unpack_numpy(x, lay)
    nll, _ = value_and_grad(x, lay, prep)
```

```text
== 3.5 start_values ==
null : size 112, loglik at start -664.420, gamma start -0.270 (speed 0.764; truth 2)
   b start: [[0.27, 0.094, 0.056, -0.017], [0.069, 0.297, 0.107, -0.076]]
   diag Q start: [[0.144, 0.094, 0.108, 0.123], [0.184, 0.167, 0.177, 0.185]] | sigma: [0.3 0.3] | tau: [0.1 0.1]
free : size 119, loglik at start -644.841
   b start: [[0.27, 0.094, 0.056, -0.017], [0.047, 0.249, 0.1, -0.075]]
   diag Q start: [[0.144, 0.094, 0.108, 0.123], [0.14, 0.127, 0.135, 0.141]] | sigma: [0.3 0.3] | tau: [0.1 0.1]
joint: size 120, loglik at start -664.420, gamma start -0.270 (speed 0.764; truth 2)
   b start: [[0.27, 0.094, 0.056, -0.017], [0.069, 0.297, 0.107, -0.076]]
   diag Q start: [[0.144, 0.094, 0.108, 0.123], [0.184, 0.167, 0.177, 0.185]] | sigma: [0.3 0.3] | tau: [0.1 0.1]
truth: b = [[0.125, 0.125, -0.025, 0.0], [0.125, 0.175, -0.025, 0.0]] | diag Q = 0.10 (reference time scale) | sigma = (0.4, 0.4) | tau = (0.1, 0.1)
```

Reading it:
- With 75 participants the projected speed is far off: 0.764, against a true median ratio of 2. The rough drifts of §3.3 drive it: two of PIGD's four estimated own rates happen to be smaller in magnitude than TD's. The fit moves the speed to 1.297 (§3.7).
- The joint start equals the null start exactly, so their log-likelihoods agree (−664.420).
- PIGD's noise start in the null model (about 0.18) is the free one (about 0.14) divided by the start speed 0.764, as step 4 says.

**Design choices and pitfalls.**
- The null model has this single start (§3.8), and its start speed can be poor in small samples. The engine gate's dispersed-start check (`gate.check_dispersed_starts`; pass if the spread is below 0.1 log-likelihood units) is what tests that the null fit does not depend on it (Notes 2).
- The noise start ignores the load, the treatment terms and the curvature of $S(s)$. The floor of $10^{-6}$ keeps it positive definite even when $\Sigma_e$ is large.
- `start_values` reads only the panel and the specification. `hp._replicate` does not use it for the null fit of a replicate: it warm-starts at the observed null estimate when the support is unchanged.

### 3.6 `null_to` — the speed-only fit written in the free or joint parameters

[`fit.py:112`](../../src/pdx/engine/fit.py#L112)

**What it does.** Takes a raw speed-only vector and returns the vector of the free (or joint) model that describes exactly the same distribution of the data. `fit_pair` starts the free fit there, and that is what makes H-P's likelihood ratio non-negative.

**Inputs → outputs.** `x_null` (raw, in the null layout), `spec`, `kind` ∈ {`"free"`, `"joint"`} → a raw vector in that kind's layout, with the same log-likelihood.

**The mathematics.** *Why the speed-only model is nested in the free model.* Write the speed of a participant in phenotype $g\ge1$ as $s_i=\rho_g\,s_i'$, with $\rho_g=e^{\gamma_g}$ and $\log s_i'=\beta^\top z_i+\theta^\top\tilde y_{i1}+\sigma_gu_i$. The speed-only model's equation becomes

$$
dX=s_i'\big[\rho_gA\,X+\rho_gb_g+\rho_g\ell_i\big]dt+\sqrt{s_i'}\,\big(\rho_g^{1/2}G_g\big)\,dW .
$$

This is a free-model equation without a speed offset. Its drift is $\rho_gA$, its load $\rho_gb_g$, its participant load $\rho_g\ell_i$ and its noise factor $\rho_g^{1/2}G_g$: the speed factor multiplies everything that moves. Since $\ell_i\mid u_i\sim N(\mu_{\ell,i}+\lambda_gu_i,\mathrm{diag}\,\tau_g^2)$, the rescaled load $\rho_g\ell_i$ has mean $\rho_g(\Gamma^y_g\tilde y_{i1}+\Gamma^z_gz_i+\lambda_gu_i)$ and spread $\rho_g\tau_g$. So, for every $g\ge1$,

$$
\boxed{\;A_g^{free}=\rho_gA,\quad b_g^{free}=\rho_gb_g,\quad Q_g^{free}=\rho_gQ_g,\quad\Gamma^{y,free}_g=\rho_g\Gamma^y_g,\quad\Gamma^{z,free}_g=\rho_g\Gamma^z_g,\quad\lambda_g^{free}=\rho_g\lambda_g,\quad\tau_g^{free}=\rho_g\tau_g\;}
$$

Everything else is unchanged: TD's blocks, $\beta$, $\theta$, $\sigma_g$, $\omega$, and the treatment terms, which act on the observed scores rather than on the dynamics. $\gamma$ disappears.

*Check over one interval.* At quadrature node $u_m$ the null model runs at speed $s$ and the free model at $s'=s/\rho_g$. With $c=s\Delta t$ and $c'=s'\Delta t=c/\rho_g$, and substituting $v=\rho_gw$:

$$
E'=e^{c'\rho_gA}=e^{cA}=E,\qquad M'(b'+\ell')=\Big(\int_0^{c'}e^{\rho_gAw}dw\Big)\rho_g(b+\ell)=\Big(\int_0^{c}e^{Av}dv\Big)(b+\ell)=M(b+\ell),
$$

$$
S'=\int_0^{c'}e^{\rho_gAw}\,\rho_gQ\,e^{\rho_gA^\top w}\,dw=\int_0^ce^{Av}Qe^{A^\top v}\,dv=S .
$$

The Kalman filter on the augmented state $[X;\ell]$ therefore makes the same predictions of the observed scores at every node. (The load state is rescaled by $\rho_g$, a linear change of variables that leaves the likelihood of the scores unchanged.) The quadrature sums the same terms, so $\ell^{free}(\mathrm{null\_to}(x))=\ell^{null}(x)$.

This works only because everything about the load is phenotype-specific ([`model.py:11`](../../src/pdx/engine/model.py#L11)). If $\Gamma^y$, $\Gamma^z$, $\lambda$ or $\tau$ were shared between phenotypes, the free model could not rescale PIGD's load distribution, and the null would not be nested.

*In raw coordinates* the map is:
- own-rate raw values $r\mapsto r+\gamma_g$, since $-e^{r+\gamma_g}=\rho_g\cdot(-e^r)$; links $\times\rho_g$;
- `b`, `gam_y`, `gam_z`, `lam` $\times\rho_g$; `log_tau` $+\gamma_g$;
- the Cholesky factor of $\rho_gQ_g$ is $\rho_g^{1/2}L_g$, so its log-diagonal $+\gamma_g/2$ and its off-diagonal $\times\rho_g^{1/2}$.

The code reaches the drift and the noise through `pack_drift` and `pack_chol` instead of these formulas (see the pitfalls).

*The joint model* with $D_g=0$ and the same $\gamma$ is the speed-only model itself, so the joint map is a copy plus zero deviations.

*From the maths to the code.*

| lines | step |
|---|---|
| 114–118 | copy every non-empty block the two layouts share, except `gamma`: `A`, `beta`, `theta`, `log_sigma`, `b`, `L`, `gam_y`, `gam_z`, `lam`, `log_tau`, `omega`, `kappa`, `kappa_ldr` |
| 119–123 | joint: copy `gamma`, set `D{g}` = 0, return |
| 124–127 | free: the null's natural quantities; $\rho_g=e^{\gamma_g}$ from `log_speed`, with $\rho_0=1$; copies of `L` and `b` |
| 128–131 | for $g\ge1$: `A{g}` = `pack_drift`($\rho_gA$); $b_g\leftarrow\rho_gb_g$; $L_g\leftarrow$ `pack_chol`($\rho_gQ_g$) |
| 132–133 | store `b` and `L`, overwriting the unscaled copies |
| 135–139 | `gam_y` $(G,d)$, `gam_z` $(G,d,p)$, `lam` $(G,d)$: rows $g\ge1$ $\times\rho_g$ (the reshape broadcasts over the trailing axes) |
| 140–142 | `log_tau`: rows $g\ge1$ $+\gamma_g$ |

The derivation says "multiply everything that moves by $\rho_g$". The code copies everything first and then overwrites exactly the blocks that move: the drift, the load and its mean coefficients, the load spread and the noise.

**Worked example** ([`ex_null_to.py`](../guide_examples/s03/ex_null_to.py)). A speed-only vector with every block non-zero (the unit test's random point) and $\gamma=\log1.8$ is written in the free and joint layouts.

```python
x0 = random_point(panel, spec, "null", seed=5)          # every block non-zero (load, era noise, ...)
x0[l0.slices["gamma"][0]] = np.log(1.8)                 # PIGD runs at 1.8 times the reference speed
ll0 = float(loglik(torch.as_tensor(x0), l0, prep))
for kind in ("free", "joint"):
    l1 = Layout(spec, kind)
    x1 = null_to(x0, spec, kind)
    ll1 = float(loglik(torch.as_tensor(x1), l1, prep))
```

```text
speed-only log-likelihood: -755.8999538882
free  layout (119 parameters): -755.8999538882  difference +0.0e+00
joint layout (120 parameters): -755.8999538882  difference +0.0e+00

free-layout quantities of PIGD against e^gamma times the speed-only ones (max |difference|):
  A      0.0e+00
  b      0.0e+00
  Q      2.2e-16
  tau    0.0e+00
  lam    0.0e+00
  gam_y  0.0e+00
  gam_z  0.0e+00
unchanged: TD's blocks 0.0e+00, sigma 0.0e+00, beta 0.0e+00, theta 0.0e+00, kappa 0.0e+00, omega 0.0e+00
free log-speed offsets: [0. 0.] | PIGD own rates, speed-only vs free: [-0.6031 -0.3522 -0.3803 -0.1999] [-1.0857 -0.6339 -0.6846 -0.3598]
```

The log-likelihood agrees to every printed digit in both layouts. PIGD's own rates are multiplied by 1.8, and so is every other moving part. The unit test `test_null_is_nested_in_free_and_joint` checks the same identity to $10^{-8}$, relative.

**Design choices and pitfalls.**
- `pack_drift` floors $-\rho_gA_{jj}$ at $10^{-4}$, and `pack_chol` floors the eigenvalues of $\rho_gQ_g$ at $10^{-6}$. Beyond those floors the map is no longer exact: Notes 4 finds gaps of $10^{-3}$ to $10^{-2}$ log-likelihood units in constructed cases. Writing the raw values directly, as above, would make it exact everywhere.
- The likelihood clamps each node's log-speed to ±15 ([`likelihood.py:140`](../../src/pdx/engine/likelihood.py#L140)). The clamp acts on $\gamma_g+\dots$ in the null model and on the offset-free value in the free model, so nodes at the clamp break the identity. That happens only at absurd speeds (Notes 4).
- `kind="null"` is not supported: the free branch would look for an `A1` block that the null layout does not have.
- Three places rely on this map: `fit_pair` (LR ≥ 0), `gate.check_quadrature_lr` (the statistic at 12, 20 and 30 nodes) and `localise.joint_discovery` (the gradient at the null, in the joint layout, sets the lasso's largest penalty).

### 3.7 `fit_model` — maximum likelihood by L-BFGS-B from one or more starts

[`fit.py:148`](../../src/pdx/engine/fit.py#L148)

**What it does.** Maximises the log-likelihood of one kind of model with scipy's L-BFGS-B optimiser, using the exact gradient, from each start in turn, and keeps the best.

**Inputs → outputs.**
- `panel`, `spec`, `kind`.
- Starts: `x0` if given (for example a warm start); otherwise `start_values(...)` if `use_default_start`; plus every vector in `extra_starts`.
- `prep` (the prepared data of Chapter 2, built if not given), `chunk` (participants per batch), `penalty` (a differentiable function added to $f$), `bounds` (one (low, high) pair per coordinate; `None` means unbounded).
- Optimiser settings: `maxiter` 3000, `ftol` $10^{-10}$, `gtol` $10^{-5}$, `maxcor` 50; `maxfun` is set to 4 × `maxiter`.
- → a `Fit` (§3.1).

**The mathematics.** *Maximum likelihood.* The estimate is

$$
\hat x=\arg\max_x\ell(x)=\arg\min_xf(x),\qquad f(x)=-\ell(x)\ \big(+\ \text{penalty}(x)\big).
$$

Under the usual regularity conditions $\hat x$ is consistent and asymptotically normal, with covariance $I(\hat x)^{-1}$ (§3.9–§3.12). The independent units are the participants: the speed and the load are integrated out within each participant, so $\ell$ is a sum of independent terms.

*L-BFGS-B* (limited-memory Broyden–Fletcher–Goldfarb–Shanno with bounds; Byrd, Lu, Nocedal and Zhu 1995) is a quasi-Newton method. At iterate $x_k$ it moves along $p_k=-H_k\nabla f(x_k)$, where $H_k$ approximates the inverse Hessian $(\nabla^2f)^{-1}$. $H_k$ is built from the last $m$ pairs

$$
s_j=x_{j+1}-x_j,\qquad y_j=\nabla f(x_{j+1})-\nabla f(x_j),
$$

through the BFGS update, which makes $H$ satisfy the secant condition $Hy_j=s_j$. A line search picks the step length $\alpha$ so that the Wolfe conditions hold, among them sufficient decrease:

$$
f(x_k+\alpha p_k)\le f(x_k)+c_1\,\alpha\,\nabla f(x_k)^\top p_k .
$$

**Every accepted step therefore lowers $f$**, a property §3.8 relies on. Bounds are handled in two stages: a search along the projected steepest-descent path onto the box (the "generalised Cauchy point"), then the quasi-Newton step on the coordinates not at a bound. The settings mean the following here.
- `maxcor` $=m=50$ pairs (scipy's default is 10). More pairs give a better curvature model when parameters are strongly correlated, as speed and drift are here. The cost, $O(mn)$ per iteration, is negligible next to one likelihood evaluation.
- `ftol`: stop when $\dfrac{f_k-f_{k+1}}{\max(|f_k|,|f_{k+1}|,1)}\le\text{ftol}$. With $|f|\approx536$ in the example, the run stops once an iteration gains less than $5.4\times10^{-8}$ log-likelihood units. At a real-size $|f|$ of a few times $10^4$ the threshold is a few times $10^{-6}$. In terms of `factr`, the argument of scipy's `fmin_l_bfgs_b`, this is $\text{ftol}/\varepsilon_{mach}=4.5\times10^5$: between what scipy calls "moderate" ($10^7$) and "extremely high" (10) accuracy.
- `gtol`: stop when the largest component of the projected gradient is at most $10^{-5}$. On a log-likelihood summed over many participants this is rarely reached first; in the example `ftol` decides.
- `maxiter` and `maxfun` cap the iterations and the evaluations (the line search can take several evaluations per iteration). Hitting either sets `success = False`.

*Several starts.* The function minimises from each start and keeps the run with the lowest final objective,

$$
\hat x=\hat x^{(r^*)},\qquad r^*=\arg\min_rf\big(\hat x^{(r)}\big),
$$

with ties going to the earlier start.

*From the maths to the code.*

| lines | step |
|---|---|
| 157–158 | the layout; the prepared data, shared by all starts |
| 159–164 | the starts: `x0` if given, otherwise the default start if `use_default_start`; then the extra starts |
| 165–169 | the objective `f(v)` returns $(f,\nabla f)$ from `value_and_grad` and counts calls (in a dict, so that the closure can update it) |
| 171–178 | for each start: `minimize(..., jac=True, method="L-BFGS-B")`; record $-f$ at the final point; keep the best |
| 179–180 | the `Fit`: the best `x`, $\hat\ell=-f$, flag, message, total evaluations, `unpack_numpy`, all finals |

The maths is "minimise $f$", and scipy minimises, hence the sign. `jac=True` lets one PyTorch pass return the value and the gradient together, which halves the work against separate calls. L-BFGS-B suits a smooth objective with 100 to 350 parameters whose Hessian is expensive (§3.9). Bounds come for free, and the localisation and profile fits use them to fix coordinates. The loop over starts is the protocol's "started both at its default and at the speed-only fit" (§7.1), written generally.

**Worked example** ([`ex_fit_model.py`](../guide_examples/s03/ex_fit_model.py)):

```python
f = fit_model(panel, spec, "null", prep=prep)              # defaults: ftol 1e-10, gtol 1e-5, maxcor 50
nll, g = value_and_grad(f.x, f.layout, prep)
f6 = fit_model(panel, spec, "null", prep=prep, ftol=1e-6)
far = start_values(panel, spec, "null") + 0.3 * rng.normal(size=f.layout.size)   # a dispersed start
fm = fit_model(panel, spec, "null", prep=prep, extra_starts=[far])
```

```text
default settings: loglik -535.952465, success True, 370 evaluations, 16 s
  message: CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH
  largest |gradient| at the optimum: 1.90e-02 (gtol 1e-5)
  ftol threshold on the last decrease: 1e-10 x max(|f|, 1) = 5.4e-08
  speed ratio exp(gamma) = 1.297 (truth: median speed ratio 2)
ftol 1e-6: loglik -535.972264 (2.0e-02 below), 118 evaluations, 4 s
two starts (default, dispersed): start_logliks [-535.952465 -536.017488] | kept -535.952465 | 663 evaluations in all, 24 s
```

Reading it:
- The run stopped on `ftol`, with a gradient component of 0.019, far above `gtol`. §3.12 shows how close that is to the exact optimum: the Newton step $\tfrac12\nabla f^\top I^{-1}\nabla f$ is $7.8\times10^{-7}$ log-likelihood units.
- With `ftol` $=10^{-6}$ the run stops 0.020 units short, after a third of the evaluations. In a likelihood ratio that would be an error of up to 0.04.
- A dispersed start ended 0.065 units lower: the two runs stopped at different points of a flat region, or at two local optima. The engine gate's dispersed-start check accepts a spread below 0.1.
- The speed estimate, 1.297, comes from one small dataset. The unit test `test_fit` recovers the speed ratio (it requires Sp-1 between 1.4 and 2.8) with 400 participants.

**Design choices and pitfalls.**
- **A start at which the likelihood cannot be evaluated is reported as converged.** `value_and_grad` returns $(10^{10},\mathbf 0)$ there, and with a zero gradient L-BFGS-B stops at once with `success = True` (Notes 1).
- `x0` and the default start are exclusive: when `x0` is given, `use_default_start` is ignored. With neither, and no extra starts, the function fails with an unhelpful `AttributeError` on `None`.
- Bounds of the form $(v,v)$ fix a coordinate (`gate.profile`, `localise.anchored_tests`); L-BFGS-B then never moves it.
- Every start costs a full fit, so the free fit in `fit_pair` costs about twice a single fit.
- The result does not depend on `chunk` beyond rounding: the batch size only changes the order of summation.

### 3.8 `fit_pair` — the speed-only and free fits of H-P

[`fit.py:183`](../../src/pdx/engine/fit.py#L183)

**What it does.** Fits the speed-only model (H-P's null), then the free model (the alternative) from two starts: the null solution written in the free parameters, and the free model's own default start. Returns both fits.

**Inputs → outputs.** `panel`, `spec`, optional `prep`, `x0_null` (a warm start for the null fit), `own_free_start` (default `True`) and `**kw`, passed to both `fit_model` calls → `(f0, f1)`. `hp._replicate` and `speed.bootstrap_sample` pass the observed null estimate as `x0_null` when the replicate's support is unchanged.

**The mathematics.** H-P's statistic and its degrees of freedom are

$$
\mathrm{LR}=2\big(\hat\ell_{free}-\hat\ell_{null}\big),\qquad\mathrm{df}=\dim x_{free}-\dim x_{null}=(G-1)(n_{sup}-1).
$$

Because the null is nested (§3.6), the free fit's first start $x^{(1)}=\mathrm{null\_to}(\hat x_{null})$ already has $\ell_{free}(x^{(1)})=\hat\ell_{null}$. L-BFGS-B accepts only steps that lower $f$ (§3.7), so that run ends with $\ell\ge\hat\ell_{null}$, and taking the best of the starts can only raise it:

$$
\hat\ell_{free}=\max_r\ell_{free}\big(\hat x^{(r)}_{free}\big)\ \ge\ \ell_{free}\big(x^{(1)}\big)=\hat\ell_{null}\qquad\Longrightarrow\qquad\mathrm{LR}\ge0,
$$

up to rounding and the floors of Notes 4. `hp.lr_stat` clamps the statistic at 0 and keeps the raw value.

Why a second start: the null solution can lie in the basin of a free local optimum that is not the best one, and the free model's own start explores another basin. The protocol asks for both (§7.1). The engine gate records how much the second start ever adds (`default_start_gain` in `gate.recovery_dataset`).

*From the maths to the code.* Line 192 prepares the data once for both fits. Line 193 fits the null from `x0_null`, or from its default start when that is `None`. Lines 194–195 fit the free model with `x0 = null_to(f0.x, spec, "free")` and, if `own_free_start`, `extra_starts = [start_values(panel, spec, "free")]`. So `f1.start_logliks[0]` comes from the null solution and `f1.start_logliks[1]` from the default start. The protocol's sentence "the free model is started both at its default and at the speed-only fit, so the likelihood ratio is never negative" becomes two calls of `fit_model`, and the order of the starts fixes the meaning of `start_logliks`.

**Worked example** ([`ex_fit_pair.py`](../guide_examples/s03/ex_fit_pair.py)):

```python
f0, f1 = fit_pair(panel, spec)
lr = 2 * (f1.loglik - f0.loglik)
```

```text
null: loglik -535.952465, 112 parameters, success True, 370 evaluations
free: loglik -527.186490, 119 parameters, success True, 655 evaluations
free start_logliks [from the null solution, from its own default]: [-527.219167 -527.18649 ]
LR = 2 (l_free - l_null) = 17.5320 on df = 7 (= n_params difference 7); 57 s
speed-only speed exp(gamma) = 1.297; least-squares speed of the free drifts = 1.035
```

Reading it:
- Both free starts end above the null's −535.952, as the argument says.
- Here the free model's own default start ended 0.033 units higher than the start at the null solution, so the second start changed the statistic by 0.065.
- The statistic, 17.53, exceeds the 95% point of $\chi^2_7$, 14.07 ($\chi^2$ p = 0.014), although H-P's null is true in S1. With 75 participants and 112 to 119 parameters the asymptotic $\chi^2$ reference cannot be trusted. That is why H-P's reference is the conditional parametric bootstrap (Chapter 5), whose calibration WP3 checks.

**Design choices and pitfalls.**
- `**kw` goes to both fits, so options with layout-specific content cannot pass through it: `bounds` has the wrong length for one of the two layouts, and `x0` or `extra_starts` raise a duplicate-keyword `TypeError`.
- The null fit has one start. On the observed data it is the default start; in a bootstrap replicate it is the observed null estimate, which is the replicate's truth. If the observed null fit stopped in a worse optimum than the replicates do, the observed statistic would be inflated relative to the bootstrap distribution (Notes 2).
- The degrees of freedom are reported but do not enter the p-value.

### 3.9 `hessian` — the observed information by central differences of exact gradients

[`fit.py:199`](../../src/pdx/engine/fit.py#L199)

**What it does.** Computes the matrix of second derivatives of minus the log-likelihood at a fit, the observed information, by differencing the exact gradient along each coordinate. Raises an error if the likelihood cannot be evaluated next to the fit.

**Inputs → outputs.** `fit`, `panel`, optional `prep`, the step `h` ($10^{-5}$) and `chunk` → $\hat I$ $(n\times n)$, symmetric, in raw coordinates.

**The mathematics.** The observed information is

$$
I(\hat x)=-\nabla^2\ell(\hat x)=\nabla^2f(\hat x).
$$

Its $j$-th column is the derivative of the gradient along coordinate $j$. Expanding the exact gradient,

$$
\nabla f(\hat x\pm he_j)=\nabla f(\hat x)\pm h\,Ie_j+\tfrac{h^2}{2}\,T[e_j,e_j]\pm\tfrac{h^3}{6}\,U[e_j,e_j,e_j]+O(h^4),
$$

where $T$ and $U$ collect the third and fourth derivatives. The even terms cancel in the central difference:

$$
\hat I_{\cdot j}=\frac{\nabla f(\hat x+he_j)-\nabla f(\hat x-he_j)}{2h}=Ie_j+O(h^2).
$$

Rounding adds an error of order $\varepsilon_g/h$, where $\varepsilon_g$ is the absolute rounding error of the gradient, itself a sum over participants. With $h=10^{-5}$ both errors are small. The exact $I$ is symmetric and the difference quotient is not quite, so the code returns $\tfrac12(\hat I+\hat I^\top)$. The asymmetry it removes measures the numerical error.

*From the maths to the code.* Lines 203–205: the prepared data, $n$ and an $n\times n$ zero matrix. Lines 206–213: for each coordinate, two calls of `value_and_grad` (without penalty) at $\hat x\pm he_j$; if either value is the $10^{10}$ sentinel, raise; otherwise fill column $j$. Line 214: symmetrise. Only first derivatives are available cheaply, and `value_and_grad` already processes the participants in batches to bound memory. Differencing it costs $2n$ gradient evaluations, needs no new code, and has an error of $O(h^2)$ rather than the $O(h)$ of a forward difference. An exact alternative would be Hessian-vector products by a second backward pass, batch by batch: a similar cost, more memory and more code. Protocol §7.1 says that second derivatives are taken by automatic differentiation; the code does not do that (Notes 6).

**Worked example** ([`ex_hessian.py`](../guide_examples/s03/ex_hessian.py), first part; the null fit of §3.7):

```python
# the loop of hessian(), repeated here only to look at H before it is symmetrised
for j in range(n):
    e = np.zeros(n)
    e[j] = 1e-5
    Hraw[:, j] = (value_and_grad(f0.x + e, lay, prep)[1] - value_and_grad(f0.x - e, lay, prep)[1]) / 2e-5
H = hessian(f0, panel, prep)
H4 = hessian(f0, panel, prep, h=1e-4)
```

```text
112 coordinates, 224 gradient evaluations per Hessian, 10 s per Hessian
hessian() equals the symmetrised loop: max difference 0.0e+00
asymmetry before symmetrising: max |H - H'| / max |H| = 1.0e-08
h = 1e-4 against h = 1e-5: max |difference| / max |H| = 4.2e-06
```

The asymmetry is $10^{-8}$ of the largest entry, and a ten times larger step changes the matrix by $4\times10^{-6}$ of it. The differences are accurate to five or more significant digits relative to the largest entries.

**Design choices and pitfalls.**
- Cost: $2n$ gradient evaluations. At the 10-domain synthetic size the null model has 322 coordinates, so 644 evaluations. At the protocol's figure of 2.3 s per evaluation (§7.1) that is about 25 minutes for one Hessian; `likelihood.default_chunk` notes that smaller batches have since made evaluations up to 1.7 to 1.9 times faster.
- The step is absolute in raw coordinates. Most raw coordinates are logs or of order one, so one step suits them all; a coordinate with a very large raw value would need a relative step.
- It is the information of the unpenalised likelihood. For a bounded fit it is computed over all coordinates, fixed ones included; `localise.anchored_tests` drops those before inverting.
- Near a variance component at zero, the information about its log is close to zero (§3.12).

### 3.10 `covariance` — the inverse information without its report

[`fit.py:217`](../../src/pdx/engine/fit.py#L217)

**What it does.** Returns `covariance_checked(H)[0]`: the (pseudo-)inverse of the information, discarding the diagnostic report.

**Inputs → outputs.** $H$ → $\hat V$.

**The mathematics.** That of §3.12; nothing new.

**Worked example.** The last line of [`ex_covariance_checked.py`](../guide_examples/s03/ex_covariance_checked.py) confirms it returns the same matrix as `covariance_checked`: `covariance() gives the same matrix: True`.

**Design choices and pitfalls.** Nothing in `src/`, `scripts/` or `tests/` calls it, so it is dead code. If it were used, the report that should travel with any interval (the docstring of §3.12) would be lost (Notes 8).

### 3.11 `param_names` — a name for every coordinate

[`fit.py:222`](../../src/pdx/engine/fit.py#L222)

**What it does.** Lists a name for every coordinate of a layout's parameter vector, in vector order, as `block[flat index]`. `covariance_checked` uses the names in its flat-direction report, and `localise` and `gate` use them to find coordinates.

**Inputs → outputs.** A `Layout` → a list of `layout.size` strings, such as `'A[0]'`, `'gamma[0]'` or `'log_tau[5]'`. Empty blocks contribute no names.

**The mathematics.** Only an indexing rule. `Layout.get` reshapes a block's slice in row-major (C) order, so the flat index $k$ is:
- $k=g\,d+j$ for a block of shape $(G,d)$, and $k=(g\,d+j)\,p+q$ for shape $(G,d,p)$;
- $k=g\cdot\tfrac{d(d+1)}{2}+r$ for `L`, where $r$ runs over the lower triangle row by row: (0,0), (1,0), (1,1), (2,0), …;
- the position in `spec.sup_idx` for `A`, `A{g}` and `D{g}`: the support's (to, from) pairs, row by row.

So for a $(G,d)$ block, $(g,j)=\mathrm{divmod}(k,d)$.

*From the maths to the code.* Lines 224–227 walk the blocks in layout order and number each block's entries $0,\dots,\prod(\text{shape})-1$: the same order in which `Layout` lays them out.

**Worked example** (last line of [`ex_hessian.py`](../guide_examples/s03/ex_hessian.py)):

```python
print("param_names examples:", names[0], names[8], names[29], names[100], "| total", len(names))
```

```text
param_names examples: A[0] gamma[0] L[2] log_tau[5] | total 112
```

At $d=4$:
- `A[0]` is `spec.sup_idx[0]` = (0, 0), tremor's own rate, stored as $\log(-A_{00})$;
- `L[2]` is TD's (1, 1) Cholesky entry, stored as a log;
- `log_tau[5]` is $\mathrm{divmod}(5,4)=(1,1)$: PIGD's load spread on axial.

**Design choices and pitfalls.** The flat index hides the phenotype and the domain. A report entry such as `log_tau[7]` has to be decoded as above (PIGD, independence). The names refer to raw coordinates: `log_tau[k]` is $\log\tau$, not $\tau$.

### 3.12 `covariance_checked` — the inverse information with its checks

[`fit.py:230`](../../src/pdx/engine/fit.py#L230)

**What it does.** Inverts the observed information through its eigendecomposition and reports whether that is safe: the smallest eigenvalue, the condition number, positive definiteness, good conditioning, and the coordinates along every nearly flat direction. Directions with zero or negative information are left out of the inverse.

**Inputs → outputs.** $H$ $(n\times n)$, `max_condition` ($10^{10}$) and optional `names` → $(\hat V,\text{report})$. The report's keys are `min_eigenvalue`, `condition_number`, `positive_definite`, `well_conditioned` and `flat_directions` (a list of {eigenvalue, the top three coordinates with their loadings}).

**The mathematics.** *The covariance.* Asymptotically $\hat x\approx N(x,I^{-1})$, so the standard error (SE) of coordinate $j$ is $\sqrt{(I^{-1})_{jj}}$ and a Wald interval is $\hat x_j\pm1.96\,\mathrm{SE}_j$. The code writes the symmetrised information as

$$
H_s=\tfrac12(H+H^\top)=V\Lambda V^\top,\qquad\Lambda=\mathrm{diag}(\lambda_1\le\dots\le\lambda_n),
$$

and returns the inverse restricted to the positive eigenvalues,

$$
\hat V=\sum_{k:\,\lambda_k>0}\frac{v_kv_k^\top}{\lambda_k}.
$$

If $H_s$ is positive definite, this is $H_s^{-1}$. Otherwise the directions with $\lambda_k\le0$ get variance 0, which understates the uncertainty along them.

*The checks.*

$$
\lambda_{min}=\lambda_1,\qquad\kappa=\frac{\lambda_{max}}{\min_{k:\,\lambda_k>0}\lambda_k},\qquad\text{positive definite}=[\lambda_1>0],\qquad\text{well conditioned}=\text{positive definite and }\kappa<10^{10}.
$$

A direction is *flat* if $\lambda_k\le\lambda_{max}/10^{10}$, which includes every non-positive one. For each flat direction the report lists the three coordinates with the largest $|v_{ik}|$, so the direction can be traced to the parameters that move along it.

*Why flat directions appear at variance components near zero.* The likelihood depends on a load spread only through $\tau^2=e^{2r}$, with $r=\log\tau$. By the chain rule,

$$
\frac{\partial\ell}{\partial r}=2\tau^2\,\frac{\partial\ell}{\partial(\tau^2)},\qquad
\frac{\partial^2\ell}{\partial r^2}=4\tau^2\,\frac{\partial\ell}{\partial(\tau^2)}+4\tau^4\,\frac{\partial^2\ell}{\partial(\tau^2)^2},
$$

and both vanish as $\tau\to0$. When the data favour no load variation in a domain, the estimate drifts towards $r\to-\infty$ until `ftol` stops it, and the information there is essentially zero. The same holds for $\sigma_g$ and for the log-diagonal of the noise's Cholesky factor. Such directions are flat because of the log parametrisation, not because the rest of the model is unidentified. `gate.identification` therefore sets aside spreads below 0.01 before judging the conditioning of the rest.

*From the maths to the code.* Line 237: symmetrise. Line 238: the eigendecomposition (`eigh`, eigenvalues ascending). Lines 239–242: the positive mask, the condition number over the positive eigenvalues, and the report's flags. Lines 243–248: the flat directions, with their three largest loadings rounded to three decimals. Lines 249–250: $1/\lambda$ where positive and 0 elsewhere, then $\hat V=V\,\mathrm{diag}(1/\lambda_+)\,V^\top$. An inverse through the eigendecomposition gives the checks for free: the eigenvalues give the condition number, and the eigenvectors of the small ones name the culprits. The pseudo-inverse keeps the function total (it always returns a matrix), and the report says what was left out.

**Worked example 1, by hand** ([`ex_covariance_checked.py`](../guide_examples/s03/ex_covariance_checked.py)). A $3\times3$ information with eigenvalues 50 (ordinary), $10^{-12}$ (flat) and −0.5 (negative), rotated in the (a, b) plane:

```python
V = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])   # a rotation in the (a, b) plane
w = np.array([50.0, 1e-12, -0.5])                              # ordinary, flat, negative
H = V @ np.diag(w) @ V.T
Cov, rep = covariance_checked(H, names=["a", "b", "c"])
```

```text
eigenvalues of H: [-5.000e-01  9.998e-13  5.000e+01]
  min_eigenvalue: -0.5
  condition_number: 49965806127365.26
  positive_definite: False
  well_conditioned: False
  flat direction: {'eigenvalue': -0.5, 'coordinates': [('c', 1.0), ('a', 0.0), ('b', 0.0)]}
  flat direction: {'eigenvalue': 1.0006843454611253e-12, 'coordinates': [('b', -0.955), ('a', 0.296), ('c', 0.0)]}
covariance returned:
 [[ 8.7272e+10 -2.8213e+11  0.0000e+00]
 [-2.8213e+11  9.1204e+11  0.0000e+00]
 [ 0.0000e+00  0.0000e+00  0.0000e+00]]
it equals (1/50) v1 v1' + 1e12 v2 v2' to 0.1% (the negative direction is dropped): True
variance of c (negative information): 0.0 | covariance() gives the same matrix: True
```

Reading it:
- The condition number, $50/10^{-12}\approx5\times10^{13}$, uses the smallest *positive* eigenvalue and so ignores −0.5. `positive_definite` is what flags the negative one.
- Coordinate c has negative information and comes back with variance 0, the most confident value possible.
- The flat eigenvalue itself is known only roughly. numpy's `eigvalsh` (first line) and `eigh` (inside the function) return $9.998\times10^{-13}$ and $1.0007\times10^{-12}$, because it sits at rounding level relative to 50. Its inverse, the variance along (a, b), is therefore an arbitrary large number.

**Worked example 2, the fitted null** ([`ex_hessian.py`](../guide_examples/s03/ex_hessian.py), second part):

```python
names = param_names(lay)
V, rep = covariance_checked(H, names=names)
keep = [i for i, nm in enumerate(names) if not (nm.startswith(("log_tau", "log_sigma")) and f0.x[i] < np.log(0.01))]
_, rest = covariance_checked(H[np.ix_(keep, keep)])
```

```text
report: min eigenvalue 5.506e-12, condition number 1.291e+15, positive definite True, well conditioned False
flat directions: [{'eigenvalue': 5.506491107331602e-12, 'coordinates': [('log_tau[0]', 1.0), ('log_tau[7]', 0.0), ('log_tau[4]', 0.0)]}, {'eigenvalue': 7.744305777745628e-09, 'coordinates': [('log_tau[7]', -1.0), ('log_tau[0]', 0.0), ('log_tau[4]', -0.0)]}, {'eigenvalue': 7.100617877957897e-08, 'coordinates': [('log_tau[4]', -1.0), ('log_tau[3]', -0.0), ('log_tau[7]', 0.0)]}]
four smallest eigenvalues: ['5.5e-12', '7.7e-09', '7.1e-08', '1.2e-06'] | largest: 7108.6
smallest diagonal information: [('log_tau[0]', '5.6e-12', 'tau = 6.2e-08'), ('log_tau[7]', '7.7e-09', 'tau = 3.4e-06'), ('log_tau[4]', '7.1e-08', 'tau = 1.6e-05')]
without the 4 spreads below 0.01 (as gate.identification does): condition number 6.662e+04, well conditioned True
  gamma[0]      estimate +0.2601  standard error 0.3769
  log_sigma[0]  estimate -0.2398  standard error 0.1855
  log_sigma[1]  estimate -1.1695  standard error 0.4394
  log_tau[0]    estimate -16.5998  standard error 426150.0354
  omega[0]      estimate -0.3016  standard error 0.2513
distance to the exact optimum (Newton step), 0.5 g' I^-1 g = 7.8e-07 log-likelihood units
```

Reading it:
- In this small sample four load spreads fell below 0.01 (the truth is 0.1): TD's on tremor (`log_tau[0]`, $\tau=6\times10^{-8}$), and PIGD's on independence (`log_tau[7]`), tremor (`log_tau[4]`) and axial (`log_tau[5]`, $\tau=6.5\times10^{-5}$). Each gives a near-zero eigenvalue along its own coordinate alone, as the chain-rule argument predicts.
- Only three of them are listed as flat. The fourth, `log_tau[5]`, has eigenvalue $1.2\times10^{-6}$ (checked separately), just above the flat threshold $\lambda_{max}/10^{10}=7.1\times10^{-7}$. The report's relative threshold and `gate.identification`'s rule (spreads below 0.01) are different criteria.
- The condition number is $1.3\times10^{15}$, so the information is "not well conditioned". Without the four spreads below 0.01 it is $6.7\times10^4$, well conditioned.
- The pseudo-inverse keeps those tiny positive eigenvalues, so `log_tau[0]` gets an SE of 426,150. That number means "not identified"; it is not an interval width to report.
- The SE of $\gamma$, 0.38, gives a model-based 95% interval for $e^{\gamma}$ of $e^{0.26\pm1.96\times0.38}$, about 0.6 to 2.7, which contains the truth of 2.
- $\tfrac12\nabla f^\top\hat V\nabla f=7.8\times10^{-7}$: the `ftol` stop of §3.7 left the fit within about $10^{-6}$ log-likelihood units of the exact optimum.

**Design choices and pitfalls.**
- The two failure modes give opposite errors. A negative eigenvalue gives variance 0 (overconfident); a tiny positive one gives a huge variance (uninformative). The report flags both, but only for callers that read it. `gate.recovery_dataset` ([`gate.py:204`](../../src/pdx/gate.py#L204)) and `scripts/run_hp_real.py` (`sp1_interval`, line 94) keep only the matrix (Notes 5).
- The condition number ignores non-positive eigenvalues, so read it together with `positive_definite`.
- The flat threshold is relative ($\lambda_{max}/10^{10}$, about $7\times10^{-7}$ in example 2), so it adapts to the scale of the likelihood.
- `localise.free_draws` draws the two phenotypes' raw drift entries from $N(\hat x,\hat V)$, restricted to those coordinates, and maps own rates through $-e^{r}$, so its draws respect the constraints. Directions with variance 0 are not explored at all.

### 3.13 `_treat` — the treatment terms of one participant

[`simulate.py:18`](../../src/pdx/engine/simulate.py#L18)

**What it does.** Computes the treatment terms $h_{ik}$ of the measurement equation, for one participant at every visit, in numpy. It is the same formula as the likelihood's `model.treat_effect`, which is written in PyTorch.

**Inputs → outputs.** `par` (the fit's `params`), `spec`, and the participant's `dose`, `expo` and `on` (each $(T_i,)$), phenotype `g` → $h$ $(T_i\times d)$: zero outside the treated domains, and all zero when `spec.n_treat` is 0.

**The mathematics.** For a treated domain $j$ at visit $k$,

$$
h_{ik,j}=-\Big[\kappa_{g,j}\,f_d(\mathrm{dose}_{ik})\,\mathrm{on}_{ik}+\kappa^{ldr}_{g,j}\,f_e(\mathrm{expo}_{ik})\Big],\qquad f_d(x)=\frac{x}{ED_{50}+x},\quad f_e(x)=\frac{x}{EC^{ldr}_{50}+x},
$$

for the Emax form, and $f_d(x)=f_e(x)=x$ for the linear form. The first term is the short-duration effect of the current dose, present only when the examination is in the ON state. The second is the long-duration effect of recent exposure. A positive $\kappa$ lowers the score, an improvement, since higher means worse. Doses are in units of 100 mg of levodopa-equivalent daily dose (LEDD), and $ED_{50}=EC^{ldr}_{50}=3$ by default.

*From the maths to the code.* Lines 19–21: zeros, returned at once if there is no treatment. Lines 22–23: $f_d$ and $f_e$. Line 24: the treated domains. Line 25: the formula, broadcast over visits and treated domains. The simulator needs $h$ in numpy, one participant at a time, while the likelihood computes it in PyTorch for a batch; the formula is copied rather than shared.

**Worked example** ([`ex_treat.py`](../guide_examples/s03/ex_treat.py)). Four visits with a rising dose, in the ON state at visits 2 and 4:

```python
dose = np.array([0.0, 3.0, 4.5, 6.0])                         # 100 mg LEDD at each visit
expo = np.array([0.0, 1.5, 3.0, 4.5])                         # long-duration exposure
on = np.array([0.0, 1.0, 0.0, 1.0])                           # examination in the ON state?
h = _treat(par, spec, dose, expo, on, g=0)
```

```text
h for a TD participant (visits x domains):
 [[-0.     -0.      0.      0.    ]
 [-0.3333 -0.3333  0.      0.    ]
 [-0.125  -0.125   0.      0.    ]
 [-0.4833 -0.4833  0.      0.    ]]
by hand, visit 2, domain 1: -(0.5 * 3/(3+3) * 1 + 0.25 * 1.5/(3+1.5)) = -0.3333
group 0: max |_treat - model.treat_effect| = 0.0e+00
group 1: max |_treat - model.treat_effect| = 0.0e+00
linear form, visit 2, domain 1: -1.8750 (= -(0.5 * 3 + 0.25 * 1.5))
```

Visit 3 is in the OFF state, so only the long-duration term remains: $-0.25\times3/6=-0.125$.

**Design choices and pitfalls.**
- Two copies of one formula must stay in step. The example shows that they agree today; no unit test checks it (Notes 3).
- The dose path is the observed one: the bootstrap conditions on it (§3.16).

### 3.14 `simulate_from_fit`, lines 29–47 — setup, speed and load

[`simulate.py:29`](../../src/pdx/engine/simulate.py#L29)

**What it does.** This is the simulator of H-P's conditional parametric bootstrap (protocol §7.3.1, step 2). For every participant it keeps the phenotype, the covariates, the landmark (year-1) scores, the visit dates, the dose path, the examination states and the pattern of missing values. It redraws everything else from the fitted model: the speed, the load, the true landmark state and the later visits. This first block prepares the data and draws the speed and the load.

**Inputs → outputs** (whole function). A `Fit` of any kind (the bootstrap passes the null fit), the observed `panel` and a numpy random generator → a new `Panel` that differs from the input only in the scores of visits 2 onward.

**The mathematics.** Given the fitted parameters, participant $i$ in phenotype $g$ draws

$$
u_i\sim N(0,1),\qquad s_i=\exp\big(\gamma_g+\beta^\top z_i+\theta^\top\tilde y_{i1}+\sigma_gu_i\big),
$$

$$
\ell_i=\Gamma^y_g\tilde y_{i1}+\Gamma^z_gz_i+\lambda_gu_i+\tau_g\odot\xi_i,\qquad\xi_i\sim N(0,I_d),
$$

and the era factor of the noise, $q_i=e^{\omega\,z_{i,\mathrm{era}}}$ (1 without an era column). These are the distributions the likelihood integrates over: the quadrature replaces $u_i$ by the nodes $u_m$, and the Kalman filter integrates $\ell_i$ given $u_m$. The shared $u_i$ in the speed and in the load mean is what makes speed and load correlated: $\mathrm{corr}(\log s_i,\ell_{ij}\mid z_i,\tilde y_{i1})=\lambda_{g,j}/\sqrt{\lambda_{g,j}^2+\tau_{g,j}^2}$, Sp-2's speed–load correlation (Chapter 5).

*From the maths to the code.*

| lines | step |
|---|---|
| 30–35 | spec and parameters; `prepare(panel)`, used only for $\mu_{1g}$ and $S_{1g}$ (the landmark prior, §3.15) and $\tilde y_{i1}$ (centred landmark scores, missing ones filled with the phenotype mean); $\Sigma_e$ and its square root |
| 37–41 | per participant: an interval cache, the phenotype, scores, times and $h$ (§3.13) |
| 42–44 | $u_i$, the linear predictor $\eta_i$, $s_i=e^{\eta_i+\sigma_gu_i}$ |
| 45–46 | the load mean at $u_i$, then $\ell_i$ |
| 47 | the era factor $q_i$ |

The code follows the model's hierarchy from the top: participant-level random effects first, then states, then measurements, so that one pass per participant produces a draw from the joint distribution. Using the likelihood's own `prepare` guarantees that $\tilde y_{i1}$, $\mu_{1g}$ and $S_{1g}$ are the values the likelihood uses. Since the landmark rows are kept, `prepare` on the simulated panel reproduces them exactly.

**Worked example** ([`ex_sim_participant.py`](../guide_examples/s03/ex_sim_participant.py), first part), from the fitted null of §3.7:

```python
eta = par["log_speed"][g] + panel.z[i] @ par["beta"] + y1c[i] @ par["theta"]
print("  speed: median %.3f, 10th-90th percentile %.3f to %.3f" % (np.exp(eta), *np.exp(eta + np.array([-1.2816, 1.2816]) * par["sigma"][g])))
mu_u0 = par["gam_y"][g] * y1c[i] + par["gam_z"][g] @ panel.z[i]
```

```text
participant 3, phenotype 0: eta = 0.3899, sigma_g = 0.7868
  speed: median 1.477, 10th-90th percentile 0.539 to 4.048
  load mean at u = 0: [ 0.0419  0.0299 -0.0392 -0.1697] | per unit of u: [ 0.0063 -0.0335 -0.0277 -0.0581] | spread tau_g: [0.     0.0838 0.1365 0.0466]
```

This TD participant's speed is drawn from a log-normal distribution with median $e^{0.39}=1.48$ and 10th–90th percentiles $e^{0.39\mp1.2816\times0.787}$. The load on tremor has no spread of its own (its fitted $\tau$ is zero, §3.12), so it varies only through $u_i$.

**Design choices and pitfalls.**
- The likelihood clamps each node's log-speed to ±15; the simulator does not. They differ only at speeds above $e^{15}$.
- The simulator works for any kind of fit, because `params` already holds each phenotype's drift and speed offset (§3.1). The bootstrap passes the null fit, and that is what makes it a test of H-P's null.

### 3.15 `simulate_from_fit`, lines 48–56 — the true landmark state

[`simulate.py:48`](../../src/pdx/engine/simulate.py#L48)

**What it does.** Draws the participant's true year-1 state from its posterior given the observed year-1 scores, under the same plug-in prior that the likelihood uses.

**Inputs → outputs.** The landmark scores $y_{i1}$ (with any missing values), $h_{i1}$, the phenotype's prior $N(\mu_{1g},S_{1g})$ and $\Sigma_e$ → one draw of $X_{i1}$.

**The mathematics.** The prior of the true landmark state is $X_{i1}\sim N(\mu_{1g},S_{1g})$: the mean and covariance of the phenotype's landmark scores, minus $\Sigma_e$, floored to be positive definite (Chapter 2). Let $o$ be the observed components and $H_o$ the rows of $I_d$ that select them. The measurement is $y_{i1,o}=H_oX_{i1}+h_{i1,o}+e_o$, and the Gaussian update gives

$$
K=S_{1g}H_o^\top\big(H_oS_{1g}H_o^\top+\Sigma_{e,oo}\big)^{-1},\qquad m=\mu_{1g}+K\big(y_{i1,o}-H_o\mu_{1g}-h_{i1,o}\big),\qquad P=S_{1g}-KH_oS_{1g}.
$$

The draw is $X_{i1}=m+V\Lambda_+^{1/2}\xi$, where $\tfrac12(P+P^\top)=V\Lambda V^\top$ and negative eigenvalues are set to 0, so a covariance that is singular, or slightly indefinite through rounding, still works. The likelihood does exactly this update (`_landmark_update`, in masked form) and conditions on $y_{i1}$, which contributes no likelihood term. The landmark scores themselves are copied into the simulated panel unchanged.

*From the maths to the code.* Line 48: the observed mask. Line 49: copies of the prior. Lines 50–54: the update on the observed components only, skipped if the whole landmark visit is missing. Lines 55–56: the draw through the eigendecomposition. The likelihood's model for visits 2 onward starts from $X_{i1}\mid y_{i1}$. Simulating from the same posterior makes the simulated data's distribution, given the kept landmark scores, the one the likelihood describes. That is what "conditional" means in "conditional parametric bootstrap".

**Worked example** ([`ex_sim_participant.py`](../guide_examples/s03/ex_sim_participant.py), second part). A participant whose axial score is missing at the landmark. The posterior is computed by replaying lines 48–54 and compared with the likelihood's `_landmark_update`.

```python
m, P = mu1[g].copy(), S1[g].copy()                       # replay of simulate.py lines 48-54
Ho = np.eye(spec.d)[o]
Kg = P @ Ho.T @ np.linalg.inv(Ho @ P @ Ho.T + R[np.ix_(o, o)])
m = m + Kg @ (yi[0, o] - m[o] - h[0, o])
P = P - Kg @ Ho @ P
```

```text
  landmark scores: [0.224     nan 0.3387 0.2588] (observed: [ True False  True  True] )
  posterior mean of the true landmark state: [0.2395 0.093  0.3232 0.24  ]
  posterior sd: [0.1353 0.3408 0.1378 0.1357] | prior sd: [0.4717 0.3586 0.6321 0.5012]
  against the likelihood's _landmark_update: mean 5.6e-17, covariance 2.8e-17
```

- The observed domains' posterior SD (about 0.135) is close to the measurement-error SD, $\sqrt{0.02}=0.141$, shrunk a little by the prior.
- The missing axial score is predicted from the correlated domains, which narrows its spread only slightly (0.341 against a prior 0.359).
- The simulator and the likelihood compute the same posterior.

**Design choices and pitfalls.**
- $X_{i1}$ is drawn independently of $u_i$ and $\ell_i$, as in the likelihood, whose filter starts with zero covariance between the state and the load.
- The prior moments are plug-ins from the observed landmark scores, not parameters, so the bootstrap does not propagate their uncertainty. They are identical in every replicate because the landmark rows are kept.
- The protocol's "each participant keeps their … year-1 state" is implemented as: the year-1 *scores* are kept, and the true year-1 state is redrawn given them.

### 3.16 `simulate_from_fit`, lines 57–73 — exact transitions and the measurement equation

[`simulate.py:57`](../../src/pdx/engine/simulate.py#L57)

**What it does.** Propagates the true state exactly over each actual interval between visits, adds the treatment terms and measurement error, and puts NaN wherever the observed score was missing.

**Inputs → outputs.** $X_{i1}$, $s_i$, $\ell_i$, $q_i$, the visit times, $h_i$ and the observed missing pattern → the participant's simulated scores, with the landmark row as observed.

**The mathematics.** *Exact discretisation.* Change to the reference clock $\tilde t=s_it$. On it, $\sqrt{s_i}\,dW(t)$ has variance $s_i\,dt=d\tilde t$, so $\tilde W(\tilde t)=\sqrt{s_i}\,W(\tilde t/s_i)$ is a standard Brownian motion and

$$
dX=\big[AX+b+\ell\big]\,d\tilde t+G\,d\tilde W .
$$

A real interval $\Delta t$ is $c=s_i\Delta t$ on that clock, and the linear equation has the exact solution

$$
X_k=E\,X_{k-1}+M\,(b+\ell)+\eta_k,\qquad E=e^{cA},\quad M=\int_0^ce^{Aw}\,dw,\quad\eta_k\sim N\big(0,\ q_i\,S\big),\quad S=\int_0^ce^{Aw}Qe^{A^\top w}\,dw,
$$

with no discretisation error, however long the interval. The era factor multiplies $Q$, and $S$ is linear in $Q$, so it multiplies $S$.

*How `transition_np` computes them* ([`transitions.py:181`](../../src/pdx/engine/transitions.py#L181); Chapter 2 covers the batched versions the likelihood uses):
- $E$ and $M$ come from one block exponential, $\exp\begin{pmatrix}cA&cI\\0&0\end{pmatrix}=\begin{pmatrix}E&M\\0&I\end{pmatrix}$ (§3.4).
- $S$ comes from scaling and squaring. Differentiating $e^{Aw}Qe^{A^\top w}$ gives $e^{Aw}\mathcal L(Q)e^{A^\top w}$, so $e^{Aw}Qe^{A^\top w}=\sum_j\frac{w^j}{j!}\mathcal L^j(Q)$, and integrating,

  $$
  S(a)=\sum_{j=0}^{12}\frac{a^{j+1}}{(j+1)!}\,\mathcal L^j(Q)\quad\text{for the short interval }a=c/2^{k},\qquad k=\max\Big(0,\Big\lceil\log_2\max\big(\tfrac{c\,\|A\|_1}{0.25},1\big)\Big\rceil\Big).
  $$

  Then $k$ doublings follow: $S(2a)=S(a)+E(a)S(a)E(a)^\top$ and $E(2a)=E(a)^2$, the composition rule obtained by splitting the integral at $a$. With $a\|A\|_1\le0.25$ the omitted terms are about $10^{-15}$ of the first, and the rule needs no stability of $A$.

*Measurement.* $Y_{ik}=X_k+h_{ik}+\Sigma_e^{1/2}\epsilon_k$ with $\epsilon_k\sim N(0,I_d)$; then NaN wherever the observed $Y_{ik}$ was NaN.

*From the maths to the code.*

| lines | step |
|---|---|
| 57–58 | an output array of NaN; the landmark row copied as observed |
| 59 | the fitted drift, noise and load of phenotype $g$ (for the null fit: the shared $A$ at the reference speed) |
| 60–65 | for each later visit: $\Delta t$; $(E,M,S)$ from `transition_np(A, q_i Q, s_i, Δt)`, cached within the participant by $\Delta t$ rounded to $10^{-12}$ |
| 66–67 | $X\leftarrow EX+M(b+\ell)+V_S\Lambda_{S,+}^{1/2}\xi$ |
| 68–71 | the measurement; the observed missing pattern imposed |
| 72–73 | collect; `panel.with_y` keeps everything else |

The bootstrap must draw from the very model whose likelihood is maximised; otherwise its null distribution would describe a different model. An Euler scheme would add discretisation bias, and the exact transition removes it. The cache exploits the fact that a participant's visit grid repeats the same intervals. The speed differs between participants, so the cache cannot be shared between them.

**Worked example** ([`ex_simulate.py`](../guide_examples/s03/ex_simulate.py)), from the fitted null of §3.7:

```python
sim = simulate_from_fit(f0, panel, np.random.default_rng(0))
E, M, S = transition_np(A, Q, 1.3, 0.9)
E2, M2, S2 = _van_loan(A, Q, 1.3 * 0.9)
sims = [simulate_from_fit(f0, panel, np.random.default_rng([7, r])) for r in range(R)]
G = np.array([-value_and_grad(f0.x, f0.layout, prepare(p, spec.meas_var, spec.n_groups))[1] for p in sims])
```

```text
kept: landmark rows True, visit times True, missing pattern True, groups True, covariates True, doses True
redrawn: later visits equal to the observed ones: 0 of 1333
transition_np vs reference (speed 1.3, interval 0.9 y): max differences E 0.0e+00, M 0.0e+00, S 2.8e-17
TD   yearly change, observed      : [-0.001  0.046  0.07   0.021]
TD   yearly change, simulated mean: [0.004 0.035 0.055 0.009]  (sd over datasets [0.019 0.018 0.019 0.02 ] )
PIGD yearly change, observed      : [0.059 0.007 0.017 0.061]
PIGD yearly change, simulated mean: [0.072 0.011 0.03  0.044]  (sd over datasets [0.025 0.018 0.024 0.026] )
score check over 100 datasets, 109 coordinates (of 112; the rest have no variation): mean z^2 = 0.97, share |z| > 1.96 = 0.064, max |z| = 2.33
observed data at the fit: max |gradient| = 1.90e-02
0.04 s per simulated dataset, 0.07 s per gradient
```

Reading it:
- Everything the protocol says to keep is kept, and every later observed score is redrawn.
- One interval's transition agrees to rounding with the independent reference implementation (`reference._van_loan`: Van Loan's blocks and scipy's Lyapunov solver).
- The mean yearly change from the landmark to the last visit lies within about one simulation SD of the observed one in every domain and phenotype.
- **The score check.** For data simulated from parameters $x$, the expected score (the gradient of the log-likelihood) at $x$ is zero: $\mathbb E_x[\nabla\ell(x)]=\int\nabla p_x=\nabla\int p_x=0$. Over 100 simulated datasets, the mean gradient at the fitted null has $z$-scores with a mean square of 0.97 (1 expected) and 6.4% beyond ±1.96 (5% expected). The simulator and the likelihood therefore describe the same model, conditional on the landmark. The three coordinates without variation are the load spreads fitted at zero, whose gradient is zero in every dataset.

**Design choices and pitfalls.**
- The missing pattern, the visit times and the dose path are imposed, not regenerated. The bootstrap is conditional on them, which matches a likelihood that treats doses as fixed covariates and missingness as ignorable (missing at random). Dropout or treatment that depends on the simulated trajectory is not reproduced.
- Measurement noise is drawn for every component, missing ones included, so the random stream does not depend on the missing pattern.
- The noise draw uses an eigendecomposition with negative eigenvalues set to 0, so a rounding-level indefinite $S$ cannot make it fail.
- The simulator's $S$ comes from composition; the likelihood takes the stationary-covariance route when $A$ is clearly stable. Both are exact, and they agree to rounding (the example above; `test_both_noise_routes_agree`).
- The cost is small: 0.04 s per dataset of 75 participants, against 0.07 s for one gradient evaluation. The bootstrap's cost lies in the refits.
- No unit test checks the simulator's distribution. The score check above would make one (Notes 3).

### 3.17 Notes for the author

Ordered by likely impact. Each was reproduced on synthetic data, except where marked as a check.

1. **A start at which the likelihood cannot be evaluated comes back as a converged fit** ([`fit.py:167`](../../src/pdx/engine/fit.py#L167), with [`likelihood.py:227`](../../src/pdx/engine/likelihood.py#L227)).
   - *What:* `value_and_grad` returns $(10^{10},\mathbf 0)$ at a point it cannot evaluate. If that point is a start, L-BFGS-B sees a zero projected gradient and stops after one evaluation, with `CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL` and `success = True`.
   - *Reproduction* ([`repro_failed_start.py`](../guide_examples/s03/repro_failed_start.py)): a null start with $\log\tau=400$ for every load spread gives `null fit: loglik -1e+10, success True, 1 evaluation(s)`. The free fit's own start then reaches −527.186, so `lr_stat` = 2e+10, and `HPResult.summary` would show `null_converged = True`.
   - *Why it matters:* on the observed data this would read as an overwhelming rejection with both fits "converged". In a bootstrap replicate it counts as an exceedance (conservative). In a profile (`gate.profile`), a grid point that cannot be evaluated would look like a drop in likelihood and could mark a parameter "identified". A default start is rarely non-evaluable, but nothing would flag it if it were. A guard would: treat a final value of $10^{10}$, or zero iterations, as a failure.
2. **The null fit has a single start, and it starts differently in the observed data and in the replicates** ([`fit.py:193`](../../src/pdx/engine/fit.py#L193), [`hp.py:74`](../../src/pdx/hp.py#L74); a check).
   - The observed null fit starts at `start_values`, whose speed can be far off (0.764 against 2 in §3.5). A replicate's null fit starts at the observed null estimate, which is the replicate's truth.
   - If the observed null fit stopped in a worse optimum than replicates typically do, the observed statistic would be inflated relative to the bootstrap distribution, making the test anti-conservative. In §3.7 two null starts ended 0.065 units apart: within the gate's tolerance of 0.1, but 0.13 in units of the statistic.
   - The engine gate's dispersed-start check is the safeguard. Its null spreads at the real size are worth reading before WP4. A few dispersed starts for the observed null fit alone would be cheap next to the bootstrap.
3. **The simulator has no test of its distribution, and other branches are untested** (a check of `tests/`).
   - `simulate_from_fit` runs only inside `hp`: in the slow test `test_bootstrap_replicates_agree_however_many_run_at_once` and in `scripts/smoke_test.py`. Neither checks what it draws. `docs/code_map.md` mentions a reviewer's score check over 300 datasets; it is not in `tests/`. The score check of §3.16 (mean $z^2=0.97$ over 100 datasets, about 35 s) would make a cheap unit test, as would the agreement of `_treat` with `model.treat_effect` (§3.13). This simulator generates H-P's reference distribution, so it deserves a test.
   - No test exercises `_yearly_ols`'s fallback (fewer than $d+5$ pairs), `_drift_from_yearly`'s fallback, the `RuntimeError` of `hessian`, or the non-positive-definite branch of `covariance_checked`.
4. **The nesting of `null_to` is exact only above two floors and inside the log-speed clamp** ([`fit.py:129`](../../src/pdx/engine/fit.py#L129)).
   - *What:* `pack_drift` floors $-\rho_gA_{jj}$ at $10^{-4}$ ([`model.py:223`](../../src/pdx/engine/model.py#L223)), and `pack_chol` floors the eigenvalues of $\rho_gQ_g$ at $10^{-6}$ ([`model.py:230`](../../src/pdx/engine/model.py#L230)). The likelihood's ±15 clamp on the node log-speeds ([`likelihood.py:140`](../../src/pdx/engine/likelihood.py#L140)) acts on different quantities in the two layouts.
   - *Reproduction* ([`repro_null_to_limits.py`](../guide_examples/s03/repro_null_to_limits.py)), free minus null log-likelihood at the mapped point:

     ```text
     noise eigenvalue 1.3e-08 (x 1.8 = 2.4e-08 < 1e-6): free minus null log-likelihood +6.38e-03
     own rate -4.5e-05 (x 1.8 = -8.2e-05, floored at 1e-4): free minus null +9.85e-04
     sigma = 5, so the outer nodes reach the +-15 clamp: free minus null +1.92e-08
     unchanged vector: free minus null +0.0e+00
     ```

   - *Why it matters:* "LR ≥ 0 by construction" then holds only approximately, and the gap can have either sign. A noise variance collapsing towards zero is plausible when a domain's fixed measurement-error variance (D28) exceeds its residual variance. The free optimiser will usually recover, and `lr_stat` clamps at 0, so the effect is small. Writing the raw values directly would make the map exact at any value: own-rate raw $+\gamma_g$, links $\times\rho_g$, Cholesky log-diagonal $+\gamma_g/2$, Cholesky off-diagonal $\times\rho_g^{1/2}$.
5. **Two callers drop the covariance report** ([`gate.py:204`](../../src/pdx/gate.py#L204), [`scripts/run_hp_real.py:94`](../../scripts/run_hp_real.py#L94)).
   - `covariance_checked` gives variance 0 along negative-information directions, and turns tiny positive eigenvalues into huge variances (§3.12, both examples).
   - `gate.recovery_dataset` computes the coverage of log-speed and $\log\sigma$ from the matrix alone, and so does `run_hp_real.sp1_interval` for the model-based Sp-1 interval. If the information were not positive definite along a direction involving $\gamma$ or $\theta$, an interval could be too narrow with nothing in the output to say so. Recording `positive_definite` next to those intervals, or setting flagged SEs to NaN, would close the gap. (The protocol's Sp-1 interval is the bootstrap one; the model-based interval belongs to the first look.)
6. **Protocol §7.1 says that second derivatives come from automatic differentiation; the code differences exact gradients** ([`fit.py:199`](../../src/pdx/engine/fit.py#L199); a documentation check). The function's docstring is accurate; the protocol's sentence ("Exact gradients and second derivatives are taken through the matrix exponential by automatic differentiation") is not. The differences are accurate (§3.9: asymmetry $10^{-8}$ and step sensitivity $4\times10^{-6}$, relative to the largest entry), so the fix is to the protocol text, unless exact second derivatives are wanted.
7. **`Fit` bookkeeping that is easy to misread** ([`fit.py:179`](../../src/pdx/engine/fit.py#L179)). `n_eval` sums over all starts. `success` and `message` belong to the kept start only. With a `penalty`, `loglik` is minus the penalised objective; no caller passes one today.
8. **`covariance()` is never called** ([`fit.py:217`](../../src/pdx/engine/fit.py#L217)). It is dead code, and deleting it removes one way to lose the report.
9. **Small interface edges** ([`fit.py:159`](../../src/pdx/engine/fit.py#L159), [`fit.py:193`](../../src/pdx/engine/fit.py#L193)). `fit_pair` passes `**kw` to both fits, so `bounds` (the wrong length for one of the two layouts; from reading the code), `x0` or `extra_starts` (a duplicate keyword) cannot be passed through it. [`repro_interface_edges.py`](../guide_examples/s03/repro_interface_edges.py) shows three edges, each raising before any fitting:

   ```text
   fit_model with no start: AttributeError: 'NoneType' object has no attribute 'x'
   fit_pair with x0 in **kw: TypeError: pdx.engine.fit.fit_model() got multiple values for keyword argument 'x0'
   null_to(..., 'null'): KeyError: 'A1'
   ```
10. **`_drift_from_yearly` keeps the real part of a complex logarithm** ([`fit.py:58`](../../src/pdx/engine/fit.py#L58)). For a yearly matrix with a negative eigenvalue the start drift does not reproduce it (§3.3: $e^{-1.609}=+0.2$ for an eigenvalue of −0.2). Start values only; listed for completeness.
11. **Notation for the shared table** (for the guide's editor). $\Gamma^y_g$ is diagonal in the code: `gam_y` is $(G,d)$ and multiplies $\tilde y_{i1}$ element by element ([`likelihood.py:143`](../../src/pdx/engine/likelihood.py#L143), [`simulate.py:45`](../../src/pdx/engine/simulate.py#L45)). So "$\Gamma^y_g\tilde y_{i1}$" means $\mathrm{diag}(\gamma^y_g)\,\tilde y_{i1}$, while $\Gamma^z_g$ is a full $d\times p$ matrix. The map (§0.4) uses $\tau$ both for an interval and for the load spread; this chapter writes $\Delta t$ for intervals.

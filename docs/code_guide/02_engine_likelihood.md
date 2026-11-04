## 2. The engine I: model and likelihood

*Chapter 2 of the guide (the map is [§0](00_map.md)). Written on 4 November 2026 against commit `a9b577f`; the line numbers in the links are that commit's.*

This chapter reads the five files that turn a vector of parameters and a panel of visits into one number, the log-likelihood of the continuous-time model of protocol §2.3 and §7.1.

| file | lines | what it holds |
|---|---|---|
| [`engine/model.py`](../../src/pdx/engine/model.py) | 233 | what a model is made of (`ModelSpec`); the map from the flat parameter vector to named blocks (`Layout`); the transforms from raw numbers to drift, speed, noise and treatment terms |
| [`engine/linalg.py`](../../src/pdx/engine/linalg.py) | 24 | one helper, `bmm`, that splits batched products of small matrices for speed |
| [`engine/transitions.py`](../../src/pdx/engine/transitions.py) | 200 | the exact transition matrix $E$, intercept $M$ and noise covariance $S$ over an interval |
| [`engine/likelihood.py`](../../src/pdx/engine/likelihood.py) | 231 | the data tensors (`prepare`); the Kalman filter with the participant load in its state; the Gauss–Hermite quadrature over speed; `loglik` and `value_and_grad` |
| [`engine/reference.py`](../../src/pdx/engine/reference.py) | 100 | an independent, slow implementation of the same likelihood, used only for checks |

Fitting (`fit.py`) and simulation from a fit (`simulate.py`) are in §3.

**How the examples were run.** Every example runs the repository's own functions on synthetic data. The panels come from simulator family A, scenario S1 (PIGD runs the shared mechanism at twice TD's speed), through the unit tests' fixtures (`tests/helpers.py`). Family A has presets for 4, 6 and 10 domains, so the small examples use $d=4$ (tremor, axial, function, independence). Some examples use hand-made matrices instead. The scripts are in [`docs/guide_examples/s02/`](../guide_examples/s02/). Their shared [`_setup.py`](../guide_examples/s02/_setup.py) puts `src/` and `tests/` on the import path and imports `pdx.engine.model` first (§2.1 explains why). No script reads `data/` or `results/`. Run one from the repository folder:

```
OMP_NUM_THREADS=1 ../ppmi-causal/.venv/bin/python docs/guide_examples/s02/ex_overview.py
```

The environment was Python 3.13.12, numpy 2.5.4, scipy 1.18.1 and torch 2.14.1, single-threaded, on the M4.

---

### 2.0 Overview

**The model as equations.** Participant $i$ belongs to phenotype $g\in\{0,1\}$ (0 = TD, the reference; 1 = PIGD). Their true domain levels $X_i(t)\in\mathbb R^d$ follow a linear stochastic differential equation (SDE):

$$
dX_i(t)=s_i\,\big[A_g X_i(t)+b_g+\ell_i\big]\,dt+\sqrt{s_i\,e^{\omega\,\mathrm{era}_i}}\;G_g\,dW_i(t),
$$

where $W_i$ is a $d$-dimensional Brownian motion. The ingredients are:

- **Speed.** $\log s_i=\gamma_g+\beta^\top z_i+\theta^\top\tilde y_{i1}+\sigma_g u_i$, with $u_i\sim N(0,1)$. Speed multiplies the drift, the load and the noise variance, so a faster participant moves along the same paths in less time.
- **Load.** $\ell_i\mid u_i\sim N\big(\Gamma^y_g\tilde y_{i1}+\Gamma^z_g z_i+\lambda_g u_i,\ \mathrm{diag}(\tau_g^2)\big)$, with $\Gamma^y_g=\mathrm{diag}(\gamma^y_g)$ in the code (a per-domain coefficient). The load is a steady push on each domain. Its mean moves with the speed's random part $u_i$, so speed and load can correlate.
- **Noise.** $Q_g=G_gG_g^\top$ per unit time at the reference speed, multiplied by $e^{\omega\,\mathrm{era}_i}$.
- **Measurement.** At visit $k$, $Y_{ik}=X_i(t_{ik})+h_{ik}+e_{ik}$, with $e_{ik}\sim N(0,\Sigma_e)$, $\Sigma_e=\mathrm{diag}(\sigma^2_{e,1},\dots,\sigma^2_{e,d})$ fixed (D28).
- **Treatment** (protocol §6.1, D24). For a treated domain $j$,
  $$h_{ik,j}=-\big[\kappa_{g,j}\,f_D(D_{ik})\,\mathrm{on}_{ik}+\kappa^{\mathrm{LDR}}_{g,j}\,f_E(\mathcal E_{ik})\big],\qquad f_D(x)=\frac{x}{\mathrm{ED}_{50}+x},\ f_E(x)=\frac{x}{\mathrm{EC}_{50}+x}\ \text{(Emax) or } f(x)=x\ \text{(linear)},$$
  with $D_{ik}$ the current dose (100 mg levodopa-equivalent daily dose, LEDD), $\mathcal E_{ik}$ the long-duration exposure and $\mathrm{on}_{ik}=1$ for an examination in the ON state. $h_{ik,j}=0$ for untreated domains. Neither term is scaled by speed.
- **Landmark.** The likelihood is conditional on the year-1 (landmark) scores $Y_{i1}$. The true landmark state has a plug-in prior per phenotype, $X_i(t_{i1})\sim N(\mu_{1g},S_{1g})$, computed from the data (§2.27).

**Three kinds of model** share one parameter layout:

| kind | drift of TD | drift of PIGD | speed offset $\gamma_1$ | used for |
|---|---|---|---|---|
| `null` | $A$ | $A$ | estimated ($\gamma_0=0$) | H-P's null, the speed-only model; Sp-1 and Sp-2 |
| `free` | $A_0$ | $A_1$ | none ($\gamma\equiv 0$, absorbed in $A_1$) | H-P's alternative; Sp-3 |
| `joint` | $A$ | $A+D_1$ | estimated | localisation (protocol §7.3.2) |

**Parameter transforms.** The optimiser works on an unconstrained vector. Constraints come from transforms:
- own rates $A_{jj}=-e^{a_{jj}}<0$ (for $A$ and the free model's $A_g$; not for the joint model's deviations $D_g$);
- $G_g$ lower-triangular with diagonal $e^{l_{jj}}>0$, so $Q_g=G_gG_g^\top$ is positive definite;
- $\sigma_g=e^{\log\sigma_g}$, $\tau_{g,j}=e^{\log\tau_{g,j}}$.

Everything else ($\gamma$, $\beta$, $\theta$, $b_g$, $\Gamma^y_g$, $\Gamma^z_g$, $\lambda_g$, $\omega$, $\kappa$, $\kappa^{\mathrm{LDR}}$, the off-diagonal drift entries) enters as it is.

**From the SDE to one interval** (derived in §2.13). Given the speed $s$ and the load $\ell$, the state at the next visit, a time $\tau$ later, is Gaussian:

$$
X(t+\tau)\mid X(t)\ \sim\ N\big(E\,X(t)+M\,(b_g+\ell),\ e^{\omega\,\mathrm{era}_i}S\big),\qquad
E=e^{cA_g},\quad M=\int_0^{c}e^{A_gw}dw,\quad S=\int_0^{c}e^{A_gw}Q_ge^{A_g^\top w}dw,
$$

and all three depend on the speed and the interval only through $c=s\,\tau$. A participant twice as fast covers a year's change in half a year.

**The likelihood.** Given $u_i$, the load enters the transitions linearly. Stacking it into the state, $[X;\ell]$, makes the model linear and Gaussian, so a Kalman filter integrates out the load, the measurement error and any missing scores exactly. Only the one-dimensional integral over $u_i$ remains, done by Gauss–Hermite quadrature with $K=12$ nodes:

$$
\log L(x)=\sum_i\log\int p\big(Y_{i,2:T_i}\mid Y_{i1},u;x\big)\,\phi(u)\,du\ \approx\ \sum_i\log\sum_{m=1}^{K}w_m\,p\big(Y_{i,2:T_i}\mid Y_{i1},u_m;x\big).
$$

**How the files fit together.**

```text
Panel (data.py) ──prepare()──▶ Prepared: Y, O, dt, dose, expo, on, z, y1c, g, mu1, S1, R, Ti     (§2.24, §2.27)
                                     │
x (flat numpy vector) ──Layout──┐    │                                                              (§2.3)
                                ▼    ▼
            participant_loglik(x, lay, prep, idx)     one batch ("chunk") of participants           (§2.31–2.35)
              ├─ model: drifts, noise_cov, log_speeds, treat_effect      raw numbers → A_g, Q_g, γ_g, h (§2.4–2.8)
              ├─ gh_nodes                                                 nodes u_m and log-weights   (§2.25)
              ├─ _landmark_update                                         posterior of X(t_1)         (§2.30)
              ├─ transitions.transitions_grouped                          E, M, S per node × interval (§2.20)
              │    ├─ _grid ── _taylor, _compose (binary powers) ── linalg.bmm                       (§2.12–2.17)
              │    └─ lyapunov, is_clearly_stable (stationary route for S)                            (§2.18–2.19)
              └─ Kalman loop over visits on [X; ℓ] ── linalg.bmm
            loglik()          sum over chunks, one autograd graph (tests, small data)               (§2.36)
            value_and_grad()  −sum and its gradient, chunk by chunk ──▶ fit.py (L-BFGS-B, §3)      (§2.37)
reference.reference_loglik()  the same number by loops, scipy's expm and Lyapunov solver            (§2.38–2.39)
```

**One participant's likelihood, from start to end.**

1. **Data in.** `prepare` pads the participant's visits into fixed-size tensors: scores $Y_{ik}$ with missing values set to 0, an observed mask $O_{ik}$, intervals $\tau_{ik}=t_{i,k+1}-t_{ik}$, dose, exposure and ON state. Per phenotype, it computes the plug-in prior $N(\mu_{1g},S_{1g})$ of the true landmark state. It also computes the centred landmark scores $\tilde y_{i1}$ (§2.27).
2. **Parameters.** `Layout` cuts the flat vector into named blocks. `drifts`, `noise_cov` and `log_speeds` turn them into $A_g$, $Q_g$ and $\gamma_g$; exponentials give $\sigma_g$ and $\tau_g$ (§2.3–2.7).
3. **Speed at the nodes.** $\eta_i=\gamma_g+\beta^\top z_i+\theta^\top\tilde y_{i1}$, and for each node $s_{im}=\exp\{\mathrm{clamp}(\eta_i+\sigma_gu_m,-15,15)\}$ (§2.31).
4. **Load prior at each node.** $\ell_i\mid u_m\sim N(\mu_{\ell i}+\lambda_gu_m,\mathrm{diag}\,\tau_g^2)$, with $\mu_{\ell i}=\gamma^y_g\odot\tilde y_{i1}+\Gamma^z_gz_i$ (§2.31).
5. **Treatment terms** $h_{ik}$ from dose, exposure and the ON state (§2.8).
6. **Landmark.** The posterior $N(m_1,P_1)$ of $X_i(t_{i1})$ given the observed components of $Y_{i1}-h_{i1}$. It adds nothing to the likelihood, which is conditional on $Y_{i1}$ (§2.30).
7. **Initial filter state at each node.** $[X;\ell]\sim N\big([m_1;\ \mu_{\ell i}+\lambda_gu_m],\ \mathrm{blockdiag}(P_1,\mathrm{diag}\,\tau_g^2)\big)$ (§2.32).
8. **Transitions.** For every node and interval, $c=s_{im}\tau_{ik}$ gives $E$, $M$ and $S$; $S$ is then multiplied by $e^{\omega\,\mathrm{era}_i}$ (§2.17–2.20, §2.32).
9. **Predict** from visit $k-1$ to visit $k$: $X\leftarrow EX+M(b_g+\ell)$, $\ell$ unchanged, and the covariance blocks follow (§2.33).
10. **Update** with the observed components of $Y_{ik}-h_{ik}$. The innovation $v$ has covariance $P_{xx}+\Sigma_e$ on the observed components. $\log N(v;0,\cdot)$ is added to the node's log-likelihood $\ell\ell_{im}$, and the mean and covariance of $[X;\ell]$ are updated. A fully missing visit adds nothing (§2.34).
11. **Quadrature.** $\log L_i=\log\sum_mw_m\exp(\ell\ell_{im})$, computed by log-sum-exp (§2.35).
12. **Sum and differentiate.** $\log L=\sum_i\log L_i$ over batches of participants with similar numbers of visits. `value_and_grad` returns $-\log L$ and its gradient by automatic differentiation, batch by batch (§2.36–2.37).

The worked example of §2.31–2.35 follows these twelve steps for one synthetic participant and prints every intermediate quantity.

**Notation: symbol ↔ code.** The shared notation of [§0.4](00_map.md#04-shared-notation) maps onto the code as follows. Every row was checked against `model.py` and `likelihood.py`.

| symbol | code | exactly, in the code |
|---|---|---|
| $i$; $g$; $G$ | position in the `prep` tensors; `prep.g` (from `panel.group`); `spec.n_groups` | $g\in\{0,\dots,G-1\}$, $G=2$; 0 is the reference |
| $d$ | `spec.d` | |
| $t_{ik}$; $\tau_{ik}$ | `panel.t[i][k]`; `prep.dt[i, k]` | only differences enter, $\tau_{ik}=t_{i,k+1}-t_{ik}$, so the time origin is irrelevant (family A's landmark sits near $t=1$) |
| $Y_{ik}$ | `panel.y[i][k]`; `prep.Y`, `prep.O` | `prep.Y` holds 0 where a score is missing; `prep.O` is 1 where observed |
| $A$ | block `A` | $n_{sup}$ raw values in `spec.sup_idx` order; diagonal entries are $-e^{\text{raw}}$ |
| $A_g$ (free, $g\ge1$) | block `A{g}` | the same transform |
| $D_g$ (joint) | block `D{g}` | raw values; **no** transform on the diagonal |
| drift used for $g$ | `drifts(x, lay)[g]` | at the reference speed; shape $(G,d,d)$ |
| $\gamma_g$ | block `gamma`; `log_speeds(x, lay)` | $\gamma_0=0$; in the free model $\gamma\equiv0$ |
| $\beta$; $\theta$ | `beta` $(p)$; `theta` $(d)$ | shared by the phenotypes; `theta` exists only if `speed_on_y1` |
| $\sigma_g$ | `exp(log_sigma[g])` | |
| $z_i$ | `panel.z[i]` = `prep.z[i]` | centred by the caller; the engine does not centre $z$ |
| $\tilde y_{i1}$ | `prep.y1c[i]` | the **observed** landmark scores, a missing one replaced by its phenotype mean $\mu_{1g}$, minus the pooled mean |
| $u_m$; $w_m$ | `gh_nodes(K)` returns $(u_m,\ \log w_m)$ | $u_m=\sqrt2x_m$, $w_m=w^{GH}_m/\sqrt\pi$, $\sum_mw_m=1$ |
| $b_g$ | `b[g]` | |
| $\Gamma^y_g$ | `gam_y[g]`, shape $(d)$ | **diagonal**: $(\Gamma^y_g\tilde y)_j=\gamma^y_{g,j}\tilde y_j$ |
| $\Gamma^z_g$ | `gam_z[g]`, shape $(d,p)$ | a full matrix |
| $\lambda_g$; $\tau_g$ | `lam[g]`; `exp(log_tau[g])` | vectors of length $d$ |
| $G_g$; $Q_g$ | block `L[g]`; `noise_cov(x, lay)[g]` | `L[g]` holds the lower triangle row by row, with logs on the diagonal; $Q_g=G_gG_g^\top$ |
| $\omega$; $\mathrm{era}_i$ | `omega`; `z[i, spec.noise_era_col]` | era is the **centred** era column, so $Q_g$ is the noise at the mean era |
| $h_{ik}$ | `treat_effect(...)[i, k]` | |
| $\kappa_g$; $\kappa^{\mathrm{LDR}}_g$ | `kappa[g]`; `kappa_ldr[g]` | one value per treated domain |
| $\mathrm{ED}_{50}$; $\mathrm{EC}_{50}$ | `spec.ed50`; `spec.ec50_ldr` | fixed constants (3.0), not estimated |
| $\Sigma_e$ | `diag(spec.meas_var)` = `diag(prep.R)` | fixed |
| $\mu_{1g}$; $S_{1g}$ | `prep.mu1[g]`; `prep.S1[g]` | plug-in moments, not parameters |
| $E,M,S$ | `transitions_grouped(AG, QG, g, s * dt)` | at the reference-speed drift with $c=s\tau$; $S$ before the era factor |
| $K$ | `spec.n_quad` | 12 |

Four places where the code is narrower than, or differs from, the §0.4 table (also in §2.41):
- $\Gamma^y_g$ is diagonal;
- own rates are kept negative for $A$ and the free model's $A_g$, not in the joint model's $A+D_g$;
- $\tilde y_{i1}$ is the observed landmark score with missing values filled, not the true year-1 state $x_i(1)$ of protocol §2.3;
- $\mathrm{era}_i$ is the centred era column.

**Worked example** ([`ex_overview.py`](../guide_examples/s02/ex_overview.py)). The engine's log-likelihood for the three kinds of model on a 60-participant synthetic panel, at a perturbed start value, against the independent reference:

```python
panel, info = s1_panel(n=60, seed=1, d=4)
spec = spec_for(panel, info)            # Emax treatment, era shifts the noise, 12 nodes
prep = prepare(panel, spec.meas_var, spec.n_groups)
for kind in ("null", "free", "joint"):
    lay = layout(spec, kind)
    x = random_point(panel, spec, kind, seed=7)
    fast = float(loglik(torch.as_tensor(x), lay, prep))
    ref = reference_loglik(x, lay, panel, prep.mu1.numpy(), prep.S1.numpy(), prep.y1c.numpy())
```

```text
synthetic S1 panel, domains: ['tremor', 'axial', 'function', 'independence']
  phenotype 0: 27 participants, 144 visits, 117 transitions, 0.041 of scores missing
  phenotype 1: 27 participants, 126 visits, 99 transitions, 0.036 of scores missing
d = 4, p = 4, support entries = 8, treated domains = 2, nodes = 12, treatment = emax
Prepared tensors: Y (54, 7, 4)  dt (54, 6)  mu1 (2, 4)  S1 (2, 4, 4)

kind   params           engine        reference    |diff|   engine reference
null      112  -458.1834974738  -458.1834974738   1.7e-13   0.008s     0.23s
free      119  -420.0053549630  -420.0053549630   1.1e-13   0.008s     0.23s
joint     120  -442.1326327485  -442.1326327485   5.7e-14   0.007s     0.23s
```

Reading the output:
- Sixty simulated participants leave 54 after the phenotype rule removes the indeterminate ones.
- The free model has 7 more parameters than the null: H-P's degrees of freedom, $(G-1)(n_{sup}-1)=1\times(8-1)$.
- The engine and the reference agree to about $10^{-13}$, and the engine is about 30 times faster here.

---

### 2.1 Module docstring, `KINDS` and the float64 default — what `model.py` sets up

[`model.py:1`](../../src/pdx/engine/model.py#L1), [`model.py:29`](../../src/pdx/engine/model.py#L29), [`model.py:31`](../../src/pdx/engine/model.py#L31)

**What it does.**
- The docstring (lines 1–21) states the model of §2.0 and the three kinds of model.
- Line 29, `torch.set_default_dtype(torch.float64)`, makes every tensor created without an explicit dtype a 64-bit float. It runs once, when the module is first imported, and affects the whole Python process.
- Line 31 lists the kinds: `KINDS = ("null", "free", "joint")`.

**Inputs → outputs.** No function. Importing the module changes `torch.get_default_dtype()` from float32 to float64.

**The mathematics: why the speed-only model is nested in the free model.** The docstring claims that "the speed-only model is exactly nested in the free model". Take the null model with PIGD speed $e^{\gamma}$, and write $s_i=e^{\gamma}s'_i$, where $s'_i$ is the speed without the phenotype offset. Over an interval $\tau$, with $c=s_i\tau$ and $c'=s'_i\tau=e^{-\gamma}c$:

- **Transition.** A free-model drift $A_1=e^{\gamma}A$ gives $e^{c'A_1}=e^{c'e^{\gamma}A}=e^{cA}$, the same $E$.
- **Intercept.** $M'(c')=\int_0^{c'}e^{e^{\gamma}Aw}dw=e^{-\gamma}\int_0^{c}e^{Av}dv=e^{-\gamma}M(c)$, substituting $v=e^{\gamma}w$. With $b_1'=e^{\gamma}b_1$ and $\ell'=e^{\gamma}\ell$, the input is $M'(c')(b_1'+\ell')=M(c)(b_1+\ell)$, the same.
- **Noise.** $S'(c')=\int_0^{c'}e^{e^{\gamma}Aw}\,(e^{\gamma}Q_1)\,e^{e^{\gamma}A^\top w}dw=\int_0^{c}e^{Av}Q_1e^{A^\top v}dv=S(c)$, the same.

$\ell'=e^{\gamma}\ell$ is again normal, with mean $e^{\gamma}(\Gamma^y_1\tilde y+\Gamma^z_1z+\lambda_1u)$ and standard deviations $e^{\gamma}\tau_1$. Because every load parameter is phenotype-specific, the free model can represent it. Everything else ($\beta$, $\theta$, $\sigma_g$, $\omega$, $\kappa$, the landmark prior) is unchanged. So every null model is a free model with $A_1=e^{\gamma}A$, $b_1$, $Q_1$, $\Gamma^y_1$, $\Gamma^z_1$, $\lambda_1$ and $\tau_1$ multiplied by $e^{\gamma}$. `fit.null_to` (§3) writes exactly this, and `test_null_is_nested_in_free_and_joint` checks it to $10^{-8}$. The joint model nests the null model trivially, with $D_g=0$.

**Why float64.** The quantities that matter are small differences of large numbers: a likelihood ratio of a few units on log-likelihoods of thousands, gradients checked by finite differences with steps of $10^{-6}$, and the agreement with the reference to $10^{-8}$. Float32 carries about 7 significant digits, which is not enough. The stationary route for $S$ (§2.20) also subtracts two nearly equal matrices at small $c$.

**Worked example** ([`ex_spec_layout.py`](../guide_examples/s02/ex_spec_layout.py), first lines):

```python
import torch
print("torch default dtype before importing the engine:", torch.get_default_dtype())
from _setup import s1_panel                 # imports pdx.engine.model
print("torch default dtype after importing pdx.engine.model:", torch.get_default_dtype())
```

```text
torch default dtype before importing the engine: torch.float32
torch default dtype after importing pdx.engine.model: torch.float64
```

**Design choices and pitfalls.**
- The global default is a side effect of an import. Any other PyTorch code in the same process also switches to float64, which is harmless here.
- The reverse case is a trap. `transitions.py` does not import `model.py`. A script that imports only the transition code and builds tensors with `torch.tensor(...)` gets float32, and an error near $10^{-8}$ instead of $10^{-17}$ ([`ex_note_float32.py`](../guide_examples/s02/ex_note_float32.py), §2.41). Every engine entry point imports `model.py`, so the engine itself is safe.

### 2.2 `ModelSpec` — the fixed choices that define a model

[`model.py:35`](../../src/pdx/engine/model.py#L35)

**What it does.** A dataclass that holds everything about a model that is chosen, not estimated:

| field | default | meaning |
|---|---|---|
| `d` | — | number of domains |
| `support` | — | $(d,d)$ booleans, stored [to, from]: `support[j, k]` allows a drift entry from domain $k$ into domain $j$ |
| `p` | 0 | number of covariates in $z$ |
| `treat_mask` | none treated | $(d)$ booleans: domains that carry treatment terms |
| `treatment` | `"linear"` | `"emax"`, `"linear"` or `"none"` |
| `ed50`, `ec50_ldr` | 3.0, 3.0 | half-maximal dose and exposure of the Emax forms, in 100 mg LEDD |
| `meas_var` | zeros | $(d)$ measurement-error variances $\sigma^2_{e,j}$ |
| `n_quad` | 12 | Gauss–Hermite nodes $K$ |
| `n_groups` | 2 | phenotypes $G$ |
| `load_on_z`, `load_on_y1`, `speed_on_y1` | True | whether $\Gamma^z_g$, $\Gamma^y_g$ and $\theta$ exist |
| `negative_own_rates` | True | own rates $-e^{a}$ |
| `noise_era_col` | None | the column of $z$ holding era, if era scales the noise |

`__post_init__` (line 52) tidies the inputs. It copies the support as booleans and sets its diagonal to True, so own rates are always free. It fills the defaults of `treat_mask` and `meas_var`. It switches `treatment` to `"none"` when no domain is treated. Four properties derive what the layout needs:
- `sup_idx` (line 61): the support entries as (to, from) pairs, in row-major order (`np.argwhere`);
- `n_sup` (line 66): their number;
- `diag_pos` (line 70): the positions of the own rates within `sup_idx`;
- `n_treat` (line 75): the number of treated domains, or 0 when `treatment == "none"`.

`with_support` (line 78) returns a copy with another support. `dataclasses.replace` runs `__post_init__` again, so the diagonal is added back.

**Inputs → outputs.** `support` $(d,d)$ → `sup_idx` $(n_{sup},2)$ integers, `diag_pos` $(d)$ integers, `n_sup` and `n_treat` integers.

**The mathematics.** The support $\mathcal S\subseteq\{1,\dots,d\}^2$ always contains the pairs $(j,j)$. The drift is a sum over the support,
$$A=\sum_{r=1}^{n_{sup}}a_r\,e_{j_r}e_{k_r}^\top,\qquad (j_r,k_r)=\texttt{sup\_idx}[r],$$
so the $r$-th raw value always belongs to the $r$-th pair of `sup_idx`. Storing [to, from] makes `A[j, k]` the effect of domain $k$ on the rate of change of domain $j$, as in $dX_j=\sum_kA_{jk}X_k\,dt+\dots$

**Worked example** ([`ex_spec_layout.py`](../guide_examples/s02/ex_spec_layout.py)). The S1 scenario's expert map, and the clean-up in `__post_init__`:

```python
spec = spec_for(panel, info)
print(spec.support.astype(int)); print(spec.sup_idx.tolist(), spec.diag_pos.tolist(), spec.n_sup)
s2 = ModelSpec(d=3, support=np.zeros((3, 3)), treatment="emax")
```

```text
support [to, from] (1 = drift entry allowed), domains ['tremor', 'axial', 'function', 'independence']
[[1 0 0 0]
 [0 1 0 0]
 [1 1 1 0]
 [0 1 1 1]]
sup_idx (to, from), row-major: [[0, 0], [1, 1], [2, 0], [2, 1], [2, 2], [3, 1], [3, 2], [3, 3]]
diag_pos: [0, 1, 4, 7] | n_sup: 8 | treat_mask: [1, 1, 0, 0] | n_treat: 2 | noise_era_col: 0

ModelSpec(d=3, support=zeros, treatment='emax'): support diagonal = [True, True, True] | treatment = none | meas_var = [0.0, 0.0, 0.0]
```

Row 2 (function) receives from tremor and axial; row 3 (independence) receives from axial and function. The own rates sit at positions 0, 1, 4 and 7 of the raw drift block.

**Design choices and pitfalls.**
- `ed50` and `ec50_ldr` are constants of the specification, not parameters; `configs/protocol_v0.5.toml` marks the value 3.0 as provisional (§2.41).
- The four flags are never switched off, neither in `scripts/run_hp_real.py` nor in the tests. Their branches are untested (§2.41).
- With `meas_var` all zero, the landmark update makes the observed part of $P_1$ (nearly) zero. The filter still works, because the first prediction adds $S>0$.

### 2.3 `Layout` — names for the slices of the flat parameter vector

[`model.py:82`](../../src/pdx/engine/model.py#L82)

**What it does.** The optimiser (scipy's L-BFGS-B) works on one flat numpy vector, and automatic differentiation on one torch tensor. `Layout` gives every block of parameters a name, a slice of that vector and a shape. Its constructor builds the block list in a fixed order, the same for every kind except at the top:

| block | shape | meaning | transform when used |
|---|---|---|---|
| `A` | $(n_{sup})$ | reference drift on the support | own rates $-e^{a}$ |
| `A1`… (free) | $(n_{sup})$ each | PIGD's own drift | own rates $-e^{a}$ |
| `D1`… (joint) | $(n_{sup})$ each | PIGD's deviation | none |
| `gamma` (null, joint) | $(G-1)$ | log-speed offsets | none |
| `beta` | $(p)$ | covariates on log-speed | none |
| `theta` | $(d)$ or empty | landmark scores on log-speed | none |
| `log_sigma` | $(G)$ | spread of log-speed | $\sigma=e^{\cdot}$ |
| `b` | $(G,d)$ | phenotype loads | none |
| `L` | $(G,\,d(d+1)/2)$ | Cholesky factor of $Q_g$ | diagonal $e^{\cdot}$ |
| `gam_y` | $(G,d)$ or empty | load on the landmark scores (diagonal) | none |
| `gam_z` | $(G,d,p)$ or empty | load on the covariates | none |
| `lam` | $(G,d)$ | load on the speed's random part | none |
| `log_tau` | $(G,d)$ | load spread | $\tau=e^{\cdot}$ |
| `omega` | $(1)$ or empty | era on the noise | $e^{\omega\,\mathrm{era}}$ |
| `kappa`, `kappa_ldr` (if treated) | $(G,n_{treat})$ each | short- and long-duration treatment effects | none |

It then walks the list, giving each block the next `prod(shape)` positions, and stores `size`. Three methods use the result: `has(name)` (line 115) is true when the block exists and is not empty; `get(x, name)` (line 118) returns `x[slice].reshape(shape)` for a numpy array or a torch tensor, keeping the autograd graph; `put(x, name, value)` (line 122) writes into a numpy vector.

**Inputs → outputs.** `ModelSpec`, kind → `blocks` (list of (name, shape)), `slices` (name → (slice, shape)), `size`.

**The mathematics.** The size is
$$
n_{sup}\big(1+[\text{free or joint}](G-1)\big)+[\text{null or joint}](G-1)+p+d+G+Gd+G\tfrac{d(d+1)}{2}+Gd+Gdp+Gd+Gd+1+2Gn_{treat}
$$
with every flag on and era named. The difference between the free and null sizes is $(G-1)n_{sup}-(G-1)=(G-1)(n_{sup}-1)$, the degrees of freedom of H-P. In the S1 example ($d=p=4$, $G=2$, $n_{sup}=8$, $n_{treat}=2$) the null model has $8+1+4+4+2+8+20+8+32+8+8+1+4+4=112$ parameters.

**Worked example** ([`ex_spec_layout.py`](../guide_examples/s02/ex_spec_layout.py)):

```python
for kind in KINDS:
    lay = Layout(spec, kind)            # prints every block with its slice and shape
lay = Layout(spec, "null"); x = np.arange(lay.size, dtype=float)
print(lay.get(x, "b")); lay.put(x, "log_sigma", [np.log(0.3), np.log(0.5)])
```

```text
Layout 'null': 112 parameters
  A          [  0:  8) shape (8,)
  gamma      [  8:  9) shape (1,)
  beta       [  9: 13) shape (4,)
  theta      [ 13: 17) shape (4,)
  log_sigma  [ 17: 19) shape (2,)
  b          [ 19: 27) shape (2, 4)
  L          [ 27: 47) shape (2, 10)
  gam_y      [ 47: 55) shape (2, 4)
  gam_z      [ 55: 87) shape (2, 4, 4)
  lam        [ 87: 95) shape (2, 4)
  log_tau    [ 95:103) shape (2, 4)
  omega      [103:104) shape (1,)
  kappa      [104:108) shape (2, 2)
  kappa_ldr  [108:112) shape (2, 2)

Layout 'free': 119 parameters
  A          [  0:  8) shape (8,)
  A1         [  8: 16) shape (8,)
  beta       [ 16: 20) shape (4,)
  ...
Layout 'joint': 120 parameters
  A          [  0:  8) shape (8,)
  D1         [  8: 16) shape (8,)
  gamma      [ 16: 17) shape (1,)
  ...
get(x, 'b') on x = 0, 1, 2, ...:
 [[19. 20. 21. 22.]
 [23. 24. 25. 26.]]
after put(x, 'log_sigma', ...): [-1.204  -0.6931] | has('omega'): True | has('gamma'): True
```

The rows of `b` are the phenotypes: positions 19–22 hold TD's load and 23–26 PIGD's.

**Design choices and pitfalls.**
- The block order is part of the meaning of every saved vector (warm starts, checkpoints, gate results). A vector saved under one specification is only meaningful under the same flags, support and kind. `fit.null_to` copies blocks by name, which is safe.
- `get` returns a view: for a numpy vector, writing into the result writes into `x`.
- An empty block still has an entry in `slices`, so test with `has(name)`, not `name in lay.slices`.

### 2.4 `_fill` — support values into a drift matrix, own rates negative

[`model.py:129`](../../src/pdx/engine/model.py#L129)

**What it does.** Places $n_{sup}$ values at the `sup_idx` positions of a $d\times d$ zero matrix. With `constrain_diag=True` (and `negative_own_rates`), the own-rate values $v$ are first replaced by $-e^{v}$.

**Inputs → outputs.** `vals` $(n_{sup})$ tensor → $(d,d)$ tensor, differentiable in `vals`.

**The mathematics.**
$$
A_{jk}=\begin{cases}-e^{a_r} & j=k\ (\text{constrained})\\ a_r & (j,k)=\texttt{sup\_idx}[r],\ j\neq k\\ 0 & (j,k)\notin\mathcal S,\end{cases}\qquad \frac{\partial A_{jj}}{\partial a_r}=-e^{a_r}=A_{jj}.
$$
In the code, `dmask` marks the positions `diag_pos`, `torch.where(dmask, -torch.exp(vals), vals)` applies the transform there, and `A.index_put((rows, cols), vals)` writes the values. `index_put` is not in place, so the autograd graph is kept.

A negative own rate means mean reversion: left alone, each domain is pulled back towards its equilibrium. The transform keeps the diagonal negative without bounds in the optimiser. It does not make $A$ stable, because cross-links can still destabilise it, and the transition code does not need stability (§2.17).

**Worked example** ([`ex_drifts_speeds.py`](../guide_examples/s02/ex_drifts_speeds.py)). Three domains, links $0\to1$ and $1\to2$:

```python
raw_A = torch.tensor([np.log(0.25), 0.20, np.log(0.50), 0.30, np.log(0.10)])   # (0,0) (1,0) (1,1) (2,1) (2,2)
_fill(raw_A, spec, True); _fill(raw_A, spec, False)
```

```text
sup_idx (to, from): [[0, 0], [1, 0], [1, 1], [2, 1], [2, 2]]  diag_pos: [0, 2, 4]
raw A values: [-1.3863  0.2    -0.6931  0.3    -2.3026]
_fill(raw, constrain_diag=True):
 [[-0.25  0.    0.  ]
 [ 0.2  -0.5   0.  ]
 [ 0.    0.3  -0.1 ]]
_fill(raw, constrain_diag=False) (how a deviation D_g is filled):
 [[-1.3863  0.      0.    ]
 [ 0.2    -0.6931  0.    ]
 [ 0.      0.3    -2.3026]]
```

**Design choices and pitfalls.**
- An own rate of exactly zero, a domain that never reverts, can only be approached ($a\to-\infty$), and a positive one cannot be represented.
- `torch.where` evaluates $e^{v}$ at every position, including the cross-links. A cross-link value above about 709 would overflow to infinity and give a NaN gradient through the unused branch. Such a drift is absurd and fails evaluation anyway.

### 2.5 `drifts` — each phenotype's drift for the three kinds

[`model.py:139`](../../src/pdx/engine/model.py#L139)

**What it does.** Returns the $(G,d,d)$ stack of drifts at the reference speed. Phenotype 0 gets $A$ from block `A`. For $g\ge1$:
- free: `_fill(A{g}, True)`, PIGD's own drift;
- joint: `A + _fill(D{g}, False)`, so the deviation's own rates are not transformed;
- null: $A$ itself.

**Inputs → outputs.** `x` $(n)$ tensor, `Layout` → $(G,d,d)$ tensor.

**The mathematics.** The drift a typical participant of phenotype $g$ runs is $e^{\gamma_g}A_g$, where $A_g$ is this function's output and $\gamma_g$ comes from `log_speeds` (§2.6): $e^{\gamma_1}A$ in the null model, $A_1$ in the free model and $e^{\gamma_1}(A+D_1)$ in the joint model. In the joint model, PIGD's own rate is $-e^{a_{jj}}+D_{1,jj}$, with no sign constraint.

**Worked example** ([`ex_drifts_speeds.py`](../guide_examples/s02/ex_drifts_speeds.py)). The same reference drift in the three kinds. The free model gets PIGD's drift written as twice TD's. The joint model gets a deviation of $+0.15$ on the link $1\to2$ and $+0.30$ on domain 2's own rate:

```text
null: log_speeds = [0.     0.5878]
 drift of phenotype 1 at the reference speed:
 [[-0.25  0.    0.  ]
 [ 0.2  -0.5   0.  ]
 [ 0.    0.3  -0.1 ]]
 its eigenvalues (real parts): [-0.5  -0.25 -0.1 ] | clearly stable: True

free: log_speeds = [0. 0.]
 drift of phenotype 1 at the reference speed:
 [[-0.5  0.   0. ]
 [ 0.4 -1.   0. ]
 [ 0.   0.6 -0.2]]
 its eigenvalues (real parts): [-1.  -0.5 -0.2] | clearly stable: True

joint: log_speeds = [0.     0.5878]
 drift of phenotype 1 at the reference speed:
 [[-0.25  0.    0.  ]
 [ 0.2  -0.5   0.  ]
 [ 0.    0.45  0.2 ]]
 its eigenvalues (real parts): [-0.5  -0.25  0.2 ] | clearly stable: False
```

The joint model's PIGD own rate for domain 2 is $-0.1+0.3=+0.2$: positive, so its drift is unstable. The transition code then takes the composition route for $S$ (§2.19).

**Design choices and pitfalls.**
- §0.4 says own rates are $-e^{\text{raw}}$. That holds for $A$ and the free model's $A_g$, not for the joint model's phenotype-$g$ drift (§2.41).
- In the null model the stack holds the same $A$ twice. `transitions_grouped` still builds a table for each phenotype, because their $Q_g$ differ.
- The free model's $A_1$ includes PIGD's speed. Comparing $A_1$ with $A_0$ entry by entry mixes speed with mechanism; H-P and Sp-3 handle this (§5).

### 2.6 `log_speeds` — the phenotypes' log-speed offsets

[`model.py:154`](../../src/pdx/engine/model.py#L154)

**What it does.** Returns $(\gamma_0,\dots,\gamma_{G-1})$, with $\gamma_0=0$. In the free model every offset is 0; in the null and joint models the offsets for $g\ge1$ come from block `gamma`.

**Inputs → outputs.** `x`, `Layout` → $(G)$ tensor.

**The mathematics.** $\log s_i=\gamma_g+\beta^\top z_i+\theta^\top\tilde y_{i1}+\sigma_gu_i$. Fixing $\gamma_0=0$ makes TD the reference: TD's typical participant (at $z_i=0$, $\tilde y_{i1}=0$, $u_i=0$) has speed 1. $e^{\gamma_1}$ is then PIGD's speed relative to TD at the same covariates and landmark scores, the basis of Sp-1 (§5).

**Worked example.** In the output of §2.5, `log_speeds` is $[0,\ 0.5878]$ for the null and joint models ($\log1.8=0.5878$) and $[0,\ 0]$ for the free model.

**Design choices and pitfalls.**
- In the free model PIGD's speed is absorbed into $A_1$, $b_1$, $Q_1$ and the load parameters, so these are on a different time scale from the null model's (§2.1).
- $\beta$ and $\theta$ are shared by the phenotypes in every kind; $\sigma_g$ is phenotype-specific.

### 2.7 `noise_cov` — the noise covariance from its Cholesky factor

[`model.py:162`](../../src/pdx/engine/model.py#L162)

**What it does.** For each phenotype it does three things:
1. It places the $d(d+1)/2$ raw values of `L[g]` in a lower-triangular matrix, row by row in the order of `torch.tril_indices`.
2. It exponentiates the diagonal.
3. It returns $Q_g=G_gG_g^\top$.

**Inputs → outputs.** `x`, `Layout` → $(G,d,d)$ tensor of symmetric positive definite matrices.

**The mathematics.**
$$
(G_g)_{jk}=\begin{cases}e^{l_{jj}} & j=k\\ l_{jk} & j>k\\ 0 & j<k\end{cases},\qquad Q_g=G_gG_g^\top .
$$
A lower-triangular matrix with a positive diagonal is invertible, so $Q_g$ is positive definite for every raw vector. The map is one-to-one, because the Cholesky factor with a positive diagonal is unique. The line `L - torch.diag(dg) + torch.diag(torch.exp(dg))` swaps the diagonal for its exponential without an in-place operation, which autograd would refuse on a tensor it needs. $Q_g$ is the noise per unit time at the reference speed: over a short interval $\tau$ at speed $s$, the noise covariance is about $s\tau Q_g$, the first term of $S$ (§2.15). It is unrestricted, so it carries same-year associations between domains (protocol §6.3).

**Worked example** ([`ex_noise_cov.py`](../guide_examples/s02/ex_noise_cov.py)):

```python
G = np.array([[0.30, 0.00, 0.00], [0.05, 0.20, 0.00], [-0.02, 0.03, 0.25]])
raw = G[np.tril_indices(3)].copy(); raw[[0, 2, 5]] = np.log(np.diag(G))
Q = noise_cov(x, lay)                      # with raw written into block L for both phenotypes
```

```text
lower-triangle order, torch: [[0, 1, 1, 2, 2, 2], [0, 0, 1, 0, 1, 2]]
lower-triangle order, numpy: [[0, 1, 1, 2, 2, 2], [0, 0, 1, 0, 1, 2]]

raw L block of phenotype 0: [-1.204   0.05   -1.6094 -0.02    0.03   -1.3863]
Q_0 = G G':
 [[ 0.09    0.015  -0.006 ]
 [ 0.015   0.0425  0.005 ]
 [-0.006   0.005   0.0638]]
eigenvalues of Q_0: [0.0366 0.0647 0.095 ]
pack_chol(Q_0) - raw: 4.440892098500626e-16
```

By hand: $Q_{11}=0.05^2+0.20^2=0.0425$ and $Q_{21}=(-0.02)(0.05)+(0.03)(0.20)=0.005$.

**Design choices and pitfalls.**
- torch and numpy order the lower triangle the same way (row by row), so `noise_cov` and `pack_chol` (§2.11) are exact inverses.
- The Cholesky parametrisation depends on the order of the domains: reordering them changes which raw value describes which correlation. Maximum likelihood is unaffected, because the same set of matrices is reachable.

### 2.8 `treat_effect` — treatment terms on the observed scores

[`model.py:176`](../../src/pdx/engine/model.py#L176)

**What it does.** Computes $h_{ik}$ for a batch of participants: zero for untreated domains, and for each treated domain the short-duration term (dose, only in the ON state) plus the long-duration term (exposure, in either state). Both terms have phenotype-specific sizes.

**Inputs → outputs.** `x`, `Layout`, `dose` $(n,T)$, `expo` $(n,T)$, `on` $(n,T)$, `g` $(n)$ → `h` $(n,T,d)$.

**The mathematics.** For a treated domain $j$ (§2.0):
$$
h_{ik,j}=-\Big[\kappa_{g,j}\,f_D(D_{ik})\,\mathrm{on}_{ik}+\kappa^{\mathrm{LDR}}_{g,j}\,f_E(\mathcal E_{ik})\Big],\qquad f_D(x)=\frac{x}{\mathrm{ED}_{50}+x},\quad f_E(x)=\frac{x}{\mathrm{EC}_{50}+x}.
$$
In the Emax form, $\kappa_{g,j}$ is the maximal improvement as the dose grows, in normal-score units, and $\mathrm{ED}_{50}=3$ (300 mg LEDD) is the dose that gives half of it. In the linear form, $f(x)=x$ and $\kappa$ is the effect per 100 mg. The minus sign makes a positive $\kappa$ an improvement, because scores are oriented so that higher is worse; $\kappa$ itself is unconstrained.

Line by line:
- `kap = lay.get(x, "kappa")[g]` selects each participant's phenotype row, $(n,n_{treat})$.
- `fd`, `fe` are $f_D(D)$ and $f_E(\mathcal E)$, both $(n,T)$.
- `eff = -(kap[:, None, :] * (fd * on)[:, :, None] + kapl[:, None, :] * fe[:, :, None])` broadcasts $(n,1,n_{treat})\times(n,T,1)$ to $(n,T,n_{treat})$.
- `h.index_copy(2, tm, eff)` writes these columns into the treated domains of a zero $(n,T,d)$ tensor.

**Worked example** ([`ex_treat_effect.py`](../guide_examples/s02/ex_treat_effect.py)). One PIGD participant, domains 0 and 1 treated:

```python
dose = torch.tensor([[0.0, 1.0, 3.0, 6.0, 12.0]]); on = torch.tensor([[0.0, 1.0, 1.0, 0.0, 1.0]])
expo = torch.tensor([[0.0, 0.5, 2.0, 4.0, 8.0]])
h = treat_effect(x, lay, dose, expo, on, torch.tensor([1]))
```

```text
emax: kappa[PIGD] = [0.6 0.2], kappa_ldr[PIGD] = [0.3 0.1]
 visit  dose  on  expo   h (domains 0, 1, 2)
  0     0.0   0   0.0   [-0. -0.  0.]
  1     1.0   1   0.5   [-0.1929 -0.0643  0.    ]
  2     3.0   1   2.0   [-0.42 -0.14  0.  ]
  3     6.0   0   4.0   [-0.1714 -0.0571  0.    ]
  4    12.0   1   8.0   [-0.6982 -0.2327  0.    ]
linear: kappa[PIGD] = [0.6 0.2], kappa_ldr[PIGD] = [0.3 0.1]
 visit  dose  on  expo   h (domains 0, 1, 2)
  0     0.0   0   0.0   [-0. -0.  0.]
  1     1.0   1   0.5   [-0.75 -0.25  0.  ]
  2     3.0   1   2.0   [-2.4 -0.8  0. ]
  3     6.0   0   4.0   [-1.2 -0.4  0. ]
  4    12.0   1   8.0   [-9.6 -3.2  0. ]
by hand, Emax, visit 2, domain 0: -(0.6 * 0.500 * 1 + 0.3 * 0.400) = -0.4200
```

At visit 3 the examination is OFF, so only the long-duration term acts: $-0.3\times4/7=-0.1714$. The linear form does not saturate: 1,200 mg gives $-9.6$.

**Design choices and pitfalls.**
- $\mathrm{ED}_{50}$ and $\mathrm{EC}_{50}$ are fixed, not estimated; only the maximal effects are (§2.41).
- The terms act on the observed scores only, not on the dynamics: a dose change does not change where the disease is heading. The sensitivity analysis with a treatment effect on the drift (protocol §6.1) is not implemented (§2.41).
- $\kappa$ is phenotype-specific in every kind, so a different treatment response is never read as a mechanism difference (protocol §6.1).

### 2.9 `unpack_numpy` — named quantities for reporting and simulation

[`model.py:196`](../../src/pdx/engine/model.py#L196)

**What it does.** Turns a flat numpy vector into a dictionary of numpy arrays on the model's scale. The drift, speed and noise go through the same torch functions as the likelihood (`drifts`, `log_speeds`, `noise_cov`); $\sigma$ and $\tau$ are exponentiated. Absent blocks come back as zeros (`theta`, `gam_y`, `gam_z`, `omega`). For the joint model it adds `D` $=$ drift of $g\ge1$ minus drift of 0. Its users are the reference likelihood (§2.39), the simulator (§3) and `Fit.params`.

**Inputs → outputs.** `x` $(n)$ numpy → `A` $(G,d,d)$, `log_speed` $(G)$, `Q` $(G,d,d)$, `b` $(G,d)$, `sigma` $(G)$, `beta` $(p)$, `lam` $(G,d)$, `tau` $(G,d)$, `theta` $(d)$, `gam_y` $(G,d)$, `gam_z` $(G,d,p)$, `omega` (float), `kappa` and `kappa_ldr` $(G,n_{treat})$ if treated, `D` $(G-1,d,d)$ for the joint model.

**The mathematics.** No new mathematics. For the joint model, $D=A_g^{\text{used}}-A_0=(A+D_g)-A=D_g$ as filled by `_fill(..., False)`.

**Worked example** ([`ex_unpack_pack.py`](../guide_examples/s02/ex_unpack_pack.py)):

```text
null: {'A': (2, 4, 4), 'log_speed': (2,), 'Q': (2, 4, 4), 'b': (2, 4), 'sigma': (2,), 'beta': (4,), 'lam': (2, 4), 'tau': (2, 4), 'theta': (4,), 'gam_y': (2, 4), 'gam_z': (2, 4, 4), 'omega': 'scalar', 'kappa': (2, 2), 'kappa_ldr': (2, 2)}
joint: {..., 'kappa_ldr': (2, 2), 'D': (1, 4, 4)}
```

**Design choices and pitfalls.**
- Because the reference likelihood calls this function, the parameter transforms are shared by the engine and the reference, and their agreement does not test them (§2.41). The examples of §2.4–§2.7 check them by hand.
- Returning zeros for absent blocks spares every caller a branch.

### 2.10 `pack_drift` — a drift matrix back to raw support values

[`model.py:217`](../../src/pdx/engine/model.py#L217)

**What it does.** The inverse of `_fill`. It reads the support entries of a $(d,d)$ matrix, and replaces each own rate $v$ by $\log\max(-v,10^{-4})$.

**Inputs → outputs.** `A` $(d,d)$ numpy, `spec` → $(n_{sup})$ numpy.

**The mathematics.** $a_{jj}=\log(-A_{jj})$ when $A_{jj}<-10^{-4}$. Otherwise $a_{jj}=\log10^{-4}=-9.2103$, which maps back to $A_{jj}=-10^{-4}$. Cross-links are copied.

**Worked example** ([`ex_unpack_pack.py`](../guide_examples/s02/ex_unpack_pack.py)):

```text
A of phenotype 0 (reference speed):
 [[-0.1987  0.      0.      0.    ]
 [ 0.     -0.0785  0.      0.    ]
 [ 0.2686  0.2215 -0.1124  0.    ]
 [ 0.     -0.0168  0.2887 -0.3456]]
pack_drift(A_0) - raw block 'A': 0.0
own rate +0.02 -> raw -9.2103 -> own rate -1.0e-04
own rate -4e-05 -> raw -9.2103 -> own rate -1.0e-04
```

**Design choices and pitfalls.**
- The floor exists for start values: yearly least-squares fits can give a positive own rate, which has no raw value.
- `fit.null_to` also uses `pack_drift`, to write $e^{\gamma}A$ as the free model's $A_1$. When $e^{\gamma}|A_{jj}|<10^{-4}$ the floor changes the model, and the free model's start is no longer the null fit. A synthetic own rate of $-4.5\times10^{-5}$ moved the log-likelihood by $5.9\times10^{-4}$ ([`ex_note_nesting_clip.py`](../guide_examples/s02/ex_note_nesting_clip.py), §2.41).

### 2.11 `pack_chol` — a covariance matrix back to raw Cholesky values

[`model.py:227`](../../src/pdx/engine/model.py#L227)

**What it does.** The inverse of `noise_cov`'s parametrisation. It symmetrises $Q$, floors its eigenvalues at $10^{-6}$, takes the Cholesky factor, replaces its diagonal by logs, and returns the lower triangle row by row.

**Inputs → outputs.** `Q` $(d,d)$ numpy → $(d(d+1)/2)$ numpy.

**The mathematics.** With $\tfrac12(Q+Q^\top)=V\,\mathrm{diag}(w)\,V^\top$, it computes $\tilde Q=V\,\mathrm{diag}(\max(w,10^{-6}))\,V^\top=GG^\top$ and returns $(\log G_{jj};\ G_{jk},\ j>k)$.

**Worked example** ([`ex_noise_cov.py`](../guide_examples/s02/ex_noise_cov.py)). The round trip of §2.7 is exact to $4\times10^{-16}$. A nearly singular $Q$ (smallest eigenvalue $10^{-8}$) is changed by the floor:

```text
near-singular Q: smallest eigenvalue 1.0000000024079617e-08
after the round trip: smallest eigenvalue 1e-06 | max |change| in Q: 4.95e-07
```

**Design choices and pitfalls.**
- The floor is needed for start values, where a residual covariance minus the measurement error can be indefinite (`fit.start_values`).
- In `null_to` it makes the nesting inexact only when $e^{\gamma}Q_g$ has an eigenvalue below $10^{-6}$.

---

### 2.12 `bmm` — batched products of small matrices, split into blocks

[`linalg.py:1`](../../src/pdx/engine/linalg.py#L1), [`linalg.py:15`](../../src/pdx/engine/linalg.py#L15)

**What it does.** `bmm(a, b)` multiplies two batches of matrices, `a` of shape $(\dots,r,k)$ and `b` of shape $(\dots,k,c)$. If $r\cdot k\cdot c<400$ (`_LIMIT`) it returns `a @ b`. Otherwise it cuts the rows of `a` and the columns of `b` into blocks of
$$\texttt{blk}=\Big\lfloor\sqrt{\lfloor 399/k\rfloor}\Big\rfloor,$$
multiplies every row block by every column block, and glues the pieces back with `torch.cat`. Each block product has at most $\texttt{blk}\cdot k\cdot\texttt{blk}<400$.

**Why.** The module docstring explains: PyTorch's CPU batched product uses a fast built-in kernel while rows × columns × inner size stays below 400. Above that it calls the BLAS routine once per matrix in the batch, and for thousands of $10\times10$ matrices those calls dominate. Splitting a $10\times10$ product into $6\times10\times6$ pieces (360) keeps every call on the fast kernel.

**Inputs → outputs.** `a` $(\dots,r,k)$, `b` $(\dots,k,c)$ → $(\dots,r,c)$, the same values as `a @ b` up to rounding.

**The mathematics.** Block multiplication: with `a` split into row blocks $a_1,a_2$ and `b` into column blocks $b_1,b_2$,
$$\begin{pmatrix}a_1\\a_2\end{pmatrix}\begin{pmatrix}b_1&b_2\end{pmatrix}=\begin{pmatrix}a_1b_1&a_1b_2\\a_2b_1&a_2b_2\end{pmatrix}.$$
The inner dimension $k$ is never split, so each entry is still one complete dot product; only the order of the additions inside it can differ. The block sizes are 7 at $d=8$ ($\lfloor399/8\rfloor=49$) and 6 at $d=10$ ($\lfloor399/10\rfloor=39$). At $d\le7$, $d^3\le343<400$, so nothing is split.

The engine routes through `bmm` every batched $d\times d$ by $d\times d$ product inside the Kalman loop (`likelihood.py:172–174`, `189–191`) and inside the composition of transitions (`transitions.py:71–72`, `154`).

**Worked example** ([`ex_bmm.py`](../guide_examples/s02/ex_bmm.py)). Batches shaped like one chunk (32 participants × 12 nodes × 6 intervals), forward and backward pass, single-threaded, best of 7 rounds. Then the whole likelihood and gradient at ten domains, with `bmm` swapped for the plain product:

```python
t_plain = best(lambda: (a @ b).sum().backward())
t_block = best(lambda: bmm(a, b).sum().backward())
...
likelihood.bmm = transitions.bmm = lambda u, w: u @ w      # the blocking switched off
```

```text
threads: 1 | limit on rows x inner x columns: 400
batch (32, 12, 6) of d x d products; best time per forward + backward pass (ms)
 d  r*k*c  block   a @ b     bmm   ratio  max|difference|
 4     64     4     0.13    0.12    1.0   0.0e+00
 7    343     7     0.57    0.61    0.9   0.0e+00
 8    512     7     6.92    1.82    3.8   3.6e-15
10   1000     6     8.65    3.28    2.6   3.6e-15

value_and_grad, 186 synthetic participants, d = 10, chunks of 32:
  with bmm 526 ms, with plain @ 694 ms, ratio 1.3; |value difference| 0.0e+00
```

Reading the output:
- The plain product jumps from 0.57 ms at $d=7$ (343 < 400) to 6.92 ms at $d=8$ (512 ≥ 400): that jump is PyTorch's switch of method.
- The blocked product removes most of the jump, with identical results to $4\times10^{-15}$.
- On the whole likelihood the gain is smaller, because the Cholesky factorisations and solves are not blocked.

**Design choices and pitfalls.**
- The measured gains are well below the docstring's "about 35 times" (§2.41): 2.6× per product at $d=10$, and for the whole evaluation 1.3× here and 1.6× in an earlier run of the same computation. Timings on the M4 vary between runs, because macOS may move a single-threaded process between performance and efficiency cores.
- Some products do not go through `bmm`, and do not need to: matrix–vector products ($d\cdot d\cdot1<400$), the batched Cholesky factorisation and solve (LAPACK, one matrix at a time), the landmark update (once per chunk), and products whose second factor is one $d\times d$ matrix (`Ek @ Sig`), which PyTorch folds into a single large product.

### 2.13 Module docstring of `transitions.py` — $E$, $M$ and $S$ from the SDE, and how they compose

[`transitions.py:1`](../../src/pdx/engine/transitions.py#L1)

**What it does.** The docstring defines the three quantities, states how they compose over consecutive intervals, and outlines the method: split $c$ into coarse, fine and remainder parts, and use one of two exact routes for $S$. It notes that `transitions`, with one matrix exponential per value, is kept as a check.

**Inputs → outputs.** None; this subsection derives what the functions below compute.

**The mathematics.**

*1. The solution over one interval.* Fix a participant's speed $s$ and load $\ell$, both constant in time, and write $Q$ for $e^{\omega\,\mathrm{era}}Q_g$. Multiply $dX=s(AX+b+\ell)\,dt+\sqrt s\,G\,dW$ by the integrating factor $e^{-sAr}$:
$$d\big(e^{-sAr}X(r)\big)=e^{-sAr}\big(dX-sAX\,dr\big)=e^{-sAr}\big[s(b+\ell)\,dr+\sqrt s\,G\,dW(r)\big].$$
Integrate from $t$ to $t+\tau$ and multiply by $e^{sA(t+\tau)}$:
$$X(t+\tau)=e^{sA\tau}X(t)+\int_0^\tau e^{sA(\tau-r)}s\,dr\,(b+\ell)+\sqrt s\int_0^\tau e^{sA(\tau-r)}G\,dW(t+r).$$

*2. The intercept.* Substitute $w=s(\tau-r)$, $dw=-s\,dr$:
$$\int_0^\tau e^{sA(\tau-r)}s\,dr=\int_0^{s\tau}e^{Aw}\,dw=M(c),\qquad c=s\tau.$$

*3. The noise.* The stochastic integral is Gaussian with mean zero. By the Itô isometry its covariance is
$$s\int_0^\tau e^{sA(\tau-r)}\,GG^\top\,e^{sA^\top(\tau-r)}\,dr=\int_0^{c}e^{Aw}Qe^{A^\top w}\,dw=S(c),$$
with the same substitution. So
$$X(t+\tau)\mid X(t)\sim N\big(E(c)X(t)+M(c)(b+\ell),\ S(c)\big),\qquad E(c)=e^{cA},$$
and speed and interval enter only through $c=s\tau$.

*4. Closed forms, which the code avoids.*
- If $A$ is invertible, $M(c)=A^{-1}(e^{cA}-I)$. The code never inverts $A$, which may be singular (an own rate near zero, a marginal drift).
- The block exponential $\exp\!\big(c\begin{pmatrix}A&I\\0&0\end{pmatrix}\big)=\begin{pmatrix}E(c)&M(c)\\0&I\end{pmatrix}$ holds for any $A$. The powers of the block matrix are $\begin{pmatrix}A^k&A^{k-1}\\0&0\end{pmatrix}$ for $k\ge1$, and $\sum_{k\ge1}c^kA^{k-1}/k!=\int_0^ce^{Aw}dw$. The reference implementation uses this (§2.38).

*5. Composition.* For intervals $a$ and $b$:
$$E(a+b)=E(a)E(b),\qquad M(a+b)=M(a)+E(a)M(b),\qquad S(a+b)=S(a)+E(a)S(b)E(a)^\top .$$
- The first is the exponential law.
- For the second, split the integral: $\int_0^{a+b}=\int_0^a+\int_a^{a+b}$, and in the second part substitute $w=a+v$, so that $e^{Aw}=e^{Aa}e^{Av}$.
- For the third, split the same way:
  $$\int_a^{a+b}e^{Aw}Qe^{A^\top w}dw=e^{Aa}\Big[\int_0^be^{Av}Qe^{A^\top v}dv\Big]e^{A^\top a}.$$

$E$ and $M$ are power series in $A$, so they commute with each other. $S$ is not a function of $A$ alone, but its rule needs no commutation. Exchanging $a$ and $b$ gives the same $S(a+b)$, because both are the same integral.

*6. Short intervals.* $E\approx I+cA$, $M\approx cI$, $S\approx cQ$: over a short time the state barely moves, the load pushes it by $c(b+\ell)$, and the noise grows linearly in time.

**Worked example** ([`ex_transitions_grouped.py`](../guide_examples/s02/ex_transitions_grouped.py), first part). $A=\begin{pmatrix}-0.25&0\\0.20&-0.50\end{pmatrix}$, $Q=\begin{pmatrix}0.10&0.02\\0.02&0.08\end{pmatrix}$, speed 1.5 over one year:

```text
c = s * dt = 1.5
E:
 [[0.6873 0.    ]
 [0.1719 0.4724]]
expm(cA):
 [[0.6873 0.    ]
 [0.1719 0.4724]]
M:
 [[1.2508 0.    ]
 [0.1565 1.0553]]
A^{-1}(E - I):
 [[ 1.2508 -0.    ]
 [ 0.1565  1.0553]]
S:
 [[0.1055 0.0304]
 [0.0304 0.0681]]
Sigma - E Sigma E' (scipy):
 [[0.1055 0.0304]
 [0.0304 0.0681]]
max |differences|: E 5.6e-16, M 4.4e-16 (block expm), S 5.6e-17
```

By hand: domain 0 receives nothing ($A_{01}=0$), so its entries follow the scalar formulas with $a=0.25$:
- $E_{00}=e^{-0.375}=0.6873$;
- $M_{00}=(1-e^{-0.375})/0.25=1.2508$;
- $S_{00}=q(1-e^{-2ac})/(2a)=0.1\times(1-e^{-0.75})/0.5=0.1055$.

**Design choices and pitfalls.** Because everything depends on $c$ only, the engine never needs the speed and the interval separately. That is why a single function of $c$ serves every participant, node and visit.

### 2.14 Constants and `_tr`

[`transitions.py:36`](../../src/pdx/engine/transitions.py#L36), [`transitions.py:41`](../../src/pdx/engine/transitions.py#L41)

**What it does.**
- `TAYLOR_ORDER = 12` is the last power kept in the Taylor series.
- `STABLE_MARGIN = 1e-3` is the stability margin of §2.19.
- `_FACT` holds the factorials $0!,\dots,13!$.
- `_tr(a)` swaps the last two axes, a transpose for batches of matrices.

**Inputs → outputs.** `_tr`: $(\dots,m,n)\to(\dots,n,m)$.

**The mathematics.** The series of §2.15 are used only for $\delta\|A\|_1\le0.125$. The first omitted term of $E$ and $M$ is then at most $0.125^{13}/13!$ relative to $\|E\|\approx1$. For $S$, the operator $\mathcal L(X)=AX+XA^\top$ has $\|\mathcal L\|_1\le\|A\|_1+\|A\|_\infty$, because $\|XA^\top\|_1\le\|X\|_1\|A\|_\infty$. The first omitted term of $S$ is at most $(\delta\|\mathcal L\|_1)^{13}/14!$ relative to $\delta\|Q\|$.

**Worked example** ([`ex_taylor_compose.py`](../guide_examples/s02/ex_taylor_compose.py), first lines):

```text
_FACT = [1, 1, 2, 6] ... 6227020800 (14 entries: 0! to 13!)
first omitted term, E and M, delta ||A||_1 <= 0.125:   0.125^13 / 13! = 2.92e-22
first omitted term, S, (||A||_1 + ||A||_inf) delta <= 0.25: 0.25^13 / 14! = 1.71e-19
the same if ||A||_inf = 5 ||A||_1 (0.75):               0.75^13 / 14! = 2.73e-13
```

**Design choices and pitfalls.**
- The docstring's "truncation below 1e-18" holds for $E$ and $M$, and for $S$ when the row sums of $|A|$ are like its column sums. For a drift with one heavily fed domain ($\|A\|_\infty=5\|A\|_1$) the bound for $S$ weakens to $3\times10^{-13}$. On such a 10-domain drift the actual error of the composed $S$ stays at rounding level, so the bound is loose (end of the same script):

  ```text
  row-heavy drift: ||A||_1 = 0.5, ||A||_inf = 2.5
    c =  0.3: |S composition - (Sigma - E Sigma E')| = 6.2e-17
    c =  1.0: |S composition - (Sigma - E Sigma E')| = 2.8e-17
    c =  5.0: |S composition - (Sigma - E Sigma E')| = 8.3e-17
    c = 20.0: |S composition - (Sigma - E Sigma E')| = 6.7e-16
  ```
- With order 12 and $h=0.125/\|A\|_1$, the fine table needs 32 steps per coarse step; a higher order would allow longer steps but cost more matrix powers.

### 2.15 `_taylor` — $E$, $M$ and $S$ over short intervals by Taylor series

[`transitions.py:45`](../../src/pdx/engine/transitions.py#L45)

**What it does.** For one drift $A$, one $Q$ (or none), and any tensor of short intervals $\delta$, it returns $E(\delta)$, $M(\delta)$ and $S(\delta)$ from their Taylor series up to order 12.

**Inputs → outputs.** `A` $(d,d)$, `Q` $(d,d)$ or `None`, `delta` of any shape $(\dots)$ → `E`, `M`, `S` each $(\dots,d,d)$; `S` is `None` without `Q`.

**The mathematics.**
$$E(\delta)=\sum_{k=0}^{12}\frac{\delta^k}{k!}A^k,\qquad M(\delta)=\int_0^\delta e^{Aw}dw=\sum_{k=0}^{12}\frac{\delta^{k+1}}{(k+1)!}A^k .$$
For $S$, let $F(w)=e^{Aw}Qe^{A^\top w}$. Then $F'(w)=AF+FA^\top=\mathcal L(F)$ with $F(0)=Q$, so $F(w)=e^{w\mathcal L}(Q)=\sum_kw^k\mathcal L^k(Q)/k!$. Integrating term by term:
$$S(\delta)=\sum_{k=0}^{12}\frac{\delta^{k+1}}{(k+1)!}\,\mathcal L^k(Q),\qquad \mathcal L(X)=AX+XA^\top .$$

**From the formulas to the code.**

| formula | code |
|---|---|
| $A^0,\dots,A^{12}$ | `pows`, stacked as `P` $(13,d,d)$ |
| $k=0,\dots,12$ | `k = torch.arange(13)` |
| $\delta^k$ | `dk = delta[..., None] ** k`, shape $(\dots,13)$ |
| $\delta^{k+1}/(k+1)!$ | `w1 = dk * delta[..., None] / fact[1:]` |
| $\sum_k\frac{\delta^k}{k!}A^k$ | `einsum("...j,jab->...ab", dk / fact[:-1], P)` |
| $\mathcal L^k(Q)$ | `lyap`, built by `A @ L + L @ A.T` |
| $S(\delta)$ | `einsum("...j,jab->...ab", w1, stack(lyap))` |

The powers are computed once. The weights of all the intervals form a matrix with one row per interval, so each series is a single product of an $(N\times13)$ weight matrix and the $(13\times d^2)$ stacked powers. That is fast, and differentiable in both $\delta$ and $A$.

**Worked example** ([`ex_taylor_compose.py`](../guide_examples/s02/ex_taylor_compose.py)). The fine step of the $2\times2$ example, $h=0.125/\|A\|_1=0.25$, against scipy:

```text
||A||_1 = 0.5, fine step h = 0.125 / ||A||_1 = 0.25, Taylor order 12
E(h):
 [[0.9394 0.    ]
 [0.0455 0.8825]]
M(h):
 [[0.2423 0.    ]
 [0.0059 0.235 ]]
S(h):
 [[0.0235 0.0051]
 [0.0051 0.0179]]
max |Taylor - scipy|: E 1.1e-16  M 5.6e-17  S 6.9e-18
first terms of S: delta*Q + delta^2/2 * (AQ + QA') =
 [[0.0234 0.0052]
 [0.0052 0.0178]]
```

The first two terms already give $S(h)$ to about $10^{-4}$; the other eleven bring it to rounding level.

**Design choices and pitfalls.**
- The series is accurate only for short intervals, which is why `_grid` never calls it with $\delta\|A\|_1>0.125$ (up to rounding).
- At $\delta=0$ (padded visits), `0.0 ** 0` is 1, so $E=I$, $M=S=0$. PyTorch's derivative of `x ** k` is defined as 0 for $k=0$, so the gradient there is 0, not NaN.

### 2.16 `_compose` — joining two intervals

[`transitions.py:67`](../../src/pdx/engine/transitions.py#L67)

**What it does.** Given the triples $(E_1,M_1,S_1)$ over an interval $a$ and $(E_2,M_2,S_2)$ over an interval $b$, it returns the triple over $a+b$ (§2.13, point 5). $S$ may be `None` throughout, when only $E$ and $M$ are wanted.

**Inputs → outputs.** Two triples of broadcastable $(\dots,d,d)$ tensors → one triple.

**The mathematics.**
$$E=E_1E_2,\qquad M=M_1+E_1M_2,\qquad S=S_1+E_1S_2E_1^\top .$$
If $S_1$ and $S_2$ are positive semi-definite, so is $S$, since $E_1S_2E_1^\top\succeq0$. Composing many short pieces therefore never produces a negative variance, however long the total interval and whatever the eigenvalues of $A$. The docstring calls this "composition of positive semi-definite pieces".

**Worked example** ([`ex_taylor_compose.py`](../guide_examples/s02/ex_taylor_compose.py)). $a=0.25$, $b=0.10$:

```python
ab = _compose(_taylor(A, Q, torch.tensor(a)), _taylor(A, Q, torch.tensor(b)))
ba = _compose(_taylor(A, Q, torch.tensor(b)), _taylor(A, Q, torch.tensor(a)))
direct = _taylor(A, Q, torch.tensor(a + b))
```

```text
E: |compose(a,b) - direct(a+b)| = 1.1e-16, |compose(a,b) - compose(b,a)| = 6.9e-18
M: |compose(a,b) - direct(a+b)| = 5.6e-17, |compose(a,b) - compose(b,a)| = 0.0e+00
S: |compose(a,b) - direct(a+b)| = 6.9e-18, |compose(a,b) - compose(b,a)| = 3.5e-18
```

**Design choices and pitfalls.**
- The docstring reads "over interval a, then over b". The value does not depend on the order, because the process is time-homogeneous, so the order is only a convention.
- All products go through `bmm` (§2.12).

### 2.17 `_grid` — any interval from binary powers, a table and a Taylor remainder

[`transitions.py:75`](../../src/pdx/engine/transitions.py#L75)

**What it does.** Computes $E(c)$, $M(c)$ and (if `Q` is given) $S(c)$ for one drift and every value in a tensor $c\ge0$, without one matrix exponential per value. Each $c$ is split as
$$c=n_H\,H+m\,h+\delta,\qquad h=\frac{0.125}{\|A\|_1},\quad H=32h,\quad n_H\ge0,\ 0\le m\le31,\ 0\le\delta<h .$$
- The remainder $\delta$ comes from the Taylor series (§2.15).
- The $m$ fine steps come from a table of 32 entries built once.
- The $n_H$ coarse steps come from the binary expansion of $n_H$, using the triples over $H,2H,4H,\dots$ built by repeated squaring.

**Inputs → outputs.** `A` $(d,d)$, `Q` $(d,d)$ or `None`, `c` of any shape → `E`, `M`, `S` of shape $(\text{shape of }c,d,d)$; `S` is `None` without `Q`.

**The mathematics.** Write $n_H=\sum_b\beta_b2^b$ with bits $\beta_b\in\{0,1\}$. Let $P_b$ be the triple over $2^bH$, so $P_0$ is over $H$ and $P_{b+1}=P_b\circ P_b$, where $\circ$ is `_compose`. Then
$$\text{triple}(c)=\Big(\mathop{\bigcirc}_{b:\,\beta_b=1}P_b\Big)\circ\text{table}[m]\circ\text{Taylor}(\delta).$$
This is exact because composition is exact (§2.13). Only the Taylor series truncates, at $10^{-19}$ or less per use: once in the fine step $h$, from which the table and the powers are built, and once in the remainder $\delta$. The compositions add only rounding. The cost in compositions grows like $\log_2(c/H)$.

*The gradient with respect to $c$.* The integers $n_H$ and $m$ are computed from a detached copy of $c$, and only $\delta$ carries the derivative. That derivative is exact: since $E(c)=E(n_HH)E(mh)E(\delta)$,
$$\frac{dE}{dc}=E(n_HH)E(mh)E'(\delta)=E(c)A .$$
The same holds for $M$ and $S$. The step $h$ is a Python number computed from `A.detach()`, so it is not differentiated either. That is also correct, because the decomposition is exact for any value of $h$, which affects only the accuracy.

**From the formulas to the code, line by line.**

| lines | code | meaning |
|---|---|---|
| 78–81 | `norm`, `h`, `nh = 32`, `H` | $\Vert A\Vert_1$ (largest column sum of $\lvert A\rvert$), $h=0.125/\max(\Vert A\Vert_1,10^{-3})$, $H=32h$ |
| 85 | `step = _taylor(A, Q, h)` | the triple over one fine step |
| 86–88 | `fine` | `fine[m]` = triple over $mh$, $m=0..32$, by 32 compositions |
| 89–92 | `big`, `Pf`, `Mf`, `Sf` | triple over $H$; the table entries $m=0..31$ stacked |
| 93–96 | `cd`, `cmax` | a detached copy of $c$; a non-finite $c$ raises `ValueError` |
| 97–100 | `iH`, `rem`, `ih`, `delta` | $n_H=\lfloor c/H\rfloor$; $c-n_HH$ (keeps the gradient); $m=\lfloor\mathrm{rem}/h\rfloor$ clamped to $[0,31]$; $\delta=\mathrm{rem}-mh$ |
| 101–102 | `state` | the identity triple $(I,0,0)$ for every element |
| 103 | `nbits` | bit length of the largest $n_H$ |
| 104–111 | the loop | for bit $b$: compose `state` with $P_b$ where the bit is set (`torch.where`), then square $P_b$ |
| 112–114 | `_taylor(A, Q, delta)`, two `_compose` | remainder, table entry, then coarse part |
| 115 | `0.5 * (S + S')` | removes rounding asymmetry |

**Worked example** ([`ex_grid.py`](../guide_examples/s02/ex_grid.py)). The $2\times2$ drift ($h=0.25$, $H=8$) at seven intervals, against scipy. Then the derivative in $c$, and the cost as $c$ grows:

```text
h = 0.25, H = 32 h = 8.0
      c     iH  ih    delta  nbits   |E-expm|   |M-M_scipy|   |S-(Sig-E Sig E')|
    0.0      0   0   0.0000     0   0.0e+00     0.0e+00       0.0e+00
    0.1      0   0   0.1000     0   3.5e-18     2.8e-17       4.3e-17
    0.3      0   1   0.0500     0   2.2e-16     5.6e-17       2.8e-17
    7.5      0  30   0.0000     0   7.4e-15     1.2e-14       2.4e-16
    8.0      1   0   0.0000     1   9.6e-15     2.4e-15       2.6e-16
   60.0      7  16   0.0000     3   1.2e-21     2.7e-15       1.5e-16
 5000.0    625   0   0.0000    10   0.0e+00     2.9e-15       1.5e-16

d/dc sum(E(c)) at c = 7.5: autograd -0.071361509666, A expm(cA) summed -0.071361509666

    c_max   nbits   time for 2000 values (E, M and S)
    1e+01      1   0.0013 s
    1e+03      7   0.0018 s
    1e+05     14   0.0024 s
    1e+07     21   0.0031 s
```

Reading the output:
- $c=60$ is $7H+16h$: three bits ($7=111_2$), sixteen table steps, no remainder.
- $c=5000$ is $625H$, ten bits.
- Every value matches scipy to $10^{-14}$ or better.
- A millionfold increase of $c$ costs about 2.4 times the time, not a million times.

**Design choices and pitfalls.**
- `nbits` comes from the largest $c$ in the call. `torch.where` evaluates the composition for every element, so one participant with a huge $c$ at an extreme node makes the whole batch pay `nbits` compositions, and the autograd memory grows with them. The bound on log-speed (§2.31) caps this at about 25 bits.
- For an unstable drift (an eigenvalue with positive real part), $E$ grows with $c$ and can overflow at very large $c$. The evaluation then fails and `value_and_grad` returns its failure value (§2.37). For a marginal or unstable drift with moderate $c$, the composition is exact; no stability is needed.
- Floating point can make `rem` a hair below 0 or `rem / h` a hair above 31. The clamp keeps `ih` in the table, and $\delta$ then lies a hair outside $[0,h)$, where the Taylor series is still accurate.

### 2.18 `lyapunov` — the stationary covariance by a Kronecker solve

[`transitions.py:118`](../../src/pdx/engine/transitions.py#L118)

**What it does.** Solves $A\Sigma+\Sigma A^\top=-Q$ for $\Sigma$ by writing it as one linear system of size $d^2$, and returns the symmetrised solution.

**Inputs → outputs.** `A`, `Q` $(d,d)$ → `X` $(d,d)$.

**The mathematics.**
- *What $\Sigma$ is.* For a stable $A$ (every eigenvalue with negative real part), $\Sigma=\int_0^\infty e^{Aw}Qe^{A^\top w}dw$ is the covariance the process settles into. With $F(w)=e^{Aw}Qe^{A^\top w}$,
  $$A\Sigma+\Sigma A^\top=\int_0^\infty F'(w)\,dw=F(\infty)-F(0)=-Q,$$
  because $F(\infty)=0$ for a stable $A$.
- *The Kronecker form.* With column-major vectorisation, $\mathrm{vec}(AXB)=(B^\top\otimes A)\,\mathrm{vec}X$. So
  $$(I\otimes A+A\otimes I)\,\mathrm{vec}\,\Sigma=-\mathrm{vec}\,Q .$$
- *When it can be solved.* The eigenvalues of $K=I\otimes A+A\otimes I$ are all the sums $\lambda_i+\lambda_j$ of two eigenvalues of $A$. $K$ is invertible unless two eigenvalues sum to zero, and in particular unless $A$ has an eigenvalue 0.

In the code:
- `K = torch.kron(I, A) + torch.kron(A, I)` builds the system matrix.
- `-Q.T.reshape(d*d, 1)` is $-\mathrm{vec}\,Q$: reading $Q^\top$ row by row is reading $Q$ column by column.
- `solve(...)[:, 0].reshape(d, d).T` undoes the vectorisation the same way.
- `0.5 * (X + X.T)` removes rounding asymmetry.

The system has $d^2=100$ unknowns at ten domains, which is cheap. It is solved once per phenotype and chunk, and `torch.linalg.solve` is differentiable.

**Worked example** ([`ex_lyapunov.py`](../guide_examples/s02/ex_lyapunov.py)):

```text
Sigma (Kronecker solve):
 [[0.2   0.08 ]
 [0.08  0.112]]
|Sigma - scipy|: 1.3877787807814457e-17 | residual |A Sigma + Sigma A' + Q|: 6.938893903907228e-18
Kronecker sum K = I (x) A + A (x) I:
 [[-0.5   0.    0.    0.  ]
 [ 0.2  -0.75  0.    0.  ]
 [ 0.2   0.   -0.75  0.  ]
 [ 0.    0.2   0.2  -1.  ]]
eigenvalues of K (all sums of two eigenvalues of A): [-1.   -0.75 -0.75 -0.5 ]
```

By hand: $\Sigma_{00}=q_{00}/(2\times0.25)=0.1/0.5=0.2$. The eigenvalues of $A$ are $-0.25$ and $-0.5$, and their pairwise sums are $-0.5$, $-0.75$ (twice) and $-1$.

**Design choices and pitfalls.** The conditioning of $K$ worsens as the largest eigenvalue of $A$ approaches zero. The condition number is $1.3\times10^3$ at $-6\times10^{-4}$ and $2\times10^{16}$ (singular) at 0 (§2.19). This is why the route is used only for a clearly stable drift.

### 2.19 `is_clearly_stable` — choosing the route for $S$

[`transitions.py:127`](../../src/pdx/engine/transitions.py#L127)

**What it does.** Returns True when the largest real part of the eigenvalues of $A$ is below $-10^{-3}\|A\|_1$.

**Inputs → outputs.** `A` $(d,d)$ → bool. The eigenvalues are computed on `A.detach()`, so the test is a branch with no gradient.

**The mathematics.**
$$\max_j\mathrm{Re}\,\lambda_j(A)<-\texttt{STABLE\_MARGIN}\cdot\max(\|A\|_1,10^{-12}).$$
The test is scale-free: $sA$ passes exactly when $A$ does, for any $s>0$, so it does not depend on the speed. The margin bounds the loss of digits in $S=\Sigma-E\Sigma E^\top$. At small $c$, $S\approx cQ$ while $\|\Sigma\|\approx\|Q\|/(2|\lambda_{max}|)$, so the subtraction loses about $\log_{10}\big(1/(2|\lambda_{max}|c)\big)$ digits. Near the margin, at $c=0.01$, that is about five digits, leaving about 11.

**Worked example** ([`ex_lyapunov.py`](../guide_examples/s02/ex_lyapunov.py)):

```text
is_clearly_stable: largest real eigenvalue < -0.001 * ||A||_1
  stable (-0.25, -0.5)           max eig -0.2500  threshold -5.0e-04  clearly stable: True   cond(K) 2.4e+00
  max eig -6e-4, ||A||_1 = 0.5   max eig -0.0006  threshold -5.0e-04  clearly stable: True   cond(K) 1.3e+03
  max eig -4e-4, ||A||_1 = 0.5   max eig -0.0004  threshold -5.0e-04  clearly stable: False  cond(K) 1.9e+03
  marginal (0, -0.2)             max eig +0.0000  threshold -2.0e-04  clearly stable: False  cond(K) 2.3e+16
  unstable (+0.1, -0.5)          max eig +0.1000  threshold -5.0e-04  clearly stable: False  cond(K) 6.7e+00
  c =   0.01: |S stationary - S composition| = 2.7e-14 (|S| = 0.001)
  c =    1.0: |S stationary - S composition| = 3.7e-14 (|S| = 0.0999)
  c =  100.0: |S stationary - S composition| = 1.0e-12 (|S| = 9.42)
```

Just inside the margin, the stationary route agrees with composition to a relative $3\times10^{-11}$ at $c=0.01$ and $10^{-13}$ at $c=100$.

**Design choices and pitfalls.**
- The unstable example has a well-conditioned $K$ ($6.7$), and the Lyapunov solution exists. It is rejected because $E$ grows with $c$, so $\Sigma-E\Sigma E^\top$ would subtract huge numbers, not because $K$ is singular.
- When the parameters cross the margin during a fit, the route switches. Both routes are exact, so the likelihood changes only in its last digits (about $10^{-11}$, §2.20).

### 2.20 `transitions_grouped` — $E$, $M$, $S$ for every participant, node and interval

[`transitions.py:133`](../../src/pdx/engine/transitions.py#L133)

**What it does.** For each phenotype $k$, it selects that phenotype's participants and chooses the route for $S$:
- **stationary**, if `method == "lyapunov"`, or `"auto"` with a clearly stable $A_k$: $E$ and $M$ from `_grid` without $Q$, $\Sigma$ from `lyapunov`, and $S=\Sigma-E\Sigma E^\top$;
- **composition** otherwise: all three from `_grid` with $Q$.

It writes the results into output tensors with `index_copy`.

**Inputs → outputs.** `A`, `Q` $(G,d,d)$ at the reference speed; `g` $(n)$; `c` $(n,K,T_1)$, which is speed × interval → `E`, `M`, `S` each $(n,K,T_1,d,d)$. In the likelihood $T_1=T-1$, the number of intervals in the chunk.

**The mathematics.** The stationary identity is
$$S(c)=\Sigma-E(c)\,\Sigma\,E(c)^\top .$$
For a stable $A$, split $\Sigma=\int_0^\infty=\int_0^c+\int_c^\infty$. The second part is $e^{cA}\big[\int_0^\infty e^{Av}Qe^{A^\top v}dv\big]e^{cA^\top}=E\Sigma E^\top$.

A second proof shows that the identity needs only a solution of the Lyapunov equation, not stability. Let $F(c)=\Sigma-E\Sigma E^\top$. Then $F(0)=0$, and
$$F'(c)=-E(A\Sigma+\Sigma A^\top)E^\top=EQE^\top=S'(c),$$
so $F=S$. Stability is what makes the identity safe in floating point: $E\to0$ as $c$ grows, so $S$ tends smoothly to $\Sigma$ at any $c$, where an unstable $E$ would make the subtraction blow up.

**Worked example** ([`ex_transitions_grouped.py`](../guide_examples/s02/ex_transitions_grouped.py), second part). Three participants (phenotypes 0, 1, 0), two nodes, two intervals. Phenotype 1's drift is $2A$ with noise $2Q$, so its speed-1 transitions over $c$ must equal phenotype 0's over $2c$:

```text
shapes: (3, 2, 2, 2, 2) (3, 2, 2, 2, 2) (3, 2, 2, 2, 2)
participant 1 (phenotype 1, drift 2A) at c = 0.5 equals phenotype 0 at c = 1.0: 0.0
routes agree: max |S_lyapunov - S_composition| = 1.3877787807814457e-16
marginal drift, c = 0.3, 2, 7.5: max |grouped - Van Loan| = 4.440892098500626e-15
marginal drift, c = 5000: S =
 [[275.1375 274.9125]
 [274.9125 275.0375]]
 max |grouped - transition_np| = 2.852402758435346e-10
```

The marginal drift $\begin{pmatrix}-0.1&0.1\\0.1&-0.1\end{pmatrix}$ has eigenvalues 0 and $-0.2$. Its $\Sigma$ does not exist: along the eigenvector $(1,1)$ the variance grows without bound, about $0.055\,c$ (275 at $c=5000$). Composition handles it, and agrees with the independent numpy route to a relative $10^{-12}$.

**Design choices and pitfalls.**
- `method="lyapunov"` forced on an unstable drift gives the right answer algebraically, as long as $K$ is invertible, but loses accuracy as $E$ grows. The tests force it only on stable drifts.
- The phenotypes are processed one at a time, and the transitions are computed on the reference-speed drift with $c=s\tau$. The phenotype's speed offset $e^{\gamma_g}$ is already inside $s$.

### 2.21 `transitions` — the matrix-exponential version, and why Van Loan breaks at large $c$

[`transitions.py:164`](../../src/pdx/engine/transitions.py#L164)

**What it does.** The check version. For every value of $c$ it computes two $2d\times2d$ matrix exponentials with `torch.linalg.matrix_exp`:
- $\exp\!\big(c\begin{pmatrix}A&I\\0&0\end{pmatrix}\big)$, which gives $E$ and $M$ (§2.13, point 4);
- Van Loan's block $\exp\!\big(c\begin{pmatrix}-A&Q\\0&A^\top\end{pmatrix}\big)$, which gives $S$.

**Inputs → outputs.** `A`, `Q` $(n,d,d)$, one per participant; `sdt` $(n,K,T_1)$ → `E`, `M`, `S` each $(n,K,T_1,d,d)$.

**The mathematics.** For a block upper-triangular matrix,
$$\exp\begin{pmatrix}X&Y\\0&Z\end{pmatrix}=\begin{pmatrix}e^X&\int_0^1e^{X(1-r)}Ye^{Zr}dr\\0&e^Z\end{pmatrix}.$$
With $X=-cA$, $Y=cQ$, $Z=cA^\top$ and the substitution $r'=cr$, the upper-right block is $F_{12}=\int_0^ce^{-A(c-r)}Qe^{A^\top r}dr$ and the lower-right block is $F_{22}=e^{cA^\top}$. Then
$$F_{22}^\top F_{12}=e^{cA}\int_0^ce^{-A(c-r)}Qe^{A^\top r}dr=\int_0^ce^{Ar}Qe^{A^\top r}dr=S(c).$$
The code reads `F2[..., d:, d:]` as $F_{22}$ and `F2[..., :d, d:]` as $F_{12}$.

**Why it breaks at large $c$.** For a stable $A$, the block $e^{-cA}$ grows like $e^{c|\lambda|_{max}}$, where $|\lambda|_{max}$ is the largest decay rate. The computed exponential carries rounding errors relative to its largest entries, about $\epsilon\,e^{c|\lambda|_{max}}$ with $\epsilon\approx10^{-16}$. Multiplying by $F_{22}^\top$, which decays like $e^{-c|\lambda|_{min}}$, leaves an absolute error up to about $\epsilon\,e^{c(|\lambda|_{max}-|\lambda|_{min})}$ in a result $S$ that stays of order $\Sigma$. Whether that loss appears before outright overflow ($c|\lambda|_{max}>709$) depends on the matrix.

**Worked example** ([`ex_van_loan.py`](../guide_examples/s02/ex_van_loan.py)). A dense random stable $4\times4$ drift and the triangular $2\times2$ drift, error of $S$ against the stationary identity computed with scipy:

```text
dense 4 x 4: eigenvalues [-0.935+0.j    -1.362+0.511j -1.362-0.511j -1.401+0.j   ]
      c   |S| (true)   composition   stationary   Van Loan (torch matrix_exp)
      1   4.123e-02     1.0e-16      4.9e-17     6.2e-17
     10   4.742e-02     1.4e-16      6.4e-17     5.4e-16
     30   4.742e-02     1.4e-16      6.4e-17     8.9e-12
     50   4.742e-02     1.3e-16      6.4e-17     4.9e-08
    100   4.742e-02     1.4e-16      6.4e-17     3.8e+02
    200   4.742e-02     1.3e-16      6.4e-17     3.9e+22
    600   4.742e-02     1.4e-16      6.4e-17     nan
   3000   4.742e-02     1.3e-16      6.4e-17     nan
triangular 2 x 2: eigenvalues [-0.5 +0.j -0.25+0.j]
      c   |S| (true)   composition   stationary   Van Loan (torch matrix_exp)
      1   7.869e-02     2.8e-17      6.9e-17     4.3e-18
     10   1.987e-01     1.5e-16      1.4e-17     3.6e-16
     30   2.000e-01     1.5e-16      0.0e+00     7.2e-16
     50   2.000e-01     1.7e-16      0.0e+00     1.1e-15
    100   2.000e-01     1.5e-16      0.0e+00     2.1e-15
    200   2.000e-01     1.7e-16      0.0e+00     4.1e-15
    600   2.000e-01     1.7e-16      0.0e+00     2.1e-14
   3000   2.000e-01     1.7e-16      0.0e+00     nan
```

Reading the output:
- For the dense drift ($|\lambda|_{max}-|\lambda|_{min}=0.47$), Van Loan's $S$ has a relative error of $10^{-6}$ at $c=50$ and is meaningless at $c=100$, where $\epsilon\,e^{0.47\times100}\approx10^{4}$.
- For the triangular drift the rounding errors happen not to be amplified. Van Loan fails only when $e^{0.5c}$ overflows (NaN at $c=3000$, since $0.5\times3000>709$).
- The engine's two routes stay at $10^{-16}$ throughout.

Such values of $c$ occur in practice: a participant at a high quadrature node, running at 10–30 times the reference speed over a two-year gap, has $c$ of 20–60, and the bound of §2.31 allows up to $e^{15}\tau$.

**Design choices and pitfalls.**
- `transitions` is kept only as a check. `test_grouped_transitions_match_matrix_exponential` compares it with the engine for $c\le10$, and the marginal-drift test for $c\le7.5$, where it is accurate.
- The reference likelihood uses scipy's version of the same blocks: Van Loan's $S$ only for drifts that are not stable (§2.38).

### 2.22 `transition_np` — the numpy transition used by the simulators

[`transitions.py:181`](../../src/pdx/engine/transitions.py#L181)

**What it does.** For one speed and one interval in numpy:
- $E$ and $M$ come from one block exponential (scipy's `expm`).
- $S$ comes from the Taylor series at $a=c/2^k$, followed by $k$ doublings: $S(2a)=S(a)+E(a)S(a)E(a)^\top$ and $E(2a)=E(a)^2$.
- The number of doublings $k$ is the smallest that makes $a\|A\|_1\le0.25$.

**Inputs → outputs.** `A`, `Q` $(d,d)$, `s`, `dt` floats → `E`, `M`, `S` $(d,d)$.

**The mathematics.** Scaling and squaring of the composition rule (§2.13, point 5, with $a=b$). With
$$k=\max\Big(0,\Big\lceil\log_2\frac{c\,\|A\|_1}{0.25}\Big\rceil\Big),\qquad a=c/2^k,$$
start from the Taylor values $S(a)$ and $E(a)=e^{aA}$, and apply $k$ times
$$S\leftarrow S+ESE^\top,\qquad E\leftarrow E^2,$$
which doubles the interval each time, $a\to2a\to\dots\to2^ka=c$. It needs no stability of $A$. The Taylor step is twice as long as the engine's, so for $S$ the first omitted term is about $(0.5)^{13}/14!\approx1.4\times10^{-15}$ (with $\|A\|_\infty\approx\|A\|_1$), still negligible.

**Worked example** ([`ex_transition_np.py`](../guide_examples/s02/ex_transition_np.py)):

```text
stable   c =     0.7: doublings k =  1 | max |numpy - engine|: E 1.1e-16  M 2.2e-16  S 5.6e-17
stable   c =       7: doublings k =  4 | max |numpy - engine|: E 5.5e-15  M 8.9e-15  S 5.6e-17
stable   c =    5000: doublings k = 14 | max |numpy - engine|: E 0.0e+00  M 2.9e-15  S 1.4e-16
marginal c =     0.7: doublings k =  0 | max |numpy - engine|: E 2.2e-16  M 0.0e+00  S 2.8e-17
marginal c =       7: doublings k =  3 | max |numpy - engine|: E 1.1e-15  M 2.7e-15  S 6.7e-16
marginal c =    5000: doublings k = 12 | max |numpy - engine|: E 5.2e-13  M 1.3e-09  S 2.9e-10
```

For the marginal drift at $c=5000$, $M$ and $S$ are of order $10^3$, so the gaps are relative errors near $10^{-12}$.

**Design choices and pitfalls.**
- The simulators call it once per simulated transition: family A (`sim/family_a.py:163`) at every step, and `simulate_from_fit` (§3) once per distinct interval of a participant, with a cache.
- It is written independently of `_grid` (scipy's `expm`, a different step and a different order of doubling), so the agreement above is a genuine check of both.

---

### 2.23 Module docstring and constants of `likelihood.py`

[`likelihood.py:1`](../../src/pdx/engine/likelihood.py#L1), [`likelihood.py:28`](../../src/pdx/engine/likelihood.py#L28)

**What it does.**
- The docstring states what is computed: per participant, a Gauss–Hermite sum over log-speed of the exact Kalman-filter likelihood of the visits after the landmark, given the landmark scores, with the load in the filter's state and a plug-in prior for the true landmark state.
- `LOG2PI` is $\log2\pi$.
- `LOG_SPEED_BOUND = 15.0` bounds the log-speed at a node (§2.31).
- `CHUNK = 128` and `CHUNK_SINGLE = 32` are the batch sizes of §2.28.

**Inputs → outputs.** Constants only.

**The mathematics: the likelihood as a product of one-step predictions.** For a linear Gaussian state-space model, the joint density of the later visits factorises into one-step-ahead predictions (the prediction-error decomposition):
$$
p(y_{2:T}\mid y_1,u)=\prod_{k=2}^{T}p(y_k\mid y_{1:k-1},u),\qquad y_k\mid y_{1:k-1},u\sim N\big(H\hat m_{k|k-1}+h_k,\ HP_{k|k-1}H^\top+\Sigma_e\big),
$$
where $\hat m_{k|k-1}$ and $P_{k|k-1}$ are the filter's predicted mean and covariance and $H$ picks the observed components. Each factor contributes
$$-\tfrac12\big(n_k\log2\pi+\log\det S_k+v_k^\top S_k^{-1}v_k\big),$$
with $n_k$ observed components, innovation $v_k=y_k-H\hat m_{k|k-1}-h_k$ and its covariance $S_k$. This is line 183 of the code (§2.34).

**Worked example.** The bound in speed units: $e^{15}\approx3.3\times10^{6}$ and $e^{-15}\approx3.1\times10^{-7}$ times the reference speed. The largest of 12 nodes is $u=5.50$, so with a mean log-speed of 0 the bound starts to bind at $\sigma_g>15/5.50=2.73$ ([`ex_gh_nodes.py`](../guide_examples/s02/ex_gh_nodes.py), last line):

```text
largest node 5.501: the bound |log-speed| <= 15.0 binds at that node once sigma > 2.73 (for a mean log-speed of 0)
```

**Design choices and pitfalls.**
- At $e^{15}$ times the reference speed, any stable process is at its stationary distribution within a day, and at $e^{-15}$ it does not move within a lifetime. Capping the speed there changes the likelihood only in the last digits, as §2.31 measures.
- The bound also keeps $n_H$ of §2.17 well inside 64-bit integers. Without it, a trial point with $\sigma=20$ would give $c\approx e^{110}$.

### 2.24 `Prepared` — the panel as padded tensors

[`likelihood.py:35`](../../src/pdx/engine/likelihood.py#L35)

**What it does.** A dataclass holding everything the likelihood reads from the data, as fixed-size tensors padded to the longest participant ($T$ visits).

| field | shape | content |
|---|---|---|
| `Y` | $(N,T,d)$ | scores, 0 where missing or padded |
| `O` | $(N,T,d)$ | 1 where observed, 0 where missing or padded |
| `dt` | $(N,T-1)$ | interval to the next visit; 0 for padding |
| `dose`, `expo`, `on` | $(N,T)$ | treatment inputs per visit; 0 for padding |
| `z` | $(N,p)$ | covariates |
| `y1c` | $(N,d)$ | centred landmark scores $\tilde y_{i1}$ |
| `g` | $(N)$ | phenotype |
| `mu1`, `S1` | $(G,d)$, $(G,d,d)$ | plug-in prior of the true landmark state |
| `R` | $(d)$ | measurement-error variances |
| `N`, `T` | ints | participants; longest sequence |
| `Ti` | $(N)$ | visits per participant |

**Inputs → outputs.** Built by `prepare` (§2.27).

**The mathematics.** Padding is harmless by construction. A padded interval has $\tau=0$, hence $E=I$, $M=S=0$, so the prediction leaves the state unchanged. A padded visit has $O=0$, so the update adds nothing (§2.34).

**Worked example.** In §2.0's panel, `Y` is $(54,7,4)$ and `dt` is $(54,6)$: 54 participants, at most 7 visits, 4 domains. §2.27 prints every field of a six-participant panel.

**Design choices and pitfalls.** Fixed shapes are what allow one tensor operation per visit for a whole batch of participants and nodes. The cost of padding is reduced by batching participants with similar numbers of visits (§2.29).

### 2.25 `gh_nodes` — Gauss–Hermite nodes and weights for a standard normal

[`likelihood.py:53`](../../src/pdx/engine/likelihood.py#L53)

**What it does.** Returns the $K$ nodes $u_m$ and the logs of the weights $w_m$ of the Gauss–Hermite rule, rescaled for a standard normal variable.

**Inputs → outputs.** `K` → `U` $(K)$, `logw` $(K)$, float64 tensors.

**The mathematics.** numpy's `hermgauss(K)` returns nodes $x_m$ and weights $w^{GH}_m$ for the weight function $e^{-x^2}$:
$$\int_{-\infty}^{\infty}f(x)\,e^{-x^2}dx\approx\sum_{m=1}^Kw^{GH}_mf(x_m),$$
exact when $f$ is a polynomial of degree at most $2K-1$. For $u\sim N(0,1)$, substitute $u=\sqrt2x$:
$$\int f(u)\,\phi(u)\,du=\int f(\sqrt2x)\,\frac{e^{-x^2}}{\sqrt{2\pi}}\sqrt2\,dx=\int f(\sqrt2x)\,\frac{e^{-x^2}}{\sqrt\pi}\,dx\approx\sum_m\frac{w^{GH}_m}{\sqrt\pi}\,f(\sqrt2x_m).$$
So $u_m=\sqrt2x_m$ (`U`) and $w_m=w^{GH}_m/\sqrt\pi$, returned as $\log w_m$ (`logw`), with $\sum_mw_m=1$.

**Worked example** ([`ex_gh_nodes.py`](../guide_examples/s02/ex_gh_nodes.py)):

```text
K = 12 nodes u_m: [-5.5009 -4.2718 -3.2237 -2.2595 -1.3404 -0.4444  0.4444  1.3404  2.2595  3.2237  4.2718  5.5009]
weights w_m     : [0.    0.    0.002 0.029 0.147 0.322 0.322 0.147 0.029 0.002 0.    0.   ]
sum of weights: 1.0000000000000002

 moment E[u^2j]   exact (2j-1)!!   12-node rule
   u^2            1.000000e+00       1.000000e+00
   u^4            3.000000e+00       3.000000e+00
   u^12           1.039500e+04       1.039500e+04
   u^22           1.374931e+10       1.374931e+10
   u^24           3.162341e+11       3.157551e+11

 E[exp(sigma u)] = exp(sigma^2 / 2): relative error of the rule
 sigma      K = 12      K = 20      K = 30
  0.4      5.6e-17     2.8e-16     1.7e-16
  1.0      4.4e-16     4.4e-16     1.1e-16
  2.0      4.6e-09     4.4e-16     2.2e-16
  3.0      2.1e-05     3.6e-12     0.0e+00
```

Reading the output:
- With 12 nodes the rule is exact up to $u^{22}$ (degree $\le23$) and first errs at $u^{24}$.
- For the mean speed $E[e^{\sigma u}]$, 12 nodes are exact at the speed spreads used in the simulators ($\sigma=0.4$ to $0.6$). They lose accuracy only at $\sigma=2$–3, which would mean speeds varying more than a hundredfold between patients.

**Design choices and pitfalls.**
- The nodes are fixed, not adapted to each participant. With many visits, $p(y_i\mid u)$ can be peaked in $u$, and a fixed rule then rests on a few nodes. In the trace of §2.35, four nodes carry 98% of the weight. This is why the engine gate compares 12, 20 and 30 nodes (protocol §16). Adaptive quadrature, with nodes centred and scaled at each participant's mode, would be the remedy if they disagreed.
- Log-weights feed the log-sum-exp of §2.35 directly. The smallest weight at $K=12$ is $1.5\times10^{-7}$, so nothing underflows here.

### 2.26 `_pd` — an eigenvalue floor

[`likelihood.py:58`](../../src/pdx/engine/likelihood.py#L58)

**What it does.** Symmetrises a matrix and raises every eigenvalue below `floor` (default $10^{-3}$) to the floor.

**Inputs → outputs.** `S` $(d,d)$ numpy → $(d,d)$ numpy, symmetric positive definite.

**The mathematics.** With $\tfrac12(S+S^\top)=V\,\mathrm{diag}(w)\,V^\top$, it returns $V\,\mathrm{diag}(\max(w_j,f))\,V^\top$. For symmetric matrices, clipping the eigenvalues is the nearest matrix (in the Frobenius norm) whose eigenvalues are all at least $f$.

**Worked example** ([`ex_prepare.py`](../guide_examples/s02/ex_prepare.py), the `_pd` line). $\begin{pmatrix}1&2\\2&1\end{pmatrix}$ has eigenvalues $-1$ and 3:

```text
_pd on an indefinite matrix: [[1.5005, 1.4995], [1.4995, 1.5005]] eigenvalues [0.001 3.   ]
```

**Design choices and pitfalls.** The floor of $10^{-3}$ is in squared normal-score units, a standard deviation of 0.03. A floored direction makes the prior say that the true landmark state barely varies there (§2.27).

### 2.27 `prepare` — the panel into tensors, the plug-in landmark prior and the centred landmark scores

[`likelihood.py:63`](../../src/pdx/engine/likelihood.py#L63)

**What it does.**
1. Pads every participant to $T$ visits: scores with NaN set to 0, the observed mask, the intervals `np.diff(t)`, dose, exposure and ON state (lines 64–78).
2. Takes the landmark scores $Y_{i1}$ (row 0 of each participant, line 79).
3. For each phenotype $g$ (lines 82–86):
   - $\mu_{1g}$ is the mean of each domain's observed landmark scores (`np.nanmean`);
   - $\hat C_g$ is the covariance of the landmark scores with each pair of domains using the participants observed on both (`np.ma.cov` on masked values), or the identity with two or fewer participants;
   - $S_{1g}$ is `_pd`$(\hat C_g-\Sigma_e)$.
4. Fills each missing landmark score with its phenotype's mean, and centres on the pooled mean over all participants: $\tilde y_{i1}$ (lines 87–88).
5. Converts everything to float64 tensors (lines 89–92).

**Inputs → outputs.** `panel` (§1), `meas_var` $(d)$, `n_groups` → `Prepared` (§2.24).

**The mathematics.** With $O_{j}$ the participants of phenotype $g$ whose landmark score on domain $j$ is observed:
$$\mu_{1g,j}=\frac1{|O_j|}\sum_{i\in O_j}Y_{i1,j},\qquad \hat C_{g,jk}=\frac{\sum_{i\in O_j\cap O_k}(Y_{i1,j}-\mu_{1g,j})(Y_{i1,k}-\mu_{1g,k})}{|O_j\cap O_k|-1},$$
which is what `np.ma.cov` computes: means over all available values, sums and denominators over the pairs available. Then
$$S_{1g}=\mathrm{floor}_{10^{-3}}\big(\hat C_g-\Sigma_e\big),$$
because $\mathrm{Var}(Y_{i1})=\mathrm{Var}(X_{i1})+\Sigma_e$ when the error is independent of the true level (the method of moments). The centred scores are
$$\tilde y_{i1,j}=\check y_{i1,j}-\frac1N\sum_{i'}\check y_{i'1,j},\qquad \check y_{i1,j}=\begin{cases}Y_{i1,j}&\text{observed}\\\mu_{1g(i),j}&\text{missing.}\end{cases}$$

**Worked example** ([`ex_prepare.py`](../guide_examples/s02/ex_prepare.py)). Six hand-made participants, two domains, $\Sigma_e=\mathrm{diag}(0.02,0.05)$. Participant 3 (PIGD) has a missing landmark score and a fully missing third visit; participant 2 has only the landmark:

```text
N, T: 6 4 | Ti: [3, 2, 1, 4, 2, 2]
Y[3] (missing set to 0):
 [[0.6 0. ]
 [0.7 0.9]
 [0.  0. ]
 [1.1 1. ]]
O[3]:
 [[1. 0.]
 [1. 1.]
 [0. 0.]
 [1. 1.]]
dt:
 [[1.  1.1 0. ]
 [1.2 0.  0. ]
 [0.  0.  0. ]
 [0.9 1.1 1.2]
 [1.  0.  0. ]
 [1.1 0.  0. ]]
mu1 (phenotype means of the landmark scores):
 [[0.1  0.1 ]
 [0.6  0.65]]
phenotype 0: pairwise covariance of the landmark scores
 [[ 0.09 -0.03]
 [-0.03  0.04]]
  minus diag(meas_var), floored at 1e-3 (S1[0]):
 [[ 0.0721 -0.0237]
 [-0.0237  0.0089]]
phenotype 1: pairwise covariance of the landmark scores
 [[0.09  0.09 ]
 [0.09  0.045]]
  minus diag(meas_var), floored at 1e-3 (S1[1]):
 [[0.0903 0.0595]
 [0.0595 0.0407]]
y1c (missing landmark score filled by its phenotype mean, then centred on the pooled mean):
 [[-0.25  -0.075]
 [-0.55  -0.275]
 [ 0.05  -0.475]
 [ 0.25   0.275]
 [-0.05   0.125]
 [ 0.55   0.425]]
```

By hand, for PIGD:
- Domain 0 is observed for all three participants (0.6, 0.3, 0.9): mean 0.6, variance $(0+0.09+0.09)/2=0.09$.
- Domain 1 is observed for two (0.5, 0.8): mean 0.65, variance $0.045$.
- Their covariance uses those two: $\big[(-0.3)(-0.15)+(0.3)(0.15)\big]/1=0.09$.

That pairwise matrix implies a correlation of $0.09/\sqrt{0.09\times0.045}=1.41$: impossible, and indefinite. After subtracting $\Sigma_e$ it is floored to the positive definite `S1[1]`. Participant 3's missing domain-1 score becomes 0.65, and the pooled mean of domain 1 is 0.375, so $\tilde y=0.275$.

The end of the script checks two small branches: a batch holding only a landmark-only participant contributes exactly 0, and a phenotype with two or fewer participants gets the identity, less $\Sigma_e$:

```text
landmark-only participant 2 in a batch of its own: [0.0]
S1 with two TD and one PIGD participant (identity minus meas_var):
 [[[0.98 0.  ]
  [0.   0.95]]

 [[0.98 0.  ]
  [0.   0.95]]]
```

**Design choices and pitfalls.**
- *A plug-in prior.* $\mu_{1g}$ and $S_{1g}$ are computed once from the data and treated as known. The fit and the bootstrap simulator (§3) use the same prior, so H-P's calibration is unaffected.
- *Treatment at year 1.* The prior is built from the landmark scores as recorded, which include any treatment effect at year 1, while the update subtracts $h_{i1}$ (§2.30). `docs/code_map.md` records this known approximation; it can bias $\kappa$ and the starting state when many participants are treated by year 1 (§2.41).
- *Pairwise covariances can be indefinite* (the example), hence `_pd`.
- *A floored variance.* If a domain's measurement-error variance exceeds its observed landmark variance in a phenotype, its prior variance is floored at $10^{-3}$. Every participant's true landmark state on that domain is then pulled to the phenotype mean.
- *A domain never observed at the landmark in a phenotype.* `np.nanmean` returns NaN with only a warning, so $\tilde y$ becomes NaN for everyone and every evaluation fails. The fit then "converges" at its start with a log-likelihood of $-10^{10}$ (§2.37, §2.41).
- *Phenotypes with two or fewer participants* get the identity as $\hat C_g$.
- `z` is used as given; centring it is the caller's job (`ppmi.analysis_panel` in §1, the simulators in §6).

### 2.28 `default_chunk` — the batch size by threads

[`likelihood.py:95`](../../src/pdx/engine/likelihood.py#L95)

**What it does.** Returns 32 (`CHUNK_SINGLE`) when PyTorch runs on one thread, otherwise 128 (`CHUNK`).

**Inputs → outputs.** None → int.

**The mathematics.** None: the batch size changes only the order in which participants' terms are added.

**Worked example** ([`ex_chunks.py`](../guide_examples/s02/ex_chunks.py)):

```text
default_chunk with 1 thread: 32
default_chunk with 4 threads: 128
```

**Design choices and pitfalls.**
- The docstring cites the benchmark behind the numbers (`scripts/bench_threads.py`, 4 November 2026, ten domains). One 4-thread process ran 1.7 times faster with batches of 128 than of 512, and four single-threaded jobs ran about 1.9 times faster with 32, because small batches stay in the processor's cache. This guide did not rerun that benchmark, which writes to `results/`.
- The choice is made at call time from `torch.get_num_threads()`. Workers must have their thread count set before they evaluate.

### 2.29 `chunks` — batches of participants with similar numbers of visits

[`likelihood.py:103`](../../src/pdx/engine/likelihood.py#L103)

**What it does.** Sorts the participants by their number of visits (a stable sort) and cuts the sorted list into consecutive batches of at most `size`.

**Inputs → outputs.** `Prepared`, `size` → a list of index tensors.

**The mathematics.** A batch runs its Kalman loop up to its longest participant (`T = max(Ti[idx])`, line 160), so its cost is about $n_c(\max_{i\in c}T_i-1)$ participant-intervals. Contiguous batches of the sorted list make $\max T_i$ close to every member's $T_i$, which reduces that sum to near the actual number of transitions.

**Worked example** ([`ex_chunks.py`](../guide_examples/s02/ex_chunks.py)). The 54 participants of §2.0, batches of 16:

```text
visits per participant (sorted): [2, 2, 3, 3, 4, 4, ..., 7, 7]
  chunk of 16: visits from 2 to 4, loop runs over 3 intervals
  chunk of 16: visits from 4 to 5, loop runs over 4 intervals
  chunk of 16: visits from 5 to 7, loop runs over 6 intervals
  chunk of  6: visits from 7 to 7, loop runs over 6 intervals
participant-intervals computed: sorted 244, unsorted 324, actual transitions 216
chunk   5: loglik -458.183497473848  (difference from one batch +0.0e+00, 0.032 s)
chunk  16: loglik -458.183497473848  (difference from one batch +0.0e+00, 0.015 s)
chunk  32: loglik -458.183497473848  (difference from one batch +5.7e-14, 0.010 s)
chunk 128: loglik -458.183497473848  (difference from one batch +0.0e+00, 0.009 s)
```

Sorting cuts the padded work from 324 to 244 participant-intervals, against 216 actual transitions. The log-likelihood is the same for every batch size, to $6\times10^{-14}$. On a rerun, the times and the last digit of these differences can change (another run gave $+5.7\times10^{-14}$ for batches of 5). Differences at that level are rounding and affect no result.

**Design choices and pitfalls.** The sort is stable, so the batches, and the order of summation, are reproducible.

### 2.30 `_landmark_update` — the posterior of the true landmark state

[`likelihood.py:110`](../../src/pdx/engine/likelihood.py#L110)

**What it does.** Updates the plug-in prior $N(\mu_{1g},S_{1g})$ of $X_i(t_{i1})$ with the observed components of the landmark scores, net of the treatment term. It returns the posterior mean $m_1$ and covariance $P_1$. No likelihood term is computed, because the likelihood is conditional on the landmark scores.

**Inputs → outputs.** `mu` $(n,d)$, `S` $(n,d,d)$, `y` $(n,d)$, `o` $(n,d)$, `h` $(n,d)$, `R` $(d)$ → `m` $(n,d)$, `P` $(n,d,d)$.

**The mathematics.** With $H_o$ the rows of $I_d$ for the observed components, the standard Gaussian update is
$$K=SH_o^\top\big(H_oSH_o^\top+\Sigma_{e,oo}\big)^{-1},\qquad m=\mu+K\big(y_o-\mu_o-h_o\big),\qquad P=S-KH_oS .$$
The code keeps full $d\times d$ shapes and masks instead of selecting. With $M_o=\mathrm{diag}(o)$:
$$v=(y-\mu-h)\odot o,\qquad S_v=M_o(S+\Sigma_e)M_o+(I-M_o),\qquad K=SM_oS_v^{-1}.$$
After reordering the components into observed and missing, $S_v=\begin{pmatrix}S_{oo}+\Sigma_{e,oo}&0\\0&I\end{pmatrix}$, so $S_v^{-1}$ is block-diagonal too. $SM_o$ has zero columns for the missing components, so $K=\big[S_{\cdot o}(S_{oo}+\Sigma_{e,oo})^{-1},\ 0\big]$, the standard gain with zeros added. The innovation is zero on the missing components. Hence $m$ and $P=S-KM_oS$ are exactly the standard ones.

| formula | code |
|---|---|
| $M_o$ | `Mo = torch.diag_embed(o)` |
| $v$ | `v = (y - mu - h) * o` |
| $S_v$ | `Sv = Mo @ (S + diag(R)) @ Mo + (I - Mo)` |
| $K=SM_oS_v^{-1}$ | `solve(Sv, (S @ Mo)^T)^T`: $S_v^{-1}M_oS$, transposed |
| $m$, $P$ | `mu + K v`, `S - K (Mo S)`, symmetrised |

**Worked example** ([`ex_landmark.py`](../guide_examples/s02/ex_landmark.py)). Three domains, domain 1 missing, a treatment term on domain 0:

```text
masked form:   m = [0.6731 0.6623 0.2445]
 P =
 [[0.0187 0.0062 0.0005]
 [0.0062 0.2187 0.0008]
 [0.0005 0.0008 0.0331]]
explicit form: m = [0.6731 0.6623 0.2445]
 P =
 [[0.0187 0.0062 0.0005]
 [0.0062 0.2187 0.0008]
 [0.0005 0.0008 0.0331]]
max |differences|: m 0.0e+00, P 3.5e-18
domain 1 was not observed, yet its mean moved from 0.5 to 0.6623 through its prior covariance
```

**Design choices and pitfalls.**
- Masking keeps every participant's matrices the same size in a batch, whatever their pattern of missing values. Selecting the observed components would give a different size for each participant.
- A participant with no landmark score at all keeps the prior ($K=0$).
- Treatment at the landmark is subtracted ($y-h$), but the prior's moments include it (§2.27).

### 2.31 `participant_loglik`, lines 123–152 — parameters, speed at the nodes, load prior, treatment and landmark

[`likelihood.py:123`](../../src/pdx/engine/likelihood.py#L123)

**What it does.** Opens the per-batch likelihood. It unpacks the parameters each participant needs, evaluates the speed at every node, forms the load prior, computes the treatment terms and the posterior of the true landmark state.

**Inputs → outputs.** `x` (tensor), `Layout`, `Prepared`, `idx` $(n)$ → at the end of the function, $(n)$ log-likelihood contributions. The quantities built in this block:

| quantity | code | shape |
|---|---|---|
| $u_m$, $\log w_m$ | `U`, `logw` | $(K)$ |
| $A_g$, $Q_g$ for all phenotypes | `AG`, `QG` | $(G,d,d)$ |
| $b_{g(i)}$; $\sigma_{g(i)}$ | `b`; `sig` | $(n,d)$; $(n)$ |
| $\eta_i=\gamma_{g(i)}+\beta^\top z_i+\theta^\top\tilde y_{i1}$ | `eta` | $(n)$ |
| $s_{im}$ | `s` | $(n,K)$ |
| $\mu_{\ell i}=\gamma^y_{g(i)}\odot\tilde y_{i1}+\Gamma^z_{g(i)}z_i$ | `mu_l` | $(n,d)$ |
| $\lambda_{g(i)}$; $\tau^2_{g(i)}$ | `lam`; `tau2` | $(n,d)$ |
| $h_{ik}$ | `h` | $(n,T,d)$ |
| $m_1$, $P_1$ | `m1`, `P1` | $(n,d)$, $(n,d,d)$ |

**The mathematics.**
$$s_{im}=\exp\big\{\min\big(15,\max(-15,\ \eta_i+\sigma_{g(i)}u_m)\big)\big\},\qquad \ell_i\mid u_m\sim N\big(\mu_{\ell i}+\lambda_{g(i)}u_m,\ \mathrm{diag}(\tau^2_{g(i)})\big).$$
`torch.einsum("np,ndp->nd", z, gam_z[g])` computes $\Gamma^z_{g(i)}z_i$ for every participant at once. `y1c * gam_y[g]` is an elementwise product, so $\Gamma^y$ is diagonal. `torch.exp(2 * log_tau)` is $\tau^2$.

**Worked example** ([`ex_participant_trace.py`](../guide_examples/s02/ex_participant_trace.py), block [2.31]). A PIGD participant with five visits and two kinds of gap (a missing tremor score at visit 1, a fully missing visit 2), at a perturbed start value of the null model:

```text
participant 0: phenotype 1, visit times [1.029 2.177 3.22  4.186 5.128]
scores (NaN = missing):
 [[ 0.68  -0.536  0.101 -0.237]
 [   nan -0.587  0.238 -0.181]
 [   nan    nan    nan    nan]
 [-0.01   0.217 -0.322 -1.502]
 [ 0.341  0.185  0.652 -0.36 ]]

[2.31] parameters, speed at the nodes, load prior, treatment, landmark
gamma_g = -0.3011, beta'z = 0.0164, theta'y1c = -0.0026 -> eta = -0.2873; sigma_g = 0.2728
speeds s_m at the 12 nodes: [0.167 0.234 0.311 0.405 0.521 0.665 0.847 1.082 1.39  1.808 2.406 3.365]
load prior: mean without the speed part [ 0.0554 -0.0239 -0.0227  0.0229] | lambda_g [ 0.0334 -0.017   0.0526 -0.0003] | tau_g [0.0903 0.0985 0.0956 0.1008]
dose [0. 0. 0. 0. 3.] | exposure [0.  0.  0.  0.  1.5] | ON [0. 0. 0. 0. 0.]
treatment terms h (visits x domains):
 [[-0.     -0.      0.      0.    ]
 [-0.     -0.      0.      0.    ]
 [-0.     -0.      0.      0.    ]
 [-0.     -0.      0.      0.    ]
 [-0.0117 -0.0087  0.      0.    ]]
true landmark state: prior mean [-0.029  0.468  0.222  0.189] -> posterior mean [ 0.644 -0.504  0.091 -0.222]
                     prior sd   [0.519 0.666 0.459 0.756] -> posterior sd   [0.134 0.136 0.132 0.138]
```

Reading the output:
- $\eta=-0.3011+0.0164-0.0026=-0.2873$; the speeds run from $e^{-0.2873-0.2728\times5.50}=0.167$ to $3.365$.
- The participant starts treatment before visit 4, examined OFF, so only the long-duration term acts: $-\kappa^{\mathrm{LDR}}\times1.5/4.5$.
- The landmark scores, measured with error variance 0.02 (sd 0.141), pull the true state almost onto them: posterior sd about 0.135.

The bound on log-speed, checked against the reference implementation, which has no bound ([`ex_note_reference_wild.py`](../guide_examples/s02/ex_note_reference_wild.py)):

```text
sigma =  0.30: engine -268.76181673  reference -268.76181673  difference +0.0e+00
sigma =  2.72: engine -303.79927002  reference -303.79927002  difference +0.0e+00
sigma =  4.48: engine -353.25442628  reference -353.25442628  difference +3.8e-10
sigma =  7.39: engine -374.41610021  reference -374.41610069  difference +4.9e-07
```

Even at $\sigma=7.4$, where the four outer nodes on each side are capped ($7.39\times2.26>15$), the two differ by $5\times10^{-7}$.

**Design choices and pitfalls.**
- The derivative of the clamp is zero where it binds, so capped nodes contribute no gradient with respect to $\eta$ and $\sigma$. This matters only at trial points far from any fitted value.
- $\tilde y_{i1}$ is the observed landmark score, not the true state (§2.0, §2.41).

### 2.32 `participant_loglik`, lines 153–165 — the filter state and the transitions

[`likelihood.py:153`](../../src/pdx/engine/likelihood.py#L153)

**What it does.**
- It sets up, for every participant and node, the mean and covariance of the stacked state $[X;\ell]$, and a running log-likelihood `ll` of zeros $(n,K)$.
- It finds the batch's longest sequence `T`, and returns early (with $\log\sum_mw_m=0$) when no one in the batch has a visit after the landmark.
- It computes $E$, $M$, $S$ for all participants, nodes and intervals in one call, and multiplies $S$ by the era factor.

**Inputs → outputs.** The quantities of §2.31 → `mX` $(n,K,d)$, `ml` $(n,K,d)$, `Pxx`, `Pxl`, `Pll` $(n,K,d,d)$, and `E`, `M`, `S` $(n,K,T-1,d,d)$.

**The mathematics.** Given $u_m$, the true landmark state and the load are independent (the landmark posterior does not involve $u$), so
$$\begin{pmatrix}X\\\ell\end{pmatrix}\sim N\left(\begin{pmatrix}m_1\\\mu_{\ell i}+\lambda_gu_m\end{pmatrix},\ \begin{pmatrix}P_1&0\\0&\mathrm{diag}\,\tau_g^2\end{pmatrix}\right),$$
which is `mX`, `ml`, `Pxx`, `Pxl` $=0$ and `Pll`. Only `ml` depends on the node at this point; `expand` repeats the rest across nodes without copying. The transitions use $c_{imk}=s_{im}\tau_{ik}$, computed as `s[:, :, None] * prep.dt[idx][:, None, :T-1]`, of shape $(n,K,T-1)$. Since $S$ is linear in $Q$, scaling $Q_g$ by $e^{\omega\,\mathrm{era}_i}$ scales $S$ by the same factor (line 165).

**Worked example** ([`ex_participant_trace.py`](../guide_examples/s02/ex_participant_trace.py), block [2.32]):

```text
intervals dt [1.148 1.043 0.967 0.941] | c = s dt at node 6 [0.973 0.883 0.819 0.797]
E, M, S: (1, 12, 4, 4, 4) | era factor exp(omega * era_i) = 0.9494
load prior mean at node 6 (mean + lambda u_6): [ 0.0702 -0.0314  0.0007  0.0228] (u_6 = 0.4444)
```

Node 6 (counting from 0) has $u=0.444$ and speed 0.847, so each interval of about one year becomes $c\approx0.8$–$1.0$.

**Design choices and pitfalls.**
- Computing every transition before the loop means one call per batch. The fine table and the Lyapunov solve are made once per phenotype, and every $(n,K,T-1)$ value is assembled from them (§2.17).
- The memory for $E$, $M$, $S$ is $3\cdot n\cdot K\cdot(T-1)\cdot d^2$ numbers, plus the autograd record of the binary powers. At $n=128$, $K=12$, $T-1=9$, $d=10$ that is about 33 MB for the three tensors alone, one reason for the small batches.

### 2.33 `participant_loglik`, lines 166–175 — the prediction step

[`likelihood.py:168`](../../src/pdx/engine/likelihood.py#L168)

**What it does.** Moves the stacked state from one visit to the next: $X\leftarrow EX+M(b+\ell)+\text{noise}$, $\ell\leftarrow\ell$. The covariance is updated block by block.

**Inputs → outputs.** `mX`, `ml`, `Pxx`, `Pxl`, `Pll` and the interval's `Et`, `Mt`, `St` $(n,K,d,d)$ → the predicted `mX`, `Pxx`, `Pxl`. The load's mean and covariance do not change in a prediction.

**The mathematics.** In stacked form, $[X;\ell]'=F[X;\ell]+c+w$ with
$$F=\begin{pmatrix}E&M\\0&I\end{pmatrix},\qquad c=\begin{pmatrix}Mb\\0\end{pmatrix},\qquad \mathrm{Cov}(w)=\begin{pmatrix}S&0\\0&0\end{pmatrix}.$$
The predicted mean is $X'=E\,m_X+M(b+m_\ell)$. The predicted covariance is $FPF^\top+\mathrm{Cov}(w)$. Multiplying out with $P=\begin{pmatrix}P_{xx}&P_{x\ell}\\P_{x\ell}^\top&P_{\ell\ell}\end{pmatrix}$:
$$P_{xx}'=EP_{xx}E^\top+EP_{x\ell}M^\top+MP_{x\ell}^\top E^\top+MP_{\ell\ell}M^\top+S,\qquad P_{x\ell}'=EP_{x\ell}+MP_{\ell\ell},\qquad P_{\ell\ell}'=P_{\ell\ell}.$$
The code computes the shared products once, with $D=EP_{x\ell}$ and $C=D+MP_{\ell\ell}$:
$$P_{xx}'=EP_{xx}E^\top+CM^\top+MD^\top+S .$$
Expanding $CM^\top+MD^\top=EP_{x\ell}M^\top+MP_{\ell\ell}M^\top+MP_{x\ell}^\top E^\top$ gives the three middle terms. And $P_{x\ell}'=C$.

| formula | code |
|---|---|
| $E\,m_X+M(b+m_\ell)$ | `(Et @ mX[..., None])[..., 0] + (Mt @ (b[:, None, :] + ml)[..., None])[..., 0]` |
| $D=EP_{x\ell}$ | `D = bmm(Et, Pxl)` |
| $C=D+MP_{\ell\ell}$ | `C = D + bmm(Mt, Pll)` |
| $P_{xx}'$ | `bmm(bmm(Et, Pxx), tr(Et)) + bmm(C, tr(Mt)) + bmm(Mt, tr(D)) + St` |
| $P_{x\ell}'$ | `Pxl = C` |

**Worked example** ([`ex_participant_trace.py`](../guide_examples/s02/ex_participant_trace.py)). At the first visit after the landmark, the code's block formulas are compared with the full $2d\times2d$ product $FPF^\top+\mathrm{Cov}(w)$:

```text
 check against F P F' + [S 0; 0 0]: max |difference| in mean 4.4e-16, Pxx 5.6e-17, Pxl 0.0e+00
```

**Design choices and pitfalls.**
- Working in blocks avoids building and multiplying $2d\times2d$ matrices, which would double the size of every product, and lets the load's covariance stay unchanged without being recomputed.
- $P_{x\ell}$ starts at zero but becomes $MP_{\ell\ell}$ after the first prediction. From then on, every observed change of the scores also tells the filter something about the participant's load.

### 2.34 `participant_loglik`, lines 176–193 — the masked update and the likelihood term

[`likelihood.py:176`](../../src/pdx/engine/likelihood.py#L176)

**What it does.** At each visit after the landmark, for every participant and node, it does three things:
1. It forms the innovation of the observed components and its covariance, with the same mask trick as §2.30.
2. It adds the Gaussian log-density of the innovation to `ll`.
3. It updates the mean and covariance of $[X;\ell]$.

**Inputs → outputs.** The predicted state, `Y[:, t]`, `O[:, t]`, `h[:, t]`, $\Sigma_e$ → the updated state and `ll` $(n,K)$.

**The mathematics.** The observation picks $X$ out of the stacked state: $H=[I_d\ \ 0]$, so $HPH^\top=P_{xx}$ and $HP=[P_{xx}\ \ P_{x\ell}]$. With the mask $M_o=\mathrm{diag}(o_t)$:
$$v=(y_t-m_X-h_t)\odot o_t,\qquad S_v=(P_{xx}+\Sigma_e)\odot(o_to_t^\top)+\mathrm{diag}(1-o_t).$$
The elementwise product with $o_to_t^\top$ is $M_o(\cdot)M_o$. As in §2.30, $S_v$ is block-diagonal, $(P_{xx}+\Sigma_e)_{oo}$ and $I$, after reordering. Therefore
$$\log\det S_v=\log\det(P_{xx}+\Sigma_e)_{oo},\qquad v^\top S_v^{-1}v=v_o^\top(P_{xx}+\Sigma_e)_{oo}^{-1}v_o,$$
and the term
$$\Delta\ell\ell=-\tfrac12\big(n_o\log2\pi+\log\det S_v+v^\top S_v^{-1}v\big)$$
is exactly the log-density of the observed components (§2.23). The gain and the update, in blocks:
$$\begin{pmatrix}K_x\\K_\ell\end{pmatrix}=\begin{pmatrix}P_{xx}\\P_{x\ell}^\top\end{pmatrix}M_oS_v^{-1},\qquad m_X\mathrel{+}=K_xv,\quad m_\ell\mathrel{+}=K_\ell v,$$
$$P_{xx}\mathrel{-}=K_xM_oP_{xx},\qquad P_{x\ell}\mathrel{-}=K_xM_oP_{x\ell},\qquad P_{\ell\ell}\mathrel{-}=K_\ell M_oP_{x\ell}.$$

| formula | code |
|---|---|
| $o_t$ | `o = O[:, t][:, None, :]`, shape $(n,1,d)$, broadcast over nodes |
| $v$, $S_v$ | `v`, `Sv` (lines 178–179) |
| $S_v=LL^\top$ | `Lc = torch.linalg.cholesky(Sv)` |
| $S_v^{-1}v$ | `sol = torch.cholesky_solve(v[..., None], Lc)` |
| $\log\det S_v$ | `2 * log(diag(Lc)).sum(-1)` |
| $n_o$ | `o.sum(-1)` |
| $M_o[P_{xx}\ \ P_{x\ell}]$ | `Bt = o[..., :, None] * cat([Pxx, Pxl], -1)`, shape $(n,K,d,2d)$ |
| $S_v^{-1}M_o[P_{xx}\ \ P_{x\ell}]=[K_x^\top\ \ K_\ell^\top]$ | `KT = torch.cholesky_solve(Bt, Lc)` |
| $K_x$, $K_\ell$ | `tr(KT[..., :d])`, `tr(KT[..., d:])` |
| covariance updates | `Pxx - bmm(Kx, Bt[..., :d])`, `Pll - bmm(Kl, Bt[..., d:])`, `Pxl - bmm(Kx, Bt[..., d:])` |

One Cholesky factorisation serves three purposes: the log-determinant (twice the sum of the logs of its diagonal), the quadratic form, and the gain (two triangular solves). A single `cholesky_solve` on the $d\times2d$ block `Bt` gives both gains at once.

**Worked example** ([`ex_participant_trace.py`](../guide_examples/s02/ex_participant_trace.py), block [2.33–2.34], node 6):

```text
 visit 1: 3 of 4 observed | innovation [-0.    -0.545 -0.268  0.067] | log-density term  -1.4504
 visit 2: 0 of 4 observed | innovation [-0.  0. -0.  0.] | log-density term  -0.0000
 visit 3: 4 of 4 observed | innovation [-0.813 -0.015 -1.322 -1.393] | log-density term  -7.7534
 visit 4: 4 of 4 observed | innovation [-0.027 -0.341  0.715  1.208] | log-density term -10.2546
load at node 6 after the last visit: mean [-0.0425 -0.1053  0.0127  0.0299] sd [0.0792 0.08   0.0825 0.0872]
```

Reading the output:
- At visit 1 the missing tremor score has a zero innovation and adds nothing.
- Visit 2 is entirely missing, so its term is exactly zero, and the prediction to visit 3 then spans two intervals.
- Over the visits, the load's standard deviation shrinks from its prior (about 0.1) to about 0.08: the trajectory carries some information about the participant's own load.

**Design choices and pitfalls.**
- The covariance update uses the standard form $P-KHP$, symmetrised afterwards (lines 192–193), not the Joseph form. That is accurate enough in float64, as the agreement with the reference shows (§2.39).
- If $S_v$ is not positive definite, the Cholesky factorisation raises an error. That can happen only with no measurement error and a degenerate prediction, or at absurd parameters. `value_and_grad` catches it (§2.37).
- `o` has shape $(n,1,d)$: the mask is the same for every node, while the innovations differ by node.

### 2.35 `participant_loglik`, line 194 — the quadrature in log space

[`likelihood.py:194`](../../src/pdx/engine/likelihood.py#L194)

**What it does.** Combines the nodes: $\log L_i=\log\sum_mw_m\exp(\ell\ell_{im})$, computed with `torch.logsumexp` (also on line 162 for the early return).

**Inputs → outputs.** `ll` $(n,K)$, `logw` $(K)$ → $(n)$.

**The mathematics.** With $a_m=\log w_m+\ell\ell_{im}$ and $a^*=\max_ma_m$,
$$\log\sum_me^{a_m}=a^*+\log\sum_me^{a_m-a^*}.$$
Every exponent on the right is $\le0$ and at least one is 0, so nothing overflows or underflows to a meaningless value. A direct sum fails as soon as every $\ell\ell_{im}$ is below about $-745$, where $e^{x}$ underflows to 0 in float64. One participant can reach that at a poor trial point, where an innovation of many standard deviations costs hundreds of units of log-density. The derivative of log-sum-exp is the softmax: $\partial\log L_i/\partial\ell\ell_{im}=w_mp_{im}/\sum_{m'}w_{m'}p_{im'}$, the posterior weight of node $m$ given the participant's data.

**Worked example** ([`ex_participant_trace.py`](../guide_examples/s02/ex_participant_trace.py), block [2.35]):

```text
log p(y_2..T | y_1, u_m): [-28.455 -25.503 -23.289 -21.613 -20.425 -19.71  -19.458 -19.681 -20.441 -21.92  -24.533 -29.229]
posterior weight of each node, w_m p_m / sum: [0.     0.     0.0001 0.0044 0.0736 0.3293 0.4233 0.1548 0.0144 0.0002 0.     0.    ]
log L_i (logsumexp)        = -19.7329913459
participant_loglik         = -19.7329913459
reference_loglik           = -19.7329913459
log of the plain sum of w_m exp(ll_m) = -19.7329913459
the plain sum if the terms were 50 times larger: 0.0 ; logsumexp: -974.0586
```

Reading the output:
- The data favour nodes 5–7 ($u$ between $-0.44$ and $1.34$): this participant runs at about the phenotype's typical speed.
- The replayed lines, the function itself and the independent reference give the same number to 10 digits.
- The last line multiplies the terms by 50: the plain sum underflows to 0 (a log-likelihood of $-\infty$), while log-sum-exp still returns $-974.06$.

**Design choices and pitfalls.** Four nodes carry 98% of the weight here. With longer follow-up the posterior of $u$ narrows, which is the reason for the 12/20/30-node comparison at the engine gate (§2.25).

### 2.36 `loglik` — the total, as one differentiable graph

[`likelihood.py:197`](../../src/pdx/engine/likelihood.py#L197)

**What it does.** Sums `participant_loglik` over the batches of `chunks` and returns a scalar tensor, keeping the whole autograd graph.

**Inputs → outputs.** `x` tensor, `Layout`, `Prepared`, `chunk` → scalar tensor.

**The mathematics.** $\log L=\sum_{\text{batches}}\sum_{i\in\text{batch}}\log L_i$.

**Worked example** ([`ex_loglik_vag.py`](../guide_examples/s02/ex_loglik_vag.py)):

```text
loglik = -249.3025452488; sum of one-participant calls = -249.3025452488
```

**Design choices and pitfalls.** It is meant for tests and small data. With a tensor that requires gradients, the graph of every batch is kept until `backward`, so memory grows with the panel. Fitting uses `value_and_grad` instead.

### 2.37 `value_and_grad` — the negative log-likelihood and its gradient, batch by batch

[`likelihood.py:205`](../../src/pdx/engine/likelihood.py#L205)

**What it does.**
- It turns the numpy vector into a tensor that records gradients.
- For each batch it computes the batch's negative log-likelihood, checks that it is finite, takes its gradient with `torch.autograd.grad`, and adds both to running totals. Each batch's graph is freed before the next.
- An optional `penalty(x)` (used by the penalised localisation fits, §5) is added with its gradient.
- If anything fails it returns `(1e10, zeros)`: a non-finite value or gradient, or a `RuntimeError` (which includes PyTorch's linear-algebra errors), `ValueError` (a non-finite interval, §2.17) or `ArithmeticError`.

**Inputs → outputs.** `xnp` $(n)$ numpy, `Layout`, `Prepared`, `chunk`, `penalty` → (float, $(n)$ numpy), the pair that scipy's L-BFGS-B expects with `jac=True`.

**The mathematics.** The gradient of a sum is the sum of the gradients, $\nabla_x\sum_cf_c(x)=\sum_c\nabla_xf_c(x)$, so batch-by-batch backward passes give the exact total gradient with memory bounded by one batch. Reverse-mode automatic differentiation goes through every operation of §2.31–2.35, including the binary powers, the Taylor series, the Lyapunov solve and the Cholesky factorisations. The gradient is exact up to rounding, not a finite-difference approximation.

**Worked example** ([`ex_loglik_vag.py`](../guide_examples/s02/ex_loglik_vag.py)). Sixteen coordinates, one from each kind of block, against central differences with step $10^{-6}$; then the penalty and the failure value:

```text
value_and_grad: value = 249.3025452488 (= -loglik), gradient of length 112

 coordinate        autograd       central difference (h = 1e-6)   relative gap
 A[2]               -7.96837034       -7.96837033                1.7e-09
 A[7]                8.03187309        8.03187314                5.8e-09
 gamma[0]           -0.04742881       -0.04742880                8.8e-09
 beta[1]           -12.89242071      -12.89242073                9.9e-10
 theta[0]            0.53716454        0.53716454                1.0e-09
 log_sigma[1]       -2.32444889       -2.32444890                5.6e-09
 b[3]               49.68216344       49.68216345                1.6e-10
 L[0]               -7.07492912       -7.07492913                1.8e-09
 L[13]              16.97079378       16.97079378                2.2e-10
 gam_y[2]            0.11499009        0.11499012                2.6e-08
 gam_z[5]           19.30148341       19.30148341                2.8e-10
 lam[1]             -2.30370378       -2.30370378                1.8e-09
 log_tau[4]         -4.31230275       -4.31230274                3.0e-09
 omega[0]           -6.85624136       -6.85624136                8.6e-10
 kappa[0]          -13.18692789      -13.18692787                9.5e-10
 kappa_ldr[3]       11.36090187       11.36090187                1.6e-10

with penalty 0.5 |x|^2: value rises by 34.063104 (0.5 |x|^2 = 34.063104); max |gradient change - x| = 6.6e-15
at a point where the model cannot be evaluated: value 1e+10, |gradient| 0.0
```

Every relative gap is below $3\times10^{-8}$, the size of the finite-difference error itself.

**Design choices and pitfalls.**
- *The failure value can look like convergence.* A large value with a zero gradient makes a line search back off from a bad trial point. But at the start point, a zero gradient satisfies L-BFGS-B's stopping rule: `fit_model` then returns `success=True` with a log-likelihood of $-10^{10}$ after four evaluations ([`ex_note_sentinel.py`](../guide_examples/s02/ex_note_sentinel.py)). The callers in `hp.py` and `gate.py` do not check for it (§2.41).
- *The cost per batch.* The parameters are unpacked and the transition tables rebuilt for every batch, so smaller batches repeat that fixed cost more often.
- *Hessians.* `fit.hessian` (§3) differentiates this gradient by central differences; protocol §7.1 speaks of second derivatives by automatic differentiation (§2.41).

---

### 2.38 Module docstring and `_van_loan` — the reference implementation's transitions

[`reference.py:1`](../../src/pdx/engine/reference.py#L1), [`reference.py:19`](../../src/pdx/engine/reference.py#L19)

**What it does.** The docstring states the purpose: a slow implementation of the same log-likelihood, written separately from the engine. It loops over participants, nodes and visits, uses scipy's matrix exponential with Van Loan's blocks instead of the engine's transition code, and selects the observed components explicitly instead of masking. The engine gate (protocol §16) requires the two to agree. `_van_loan(A, Q, c)` returns:
- $E$ and $M$ from scipy's `expm` of the block $\begin{pmatrix}cA&cI\\0&0\end{pmatrix}$ (§2.13, point 4);
- $S$ from the stationary covariance, with scipy's `solve_continuous_lyapunov`, when the largest real eigenvalue of $A$ is below $-10^{-6}$;
- otherwise $S$ from Van Loan's block (§2.21).

**Inputs → outputs.** `A`, `Q` $(d,d)$ numpy, `c` float → `E`, `M`, `S` $(d,d)$.

**The mathematics.** Those of §2.13, §2.20 and §2.21. scipy's Lyapunov solver uses a Schur decomposition (the Bartels–Stewart method), not a Kronecker system, and its `expm` uses a Padé approximation with scaling and squaring, not composition. So every step differs in method from the engine's.

**Worked example** ([`ex_reference.py`](../guide_examples/s02/ex_reference.py), first part):

```text
stable   (stationary    ) c =  0.5: max |reference - engine|: E 3.3e-16  M 1.7e-16  S 4.2e-17
stable   (stationary    ) c =  5.0: max |reference - engine|: E 1.4e-15  M 2.3e-15  S 4.2e-17
stable   (stationary    ) c = 50.0: max |reference - engine|: E 1.4e-20  M 3.1e-15  S 1.4e-17
marginal (Van Loan block) c =  0.5: max |reference - engine|: E 1.1e-16  M 1.1e-16  S 1.4e-17
marginal (Van Loan block) c =  5.0: max |reference - engine|: E 7.8e-16  M 8.9e-16  S 3.3e-16
marginal (Van Loan block) c = 50.0: max |reference - engine|: E 5.2e-15  M 1.4e-13  S 1.8e-13
```

**Design choices and pitfalls.**
- The reference's stability threshold (an absolute $-10^{-6}$) differs from the engine's (relative, $-10^{-3}\|A\|_1$). They need not match: both routes are exact.
- For a drift that is not stable, the reference falls back on Van Loan's block, which is accurate only for moderate $c$ (§2.21). So it cannot check the engine for an unstable drift at large $c$.

### 2.39 `reference_loglik` — the slow, independent likelihood

[`reference.py:39`](../../src/pdx/engine/reference.py#L39)

**What it does.** Computes $\sum_i\log L_i$ with plain loops and full matrices:
1. `unpack_numpy` gives the named parameters; `hermgauss` gives the nodes and log-weights.
2. For each participant it computes $\eta_i$, the load mean $\mu_{\ell i}$, the era factor, and the treatment terms (re-implemented, lines 57–61). It updates the landmark prior with the observed components only, selected by $H_o$ and using `np.linalg.inv`.
3. For each node: $s=e^{\eta_i+\sigma_gu_m}$ (no bound), $m=[m_1;\ \mu_{\ell i}+\lambda_gu_m]$, $P=\mathrm{blockdiag}(P_1,\mathrm{diag}\,\tau_g^2)$.
4. For each visit after the landmark:
   - prediction with the full $2d\times2d$ matrices: $m\leftarrow Fm+[Mb;0]$ and $P\leftarrow FPF^\top$, then $S$ is added to the top-left block;
   - if any component is observed, $H=[I_o\ \ 0]$, $v=y_o-Hm-h_o$ and $S_v=HPH^\top+\Sigma_{e,oo}$;
   - the log-density is added with `slogdet` and `inv`, and the standard update $m\mathrel{+}=Kv$, $P\mathrel{-}=KHP$ applied.
5. scipy's `logsumexp` over the nodes, and the sum over participants.

**Inputs → outputs.** `x` numpy, `Layout`, `panel`, `mu1` $(G,d)$, `S1` $(G,d,d)$, `y1c` $(N,d)$ (the last three from `prepare`) → float.

**The mathematics.** The likelihood of §2.0, written in the textbook form of a Kalman filter on $[X;\ell]$. What is, and is not, independent of the engine:

| piece | engine | reference | independent? |
|---|---|---|---|
| transitions | composition, Kronecker Lyapunov (torch) | scipy `expm`, scipy Lyapunov, Van Loan | yes |
| prediction | covariance blocks (§2.33) | full $FPF^\top$ | yes |
| missing values | masks (§2.30, §2.34) | explicit selection | yes |
| treatment terms | `treat_effect` | re-written | yes |
| landmark update | masked, `torch.linalg.solve` | selected, `np.linalg.inv` | yes |
| quadrature | `gh_nodes`, `torch.logsumexp` | `hermgauss`, scipy `logsumexp` | yes |
| speed bound | clamp at ±15 | none | they differ (§2.31) |
| batching | by number of visits | none | — |
| parameter transforms | `drifts`, `noise_cov`, `log_speeds` | the same functions, through `unpack_numpy` | **no** |
| landmark prior, $\tilde y_{i1}$ | `prepare` | passed in from `prepare` | **no** |

**Worked example** ([`ex_reference.py`](../guide_examples/s02/ex_reference.py)). A 40-participant panel with the test's extra gaps (30% of scores missing, including whole visits and some landmark scores), both treatment forms, the three kinds; then ten domains:

```text
synthetic panel with extra gaps: 31 participants, 0.302 of scores missing
treatment  kind    engine            reference         |diff|     reference time
linear     null    -587.6838978034   -587.6838978019  1.5e-09    0.21 s
linear     free    -417.7297702626   -417.7297702625  7.9e-11    0.20 s
linear     joint   -654.8682452234   -654.8682452233  1.7e-10    0.20 s
emax       null    -596.6553231138   -596.6553231123  1.5e-09    0.19 s
emax       free    -408.0862415320   -408.0862415320  8.1e-11    0.20 s
emax       joint   -659.6257384111   -659.6257384110  1.4e-10    0.20 s

ten domains, 14 participants, free model, chunks of 5: engine -1636.2328045160, reference -1636.2328045155, |diff| 5.6e-10

joint model, PIGD drift with largest real eigenvalue +0.417: engine -674.0646690134, reference -674.0646690133, |diff| 9.6e-11
```

Reading the output:
- The largest gap, $1.5\times10^{-9}$ on $-588$, is a relative $3\times10^{-12}$. The test's tolerance is $10^{-8}\times|\text{reference}|$, about $6\times10^{-6}$ here.
- The last line gives the joint model's PIGD drift a deviation of $+0.5$ on one own rate, which makes it unstable. The engine then composes $S$ and the reference uses Van Loan's block; they agree to $10^{-10}$ at these intervals. No unit test covers this case (§2.41).

**Design choices and pitfalls.**
- The reference is slow: 0.2 s for 31 participants with four domains, against milliseconds for the engine. It is meant for small panels.
- Without the speed bound it calls scipy's `expm` on huge blocks at wild trial points. At $\sigma=20$ it did not return within 20 seconds, while the engine took 0.01 s ([`ex_note_reference_wild.py`](../guide_examples/s02/ex_note_reference_wild.py), §2.41).
- The parameter transforms and the landmark prior are shared with the engine, so their agreement says nothing about them (§2.41).

### 2.40 What `tests/test_engine.py` checks

[`tests/test_engine.py:1`](../../tests/test_engine.py#L1)

**What it does.** Eight checks of protocol §16, all on synthetic panels from `tests/helpers.py` (scenario S1, emax treatment, era on the noise, 12 nodes):

| test | line | what it checks | tolerance | code covered |
|---|---|---|---|---|
| `test_likelihood_matches_reference` | 33 | engine vs reference; 40 participants with extra gaps; linear and emax; null, free and joint | $10^{-8}\max(1,\lvert\text{ref}\rvert)$ | the whole likelihood, masks, treatment |
| `test_gradient_matches_finite_differences` | 47 | 12 random coordinates, null and free, central differences ($h=10^{-6}$) | $10^{-4}\max(1,\lvert g\rvert)$ | `value_and_grad` |
| `test_null_is_nested_in_free_and_joint` | 63 | the null fit written by `null_to` in the free and joint models gives the same log-likelihood | $10^{-8}\lvert\ell_0\rvert$ | layout, nesting (§2.1) |
| `test_grouped_transitions_match_matrix_exponential` | 76 | `transitions_grouped` vs `transitions` (Van Loan) for $c<10$; vs scipy's stationary identity for $c<60$ | $10^{-10}$; $10^{-12}$ | §2.17–2.21 |
| `test_ten_domain_likelihood_matches_reference` | 100 | ten domains, 14 participants, batches of 5 (exercises the blocking of `bmm`) | $10^{-8}\max(1,\lvert\text{ref}\rvert)$ | §2.12 |
| `test_transitions_for_marginal_drift_and_long_intervals` | 114 | eigenvalues 0 and $-0.2$: vs Van Loan for $c\le7.5$; $c=5000$ within 1 s and vs `transition_np` | $10^{-10}$; rtol $10^{-8}$ | composition route, `transition_np` |
| `test_wild_speed_parameters_stay_cheap` | 134 | $\sigma=20$: a finite value within 10 s | — | speed bound |
| `test_both_noise_routes_agree` | 147 | forced stationary vs forced composition, $c<80$ | $10^{-11}$ | §2.20 |

**Inputs → outputs.** Each test takes no arguments, builds its own synthetic data with fixed seeds, and raises `AssertionError` on failure. The plain runner `tests/run_tests.py` calls every `test_*` function and prints one line per test.

**The mathematics: what the tolerances allow.**
- *Likelihood.* $|\ell_{engine}-\ell_{ref}|<10^{-8}\max(1,|\ell_{ref}|)$ is a relative error of $10^{-8}$. Two exact implementations in float64 differ by about $10^{-12}$ relative (§2.39), so the margin is wide, while a modelling error (a wrong sign, a missing term) shifts the log-likelihood by whole units.
- *Gradient.* A central difference with step $h$ has truncation error of order $h^2f'''$ and rounding error of order $\epsilon|f|/h$. With $h=10^{-6}$ and $|f|\approx250$ the rounding part is about $2.5\times10^{-8}$, which is what §2.37 sees. The tolerance $10^{-4}$ (relative) is far above that and far below any real error in a derivative.
- *Nesting.* The null and free log-likelihoods must agree to $10^{-8}$ relative, because the two models are the same model (§2.1), up to the floors of §2.10–§2.11.

**Worked example.** The run used for this guide:

```
OMP_NUM_THREADS=1 ../ppmi-causal/.venv/bin/python tests/run_tests.py engine
```

```text
  ok    test_engine.test_both_noise_routes_agree  (0.0 s)
  ok    test_engine.test_gradient_matches_finite_differences  (0.6 s)
  ok    test_engine.test_grouped_transitions_match_matrix_exponential  (0.0 s)
  ok    test_engine.test_likelihood_matches_reference  (0.7 s)
  ok    test_engine.test_null_is_nested_in_free_and_joint  (0.0 s)
  ok    test_engine.test_ten_domain_likelihood_matches_reference  (0.2 s)
  ok    test_engine.test_transitions_for_marginal_drift_and_long_intervals  (0.0 s)
  ok    test_engine.test_wild_speed_parameters_stay_cheap  (0.0 s)

8 passed, 0 failed, 0 skipped
```

**Design choices and pitfalls.** What the tests do not reach is listed in §2.41: the shared parameter transforms, the optional flags, an unstable drift in the likelihood, and the failure value.

### 2.41 Notes for the author

Things noticed while reading and running the code. Nothing was changed. Each item gives the place, what was seen, why it matters and, where one was run, a reproduction (all synthetic).

**Possible bugs and robustness.**

1. **The failure value is read as convergence at a start point.** [`likelihood.py:216–230`](../../src/pdx/engine/likelihood.py#L216) returns `(1e10, zeros)` on any failure. At the start point a zero gradient meets L-BFGS-B's stopping rule ("CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL"), so `fit.fit_model` records `success=True` with `loglik = -1e10`. Neither `hp.py` (lines 47, 54, 83, 153, 173) nor `gate.py` (lines 98, 153, 197) checks for it. *Why it matters:* a null fit that fails this way gives a likelihood ratio near $2\times10^{10}$; a free fit that fails gives a ratio clipped to 0 (`hp.py:54`). Either way the fit is counted as converged. *Reproduction:* [`ex_note_sentinel.py`](../guide_examples/s02/ex_note_sentinel.py), part (1), output below with part (2).
2. **A domain never observed at the landmark within a phenotype makes every evaluation fail, with only a warning.** [`likelihood.py:84`](../../src/pdx/engine/likelihood.py#L84) (`np.nanmean` gives NaN and "Mean of empty slice") and lines 87–88 ($\tilde y$ becomes NaN for everyone). Item 1 then applies: the fit "converges" at its start after one evaluation. *Why it matters:* this can happen in a bootstrap sample, a small subgroup, or with an optional domain. *Reproduction:* `ex_note_sentinel.py`, part (2):

   ```text
   (1) start with a NaN entry: success=True, loglik=-1e+10, evaluations=4
       message: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL
   (2) mu1 of PIGD = [  nan 0.464 0.182 0.181]; warnings: ['Mean of empty slice']
       value at a good start: 10000000000.0
       fit from the default start: success=True, loglik=-1e+10, evaluations=1
   ```
3. **The floor in `pack_drift` breaks the exact nesting for very small own rates.** [`model.py:223`](../../src/pdx/engine/model.py#L223); used by `fit.null_to` (§3). When PIGD's own rate at its speed, $e^{\gamma}|A_{jj}|$, is below $10^{-4}$, it is moved to $-10^{-4}$, and the free model's start is no longer the null fit. *Why it matters:* a domain that progresses steadily, with almost no mean reversion, is fitted with an own rate near zero and its load carrying the slope, so such values are plausible. The free fit is still optimised from that start, so a negative likelihood ratio stays unlikely, but the "exactly nested start" is not guaranteed. The eigenvalue floor of `pack_chol` (`model.py:230`, $10^{-6}$) is the same kind of issue for a nearly singular $Q_g$. *Reproduction:* [`ex_note_nesting_clip.py`](../guide_examples/s02/ex_note_nesting_clip.py):

   ```text
   own rate of domain 0 (reference speed) | x 1.8 (PIGD) | loglik null | loglik free from null | difference
       -2.50e-01                     -4.50e-01   -463.19037327   -463.19037327   +5.7e-14
       -1.23e-04                     -2.22e-04   -472.17186960   -472.17186960   +0.0e+00
       -4.54e-05                     -8.17e-05   -472.18091102   -472.18032261   +5.9e-04
       -6.14e-06                     -1.11e-05   -472.18546225   -472.18259904   +2.9e-03
   ```
4. **float64 depends on an import side effect.** [`model.py:29`](../../src/pdx/engine/model.py#L29). `transitions.py` does not import `model.py`, and works in the dtype of its inputs. A script that imports only the transition code and builds float32 tensors loses about eight digits. *Why it matters:* the engine's entry points are safe today; a future module or a check script might not be. *Reproduction:* [`ex_note_float32.py`](../guide_examples/s02/ex_note_float32.py):

   ```text
   only transitions imported         : dtype torch.float32, max |S - S_true| at c = 10: 5.9e-09
   after importing pdx.engine.model  : dtype torch.float64, max |S - S_true| at c = 10: 1.4e-17
   ```

**Checks that check less than they seem.**

5. **The reference shares the parameter transforms and the landmark prior with the engine.** [`reference.py:16`](../../src/pdx/engine/reference.py#L16) and line 43 call `unpack_numpy`, which uses `model.drifts`, `noise_cov` and `log_speeds`. `mu1`, `S1` and `y1c` are passed in from `prepare` (lines 39–40). *Why it matters:* an error in `_fill`, `noise_cov`, `log_speeds`, `Layout` or `prepare` would be shared and pass `test_likelihood_matches_reference`. The hand checks of §2.4–§2.7 and §2.27 cover them on small cases, and the nesting test covers part of the layout.
6. **The reference has no speed bound and cannot be evaluated at wild trial points.** The engine clamps $|\log s|\le15$ at the nodes ([`likelihood.py:140`](../../src/pdx/engine/likelihood.py#L140)); the reference does not (`reference.py:73`). They agree to $4.9\times10^{-7}$ at $\sigma=7.4$, so the clamp is harmless. At $\sigma=20$ the reference did not return within 20 s, because scipy 1.18.1's `expm` does not return on blocks of norm near $10^{39}$ or more; the engine took 0.01 s. *Why it matters:* only for the gate's design; the reference is a valid check at plausible values only. *Reproduction:* [`ex_note_reference_wild.py`](../guide_examples/s02/ex_note_reference_wild.py) (the first four lines are in §2.31):

   ```text
   sigma = 20.09: engine -378.504174 in 0.01 s
   reference at sigma = 20.09: no result after 20 s (stopped)
   ```
7. **The "about 35 times slower" figure was not reproduced.** [`linalg.py:3–6`](../../src/pdx/engine/linalg.py#L3) and `docs/code_map.md`. On the M4, single-threaded, the blocked product was 2.6× faster than the plain one at $d=10$ (3.8× at $d=8$), and the whole likelihood and gradient at $d=10$ 1.3× faster (1.6× in an earlier run of the same computation). Results are identical. *Why it matters:* only the documentation; the claim should be re-measured or reworded. *Reproduction:* [`ex_bmm.py`](../guide_examples/s02/ex_bmm.py).
8. **Untested branches.** No test in `tests/` evaluates the likelihood with any of the following:
   - `treatment="none"` (`model.py:182–183`) or `noise_era_col=None`;
   - `load_on_z`, `load_on_y1` or `speed_on_y1` set to False, or `negative_own_rates=False`;
   - `n_groups` above 2;
   - an unstable drift, which the joint model allows (item 10) and trial points can reach.

   For this guide, three of these were checked by hand: a phenotype with two or fewer participants (`likelihood.py:85`) and a batch of landmark-only participants (lines 161–162), in `ex_prepare.py` (§2.27); and an unstable joint-model drift against the reference at moderate $c$, which agreed to $10^{-10}$ (`ex_reference.py`, §2.39). A unit test of the unstable case would be cheap to add.

**Code compared with the protocol and the shared notation.**

9. **The year-1 state is the observed score, not the true state.** [`likelihood.py:87–88`](../../src/pdx/engine/likelihood.py#L87), 138 and 143: $\theta$ and $\Gamma^y$ multiply $\tilde y_{i1}$, the observed landmark scores with missing values filled by the phenotype mean. Protocol §2.3 writes $\theta'x_i(1)$ and a load mean that depends on $x_i(1)$, the true state. *Why it matters:* the code's version is a coherent model conditional on the observed scores, conditioning on observed initial values in the spirit of Wooldridge 2005. But $\theta$ and $\Gamma^y$ are then coefficients on error-prone scores, attenuated relative to the true state's, and the filled values carry no correction. The protocol (or the notation table, §0.4) should say which is meant.
10. **Own rates are constrained only in the reference and free drifts.** [`model.py:148`](../../src/pdx/engine/model.py#L148) fills $D_g$ with `constrain_diag=False`, so the joint model's phenotype-$g$ own rate $-e^{a}+D_{g,jj}$ can be positive. `ex_drifts_speeds.py` gives $+0.2$ and an unstable drift, which the engine handles by composition. *Why it matters:* the §0.4 notation says "own rates $-e^{\text{raw}}$" for all drifts. Localisation (§5) should know that a deviation on an own rate can flip its sign.
11. **$\Gamma^y_g$ is diagonal.** [`model.py:100`](../../src/pdx/engine/model.py#L100) gives `gam_y` the shape $(G,d)$, and `likelihood.py:143` multiplies it elementwise (as does `reference.py:53`). The load on domain $j$ depends only on domain $j$'s landmark score. *Why it matters:* the §0.4 notation writes a matrix $\Gamma^y_g\tilde y_{i1}$, and the protocol does not say. The restriction should be stated.
12. **$\mathrm{ED}_{50}$ and $\mathrm{EC}_{50}$ are fixed.** [`model.py:41–42`](../../src/pdx/engine/model.py#L41); `configs/protocol_v0.5.toml` sets both to 3.0 and marks them provisional. Only $\kappa$ and $\kappa^{\mathrm{LDR}}$ are estimated. *Why it matters:* protocol §6.1 (D24) says "an Emax function" without saying whether the half-maximal dose is estimated. Fixing it is a modelling choice worth recording, perhaps with a sensitivity analysis over its value.
13. **Two protocol items are not implemented in the engine:**
    - the latent-threshold model for domains with more than half their scores at a floor or ceiling (§7.1, D26): only leaving such a domain out is possible now;
    - the sensitivity analysis with a treatment effect on the drift (§6.1, last bullet).
14. **The protocol's description of the computation is out of date.** Protocol §7.1 says the intercept follows "from the same matrix exponential (Van Loan 1978)" and the noise covariance "from the stationary covariance". The code composes $E$ and $M$ from binary powers with Taylor remainders, and uses the stationary covariance only for clearly stable drifts, composing $S$ otherwise (§2.17–§2.20). Both are exact; the Van Loan blocks survive only in the check versions. §7.1 also says "exact gradients and second derivatives … by automatic differentiation". The gradients are, but the Hessian is central differences of the exact gradient (`fit.hessian`, §3).
15. **Treatment at year 1 is in the landmark prior** (a known approximation, restated). [`likelihood.py:79–86`](../../src/pdx/engine/likelihood.py#L79) builds $\mu_{1g}$ and $S_{1g}$ from the recorded year-1 scores, while `_landmark_update` subtracts $h_{i1}$ (line 115). `docs/code_map.md` lists it. It can bias $\kappa$ and the starting state when many participants are treated by year 1.

**Numerical edge cases.**

16. **A floored landmark variance pins the true landmark state.** [`likelihood.py:86`](../../src/pdx/engine/likelihood.py#L86) floors the eigenvalues of $\hat C_g-\Sigma_e$ at $10^{-3}$. If a domain's fixed error variance exceeds its observed landmark variance in a phenotype (a low reliability, or a small group), every participant's true landmark state on that domain is pulled to the phenotype mean. Worth checking WP1's reliabilities against the observed landmark variances before the confirmatory fit.
17. **Fixed quadrature nodes.** In the trace of §2.35, four of twelve nodes carry 98% of the weight. With long follow-up the posterior of $u$ narrows further, and a fixed rule rests on fewer nodes. The engine gate's comparison of 12, 20 and 30 nodes (protocol §16) is the right check. If they disagree, adaptive Gauss–Hermite, with nodes centred and scaled at each participant's mode, would be the remedy.
18. **Checked and fine.**
    - Marginal and unstable drifts: the composition route stays exact for a marginal drift to $c=5000$ (§2.20), and the likelihood with an unstable joint-model drift matches the reference to $10^{-10}$ (§2.39).
    - The switch between the two routes for $S$ at the stability margin changes the likelihood only in its last digits (§2.19).
    - The speed bound changes the likelihood by less than $10^{-6}$ even at $\sigma=7.4$ (§2.31).
    - The batch size changes only the order of summation (§2.29).
    - Gradients match finite differences for every kind of block (§2.37).


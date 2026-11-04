## 5. H-P, the speed estimands and localisation

*Files: [`hp.py`](../../src/pdx/hp.py), [`speed.py`](../../src/pdx/speed.py), [`localise.py`](../../src/pdx/localise.py), [`multiplicity.py`](../../src/pdx/multiplicity.py), with the engine's [`model.py`](../../src/pdx/engine/model.py) and [`fit.py`](../../src/pdx/engine/fit.py) for context (they are explained in §2 and §3). Protocol: §2.3–§2.6, §7.1 ("What H-P needs, and what localisation needs"), §7.3, §7.6–§7.8. Tests: `tests/test_hp.py`, `tests/test_localise.py`, `tests/test_multiplicity.py`. Line numbers are those of commit `a9b577f`.*

**Contents.**
[5.0 Overview](#50-overview) ·
**hp.py** [5.1 HPResult](#51-hpresult--what-one-h-p-run-returns) · [5.2 lr_stat](#52-lr_stat--the-likelihood-ratio) · [5.3 h_p_df](#53-h_p_df--the-degrees-of-freedom) · [5.4 _support](#54-_support--the-support-a-dataset-is-tested-on) · [5.5 _replicate](#55-_replicate--one-replicate-of-the-conditional-parametric-bootstrap) · [5.6 boot_setup, boot_replicate](#56-boot_setup-and-boot_replicate--replicates-with-their-own-seeds-for-parallel-runs) · [5.7 sequential_p](#57-sequential_p--the-besagclifford-sequential-monte-carlo-p-value) · [5.8 observed_fits](#58-observed_fits--selection-and-the-two-observed-fits) · [5.9 run_hp](#59-run_hp--the-whole-h-p-procedure) · [5.10 warp_replicate](#510-warp_replicate--one-dataset-of-the-warp-speed-calibration) · [5.11 reference_distribution](#511-reference_distribution--the-definition-matched-robustness-analysis) ·
**speed.py** [5.12 Z90, _y1c](#512-z90-and-_y1c--constants-and-the-centred-landmark-state) · [5.13 sp1](#513-sp1--sp-1-the-speed-ratio) · [5.14 sp2](#514-sp2--sp-2-the-spread-of-speed-its-predictors-and-the-speedload-correlation) · [5.15 sp3](#515-sp3--sp-3-the-share-of-the-drift-difference-one-speed-explains) · [5.16 speed_estimates, _flat](#516-speed_estimates-and-_flat--all-estimates-as-one-vector) · [5.17 sample_setup, bootstrap_sample](#517-sample_setup-and-bootstrap_sample--one-participant-bootstrap-sample) · [5.18 bootstrap_speed](#518-bootstrap_speed--intervals-the-sp-3-correction-and-its-reportability-test) ·
**localise.py** [5.19 CLASSES](#519-classes-and-the-module-docstring--the-vocabulary-of-verdicts) · [5.20 weighted_median](#520-weighted_median--the-lower-weighted-median) · [5.21 _usable, speed_rule](#521-_usable-and-speed_rule--the-minimal-l1-speed) · [5.22 speed_rule_batch](#522-speed_rule_batch--the-same-rule-for-every-monte-carlo-draw) · [5.23 speed_mode](#523-speed_mode--the-plurality-rule-diagnostic) · [5.24 speed_ls, speed_diagnostics](#524-speed_ls-and-speed_diagnostics--the-other-diagnostic-speeds) · [5.25 _natural, FreeDraws, free_draws](#525-_natural-freedraws-and-free_draws--monte-carlo-draws-of-the-free-fit) · [5.26 _mc_p](#526-_mc_p--monte-carlo-p-values) · [5.27 _basic_ci, _opt_kw](#527-_basic_ci-and-_opt_kw--basic-intervals-from-draws-and-fit-options) · [5.28 SpeedView, LinkResults](#528-speedview-and-linkresults--containers-for-the-verdicts) · [5.29 per_link_tests](#529-per_link_tests--the-anchor-free-per-link-tests) · [5.30 _reject, _nonzero, _within](#530-_reject-_nonzero-and-_within--three-helpers) · [5.31 _classes_at](#531-_classes_at--the-class-rules-at-one-speed) · [5.32 classify_links](#532-classify_links--classes-that-must-hold-across-the-speed-set) · [5.33 speed_and_load_only](#533-speed_and_load_only--the-intersectionunion-verdict) · [5.34 _domain_p, per_domain_tests](#534-_domain_p-and-per_domain_tests--h-d-the-per-domain-tests) · [5.35 recalibrate](#535-recalibrate--per-link-p-values-against-their-bootstrap-null) · [5.36 free_to_joint, _d_slice](#536-free_to_joint-and-_d_slice--the-free-fit-in-the-joint-models-parameters) · [5.37 anchored_tests](#537-anchored_tests--the-anchored-per-link-tests-sensitivity) · [5.38 _penalised_fit, _to_z](#538-_penalised_fit-and-_to_z--the-adaptive-lasso-fit) · [5.39 joint_discovery](#539-joint_discovery--speed-adjusted-joint-discovery) · [5.40 stability_selection](#540-stability_selection--selection-frequencies-over-complementary-halves) · [5.41 commutator_test](#541-commutator_test--the-shared-eigenvector-check) · [5.42 embeddability_check](#542-embeddability_check--complex-or-negative-eigenvalues-of-yearly-matrices) ·
**multiplicity.py** [5.43 holm, holm_adjusted](#543-holm-and-holm_adjusted--holms-step-down-procedure) · [5.44 bh, by](#544-bh-and-by--benjaminihochberg-and-benjaminiyekutieli) · [5.45 simultaneous_intervals](#545-simultaneous_intervals--max-t-simultaneous-intervals) · [5.46 iut_similar](#546-iut_similar--every-interval-inside-ε) ·
[5.N Notes for the author](#5n-notes-for-the-author)

---

### 5.0 Overview

This chapter covers the primary test and everything read off the same fits:
- **H-P** (`hp.py`), the only primary test: do tremor-dominant (TD) and postural instability/gait difficulty (PIGD) patients share one mechanism up to a speed factor?
- **The speed estimands Sp-1 to Sp-3** (`speed.py`), the key secondary results, with participant-bootstrap intervals.
- **Localisation** (`localise.py`): which links and which domains differ beyond speed, under one identifying assumption.
- **The multiplicity tools** (`multiplicity.py`) that the localisation verdicts use.

**The logic in four steps.**

**1. A pure speed difference makes yearly links look different.** Suppose PIGD runs TD's mechanism $s$ times as fast. In continuous time its drift is then exactly $A_1=sA_0$: every entry, own rates included, is multiplied by the same $s$. Its yearly transition matrix, however, is $e^{sA_0}$, which is not an entry-by-entry rescaling of $e^{A_0}$. Take two domains, a motor sign driving daily function, $A_0=\begin{pmatrix}a&0\\c&b\end{pmatrix}$. Then

$$
e^{tA_0}=\begin{pmatrix}e^{ta}&0\\[2pt] c\,\dfrac{e^{ta}-e^{tb}}{a-b}&e^{tb}\end{pmatrix}.
$$

At speed $s$ the yearly persistence $e^{sa}$ falls, while the yearly cross-link $c(e^{sa}-e^{sb})/(a-b)$ changes by a factor that depends on $a$ and $b$, not by $s$. With $a=-0.5$, $b=-0.4$, $c=0.3$ and $s=2$, the cross-link goes from 0.191 to 0.244 (×1.28) and the persistences fall by ×0.61 and ×0.67 (§0.5, idea 1, prints these numbers). A comparison of yearly coefficients would report "a stronger motor → function link and weaker persistence in PIGD", which is one mechanism run faster. This is why the engine works in continuous time, where a pure speed difference is one number.

**2. H-P compares "one drift at two speeds" with "two drifts", and needs no split of speed from deviations.**
- Null: $A_1=sA_0$ for some $s>0$. Everything else may differ: the speed's spread, the load, the noise, the treatment response and the starting state.
- Alternative: $A_1$ is free on the support $\mathcal S$.
- Statistic: the likelihood ratio, $\mathrm{LR}=2(\hat\ell_{free}-\hat\ell_{null})$.

Write PIGD's drift as $A_1=s(A+D)$, the joint model's parameters (§5.36). For every $c>0$ the pair $\big(cs,\ (A+D)/c-A\big)$ gives the same $A_1$, so the same likelihood: speed and deviations trade off exactly. But $\hat\ell_{free}$ is the maximum over all distributions in the free model, and it does not depend on which pair $(s,D)$ is used to write the maximiser down. The null "$A_1$ is proportional to $A_0$" is a statement about the distribution, not about a particular split. So H-P needs no identifying assumption. Neither do Sp-1 and Sp-2: they come from the speed-only model, where the drift is shared and the speed is identified. Sp-3 uses the least-squares speed $\hat s_{LS}$, a property of the pair $(A_0,A_1)$ that assumes nothing about which links differ.

**3. Localisation needs the split.** "Which link differs beyond speed" asks which entries of $\Delta(s)=a_1/s-a_0$ are non-zero, and every $s>0$ gives a $\Delta(s)$ that fits equally well (§5.36 shows identical log-likelihoods at $s=1$, 1.5 and 2.5). The code takes the **minimal-L1 speed**

$$
\hat s_{L1}=\arg\min_{s>0}\ \sum_{k\in U}\big|a_{1k}-s\,a_{0k}\big| ,
$$

which is the weighted median of the entry ratios $r_k=a_{1k}/a_{0k}$ with weights $|a_{0k}|$ (§5.20–§5.21). $U$ is the set of entries allowed to vote: reference entries clearly away from zero, and no sign change. The **identifying assumption** is weighted sparsity. Suppose the true deviations are non-zero only on some entries. Each such entry's ratio moves above or below the true $s$, and the weighted median still equals $s$ as long as the entries whose ratio moves up carry at most half of the weight, and those whose ratio moves down carry less than half. In the protocol's words, "the entries that deviate in the same direction carry less than half of the total weight". The data cannot check this, because a dense pattern of deviations at another speed fits exactly as well. The code therefore:
- computes diagnostic speeds (own rates only, cross-links only, the mode, least squares);
- forms the set of speeds the data cannot tell apart from Monte Carlo draws plus those diagnostics;
- reports a localised verdict only if it holds at every probe speed across that set, and otherwise "speed and mechanism not separable".

**4. The bootstraps repeat everything that used the data.** H-P's replicates re-select the support and refit both models (§5.5). The speed bootstrap re-selects the support in every participant sample (§5.17).

**Four speeds, four targets.** They answer different questions and are not estimates of one number, even under the null.

| quantity | code | fitted model | what it estimates | assumption |
|---|---|---|---|---|
| Sp-1 $=\hat s_0$ | `speed.sp1` | speed-only | $\exp(\gamma_1+\theta^\top(\bar y_1-\bar y_0))$: ratio of geometric-mean speeds, covariates standardised to their pooled distribution, each phenotype at its own mean landmark state | none |
| $e^{\hat\gamma_1}$ | `null_fit.params["log_speed"][1]` | speed-only | the ratio for two participants with the same covariates, landmark state and $u$ | none |
| $\hat s_{LS}$ | `speed.sp3`, `localise.speed_ls` | free | the single factor that best maps $A_0$ onto $A_1$ (least squares); $e^{\gamma_1}$ under the null | none |
| $\hat s_{L1}$ | `localise.speed_rule` | free | $e^{\gamma_1}$ | minimal-L1 |

Here $\bar y_g$ is phenotype $g$'s mean centred landmark state. The shared notation of §0.4 maps $\hat s_0$ to `speed.sp1`. That is Sp-1 of protocol §2.5, which equals $e^{\hat\gamma_1}$ only when $\hat\theta^\top(\bar y_1-\bar y_0)=0$. The free-fit speeds $\hat s_{LS}$ and $\hat s_{L1}$ target $e^{\gamma_1}$, because the free model absorbs $e^{\gamma_1}$ into $A_1$ and keeps $\theta$ (§5.13; Notes, item 1).

**Where the results go.** `scripts/run_hp_real.py` (§7) writes these keys of `results/wp4/hp_tdpigd.json`:

| report key | function |
|---|---|
| `hp` (LR, p, df, replicates, failures, convergence) | `run_hp` → `HPResult.summary` (§5.1, §5.9) |
| `bootstrap_lr` | `HPResult.lr_star` |
| `speed` (`estimates`, `sp1_ci`, `sigma_ci`, …, `sp3_reportable`) | `speed.bootstrap_speed` (§5.18) |
| `links_exploratory` | `localise.classify_links(…, "by")` (§5.32) |
| `speed_and_load_only` | `localise.speed_and_load_only` (§5.33) |
| `speed_diagnostics`, `speed_set` | `LinkResults.diagnostics`, `.speed_set` (§5.29) |
| `domains` | `localise.per_domain_tests` (§5.34) |

**The data flow of this chapter.**

```text
scripts/run_hp_real.py
 ├─ hp.observed_fits ── step1 support_fn ── engine.fit_pair ──► (spec_s, f0 = null fit, f1 = free fit)
 ├─ hp.run_hp(fits, draw) ── hp.sequential_p ◄── parallel.ordered(hp.boot_replicate, i = 0, 1, …)
 │                                                 └─ hp._replicate: simulate_from_fit(f0) → support_fn
 │                                                                   → fit_pair → lr_stat
 ├─ speed.bootstrap_speed ── parallel.ordered(speed.bootstrap_sample, b = 0, 1, …)
 │                             └─ resample participants within phenotype → support_fn → fit_pair
 │                                → speed_estimates (sp1, sp2, sp3)
 └─ localise.per_link_tests ── free_draws (Hessian of f1) → speed_rule, speed_rule_batch, speed_diagnostics
       ├─ classify_links ── _classes_at ── multiplicity.by, multiplicity.simultaneous_intervals
       ├─ speed_and_load_only
       └─ per_domain_tests ── _domain_p ── multiplicity.by
 not called by run_hp_real.py: anchored_tests, joint_discovery, stability_selection, commutator_test
 (smoke test only), embeddability_check, recalibrate, reference_distribution (no caller yet),
 warp_replicate (qualification.py, §6)
```

**The engine in two paragraphs** (details in §2 and §3). `ModelSpec` holds the support (stored as `[to, from]`, diagonal always on), the number of domains $d$, covariates $p$, treatment form, measurement variances and the switches for the load and the landmark terms. `Layout(spec, kind)` maps names to slices of one flat parameter vector for the three kinds:
- **null**: blocks `A`, `gamma` ($G-1$ speed offsets);
- **free**: `A`, `A1`, …, `A{G-1}`, no `gamma`;
- **joint**: `A`, `D1`, …, `gamma`.

Every kind then shares `beta`, `theta`, `log_sigma`, `b`, `L` (noise Cholesky factor), `gam_y`, `gam_z`, `lam`, `log_tau`, `omega` and the treatment blocks. Own rates are stored as $\log(-a_{jj})$ and mapped back through $-e^{(\cdot)}$, so they stay negative. The deviations `D` are stored on their natural scale. `unpack_numpy` returns named quantities: `params["A"]` is the $(G,d,d)$ stack of drifts (for the free model, PIGD's drift is on the observed time scale, its speed absorbed); `params["log_speed"]` is $(0,\gamma_1)$ for the null and joint models and zeros for the free model.

`fit_model` maximises the likelihood by L-BFGS-B from one or more starts and keeps the best. `fit_pair` fits the null model, then starts the free model at the null solution written in its own parameters (`null_to`) and, by default, also at its own default start (`own_free_start=True`). This makes $\hat\ell_{free}\ge\hat\ell_{null}$ up to rounding. `null_to` multiplies PIGD's drift, load mean, load spread and noise by $e^{\gamma_1}$: the speed-only model is exactly nested in the free model (§3). `hessian` returns the observed information by central differences of the exact gradient; `covariance_checked` inverts it, dropping non-positive directions and reporting the condition number.

**How the examples were run.** No example uses PPMI data. Each script is in [`docs/guide_examples/s05/`](../guide_examples/s05/) and runs from the repository folder, for example

```
OMP_NUM_THREADS=1 ../ppmi-causal/.venv/bin/python docs/guide_examples/s05/ex_speed.py
```

with Python 3.13.12, numpy 2.5.4, scipy 1.18.1 and torch 2.14.1. The scripts that fit models share a fixture, [`_fixture.py`](../guide_examples/s05/_fixture.py). Its `engine_panel` builds a small panel in two steps:
1. The design comes from simulator family A, scenario S0 (§6): phenotype labels, covariates, visit dates, landmark scores and missing values.
2. The later visits are simulated from the engine itself, with known parameters, by `simulate_from_fit` applied to a hand-built "fit" that holds the truth. This is the routine the H-P bootstrap uses, so the fitted model is exactly the model that generated the data.

The truth:
- the drift $A$ has own rates $-0.25$ and three links, tremor → function 0.10, axial → function 0.20 and function → independence 0.25;
- PIGD runs at speed 2;
- the random part of log-speed has $\sigma=0.3$, the load spread is $\tau=0.1$, the noise is $Q=0.1I$ and the measurement variance is 0.02;
- the support is the scenario's expert map: the three links plus the candidate axial → independence, which is zero unless an example adds it.

The specification is deliberately light, so that one pair of fits takes about 15–40 seconds on one thread: no treatment terms, a load without covariate shifts (`load_on_z=False`, `load_on_y1=False`), and 6 quadrature nodes. Most examples also skip the free model's second start (`own_free_start=False`); the protocol keeps it. Panels of 55–140 participants give noisy estimates. [`ex_fixture_check.py`](../guide_examples/s05/ex_fixture_check.py) shows that the fixture recovers its truth at about 400 participants:

```text
n = 416 (61 s): exp(gamma-hat) = 1.956 (truth 2), minimal-L1 speed 1.786, LR 5.18 on 7 df
speed-only fit, A-hat:
 [[-0.224  0.     0.     0.   ]
 [ 0.    -0.189  0.     0.   ]
 [ 0.12   0.222 -0.285  0.   ]
 [ 0.    -0.013  0.279 -0.248]]
truth A:
 [[-0.25  0.    0.    0.  ]
 [ 0.   -0.25  0.    0.  ]
 [ 0.1   0.2  -0.25  0.  ]
 [ 0.    0.    0.25 -0.25]]
sigma-hat [0.27  0.308] (truth 0.3, 0.3)
```

**New notation for this chapter.**
- $a_0,a_1\in\mathbb R^{n_{sup}}$: the free fit's support entries for TD (reference) and PIGD, PIGD's on the observed time scale, in `spec.sup_idx` order (row-major over (to, from)). $r_k=a_{1k}/a_{0k}$; $w_k=|a_{0k}|$.
- $c$ = `near_zero` = 0.05; $U=\{k:|a_{0k}|>c,\ r_k>0\}$, the entries that vote on the speed.
- $a_0^{(m)},a_1^{(m)}$, $m=1,\dots,M$: Monte Carlo draws from the free fit's asymptotic normal distribution ($M=2000$, `[localisation] mc_draws`). $s^{(m)}$ is the minimal-L1 speed of draw $m$, and $\Delta^{(m)}=a_1^{(m)}/s^{(m)}-a_0^{(m)}$.
- $T$, $T^*_b$: an observed statistic and its $b$-th bootstrap or Monte Carlo replicate. $I_b=1$ if replicate $b$ is "at least as large" (or failed). $e$ is the count of such replicates, $N$ the number drawn.
- $\vartheta$: the vector of bootstrapped speed quantities (§5.16), $\vartheta^*_b$ its value in sample $b$. In `bootstrap_speed` the code calls this vector `theta`; it is not the model's $\theta$.

---

### 5.1 `HPResult` — what one H-P run returns

[`hp.py:26`](../../src/pdx/hp.py#L26) (class), [`hp.py:43`](../../src/pdx/hp.py#L43) (`summary`)

**What it does.** A dataclass holding everything one H-P run produces, plus `summary()`, the dictionary written to the report.

**Inputs → outputs.** Fields, in order:
- `lr` (the statistic, clamped at 0), `p`, `n_drawn`, `n_exceed`, `stopped_early`, `df`;
- `null_fit`, `free_fit` (engine `Fit` objects), `support` (the selected support, a $d\times d$ boolean array);
- `lr_star` (every replicate statistic in order, `inf` for failures), `failures`, `non_converged`;
- `lr_raw` (the unclamped $2(\hat\ell_{free}-\hat\ell_{null})$) and `extras` (per-replicate output of `on_replicate`).

`summary()` returns `lr`, `lr_raw`, `p`, `df`, `n_drawn`, `n_exceed`, `stopped_early`, `failures`, `replicates_not_converged` (= `non_converged`), `null_converged`, `free_converged` and `free_start_logliks`: the final log-likelihood from each start of the free fit, first the start at the null solution, then the free model's own default start. The engine gate (`gate.py:196`) takes their difference, to see whether the second start ever changes the ratio.

**The mathematics.** None of its own; §5.7 and §5.9 explain each count.

**Worked example.** `run_hp` with stand-in fits (only their log-likelihoods are read) and a scripted stream of replicate outcomes `(statistic, ok, converged)`; observed LR $=2(-95-(-100))=10$, $h=3$ ([`ex_run_hp_bookkeeping.py`](../guide_examples/s05/ex_run_hp_bookkeeping.py)):

```python
script = [(12.0, True, True),       # above the observed LR of 10: exceedance 1
          (np.inf, False, False),   # a failed replicate: exceedance 2
          (3.0, True, False),       # below, and its fits stopped short of convergence
          (11.0, True, True),       # exceedance 3 = h: stop
          (0.5, True, True)]        # never drawn
res = hp.run_hp(None, spec, fits=(spec, f0, f1), n_max=999, h=3, draw=lambda: next(it), callback=...)
```

```text
callback calls (drawn, exceed, statistic, ok): [(1, 1, 12.0, True), (2, 2, inf, False), (3, 2, 3.0, True), (4, 3, 11.0, True)]
summary: {'lr': 10.0, 'lr_raw': 10.0, 'p': 0.75, 'df': 7, 'n_drawn': 4, 'n_exceed': 3, 'stopped_early': True, 'failures': 1, 'replicates_not_converged': 2, 'null_converged': True, 'free_converged': True, 'free_start_logliks': [-95.0, -95.2]}
lr_star: [12.0, inf, 3.0, 11.0]
resumed after two replicates: p 0.75 | failures 1 | replicates_not_converged 1
```

The third exceedance comes at the fourth replicate, so $p=h/N=3/4$. `replicates_not_converged` is 2: the replicate whose fits stopped short, **and the failed one**. After a resume from the first two replicates it is 1, because the resumed failure is not re-counted.

**Design choices and pitfalls.**
- `replicates_not_converged` mixes two things: failed replicates and replicates whose fits did not meet the convergence criterion. It also depends on whether the run was resumed (Notes, item 4). `failures` is reliable on its own.
- `lr_star` keeps `inf` for failures, so a histogram of the bootstrap distribution must drop or mark them.
- `df` is that of the observed support; a replicate with a different selected support has a different number of degrees of freedom.

### 5.2 `lr_stat` — the likelihood ratio

[`hp.py:51`](../../src/pdx/hp.py#L51)

**What it does.** Returns $\max\{2(\hat\ell_{free}-\hat\ell_{null}),\,0\}$ from two fits.

**Inputs → outputs.** `f0` (null fit), `f1` (free fit) → a float ≥ 0.

**The mathematics.** The likelihood-ratio statistic for nested models is

$$
\mathrm{LR}=2\Big(\sup_{\vartheta\in\Theta_{free}}\ell(\vartheta)-\sup_{\vartheta\in\Theta_{null}}\ell(\vartheta)\Big)\;\ge 0 ,
$$

non-negative because $\Theta_{null}\subset\Theta_{free}$ after the reparametrisation `null_to`. That map writes the null's parameters in the free model's: PIGD's drift $e^{\gamma_1}A$, load $e^{\gamma_1}b_1$, noise $e^{\gamma_1}Q_1$, load-mean coefficients times $e^{\gamma_1}$, and $\log\tau_1+\gamma_1$. To see why this is exact, write the participant speed as $s_i=e^{\gamma_1}\tilde s_i$, with $\tilde s_i=\exp(\beta^\top z_i+\theta^\top\tilde y_{i1}+\sigma_1u_i)$:

$$
dX=e^{\gamma_1}\tilde s_i\,[AX+b_1+\ell_i]\,dt+\sqrt{e^{\gamma_1}\tilde s_i}\,G_1\,dW
=\tilde s_i\,[(e^{\gamma_1}A)X+e^{\gamma_1}b_1+e^{\gamma_1}\ell_i]\,dt+\sqrt{\tilde s_i}\,(e^{\gamma_1/2}G_1)\,dW .
$$

The right-hand side is the free model with $A_1=e^{\gamma_1}A$ and the load and noise rescaled. So every null distribution is a free distribution. Under regularity (the true null value is in the interior, and the null is a smooth submanifold) Wilks' theorem gives $\mathrm{LR}\to\chi^2_{df}$ (§5.3). The protocol uses a bootstrap instead, because support selection, bounded scores, dropout and measurement error make the $\chi^2$ reference doubtful at PPMI's sizes (protocol §7.3.1).

*From the maths to the code.* In exact arithmetic the free fit cannot end below the null. `fit_pair` starts it at `null_to(f0.x)`, which has exactly the null's likelihood, and L-BFGS-B never returns a point worse than its start. A raw negative value can come only from rounding in the change of parameters: `pack_drift` floors $-a_{jj}$ at $10^{-4}$ before the log, and `pack_chol` floors the noise eigenvalues at $10^{-6}$. So the code clamps at 0 and keeps the raw value in `HPResult.lr_raw`. Line 54 is the whole function: `max(2.0 * (f1.loglik - f0.loglik), 0.0)`.

**Worked example** ([`ex_hp_observed.py`](../guide_examples/s05/ex_hp_observed.py), on a panel simulated from the speed-only engine, so H-P's null is true):

```text
participants by phenotype: [42 31] | transitions: 300
support (rows = to, columns = from):
 [[1 0 0 0]
 [0 1 0 0]
 [1 1 1 0]
 [0 1 1 1]]
parameters: null 63, free 70, difference 7; (G - 1)(n_sup - 1) = (2 - 1)(8 - 1) = 7
observed fits: 18 s; converged: null True, free True
log-likelihood: null fit -450.9226; free model at the null solution -450.9226; free fit -446.8804
LR raw = 2 x (-446.8804 - (-450.9226)) = 8.0844; lr_stat = 8.0844
chi-square(7) p-value, for orientation only: 0.325
speed-only fit: PIGD speed exp(gamma) = 2.571 (truth 2.0)
free fit, TD drift A0:
 [[-0.113  0.     0.     0.   ]
 [ 0.    -0.217  0.     0.   ]
 [ 0.045  0.367 -0.492  0.   ]
 [ 0.    -0.12   0.285 -0.304]]
free fit, PIGD drift A1 (observed time scale):
 [[-0.939  0.     0.     0.   ]
 [ 0.    -0.942  0.     0.   ]
 [ 0.128  0.572 -0.551  0.   ]
 [ 0.    -0.23   0.779 -0.771]]
```

The free model evaluated at `null_to(f0.x)` reproduces the null log-likelihood to four decimals: the nesting is exact. The free fit climbs 4.04 log-likelihood units from there, so LR = 8.08 on 7 degrees of freedom. The drifts of a 73-participant panel are noisy (TD's tremor own rate is −0.11 against a true −0.25); the fixture check above shows that this is sampling noise.

**Design choices and pitfalls.**
- A clamped zero is a legitimate value. In `sequential_p` every replicate is then "at least as large", so $p=h/h=1$.
- The statistic is only as good as the two optimisations. A free fit stuck below its optimum shrinks LR (conservative). A null fit stuck below its optimum inflates LR (anti-conservative), and the warm-start asymmetry of the bootstrap (§5.5) matters there.

### 5.3 `h_p_df` — the degrees of freedom

[`hp.py:57`](../../src/pdx/hp.py#L57)

**What it does.** Returns $(G-1)(n_{sup}-1)$.

**Inputs → outputs.** A `ModelSpec` → an integer (7 for two phenotypes and 8 support entries).

**The mathematics.** Count the drift parameters; every other block has the same size in both layouts (§5.0).
- Null: $A$ on the support ($n_{sup}$) plus $G-1$ speed offsets $\gamma_g$: $n_{sup}+G-1$.
- Free: one drift per phenotype: $G\,n_{sup}$.
- Difference: $G\,n_{sup}-n_{sup}-(G-1)=(G-1)(n_{sup}-1)$.

Geometrically, the null set $\{(A, e^{\gamma}A)\}$ is an $(n_{sup}+1)$-dimensional cone inside the $2n_{sup}$-dimensional space of drift pairs (for $G=2$). Its codimension, $n_{sup}-1$, is the df of the $\chi^2$ limit. Each phenotype beyond the first adds one speed and costs one full drift.

*From the maths to the code.* Line 59: `(spec.n_groups - 1) * (spec.n_sup - 1)`, with `n_sup = support.sum()` including the diagonal.

**Worked example** ([`ex_hp_observed.py`](../guide_examples/s05/ex_hp_observed.py), the support printed in §5.2: four own rates and four links):

```text
parameters: null 63, free 70, difference 7; (G - 1)(n_sup - 1) = (2 - 1)(8 - 1) = 7
```

The layouts differ by exactly the formula's count.

**Design choices and pitfalls.**
- The df is used only for reporting and for the χ² p-value of the exploratory first look (`run_hp_real.py --first-look`); the protocol's p-value comes from the bootstrap.
- With support selection the df is random: a replicate whose selection adds a link tests one more df.

### 5.4 `_support` — the support a dataset is tested on

[`hp.py:62`](../../src/pdx/hp.py#L62)

**What it does.** Returns `spec` unchanged if `support_fn` is `None`, else `spec.with_support(support_fn(panel))`.

**Inputs → outputs.** A panel, a spec and a support function (or `None`) → a spec.

**The mathematics.** The support used for H-P is $\hat{\mathcal S}(Y)=\mathcal S_{expert}\cup \mathrm{FCI}(Y)\cup\{\text{diagonal}\}$: the expert map's links, plus the lagged adjacencies that Step 1's learner (FCI, the Fast Causal Inference algorithm, in its fast configuration) finds on these data, plus every own rate (`step1.select_support`, §4). `make_support_fn(expert, cfg)` returns the closure `lambda panel: select_support(panel, expert, cfg)`. Selection can only add links to the expert map, never remove one.

*From the maths to the code.* `ModelSpec.with_support` is `dataclasses.replace`, which re-runs `__post_init__`, so the diagonal is switched on again even if the function forgot it.

**Worked example** ([`ex_hp_bootstrap.py`](../guide_examples/s05/ex_hp_bootstrap.py); Step 1's fast configuration with depth 2 around the scenario's map):

```text
observed: selected support adds 0 link(s) to the map; LR 8.084 on 7 df (14 s)
with a map lacking (3,2), the selection would add: [(3, 2)]
```

The map already holds every true link, so the selection adds nothing. Remove function → independence from the map and FCI finds it again.

**Design choices and pitfalls.** The same closure is used for the observed data, for every bootstrap replicate (§5.5) and for every speed-bootstrap sample (§5.17). That is what makes the selection part of the procedure being calibrated.

### 5.5 `_replicate` — one replicate of the conditional parametric bootstrap

[`hp.py:66`](../../src/pdx/hp.py#L66)

**What it does.** Simulates one dataset from the fitted null model, re-selects the support on it, refits both models and returns the replicate's statistic.

**Inputs → outputs.** The null fit `f0`, the observed panel, the spec (before selection), the support function, a random generator, fit options and an optional `on_replicate(sim, g0, g1)` → `(statistic, ok, converged, extra)`:
- success: `(LR*, True, g0.success and g1.success, extra)`;
- any exception in simulation, selection, preparation or fitting: `(inf, False, False, None)`;
- a non-finite statistic: `(inf, False, False, extra)`.

**The mathematics.** Let $\hat\vartheta_0$ be the null fit on the selected support $\hat{\mathcal S}(Y)$, and write the observed statistic as one function of the data, selection included:

$$
T(Y)=\mathrm{LR}\big(Y;\ \hat{\mathcal S}(Y)\big).
$$

The parametric bootstrap estimates $P_{\hat\vartheta_0}\{T(Y^*)\ge T(Y)\}$, where $Y^*$ is drawn from the fitted null **conditionally on** everything the analysis treats as fixed. `simulate_from_fit` (§3) keeps each participant's phenotype, covariates, landmark scores, visit dates, dose path, examination states and pattern of missing values. For each participant it draws a speed $s_i$ and a load $\ell_i$ from the fitted distributions, draws the true landmark state from its posterior given the recorded landmark scores, and simulates the later visits exactly over the actual intervals.

**Why the support selection must be repeated in every replicate.** The statistic of record is $T(Y)$, a composition of selection and testing, so its null distribution is the distribution of that composition. A bootstrap on the observed support $\hat{\mathcal S}(Y)$ held fixed would instead estimate the distribution of $\mathrm{LR}(Y^*;\hat{\mathcal S}(Y))$: a test on a support chosen in advance. Two things are lost:
- the number of degrees of freedom varies with the selection (a selected link adds one), and $T^*$ grows with the df;
- the selection is adaptive: it adds links where these data show a lagged association, so the support, and with it the statistic, depends on the same data. A fixed-support bootstrap ignores that dependence, and its error could go either way.

Repeating the selection makes the bootstrap reflect the procedure that was actually applied, so the p-value inherits the selection's effect, whichever direction it has. The replicate's data come from a null whose drift lives on $\hat{\mathcal S}(Y)$, and the re-selection may add links that are zero in the data-generating model, just as the real selection may add links that are absent in truth. Validity is that of a parametric bootstrap: the p-value is exactly right if the null holds at $\hat\vartheta_0$, and approximately right if it holds at a nearby $\vartheta_0$, as $\hat\vartheta_0\to\vartheta_0$ (protocol §7.3.1: "valid if that model holds, up to its estimated parameters").

*From the maths to the code.*
- Line 72: $Y^*$ = `simulate_from_fit(f0, panel, rng)`. The simulation uses `f0.spec`, so the drift lives on the observed selected support.
- Line 73: $\hat{\mathcal S}(Y^*)$ via `_support(sim, spec, support_fn)`. Here `spec` is the spec *before* selection; `support_fn` replaces its support.
- Line 74: a warm start. If the re-selected support equals the observed one, the null fit starts at `f0.x`, the data-generating parameters of this replicate; otherwise at the default start.
- Lines 75–76: `fit_pair(sim, spec_b, prep=prepare(sim, …), x0_null=warm, **fit_kw)`. The free fit starts at the replicate's null solution and, unless `fit_kw` says `own_free_start=False`, at its own default too.
- Line 77: $T^*$ = `lr_stat(g0, g1)`.
- Line 78: the optional extra output, for example per-link p-values for recalibration (§5.35).
- Lines 79–83: the failure semantics, everything inside one `try`.

**Worked example** ([`ex_hp_bootstrap.py`](../guide_examples/s05/ex_hp_bootstrap.py); shown in §5.9). Replicate $i=0$'s dataset, rebuilt with the same seed, keeps the landmark scores and the missing-value pattern exactly:

```text
replicate i = 0: landmark scores kept: True | missing pattern kept: True | re-selected support adds 0 link(s)
```

**Design choices and pitfalls.**
- **Failures are conservative.** A replicate that raises counts as at least as large as the observed statistic (§5.7), which can only raise the p-value.
- **Non-convergence is not failure.** A replicate whose fits stop at the iteration limit still contributes its statistic, and `converged=False` is recorded.
- **Warm-start asymmetry.** The observed null fit starts at the default start, while each replicate's null fit starts at the true values of its own data-generating process. If the null likelihood had several local optima, the replicates would find the global one more easily than the observed fit, making $T^*$ smaller than a comparable $T$ (anti-conservative). `docs/code_map.md` lists this as a known approximation, checked by the engine gate's dispersed starts.
- `prepare(sim, …)` recomputes the plug-in landmark prior from the simulated panel, but the landmark scores are unchanged, so it is identical to the observed one.

### 5.6 `boot_setup` and `boot_replicate` — replicates with their own seeds for parallel runs

[`hp.py:86`](../../src/pdx/hp.py#L86), [`hp.py:94`](../../src/pdx/hp.py#L94)

**What it does.**
- `boot_setup(f0, panel, spec, expert, step1_cfg, fit_kw, seed)` builds the shared state of the bootstrap workers: the fits, the panel, the spec, the fit options, the seed and the support function `make_support_fn(expert, step1_cfg)` (`None` if `step1_cfg` is `None`).
- `boot_replicate(shared, i)` runs replicate $i$ with its own random stream and returns `(statistic, ok, converged)`.

**Inputs → outputs.** As above. `parallel.ordered(boot_replicate, range(start, n_max), jobs, boot_setup, args)` yields the replicates in index order, whether they ran in one process or in `jobs` worker processes (§7).

**The mathematics.** Replicate $i$ uses the generator `np.random.default_rng([seed, 3, i])`. NumPy turns the integer list into a `SeedSequence` whose entropy is $(\text{seed},3,i)$, and distinct lists give streams that are independent for all practical purposes. The middle number tags the use:
- 1: speed-bootstrap samples (§5.17);
- 2: the localisation draws in `run_hp_real.py`;
- 3: H-P replicates.

Replicate $i$'s dataset and statistic therefore depend only on $(\text{seed},i)$: not on how many workers run, in which order they finish, or whether the run was interrupted and resumed. Because `parallel.ordered` yields in index order, `sequential_p` applies Besag–Clifford's rule to exactly the sequence a serial run would produce.

*From the maths to the code.* Line 97 creates the stream; line 98 calls `_replicate(…)[:3]`, dropping the fourth element (`extra`). `run_hp_real.py` resumes with `range(len(state["replicates"]), n_max)`, so the indices continue where the checkpoint stopped.

**Worked example** (from [`ex_hp_bootstrap.py`](../guide_examples/s05/ex_hp_bootstrap.py)): replicate $i=1$ recomputed on its own gives the same statistic as inside the run.

```text
replicate i = 1 run again on its own: LR* 7.373315 (in the run: 7.373315)
```

**Design choices and pitfalls.**
- With `jobs > 1`, workers may finish replicates beyond the stopping point. That costs compute but cannot change $p$, because the rule reads replicates in order and the generator is closed when it stops.
- One job runs in the calling process with its threads; the order of floating-point summation then differs from the single-threaded workers, so statistics agree to the fits' tolerance (below $10^{-4}$), not bit for bit (`parallel.py` docstring; `tests/test_hp.py::test_bootstrap_replicates_agree_however_many_run_at_once`, a slow test).
- Dropping `extra` means the parallel path cannot carry `on_replicate` output, so per-link recalibration (§5.35) cannot run through it (Notes, item 9).
- `[seed, 3, i]` in `run_hp_real.py` and the serial `run_hp` path (one shared `rng`) give different replicate sequences. Both are valid; only the parallel path is reproducible across job counts.

### 5.7 `sequential_p` — the Besag–Clifford sequential Monte Carlo p-value

[`hp.py:101`](../../src/pdx/hp.py#L101)

**What it does.** Draws replicates one at a time and stops at the $h$-th one that is at least as large as the observed statistic, or at $n_{max}$. Returns the p-value with its counts.

**Inputs → outputs.**
- `draw()`, which returns `(statistic, ok)` or a longer tuple starting with them;
- the observed statistic, `n_max` (999), `h` (20);
- an optional `callback(drawn, exceed, statistic, ok)`, called after each replicate;
- an optional `resume` list of earlier `(statistic, ok)` pairs.

→ a dict with `p`, `n_drawn`, `n_exceed`, `stopped_early`, `failures`, `stars` (all statistics) and `rest` (the extra tuple elements of the newly drawn replicates).

**The mathematics.** Let $t$ be the observed statistic and $I_b=\mathbf 1\{T^*_b\ge t\ \text{or replicate } b \text{ failed}\}$. Let $L$ be the index of the $h$-th exceedance, if it comes by $n_{max}$, and $e=\sum_{b\le n_{max}}I_b$ otherwise. Besag & Clifford's (1991) p-value is

$$
p=\begin{cases}h/L & \text{if the } h\text{-th exceedance occurs at } L\le n_{max},\\[2pt](e+1)/(n_{max}+1) & \text{otherwise.}\end{cases}
$$

*Why it is exactly valid.* Take continuous statistics, no failures, and replicates drawn from the true null distribution $F$. Given $t$, the indicators are independent Bernoulli($q$) with $q=1-F(t)$, and under $H_0$, $q\sim U(0,1)$. For a stopping value $h/l$ ($h\le l\le n_{max}$):

$$
P(p\le h/l)=P(\text{at most } h-1 \text{ exceedances among the first } l-1)=\int_0^1\sum_{k=0}^{h-1}\binom{l-1}{k}q^k(1-q)^{l-1-k}\,dq=\frac hl ,
$$

using $\int_0^1\binom nk q^k(1-q)^{n-k}dq=1/(n+1)$. The first equality holds because $p\le h/l$ happens either when the procedure stops at $L\ge l$, or when it never stops, in which case $p\le h/(n_{max}+1)<h/l$. For a non-stopping value, $g\le h-1$:

$$
P\Big(p\le\tfrac{g+1}{n_{max}+1}\Big)=P(e\le g)=\int_0^1\sum_{k=0}^{g}\binom{n_{max}}{k}q^k(1-q)^{n_{max}-k}\,dq=\frac{g+1}{n_{max}+1}.
$$

So $P(p\le a)=a$ at every attainable $a$: the test is exact, like a fixed Monte Carlo test with $n_{max}$ replicates, but it stops early when $p$ is clearly large. In the parametric bootstrap $F$ is the fitted null's distribution, so exactness becomes approximate validity (§5.5).

*How many replicates.* $P(N>l)=P(\text{fewer than } h \text{ exceedances in } l)$, which averages to $\min(1,h/(l+1))$ under $H_0$. So

$$
E[N]=\sum_{l=0}^{n_{max}-1}\min\Big(1,\frac h{l+1}\Big)=h\Big(1+\sum_{m=h+1}^{n_{max}}\frac1m\Big)\approx h\big(1+\ln(n_{max}/h)\big)=97.7
$$

for $(h,n_{max})=(20,999)$. Under a clear alternative ($q\approx0$) the procedure runs to $n_{max}$. To reject at 5% it needs $L\ge400$: $p=20/400=0.05$.

*Why failures and ties count as exceedances.* The p-value is non-decreasing in every indicator. Turn one $I_b$ from 0 to 1:
- if the procedure had stopped at $L>b$, it now stops at some $L'\le L$, so $h/L'\ge h/L$;
- if it had not stopped, either $e$ rises by one, or the $h$-th exceedance now arrives at some $L'\le n_{max}$, and then $h/L'\ge h/n_{max}>h/(n_{max}+1)=(e+1)/(n_{max}+1)$.

So counting a failed replicate as an exceedance gives an upper bound on the p-value that any value of its statistic could have produced: conservative. Ties (`>=`) do the same for discrete statistics, for example many statistics clamped at 0.

*From the maths to the code.*
- Lines 107–112: resumed pairs are counted exactly like new draws. (`run_hp_real.py` saves a non-finite statistic as `None` in its JSON checkpoint and maps it back to `inf` before passing `resume`, since `float(None)` would raise.)
- Line 113: `while drawn < n_max and exceed < h` is the stopping rule.
- Lines 114–123: one draw; `rest` keeps `out[2:]`; line 120: `not ok or stat >= observed` is $I_b$.
- Lines 124–126: stopped, $p=h/N$. This includes $L=n_{max}$, where the protocol's rule also gives $h/L$.
- Lines 127–128: not stopped, $p=(e+1)/(n_{max}+1)$.

**Worked example** ([`ex_sequential_p.py`](../guide_examples/s05/ex_sequential_p.py), made-up streams; part 6 draws $\chi^2_7$ statistics under the null 20,000 times):

```text
1. early stop: observed 4.0, h = 3, n_max = 99
   {'p': 0.75, 'n_drawn': 4, 'n_exceed': 3, 'stopped_early': True} -> p = h / drawn = 3/4
2. no stop: 99 replicates all below the observed
   {'p': 0.01, 'n_drawn': 99, 'n_exceed': 0, 'stopped_early': False} -> p = (0 + 1)/(99 + 1)
3. a tie and a failed replicate both count as at least as large
   {'p': 0.75, 'n_drawn': 3, 'n_exceed': 2, 'failures': 1} -> p = (2 + 1)/(3 + 1)
4. resume from a checkpoint gives the same answer
   full   : {'p': 0.5, 'n_drawn': 6, 'n_exceed': 3}
   resumed: {'p': 0.5, 'n_drawn': 6, 'n_exceed': 3}
5. monotone: turning any replicate into an exceedance never lowers p (all 2^7 sequences, h = 3)
   smallest change in p after a flip: 0.0
6. exact validity under the null and the expected number of replicates
   P(p <= 0.01) = 0.0106  (MC SE 0.0007)
   P(p <= 0.05) = 0.0498  (MC SE 0.0015)
   P(p <= 0.10) = 0.1008  (MC SE 0.0021)
   P(p <= 0.50) = 0.4972  (MC SE 0.0035)
   mean replicates drawn: 97.8; formula h(1 + sum_(m=h+1..n_max) 1/m) = 97.7
   attainable p-values near 0.05 when stopping early: [0.0526, 0.05, 0.0476]
```

By hand, case 1: the stream is 5.0, 1.0, 6.0, 7.0 against $t=4$, so the exceedances are draws 1, 3 and 4, and $p=3/4$. Case 3: the tie (4.0) and the failure count, the 0.0 does not, so $e=2$ and $p=3/4$. Part 5 checks the monotonicity argument on all 128 indicator sequences of length 7. Part 6 confirms the exactness and the 97.7 replicates expected under the null.

**Design choices and pitfalls.**
- The smallest attainable p-value is $1/(n_{max}+1)=0.001$. When stopping early, $p\ge h/n_{max}\approx0.020$; smaller values arise only from running to the cap.
- The p-value is coarse where it does not matter ($p=1$ after 20 straight exceedances) and fine near 0.05, where about 400 replicates are needed.
- Resumed pairs count towards `failures` and `stars` but not towards `rest`, so anything computed from `rest` (non-convergence, extras) covers only the replicates drawn in the current session (§5.9).
- If a checkpoint already holds $h$ exceedances, no new draw is made and $p=h/N$ with $N$ the saved count. That is correct only if the saved list ends at the $h$-th exceedance, which is what the loop writes.

### 5.8 `observed_fits` — selection and the two observed fits

[`hp.py:131`](../../src/pdx/hp.py#L131)

**What it does.** Selects the support on the observed panel and fits both models.

**Inputs → outputs.** A panel, a spec, a support function and fit options → `(spec_s, f0, f1)`: the spec with the selected support, the null fit and the free fit.

**The mathematics.** $\hat{\mathcal S}(Y)$ (§5.4), then $\hat\vartheta_0=\arg\max_{\Theta_{null}}\ell$ and $\hat\vartheta_1=\arg\max_{\Theta_{free}}\ell$ on that support.

*From the maths to the code.* Two lines: `_support`, then `fit_pair(panel, spec_s, **fit_kw)`. `run_hp_real.py` calls it once and passes the triple to `run_hp(fits=…)`, so the observed fits are not repeated; it also uses the observed LR to decide whether a checkpoint can be resumed.

**Worked example.** §5.2 (no selection) and §5.9 (with selection: `observed: selected support adds 0 link(s) to the map`).

**Design choices and pitfalls.** The observed fits are deterministic (no random numbers), so rerunning them on resume reproduces the statistic to rounding; `run_hp_real.py` accepts a checkpoint only if $|\mathrm{LR}-\mathrm{LR}_{saved}|<10^{-6}$.

### 5.9 `run_hp` — the whole H-P procedure

[`hp.py:138`](../../src/pdx/hp.py#L138)

**What it does.** Observed fits (or the ones passed in), the statistic, the sequential bootstrap and the result.

**Inputs → outputs.** A panel, the spec before selection, the support function, `n_max`, `h`, a generator, fit options, `callback`, `resume`, `on_replicate`, `fits` (output of `observed_fits`) and `draw` (a replicate supplier, for example `boot_replicate` through `parallel.ordered`) → `HPResult`.

**The mathematics.** $p=\mathrm{BC}\big(T(Y);\ T(Y^*_1),T(Y^*_2),\dots\big)$, with $T$ and $Y^*$ as in §5.5 and BC as in §5.7.

*From the maths to the code.*
- Line 152: the observed fits, reused if given.
- Lines 153–154: `lr_raw` and the clamped `lr`.
- Line 155: the default replicate supplier, `_replicate` with the shared generator `rng`.
- Line 156: `sequential_p(draw, lr, n_max, h, callback, resume)`.
- Line 157: `non_conv` counts the newly drawn replicates whose third element (`converged`) is false. A failed replicate returns `converged=False`, so failures are counted here too.
- Line 158: `extras`, the `on_replicate` outputs (`None` entries if no hook was given).
- Lines 159–160: `HPResult`, with `df = h_p_df(spec_s)` and `support = spec_s.support`.

**Worked example** ([`ex_hp_bootstrap.py`](../guide_examples/s05/ex_hp_bootstrap.py)): the panel of §5.2, Step 1's fast configuration with depth 2 around the expert map, and a three-replicate bootstrap with $h=2$, run as `run_hp_real.py` runs it.

```python
draws = parallel.ordered(hp.boot_replicate, range(3), 1, hp.boot_setup, (f0, panel, spec, expert, cfg, fit_kw, seed))
res = hp.run_hp(panel, spec, support_fn, n_max=3, h=2, fits=fits, draw=lambda: next(draws), callback=...)
```

```text
observed: selected support adds 0 link(s) to the map; LR 8.084 on 7 df (14 s)
with a map lacking (3,2), the selection would add: [(3, 2)]
replicate i = 0: landmark scores kept: True | missing pattern kept: True | re-selected support adds 0 link(s)
replicate i = 0: LR* 4.703, ok True, exceedances 0
replicate i = 1: LR* 7.373, ok True, exceedances 0
replicate i = 2: LR* 8.769, ok True, exceedances 1 (32 s)
result: {'lr': 8.0844, 'lr_raw': 8.0844, 'p': 0.5, 'df': 7, 'n_drawn': 3, 'n_exceed': 1, 'stopped_early': False, 'failures': 0, 'replicates_not_converged': 0, 'null_converged': True, 'free_converged': True, 'free_start_logliks': [-446.88038831605695]}
replicate i = 1 run again on its own: LR* 7.373315 (in the run: 7.373315)
```

Here the map already holds every true link, so the selection adds none. The second line shows what the selection does when the map lacks one: FCI finds function → independence again. One of three replicates reaches 8.084, the procedure does not stop, and $p=(1+1)/(3+1)=0.5$. `free_start_logliks` has one entry because the example skipped the free model's second start.

**Design choices and pitfalls.**
- The replicates receive the spec *before* selection, and `support_fn` re-selects. If `fits` came from `observed_fits` with a support function but `run_hp` is called with `support_fn=None` and no `draw`, the replicates would test the unselected support while the observed statistic used the selected one. `run_hp_real.py` passes both, so this is only a calling hazard (Notes, item 16).
- `non_converged` and `extras` cover only replicates drawn in this session (§5.7).

### 5.10 `warp_replicate` — one dataset of the warp-speed calibration

[`hp.py:163`](../../src/pdx/hp.py#L163)

**What it does.** For one simulated dataset: selects the support, fits both models, computes the observed statistic, and draws **one** bootstrap replicate. `qualification.dataset_task` (§6) calls it for each simulated dataset of WP3, the method qualification.

**Inputs → outputs.** A panel, the spec, the support function, a generator and fit options → `{"lr", "lr_star", "replicate_ok", "replicate_converged", "converged", "df", "null_fit", "free_fit"}`, with `lr_star = inf` for a failed replicate.

**The mathematics.** Checking H-P's size by brute force over $R$ simulated datasets would need $R\times B$ bootstrap refits. The warp-speed method (Giacomini, Politis & White 2013) keeps one replicate per dataset:
1. for dataset $r$, compute $T_r$ and a single $T^*_r$ from the bootstrap of dataset $r$;
2. pool $\{T^*_1,\dots,T^*_R\}$ and take its empirical $(1-\alpha)$ quantile $\hat c$;
3. estimate the size as $R^{-1}\sum_r\mathbf 1\{T_r>\hat c\}$.

Why it works: dataset $r$'s bootstrap distribution $F^*_r$ depends on the data only through the fitted parameters, which converge to the truth, so each $F^*_r$ approaches one limit $F^*$. The pooled single draws are then draws from (approximately) $F^*$, and $\hat c$ estimates its quantile, the critical value every full bootstrap would use. The cost falls from $R(B+1)$ fits to $2R$. `qualification.calibration_summary` (§6) computes $\hat c$ with failed replicates entering the pool as $+\infty$ (`_rank_quantile`) and failed datasets counting as rejections. Since Besag–Clifford rejects at 5% exactly when fewer than 20 of the first 399 replicates reach $T$, its rejection rate is that of the test with critical value at the bootstrap's 95% quantile, up to Monte Carlo error, which is what the warp-speed rate estimates.

*From the maths to the code.* Lines 169–170 are `observed_fits` inlined; line 171 is one `_replicate` with the dataset's own generator; line 172 assembles the dict.

**Worked example** ([`ex_warp_speed.py`](../guide_examples/s05/ex_warp_speed.py)). Part 1 is a toy where the exact answer is known: $T=n\bar x^2/s^2\sim F(1,19)$ under the null, with the parametric bootstrap drawing from $N(0,\hat s_0^2)$. Part 2 runs `warp_replicate` on one small engine panel.

```text
1. toy: X_1..X_20 ~ N(0, 1); statistic T = n xbar^2 / s^2; parametric bootstrap from N(0, s0^2)
   full bootstrap (199 per dataset): rejection rate 0.0457
   warp speed (1 per dataset): critical value 4.391 (exact F(1,19): 4.381), rejection rate 0.0457
   bootstrap fits needed: 796000 against 4000
2. hp.warp_replicate on one small engine panel (speed-only model true)
   {'lr': 10.593, 'lr_star': 6.64, 'replicate_ok': True, 'replicate_converged': True, 'converged': True, 'df': 7} (17 s)
```

The pooled critical value (4.391) lands next to the exact $F(1,19)$ quantile (4.381), and the warp-speed rejection rate equals the full bootstrap's (both 0.0457, Monte Carlo SE about 0.0033) at 1/199 of the cost.

**Design choices and pitfalls.**
- The method estimates the size of the bootstrap test as a whole (averaged over datasets), not the p-value of any one dataset.
- Pooling is valid only within one configuration (sample size, support rule, fit options), which is why `qualification.config_key` hashes every setting except the seed and scenario.

### 5.11 `reference_distribution` — the definition-matched robustness analysis

[`hp.py:177`](../../src/pdx/hp.py#L177)

**What it does.** Computes H-P's statistic, with support selection, on `n_datasets` datasets from a generator, and returns the list. It serves protocol §7.3.6: a generator fitted to the pooled population under one shared mechanism, which re-applies the phenotype rule to simulated baselines, so that the simulated phenotypes inherit the rule's artefacts (range restriction, collider bias, regression to the mean) but no mechanism difference.

**Inputs → outputs.** `generate(rng) → Panel`, the number of datasets, the spec, the support function, a generator and fit options → a list of statistics, `nan` for a dataset whose selection or fits raised.

**The mathematics.** The output is a sample from the distribution of $T(Y)$ under the generator. The protocol's reading is descriptive: if the observed $T$ lies within this distribution, the phenotype rule's artefacts could explain it. The function computes no p-value.

*From the maths to the code.* A plain loop: `generate`, `_support`, `fit_pair`, `lr_stat`, with `nan` on exception.

**Worked example** ([`ex_warp_speed.py`](../guide_examples/s05/ex_warp_speed.py), part 3, a generator that returns fresh engine panels):

```text
3. hp.reference_distribution on two datasets from a generator
   statistics: [ 9.056 14.76 ] (27 s)
```

**Design choices and pitfalls.**
- No caller yet: the generator fitted to PPMI comes in WP3 (`docs/code_map.md`).
- Failures become `nan`, whereas the bootstrap, the warp-speed pool and `qualification` treat failures as $+\infty$ (Notes, item 15).

---

### 5.12 `Z90` and `_y1c` — constants and the centred landmark state

[`speed.py:27`](../../src/pdx/speed.py#L27), [`speed.py:30`](../../src/pdx/speed.py#L30)

**What it does.** `Z90 = 1.2815515655446004` is $z_{0.9}=\Phi^{-1}(0.9)$, the 90th percentile of the standard normal. `_y1c(panel, spec)` returns the centred landmark scores $\tilde y_{i1}$ exactly as the likelihood uses them (`prepare(...).y1c`).

**Inputs → outputs.** A panel and a spec → an $(n,d)$ array.

**The mathematics.** `prepare` (§2) fills a missing landmark score with the phenotype's mean of the observed landmark scores, then centres each column at its pooled mean:

$$
\tilde y_{i1,j}=y^\circ_{i1,j}-\frac1n\sum_{i'}y^\circ_{i'1,j},\qquad y^\circ_{i1,j}=\begin{cases}y_{i1,j}&\text{observed}\\ \bar y^{obs}_{g_i,j}&\text{missing.}\end{cases}
$$

Filling with the phenotype mean does not change that phenotype's mean, so a difference of phenotype means of $\tilde y$ is the difference of the observed landmark means, domain by domain.

*From the maths to the code.* One line: `prepare(panel, spec.meas_var, spec.n_groups).y1c.numpy()`.

**Worked example.** In §5.13 the PIGD − TD difference of mean landmark states is `[-0.578  0.959  0.157  0.047]`: PIGD starts with less tremor and more axial impairment, as the phenotype rule implies.

**Design choices and pitfalls.** `_y1c` builds the whole padded data tensors to read one array. That is cheap at these sizes, and it guarantees that Sp-1 and Sp-2 use exactly the likelihood's $\tilde y$.

### 5.13 `sp1` — Sp-1, the speed ratio

[`speed.py:34`](../../src/pdx/speed.py#L34)

**What it does.** Returns Sp-1, the ratio of PIGD's to TD's geometric-mean speed in the speed-only model, with the covariates standardised to their pooled distribution and each phenotype averaged over its own landmark states (protocol §2.5).

**Inputs → outputs.** The null fit and the panel → a float.

**The mathematics.** Log-speed is $\log s_i=\gamma_{g_i}+\beta^\top z_i+\theta^\top\tilde y_{i1}+\sigma_{g_i}u_i$, with $\gamma_0=0$. The geometric mean of $s$ is $\exp(E\log s)$. Standardise $z$ to its pooled distribution $P_z$, and average $\tilde y$ over phenotype $g$'s own participants:

$$
\log \mathrm{GM}_g=E_{z\sim P_z}\big[\gamma_g+\beta^\top z\big]+\theta^\top\bar{\tilde y}_g+\sigma_g\,E[u]=\gamma_g+\beta^\top\bar z+\theta^\top\bar{\tilde y}_g .
$$

Then

$$
\text{Sp-1}=\frac{\mathrm{GM}_1}{\mathrm{GM}_0}=\exp\!\Big(\gamma_1+\theta^\top\big(\bar{\tilde y}_1-\bar{\tilde y}_0\big)\Big).
$$

Two things drop out, which is why the code needs neither $z$ nor $\sigma$:
- $\beta^\top\bar z$ is common to both phenotypes, because both are standardised to the same $P_z$;
- $\sigma_g E[u]=0$. This is the reason for a geometric mean: an arithmetic mean would carry $e^{\sigma_g^2/2}$, which differs between phenotypes.

The term $\theta^\top(\bar{\tilde y}_1-\bar{\tilde y}_0)$ is part of the estimand by design. The protocol says each phenotype is averaged over its own year-1 states, "which partly reflect the phenotype itself".

*From the maths to the code.*
- Line 36: $\tilde y$ from `_y1c`.
- Line 38: `shift` $=\hat\theta^\top(\bar{\tilde y}_1-\bar{\tilde y}_0)$ with the means over `g == 1` and `g == 0`. The guard `if par["theta"].size else 0.0` never triggers: `unpack_numpy` returns zeros of length $d$ when the model has no $\theta$.
- Line 39: `exp(log_speed[1] + shift)`, with `log_speed[1]` $=\hat\gamma_1$.

**Worked example** ([`ex_speed.py`](../guide_examples/s05/ex_speed.py); engine panel with PIGD at speed 2 and $\theta=(0.3,0.3,0,0)$, so faster progression with more tremor or axial impairment at the landmark):

```text
fits: 13 s
mean centred landmark state, PIGD - TD: [-0.578  0.959  0.157  0.047]
theta-hat: [ 0.124  0.251 -0.013  0.027] | gamma-hat: 0.401
Sp-1 = exp(gamma + theta.dy) = exp(0.401 + 0.169) = 1.767; speed.sp1 = 1.767
truth: exp(gamma) = 2.000, Sp-1 = exp(log 2 + theta.dy) = 2.242
free-fit speeds: minimal-L1 1.261, least squares 1.271
```

By hand: $\hat\theta^\top\Delta\bar y=0.124(-0.578)+0.251(0.959)-0.013(0.157)+0.027(0.047)=0.168$, and $e^{0.401+0.168}=1.77$. In truth, PIGD's landmark states add $0.3(0.959-0.578)=0.114$ to its mean log-speed, so the true Sp-1 is $2e^{0.114}=2.242$, not 2. The free-fit speeds target $e^{\gamma_1}=2$, not 2.242, because the free model absorbs $e^{\gamma_1}$ into $A_1$ and keeps $\theta$. On this 73-participant panel all the estimates are far off; the fixture check in §5.0 shows the estimates settle at about 400 participants.

**Design choices and pitfalls.**
- Sp-1, $e^{\hat\gamma_1}$, $\hat s_{LS}$ and $\hat s_{L1}$ answer different questions (table in §5.0). The protocol's "minimal-L1 speed shown beside Sp-1" (§2.5) compares numbers that differ by $\theta^\top\Delta\bar y$ even under the null (Notes, item 1).
- The index `[1]` assumes two phenotypes. An axis with three groups (protocol §7.3.8) would need one ratio per group.
- The model-based interval for Sp-1 used by the first look (`run_hp_real.sp1_interval`) differentiates the same expression: gradient 1 for $\gamma_1$ and $\Delta\bar y$ for $\theta$.

### 5.14 `sp2` — Sp-2, the spread of speed, its predictors and the speed–load correlation

[`speed.py:42`](../../src/pdx/speed.py#L42)

**What it does.** Returns, from the null fit:
- $\sigma_g$ for each phenotype;
- the 90th/10th percentile speed ratio;
- $\beta$ and $\theta$, the effects of the covariates and of the landmark state on log-speed;
- the share of between-participant variation in log-speed that phenotype, covariates and landmark state explain;
- the correlation between each domain's participant load and the participant's speed.

**Inputs → outputs.** The null fit and the panel → a dict: `sigma` ($G$), `ratio_90_10` ($G$), `beta` ($p$), `theta` ($d$), `share_explained`, `speed_load_corr` ($G\times d$).

**The mathematics.**
- *Spread.* Given its covariates, a participant's log-speed is $\eta_i+\sigma_gu_i$ with $\eta_i=\gamma_{g_i}+\beta^\top z_i+\theta^\top\tilde y_{i1}$. Its $q$-quantile is $\eta_i+\sigma_g z_q$, so

$$
\frac{s_{0.9}}{s_{0.1}}=\exp\big(\sigma_g(z_{0.9}-z_{0.1})\big)=\exp(2z_{0.9}\,\sigma_g).
$$

  This is the spread among participants of one phenotype *with the same characteristics*.
- *Share explained.* By the law of total variance, with $u_i$ independent of $(g,z,\tilde y)$,

$$
\mathrm{Var}(\log s)=\mathrm{Var}(\eta)+E[\sigma_g^2],\qquad \text{share}=\frac{\mathrm{Var}(\eta)}{\mathrm{Var}(\eta)+\sum_g\pi_g\sigma_g^2},
$$

  computed over the sample: $\mathrm{Var}(\eta)$ is the variance of the fitted $\eta_i$ over participants, and $\pi_g=n_g/n$.
- *Speed–load correlation.* The load of domain $j$ is $\ell_{ij}=\mu_{ij}+\lambda_{gj}u_i+\tau_{gj}\varepsilon_{ij}$, with $\varepsilon\perp u$ both standard normal. Given the covariates, $\mathrm{Cov}(u,\ell_j)=\lambda_{gj}$ and $\mathrm{Var}(\ell_j)=\lambda_{gj}^2+\tau_{gj}^2$, so

$$
\mathrm{Corr}(\log s_i,\ell_{ij}\mid g,z,\tilde y)=\mathrm{Corr}(u_i,\ell_{ij}\mid\cdot)=\frac{\lambda_{gj}}{\sqrt{\lambda_{gj}^2+\tau_{gj}^2}} .
$$

*From the maths to the code.*
- Line 46: `eta` $=\hat\gamma_{g_i}+z_i^\top\hat\beta+\tilde y_{i1}^\top\hat\theta$.
- Lines 47–48: `sig` $=\hat\sigma_g$; `pi` $=n_g/n$.
- Line 49: `within` $=\sum_g\pi_g\hat\sigma_g^2$.
- Line 50: `between` $=$ `np.var(eta)`, the population variance (divisor $n$).
- Line 51: `ratio_90_10` $=\exp(2\,z_{0.9}\,\hat\sigma_g)$.
- Line 53: the share, 0 if both parts are 0.
- Line 54: `lam / sqrt(lam**2 + tau**2)`, elementwise over $(g,j)$.

**Worked example** ([`ex_speed.py`](../guide_examples/s05/ex_speed.py)):

```text
Sp-2: sigma [0.174 0.093] | 90/10 ratio exp(2 x 1.2816 x sigma) [1.562 1.271]
  between = var(eta) = 0.1206; within = sum pi_g sigma_g^2 = 0.0211; share = 0.8509 (code 0.8509)
  speed-load correlation lam/sqrt(lam^2 + tau^2), TD row: [-1.    -1.     0.066  0.307]
```

By hand: $e^{2\times1.2816\times0.174}=e^{0.446}=1.56$; share $=0.1206/(0.1206+0.0211)=0.851$. The true $\sigma$ is 0.3: with 73 participants the random part of speed is hard to tell from the noise. The correlations of −1 for tremor and axial in TD are a boundary: the fitted load spread $\tau$ went to 0 there, so the load became an exact multiple of $u$.

**Design choices and pitfalls.**
- The 90/10 ratio is conditional on the characteristics. The protocol's "ratio between the 90th- and 10th-percentile patients' speeds" could also be read as the marginal spread within the phenotype, which adds the spread of $\beta^\top z+\theta^\top\tilde y$ within the phenotype (Notes, item 11).
- A correlation of ±1 signals $\hat\tau\to0$, an estimate on the boundary of the parameter space. Its bootstrap interval then has poor coverage.
- If $\lambda=\tau=0$ for some $(g,j)$, the correlation is $0/0$ = NaN.

### 5.15 `sp3` — Sp-3, the share of the drift difference one speed explains

[`speed.py:57`](../../src/pdx/speed.py#L57)

**What it does.** From the free fit: the least-squares speed $\hat s_{LS}$, the share of $\|A_1-A_0\|^2$ that this one speed explains, and $\|A_1-A_0\|$.

**Inputs → outputs.** The free fit → `{"s_ls", "share", "norm_diff"}`.

**The mathematics.** With the Frobenius inner product $\langle X,Y\rangle=\sum_{jk}X_{jk}Y_{jk}$:

$$
\hat s_{LS}=\arg\min_s\|A_1-sA_0\|^2=\frac{\langle A_1,A_0\rangle}{\|A_0\|^2},\qquad \text{Sp-3}=1-\frac{\|A_1-\hat s_{LS}A_0\|^2}{\|A_1-A_0\|^2}.
$$

*Geometry.* The residual $R=A_1-\hat s_{LS}A_0$ is orthogonal to $A_0$, so $A_1-A_0=R+(\hat s_{LS}-1)A_0$ and, by Pythagoras, $\|A_1-A_0\|^2=\|R\|^2+(\hat s_{LS}-1)^2\|A_0\|^2$. Hence

$$
\text{Sp-3}=\frac{(\hat s_{LS}-1)^2\|A_0\|^2}{\|A_1-A_0\|^2}=\cos^2\angle\big(A_1-A_0,\ A_0\big)\in[0,1].
$$

Sp-3 is the squared cosine between the phenotype difference and the "speed direction" $A_0$: 1 for a pure speed difference ($A_1=sA_0$, $s\ne1$), 0 when the difference is orthogonal to $A_0$. It is undefined when $A_1=A_0$, and unstable when $A_1-A_0$ is small, because estimation noise then dominates both the numerator and the denominator. If the estimated difference were pure noise spread evenly over the $n_{sup}$ support entries, only one direction in $n_{sup}$ would lie along $A_0$, and the share would average about $1/n_{sup}$ whatever the truth.

*From the maths to the code.*
- Line 59: the full $d\times d$ matrices, flattened. Entries off the support are zero in both phenotypes, so they add nothing, and the result equals the same formula on the support entries (`localise.speed_ls`).
- Line 60: $\hat s_{LS}$, with the denominator floored at $10^{-12}$.
- Line 61: $\|A_1-A_0\|^2$.
- Line 62: the share, NaN if the difference is exactly 0.

**Worked example** ([`ex_speed.py`](../guide_examples/s05/ex_speed.py)):

```text
Sp-3 by hand: s_LS = <A1, A0>/<A0, A0> = 1.2706; share = 0.0594; code {'s_ls': 1.2705825864886777, 'share': 0.0594007930648347, 'norm_diff': 0.8745120583090491}
```

On this small panel one speed explains 6% of the drift difference, because noise in the free fits points in every direction. §5.18 shows that the reportability test then withholds Sp-3. On the true parameters ($A_1=2A_0$) the formula gives $\hat s_{LS}=2$ and share 1.

**Design choices and pitfalls.**
- The share weights entries by their squared size. Large own rates dominate $\|A_0\|^2$, so Sp-3 says mostly whether the own rates scale together.
- `norm_diff` is the reportability statistic of §5.18.

### 5.16 `speed_estimates` and `_flat` — all estimates as one vector

[`speed.py:66`](../../src/pdx/speed.py#L66), [`speed.py:70`](../../src/pdx/speed.py#L70)

**What it does.** `speed_estimates` bundles `sp1`, `sp2` and `sp3`. `_flat` writes every bootstrapped quantity into one vector $\vartheta$, on the log scale for ratios and spreads.

**Inputs → outputs.** `_flat(e, free_fit)` → a vector of length $1+G+p+d+1+Gd+1$, in this order:

| block | length | scale |
|---|---|---|
| Sp-1 | 1 | log |
| $\sigma_g$ | $G$ | log |
| $\beta$ | $p$ | natural |
| $\theta$ | $d$ | natural |
| share explained | 1 | natural |
| speed–load correlation | $G\cdot d$ | natural |
| Sp-3 share | 1 | natural |

**The mathematics.** None beyond the log transform: intervals for positive quantities are formed on the log scale (§5.18) and mapped back.

*From the maths to the code.* Lines 73–74 concatenate the blocks. `bootstrap_speed` later cuts the vector with the cumulative block lengths (line 130).

**Worked example.**

```text
bootstrapped vector (_flat): 21 entries = 1 + G + p + d + 1 + G*d + 1 = 21
```

for $G=2$, $p=4$, $d=4$.

**Design choices and pitfalls.**
- `_flat`'s `free_fit` argument is unused.
- The length depends only on $G$, $p$ and $d$, never on the support, so samples with different selected supports stack into one matrix.

### 5.17 `sample_setup` and `bootstrap_sample` — one participant-bootstrap sample

[`speed.py:77`](../../src/pdx/speed.py#L77), [`speed.py:86`](../../src/pdx/speed.py#L86)

**What it does.**
- `sample_setup(panel, spec, null_fit, support, fit_kw, seed)` builds the shared state: the panel, the spec, the null fit (for warm starts), the fit options, the seed, the support function (from `support = (expert map, Step1Config)`, or `None` to keep the spec's support) and each phenotype's participant indices.
- `bootstrap_sample(shared, b)` draws sample $b$, refits and returns `(ϑ*_b as a list, vec(A*_1 − A*_0) as a list)`, or `None` if anything raises.

**Inputs → outputs.** As above.

**The mathematics.** The participant (cluster) bootstrap, stratified by phenotype. Participants are the independent units; their visits are dependent and stay together. Sample $b$ draws $n_g$ participants with replacement from phenotype $g$, for each $g$, so the group sizes stay as observed. The bootstrap distribution of $\vartheta^*-\hat\vartheta$ then approximates the sampling distribution of $\hat\vartheta-\vartheta$ given the group sizes. Every data-dependent step is repeated in each sample: the support selection, the landmark prior and centring (inside `prepare`), and both fits.

*From the maths to the code.*
- Line 89: the sample's own generator, `default_rng([seed, 1, b])`.
- Line 90: within-phenotype resampling, `rng.choice(gi, size=len(gi))` for each phenotype.
- Line 91: `panel.subset(idx)`, a panel with duplicated participants.
- Line 93: support re-selection.
- Line 94: warm start of the null fit at the observed null fit, if the support is unchanged.
- Lines 95–97: `fit_pair`, `speed_estimates`, `_flat`, and the drift difference on the full $d\times d$ matrices.
- Lines 98–99: any exception gives `None`.

**Worked example.** Run inside `bootstrap_speed` (§5.18).

**Design choices and pitfalls.**
- The covariates $z$ stay centred at the full-sample means, so a resampled panel has $\bar z^*\ne0$. That shifts every participant's log-speed by the same $\beta^\top\bar z^*$, which the scale of the drift absorbs: speed ratios, drift ratios and Sp-3's share are unchanged, and the share explained uses only the variance of $\eta$.
- The landmark centring and prior are recomputed in each sample, so they vary across samples as they would across studies.
- A failed sample is dropped (complete-case bootstrap). If failures were more likely in some kinds of samples, the intervals would be conditional on success; `failures` is reported.

### 5.18 `bootstrap_speed` — intervals, the Sp-3 correction and its reportability test

[`speed.py:102`](../../src/pdx/speed.py#L102)

**What it does.** Runs $B$ participant-bootstrap samples (200 in the protocol, `[speed] bootstrap`), possibly in parallel and from a checkpoint, and returns:
- basic 95% intervals for every quantity;
- the bias-corrected Sp-3 and its one-sided lower 95% bound;
- the test that decides whether Sp-3 may be reported.

**Inputs → outputs.** The panel, the spec, both fits, `B`, `seed`, `fit_kw`, `level` (0.95), `support`, `done` (the checkpoint: `rows`, `diffs`, `failures`), `on_sample(done)` and `jobs` → a dict: `estimates`, `B`, `B_used`, `failures`, `sp1_ci`, `sigma_ci`, `ratio_90_10_ci`, `beta_ci`, `theta_ci`, `share_explained_ci`, `speed_load_corr_ci`, `sp3_bias_corrected`, `sp3_lower_95`, `sp3_test_statistic`, `sp3_threshold`, `sp3_reportable`.

**The mathematics.**
- *Basic intervals.* Let $q^*_a$ be the $a$-quantile of $\vartheta^*$ over the samples. If $\vartheta^*-\hat\vartheta$ mimics $\hat\vartheta-\vartheta$, then $P\big(q^*_{\alpha/2}-\hat\vartheta\le\hat\vartheta-\vartheta\le q^*_{1-\alpha/2}-\hat\vartheta\big)\approx1-\alpha$, which rearranges to

$$
\vartheta\in\big[\,2\hat\vartheta-q^*_{1-\alpha/2},\ \ 2\hat\vartheta-q^*_{\alpha/2}\,\big].
$$

  For Sp-1 and $\sigma_g$ this is applied to the logs and then exponentiated. For Sp-1 it reads $\big[\widehat{\text{Sp-1}}^2/Q_{0.975},\ \widehat{\text{Sp-1}}^2/Q_{0.025}\big]$ with $Q_a=\exp\{q^*_a(\log\text{Sp-1}^*)\}$. The 90/10 interval is the $\sigma$ interval pushed through the increasing map $\sigma\mapsto e^{2z_{0.9}\sigma}$.
- *Sp-3, bias-corrected.* The bootstrap bias estimate is $\overline{\text{Sp-3}^*}-\widehat{\text{Sp-3}}$, so the corrected value is $2\widehat{\text{Sp-3}}-\overline{\text{Sp-3}^*}$. The one-sided basic lower bound is $2\widehat{\text{Sp-3}}-q^*_{0.95}$. Both are clipped to $[0,1]$.
- *Reportability.* Let $\hat D=\mathrm{vec}(\hat A_1-\hat A_0)$ and $D^*_b$ its value in sample $b$. The statistic is $T=\|\hat D\|$; its reference is the centred bootstrap $T^*_b=\|D^*_b-\hat D\|$, which mimics $\|\hat D-D\|$. Under $H_0: D=0$, $T=\|\hat D-D\|$, so

$$
\text{reportable}\iff\|\hat D\|>q_{0.95}\big(\|D^*_b-\hat D\|\big),
$$

  a 5% bootstrap test that the free drifts differ at all. It is the protocol's "‖A_PIGD − A_TD‖ clearly exceeds its bootstrap noise".

*From the maths to the code.*
- Lines 109–111: the estimates, $\hat\vartheta$ (named `theta` in the code) and $\hat D$.
- Lines 112–114: the checkpoint. `start = len(rows) + failures` is the index of the next sample; each sample has its own seed, so a resumed or parallel run draws exactly the samples an uninterrupted serial run would.
- Lines 115–123: `parallel.ordered` over samples `start … B−1`; successes are appended, failures counted, and `on_sample(done)` is called after each.
- Lines 125–128: `lo` $=2\hat\vartheta-q^*_{1-a}$ and `hi` $=2\hat\vartheta-q^*_{a}$ with $a=0.025$ (`np.nanquantile`, linear interpolation).
- Lines 129–132: the cut points `k = cumsum([1, G, p, d, 1, G*d, 1])` split `lo` and `hi` into the blocks of §5.16.
- Lines 133–137: Sp-3 and its samples (the last column), $T$ and $T^*_b$.
- Lines 138–149: the output. `exp` for the Sp-1 and $\sigma$ blocks; `clip` to $[0,1]$ for the two Sp-3 numbers only.

**Worked example** ([`ex_speed.py`](../guide_examples/s05/ex_speed.py); $B=3$ samples with support re-selection, only to show the arithmetic, where the protocol uses 200):

```python
out = speed.bootstrap_speed(panel, spec, f0, f1, B=3, seed=7, fit_kw={"own_free_start": False},
                            support=(spec.support, Step1Config(depth=2)), on_sample=lambda done: seen.append(...))
```

```text
bootstrap: 47 s; B_used 3, failures 0
  sp1_ci: [1.374, 1.889]
  sigma_ci: [[0.088, 0.176], [0.03, 0.466]]
  ratio_90_10_ci: [[1.252, 1.57], [1.079, 3.302]]
  share_explained_ci: [0.874, 1.016]
  sp3_bias_corrected: 0.0
  sp3_lower_95: 0.0
  sp3_test_statistic: 0.875
  sp3_threshold: 0.923
  sp3_reportable: False
  Sp-1 basic interval by hand: exp(2 log Sp-1 - q*) = [1.374, 1.889]
  Sp-3 test by hand: ||A1 - A0|| = 0.875; centred bootstrap norms [0.956 0.62  0.579]
```

The Sp-1 interval recomputed from the saved samples matches. The upper end of the share-explained interval is 1.016, above the largest possible share: basic intervals are not clipped for this quantity. The drift difference (norm 0.875) does not exceed the 95th percentile of its centred bootstrap noise (0.923, interpolated between 0.62 and 0.956), so Sp-3 is not reportable. Its bias-corrected value and lower bound are clipped to 0.

**Design choices and pitfalls.**
- Basic intervals for the share explained and the speed–load correlations can leave $[0,1]$ and $[-1,1]$ (Notes, item 12). A logit or Fisher-$z$ scale, or clipping, would keep them in range.
- If every sample fails, line 134 (`R[:, -1]` on an empty array) raises `IndexError` (Notes, item 3). `run_hp_real.py` skips the function only when $B=0$.
- The function returns the bias-corrected Sp-3 and its bound whether or not `sp3_reportable` is true; the report must apply the flag.
- The `done` lists passed in are appended to in place, so the caller's checkpoint object grows as the run proceeds. `run_hp_real.py` relies on this.

---

### 5.19 `CLASSES` and the module docstring — the vocabulary of verdicts

[`localise.py:1`](../../src/pdx/localise.py#L1) (docstring), [`localise.py:32`](../../src/pdx/localise.py#L32) (`CLASSES`)

**What it does.** The docstring states the module's premise: everything here except the shared-eigenvector check (§5.41) needs an identifying assumption, because speed and deviations trade off exactly in the free model. `CLASSES` lists the five link classes of protocol §2.3: "similar", "different strength beyond speed", "structurally different", "different, type unresolved", "unresolved". The functions add four more labels:
- "outside the family": a support entry that is not tested, by default an own rate;
- "speed and mechanism not separable": a class that changes across the speed set (§5.32);
- "speed not identified": no reference entry can vote on the speed (§5.29);
- for domains, "differs beyond speed" and "no difference detected" (§5.34).

**Inputs → outputs.** Constants.

**The mathematics.** Deviations are on the reference-speed scale, $\Delta=a_1/s-a_0$, because $\varepsilon$ is defined per year at the reference speed (protocol §2.3).

**Worked example.** Every label appears in the outputs of §5.29–§5.34.

**Design choices and pitfalls.** `CLASSES` itself is not used by the code: the strings are repeated literally in `_classes_at`, so a renamed class must be changed in both places.

### 5.20 `weighted_median` — the lower weighted median

[`localise.py:38`](../../src/pdx/localise.py#L38)

**What it does.** Returns the smallest value $v_{(k)}$ whose cumulative weight reaches half of the total.

**Inputs → outputs.** Values $v$ and non-negative weights $w$ → a float; NaN if there are no values or the weights sum to 0.

**The mathematics.** The weighted median minimises $f(s)=\sum_jw_j|v_j-s|$. $f$ is convex and piecewise linear, and its right derivative is

$$
f'_+(s)=\sum_{v_j\le s}w_j-\sum_{v_j>s}w_j=2C(s)-W,\qquad C(s)=\sum_{v_j\le s}w_j,\ \ W=\sum_jw_j .
$$

A point $s$ minimises $f$ when $f'_-(s)\le0\le f'_+(s)$, that is $C(s^-)\le W/2\le C(s)$. The smallest such point is $v_{(k^*)}$ with $k^*=\min\{k:\sum_{j\le k}w_{(j)}\ge W/2\}$, the *lower* weighted median. If $C(v_{(k^*)})=W/2$ exactly, every $s\in[v_{(k^*)},v_{(k^*+1)}]$ minimises $f$, and the code returns the left end.

*From the maths to the code.*
- Line 40: NaN if empty or $W\le0$.
- Lines 42–43: sort; cumulative weights $C$.
- Line 44: `np.searchsorted(c, 0.5 * c[-1])`, with the default `side="left"`, is $k^*$, the first index with $C\ge W/2$.
- Line 45: `min(k, len(v) - 1)` guards against rounding.

**Worked example.** The unit test's cases: values (1, 2, 3) with weights (1, 1, 1) give 2; with weights (5, 1, 1) they give 1, because the first value already carries 5 of 7. The minimal-L1 example of §5.21 shows the cumulative sums.

**Design choices and pitfalls.** With an exact tie at half the total, the lower end is returned, so the estimate can sit one gap below the midpoint of the minimising interval. With estimated, continuous ratios, exact ties essentially never happen.

### 5.21 `_usable` and `speed_rule` — the minimal-L1 speed

[`localise.py:48`](../../src/pdx/localise.py#L48), [`localise.py:52`](../../src/pdx/localise.py#L52)

**What it does.** `_usable` marks the entries allowed to vote: $|a_{0k}|>c$ and $a_{1k}/a_{0k}>0$. `speed_rule` returns the weighted median of their ratios, with weights $|a_{0k}|$, optionally within a mask (own rates only, cross-links only).

**Inputs → outputs.** Two vectors of support entries (reference $a_0$, other phenotype $a_1$), `near_zero` ($c=0.05$, `[localisation] near_zero`, provisional) and an optional mask → $\hat s_{L1}$, or NaN if no entry can vote.

**The mathematics.** For $k\in U$, $|a_{1k}-s\,a_{0k}|=|a_{0k}|\,|r_k-s|$, so

$$
\hat s_{L1}=\arg\min_{s}\sum_{k\in U}|a_{1k}-s\,a_{0k}|=\arg\min_s\sum_{k\in U}|a_{0k}|\,|r_k-s| ,
$$

the weighted median of $\{r_k\}_{k\in U}$ with weights $|a_{0k}|$ (§5.20). The objective is written on the other phenotype's time scale, $a_1-sa_0$, as the protocol states it.

*When it recovers the true speed.* Suppose the truth is $a_{1k}=s\,(a_{0k}+\Delta_k)$. An entry with $\Delta_k=0$ has $r_k=s$ exactly. An entry with $\Delta_k\ne0$ (and no sign change) has $r_k=s(1+\Delta_k/a_{0k})$: above $s$ if $\Delta_k/a_{0k}>0$, below if $\Delta_k/a_{0k}<0$. Let $W_{up}$ and $W_{down}$ be the weights of the entries whose ratio moves up and down, and $W$ the total. By §5.20, the lower weighted median equals $s$ exactly when

$$
W_{down}<W/2\quad\text{and}\quad W_{up}\le W/2 .
$$

This is the protocol's "the entries that deviate in the same direction carry less than half of the total weight". "Direction" refers to the ratio, not to the sign of $\Delta$: a stronger positive cross-link moves its ratio up, and so does an own rate that is more negative beyond speed. Once the deviating entries on one side carry more than half of the weight, the rule returns their ratio instead.

*From the maths to the code.*
- Line 49: `_usable`. `np.where(a0 == 0, 1.0, a0)` avoids dividing by zero; such entries fail the first condition anyway.
- Line 56: an optional mask restricts the vote.
- Line 58: `weighted_median(a1[ok] / a0[ok], np.abs(a0[ok]))`.

**Worked example** ([`ex_minimal_l1.py`](../guide_examples/s05/ex_minimal_l1.py); hand-made entries where PIGD = 2 × TD, except one link stronger beyond speed and one near-zero reference entry with a sign change):

```text
link (to,from): [(0, 0), (1, 1), (2, 0), (2, 1), (2, 2), (3, 1), (3, 2), (3, 3)]
a0: [-0.25 -0.25  0.1   0.2  -0.25  0.02  0.25 -0.25]
a1: [-0.5 -0.5  0.2  0.9 -0.5 -0.1  0.5 -0.5]
ratio a1/a0  : [ 2.   2.   2.   4.5  2.  -5.   2.   2. ]
usable       : [1 1 1 1 1 0 1 1] (|a0| > 0.05 and same sign)
sorted ratios: [2.  2.  2.  2.  2.  2.  4.5]
their weights: [0.25 0.25 0.1  0.25 0.25 0.25 0.2 ]  cumulative: [0.25 0.5  0.6  0.85 1.1  1.35 1.55]  half: 0.775
weighted_median by hand -> first sorted ratio whose cumulative weight reaches half
speed_rule: 2.0 | weighted_median: 2.0
grid argmin of sum|a1 - s a0| over usable entries: 2.0
grid argmin over all entries (sign change and near-zero entry included): 2.0

breakdown: more and more entries deviate the same way (their ratio becomes 3 instead of 2)
  0 deviating entries carrying 0.00 of 1.55: speed_rule = 2.000
  1 deviating entries carrying 0.10 of 1.55: speed_rule = 2.000
  2 deviating entries carrying 0.35 of 1.55: speed_rule = 2.000
  3 deviating entries carrying 0.55 of 1.55: speed_rule = 2.000
  4 deviating entries carrying 0.80 of 1.55: speed_rule = 3.000
  5 deviating entries carrying 1.05 of 1.55: speed_rule = 3.000
```

The (3,1) entry has $|a_0|=0.02<0.05$ and changes sign, so it does not vote. The first cumulative weight to reach $0.775$ is $0.85$, at a ratio of 2. A brute-force grid search of $\sum|a_1-sa_0|$ agrees. The breakdown table shows the assumption at work. Up to a deviating weight of 0.55 the rule returns the true speed 2. When the deviating weight (0.80) passes half of the voting weight (0.775), it returns 3, the deviating entries' ratio. Nothing in the data marks the change.

**Design choices and pitfalls.**
- The threshold $c=0.05$ is absolute, in drift units per year on the normal-scores scale. It is provisional (config).
- "Near zero" and "sign change" are judged on point estimates. In the Monte Carlo draws, entries move in and out of $U$ (§5.22).
- Leaving sign changes out of the vote is a choice. In the full L1 objective they would pull $s$ downwards, towards their negative ratios. In the example their weight is too small to matter (the grid search over all entries also gives 2).
- Own rates are always negative in both phenotypes (they are $-e^{\text{raw}}$), so they always vote unless $|a_{0jj}|\le0.05$. They often carry most of the weight (1.00 of 1.55 here), and the "own rates only" diagnostic shows whether they dominate.

### 5.22 `speed_rule_batch` — the same rule for every Monte Carlo draw

[`localise.py:61`](../../src/pdx/localise.py#L61)

**What it does.** Applies `speed_rule` row by row to $(M,k)$ arrays of draws, without a Python loop.

**Inputs → outputs.** `a0`, `a1` of shape $(M,k)$ and `near_zero` → $M$ speeds, NaN for rows where no entry can vote.

**The mathematics.** As §5.21, for each row.

*From the maths to the code.*
- Line 63: usable entries. `a1 * a0 > 0` replaces the ratio test; it is equivalent because $a_0\ne0$ there.
- Lines 64–65: unusable entries get ratio $+\infty$ and weight 0, so they sort last and never reach half the weight.
- Lines 66–68: sort each row; cumulative weights.
- Line 69: `k = (c < 0.5 * c[:, -1:]).sum(1)` counts the cumulative weights strictly below half, which is the index of the first one at or above it: the same $k^*$ as `searchsorted(…, side="left")`.
- Lines 70–71: pick $r_{(k^*)}$; NaN where the total weight is 0.

**Worked example** ([`ex_minimal_l1.py`](../guide_examples/s05/ex_minimal_l1.py)):

```text
batch version equals the scalar one row by row
  batch : [2.       2.181818 1.777778]
  scalar: [2.       2.181818 1.777778]
```

`tests/test_localise.py::test_batch_matches_scalar` checks the same on 200 random rows.

**Design choices and pitfalls.** The cumulative weight reaches the total at the last usable entry, so $k^*$ never lands on a $+\infty$ placeholder.

### 5.23 `speed_mode` — the plurality-rule diagnostic

[`localise.py:74`](../../src/pdx/localise.py#L74)

**What it does.** Returns the mode of the weighted distribution of log-ratios: the speed that most entries agree on (the plurality rule, Guo 2018; the mode-based estimate of Hartwig 2017).

**Inputs → outputs.** `a0`, `a1`, `near_zero` → a float; NaN if no entry can vote; the ratio itself if only one can.

**The mathematics.** With $\ell_k=\log r_k$ and normalised weights $\omega_k=|a_{0k}|/\sum_{k'\in U}|a_{0k'}|$, the weighted kernel density estimate is

$$
\hat f(x)\propto\sum_{k\in U}\omega_k\,\phi\Big(\frac{x-\ell_k}{b}\Big),\qquad b=0.9\cdot\max\{\tilde\sigma,10^{-3}\}\cdot K^{-1/5},
$$

with $K=|U|$, and $\tilde\sigma=\min(s_\omega,\mathrm{MAD})$ if that minimum is positive, else $\max(s_\omega,\mathrm{MAD})$. Here $s_\omega$ is the weighted standard deviation (SD) of the $\ell_k$, and the median absolute deviation is MAD $=1.4826\cdot\mathrm{median}|\ell-\mathrm{median}\,\ell|$ (unweighted). This is Silverman's rule with the robust spread, Hartwig's "modified Silverman" bandwidth with tuning parameter 1. The estimate is $\exp(\arg\max_x\hat f(x))$.

*From the maths to the code.*
- Lines 77–82: usable entries; log-ratios and weights; the single-entry case.
- Lines 83–85: normalised weights, $s_\omega$ and the MAD (`scale="normal"` gives the 1.4826 factor).
- Line 86: the bandwidth.
- Lines 87–89: a grid of 2001 points over $[\min\ell-3b,\ \max\ell+3b]$, the density on it, and $\exp$ of the argmax.

**Worked example** ([`ex_minimal_l1.py`](../guide_examples/s05/ex_minimal_l1.py)):

```text
mode bandwidth: weighted sd 0.2719, MAD 0.0000 -> 0.9 x 0.2719 x 7^(-1/5)
mode: 2.0 | least squares a1.a0/a0.a0: 2.268 = 2.268
```

Six of the seven voting log-ratios are identical, so the MAD is 0 and the bandwidth falls back on the weighted SD. The mode is 2.

**Design choices and pitfalls.**
- Hartwig weights by inverse variances; here the weights are $|a_0|$, as for the median.
- The grid resolution is about $(\max\ell-\min\ell+6b)/2000$ on the log scale.
- The mode is a diagnostic: it widens the speed set (§5.29) but never sets the speed.

### 5.24 `speed_ls` and `speed_diagnostics` — the other diagnostic speeds

[`localise.py:92`](../../src/pdx/localise.py#L92), [`localise.py:97`](../../src/pdx/localise.py#L97)

**What it does.** `speed_ls` is the least-squares speed $\langle a_1,a_0\rangle/\langle a_0,a_0\rangle$ over all support entries (no exclusions). `speed_diagnostics` returns the five speeds the protocol pre-specifies: all entries, own rates only, cross-links only, the mode and least squares.

**Inputs → outputs.** `speed_diagnostics(a0, a1, spec, near_zero)` → `{"all", "own_rates", "cross_links", "mode", "least_squares"}`.

**The mathematics.** On the support entries, $\hat s_{LS}$ equals Sp-3's $\hat s_{LS}$ (§5.15), because the entries off the support are zero. Least squares has no robustness: one deviating entry moves it.

*From the maths to the code.* Lines 98–99: the own-rate mask from `spec.sup_idx`. Lines 100–102: `speed_rule` with the mask, its complement and no mask; `speed_mode`; `speed_ls`.

**Worked example** ([`ex_minimal_l1.py`](../guide_examples/s05/ex_minimal_l1.py)):

```text
diagnostics: {'all': 2.0, 'own_rates': 2.0, 'cross_links': 2.0, 'mode': 2.0, 'least_squares': 2.268}
```

By hand, $\langle a_1,a_0\rangle=0.823$ and $\langle a_0,a_0\rangle=0.3629$, so $\hat s_{LS}=2.268$. The one deviating link (0.9 where 0.4 was due) pulls least squares up, while the four robust speeds stay at 2.

**Design choices and pitfalls.**
- The vectors must be in `spec.sup_idx` order, or the own-rate mask picks the wrong entries.
- Least squares is reported but left out of the speed set, because it is not robust.

### 5.25 `_natural`, `FreeDraws` and `free_draws` — Monte Carlo draws of the free fit

[`localise.py:107`](../../src/pdx/localise.py#L107), [`localise.py:115`](../../src/pdx/localise.py#L115), [`localise.py:124`](../../src/pdx/localise.py#L124)

**What it does.** Draws both phenotypes' drift entries from the free fit's asymptotic normal distribution. The raw parameters are drawn, and own rates are mapped through $-\exp$, so every draw respects the sign constraint. `FreeDraws` holds the estimates $a_0,a_1$, the draws `d0`, `d1` $(M\times n_{sup})$, the covariance and its report. `_natural` maps raw support values to drift entries.

**Inputs → outputs.** The free fit, the panel, $M$, a generator, an optional `prep`, an optional covariance (to skip the Hessian) and `group` (1) → `FreeDraws`.

**The mathematics.** Let $\xi$ be the raw parameters (own rates on the scale $\log(-a_{jj})$, cross-links as they are). Asymptotically $\hat\xi\approx N(\xi,V)$, with $V=\mathcal I(\hat\xi)^{-1}$ the inverse observed information. The code draws

$$
\xi^{(m)}=\hat\xi+Lz^{(m)},\quad z^{(m)}\sim N(0,I),\quad LL^\top=V_+ ,
$$

where $V_+$ keeps the non-negative part of the spectrum of $V$, and maps each draw to drift entries with $g(\xi)=(-e^{\xi_{jj}};\ \xi_{jk})$. The draws $g(\xi^{(m)})$ approximate the sampling distribution of $g(\hat\xi)$. This is the Monte Carlo version of the delta method: it keeps the non-linearity of $g$, and later of the median speed. Only the coordinates of `A` and `A1` are drawn, with their joint covariance taken from the full $V$, so the uncertainty of every other parameter enters through it.

*From the maths to the code.*
- Line 132: $V$ from `covariance_checked(hessian(…))`. This is the inverse observed information, dropping non-positive directions, with a report (smallest eigenvalue, condition number, flat directions).
- Lines 133–135: the indices of the `A` and `A{group}` blocks; $V$ restricted to them and symmetrised.
- Lines 136–137: `eigh`, then $L=U\,\mathrm{diag}(\sqrt{\max(w,0)})$.
- Line 138: raw draws $\hat\xi+ZL^\top$.
- Lines 139–141: `_natural` on the estimates and on both halves of the draws.

**Worked example** ([`ex_localise_fit.py`](../guide_examples/s05/ex_localise_fit.py); engine panel of 140 participants, PIGD at speed 2 plus one link only PIGD has, axial → independence with $\Delta=0.30$):

```text
n = 140; fits and Hessian: 38 s; information: positive definite True, condition number 1.95e+10
links (to, from): [(0, 0), (1, 1), (2, 0), (2, 1), (2, 2), (3, 1), (3, 2), (3, 3)]
a0 (TD)     : [-0.426 -0.142  0.175  0.269 -0.348 -0.017  0.362 -0.323]
a1 (PIGD)   : [-0.519 -0.657  0.15   0.377 -0.492  0.482  0.491 -0.4  ]
own rates: estimate [-0.426 -0.142 -0.348 -0.323] | mean of draws [-0.454 -0.147 -0.357 -0.328] (draws are -exp(normal), so not centred on the estimate)
95% interval of TD's tremor own rate: basic [-0.629, -0.039], percentile [-0.812, -0.223]
```

An own rate's draws are $-e^{N(\hat\mu,v)}$. Their mean is $-e^{\hat\mu+v/2}$, more negative than the estimate $-e^{\hat\mu}$ (−0.454 against −0.426), and their distribution has a long tail towards more negative values. The percentile interval $[-0.812,-0.223]$ is, up to Monte Carlo error, $-\exp(\hat\mu\pm1.96\,se)$: the Wald interval on the raw scale, mapped back. The basic interval reflects the skew to the wrong side, $[-0.629,-0.039]$, nearly touching 0. The condition number 1.95e+10 exceeds the $10^{10}$ threshold, so `cov_info["well_conditioned"]` is false here: the small panel leaves a nearly flat direction.

**Design choices and pitfalls.**
- Directions with non-positive information are dropped, which understates the uncertainty along them. The report travels in `LinkResults.extra["cov_info"]`.
- The skew of the own-rate draws matters wherever a basic interval of an own-rate quantity is used: the non-zero and equivalence tests of §5.31 if own rates join a family, and the own-rate deviations in `speed_and_load_only` (Notes, item 8). Cross-links are drawn on their natural scale, where the draws are normal and the basic and percentile intervals agree.
- `group=1` hard-codes two phenotypes.

### 5.26 `_mc_p` — Monte Carlo p-values

[`localise.py:144`](../../src/pdx/localise.py#L144)

**What it does.** Two-sided Monte Carlo p-values for $H_0:\Delta_k=0$, from draws centred on the estimate.

**Inputs → outputs.** Estimates $\hat\Delta$ $(k)$ and draws $\Delta^{(m)}$ $(M\times k)$ → $k$ p-values, NaN where the estimate is not finite.

**The mathematics.** The draws approximate the sampling distribution of $\hat\Delta$ around the truth, shifted to centre at $\hat\Delta$. Under $H_0:\Delta_k=0$ the estimator's error is $\hat\Delta_k-0$, distributed approximately like $\Delta^{(m)}_k-\hat\Delta_k$. So

$$
p_k=\frac{1+\#\{m:\ |\Delta^{(m)}_k-\hat\Delta_k|\ge|\hat\Delta_k|\}}{M+1},
$$

the two-sided tail probability of $|\hat\Delta_k|$ under the centred draws. The $+1$ in numerator and denominator counts the observed value as one of the draws, which keeps $p\ge1/(M+1)$ and the test valid at Monte Carlo resolution. This is the inference of Chen Y (2023) for a median-type speed.

*From the maths to the code.* Line 149 is the formula, vectorised over links; line 150 sets NaN for non-finite estimates.

**Worked example** ([`ex_link_classes.py`](../guide_examples/s05/ex_link_classes.py)):

```text
Monte Carlo p of (1,0) by hand: (1 + #|D* - Delta| >= |Delta|)/(M + 1) = 0.0002; _mc_p = 0.0002
```

None of the 4000 centred draws reaches $|\hat\Delta|=0.12$, so $p=1/4001=0.00025$.

**Design choices and pitfalls.**
- The reference treats the error distribution as the same at $\Delta=0$ as at $\hat\Delta$ (a translation family). That is reasonable near 0 and less so for entries with skewed draws.
- **Resolution.** With $M=2000$ the smallest p-value is $1/2001\approx0.0005$. Benjamini–Yekutieli (BY, §5.44) rejects a lone strong link only if $p\le\alpha/(m\,c(m))$, which falls below 0.0005 once the family has 26 or more links ($c(26)=3.85$). A single strongly deviating link in a family that large could then never be rejected (Notes, item 13). The provisional map has 14 links, for which the threshold is 0.0011.

### 5.27 `_basic_ci` and `_opt_kw` — basic intervals from draws, and fit options

[`localise.py:153`](../../src/pdx/localise.py#L153), [`localise.py:159`](../../src/pdx/localise.py#L159)

**What it does.** `_basic_ci(est, draws, level)` returns basic intervals from Monte Carlo draws. `_opt_kw` keeps only the fit options the joint and penalised fits accept (`maxiter`, `chunk`, `ftol`, `gtol`).

**Inputs → outputs.** Estimates, draws and a level → `(lower, upper)` arrays. A dict → a filtered dict.

**The mathematics.** With $c^{(m)}=\Delta^{(m)}-\hat\Delta$ and $a=(1-\text{level})/2$:

$$
\big[\hat\Delta-q_{1-a}(c),\ \hat\Delta-q_a(c)\big]=\big[2\hat\Delta-q_{1-a}(\Delta^{(m)}),\ 2\hat\Delta-q_a(\Delta^{(m)})\big],
$$

the same construction as the bootstrap's basic interval (§5.18), with draws in place of samples. A 90% basic interval's ends are the two one-sided 95% bounds used by the equivalence test (two one-sided tests, TOST; §5.31).

*From the maths to the code.* Lines 154–156.

**Worked example** ([`ex_link_classes.py`](../guide_examples/s05/ex_link_classes.py)):

```text
TOST for (2,1): lower 95% bound -0.0170 > -0.10 and upper 95% bound 0.0184 < 0.10; basic 90% interval [-0.0170, 0.0184]
```

**Design choices and pitfalls.** For quantities whose draws are skewed (own rates, §5.25), the basic interval is reflected against the skew. `_opt_kw` silently drops other options, for example `own_free_start`.

### 5.28 `SpeedView` and `LinkResults` — containers for the verdicts

[`localise.py:166`](../../src/pdx/localise.py#L166), [`localise.py:177`](../../src/pdx/localise.py#L177)

**What it does.** `SpeedView` holds everything a verdict needs at one speed: the speed, $\hat\Delta$ and its draws, $a_1/s$ and its draws, and the p-values. `LinkResults` holds:
- the links (`(to, from)` pairs) and the own-rate mask;
- $\hat s_{L1}$, the speed set and the diagnostics;
- the main view, at $\hat s_{L1}$ with the speed's uncertainty, and the probe views at fixed speeds;
- $a_0$ and its draws, the family mask, the draws, `speed_identified` and `extra`.

Properties: `delta` and `p` (of the main view), `probe_speeds`, and `ci90`, the basic 90% interval of the main view's $\hat\Delta$.

**Inputs → outputs.** Built by `per_link_tests`; read by `classify_links`, `speed_and_load_only` and `per_domain_tests`.

**The mathematics.** None.

**Worked example.** See §5.29.

**Design choices and pitfalls.** `a0_draws` holds only the draws that identify the speed, while `draws.d0` holds all of them. The main view uses the former and the probes the latter (§5.32).

### 5.29 `per_link_tests` — the anchor-free per-link tests

[`localise.py:209`](../../src/pdx/localise.py#L209)

**What it does.** Protocol §7.3.3, primary version:
1. draw the free fit;
2. identify the speed by the minimal-L1 rule;
3. compute each link's speed-adjusted deviation and its Monte Carlo p-value;
4. form the set of speeds the data cannot tell apart;
5. re-derive every quantity at probe speeds across that set.

**Inputs → outputs.** The free fit, the panel, $M$ (2000), a generator, `near_zero`, `prep`, `cov`, `family` (default: the cross-links), `n_probe` (7), `set_level` (0.95), `include_diagnostics` (True) and `draws` (precomputed `FreeDraws`) → `LinkResults`.

**The mathematics.**
- $\hat s=\hat s_{L1}(a_0,a_1)$ and, for each draw, $s^{(m)}=\hat s_{L1}(a_0^{(m)},a_1^{(m)})$.
- Deviations on the reference scale: $\hat\Delta=a_1/\hat s-a_0$ and $\Delta^{(m)}=a_1^{(m)}/s^{(m)}-a_0^{(m)}$. Each draw uses its own speed, so the main view's draws carry the speed's uncertainty.
- p-values $p_k$ by §5.26.
- Speed set: $[q_{0.025}(s^{(m)}),\ q_{0.975}(s^{(m)})]$, widened to cover the diagnostic speeds except least squares.
- Probe speeds: 7 points spaced geometrically across the set, plus $\hat s$ (speed acts multiplicatively, hence geometric spacing). At a probe speed $s$, $\Delta(s)=a_1/s-a_0$ with draws $a_1^{(m)}/s-a_0^{(m)}$: the speed is held fixed.

Why the reference scale: in the joint model $A_1=s(A+D)$, so $\Delta=D$, a deviation per year at the reference speed, comparable with $\varepsilon$.

*From the maths to the code.*
- Line 223: the draws (or the ones given).
- Lines 224–226: the own-rate mask and the family (default `~own`).
- Lines 227–228: $\hat s$ and the diagnostics.
- Lines 230–235: if $\hat s$ is undefined, every output is NaN and `speed_identified=False`.
- Lines 236–238: $s^{(m)}$; only draws with a finite speed are kept.
- Lines 239–241: $\hat\Delta$, $\Delta^{(m)}$ and the main view.
- Lines 242–246: the speed set and its widening by the diagnostics.
- Lines 247–250: the probes, `np.unique(np.r_[np.geomspace(lo, hi, n_probe), s_hat])`.
- Lines 251–252: the result, with `extra["draws_used"]` and the covariance report.

**Worked example 1** ([`ex_link_classes.py`](../guide_examples/s05/ex_link_classes.py); hand-made estimates with draws = estimate + N(0, 0.01²), $M=4000$). The entries: four own rates (−0.25 in TD, −0.5 in PIGD); (1,0) stronger beyond speed (0.20 → 0.64, so $\Delta=0.64/2-0.20=0.12$); (2,0) present only in PIGD (0 → 0.50); three links that differ by speed alone (0.15 → 0.30):

```text
s_hat = 2.000; speed set = (1.920, 2.118); probe speeds = [1.92  1.951 1.984 2.    2.016 2.05  2.084 2.118]
diagnostics: {'all': 2.0, 'own_rates': 2.0, 'cross_links': 2.0, 'mode': 2.0, 'least_squares': 2.134}
link         a0     a1   Delta       p   ci90             verdict
(0, 0)    -0.25  -0.50   0.000  1.0000   [-0.020, 0.017] outside the family
(1, 0)     0.20   0.64   0.120  0.0002   [ 0.099, 0.145] different strength beyond speed
(1, 1)    -0.25  -0.50   0.000  1.0000   [-0.020, 0.016] outside the family
(2, 0)     0.00   0.50   0.250  0.0002   [ 0.231, 0.273] structurally different
(2, 1)     0.15   0.30   0.000  1.0000   [-0.017, 0.018] similar
(2, 2)    -0.25  -0.50   0.000  1.0000   [-0.021, 0.018] outside the family
(3, 1)     0.15   0.30   0.000  1.0000   [-0.018, 0.019] similar
(3, 2)     0.15   0.30   0.000  1.0000   [-0.017, 0.019] similar
(3, 3)    -0.25  -0.50   0.000  1.0000   [-0.020, 0.017] outside the family
```

The weighted median is exactly 2, so the shared entries have $\hat\Delta=0$ and $p=1$. The entry (2,0) has $a_0=0$, so it cannot vote; its deviation $0.50/2=0.25$ is still estimated and tested.

**Worked example 2** ([`ex_localise_fit.py`](../guide_examples/s05/ex_localise_fit.py); the real free fit of §5.25):

```text
minimal-L1 speed 1.358; speed set (0.784, 2.049); 2000 of 2000 draws identify the speed
diagnostics: {'all': 1.358, 'own_rates': 1.239, 'cross_links': 1.358, 'mode': 1.326, 'least_squares': 1.38}
  tremor -> function     Delta  -0.064 (true  0.00)  p 0.5247  ci90 [-0.241, 0.097]  unresolved
   axial -> function     Delta   0.009 (true  0.00)  p 0.9415  ci90 [-0.184, 0.140]  unresolved
   axial -> independence Delta   0.371 (true  0.30)  p 0.0235  ci90 [ 0.085, 0.556]  unresolved
  function -> independence Delta   0.000 (true  0.00)  p 1.0000  ci90 [-0.213, 0.130]  unresolved
per-domain p: [0.7956 0.0725 0.8636 0.033 ] ['no difference detected', 'no difference detected', 'no difference detected', 'no difference detected']
speed and load only: False
```

With 140 participants the speed set runs from 0.78 to 2.05. The true deviation (axial → independence) is estimated at 0.37 with $p=0.024$. That does not survive Benjamini–Yekutieli across the four cross-links: the smallest p-value must be below $0.05/(4\times2.083)=0.006$. So every link is "unresolved", the outcome the protocol expects for many links at PPMI's sizes (§2.3). The function → independence link has $\hat\Delta=0.000$ exactly: its ratio is the weighted median.

**Design choices and pitfalls.**
- Dropping the draws whose speed is undefined conditions on identification; `extra["draws_used"]` records how many were kept.
- If the point speed is defined but **no** draw's speed is, line 243 (`np.quantile` of an empty array) raises `IndexError` (Notes, item 2).
- The main view carries the speed's uncertainty; the probe views do not. A probe view's intervals are therefore narrower, but centred at a different speed.

### 5.30 `_reject`, `_nonzero` and `_within` — three helpers

[`localise.py:255`](../../src/pdx/localise.py#L255), [`localise.py:267`](../../src/pdx/localise.py#L267), [`localise.py:271`](../../src/pdx/localise.py#L271)

**What it does.**
- `_reject(p, method, alpha)` returns the rejections of the family under `"holm"`, `"bh"`, `"by"` or `"none"` ($p\le\alpha$), and raises `ValueError` otherwise.
- `_nonzero(lo, hi)` is `(lo > 0) | (hi < 0)`: the interval excludes 0.
- `_within(lo, hi, eps)` is `(lo > -eps) & (hi < eps)`: the interval lies strictly inside $(-\varepsilon,\varepsilon)$.

**Inputs → outputs.** Arrays → boolean arrays.

**The mathematics.** A 95% interval that excludes 0 is a two-sided 5% test of $a\ne0$. A 90% interval inside $\pm\varepsilon$ is the equivalence test of §5.31.

**Worked example.** Used in §5.31.

**Design choices and pitfalls.** The inequalities are strict, so an interval ending exactly at $\pm\varepsilon$ is not "within".

### 5.31 `_classes_at` — the class rules at one speed

[`localise.py:275`](../../src/pdx/localise.py#L275)

**What it does.** Assigns each support entry one of the classes of protocol §2.3, at one speed view.

**Inputs → outputs.** A `SpeedView`, $a_0$ and its draws, the family mask, $\varepsilon$, the multiplicity method and $\alpha$ → a list of class labels.

**The mathematics.**

*The equivalence test (two one-sided tests, TOST).* For $H_0:|\Delta|\ge\varepsilon$ against $H_1:|\Delta|<\varepsilon$, test $H_0^-:\Delta\le-\varepsilon$ and $H_0^+:\Delta\ge\varepsilon$, each one-sided at 5%. Reject $H_0^-$ if the lower one-sided 95% bound exceeds $-\varepsilon$; reject $H_0^+$ if the upper one-sided 95% bound is below $\varepsilon$. $H_0$ is the union of $H_0^-$ and $H_0^+$, so rejecting it requires rejecting both, and the level is at most 5% (§5.33). The two one-sided 95% bounds are the ends of the two-sided 90% interval, so the test is "the 90% interval lies inside $\pm\varepsilon$".

*The rules, as coded.* For a family link $k$, define:
- $R_k$: link $k$ is rejected by the family procedure (Benjamini–Yekutieli by default) on the p-values $p_k$;
- $NZ_{0k}$, $NZ_{1k}$: the 95% basic interval of $a_{0k}$, respectively of $a_{1k}/s$, excludes 0;
- $EQ_{0k}$, $EQ_{1k}$: the 90% basic interval of $a_{0k}$, respectively of $a_{1k}/s$, lies inside $\pm\varepsilon$;
- $\mathrm{sign}_{0k}$, $\mathrm{sign}_{1k}$: the sign of the midpoint of those 95% intervals, which is the sign of the whole interval whenever it excludes 0;
- $SIM_k$: the simultaneous 90% interval of $\Delta_k$ over the family (max-|t|, §5.45) lies inside $\pm\varepsilon$.

| condition | class |
|---|---|
| $R_k$ and $NZ_{0k}$ and $NZ_{1k}$ and same sign | different strength beyond speed |
| $R_k$ and [$NZ_{0k}$ and $EQ_{1k}$, or $NZ_{1k}$ and $EQ_{0k}$, or $NZ_{0k}$ and $NZ_{1k}$ with opposite signs] | structurally different |
| $R_k$ otherwise | different, type unresolved |
| not $R_k$, and $SIM_k$ | similar |
| otherwise | unresolved |

A support entry outside the family is "outside the family". The rows follow protocol §2.3: "non-zero with the same sign in both phenotypes at a common speed"; "non-zero in one phenotype with its 90% interval within ±ε in the other, or opposite signs".

*From the maths to the code.*
- Lines 277–279: the rejections within the family.
- Lines 280–283: simultaneous 90% intervals of $\Delta$ over the family.
- Lines 284–287: the 95% and 90% basic intervals of $a_0$ and $a_1/s$.
- Lines 288–290: $NZ$, $EQ$ and the signs.
- Lines 291–308: the table above, row by row.

**Worked example** ([`ex_link_classes.py`](../guide_examples/s05/ex_link_classes.py); a hand-made view with six family links, the rejections set by hand with `method="none"`; $a_0=0.20$ or $0.15$ with SD 0.02):

```text
class rules on a hand-made speed view (rejections set by hand, method 'none')
  both non-zero, same sign   -> different strength beyond speed
  PIGD within +-eps          -> structurally different
  opposite signs             -> structurally different
  PIGD interval wide         -> different, type unresolved
  not rejected, narrow       -> similar
  not rejected, wide         -> unresolved
```

By hand, row 2: $a_1/s=0.02$ with SD 0.02 has a 90% interval of about $[-0.013,0.053]$, inside $\pm0.10$ ($EQ_1$), while TD's 0.20 is clearly non-zero ($NZ_0$): structurally different. Row 4: $a_1/s=0.10$ with SD 0.15 has a 95% interval of about $[-0.19,0.39]$ (not $NZ_1$) and a 90% interval of about $[-0.15,0.35]$ (not $EQ_1$), so neither rule holds.

**Design choices and pitfalls.**
- Rejection is checked first. A link that differs significantly but by less than $\varepsilon$ falls into a "different" class, never "similar".
- **Protocol difference.** Protocol §7.6 and the docstring of `multiplicity.py` say that "similar" *and* "structurally different" use simultaneous 90% intervals across the family. The code uses simultaneous intervals for "similar" only. The equivalence part of "structurally different" uses marginal 90% intervals, and its non-zero parts marginal 95% intervals, as `classify_links`' docstring says (Notes, item 5).
- The non-zero and equivalence tests on $a_0$ and $a_1/s$ are not adjusted for multiplicity.

### 5.32 `classify_links` — classes that must hold across the speed set

[`localise.py:311`](../../src/pdx/localise.py#L311)

**What it does.** Applies `_classes_at` at $\hat s_{L1}$ (with the speed's uncertainty) and at every probe speed. It reports a class only if it holds across the whole speed set; otherwise the verdict is "speed and mechanism not separable".

**Inputs → outputs.** `LinkResults`, $\varepsilon$, the method (default `"by"`), $\alpha$ and an optional family → one dict per support entry: `link`, `own_rate`, `class` (at $\hat s$), `stable_across_speeds`, `verdict`, `delta`, `p`, `ci90` and `max_deviation_allowed`.

**The mathematics.** Let $C_k(s)$ be link $k$'s class at speed $s$. The verdict is

$$
\text{verdict}_k=\begin{cases}C_k(\hat s) & \text{if } C_k(\hat s)\in\{\text{unresolved, outside}\},\ \text{or } C_k(s)=C_k(\hat s) \text{ at every probe } s\ \text{(and, for a "different" class, } \mathrm{sign}\,\Delta_k(s) \text{ is constant)},\\ \text{not separable} & \text{otherwise.}\end{cases}
$$

`max_deviation_allowed` is $\max(|lo_k|,|hi_k|)$ of the marginal 90% interval at $\hat s$: "the largest deviation the data allow" for an unresolved link (protocol §2.3).

*From the maths to the code.*
- Lines 323–327: an unidentified speed gives "speed not identified" everywhere.
- Line 328: the classes at $\hat s$, with `res.a0_draws` (the identified draws).
- Line 329: the classes at each probe, with `res.draws.d0` (all draws).
- Line 330: `ci90`.
- Lines 332–338: the stability rule.
- Lines 339–344: the output.

**Worked example** ([`ex_link_classes.py`](../guide_examples/s05/ex_link_classes.py)). The table of §5.29 is the stable case. The next two blocks show what stability and simultaneity do:

```text
a marginal 90% interval inside +-eps is not enough for 'similar' (simultaneous over 6 links)
  (1, 0)  marginal 90% [-0.084, 0.083]  simultaneous [-0.119, 0.119]  -> unresolved / unresolved
  (2, 0)  marginal 90% [-0.084, 0.079]  simultaneous [-0.117, 0.117]  -> unresolved / unresolved
  (2, 1)  marginal 90% [-0.083, 0.081]  simultaneous [-0.119, 0.119]  -> unresolved / unresolved
  (3, 0)  marginal 90% [-0.086, 0.081]  simultaneous [-0.118, 0.118]  -> unresolved / unresolved
  (3, 1)  marginal 90% [-0.082, 0.078]  simultaneous [-0.116, 0.116]  -> unresolved / unresolved
  (3, 2)  marginal 90% [-0.084, 0.080]  simultaneous [-0.118, 0.118]  -> unresolved / unresolved

diagnostics that disagree make the verdict 'not separable'
  diagnostics: {'all': 2.0, 'own_rates': 2.0, 'cross_links': 3.2, 'mode': 2.001, 'least_squares': 2.192} speed set: [1.922 3.2  ]
  (1, 0): class at s_hat 'different strength beyond speed', stable False -> speed and mechanism not separable
  (2, 1): class at s_hat 'similar', stable False -> speed and mechanism not separable
```

- First block: six links that differ by speed alone, each with draw SD 0.045. Every marginal 90% interval (the reported `ci90`) lies inside $\pm0.10$, but the simultaneous intervals over six links reach $\pm0.12$, so every link is "unresolved". Its `max_deviation_allowed` (about 0.085) would nevertheless suggest a deviation within $\varepsilon$ (Notes, item 6).
- Second block: three domains. The own rates say the speed is 2, but the cross-links alone say 3.2, so the speed set runs to 3.2. At $s=3.2$ the (1,0) link has $\Delta=0.64/3.2-0.20=0$, so its class changes. The (2,1) link has $\Delta=0.30/3.2-0.15=-0.056$, which is significantly non-zero with draws this precise. Both verdicts become "not separable".

**Design choices and pitfalls.**
- The default method is Benjamini–Yekutieli (exploratory, §7.3.3). Step 5's confirmatory family would call `classify_links(res, eps, "holm", family=…)`; nothing in the code defines that family yet.
- The reported `ci90` is marginal, while "similar" is judged on the simultaneous interval (first block above).

### 5.33 `speed_and_load_only` — the intersection–union verdict

[`localise.py:348`](../../src/pdx/localise.py#L348)

**What it does.** Returns True if every support entry's 90% interval of $\Delta$ lies inside $\pm\varepsilon$, at $\hat s_{L1}$ and at every probe speed: the overall verdict "speed and load only" (protocol §2.3).

**Inputs → outputs.** `LinkResults` and $\varepsilon$ → a bool; False if the speed is not identified.

**The mathematics.** The intersection–union test (IUT; Berger 1982). The claim is $H_1:|\Delta_k|<\varepsilon$ for every $k$, so the null is the union $H_0=\bigcup_k\{|\Delta_k|\ge\varepsilon\}$. Reject $H_0$ only if every individual equivalence test rejects at level $\alpha$. For any parameter value in $H_0$ there is some $k^*$ with $|\Delta_{k^*}|\ge\varepsilon$, and

$$
P(\text{reject }H_0)\le P(\text{TOST}_{k^*}\text{ rejects})\le\alpha ,
$$

so the overall test has level $\alpha$ with no multiplicity adjustment: each additional link can only make the claim harder.

*From the maths to the code.* Line 353: the main view and the probes. Line 354: `_within(*_basic_ci(v.delta, v.d_draws, 0.90), eps)` for every entry in every view.

**Worked example** ([`ex_link_classes.py`](../guide_examples/s05/ex_link_classes.py)):

```text
speed and load only (BASE): False
speed and load only (pure speed, noise 0.01): True
```

**Design choices and pitfalls.**
- Every support entry is tested, own rates included, while the per-link family defaults to the cross-links. The protocol says "every link's 90% interval" (Notes, item 7).
- Requiring the verdict at every probe speed is stricter than the IUT needs; it is the speed-set rule of §7.1.
- The intervals are marginal basic ones. For own-rate entries they inherit the skew problem of §5.25.

### 5.34 `_domain_p` and `per_domain_tests` — H-D, the per-domain tests

[`localise.py:357`](../../src/pdx/localise.py#L357), [`localise.py:369`](../../src/pdx/localise.py#L369)

**What it does.** For each domain, tests all deviations into it at once (own rate included) with a Mahalanobis statistic and a Monte Carlo reference, then applies Benjamini–Yekutieli across domains. A domain "differs beyond speed" only if it is rejected at $\hat s_{L1}$ and at every probe speed.

**Inputs → outputs.** `per_domain_tests(res, spec, alpha)` → `{"p": [d p-values at ŝ], "reject_by": [d], "verdict": [d]}`.

**The mathematics.** For domain $j$ let $L_j=\{k:\text{to}(k)=j\}$, its incoming links and its own rate. With $\hat\Sigma_j$ the sample covariance of the draws $\Delta^{(m)}_{L_j}$:

$$
T_j=\hat\Delta_{L_j}^\top\hat\Sigma_j^{+}\hat\Delta_{L_j},\qquad T_j^{(m)}=\big(\Delta^{(m)}_{L_j}-\hat\Delta_{L_j}\big)^\top\hat\Sigma_j^{+}\big(\Delta^{(m)}_{L_j}-\hat\Delta_{L_j}\big),
$$

$$
p_j=\frac{1+\#\{m:T_j^{(m)}\ge T_j\}}{M+1}.
$$

Under $H_0:\Delta_{L_j}=0$, $\hat\Delta_{L_j}\approx N(0,\Sigma_j)$ and the centred draws $\approx N(0,\Sigma_j)$, so $T_j$ and $T^{(m)}_j$ share one distribution (about $\chi^2_{|L_j|}$). The p-values of the $d$ domains then go through Benjamini–Yekutieli (§5.44). Verdicts: rejected at $\hat s$ and at every probe → "differs beyond speed"; at $\hat s$ only → "speed and mechanism not separable"; otherwise "no difference detected".

*From the maths to the code.*
- Lines 361–365: for each domain, `L`, the centred draws `D`, `Si = pinv(cov(D.T))`, `t_obs`, and the Monte Carlo count via `einsum`.
- Line 376: the p-values at $\hat s$; line 377: BY.
- Lines 378–380: the rejections must also hold at every probe.
- Lines 381–384: the verdicts.

**Worked example** ([`ex_link_classes.py`](../guide_examples/s05/ex_link_classes.py); the hand-made entries of §5.29):

```text
per-domain p: [1.e+00 2.e-04 2.e-04 1.e+00] verdicts: ['no difference detected', 'differs beyond speed', 'differs beyond speed', 'no difference detected']
domain 2 by hand: entries [(2, 0), (2, 1), (2, 2)], T = 376.6, p = 0.00025 (code 0.00025)
```

Domains 1 and 2 receive the two deviating links. Domain 2's statistic over its three entries is 376.6, which no centred draw reaches, so $p=1/4001$, the same as the code.

**Design choices and pitfalls.**
- The covariance used in $T_j$ comes from the same draws as the reference, which is fine for a Monte Carlo test.
- `pinv` handles a singular covariance (for example a degenerate own-rate draw).
- Only the p-values at $\hat s$ are reported; the probe p-values enter only through the verdict.

### 5.35 `recalibrate` — per-link p-values against their bootstrap null

[`localise.py:387`](../../src/pdx/localise.py#L387)

**What it does.** Recalibrates observed per-link p-values against their null distribution from the H-P bootstrap replicates, as protocol §7.3.3 requires when those null p-values are not uniform.

**Inputs → outputs.** `p_obs` $(k)$ and `p_null` $(B\times k)$, one row per replicate → $k$ recalibrated p-values.

**The mathematics.** The H-P replicates are simulated from the speed-only fit, where every $\Delta=0$. The per-link p-values computed on them are draws from each link's null distribution of $p$. The recalibrated p-value is that empirical distribution function at the observed p-value, with the Monte Carlo $+1$:

$$
\tilde p_k=\frac{1+\#\{b:p^*_{bk}\le p_k\}}{B+1}.
$$

If the null p-values are uniform, $\tilde p_k\approx p_k$.

*From the maths to the code.* Line 391 is the formula, vectorised over links.

**Worked example** ([`ex_recalibrate.py`](../guide_examples/s05/ex_recalibrate.py); made-up null p-values for three links: uniform, too small (Beta(0.5, 1)) and too large (Beta(2, 1))):

```text
share of null p-values <= 0.05 per link: [0.045 0.216 0.002]
observed p      : [0.03 0.03 0.03]
recalibrated p  : [0.034 0.172 0.001]
by hand, link 1 : (1 + #{p_null <= 0.03}) / (B + 1) = 0.172
```

An observed 0.03 stays 0.034 for the calibrated link. It becomes 0.172 for the anti-conservative link, whose null p-values fall below 0.05 in 22% of replicates, and 0.001 for the conservative one.

**Design choices and pitfalls.**
- Nothing calls this function. `run_hp(on_replicate=…)` can return per-replicate output, but the parallel path used by `run_hp_real.py` drops it (§5.6), and no script passes such a hook (Notes, item 9).
- Recalibration uses the replicates of the speed-only null, so it calibrates the per-link test under "no mechanism difference at all", not under "other links differ".

### 5.36 `free_to_joint` and `_d_slice` — the free fit in the joint model's parameters

[`localise.py:396`](../../src/pdx/localise.py#L396), [`localise.py:428`](../../src/pdx/localise.py#L428)

**What it does.** Writes the free fit in the joint model's parameters, at a chosen speed $s$ for PIGD. Used as the start of the anchored fit (§5.37); it also shows the speed–deviation trade-off directly. `_d_slice` returns the slice of the deviation block `D{group}`.

**Inputs → outputs.** The free fit, $s$ and the group → a joint-layout parameter vector.

**The mathematics.** In the free model PIGD has drift $A_1$, load $b_1$, load spread $\tau_1$ and noise $Q_1$, all on the observed time scale, and participant factor $\tilde s_i$. The joint model runs PIGD at $s\,\tilde s_i$ with drift $A+D$, load $b_1'$, participant load $\ell_i'$ and noise $Q_1'$. The two give the same process when

$$
s(A+D)=A_1,\quad s\,b_1'=b_1,\quad s\,\ell_i'=\ell_i,\quad s\,Q_1'=Q_1
\;\Longrightarrow\;
D=\frac{A_1}{s}-A,\quad b_1'=\frac{b_1}{s},\quad \Gamma'=\frac{\Gamma}{s},\ \lambda'=\frac{\lambda}{s},\ \log\tau'=\log\tau-\log s,\quad Q_1'=\frac{Q_1}{s},
$$

where $\Gamma$ stands for the load-mean coefficients `gam_y` and `gam_z`. The treatment terms are in the measurement equation, not in the dynamics, so they are not rescaled. The joint likelihood at `free_to_joint(free, s)` therefore equals the free likelihood for **every** $s>0$: this is the flat direction that makes localisation need an assumption.

*From the maths to the code.*
- Lines 402–404: copy every block the two layouts share (`A` raw, `beta`, `theta`, `log_sigma`, …).
- Lines 405–407: $\gamma=\log s$ for the group.
- Lines 408–411: $D_g=a_g/s_g-a_0$, with $s_g=s$ for the chosen group and 1 for any other.
- Lines 412–416: $b_1/s$, and $Q_1/s$ re-packed as a Cholesky factor.
- Lines 417–421: `gam_y`, `gam_z`, `lam` of the group divided by $s$.
- Lines 422–424: $\log\tau_1-\log s$.

**Worked example** ([`ex_joint.py`](../guide_examples/s05/ex_joint.py); engine panel of 95 participants, PIGD at speed 2 plus axial → independence, $\Delta=0.30$):

```text
n = 95; fits 18 s; free log-likelihood -711.9459
links (to, from): [(0, 0), (1, 1), (2, 0), (2, 1), (2, 2), (3, 1), (3, 2), (3, 3)]
joint model at speed 1.0: log-likelihood -711.9459; D = [-0.127 -0.153  0.046  0.008 -0.052  0.565 -0.09   0.012]
joint model at speed 1.5: log-likelihood -711.9459; D = [-0.014  0.028 -0.024 -0.125  0.076  0.396 -0.167  0.094]
joint model at speed 2.5: log-likelihood -711.9459; D = [ 0.076  0.172 -0.08  -0.231  0.178  0.261 -0.228  0.159]
```

Three different deviation patterns, one likelihood. At speed 1 the axial → independence link and the tremor and axial own rates carry most of the difference; at speed 2.5 every entry deviates by at least 0.07. Nothing in the data prefers one pattern to another.

**Design choices and pitfalls.** `pack_chol` floors the eigenvalues of $Q_1/s$ at $10^{-6}$, so for an extreme $s$ the noise could be changed slightly, and the likelihood with it.

### 5.37 `anchored_tests` — the anchored per-link tests (sensitivity)

[`localise.py:434`](../../src/pdx/localise.py#L434)

**What it does.** Protocol §7.3.3, sensitivity version: fits the joint model with every cross-link deviation free and every own-rate deviation fixed at zero ($D_{jj}=0$), which anchors the speed to the own rates, then runs Wald tests.

**Inputs → outputs.** The panel, the spec, the free fit, `prep`, `near_zero`, fit options and $\alpha$ → `speed`, `delta`, `se`, `p` (NaN for own rates), `reject_by`, `p_domain`, `converged`, `loglik` and `information` (the covariance report).

**The mathematics.** With $D_{jj}=0$, every own rate obeys $a_{1jj}=s\,A_{jj}$, so the speed is identified, and over-identified once $d>1$; maximum likelihood finds the compromise. Then:
- the constrained maximum-likelihood estimate $(\hat\gamma,\hat D,\dots)$ with its covariance $\hat V$ from the inverse observed information over the free coordinates;
- for each cross-link, $z_k=\hat D_k/\widehat{se}_k$ and $p_k=2\Phi(-|z_k|)$;
- for each domain, the Wald statistic over its incoming cross-links $L$, $W_j=\hat D_L^\top\hat V_{LL}^{-1}\hat D_L\sim\chi^2_{|L|}$;
- Benjamini–Yekutieli over the cross-link p-values.

*From the maths to the code.*
- Line 444: start at the own-rate minimal-L1 speed (or $\max(\hat s_{LS},0.1)$ if it is undefined).
- Lines 445–447: `free_to_joint` at that speed, with the own-rate $D$ set to 0.
- Lines 448–451: bounds $(0,0)$ fix the own-rate $D$ (L-BFGS-B accepts fixed variables); `fit_model(…, "joint", bounds=…)`.
- Lines 452–456: the Hessian, restricted to the free coordinates, and `covariance_checked`.
- Lines 457–461: $\hat D$, the standard errors, $z$ and two-sided normal p-values.
- Lines 462–467: the per-domain Wald tests; domains with no incoming cross-link keep $p=1$.
- Lines 468–469: BY over the cross-links.
- Lines 470–472: the output, with the speed $e^{\hat\gamma_1}$.

**Worked example** ([`ex_joint.py`](../guide_examples/s05/ex_joint.py)):

```text
anchored (10 s): speed 1.200, converged True
  (2, 0): D   0.014  se 0.093  p 0.8776
  (2, 1): D  -0.063  se 0.130  p 0.6255
  (3, 1): D   0.494  se 0.147  p 0.0008
  (3, 2): D  -0.085  se 0.075  p 0.2552
  per-domain Wald p: [1.     1.     0.8656 0.0034]
```

The anchored fit finds the true deviation (axial → independence, $p=0.0008$, which survives BY: $0.0008\le0.05/(4\times2.083)=0.006$). Domains 0 and 1 have no incoming cross-links, so their p-values stay at 1.

**Design choices and pitfalls.**
- The anchor is an assumption too: every domain's own rate shares the speed. If a phenotype changes a domain's persistence beyond speed, the anchored deviations are biased.
- The speed here (1.20, on 95 participants) is well below the truth (2); the anchored test inherits that error through $D=A_1/s-A$.

### 5.38 `_penalised_fit` and `_to_z` — the adaptive-lasso fit

[`localise.py:477`](../../src/pdx/localise.py#L477), [`localise.py:512`](../../src/pdx/localise.py#L512)

**What it does.** Fits the joint model with an adaptive-lasso penalty on the deviations, by splitting each deviation into positive and negative parts so that L-BFGS-B can handle the non-smooth penalty. `_to_z` maps a joint parameter vector into the split variables.

**Inputs → outputs.** The joint layout, `prep`, a start in split variables `z0`, $\lambda$, the weights $w$, an optional mask of deviations fixed at 0, and optimiser options → `(optimiser result, x, unpenalised log-likelihood)`.

**The mathematics.** The problem is

$$
\min_x\ -\ell(x)+\lambda\sum_kw_k|D_k| .
$$

Write $D_k=u_k-v_k$ with $u_k,v_k\ge0$ and solve the smooth, bound-constrained problem

$$
\min_{x_{-D},\,u\ge0,\,v\ge0}\ -\ell\big(x(u,v)\big)+\lambda\sum_kw_k(u_k+v_k).
$$

At a solution $\min(u_k,v_k)=0$: otherwise lowering both by $\min(u_k,v_k)$ leaves $D_k$ unchanged and cuts the penalty by $2\lambda w_k\min(u_k,v_k)$. So $u_k+v_k=|D_k|$, and the two problems have the same solutions. With $g=\nabla_x(-\ell)$, the gradients are $\partial/\partial u_k=g_{D_k}+\lambda w_k$ and $\partial/\partial v_k=-g_{D_k}+\lambda w_k$.

*From the maths to the code.*
- Lines 485–488: `to_x`, $D=u-v$, with $u$ stored in the `D` slice of `z` and $v$ appended after the end.
- Lines 490–495: the objective and its gradient, as above.
- Lines 497–503: bounds $u,v\ge0$ (both $(0,0)$ for a deviation fixed at 0).
- Lines 504–506: L-BFGS-B.
- Lines 507–509: back to $x$, and the *unpenalised* log-likelihood for the Bayesian information criterion (BIC, §5.39).
- `_to_z` (lines 513–517): $u=\max(D,0)$, $v=\max(-D,0)$.

**Worked example.** See the path in §5.39: at $\lambda_{max}$ every deviation is exactly 0, and the active set grows as $\lambda$ falls.

**Design choices and pitfalls.**
- Exact zeros come from the bounds, which is why `joint_discovery` can use a small tolerance ($10^{-6}$) to define the active set.
- `fixed_zero` is available but not used by `joint_discovery`.

### 5.39 `joint_discovery` — speed-adjusted joint discovery

[`localise.py:520`](../../src/pdx/localise.py#L520)

**What it does.** Protocol §7.3.2, exploratory: an adaptive lasso on the deviations of the joint model (own rates included, speed unpenalised), with weights from the free fits at the minimal-L1 speed, a path of $\lambda$ values, BIC selection and an unpenalised refit of the selected support. Two phenotypes only.

**Inputs → outputs.** The panel, the spec, both fits, `prep`, `n_lambda` (12), `ratio` ($10^{-3}$), `near_zero`, `n_units` (`"participants"` or transitions), `refit`, fit options and `tol` → `speed_weights`, `lambda_max`, `path` (per $\lambda$: log-likelihood, active set, $D$, BIC, convergence), `best`, `selected`, `identified` and, if refitted, `refit` (`D`, `speed`, `loglik`, `converged`).

**The mathematics.**
- *Weights.* $\tilde\Delta=a_1/\hat s_{L1}-a_0$ and $w_k=1/\max(|\tilde\Delta_k|,10^{-4})$: the median-initialised adaptive lasso (Windmeijer 2019).
- *Largest useful $\lambda$.* At the null fit written in the joint layout ($D=0$), the other parameters are already optimal, because the joint model with $D=0$ is the null model. So $D=0$ solves the penalised problem exactly when $|\partial(-\ell)/\partial D_k|\le\lambda w_k$ for every $k$, the lasso's optimality condition:

$$
\lambda_{max}=\max_k\frac{|g_k|}{w_k},\qquad g=\nabla_D(-\ell)\big|_{\text{null fit}} .
$$

  The code multiplies it by 1.01, and the path is $\lambda_{max}\times$ `geomspace(1, ratio, n_lambda)`.
- *BIC.* $\mathrm{BIC}(\lambda)=-2\hat\ell(\lambda)+\log(n_{eff})\cdot\mathrm{df}(\lambda)$, with df = the joint model's parameters other than $D$, plus the active deviations ($|\hat D_k|>10^{-6}$), and $n_{eff}$ = participants (protocol §7.3.2: once the random effects are integrated out, participants are the independent units).
- *Identification of the refit.* The refit fixes the unselected deviations at 0. If one of them sits on a clearly non-zero reference entry ($|a_{0k}|>c$), then $a_{1k}=s\,a_{0k}$ pins down $s$. Otherwise the refit would have a flat direction, and it is skipped.
- *The rule the penalty implies.* Along the flat direction (speed $s'$, $D(s')=a_1/s'-a_0$) the penalty is $\sum_kw_k|a_{1k}/s'-a_{0k}|=\sum_kw_k|a_{1k}|\,\big|1/s'-a_{0k}/a_{1k}\big|$, minimised when $1/s'$ is a weighted median of the inverse ratios $a_{0k}/a_{1k}$ with weights $w_k|a_{1k}|$. With adaptive weights the entries with small $|\tilde\Delta_k|$ dominate that median.

*From the maths to the code.*
- Lines 532–536: the speed for the weights (fallback $\max(\hat s_{LS},0.1)$) and the weights.
- Lines 537–539: the null fit in the joint layout (`null_to`), its gradient over `D`, $\lambda_{max}$.
- Lines 540–541: the path and $n_{eff}$.
- Lines 543–552: penalised fits down the path, each warm-started at the previous solution; active sets, df and BIC.
- Lines 553–559: the BIC choice, the identification check and the output.
- Lines 560–568: the unpenalised refit with `bounds` $(0,0)$ on the unselected deviations, started at the path solution with those deviations set to 0.

**Worked example** ([`ex_joint.py`](../guide_examples/s05/ex_joint.py)):

```text
adaptive weights at the minimal-L1 speed 1.158: [1.24e+01 1.26e+01 5.66e+01 2.15e+01 1.00e+04 2.00e+00 8.20e+00 2.21e+01]
joint discovery (24 s): lambda_max 17.450
  lambda  17.4497  loglik  -720.417  active [0 0 0 0 0 0 0 0]  BIC  1727.73
  lambda   1.7450  loglik  -713.364  active [0 0 0 0 0 1 0 0]  BIC  1718.18  <- best
  lambda   0.1745  loglik  -712.229  active [1 1 0 1 0 1 1 1]  BIC  1738.68
  lambda   0.0174  loglik  -711.953  active [1 1 1 1 0 1 1 1]  BIC  1742.68
  selected: [(3, 1)] | identified: True
  refit: speed 1.132, D = [0.    0.    0.    0.    0.    0.441 0.    0.   ], converged True
```

- At $\lambda_{max}$ nothing is active and the log-likelihood is the null's (−720.417).
- The BIC picks the second step, where only axial → independence (3,1) is active: the true deviation. The refit estimates it at 0.441 (truth 0.30) at speed 1.13.
- Entry (2,2), function's own rate, has weight $10^4$. Its ratio *is* the weighted median, so $\tilde\Delta=0$ exactly and the weight hits its cap. It stays inactive along the whole path, even at the smallest $\lambda$, where the free model's likelihood (−711.946) is almost reached. The penalised fits are in effect anchored on that single entry (Notes, item 14).

**Design choices and pitfalls.**
- The df counts $\gamma$ even when every deviation is active, one more than the identified dimension; it matters only at the end of the path.
- The weights depend on $\hat s_{L1}$, so joint discovery shares the minimal-L1 assumption.
- `n_units="transitions"` is the alternative penalty that WP3 compares (protocol §7.3.2).

### 5.40 `stability_selection` — selection frequencies over complementary halves

[`localise.py:572`](../../src/pdx/localise.py#L572)

**What it does.** Splits the participants into two random halves within phenotype, `B // 2` times; refits the null and free models on each half from the full-data fits; runs `joint_discovery` without the refit; and reports each support entry's selection frequency (Shah & Samworth 2013).

**Inputs → outputs.** The panel, the spec, both fits, `B` (50 halves, `[localisation] half_samples`), a generator, fit options and options for `joint_discovery` → `frequency`, `halves_used` and `links`.

**The mathematics.** Complementary-pairs stability selection. For $b=1,\dots,B/2$, draw a random split $(I_b,\bar I_b)$ of the participants, made within each phenotype so both halves keep the phenotype ratio. The frequency of entry $k$ is

$$
\hat\Pi_k=\frac1{B}\sum_{b=1}^{B/2}\big(\mathbf 1\{k\in\hat S(I_b)\}+\mathbf 1\{k\in\hat S(\bar I_b)\}\big).
$$

Shah & Samworth's error bounds need a threshold on $\hat\Pi_k$. The protocol asks only for the frequencies, and the code reports only them.

*From the maths to the code.*
- Lines 583–585: a permutation within each phenotype; the first half of each, and the rest.
- Lines 587–592: the sub-panel, warm-started null and free fits, and `joint_discovery(…, refit=False)`.
- Lines 593–596: a half that raises is skipped; selections are summed.
- Lines 597–598: frequencies over the halves used.

**Worked example** ([`ex_stability.py`](../guide_examples/s05/ex_stability.py); one complementary pair on the panel of §5.36, `n_lambda=3`):

```text
2 halves used (25 s)
  (0, 0): selected in 0% of halves
  (1, 1): selected in 0% of halves
  (2, 0): selected in 50% of halves
  (2, 1): selected in 0% of halves
  (2, 2): selected in 0% of halves
  (3, 1): selected in 50% of halves
  (3, 2): selected in 50% of halves
  (3, 3): selected in 0% of halves
```

With two halves the frequencies can only be 0, 50% or 100%. The protocol uses 50 halves.

**Design choices and pitfalls.**
- Each half starts from the full-data fits (a warm start), and the free fit on a half starts only there, not at the half's own null solution.
- A failed half is dropped silently; `halves_used` shows how many remained.

### 5.41 `commutator_test` — the shared-eigenvector check

[`localise.py:603`](../../src/pdx/localise.py#L603)

**What it does.** Protocol §7.3.7, a sensitivity analysis with no continuous-time model: do the two phenotypes' yearly least-squares matrices commute, as one mechanism at any speeds implies? It uses a centred participant bootstrap of the commutator's norm.

**Inputs → outputs.** A panel, $B$ (500) and a generator → `statistic`, `p`, `B`.

**The mathematics.** If $B_g=f_g(A)$ for power series $f_g$ (for example $e^{s_gA}$, or an average of $e^{sA}$ over a phenotype's participant speeds), then $B_0B_1=B_1B_0$, because two functions of one matrix commute. Different mechanisms generally break this. With $\hat B_g$ the yearly least-squares matrices (from `engine.fit._yearly_ols`: consecutive visits 0.5–1.6 years apart, complete rows, regression with intercept):

$$
\hat C=\hat B_0\hat B_1-\hat B_1\hat B_0,\qquad T=\|\hat C\|_F,\qquad T^*_b=\|\hat C^*_b-\hat C\|_F ,
$$

$$
p=\frac{1+\#\{b:T^*_b\ge T\}}{B+1}.
$$

Centring at $\hat C$ makes $T^*$ mimic the distribution of $\|\hat C-C\|$, which is the distribution of $T$ when $C=0$.

*From the maths to the code.*
- Lines 609–613: the two yearly matrices and the commutator.
- Lines 615–620: resampling within phenotype and the centred norms.
- Line 622: the p-value.

**Worked example** ([`ex_commutator.py`](../guide_examples/s05/ex_commutator.py); exact matrices first, then family-A panels of 543 participants):

```text
||B0 B1 - B1 B0||, same A at speeds 1 and 2: 3.10e-17
||B0 B1 - B1 B0||, one link stronger in PIGD : 0.0354
S0: n = 543, statistic 0.0344, p = 0.144 (2 s)
S1: n = 543, statistic 0.0267, p = 0.687 (2 s)
S2: n = 543, statistic 0.0477, p = 0.328 (3 s)
```

$e^{A}$ and $e^{2A}$ commute to rounding error; strengthening one link in PIGD breaks commutation (0.035). In these three datasets the check rejects neither S0 nor S1 (one mechanism), and it also misses S2's real difference at this size. The protocol's synthetic check found 28–78% power against real differences (§7.3.7): a supporting check, never a replacement for H-P.

**Design choices and pitfalls.**
- Measurement error, participant loads and phenotype-specific state covariances bias the least-squares matrices and can break commutation even under one mechanism (`docs/code_map.md`; WP3 measures it).
- If a phenotype has fewer than $d+5$ usable yearly transitions, `_yearly_ols` returns $0.8I$, which commutes with anything (Notes, item 19).
- Only the first two phenotypes are compared.

### 5.42 `embeddability_check` — complex or negative eigenvalues of yearly matrices

[`localise.py:625`](../../src/pdx/localise.py#L625)

**What it does.** For each phenotype, reports whether the yearly least-squares matrix has complex eigenvalues or non-positive real ones, and its smallest real part (protocol §6.3).

**Inputs → outputs.** A panel → `{group: {"complex", "negative_real", "min_real"}}`.

**The mathematics.** Culver (1966): a real non-singular matrix has a real logarithm if and only if each Jordan block of a negative eigenvalue occurs an even number of times, and the real logarithm is unique if and only if all eigenvalues are positive and real and no Jordan block repeats. A complex pair therefore means that several real drifts give the same yearly matrix (logarithms differ by $2\pi i k$ on the pair's eigenvalues). A negative real eigenvalue may mean that no real drift exists.

*From the maths to the code.* Line 630: `np.linalg.eigvals` of the yearly matrix. Lines 631–633: complex if $|\mathrm{Im}|>10^{-8}$; negative real if real part $\le0$ and $|\mathrm{Im}|\le10^{-8}$; the smallest real part.

**Worked example** ([`ex_commutator.py`](../guide_examples/s05/ex_commutator.py)):

```text
embeddability (S2 panel): {'0': {'complex': True, 'negative_real': False, 'min_real': 0.6931965374740569}, '1': {'complex': True, 'negative_real': False, 'min_real': 0.4974323353763193}}
eigenvalues of the yearly least-squares matrix, TD: [0.693+0.072j 0.693-0.072j 0.876+0.j    0.797+0.j   ]
eigenvalues of the true yearly matrix expm(A):     [0.779+0.j 0.779+0.j 0.779+0.j 0.779+0.j]
```

The true yearly matrix has one eigenvalue, $e^{-0.25}=0.779$, four times over (equal own rates). Sampling noise splits a repeated eigenvalue easily, here into a complex pair, so the flag fires although the true yearly matrix is exactly $e^{A}$ for a real drift $A$.

**Design choices and pitfalls.**
- The check reports; it does not test. Close or repeated true eigenvalues make complex estimates common (Notes, item 20).
- It describes the yearly least-squares matrices, not the engine's continuous-time drifts, which are fitted directly and need no logarithm.

---

### 5.43 `holm` and `holm_adjusted` — Holm's step-down procedure

[`multiplicity.py:12`](../../src/pdx/multiplicity.py#L12), [`multiplicity.py:26`](../../src/pdx/multiplicity.py#L26)

**What it does.** `holm(p, alpha)` returns Holm's rejections; `holm_adjusted(p)` returns Holm-adjusted p-values. Holm is the procedure of Step 5's confirmatory links (protocol §2.6, §7.6), available in `classify_links(method="holm")`.

**Inputs → outputs.** p-values (and $\alpha$) → a boolean array, or adjusted p-values, in the original order.

**The mathematics.** Sort $p_{(1)}\le\dots\le p_{(m)}$. Reject $H_{(1)},\dots,H_{(k^*-1)}$, where $k^*=\min\{k:p_{(k)}>\alpha/(m-k+1)\}$.

*Why the family-wise error is at most $\alpha$, under any dependence.* Let $m_0$ be the number of true nulls, and let the smallest true-null p-value have rank $j$. Any false rejection requires rejecting it, at step $j$. Since at most $m-m_0$ hypotheses rank before it, $j\le m-m_0+1$, so the step-$j$ threshold is $\alpha/(m-j+1)\le\alpha/m_0$. Hence

$$
P(\text{any false rejection})\le P\Big(\min_{i\ \text{true null}}p_i\le\frac{\alpha}{m_0}\Big)\le m_0\cdot\frac{\alpha}{m_0}=\alpha
$$

by Bonferroni. The adjusted p-values are $\tilde p_{(i)}=\max_{j\le i}\min\{1,(m-j+1)p_{(j)}\}$; then $H_{(i)}$ is rejected at level $\alpha$ exactly when $\tilde p_{(i)}\le\alpha$.

*From the maths to the code.* `holm`, lines 16–23: `rank` counts from 0, so the threshold is `alpha / (m - rank)`; the loop stops at the first failure. `holm_adjusted`, lines 29–35: the running maximum of `min(1, (m - rank) * p)`.

**Worked example** ([`ex_multiplicity.py`](../guide_examples/s05/ex_multiplicity.py); eight made-up p-values):

```text
sorted p        : [4.0e-04 8.9e-03 1.2e-02 3.1e-02 3.3e-02 4.9e-02 2.0e-01 7.4e-01]
Holm thresholds : [0.00625 0.00714 0.00833 0.01    0.0125  0.01667 0.025   0.05   ]
BH thresholds   : [0.00625 0.0125  0.01875 0.025   0.03125 0.0375  0.04375 0.05   ]
BY constant c(m) = sum 1/j = 2.7179; BY thresholds: [0.0023 0.0046 0.0069 0.0092 0.0115 0.0138 0.0161 0.0184]
rejected, in the original order:
  p    : [1.2e-02 4.0e-04 4.9e-02 3.1e-02 2.0e-01 8.9e-03 7.4e-01 3.3e-02]
  holm : [0 1 0 0 0 0 0 0]
  bh   : [1 1 0 0 0 1 0 0]
  by   : [0 1 0 0 0 0 0 0]
  Holm-adjusted p: [0.072  0.0032 0.155  0.155  0.4    0.0623 0.74   0.155 ]
```

Holm rejects 0.0004 (≤ 0.00625) and stops at 0.0089 (> 0.00714). By hand, the adjusted value of 0.033 is $\max(\dots,5\times0.031,\ 4\times0.033)=\max(0.155,0.132)=0.155$. `tests/test_multiplicity.py` checks all three procedures against `statsmodels` on 50 random families.

**Design choices and pitfalls.** NaN p-values sort last and count in $m$, which is conservative.

### 5.44 `bh` and `by` — Benjamini–Hochberg and Benjamini–Yekutieli

[`multiplicity.py:38`](../../src/pdx/multiplicity.py#L38), [`multiplicity.py:54`](../../src/pdx/multiplicity.py#L54)

**What it does.** Rejections under Benjamini–Hochberg (BH) and Benjamini–Yekutieli (BY). BY is the protocol's control for the exploratory families: per-link tests over all map links (§7.3.3), per-domain tests (§7.3.4) and the anchored tests.

**Inputs → outputs.** p-values and $\alpha$ → a boolean array.

**The mathematics.**
- BH finds $\hat k=\max\{k:p_{(k)}\le k\alpha/m\}$ and rejects the $\hat k$ smallest p-values. It is a step-up rule: a p-value above its own threshold is still rejected if a larger one meets its threshold. The false-discovery rate is at most $(m_0/m)\,\alpha$ under independence, or under positive regression dependence (PRDS).
- BY runs BH at $\alpha/c(m)$ with $c(m)=\sum_{j=1}^m1/j$. Benjamini & Yekutieli (2001, Theorem 1.3) showed that BH at level $q$ has false-discovery rate at most $q\,(m_0/m)\,c(m)$ under *any* dependence, so dividing by $c(m)$ restores control at $\alpha$. The per-link deviations share one speed estimate and correlated free-fit entries, and their two-sided p-values are not PRDS in general, which is why the protocol uses BY.

*From the maths to the code.* `bh`, lines 42–51: an empty family gives no rejections; `thresh` $=\alpha k/m$; `kmax` is the largest $k$ that meets its threshold; the `kmax + 1` smallest are rejected. `by`, lines 56–58: $c(m)$, then `bh(p, alpha / c)`.

**Worked example** (output in §5.43). BH: $p_{(3)}=0.012\le0.01875$ while every later $p_{(k)}$ misses its threshold, so $\hat k=3$, and 0.0004, 0.0089 and 0.012 are rejected. BY: $c(8)=2.718$, so the thresholds shrink to 0.0023, 0.0046, … and only 0.0004 is rejected.

**Design choices and pitfalls.**
- $m$ counts NaN p-values, which is conservative.
- At $M=2000$ Monte Carlo draws, BY cannot reject a lone link in families of 26 or more (§5.26; Notes, item 13).

### 5.45 `simultaneous_intervals` — max-|t| simultaneous intervals

[`multiplicity.py:61`](../../src/pdx/multiplicity.py#L61)

**What it does.** Simultaneous intervals for a family of estimates from Monte Carlo draws, by the maximum standardised deviation. Used for the "similar" class (§5.31).

**Inputs → outputs.** Estimates $(k)$, draws $(M\times k)$ and a level (0.90) → `(lower, upper)`.

**The mathematics.** With $s_k$ the SD of the draws of estimate $k$:

$$
T^{(m)}=\max_k\frac{|X^{(m)}_k-\bar X_k|}{s_k},\qquad c=q_{level}\big(T^{(m)}\big),\qquad \text{interval}_k=\hat\theta_k\pm c\,s_k .
$$

If the centred draws mimic the joint sampling distribution of $\hat\theta-\theta$, then $P(\forall k:|\hat\theta_k-\theta_k|\le c\,s_k)\approx$ level: every interval covers at once. Correlation between the estimates is used, which makes $c$ smaller than a Bonferroni constant.

*From the maths to the code.* Lines 63–68: `sd` with `ddof=1`; a zero SD becomes $\infty$, so that column gets an infinite interval and can never be called similar; `tmax` per draw; its quantile; $\hat\theta\pm c\,s$.

**Worked example** ([`ex_multiplicity.py`](../guide_examples/s05/ex_multiplicity.py); six estimates with draw SD 0.04 and correlation 0.5):

```text
simultaneous 90% intervals over 6 correlated estimates (max-|t|)
  critical value c = 2.294 (marginal 90% would use 1.645)
  simultaneous lo: [-0.072 -0.101 -0.061 -0.092 -0.041 -0.131]
  simultaneous hi: [0.112 0.081 0.121 0.092 0.141 0.051]
  joint coverage of the centred draws: 0.9
```

**Design choices and pitfalls.**
- The intervals are symmetric ($\hat\theta\pm c\,s$), while every other interval in `localise` is basic (reflected quantiles). For skewed draws the two disagree.
- The draws are centred at their own mean, not at the estimate.

### 5.46 `iut_similar` — every interval inside ±ε

[`multiplicity.py:71`](../../src/pdx/multiplicity.py#L71)

**What it does.** Returns True if every interval lies strictly inside $(-\varepsilon,\varepsilon)$: the intersection–union verdict, given intervals.

**Inputs → outputs.** Lower and upper bounds and $\varepsilon$ → a bool.

**The mathematics.** §5.33. The intersection–union test needs each interval to be an individual 90% interval. Simultaneous intervals would make it needlessly strict.

*From the maths to the code.* Line 74.

**Worked example** ([`ex_multiplicity.py`](../guide_examples/s05/ex_multiplicity.py)):

```text
  marginal 90% lo: [-0.046 -0.076 -0.035 -0.066 -0.015 -0.106]
  marginal 90% hi: [0.086 0.056 0.095 0.066 0.115 0.026]
  eps 0.10: iut_similar(marginal) = False, iut_similar(simultaneous) = False
  eps 0.12: iut_similar(marginal) = True, iut_similar(simultaneous) = False
  eps 0.15: iut_similar(marginal) = True, iut_similar(simultaneous) = True
```

At $\varepsilon=0.12$ the marginal intervals, which Berger's test needs, pass; simultaneous intervals would not.

**Design choices and pitfalls.** `localise.speed_and_load_only` does not call this function; it applies the same rule with `_within`. Only the unit test uses `iut_similar`.

---

### 5.N Notes for the author

Ordered by likely impact. Each was reproduced on synthetic data where an example is named; the others come from reading the code.

1. **Sp-1 and the free-fit speeds estimate different quantities** ([`speed.py:38`](../../src/pdx/speed.py#L38), [`localise.py:227`](../../src/pdx/localise.py#L227)).
   - *What:* Sp-1 $=\exp(\hat\gamma_1+\hat\theta^\top\Delta\bar y)$ includes the phenotypes' different mean landmark states. $\hat s_{LS}$ and $\hat s_{L1}$ come from the free fits, which absorb only $e^{\gamma_1}$ into $A_1$ and keep $\theta$, so under the null they estimate $e^{\gamma_1}$. In `ex_speed.py` the true values are Sp-1 = 2.242 against $e^{\gamma_1}=2$.
   - *Why it matters:* protocol §2.5 shows $\hat s_{L1}$ "beside" Sp-1 when H-P rejects, and a reader could take the gap $\hat\theta^\top\Delta\bar y$ for a mechanism effect. Protocol §2.3 defines $s_g=\exp(\gamma_g)$ "at given characteristics" and $\hat s_0$ "from the speed-only model", while the shared notation maps $\hat s_0$ to `speed.sp1`. §7.8's speed part, $\exp(h\hat s_0A_{TD})$, inherits the ambiguity. Report $e^{\hat\gamma_1}$ beside Sp-1, or state which one $\hat s_0$ is.
2. **`per_link_tests` raises when the point speed is identified but no draw is** ([`localise.py:236–243`](../../src/pdx/localise.py#L236)).
   - *What:* if every draw's speed is NaN, `np.quantile` of an empty array raises `IndexError`. Minimal reproduction, with no fit ([`ex_notes_reproductions.py`](../guide_examples/s05/ex_notes_reproductions.py)):
     ```python
     sup = np.eye(2, dtype=bool); sup[1, 0] = True; spec2 = ModelSpec(d=2, support=sup)
     a0, a1 = np.array([-0.04, -0.06, 0.03]), np.array([-0.05, -0.12, 0.05])
     d0 = np.tile(a0, (50, 1)); d0[:, 1] = -0.04          # the only voting entry falls below 0.05 in every draw
     per_link_tests(SimpleNamespace(spec=spec2), None, draws=FreeDraws(a0, a1, d0, np.tile(a1, (50, 1))))
     ```
     ```text
     Note 2: per_link_tests raised IndexError: index -1 is out of bounds for axis 0 with size 0
     ```
   - *Why it matters:* the case arises when the voting reference entries sit near the `near_zero` threshold. Partial losses are silent: draws without a speed are dropped (conditioning on identification), reported only in `extra["draws_used"]`.
3. **`bootstrap_speed` raises when every sample fails** ([`speed.py:134`](../../src/pdx/speed.py#L134)). `R[:, -1]` on an empty array raises `IndexError`. The reproduction in [`ex_notes_reproductions.py`](../guide_examples/s05/ex_notes_reproductions.py) passes an invalid fit option, so that every sample's fit raises:
   ```text
   Note 3: bootstrap_speed raised IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed
   ```
   In `run_hp_real.py` this would stop the run after the H-P bootstrap, before the report is written (the replicates survive in the checkpoint), instead of reporting $B_{used}=0$.
4. **`replicates_not_converged` counts failed replicates, and forgets resumed ones** ([`hp.py:157`](../../src/pdx/hp.py#L157)). A failed replicate returns `converged=False`, so it is counted both in `failures` and here. Replicates restored from a checkpoint are not in `rest`, so they are not counted here at all. In `ex_run_hp_bookkeeping.py` the same four replicates give 2 uninterrupted and 1 after a resume. Count non-convergence only among `ok` replicates, and save `converged` in the checkpoint.
5. **"Structurally different" uses marginal intervals** ([`localise.py:284–289`](../../src/pdx/localise.py#L284), [`300`](../../src/pdx/localise.py#L300)). Protocol §7.6 and the docstring of `multiplicity.py` say that "similar" and "structurally different" labels use simultaneous 90% intervals across the family. The code uses simultaneous intervals for "similar" only; the equivalence part of "structurally different" uses marginal 90% intervals, and the non-zero parts marginal 95% intervals. Align the protocol or the code.
6. **The reported interval is not the one behind "similar"** ([`localise.py:330`](../../src/pdx/localise.py#L330), [`343–344`](../../src/pdx/localise.py#L343)). `ci90` and `max_deviation_allowed` come from marginal basic intervals, while "similar" is judged on simultaneous ones. In `ex_link_classes.py` six links have `ci90` inside $\pm0.10$ (largest bound 0.086) but are "unresolved", because their simultaneous intervals reach $\pm0.12$. Report the simultaneous interval next to the class.
7. **`speed_and_load_only` tests the own rates too** ([`localise.py:353–354`](../../src/pdx/localise.py#L353)). Every support entry must pass, including the diagonal, while the per-link family defaults to the cross-links and protocol §2.3 says "every link's 90% interval". This is stricter than the text, and the own-rate intervals have the skew problem of item 8. State which is intended.
8. **Basic intervals are reflected against the skew of the own-rate draws** ([`localise.py:153–156`](../../src/pdx/localise.py#L153) with [`107–111`](../../src/pdx/localise.py#L107)). Own-rate draws are $-e^{N(\hat\mu,v)}$. In `ex_localise_fit.py` TD's tremor own rate has basic 95% interval $[-0.629,-0.039]$ against $[-0.812,-0.223]$ for the percentile interval, which equals the raw-scale Wald interval mapped back. This affects the non-zero and equivalence tests whenever own rates are tested (item 7, or a family that includes them) and the intervals of own-rate deviations. Percentile intervals, or intervals on the raw scale, respect the transformation.
9. **The recalibration of per-link p-values is not wired** ([`localise.py:387`](../../src/pdx/localise.py#L387), [`hp.py:98`](../../src/pdx/hp.py#L98)). Protocol §7.3.3 requires checking each link's null p-values on the H-P replicates and recalibrating if they are not uniform. `recalibrate` has no caller; `boot_replicate`, the parallel path `run_hp_real.py` uses, drops the `on_replicate` output; and no script passes `on_replicate`.
10. **Protocol parts not implemented in these modules.**
    - §7.7: the load and noise differences beyond speed, with BY across domains.
    - §7.8: model-implied propagation, the speed part and remainder, speed-matched horizons.
    - §7.6: Step 5's confirmatory family and its controls (only the `method="holm"` hook exists).
    - §7.3.2: participant-bootstrap intervals for the selected deviations, sample splitting, and eBIC.

    Not bugs, but still to come.
11. **The 90/10 speed ratio is conditional on the characteristics** ([`speed.py:51`](../../src/pdx/speed.py#L51)). $\exp(2z_{0.9}\sigma_g)$ is the spread among participants with the same covariates and landmark state. Protocol §2.5's "ratio between the 90th- and 10th-percentile patients' speeds" could be read as the marginal spread within the phenotype, which is wider. Say which.
12. **Some intervals leave the parameter range** ([`speed.py:144–145`](../../src/pdx/speed.py#L144), [`54`](../../src/pdx/speed.py#L54)). Basic intervals for the share explained and the speed–load correlations are not clipped: `ex_speed.py` shows an upper bound of 1.016 for a share. The correlations are ±1 whenever $\hat\tau\to0$ (−1 for two TD domains in `ex_speed.py`), a boundary estimate whose bootstrap interval is unreliable.
13. **Monte Carlo resolution limits Benjamini–Yekutieli** ([`localise.py:149`](../../src/pdx/localise.py#L149), [`multiplicity.py:54–58`](../../src/pdx/multiplicity.py#L54), `[localisation] mc_draws = 2000`). The smallest p-value is $1/2001=0.0005$, and BY's threshold for a lone rejection, $\alpha/(m\,c(m))$, is 0.00052 at $m=25$ and 0.00050 at $m=26$. A single strongly deviating link in a family of 26 or more links could never be rejected. The provisional map has 14 links (threshold 0.0011), but Step 1 can add links to the support. Scale $M$ with the family size: at least $m\,c(m)/\alpha$ (2,400 at $m=30$), and several times that for precision near the threshold. The draws are cheap; no refit is needed.
14. **Joint discovery is anchored on the median entry** ([`localise.py:536`](../../src/pdx/localise.py#L536)). The entry whose ratio is the weighted median has $\tilde\Delta=0$ exactly, so its weight is the cap $10^4$, and it stays at zero along the whole path. In `ex_joint.py` function's own rate has weight $10^4$ and is inactive even at the smallest $\lambda$. This is the logic of median initialisation, but the result then rests on one entry; near-ties in the median would switch the anchor. Also, the BIC's df counts $\gamma$ even when every deviation is active ([`549`](../../src/pdx/localise.py#L549)); minor.
15. **`reference_distribution` records failures as NaN** ([`hp.py:191–192`](../../src/pdx/hp.py#L191)), whereas the bootstrap, the warp-speed pool and `qualification` count failures as $+\infty$. If the §7.3.6 reading drops NaN, failed datasets leave the distribution silently.
16. **`run_hp` with `fits` but without `support_fn`** ([`hp.py:155`](../../src/pdx/hp.py#L155)). The internal replicates receive the spec before selection. If `fits` came from `observed_fits` with a support function and `run_hp` is called with `support_fn=None` and no `draw`, the replicates test the unselected support while the observed statistic used the selected one. `run_hp_real.py` passes both, so this is a calling hazard only.
17. **Two phenotypes are hard-coded in places**: `sp1` (`log_speed[1]`, [`speed.py:39`](../../src/pdx/speed.py#L39)), `free_draws(group=1)`, `free_to_joint`, `joint_discovery` ("D1 only"), `commutator_test` (`[:2]`). `h_p_df` and `run_hp` handle any $G$. A secondary axis with three groups (the clinical subtype, protocol §7.3.8) would need the localisation and Sp-1 generalised.
18. **Warm-start asymmetry in the bootstrap** ([`hp.py:74`](../../src/pdx/hp.py#L74)). Each replicate's null fit starts at the true values of its own data-generating process; the observed null fit starts at the default start. `docs/code_map.md` lists this, and the engine gate's dispersed starts check it. If the null likelihood had local optima, the asymmetry would push the test towards anti-conservative.
19. **The commutator's fallback** ([`fit.py:47–48`](../../src/pdx/engine/fit.py#L47), used at [`localise.py:610`](../../src/pdx/localise.py#L610)). With fewer than $d+5$ usable yearly transitions in a phenotype, `_yearly_ols` returns $0.8I$, which commutes with everything; the statistic is then about the other phenotype's matrix only, and a bootstrap sample can switch between the two regimes. Unlikely at PPMI's size, possible in small subgroups.
20. **`embeddability_check` flags sampling noise** ([`localise.py:631`](../../src/pdx/localise.py#L631)). Close or repeated true eigenvalues split into complex pairs easily: in `ex_commutator.py` the true yearly matrix has one real eigenvalue four times over, and the estimate shows a complex pair. The docstring's "other drift matrices can give the same yearly transitions" holds for complex pairs; for negative real eigenvalues there may be no real drift at all.
21. **Untested branches.** The unit tests cover `sequential_p` (stopping rules, uniform null p-values), the classes on hand-made draws, the weighted median and its batch version, and the multiplicity procedures against `statsmodels`. Nothing tests:
    - `sp2`'s formulas, `sp3`'s share, or `bootstrap_speed`'s interval arithmetic (a slow test checks only that resumed and parallel runs agree);
    - `free_draws`, `_domain_p` by itself, `recalibrate`;
    - `anchored_tests`, `joint_discovery`, `stability_selection`, `commutator_test`, `embeddability_check`;
    - `warp_replicate` (`tests/test_qualification.py` tests only the pooling) and `reference_distribution`.

    The smoke test runs some of these end to end without checking values. The examples in `docs/guide_examples/s05/` could become tests: Sp-1 and the share explained recomputed by hand, the flat likelihood of `free_to_joint`, and the class rules on a hand-made view.
22. **Small things.**
    - `CLASSES` is never used; the class strings are repeated literally in `_classes_at`.
    - `_flat`'s `free_fit` argument is unused.
    - In `bootstrap_speed` the vector of estimates is called `theta` (not the model's $\theta$).
    - The `if par["theta"].size` guard in `sp1` and `sp2` never triggers.
    - `iut_similar` duplicates `_within` and is not used by `localise`.
    - `run_hp`'s serial path (one shared generator) and the parallel path (`[seed, 3, i]`) give different, equally valid, replicate sequences.
    - The exploratory per-link family is every cross-link of the *selected* support, including links that Step 1 added ([`localise.py:226`](../../src/pdx/localise.py#L226)); protocol §7.3.3 says "over all map links".
23. **Shared notation.** Every symbol of §0.4 matches `model.py`: $\log s_i=\gamma_g+\beta^\top z_i+\theta^\top\tilde y_{i1}+\sigma_gu_i$ (`log_speeds`, `participant_loglik`), drifts $A$, $A_g$ and $A+D_g$ (`drifts`), support entries as (to, from) (`sup_idx`), LR and df (`lr_stat`, `h_p_df`), $h=20$ and $n_{max}=999$ (`[hp]`), $\varepsilon=0.10$ (`[localisation]`). The one mismatch is item 1: $\hat s_0$ = `speed.sp1` is Sp-1, not $e^{\hat\gamma_1}$.

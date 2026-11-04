# WP3: fitting the simulators to PPMI (protocol §8.2): plan

> 4 November 2026. The step after WP1's map and before the qualification runs. Nothing here has been run on PPMI data.

## What §8.2 asks for

- **Two simulator families**, both fitted to PPMI's pooled primary population over the frozen expert map:
  - family A, the engine's class;
  - family B, outside it and coded separately.
- **H-P must hold its level in both** (H-M1). A **fidelity dashboard** compares real and simulated data; WP2 fixes its thresholds.
- **Today's simulators run on preset values** (`src/pdx/sim/family_a.py`, `family_b.py`). That was enough for the power pilot and the engine gate. It is not enough for H-M1.

## Keeping the confirmatory analyses blind

The fit uses the **pooled** population as **one group**. It never estimates anything by phenotype:
- **Speed.** A phenotype-specific fit would estimate PIGD's speed ratio (Sp-1) from the real data before registration.
- **Phenotype differences** enter the simulated data only by design, scenario by scenario: S0 none, S1 a speed ratio from the stress grid (1.25, 1.5, 2), and so on.
- **The phenotype rule** (Stebbins) is applied to the simulated baselines, so the artefacts of defining phenotypes from graph variables are reproduced (§8.2, "Baselines").

## Family A: what is fitted, and how

| Part | Source in PPMI (pooled) | How | Code |
|---|---|---|---|
| Links, load, noise covariance, speed spread, speed–load link | The analysis domains, year 1 on | The engine's speed-only model with **one group**, over the frozen map's support | Engine exists; a one-group panel and a fit-to-`Truth` translation are new |
| Effects of the pre-specified characteristics on speed | Era, age at onset, sex, duration | The same fit | `Truth` has `beta_age` only; to extend |
| Treatment terms (Emax) | Dose and ON state | The same fit | Exists in the engine and in `Truth` |
| Treatment initiation | First treated visit | Logistic model on the state and era | `start_hazard` is preset; a fitted model is new |
| Dropout and death | The last visit, Conclusion of Study | Logistic model on the simulated state and era | `dropout` is preset; a fitted model is new |
| Curvature | Transition residuals | Splines in the state and in disease duration | `Truth` has one `tanh` term; to extend |
| Missingness and intervals | WP0's visit grid | Resampled by era, year and treatment status | WP0's marginal shares are used today; resampling the patterns is new |
| Baselines with the phenotype items | Year-0 scores and the Stebbins items | Joint model (Gaussian copula on the normal scores) | New |
| Measurement error | `configs/measurement_error.json` (D28) | Fixed | Exists |

**Noise.** The engine's noise covariance is unrestricted, while `Truth` holds only a diagonal variance (`q`). The translation needs `Truth` to take a full covariance.

## Family B: separate code

Family B is a discrete-time latent-state model. Its parts:
- non-linear dynamics;
- a time-varying latent burden;
- measurement that differs by era through a common scale;
- site effects;
- its own treatment model.

It is fitted by a separate, simpler estimator, for example moment matching on the dashboard's statistics, so that it does not share the engine's assumptions. Loadings that differ by phenotype stay a named stress, outside the null scenarios.

## Fidelity dashboard

`src/pdx/fidelity.py` already computes most of the pooled summaries. It places each real value within the 2.5–97.5% range of the simulated datasets.

| Comparison | In `fidelity.py` | Still to add |
|---|---|---|
| Marginals | Quantiles per domain | By year |
| Dynamics | Transition residuals, lagged and same-year correlations | Autocorrelations |
| Within-person | Variance across visits | — |
| Phenotypes | PIGD share | Year-1 stability |
| Treatment | Share treated at the last visit | Timing of initiation, dose |
| Follow-up | Missing shares, visits per participant | Dropout, intervals |
| Groups | — | By era and by site |

- **Pooled real-data summaries.** They are never split by phenotype, so the dashboard reveals nothing H-P tests.
- **Aggregates only,** under WP0's small-cell rule.
- **WP2 fixes the thresholds:** which failures stop the qualification, which trigger a refit, and which only limit interpretation (§8.2).

## Order and approvals

1. **WP1 freezes the map and the domains.** The fit's support depends on them.
2. **Code first, on the mock download** (`tests/ppmi_mock.py`): the one-group fit, the translation, the new parts of family A, family B and the dashboard's missing statistics, each with tests.
3. **Approval to read PPMI** for the pooled fit (§14). It is a new use of the data, after WP0's.
4. **Fit, then check the dashboard.**
5. **Qualification runs** (WP3), on the compute settled by D14.

**Compute.** The one-group fit is about as costly as one speed-only fit at the real size: about half an hour to an hour on the M4. The dashboard needs only a few hundred simulated datasets, without any fitting. The cost of WP3 lies in the qualification runs, not in this step.

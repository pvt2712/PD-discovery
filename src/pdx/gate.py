"""The engine gate (protocol v0.5, §16), run on synthetic data before registration.

The complete engine passes if:
1. the likelihood agrees with the independent implementation, and gradients agree with finite
   differences;
2. parameters are recovered without material bias and intervals cover close to their nominal
   level (at least 20 datasets of the real size, under the null and with one changed link);
3. fits converge from dispersed starts, and no likelihood ratio is negative;
4. profile likelihoods show speed, load, noise and the treatment terms are identified, with
   acceptable Hessian condition numbers;
5. results agree across 12, 20 and 30 quadrature nodes;
6. a whole real-size bootstrap replicate, with support selection and every refit, is timed.
Every check returns plain numbers; run_gate collects them with pass flags for the JSON report.
"""
from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import torch

from .engine.fit import Fit, covariance_checked, fit_model, fit_pair, hessian, param_names, start_values
from .engine.likelihood import loglik, prepare, value_and_grad
from .engine.model import Layout, ModelSpec
from .engine.reference import reference_loglik
from .hp import _replicate
from .qualification import spec_for
from .sim import scenarios
from .step1 import Step1Config, make_support_fn


def check_reference(panel, spec: ModelSpec, rng, n_points: int = 3) -> dict:
    prep = prepare(panel, spec.meas_var, spec.n_groups)
    worst = 0.0
    for kind in ("null", "free", "joint"):
        lay = Layout(spec, kind)
        for _ in range(n_points):
            x = start_values(panel, spec, kind) + 0.05 * rng.normal(size=lay.size)
            a = float(loglik(torch.as_tensor(x), lay, prep))
            b = reference_loglik(x, lay, panel, prep.mu1.numpy(), prep.S1.numpy(), prep.y1c.numpy())
            worst = max(worst, abs(a - b) / max(1.0, abs(b)))
    return {"max_relative_difference": worst, "pass": worst < 1e-8}


def check_gradient(panel, spec: ModelSpec, rng, n_coords: int = 20, h: float = 1e-6) -> dict:
    prep = prepare(panel, spec.meas_var, spec.n_groups)
    worst = 0.0
    for kind in ("null", "free"):
        lay = Layout(spec, kind)
        x = start_values(panel, spec, kind) + 0.05 * rng.normal(size=lay.size)
        _, g = value_and_grad(x, lay, prep)
        for j in rng.choice(lay.size, size=min(n_coords, lay.size), replace=False):
            e = np.zeros(lay.size)
            e[j] = h
            fd = (value_and_grad(x + e, lay, prep)[0] - value_and_grad(x - e, lay, prep)[0]) / (2 * h)
            worst = max(worst, abs(fd - g[j]) / max(1.0, abs(g[j])))
    return {"max_relative_difference": worst, "pass": worst < 1e-4}


def check_quadrature(fit: Fit, panel, nodes=(12, 20, 30)) -> dict:
    out = {}
    for K in nodes:
        spec = replace(fit.spec, n_quad=K)
        lay = Layout(spec, fit.kind)
        prep = prepare(panel, spec.meas_var, spec.n_groups)
        out[str(K)] = float(loglik(torch.as_tensor(fit.x), lay, prep))
    spread = max(out.values()) - min(out.values())
    return {"loglik": out, "spread": spread, "pass": spread < 0.01}


def check_dispersed_starts(panel, spec: ModelSpec, rng, n_starts: int = 5, scale: float = 0.3,
                           fit_kw: dict | None = None) -> dict:
    """Both models from their default start and from dispersed starts around it."""
    prep = prepare(panel, spec.meas_var, spec.n_groups)
    out = {}
    for kind in ("null", "free"):
        x_def = start_values(panel, spec, kind)
        lls = [fit_model(panel, spec, kind, prep=prep, **(fit_kw or {})).loglik]
        for _ in range(n_starts):
            x0 = x_def + scale * rng.normal(size=len(x_def))
            lls.append(fit_model(panel, spec, kind, x0=x0, prep=prep, **(fit_kw or {})).loglik)
        out[kind] = {"logliks": lls, "spread": max(lls) - min(lls)}
    out["pass"] = all(out[k]["spread"] < 0.1 for k in ("null", "free"))
    return out


def profile(fit: Fit, panel, name: str, index: int, se: float, half_width_se: float = 4.0, n_grid: int = 9,
            fit_kw: dict | None = None) -> dict:
    """Profile log-likelihood over one raw parameter across +- half_width_se standard errors;
    identified if it falls by at least 1.92 (the 95% chi-square cut-off) on both sides."""
    lay = fit.layout
    j = lay.slices[name][0].start + index
    prep = prepare(panel, fit.spec.meas_var, fit.spec.n_groups)
    width = half_width_se * (se if np.isfinite(se) and se > 0 else 1.0)
    grid = fit.x[j] + np.linspace(-width, width, n_grid)
    vals = []
    for v in grid:
        bounds = [(None, None)] * lay.size
        bounds[j] = (v, v)
        x0 = fit.x.copy()
        x0[j] = v
        vals.append(fit_model(panel, fit.spec, fit.kind, x0=x0, prep=prep, bounds=bounds,
                              **(fit_kw or {})).loglik)
    vals = np.array(vals)
    top = max(vals.max(), fit.loglik)
    mid = n_grid // 2
    return {"parameter": f"{name}[{index}]", "grid": grid.tolist(), "loglik": vals.tolist(), "se": float(se),
            "identified": bool(top - vals[:mid].min() >= 1.92 and top - vals[mid + 1:].min() >= 1.92)}


def identification(fit: Fit, panel, H: np.ndarray | None = None, boundary: float = 0.01) -> dict:
    """Conditioning of the observed information. Spreads (load tau, speed sigma) estimated below
    `boundary` sit at the edge of their range, where the likelihood is flat in their log; they
    are listed and set aside when judging the conditioning of the rest."""
    H = hessian(fit, panel) if H is None else H
    names = param_names(fit.layout)
    V, info = covariance_checked(H, names=names)
    at_edge = [i for i, nm in enumerate(names)
               if nm.split("[")[0] in ("log_tau", "log_sigma") and fit.x[i] < np.log(boundary)]
    keep = np.setdiff1d(np.arange(len(names)), at_edge)
    _, rest = covariance_checked(H[np.ix_(keep, keep)])
    sd = np.sqrt(np.clip(np.diag(V), 1e-300, None))
    C = V / np.outer(sd, sd)
    np.fill_diagonal(C, 0.0)
    return {**info, "spreads_at_zero": [names[i] for i in at_edge],
            "condition_number_without_them": rest["condition_number"],
            "well_conditioned_without_them": rest["well_conditioned"],
            "max_abs_correlation": float(np.abs(C).max()), "se": sd}


def recovery(scenario: str, n: int, n_datasets: int, seed: int = 0, d: int = 4, fit_kw: dict | None = None,
             with_intervals: bool = True) -> dict:
    """Recovery of known parameters; LR >= 0 in every dataset.

    Under a shared mechanism (S1) the speed-only fit is compared with the truth: the speed ratio,
    the speed spreads and the reference drift, with the coverage of model-based 95% intervals.
    Under a changed link (S2) the free fit is compared with the truth: the reference drift with
    TD's, and the other phenotype's drift with the speed ratio times PIGD's.
    """
    rows = []
    for r in range(n_datasets):
        rng = np.random.default_rng(seed + r)
        panel, info = scenarios.generate(scenario, n, rng, d=d)
        spec = spec_for(panel, info)
        t = info["truth"]
        f0, f1 = fit_pair(panel, spec, **(fit_kw or {}))
        gain = f1.start_logliks[1] - f1.start_logliks[0] if len(f1.start_logliks) > 1 else 0.0
        row = {"lr": 2 * (f1.loglik - f0.loglik), "converged": f0.success and f1.success,
               "default_start_gain": float(gain)}
        if scenario in scenarios.NULL_SCENARIOS:
            lay = f0.layout
            row["error_log_speed"] = float(lay.get(f0.x, "gamma")[0] - np.log(t.speed_ratio))
            row["error_A"] = (f0.params["A"][0][spec.support] - t.A[spec.support]).tolist()
            if with_intervals:
                V = covariance_checked(hessian(f0, panel))[0]
                jg = lay.slices["gamma"][0].start
                row["covered_log_speed"] = bool(abs(row["error_log_speed"]) <= 1.96 * np.sqrt(max(V[jg, jg], 0.0)))
                js = lay.slices["log_sigma"][0]
                err_s = lay.get(f0.x, "log_sigma") - np.log(np.array(t.sigma))
                row["covered_log_sigma"] = (np.abs(err_s) <= 1.96 * np.sqrt(np.clip(np.diag(V)[js], 0.0, None))).tolist()
        else:
            row["error_A"] = (f1.params["A"][0][spec.support] - t.A[spec.support]).tolist()
            row["error_A_other"] = (f1.params["A"][1][spec.support]
                                    - t.speed_ratio * t.A_pigd[spec.support]).tolist()
        rows.append(row)
    out = {"scenario": scenario, "datasets": n_datasets, "n": n,
           "mean_abs_error_A": float(np.mean(np.abs([r["error_A"] for r in rows]))),
           "min_lr": float(min(r["lr"] for r in rows)), "non_converged": int(sum(not r["converged"] for r in rows)),
           "max_gain_from_free_default_start": float(max(r["default_start_gain"] for r in rows))}
    if "error_log_speed" in rows[0]:
        e = np.array([r["error_log_speed"] for r in rows])
        out.update({"bias_log_speed": float(e.mean()), "rmse_log_speed": float(np.sqrt(np.mean(e ** 2)))})
    if "error_A_other" in rows[0]:
        out["mean_abs_error_A_other"] = float(np.mean(np.abs([r["error_A_other"] for r in rows])))
    if with_intervals and "covered_log_speed" in rows[0]:
        out["coverage_log_speed"] = float(np.mean([r["covered_log_speed"] for r in rows]))
        out["coverage_log_sigma"] = float(np.mean([r["covered_log_sigma"] for r in rows]))
    out["pass_no_negative_lr"] = out["min_lr"] > -1e-6
    return out


def time_replicate(n: int, d: int, seed: int = 0, fit_kw: dict | None = None, step1: bool = True) -> dict:
    """Seconds for the observed fits and one whole bootstrap replicate (support selection and refits)."""
    rng = np.random.default_rng(seed)
    panel, info = scenarios.generate("S1", n, rng, d=d)
    spec = spec_for(panel, info)
    support_fn = make_support_fn(info["support"], Step1Config()) if step1 else None
    t0 = time.time()
    spec_s = spec if support_fn is None else spec.with_support(support_fn(panel))
    t1 = time.time()
    f0, f1 = fit_pair(panel, spec_s, **(fit_kw or {}))
    t2 = time.time()
    _replicate(f0, panel, spec, support_fn, rng, fit_kw or {})
    t3 = time.time()
    return {"n": panel.n, "d": d, "support_selection_s": t1 - t0, "observed_fits_s": t2 - t1,
            "replicate_s": t3 - t2, "null_evaluations": f0.n_eval, "free_evaluations": f1.n_eval}


def run_gate(n: int = 1400, d: int = 4, n_datasets: int = 20, seed: int = 2026, quick: bool = False,
             fit_kw: dict | None = None) -> dict:
    """All checks; quick=True shrinks sizes so the harness itself can be tested in minutes."""
    rng = np.random.default_rng(seed)
    if quick:
        n, n_datasets = min(n, 200), min(n_datasets, 2)
    small, info = scenarios.generate("S1", 40, rng, d=d)
    spec_small = spec_for(small, info)
    report = {"settings": {"n": n, "d": d, "n_datasets": n_datasets, "seed": seed, "quick": quick},
              "reference": check_reference(small, spec_small, rng),
              "gradient": check_gradient(small, spec_small, rng)}
    panel, info = scenarios.generate("S1", n, rng, d=d)
    spec = spec_for(panel, info)
    f0 = fit_model(panel, spec, "null", **(fit_kw or {}))
    report["quadrature"] = check_quadrature(f0, panel)
    report["dispersed_starts"] = check_dispersed_starts(panel, spec, rng, n_starts=2 if quick else 5,
                                                        fit_kw=fit_kw)
    ident = identification(f0, panel)
    se = ident.pop("se")
    report["identification"] = ident
    lay = f0.layout
    names = [("gamma", 0), ("log_sigma", 1), ("b", spec.d + 1), ("L", 0), ("log_tau", 0), ("lam", 0)]
    names += [("kappa", 0)] if spec.n_treat else []
    report["profiles"] = [profile(f0, panel, nm, ix, se[lay.slices[nm][0].start + ix], n_grid=5 if quick else 9,
                                  fit_kw=fit_kw) for nm, ix in names if lay.has(nm)]
    report["recovery_null"] = recovery("S1", n, n_datasets, seed + 100, d, fit_kw, with_intervals=not quick)
    report["recovery_changed_link"] = recovery("S2", n, n_datasets, seed + 200, d, fit_kw, with_intervals=False)
    report["timing"] = time_replicate(n, d, seed + 300, fit_kw)
    checks = [report["reference"]["pass"], report["gradient"]["pass"], report["quadrature"]["pass"],
              report["dispersed_starts"]["pass"], report["identification"]["well_conditioned_without_them"],
              all(p["identified"] for p in report["profiles"]),
              report["recovery_null"]["pass_no_negative_lr"], report["recovery_changed_link"]["pass_no_negative_lr"]]
    report["all_automatic_checks_pass"] = bool(all(checks))
    report["note"] = ("Bias and coverage are judged by the investigator and statistician against §16; "
                      "the timing revises the compute budget (§8.8).")
    return report

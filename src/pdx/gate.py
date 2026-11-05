"""The engine gate (protocol v0.5, §16), run on synthetic data before registration.

The complete engine passes if:
1. the likelihood agrees with the independent implementation, and gradients agree with finite
   differences;
2. parameters are recovered without material bias and intervals cover close to their nominal
   level (at least 20 datasets of the real size, under the null and with one changed link);
3. fits converge from dispersed starts, and no likelihood ratio is negative;
4. profile likelihoods show speed, load, noise and the treatment terms are identified, with
   acceptable Hessian condition numbers;
5. results agree across the chosen number of quadrature nodes and 10 and 20 more (20, 30, 40 since D32);
6. a whole real-size bootstrap replicate, with support selection and every refit, is timed.
Every check returns plain numbers; run_gate collects them with pass flags for the JSON report.
At the real size the gate runs for days, so it saves every finished section, dispersed-start fit,
recovery dataset and profile to a checkpoint; a restarted run skips them. Each section has its own
seed, so a resumed run gives the same results as an uninterrupted one.
"""
from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from . import parallel
from .engine.fit import Fit, covariance_checked, fit_model, fit_pair, hessian, null_to, param_names, start_values
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


def check_quadrature_lr(f0: Fit, panel, fit_kw: dict | None = None, nodes=(12, 20, 30)) -> dict:
    """The speed-only fit's log-likelihood at each node count (pass: spread below 0.01), and H-P's
    statistic at each, from a free fit started at the speed-only one: errors that are common to
    both models cancel in the statistic, which is the result that has to agree (§16)."""
    f1 = fit_model(panel, f0.spec, "free", x0=null_to(f0.x, f0.spec, "free"), **(fit_kw or {}))
    q0, q1 = check_quadrature(f0, panel, nodes), check_quadrature(f1, panel, nodes)
    lr = {K: 2 * (q1["loglik"][K] - q0["loglik"][K]) for K in q0["loglik"]}
    return {**q0, "free_loglik": q1["loglik"], "free_spread": q1["spread"], "lr": lr,
            "lr_spread": max(lr.values()) - min(lr.values())}


def start_setup(panel, spec: ModelSpec, fit_kw: dict | None) -> dict:
    """Shared state of the dispersed-start workers (parallel.ordered)."""
    return {"panel": panel, "spec": spec, "fit_kw": fit_kw or {},
            "prep": prepare(panel, spec.meas_var, spec.n_groups)}


def start_fit(shared: dict, job: tuple) -> float:
    """One fit of one model (`kind`) from the start x0 (None: the default start)."""
    kind, x0 = job
    return fit_model(shared["panel"], shared["spec"], kind, x0=x0, prep=shared["prep"], **shared["fit_kw"]).loglik


def check_dispersed_starts(panel, spec: ModelSpec, rng, n_starts: int = 5, scale: float = 0.3,
                           fit_kw: dict | None = None, done: dict | None = None, on_fit=None,
                           jobs: int = 1) -> dict:
    """Both models from their default start and from dispersed starts around it. `done` holds the
    log-likelihoods already found (from a checkpoint); on_fit(done) is called after each new one;
    `jobs` fits run at once."""
    done = {k: list(v) for k, v in (done or {}).items()}
    queue = []
    for kind in ("null", "free"):
        x_def = start_values(panel, spec, kind)
        starts = [None] + [x_def + scale * rng.normal(size=len(x_def)) for _ in range(n_starts)]
        queue += [(kind, x0) for x0 in starts[len(done.setdefault(kind, [])):]]
    for (kind, _), ll in zip(queue, parallel.ordered(start_fit, queue, jobs, start_setup, (panel, spec, fit_kw))):
        done[kind].append(ll)
        if on_fit:
            on_fit(done)
    out = {}
    for kind in ("null", "free"):
        lls = done[kind]
        out[kind] = {"logliks": lls, "spread": max(lls) - min(lls)}
    out["pass"] = all(out[k]["spread"] < 0.1 for k in ("null", "free"))
    return out


def profile_setup(fit: Fit, panel, n_grid: int, fit_kw: dict | None) -> dict:
    """Shared state of the profile workers (parallel.ordered)."""
    return {"fit": fit, "panel": panel, "n_grid": n_grid, "fit_kw": fit_kw}


def profile_job(shared: dict, job: tuple) -> dict:
    """The profile of one parameter: job is (name, index, standard error)."""
    name, index, se = job
    return plain(profile(shared["fit"], shared["panel"], name, index, se, n_grid=shared["n_grid"],
                         fit_kw=shared["fit_kw"]))


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


def recovery_setup(scenario: str, n: int, seed: int, d: int, fit_kw: dict | None, with_intervals: bool,
                   sim_kw: dict | None, engine: dict | None = None) -> dict:
    """Shared state of the recovery workers (parallel.ordered); engine holds the analysis model's
    settings (quadrature nodes, whether the load moves with speed)."""
    return {"scenario": scenario, "n": n, "seed": seed, "d": d, "fit_kw": fit_kw or {},
            "with_intervals": with_intervals, "sim_kw": sim_kw or {}, "engine": engine or {}}


def recovery_dataset(shared: dict, r: int) -> dict:
    """Dataset r of a recovery check, simulated with its own seed, fitted and compared with the truth."""
    scenario, with_intervals = shared["scenario"], shared["with_intervals"]
    rng = np.random.default_rng(shared["seed"] + r)
    panel, info = scenarios.generate(scenario, shared["n"], rng, d=shared["d"], **shared["sim_kw"])
    spec = spec_for(panel, info, **shared.get("engine", {}))
    t = info["truth"]
    f0, f1 = fit_pair(panel, spec, **shared["fit_kw"])
    gain = f1.start_logliks[1] - f1.start_logliks[0] if len(f1.start_logliks) > 1 else 0.0
    row = {"lr": 2 * (f1.loglik - f0.loglik), "converged": f0.success and f1.success,
           "failed": f0.failed or f1.failed, "default_start_gain": float(gain)}
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
    return plain(row)


def recovery(scenario: str, n: int, n_datasets: int, seed: int = 0, d: int = 4, fit_kw: dict | None = None,
             with_intervals: bool = True, sim_kw: dict | None = None, rows: list | None = None,
             on_row=None, jobs: int = 1, engine: dict | None = None) -> dict:
    """Recovery of known parameters; LR >= 0 in every dataset.

    Under a shared mechanism (S1) the speed-only fit is compared with the truth: the speed ratio,
    the speed spreads and the reference drift, with the coverage of model-based 95% intervals.
    Under a changed link (S2) the free fit is compared with the truth: the reference drift with
    TD's, and the other phenotype's drift with the speed ratio times PIGD's. `rows` are datasets
    already done (from a checkpoint); on_row(rows) is called after each new one; `jobs` datasets
    run at once.
    """
    rows = list(rows or [])
    for row in parallel.ordered(recovery_dataset, range(len(rows), n_datasets), jobs, recovery_setup,
                                (scenario, n, seed, d, fit_kw, with_intervals, sim_kw, engine)):
        rows.append(row)
        if on_row:
            on_row(rows)
    out = {"scenario": scenario, "datasets": n_datasets, "n": n,
           "mean_abs_error_A": float(np.mean(np.abs([r["error_A"] for r in rows]))),
           "min_lr": float(min(r["lr"] for r in rows)), "non_converged": int(sum(not r["converged"] for r in rows)),
           "failed": int(sum(bool(r.get("failed")) for r in rows)),
           "max_gain_from_free_default_start": float(max(r["default_start_gain"] for r in rows))}
    if "error_log_speed" in rows[0]:
        e = np.array([r["error_log_speed"] for r in rows])
        out.update({"bias_log_speed": float(e.mean()), "rmse_log_speed": float(np.sqrt(np.mean(e ** 2)))})
    if "error_A_other" in rows[0]:
        out["mean_abs_error_A_other"] = float(np.mean(np.abs([r["error_A_other"] for r in rows])))
    if with_intervals and "covered_log_speed" in rows[0]:
        out["coverage_log_speed"] = float(np.mean([r["covered_log_speed"] for r in rows]))
        out["coverage_log_sigma"] = float(np.mean([r["covered_log_sigma"] for r in rows]))
    out["pass_no_negative_lr"] = out["min_lr"] > -1e-6 and out["failed"] == 0
    return out


def time_replicate(n: int, d: int, seed: int = 0, fit_kw: dict | None = None, step1: bool = True,
                   sim_kw: dict | None = None, engine: dict | None = None) -> dict:
    """Seconds for the observed fits and one whole bootstrap replicate (support selection and refits)."""
    rng = np.random.default_rng(seed)
    panel, info = scenarios.generate("S1", n, rng, d=d, **(sim_kw or {}))
    spec = spec_for(panel, info, **(engine or {}))
    support_fn = make_support_fn(info["support"], Step1Config()) if step1 else None
    t0 = time.time()
    spec_s = spec if support_fn is None else spec.with_support(support_fn(panel))
    t1 = time.time()
    f0, f1 = fit_pair(panel, spec_s, **(fit_kw or {}))
    t2 = time.time()
    stat, ok, conv, _, info = _replicate(f0, panel, spec, support_fn, rng, fit_kw or {})
    t3 = time.time()
    return {"n": panel.n, "d": d, "support_selection_s": t1 - t0, "observed_fits_s": t2 - t1,
            "replicate_s": t3 - t2, "null_evaluations": f0.n_eval, "free_evaluations": f1.n_eval,
            "replicate_ok": bool(ok), "replicate_statistic": float(stat) if ok else None, "replicate": info}


def plain(obj):
    """JSON-ready values (numpy arrays and scalars as lists and numbers), so results can be checkpointed."""
    def conv(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.generic):
            return o.item()
        raise TypeError(type(o).__name__)
    return json.loads(json.dumps(obj, default=conv))


class Checkpoint:
    """The gate's finished work, saved after each piece; a restart with other settings is refused.
    log(message), if given, reports each section as it finishes or is skipped."""

    def __init__(self, path: Path | None, settings: dict, log=None):
        self.path = Path(path) if path else None
        self.log = log or (lambda message: None)
        self.data = json.loads(self.path.read_text(encoding="utf-8")) if self.path and self.path.exists() else {}
        if self.data.get("settings", plain(settings)) != plain(settings):
            raise RuntimeError(f"{self.path} holds a gate run with other settings; move it away to start afresh")
        self.data["settings"] = plain(settings)

    def section(self, key: str, fn):
        if key in self.data:
            self.log(f"{key}: done earlier, skipped")
        else:
            t0 = time.time()
            self.data[key] = plain(fn())
            self.save()
            self.log(f"{key}: done ({(time.time() - t0) / 60:.0f} min)")
        return self.data[key]

    def save(self) -> None:
        if self.path:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.data))
            for attempt in range(20):            # Windows refuses to replace a file another program has open
                try:
                    tmp.replace(self.path)
                    return
                except PermissionError:
                    if attempt == 19:
                        raise
                    time.sleep(0.5)


def run_gate(n: int = 1400, d: int = 4, n_datasets: int = 20, seed: int = 2026, quick: bool = False,
             fit_kw: dict | None = None, sim_kw: dict | None = None, checkpoint: Path | None = None,
             log=None, jobs: int = 1, engine: dict | None = None) -> dict:
    """All checks; quick=True shrinks sizes so the harness itself can be tested in minutes.

    sim_kw shapes every real-size dataset (for example WP0's group sizes, visits and missingness;
    target_counts then fix the size, and n is the pool they are drawn from). With a checkpoint path,
    finished work is saved as it goes and skipped when the same run is restarted. log(message), if
    given, reports progress. `jobs` dispersed starts, profiles or recovery datasets run at once,
    each in a single-threaded worker; the results do not depend on it. engine holds the analysis
    model's settings (n_quad, load_on_speed; protocol_v0.5.toml's [engine]); quadrature is checked
    at n_quad and 10 and 20 nodes more.
    """
    engine = dict(engine or {})
    log = log or (lambda message: None)
    sim_kw = dict(sim_kw or {})
    if quick:
        n_datasets = min(n_datasets, 2)
        if sim_kw.get("target_counts"):
            tc = sim_kw["target_counts"]
            sim_kw["target_counts"] = tuple(max(2, round(200 * c / sum(tc))) for c in tc)
            n = 5 * sum(sim_kw["target_counts"])
        else:
            n = min(n, 200)
    ck = Checkpoint(checkpoint, {"n": n, "d": d, "n_datasets": n_datasets, "seed": seed, "quick": quick,
                                 "sim_kw": sim_kw, "fit_kw": fit_kw or {}, "engine": engine}, log)
    log(f"gate: {d} domains, {n_datasets} recovery datasets, {torch.get_num_threads()} threads, {jobs} parallel jobs")
    rng = lambda k: np.random.default_rng([seed, k])                     # one stream per section
    small, info = scenarios.generate("S1", 40, rng(0), d=d)
    spec_small = spec_for(small, info, **engine)
    report = {"settings": ck.data["settings"],
              "reference": ck.section("reference", lambda: check_reference(small, spec_small, rng(1))),
              "gradient": ck.section("gradient", lambda: check_gradient(small, spec_small, rng(2)))}
    panel, info = scenarios.generate("S1", n, rng(3), d=d, **sim_kw)
    spec = spec_for(panel, info, **engine)
    report["participants"] = int(panel.n)
    f0 = None
    if not all(k in ck.data for k in ("quadrature", "identification", "profiles")):
        x0 = np.array(ck.data["null_fit_x"]) if "null_fit_x" in ck.data else None   # a saved fit converges at once
        t0 = time.time()
        f0 = fit_model(panel, spec, "null", x0=x0, **(fit_kw or {}))
        ck.data["null_fit_x"] = f0.x.tolist()
        ck.save()
        log(f"speed-only fit of the gate dataset ({panel.n} participants): done ({(time.time() - t0) / 60:.0f} min)")
    k = spec.n_quad
    report["quadrature"] = ck.section("quadrature", lambda: check_quadrature_lr(f0, panel, fit_kw, (k, k + 10, k + 20)))
    n_starts = 2 if quick else 5

    def keep_starts(done):
        ck.data["dispersed_starts_done"] = done
        ck.save()
        log(f"dispersed_starts: fit {sum(map(len, done.values()))} of {2 * (n_starts + 1)} done")
    report["dispersed_starts"] = ck.section("dispersed_starts", lambda: check_dispersed_starts(
        panel, spec, rng(4), n_starts=n_starts, fit_kw=fit_kw, done=ck.data.get("dispersed_starts_done"),
        on_fit=keep_starts, jobs=jobs))
    if "identification" not in ck.data:
        ident = identification(f0, panel)
        ck.data["identification_se"] = plain(ident.pop("se"))
        ck.section("identification", lambda: ident)
    report["identification"] = ck.data["identification"]
    se = np.array(ck.data["identification_se"])
    lay = Layout(spec, "null")
    names = [("gamma", 0), ("log_sigma", 1), ("b", spec.d + 1), ("L", 0), ("log_tau", 0), ("lam", 0)]
    names += [("kappa", 0)] if spec.n_treat else []
    done = ck.data.setdefault("profiles_done", [])
    todo = [(nm, ix, float(se[lay.slices[nm][0].start + ix])) for nm, ix in names
            if lay.has(nm) and f"{nm}[{ix}]" not in [p["parameter"] for p in done]]
    t0 = time.time()
    for prof in parallel.ordered(profile_job, todo, jobs, profile_setup, (f0, panel, 5 if quick else 9, fit_kw)):
        done.append(prof)
        ck.save()
        log(f"profile of {prof['parameter']}: done ({(time.time() - t0) / 60:.0f} min since the profiles began)")
    report["profiles"] = ck.section("profiles", lambda: done)
    for key, scenario, offset, intervals in (("recovery_null", "S1", 100, not quick),
                                             ("recovery_changed_link", "S2", 200, False)):
        def keep_rows(rows, key=key):
            ck.data[key + "_rows"] = rows
            ck.save()
            log(f"{key}: dataset {len(rows)} of {n_datasets} done")
        report[key] = ck.section(key, lambda: recovery(scenario, n, n_datasets, seed + offset, d, fit_kw, intervals,
                                                       sim_kw, ck.data.get(key + "_rows"), keep_rows, jobs, engine))
    report["timing"] = ck.section("timing", lambda: time_replicate(n, d, seed + 300, fit_kw, sim_kw=sim_kw,
                                                                  engine=engine))
    checks = [report["reference"]["pass"], report["gradient"]["pass"], report["quadrature"]["pass"],
              report["dispersed_starts"]["pass"], report["identification"]["well_conditioned_without_them"],
              all(p["identified"] for p in report["profiles"]),
              report["recovery_null"]["pass_no_negative_lr"], report["recovery_changed_link"]["pass_no_negative_lr"]]
    report["all_automatic_checks_pass"] = bool(all(checks))
    report["note"] = ("Bias and coverage are judged by the investigator and statistician against §16; "
                      "the timing revises the compute budget (§8.8).")
    return report

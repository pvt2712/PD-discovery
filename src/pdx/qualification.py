"""WP3 method qualification (protocol v0.5, §8): H-P calibration by the warp-speed method
(Giacomini 2013), power, and the bias and coverage of the speed estimates, on simulator families
A and B. Synthetic data only.

One dataset per job: each result is saved as JSON as it finishes and skipped on rerun, with at
most 4 parallel jobs (§16). The H-M1 gate passes when every shared-mechanism scenario of both
families has a false-rejection probability of at most 7.5% (Bradley 1978).

The simulators here use preset parameters; fitting them to PPMI's pooled population (§8.2) is a
WP3 step that needs approved data access.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
from scipy import stats

from .engine.model import ModelSpec
from .engine.fit import fit_pair
from .hp import h_p_df, lr_stat, warp_replicate
from .sim import scenarios
from .speed import sp1, sp3
from .step1 import Step1Config, make_support_fn

NULL_A = scenarios.NULL_SCENARIOS
NULL_B = ("S0", "S1")


def spec_for(panel, info, treatment: str = "emax", n_quad: int = 12, era: bool = True) -> ModelSpec:
    """The analysis model for a simulated dataset: the support the analysis is given (family A:
    the true union, or a stressed one; family B: its map) and the fixed measurement variances."""
    truth = info.get("truth")
    d = panel.d
    tm = truth.treat_mask if truth is not None else np.isin(np.arange(d), [0, 1])
    mv = truth.meas_var if truth is not None else np.full(d, 0.02)
    return ModelSpec(d=d, support=info["support"], p=panel.p, treat_mask=tm, treatment=treatment,
                     meas_var=mv, n_quad=n_quad, noise_era_col=0 if era and panel.p else None)


def dataset_task(task: dict) -> dict:
    """One simulated dataset: H-P's statistic and one bootstrap replicate (warp-speed), plus the
    speed estimates from the same fits."""
    rng = np.random.default_rng(task["seed"])
    panel, info = scenarios.generate(task["scenario"], task["n"], rng, family=task.get("family", "A"),
                                     d=task.get("d", 4), **dict(task.get("sim_kw", {})))
    spec = spec_for(panel, info, task.get("treatment", "emax"))
    support_fn = None
    if task.get("step1", True):
        support_fn = make_support_fn(info["support"], Step1Config(**task.get("step1_cfg", {})))
    t0 = time.time()
    if task.get("replicate", True):
        r = warp_replicate(panel, spec, support_fn, rng, task.get("fit_kw", {}))
    else:                                  # power only: the observed statistic, no bootstrap replicate
        spec_s = spec if support_fn is None else spec.with_support(support_fn(panel))
        g0, g1 = fit_pair(panel, spec_s, **task.get("fit_kw", {}))
        r = {"lr": lr_stat(g0, g1), "lr_star": float("nan"), "replicate_ok": False, "replicate_converged": False,
             "converged": bool(g0.success and g1.success), "df": h_p_df(spec_s), "null_fit": g0, "free_fit": g1}
    f0, f1 = r["null_fit"], r["free_fit"]
    return {"task": task, "config": config_key(task), "scenario": task["scenario"],
            "family": task.get("family", "A"), "seed": task["seed"],
            "n": panel.n, "lr": r["lr"], "lr_star": r["lr_star"] if np.isfinite(r["lr_star"]) else None,
            "df": r["df"], "converged": r["converged"], "replicate_ok": r["replicate_ok"],
            "replicate_converged": r["replicate_converged"],
            "sp1": sp1(f0, panel), "sp3_share": sp3(f1)["share"], "true_speed_ratio": info["speed_ratio"],
            "seconds": time.time() - t0, "counts": panel.aggregate_summary()}


def config_key(task: dict) -> str:
    """A hash of everything in a task except its seed and scenario: datasets are pooled only
    within one configuration (size, domains, Step 1, treatment form, fitting options, stresses),
    and the same key links a power scenario to its null reference."""
    cfg = {k: v for k, v in task.items() if k not in ("seed", "scenario")}
    return hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:10]


def _path(out_dir: Path, task: dict) -> Path:
    return out_dir / f"{task.get('family', 'A')}_{task['scenario']}_{config_key(task)}_seed{task['seed']}.json"


def _run_one(args):
    task, out_dir = args
    path = _path(out_dir, task)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    try:
        res = dataset_task(task)
    except Exception as exc:                      # a failed dataset is logged, not dropped
        res = {"task": task, "config": config_key(task), **{k: task[k] for k in ("scenario", "seed", "n")},
               "family": task.get("family", "A"), "error": repr(exc)}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(res))
    tmp.replace(path)
    return res


def run_tasks(tasks: list, out_dir: str | Path, jobs: int = 4) -> list:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    args = [(t, out_dir) for t in tasks]
    if jobs <= 1:
        return [_run_one(a) for a in args]
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(v, "1")
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(min(jobs, 4)) as pool:
        return pool.map(_run_one, args, chunksize=1)


def make_tasks(scenario: str, n_datasets: int, n: int, family: str = "A", seed0: int = 0, **kw) -> list:
    base = {"A": 1_000_000, "B": 2_000_000}[family] + 10_000 * (list(scenarios.NULL_SCENARIOS +
                                                                     scenarios.POWER_SCENARIOS).index(scenario))
    return [{"scenario": scenario, "family": family, "n": n, "seed": base + seed0 + r, **kw}
            for r in range(n_datasets)]


# ---------------------------------------------------------------- summaries

def _mcse(p: float, R: int) -> float:
    return float(np.sqrt(p * (1 - p) / R)) if R else float("nan")


def _rank_quantile(values: list, q: float) -> float:
    """The ceil(q R)-th smallest value; failed replicates enter as +inf (at least as large as
    anything), as in the sequential procedure."""
    v = np.sort(np.array([np.inf if x is None or not np.isfinite(x) else x for x in values], float))
    if len(v) == 0:
        return float("inf")
    return float(v[min(len(v) - 1, int(np.ceil(q * len(v))) - 1)])


def _groups(rows: list) -> dict:
    out = {}
    for r in rows:
        out.setdefault((r["family"], r["scenario"], r.get("config", "")), []).append(r)
    return out


def calibration_summary(rows: list, alpha: float = 0.05, bradley: float = 0.075) -> dict:
    """Warp-speed rejection rate per scenario and configuration: each dataset's statistic
    against the (1 - alpha) quantile of the pooled single bootstrap statistics. A failed
    replicate counts as +inf in the pool; a failed dataset counts as a rejection."""
    out = {}
    for (fam, sc, cfg), rs in sorted(_groups(rows).items()):
        ok = [r for r in rs if "error" not in r]
        crit = _rank_quantile([r["lr_star"] for r in ok], 1 - alpha)
        rej = [("error" in r) or (r["lr"] > crit) for r in rs]
        chi = [("error" in r) or (r["lr"] > stats.chi2.ppf(1 - alpha, r["df"])) for r in rs]
        p = float(np.mean(rej)) if rs else float("nan")
        out[f"{fam}:{sc}:{cfg}"] = {
            "family": fam, "scenario": sc, "config": cfg, "task_example": rs[0].get("task"),
            "datasets": len(rs), "failed": len(rs) - len(ok), "critical_value": crit,
            "rejection_rate": p, "mcse": _mcse(p, len(rs)), "passes_bradley": bool(p <= bradley),
            "rejection_rate_chi2": float(np.mean(chi)) if rs else float("nan"),
            "non_converged": int(sum(not r.get("converged", False) for r in ok)),
            "replicates_failed": int(sum(not r.get("replicate_ok", False) for r in ok)),
            "replicates_not_converged": int(sum(not r.get("replicate_converged", False) for r in ok)),
            "median_seconds": float(np.median([r["seconds"] for r in ok])) if ok else float("nan")}
    return out


def h_m1(calibration: dict, family_a=NULL_A, family_b=NULL_B, config: dict | None = None) -> dict:
    """The gate of §8.7, with the partial pass when only S4 (curvature) fails. `config` maps each
    family to the configuration key of the protocol run (needed when several were run)."""
    by_cell = {}
    for v in calibration.values():
        if config and config.get(v["family"]) not in (None, v["config"]):
            continue
        by_cell.setdefault(f"{v['family']}:{v['scenario']}", []).append(v)
    need = [f"A:{s}" for s in family_a] + [f"B:{s}" for s in family_b]
    missing = [k for k in need if k not in by_cell]
    ambiguous = [k for k in need if len(by_cell.get(k, [])) > 1]
    failed = [k for k in need if k in by_cell and not all(v["passes_bradley"] for v in by_cell[k])]
    if missing or ambiguous:
        verdict = "incomplete" if missing else "ambiguous: several configurations; pass `config`"
    elif not failed:
        verdict = "pass"
    elif failed == ["A:S4"]:
        verdict = "partial pass (conditional on linear dynamics)"
    else:
        verdict = "fail"
    return {"verdict": verdict, "failed": failed, "missing": missing, "ambiguous": ambiguous}


def power_summary(rows: list, critical: dict, alpha: float = 0.05) -> dict:
    """Power in S2, S2b and S3 against the critical value of the matching null scenario (S1 of
    the same family and configuration apart from the scenario)."""
    crit = {(v["family"], v["config"]): v["critical_value"] for v in critical.values() if v["scenario"] == "S1"}
    out = {}
    for (fam, sc, cfg), rs in sorted(_groups(rows).items()):
        c = crit.get((fam, cfg), float("inf"))
        ok = [r for r in rs if "error" not in r]
        p = float(np.mean([r["lr"] > c for r in ok])) if ok else float("nan")
        out[f"{fam}:{sc}:{cfg}"] = {"datasets": len(ok), "failed": len(rs) - len(ok), "critical_value": c,
                                    "power": p, "mcse": _mcse(p, len(ok))}
    return out


def speed_summary(rows: list) -> dict:
    """Bias and root-mean-square error of log Sp-1 against the true speed ratio, per scenario
    and configuration."""
    out = {}
    for (fam, sc, cfg), rs in sorted(_groups(rows).items()):
        rs = [r for r in rs if "error" not in r]
        if not rs:
            continue
        e = np.log([r["sp1"] for r in rs]) - np.log([r["true_speed_ratio"] for r in rs])
        out[f"{fam}:{sc}:{cfg}"] = {"datasets": len(rs), "bias_log_sp1": float(e.mean()),
                                    "rmse_log_sp1": float(np.sqrt(np.mean(e ** 2))),
                                    "median_sp3_share": float(np.nanmedian([r["sp3_share"] for r in rs]))}
    return out

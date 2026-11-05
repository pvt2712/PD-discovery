"""Speed estimands, key secondary (protocol v0.5, §2.5).

Sp-1  the speed ratio: the ratio of geometric-mean speeds in the speed-only model, with the
      baseline covariates standardised to their pooled distribution and each phenotype averaged
      over its own landmark states.
Sp-2  the spread of speed within each phenotype (sigma_g and the 90th/10th percentile ratio), its
      predictors, the share of between-participant variation in log-speed they explain, and the
      speed-load correlation.
Sp-3  the share of the difference between the phenotypes' drifts that one speed explains,
      1 - ||A1 - s_LS A0||^2 / ||A1 - A0||^2 from the free fits (least-squares speed s_LS).
None of the three needs the minimal-L1 assumption. Intervals come from a participant bootstrap
(resampling participants within phenotype): basic intervals, on the log scale for ratios and
spreads. Sp-3 is bias-corrected with a one-sided lower bound, and is reported only when the
free fits clearly differ: ||A1 - A0|| above the 95th percentile of its centred bootstrap
distribution ||(A1 - A0)* - (A1 - A0)||. A sample whose fits fail is a failure; a sample that
re-selects another support is warm-started at the null fit written for that support.
"""
from __future__ import annotations

import numpy as np

from . import parallel
from .data import Panel
from .engine.fit import Fit, fit_pair, remap_support
from .engine.likelihood import prepare
from .engine.model import ModelSpec

Z90 = 1.2815515655446004


def _y1c(panel: Panel, spec: ModelSpec) -> np.ndarray:
    return prepare(panel, spec.meas_var, spec.n_groups).y1c.numpy()


def sp1(null_fit: Fit, panel: Panel) -> float:
    par = null_fit.params
    y1c = _y1c(panel, null_fit.spec)
    g = panel.group
    shift = par["theta"] @ (y1c[g == 1].mean(0) - y1c[g == 0].mean(0)) if par["theta"].size else 0.0
    return float(np.exp(par["log_speed"][1] + shift))


def sp2(null_fit: Fit, panel: Panel) -> dict:
    par = null_fit.params
    y1c = _y1c(panel, null_fit.spec)
    g = panel.group
    eta = par["log_speed"][g] + panel.z @ par["beta"] + (y1c @ par["theta"] if par["theta"].size else 0.0)
    sig = par["sigma"]
    pi = np.bincount(g, minlength=len(sig)) / len(g)
    within = float(np.sum(pi * sig ** 2))
    between = float(np.var(eta))
    return {"sigma": sig.tolist(), "ratio_90_10": np.exp(2 * Z90 * sig).tolist(),
            "beta": par["beta"].tolist(), "theta": par["theta"].tolist(),
            "share_explained": between / (between + within) if between + within > 0 else 0.0,
            # empty when the load does not move with speed (lam = 0, D31)
            "speed_load_corr": ((par["lam"] / np.sqrt(par["lam"] ** 2 + par["tau"] ** 2)).tolist()
                                if null_fit.layout.has("lam") else [])}


def sp3(free_fit: Fit) -> dict:
    """On the full drift matrices (entries off the support are zero in both phenotypes)."""
    A0, A1 = free_fit.params["A"][0].ravel(), free_fit.params["A"][1].ravel()
    s_ls = float(A1 @ A0 / max(A0 @ A0, 1e-12))
    diff = float(np.sum((A1 - A0) ** 2))
    share = 1.0 - float(np.sum((A1 - s_ls * A0) ** 2)) / diff if diff > 0 else np.nan
    return {"s_ls": s_ls, "share": share, "norm_diff": float(np.sqrt(diff))}


def speed_estimates(null_fit: Fit, free_fit: Fit, panel: Panel) -> dict:
    return {"sp1": sp1(null_fit, panel), "sp2": sp2(null_fit, panel), "sp3": sp3(free_fit)}


def _flat(e: dict, free_fit: Fit) -> np.ndarray:
    """Every bootstrapped quantity in one vector (logs for ratios and spreads)."""
    s2 = e["sp2"]
    return np.concatenate([[np.log(e["sp1"])], np.log(s2["sigma"]), s2["beta"], s2["theta"],
                           [s2["share_explained"]], np.ravel(s2["speed_load_corr"]), [e["sp3"]["share"]]])


def sample_setup(panel: Panel, spec: ModelSpec, null_fit: Fit, support, fit_kw: dict | None, seed: int) -> dict:
    """Shared state of the speed-bootstrap workers (parallel.ordered); support is (expert map,
    Step 1 configuration) to re-select the support in every sample, or None to keep the spec's."""
    from .step1 import make_support_fn
    return {"panel": panel, "spec": spec, "null_fit": null_fit, "fit_kw": fit_kw or {}, "seed": seed,
            "support_fn": None if support is None else make_support_fn(*support),
            "groups": [np.where(panel.group == g)[0] for g in range(panel.n_groups)]}


def bootstrap_sample(shared: dict, b: int):
    """Bootstrap sample b, participants resampled within phenotype with the sample's own random
    stream: (estimates as one vector, drift difference), or None if the fits fail."""
    rng = np.random.default_rng([shared["seed"], 1, b])
    idx = np.concatenate([rng.choice(gi, size=len(gi)) for gi in shared["groups"]])
    bp, spec, support_fn = shared["panel"].subset(idx), shared["spec"], shared["support_fn"]
    try:
        sb = spec if support_fn is None else spec.with_support(support_fn(bp))
        nf = shared["null_fit"]
        warm = nf.x if np.array_equal(sb.support, spec.support) else remap_support(nf.x, nf.spec, sb, "null")
        f0, f1 = fit_pair(bp, sb, x0_null=warm, **shared["fit_kw"])
        if f0.failed or f1.failed:
            return None
        e = speed_estimates(f0, f1, bp)
        return _flat(e, f1).tolist(), (f1.params["A"][1] - f1.params["A"][0]).ravel().tolist()
    except Exception:
        return None


def bootstrap_speed(panel: Panel, spec: ModelSpec, null_fit: Fit, free_fit: Fit, B: int = 200,
                    seed: int = 0, fit_kw: dict | None = None, level: float = 0.95, support=None,
                    done: dict | None = None, on_sample=None, jobs: int = 1) -> dict:
    """Participant bootstrap of Sp-1 to Sp-3. support (expert map, Step 1 configuration), if
    given, re-selects the support in every sample, as for the observed fits. Each sample has its
    own seed, so `jobs` samples can run at once with the same result. `done` holds the samples
    already fitted (from a checkpoint); on_sample(done) is called after each new one."""
    est = speed_estimates(null_fit, free_fit, panel)
    theta = _flat(est, free_fit)
    d_hat = (free_fit.params["A"][1] - free_fit.params["A"][0]).ravel()
    done = {"rows": [], "diffs": [], "failures": 0, **(done or {})}
    rows, diffs = done["rows"], done["diffs"]
    start = len(rows) + done["failures"]
    for out in parallel.ordered(bootstrap_sample, range(start, B), jobs, sample_setup,
                                (panel, spec, null_fit, support, fit_kw, seed)):
        if out is None:
            done["failures"] += 1
        else:
            rows.append(out[0])
            diffs.append(out[1])
        if on_sample:
            on_sample(done)
    failures = done["failures"]
    if not rows:
        return {"estimates": est, "B": B, "B_used": 0, "failures": failures,
                "note": "every bootstrap sample failed, so there are no intervals"}
    R, Dm = np.array(rows), np.array(diffs)
    a = (1 - level) / 2
    lo = 2 * theta - np.nanquantile(R, 1 - a, axis=0)
    hi = 2 * theta - np.nanquantile(R, a, axis=0)
    G, p, d = panel.n_groups, len(est["sp2"]["beta"]), len(est["sp2"]["theta"])
    k = np.cumsum([1, G, p, d, 1, len(np.ravel(est["sp2"]["speed_load_corr"])), 1])
    cut = lambda v: np.split(v, k[:-1])
    L, U = cut(lo), cut(hi)
    s3 = est["sp3"]["share"]
    s3_star = R[:, -1]
    t_obs = float(np.linalg.norm(d_hat))
    t_null = np.linalg.norm(Dm - d_hat, axis=1)
    clip = lambda v: float(np.clip(v, 0.0, 1.0))
    return {"estimates": est, "B": B, "B_used": int(len(R)), "failures": failures,
            "sp1_ci": np.exp([L[0][0], U[0][0]]).tolist(),
            "sigma_ci": np.exp(np.column_stack([L[1], U[1]])).tolist(),
            "ratio_90_10_ci": np.exp(2 * Z90 * np.exp(np.column_stack([L[1], U[1]]))).tolist(),
            "beta_ci": np.column_stack([L[2], U[2]]).tolist(),
            "theta_ci": np.column_stack([L[3], U[3]]).tolist(),
            "share_explained_ci": np.clip([float(L[4][0]), float(U[4][0])], 0.0, 1.0).tolist(),
            "speed_load_corr_ci": np.clip(np.column_stack([L[5], U[5]]), -1.0, 1.0).tolist(),
            "sp3_bias_corrected": clip(2 * s3 - np.nanmean(s3_star)),
            "sp3_lower_95": clip(2 * s3 - np.nanquantile(s3_star, 0.95)),
            "sp3_test_statistic": t_obs, "sp3_threshold": float(np.quantile(t_null, 0.95)),
            "sp3_reportable": bool(t_obs > np.quantile(t_null, 0.95))}

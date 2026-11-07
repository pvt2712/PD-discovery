"""H-P, the only primary test (protocol v0.5, §2.4, §7.3.1).

Null: the phenotypes share one drift up to a positive speed factor (with speed, load, noise,
treatment response and starting state free). Alternative: free drifts on the same support.
Statistic: the likelihood ratio. Reference: a conditional parametric bootstrap of the fitted
null model, repeating the support selection in every replicate, with sequential Monte Carlo
p-values (Besag & Clifford 1991): replicates are drawn one at a time up to n_max; drawing stops
once h simulated ratios are at least as large as the observed one, and then p = h / drawn;
otherwise p = (exceedances + 1) / (n_max + 1). Ties count as at least as large. A replicate that
fails (raises, gives a non-finite statistic, or has a fit that could not be evaluated) counts as at
least as large (conservative); a replicate whose fits stop short of the convergence criterion is
counted and its statistic used. A failed observed fit stops the procedure. If the first
GUARD_REPLICATES replicates all fail, all give exactly 0, or all have fits that stopped at their
start, the bootstrap is broken and stops with the reason (the PC run of 4 November 2026 drew 199
replicates of exactly 0 before anyone could see it).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from . import progress
from .data import Panel
from .engine.fit import Fit, fit_pair, remap_support
from .engine.likelihood import prepare
from .engine.model import ModelSpec
from .engine.simulate import simulate_from_fit

GUARD_REPLICATES = 5


@dataclass
class HPResult:
    lr: float
    p: float
    n_drawn: int
    n_exceed: int
    stopped_early: bool
    df: int
    null_fit: Fit
    free_fit: Fit
    support: np.ndarray
    lr_star: list = field(default_factory=list)
    failures: int = 0
    non_converged: int = 0
    lr_raw: float = 0.0
    extras: list = field(default_factory=list)
    replicate_info: list = field(default_factory=list)

    def summary(self) -> dict:
        info = [r for r in self.replicate_info if r]
        evals = [r["null_eval"] + r["free_eval"] for r in info if "null_eval" in r]
        return {"lr": self.lr, "lr_raw": self.lr_raw, "p": self.p, "df": self.df, "n_drawn": self.n_drawn,
                "n_exceed": self.n_exceed, "stopped_early": self.stopped_early,
                "failures": self.failures, "replicates_not_converged": self.non_converged,
                "null_converged": self.null_fit.success, "free_converged": self.free_fit.success,
                "free_start_logliks": self.free_fit.start_logliks,
                "null_fit": self.null_fit.summary(), "free_fit": self.free_fit.summary(),
                "replicates": {"median_evaluations": float(np.median(evals)) if evals else None,
                               "warm_started": sum(bool(r.get("warm")) for r in info),
                               "support_changed": sum(bool(r.get("support_changed")) for r in info),
                               "failure_reasons": sorted({r["error"] for r in info if r.get("error")})[:5]}}


def lr_stat(f0: Fit, f1: Fit) -> float:
    """The likelihood ratio, clamped at 0 (the free fit starts at the null solution, so a negative
    raw value would only reflect rounding; HPResult keeps the raw value). NaN if either fit
    failed, so a failed replicate counts as a failure."""
    if f0.failed or f1.failed:
        return float("nan")
    return max(2.0 * (f1.loglik - f0.loglik), 0.0)


def degenerate(outs: list) -> str | None:
    """Why the first bootstrap replicates look broken, or None: every one failed, every statistic
    is exactly 0, or every fit stopped at its start. outs are _replicate's tuples."""
    if not outs:
        return None
    infos = [o[4] if len(o) > 4 and o[4] else {} for o in outs]
    if not any(o[1] for o in outs):
        reasons = sorted({i.get("error", "unknown") for i in infos})
        return f"all of the first {len(outs)} replicates failed ({'; '.join(reasons)[:300]})"
    if all(o[1] and o[0] == 0.0 for o in outs):
        return f"all of the first {len(outs)} replicates gave a statistic of exactly 0"
    if all(i.get("null_eval", 99) <= 3 * max(i.get("null_starts", 1), 1) and
           i.get("free_eval", 99) <= 3 * max(i.get("free_starts", 1), 1) for i in infos):
        return f"in all of the first {len(outs)} replicates the fits stopped at their start"
    return None


def h_p_df(spec: ModelSpec) -> int:
    """Degrees of freedom of H-P: (G - 1)(n_sup - 1)."""
    return (spec.n_groups - 1) * (spec.n_sup - 1)


def _support(panel: Panel, spec: ModelSpec, support_fn) -> ModelSpec:
    return spec if support_fn is None else spec.with_support(support_fn(panel))


def _links(spec: ModelSpec) -> int:
    """Links between domains in the support (its diagonal holds each domain's own rate)."""
    return int(spec.support.sum() - spec.d)


def fit_info(g0: Fit, g1: Fit) -> dict:
    """What a replicate records about its two fits (evaluations, iterations, how they stopped)."""
    info = {"null_eval": g0.n_eval, "free_eval": g1.n_eval, "null_iter": g0.n_iter, "free_iter": g1.n_iter,
            "null_starts": len(g0.starts), "free_starts": len(g1.starts),
            "null_stopped_by": g0.message, "free_stopped_by": g1.message}
    if g0.failed or g1.failed:
        info["error"] = (g0 if g0.failed else g1).message
    return info


def _replicate(f0: Fit, panel: Panel, spec: ModelSpec, support_fn, rng, fit_kw, on_replicate=None):
    """One bootstrap replicate: simulate from the fitted null, re-select the support, refit both
    models, the null warm-started at the fitted null (written for the replicate's support when that
    differs). Returns (statistic, ok, converged, extra, info). A replicate that raises, gives a
    non-finite statistic or has a failed fit is a failure (ok False), counted as at least as large
    as the observed statistic; non-convergence is recorded but the statistic is used."""
    try:
        sim = simulate_from_fit(f0, panel, rng)
        spec_b = _support(sim, spec, support_fn)
        changed = not np.array_equal(spec_b.support, f0.spec.support)
        progress.report("dataset simulated from the fitted speed-only model; support "
                        + ("kept" if support_fn is None else
                           f"re-selected: {'changed, ' + str(_links(spec_b)) + ' links' if changed else 'the same as observed'}"))
        warm = remap_support(f0.x, f0.spec, spec_b, "null") if changed else f0.x
        g0, g1 = fit_pair(sim, spec_b, prep=prepare(sim, spec_b.meas_var, spec_b.n_groups),
                          x0_null=warm, **fit_kw)
        stat = lr_stat(g0, g1)
        info = {**fit_info(g0, g1), "warm": True, "support_changed": changed}
        extra = on_replicate(sim, g0, g1) if on_replicate is not None and np.isfinite(stat) else None
    except Exception as e:
        return np.inf, False, False, None, {"error": f"{type(e).__name__}: {str(e)[:200]}"}
    if not np.isfinite(stat):
        return np.inf, False, False, extra, info
    return stat, True, bool(g0.success and g1.success), extra, info


def boot_setup(f0: Fit, panel: Panel, spec: ModelSpec, expert, step1_cfg, fit_kw: dict | None, seed: int) -> dict:
    """Shared state of the bootstrap workers (parallel.ordered): the support is re-selected with
    `step1_cfg` around the `expert` map, as for the observed fits; step1_cfg None keeps the spec's."""
    from .step1 import make_support_fn
    return {"f0": f0, "panel": panel, "spec": spec, "fit_kw": fit_kw or {}, "seed": seed,
            "support_fn": None if step1_cfg is None else make_support_fn(expert, step1_cfg)}


def boot_replicate(shared: dict, i: int) -> tuple:
    """Replicate i with its own random stream, so the bootstrap does not depend on how many
    replicates run at once: (statistic, ok, converged, None, info)."""
    rng = np.random.default_rng([shared["seed"], 3, i])
    t0 = time.time()
    with progress.piece(f"replicate {i + 1}"):
        stat, ok, conv, _, info = _replicate(shared["f0"], shared["panel"], shared["spec"], shared["support_fn"],
                                             rng, shared["fit_kw"])
        progress.report(f"finished in {progress.duration(time.time() - t0)}: "
                        + (f"statistic {stat:.2f}" if ok else f"failed ({info.get('error', 'unknown')})"))
    return stat, ok, conv, None, {**info, "seconds": time.time() - t0}


def sequential_p(draw, observed: float, n_max: int = 999, h: int = 20, callback=None, resume: list | None = None) -> dict:
    """Besag-Clifford sequential Monte Carlo p-value. draw() returns (statistic, ok) or a longer
    tuple starting with them; a failed replicate (ok False) counts as at least as large as the
    observed statistic. resume continues from earlier (statistic, ok) pairs (checkpointing).
    callback(drawn, exceed, statistic, ok, rest) is called after each new replicate, rest being the
    draw's tuple after (statistic, ok)."""
    exceed, drawn, failures, stars = 0, 0, 0, []
    rest = []
    for prev in (resume or []):
        stat, ok = float(prev[0]), bool(prev[1])
        drawn += 1
        failures += int(not ok)
        stars.append(stat)
        exceed += int(not ok or stat >= observed)
    while drawn < n_max and exceed < h:
        out = draw()
        stat, ok = out[0], out[1]
        rest.append(out[2:])
        drawn += 1
        failures += int(not ok)
        stars.append(float(stat))
        if not ok or stat >= observed:
            exceed += 1
        if callback is not None:
            callback(drawn, exceed, float(stat), bool(ok), tuple(out[2:]))
    if exceed >= h:
        return {"p": h / drawn, "n_drawn": drawn, "n_exceed": exceed, "stopped_early": True,
                "failures": failures, "stars": stars, "rest": rest}
    return {"p": (exceed + 1) / (n_max + 1), "n_drawn": drawn, "n_exceed": exceed, "stopped_early": False,
            "failures": failures, "stars": stars, "rest": rest}


def observed_fits(panel: Panel, spec: ModelSpec, support_fn=None, fit_kw: dict | None = None):
    """Support selection and the two observed fits: (spec with the selected support, null, free).
    Raises if either fit failed: there is no statistic to test."""
    t0 = time.time()
    if support_fn is not None:
        progress.report("support selection (Step 1): started")
    spec_s = _support(panel, spec, support_fn)
    if support_fn is not None:
        progress.report(f"support selected in {progress.duration(time.time() - t0)}: {_links(spec_s)} links "
                        f"between domains; next, the speed-only and free fits")
    f0, f1 = fit_pair(panel, spec_s, **(fit_kw or {}))
    for name, f in (("speed-only", f0), ("free", f1)):
        if f.failed:
            raise RuntimeError(f"the observed {name} fit failed: {f.message}")
    return spec_s, f0, f1


def run_hp(panel: Panel, spec: ModelSpec, support_fn=None, n_max: int = 999, h: int = 20,
           rng: np.random.Generator | None = None, fit_kw: dict | None = None,
           callback=None, resume: list | None = None, on_replicate=None, fits: tuple | None = None,
           draw=None) -> HPResult:
    """The full H-P procedure on one dataset (real data, or a single simulated dataset).

    callback(drawn, exceed, statistic, ok, rest) is called after every replicate (for
    checkpoints); resume continues from saved (statistic, ok) pairs; on_replicate(sim, null_fit,
    free_fit) returns extra per-replicate output (for example per-link p-values for
    recalibration); fits reuses the output of observed_fits; draw, if given, supplies the
    replicates instead (for example boot_replicate run in parallel, in replicate order). Unless the
    run resumes past them, the first GUARD_REPLICATES replicates are checked (degenerate()) and a
    broken bootstrap stops with RuntimeError.
    """
    rng = rng or np.random.default_rng()
    fit_kw = fit_kw or {}
    spec_s, f0, f1 = fits if fits is not None else observed_fits(panel, spec, support_fn, fit_kw)
    lr_raw = 2.0 * (f1.loglik - f0.loglik)
    lr = lr_stat(f0, f1)
    draw = draw or (lambda: _replicate(f0, panel, spec, support_fn, rng, fit_kw, on_replicate))
    first, k = [], min(GUARD_REPLICATES, n_max) - len(resume or [])

    def checked_draw():
        out = draw()
        if len(first) < k:
            first.append(out)
            reason = degenerate(first) if len(first) == k else None
            if reason:
                raise RuntimeError(f"H-P bootstrap stopped: {reason}")
        return out

    seq = sequential_p(checked_draw, lr, n_max, h, callback, resume)
    non_conv = sum(1 for r in seq["rest"] if r and not r[0])
    extras = [r[1] for r in seq["rest"] if r and len(r) > 1 and r[1] is not None]
    infos = [r[2] if len(r) > 2 else {} for r in seq["rest"]]
    return HPResult(lr, seq["p"], seq["n_drawn"], seq["n_exceed"], seq["stopped_early"], h_p_df(spec_s),
                    f0, f1, spec_s.support, seq["stars"], seq["failures"], non_conv, lr_raw, extras, infos)


def warp_replicate(panel: Panel, spec: ModelSpec, support_fn=None, rng: np.random.Generator | None = None,
                   fit_kw: dict | None = None) -> dict:
    """One dataset of the warp-speed calibration (Giacomini 2013): the observed ratio and a
    single bootstrap ratio. Pooled over datasets, the bootstrap ratios give the reference."""
    rng = rng or np.random.default_rng()
    fit_kw = fit_kw or {}
    spec_s = _support(panel, spec, support_fn)
    f0, f1 = fit_pair(panel, spec_s, **fit_kw)
    lr_b, ok, conv, _, _ = _replicate(f0, panel, spec, support_fn, rng, fit_kw)
    return {"lr": lr_stat(f0, f1), "lr_star": float(lr_b), "replicate_ok": ok, "replicate_converged": conv,
            "converged": bool(f0.success and f1.success), "df": h_p_df(spec_s),
            "null_fit": f0, "free_fit": f1}


def reference_distribution(generate, n_datasets: int, spec: ModelSpec, support_fn=None,
                           rng: np.random.Generator | None = None, fit_kw: dict | None = None) -> list:
    """The definition-matched robustness analysis (§7.3.6): H-P's statistic, with support
    selection, on datasets from a generator that re-applies the phenotype rule to simulated
    baselines under one shared mechanism. generate(rng) returns a Panel."""
    rng = rng or np.random.default_rng()
    fit_kw = fit_kw or {}
    out = []
    for _ in range(n_datasets):
        p = generate(rng)
        try:
            s = _support(p, spec, support_fn)
            f0, f1 = fit_pair(p, s, **fit_kw)
            out.append(lr_stat(f0, f1))
        except Exception:
            out.append(float("nan"))
    return out

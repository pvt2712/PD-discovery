"""H-P, the only primary test (protocol v0.5, §2.4, §7.3.1).

Null: the phenotypes share one drift up to a positive speed factor (with speed, load, noise,
treatment response and starting state free). Alternative: free drifts on the same support.
Statistic: the likelihood ratio. Reference: a conditional parametric bootstrap of the fitted
null model, repeating the support selection in every replicate, with sequential Monte Carlo
p-values (Besag & Clifford 1991): replicates are drawn one at a time up to n_max; drawing stops
once h simulated ratios are at least as large as the observed one, and then p = h / drawn;
otherwise p = (exceedances + 1) / (n_max + 1). Ties count as at least as large. A replicate that
fails (raises, or gives a non-finite statistic) counts as at least as large (conservative); a
replicate whose fits stop short of the convergence criterion is counted and its statistic used.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .data import Panel
from .engine.fit import Fit, fit_pair
from .engine.likelihood import prepare
from .engine.model import ModelSpec
from .engine.simulate import simulate_from_fit


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

    def summary(self) -> dict:
        return {"lr": self.lr, "lr_raw": self.lr_raw, "p": self.p, "df": self.df, "n_drawn": self.n_drawn,
                "n_exceed": self.n_exceed, "stopped_early": self.stopped_early,
                "failures": self.failures, "replicates_not_converged": self.non_converged,
                "null_converged": self.null_fit.success, "free_converged": self.free_fit.success,
                "free_start_logliks": self.free_fit.start_logliks}


def lr_stat(f0: Fit, f1: Fit) -> float:
    """The likelihood ratio, clamped at 0 (the free fit starts at the null solution, so a negative
    raw value would only reflect rounding; HPResult keeps the raw value)."""
    return max(2.0 * (f1.loglik - f0.loglik), 0.0)


def h_p_df(spec: ModelSpec) -> int:
    """Degrees of freedom of H-P: (G - 1)(n_sup - 1)."""
    return (spec.n_groups - 1) * (spec.n_sup - 1)


def _support(panel: Panel, spec: ModelSpec, support_fn) -> ModelSpec:
    return spec if support_fn is None else spec.with_support(support_fn(panel))


def _replicate(f0: Fit, panel: Panel, spec: ModelSpec, support_fn, rng, fit_kw, on_replicate=None):
    """One bootstrap replicate: simulate from the fitted null, re-select the support, refit both
    models. Returns (statistic, ok, converged, extra). A replicate that raises or gives a
    non-finite statistic is a failure (ok False), counted as at least as large as the observed
    statistic; non-convergence is recorded but the statistic is used."""
    try:
        sim = simulate_from_fit(f0, panel, rng)
        spec_b = _support(sim, spec, support_fn)
        warm = f0.x if np.array_equal(spec_b.support, f0.spec.support) else None
        g0, g1 = fit_pair(sim, spec_b, prep=prepare(sim, spec_b.meas_var, spec_b.n_groups),
                          x0_null=warm, **fit_kw)
        stat = lr_stat(g0, g1)
        extra = on_replicate(sim, g0, g1) if on_replicate is not None else None
    except Exception:
        return np.inf, False, False, None
    if not np.isfinite(stat):
        return np.inf, False, False, extra
    return stat, True, bool(g0.success and g1.success), extra


def sequential_p(draw, observed: float, n_max: int = 999, h: int = 20, callback=None, resume: list | None = None) -> dict:
    """Besag-Clifford sequential Monte Carlo p-value. draw() returns (statistic, ok) or a longer
    tuple starting with them; a failed replicate (ok False) counts as at least as large as the
    observed statistic. resume continues from earlier (statistic, ok) pairs (checkpointing)."""
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
            callback(drawn, exceed, float(stat), bool(ok))
    if exceed >= h:
        return {"p": h / drawn, "n_drawn": drawn, "n_exceed": exceed, "stopped_early": True,
                "failures": failures, "stars": stars, "rest": rest}
    return {"p": (exceed + 1) / (n_max + 1), "n_drawn": drawn, "n_exceed": exceed, "stopped_early": False,
            "failures": failures, "stars": stars, "rest": rest}


def observed_fits(panel: Panel, spec: ModelSpec, support_fn=None, fit_kw: dict | None = None):
    """Support selection and the two observed fits: (spec with the selected support, null, free)."""
    spec_s = _support(panel, spec, support_fn)
    f0, f1 = fit_pair(panel, spec_s, **(fit_kw or {}))
    return spec_s, f0, f1


def run_hp(panel: Panel, spec: ModelSpec, support_fn=None, n_max: int = 999, h: int = 20,
           rng: np.random.Generator | None = None, fit_kw: dict | None = None,
           callback=None, resume: list | None = None, on_replicate=None, fits: tuple | None = None) -> HPResult:
    """The full H-P procedure on one dataset (real data, or a single simulated dataset).

    callback(drawn, exceed, statistic, ok) is called after every replicate (for checkpoints);
    resume continues from saved (statistic, ok) pairs; on_replicate(sim, null_fit, free_fit)
    returns extra per-replicate output (for example per-link p-values for recalibration);
    fits reuses the output of observed_fits.
    """
    rng = rng or np.random.default_rng()
    fit_kw = fit_kw or {}
    spec_s, f0, f1 = fits if fits is not None else observed_fits(panel, spec, support_fn, fit_kw)
    lr_raw = 2.0 * (f1.loglik - f0.loglik)
    lr = lr_stat(f0, f1)
    seq = sequential_p(lambda: _replicate(f0, panel, spec, support_fn, rng, fit_kw, on_replicate),
                       lr, n_max, h, callback, resume)
    non_conv = sum(1 for r in seq["rest"] if r and not r[0])
    extras = [r[1] for r in seq["rest"] if r and len(r) > 1]
    return HPResult(lr, seq["p"], seq["n_drawn"], seq["n_exceed"], seq["stopped_early"], h_p_df(spec_s),
                    f0, f1, spec_s.support, seq["stars"], seq["failures"], non_conv, lr_raw, extras)


def warp_replicate(panel: Panel, spec: ModelSpec, support_fn=None, rng: np.random.Generator | None = None,
                   fit_kw: dict | None = None) -> dict:
    """One dataset of the warp-speed calibration (Giacomini 2013): the observed ratio and a
    single bootstrap ratio. Pooled over datasets, the bootstrap ratios give the reference."""
    rng = rng or np.random.default_rng()
    fit_kw = fit_kw or {}
    spec_s = _support(panel, spec, support_fn)
    f0, f1 = fit_pair(panel, spec_s, **fit_kw)
    lr_b, ok, conv, _ = _replicate(f0, panel, spec, support_fn, rng, fit_kw)
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

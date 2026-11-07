"""Maximum-likelihood fitting of the engine (protocol v0.5, §7.1).

Start values come from yearly least-squares fits per phenotype mapped to continuous time.
The free model is started both at its own start and at the null fit written in its parameters,
so the likelihood ratio of H-P cannot come out negative from a poor local optimum. The null is
then restarted from the free fit projected onto it (one speed per phenotype), and the two fits
alternate until neither improves, so a speed-only fit stuck in the wrong speed basin cannot
inflate the ratio (audit A3, after the PC run of 4 November 2026).

A fit stops when L-BFGS-B converges, or when the log-likelihood has gained less than `min_gain`
over the last `window` iterations, or at `maxiter`. A start where the model cannot be evaluated
is never "converged": the fit is marked failed (audit A1).

While a fit runs it reports progress (pdx.progress, when a script asks for it): each start's
beginning and end, and every few minutes its iteration, evaluations, log-likelihood and gain over
the stopping window. Reporting never changes the fit.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy import linalg, optimize

from .. import progress
from ..data import Panel
from .likelihood import FAILED, Prepared, prepare, value_and_grad
from .model import Layout, ModelSpec, pack_chol, pack_drift, unpack_numpy


@dataclass
class Fit:
    kind: str
    spec: ModelSpec
    layout: Layout
    x: np.ndarray
    loglik: float                 # -inf if the fit failed
    success: bool                 # stopped by a convergence rule (L-BFGS-B's or the window), not a limit
    message: str
    n_eval: int                   # likelihood evaluations over all starts
    params: dict = field(default_factory=dict)
    start_logliks: list = field(default_factory=list)   # final log-likelihood from each start (-inf: failed)
    n_iter: int = 0               # iterations of the kept start
    failed: bool = False          # the model could not be evaluated at any start (or at the end)
    starts: list = field(default_factory=list)          # per start: label, loglik, stopped_by, n_eval, n_iter

    @property
    def n_params(self) -> int:
        return self.layout.size

    def summary(self) -> dict:
        """What a report keeps about the fit (no participant data)."""
        return {"kind": self.kind, "loglik": self.loglik, "success": self.success, "failed": self.failed,
                "message": self.message, "n_eval": self.n_eval, "n_iter": self.n_iter, "starts": self.starts}

    def to_record(self) -> dict:
        """The fit as plain values, parameters included, for a checkpoint or a report."""
        return {**self.summary(), "x": [float(v) for v in self.x], "start_logliks": list(self.start_logliks)}


def fit_from_record(rec: dict, spec: ModelSpec) -> Fit:
    """A Fit rebuilt from Fit.to_record() for the same model (a checkpoint's observed fits)."""
    lay = Layout(spec, rec["kind"])
    x = np.asarray(rec["x"], float)
    if len(x) != lay.size:
        raise ValueError(f"the saved {rec['kind']} fit has {len(x)} parameters, the model {lay.size}")
    ll = -np.inf if rec.get("loglik") is None else float(rec["loglik"])
    return Fit(rec["kind"], spec, lay, x, ll, bool(rec["success"]), rec["message"], int(rec["n_eval"]),
               unpack_numpy(x, lay), [(-np.inf if v is None else v) for v in rec.get("start_logliks", [])],
               int(rec.get("n_iter", 0)), bool(rec.get("failed", False)), list(rec.get("starts", [])))


# ---------------------------------------------------------------- start values

def _yearly_ols(panel: Panel, g: int, d: int):
    Xs, Ys = [], []
    for i in np.where(panel.group == g)[0]:
        y, t = panel.y[i], panel.t[i]
        for k in range(len(t) - 1):
            if 0.5 <= t[k + 1] - t[k] <= 1.6 and not (np.isnan(y[k]).any() or np.isnan(y[k + 1]).any()):
                Xs.append(y[k])
                Ys.append(y[k + 1])
    if len(Xs) < d + 5:
        return 0.8 * np.eye(d), np.zeros(d), 0.3 * np.eye(d)
    X, Y = np.array(Xs), np.array(Ys)
    Z = np.column_stack([np.ones(len(X)), X])
    C, *_ = np.linalg.lstsq(Z, Y, rcond=None)
    res = Y - Z @ C
    return C[1:].T, C[0], np.cov(res.T).reshape(d, d)


def _drift_from_yearly(B: np.ndarray, spec: ModelSpec) -> np.ndarray:
    try:
        A = np.real(linalg.logm(B))
        if not np.all(np.isfinite(A)):
            raise ValueError
    except Exception:
        A = B - np.eye(len(B))
    A = np.where(spec.support, A, 0.0)
    dg = np.minimum(np.diag(A), -0.05)
    A[np.diag_indices(len(A))] = dg
    return A


def _load_from_intercept(A: np.ndarray, a: np.ndarray, s: float) -> np.ndarray:
    d = len(A)
    B1 = np.zeros((2 * d, 2 * d))
    B1[:d, :d], B1[:d, d:] = s * A, s * np.eye(d)
    M = linalg.expm(B1)[:d, d:]
    return np.linalg.solve(M, a)


def start_values(panel: Panel, spec: ModelSpec, kind: str) -> np.ndarray:
    lay = Layout(spec, kind)
    d, G = spec.d, spec.n_groups
    fits = [_yearly_ols(panel, g, d) for g in range(G)]
    As = [_drift_from_yearly(B, spec) for B, _, _ in fits]
    R = np.diag(spec.meas_var)
    x = np.zeros(lay.size)
    A0 = As[0]
    sp = np.ones(G)
    if kind in ("null", "joint"):
        for g in range(1, G):
            sp[g] = float(np.clip(np.sum(As[g] * A0) / max(np.sum(A0 * A0), 1e-8), 0.3, 5.0))
        lay.put(x, "gamma", np.log(sp[1:]))
    lay.put(x, "A", pack_drift(A0, spec))
    if kind == "free":
        for g in range(1, G):
            lay.put(x, f"A{g}", pack_drift(As[g], spec))
    if kind == "joint":
        for g in range(1, G):
            lay.put(x, f"D{g}", np.zeros(spec.n_sup))
    b = np.zeros((G, d))
    L = np.zeros((G, d * (d + 1) // 2))
    for g, (B, a, Sig) in enumerate(fits):
        Ag = As[g] if kind == "free" else A0
        s = 1.0 if kind == "free" else sp[g]
        b[g] = _load_from_intercept(Ag, a, s)
        Qg = (Sig - R - B @ R @ B.T) / s
        L[g] = pack_chol(Qg)
    lay.put(x, "b", b)
    lay.put(x, "L", L)
    lay.put(x, "log_sigma", np.full(G, np.log(0.3)))
    lay.put(x, "log_tau", np.full((G, d), np.log(0.1)))
    return x


def null_to(x_null: np.ndarray, spec: ModelSpec, kind: str) -> np.ndarray:
    """The null fit written in the parameters of the free or joint model."""
    l0, l1 = Layout(spec, "null"), Layout(spec, kind)
    x = np.zeros(l1.size)
    for name, _ in l1.blocks:
        if name in l0.slices and l0.has(name) and name != "gamma":
            l1.put(x, name, l0.get(x_null, name))
    if kind == "joint":
        l1.put(x, "gamma", l0.get(x_null, "gamma"))
        for g in range(1, spec.n_groups):
            l1.put(x, f"D{g}", np.zeros(spec.n_sup))
        return x
    par = unpack_numpy(x_null, l0)
    eg = np.exp(par["log_speed"])
    L = l0.get(x_null, "L").copy()
    b = par["b"].copy()
    for g in range(1, spec.n_groups):
        l1.put(x, f"A{g}", pack_drift(eg[g] * par["A"][g], spec))
        b[g] = eg[g] * b[g]
        L[g] = pack_chol(eg[g] * par["Q"][g])
    l1.put(x, "b", b)
    l1.put(x, "L", L)
    # the load distribution is on the reference time scale, so it is rescaled with the speed too
    for name in ("gam_y", "gam_z", "lam"):
        if l1.has(name):
            v = l0.get(x_null, name).copy()
            v[1:] *= eg[1:].reshape(-1, *([1] * (v.ndim - 1)))
            l1.put(x, name, v)
    lt = l0.get(x_null, "log_tau").copy()
    lt[1:] += par["log_speed"][1:, None]
    l1.put(x, "log_tau", lt)
    return x


def free_to_null(x_free: np.ndarray, spec: ModelSpec, max_ratio: float = 20.0) -> np.ndarray:
    """The free fit projected onto the null model, as a start for it (the inverse of null_to): the
    reference phenotype's drift as the shared one, and for each other phenotype the speed that
    brings its drift closest to it (least squares over the drift entries, kept within
    [1/max_ratio, max_ratio]), which also rescales its load mean, intercept and noise."""
    lf, l0 = Layout(spec, "free"), Layout(spec, "null")
    par = unpack_numpy(x_free, lf)
    x = np.zeros(l0.size)
    for name, _ in l0.blocks:
        if name not in ("A", "gamma") and lf.has(name):
            l0.put(x, name, lf.get(x_free, name))
    A0 = par["A"][0]
    s = np.ones(spec.n_groups)
    for g in range(1, spec.n_groups):
        s[g] = float(np.clip(np.sum(par["A"][g] * A0) / max(np.sum(A0 * A0), 1e-12), 1 / max_ratio, max_ratio))
    l0.put(x, "A", pack_drift(A0, spec))
    l0.put(x, "gamma", np.log(s[1:]))
    b, L = par["b"].copy(), lf.get(x_free, "L").copy()
    for g in range(1, spec.n_groups):
        b[g] = b[g] / s[g]
        L[g] = pack_chol(par["Q"][g] / s[g])
    l0.put(x, "b", b)
    l0.put(x, "L", L)
    for name in ("gam_y", "gam_z", "lam"):
        if l0.has(name):
            v = lf.get(x_free, name).copy()
            v[1:] /= s[1:].reshape(-1, *([1] * (v.ndim - 1)))
            l0.put(x, name, v)
    lt = lf.get(x_free, "log_tau").copy()
    lt[1:] -= np.log(s[1:])[:, None]
    l0.put(x, "log_tau", lt)
    return x


def remap_support(x: np.ndarray, spec_from: ModelSpec, spec_to: ModelSpec, kind: str) -> np.ndarray:
    """Parameters of a fit on one support written for another (a warm start when a bootstrap
    sample re-selects a different support): drift entries on both supports are copied, entries
    new to spec_to start at 0, and everything that does not depend on the support is copied."""
    la, lb = Layout(spec_from, kind), Layout(spec_to, kind)
    pos = {tuple(ij): k for k, ij in enumerate(spec_from.sup_idx)}
    take = np.array([pos.get(tuple(ij), -1) for ij in spec_to.sup_idx])
    out = np.zeros(lb.size)
    for name, _ in lb.blocks:
        if not lb.has(name):
            continue
        v = la.get(x, name)
        if name == "A" or name[0] in "AD" and name[1:].isdigit():
            w = np.where(take >= 0, v[np.maximum(take, 0)], 0.0)
            lb.put(out, name, w)
        else:
            lb.put(out, name, v)
    return out


def sigma_start(panel: Panel, spec: ModelSpec, kind: str, sigma: float = 0.6) -> np.ndarray:
    """The default start with the speed spread well away from zero (audit A3: one speed-only
    start stalled at a spread of zero)."""
    x = start_values(panel, spec, kind)
    Layout(spec, kind).put(x, "log_sigma", np.full(spec.n_groups, np.log(sigma)))
    return x


# ---------------------------------------------------------------- fitting

KIND_NAMES = {"null": "speed-only fit", "free": "free fit", "joint": "joint fit"}


def fit_model(panel: Panel, spec: ModelSpec, kind: str, x0: np.ndarray | None = None,
              extra_starts: list | None = None, prep: Prepared | None = None,
              maxiter: int = 3000, chunk: int | None = None, penalty=None, ftol: float = 1e-10,
              gtol: float = 1e-5, bounds=None, use_default_start: bool = True, maxcor: int = 50,
              window: int = 100, min_gain: float = 0.01, labels: list | None = None) -> Fit:
    """Fit one kind of model; the best of the starts by likelihood is kept.

    x0 replaces the default start (for example a warm start in a bootstrap replicate);
    extra_starts are tried as well; labels name the starts in Fit.starts. A start where the
    model cannot be evaluated is skipped and recorded; if no start can be evaluated the fit is
    failed (loglik -inf, success False), never converged. Each start stops at L-BFGS-B's own
    convergence, when the log-likelihood gained less than min_gain over the last `window`
    iterations (window 0 switches this off), or at maxiter.
    """
    lay = Layout(spec, kind)
    prep = prep or prepare(panel, spec.meas_var, spec.n_groups)
    starts = []
    if x0 is not None:
        starts.append(x0)
    elif use_default_start:
        starts.append(start_values(panel, spec, kind))
    starts += list(extra_starts or [])
    labels = list(labels or []) + [f"start {k + 1}" for k in range(len(labels or []), len(starts))]
    count = {"n": 0}
    clock = {"eval": 0.0, "gmax": float("nan")}       # seconds spent evaluating; the last gradient's size
    errors: list = []
    name = KIND_NAMES.get(kind, f"{kind} fit")
    rule = f"; a start ends once it gains less than {min_gain:g} over {window} iterations" if window else ""

    def f(v):
        count["n"] += 1
        t = time.perf_counter()
        out = value_and_grad(v, lay, prep, chunk, penalty, errors)
        clock["eval"] += time.perf_counter() - t
        if len(out[1]):
            clock["gmax"] = float(np.max(np.abs(out[1])))
        return out

    best, finals, records = None, [], []
    for k, (s0, label) in enumerate(zip(starts, labels)):
        s0 = np.asarray(s0, float)
        n0, e0, t_start = count["n"], clock["eval"], time.time()
        where = f"{name}, start '{label}' ({k + 1} of {len(starts)})"
        v0, _ = f(s0)
        if v0 >= FAILED:
            progress.report(f"{where}: the model cannot be evaluated at this start, so it is skipped")
            finals.append(-np.inf)
            records.append({"label": label, "loglik": -np.inf, "stopped_by": "not evaluable at the start",
                            "error": errors[-1] if errors else "", "n_eval": count["n"] - n0, "n_iter": 0})
            continue
        progress.report(f"{where}: begins at log-likelihood {-v0:.2f}; one evaluation takes "
                        f"{clock['eval'] - e0:.1f} s; at most {maxiter} iterations{rule}")
        hist, stalled = [], {"flag": False}
        beat = progress.Heartbeat()

        def callback(intermediate_result):
            hist.append(float(intermediate_result.fun))
            if beat.due():
                w = min(window, len(hist) - 1) if window else len(hist) - 1
                n_ev = count["n"] - n0
                progress.report(f"{where}: iteration {len(hist)}, {n_ev} evaluations, log-likelihood "
                                f"{-hist[-1]:.2f}, gained {hist[-w - 1] - hist[-1]:.3g} over the last {w} iterations, "
                                f"largest gradient {clock['gmax']:.2g}, {(clock['eval'] - e0) / max(n_ev, 1):.1f} s "
                                f"per evaluation, {progress.duration(time.time() - t_start)} so far")
            if window and len(hist) > window and hist[-window - 1] - hist[-1] < min_gain:
                stalled["flag"] = True
                raise StopIteration

        r = optimize.minimize(f, s0, jac=True, method="L-BFGS-B", bounds=bounds, callback=callback,
                              options={"maxiter": maxiter, "maxfun": 4 * maxiter, "ftol": ftol, "gtol": gtol,
                                       "maxcor": maxcor})
        ok = np.isfinite(r.fun) and r.fun < FAILED
        stopped = ("window" if stalled["flag"] else "converged" if r.success
                   else "iteration limit" if r.nit >= maxiter or "LIMIT" in str(r.message).upper()
                   else str(r.message))
        finals.append(-float(r.fun) if ok else -np.inf)
        records.append({"label": label, "loglik": finals[-1], "stopped_by": stopped if ok else "not evaluable at the end",
                        "n_eval": count["n"] - n0, "n_iter": int(r.nit)})
        if ok and (best is None or r.fun < best[0].fun):
            best = (r, stopped)
        progress.report(f"{where}: ended ({records[-1]['stopped_by']}) after {int(r.nit)} iterations and "
                        f"{count['n'] - n0} evaluations in {progress.duration(time.time() - t_start)}; log-likelihood "
                        f"{finals[-1]:.2f}" + ("; the best start so far" if len(starts) > 1 and best and best[0] is r else ""))
    if best is None:
        reason = errors[-1] if errors else "no start could be evaluated"
        return Fit(kind, spec, lay, np.asarray(starts[0], float), -np.inf, False,
                   f"failed: the model could not be evaluated ({reason})", count["n"],
                   unpack_numpy(np.asarray(starts[0], float), lay), finals, 0, True, records)
    r, stopped = best
    return Fit(kind, spec, lay, r.x, -float(r.fun), stopped in ("converged", "window"), stopped, count["n"],
               unpack_numpy(r.x, lay), finals, int(r.nit), False, records)


def _merge(old: Fit, new: Fit) -> Fit:
    """The better of two fits of the same model, with both fits' start records."""
    best = new if new.loglik > old.loglik else old
    return Fit(best.kind, best.spec, best.layout, best.x, best.loglik, best.success, best.message,
               old.n_eval + new.n_eval, best.params, old.start_logliks + new.start_logliks,
               best.n_iter, old.failed and new.failed, old.starts + new.starts)


def fit_pair(panel: Panel, spec: ModelSpec, prep: Prepared | None = None, x0_null: np.ndarray | None = None,
             own_free_start: bool = True, sigma_starts: bool | None = None, rounds: int = 3, tol: float = 0.01,
             **kw) -> tuple[Fit, Fit]:
    """The null and free fits of H-P (§7.1).

    The speed-only (null) fit starts at x0_null if given (a bootstrap replicate's warm start, at the
    model that generated it), otherwise at the default start and, with sigma_starts (the default
    when there is no warm start), also with the speed spread away from zero. The free fit starts at
    the null solution written in its parameters, so its likelihood is at least the null's and the
    likelihood ratio is never negative, and at its own default start (own_free_start). Then the
    null is restarted from the free fit projected onto it (free_to_null), and while that improves
    the null by more than `tol` the free fit is restarted from the new null; at most `rounds`
    rounds. Observed and replicate fits follow these same rules; the observed fit, which has no
    warm start, gets the extra default starts instead. If either fit fails, both are returned as
    they are and the caller decides (H-P counts a failed replicate as at least as large).
    """
    prep = prep or prepare(panel, spec.meas_var, spec.n_groups)
    t0 = time.time()
    sigma_starts = x0_null is None if sigma_starts is None else sigma_starts
    if x0_null is not None:
        null_starts, null_labels = [x0_null], ["warm"]
    else:
        null_starts, null_labels = [start_values(panel, spec, "null")], ["default"]
    if sigma_starts:
        null_starts.append(sigma_start(panel, spec, "null"))
        null_labels.append("sigma 0.6")
    f0 = fit_model(panel, spec, "null", x0=null_starts[0], extra_starts=null_starts[1:], prep=prep,
                   labels=null_labels, **kw)
    if f0.failed:
        progress.report("the speed-only fit could not be evaluated at any start, so the free fit is not run")
        lay = Layout(spec, "free")
        x = null_to(f0.x, spec, "free")
        return f0, Fit("free", spec, lay, x, -np.inf, False, "not fitted: the speed-only fit failed", 0,
                       unpack_numpy(x, lay), [], 0, True, [])
    f1 = fit_model(panel, spec, "free", x0=null_to(f0.x, spec, "free"), prep=prep,
                   extra_starts=[start_values(panel, spec, "free")] if own_free_start else None,
                   labels=["null embedding", "own default"], **kw)
    if f1.failed:
        progress.report(f"the free fit could not be evaluated at any start ({f1.message})")
    else:
        progress.report(f"speed-only fit {f0.loglik:.2f}, free fit {f1.loglik:.2f}: likelihood ratio so far "
                        f"{2 * (f1.loglik - f0.loglik):.2f}"
                        + ("; next, the speed-only fit restarts from the free fit" if rounds else ""))
    for k in range(rounds):
        if f1.failed:
            break
        r0 = fit_model(panel, spec, "null", x0=free_to_null(f1.x, spec), prep=prep,
                       labels=[f"projection of the free fit, round {k + 1}"], **kw)
        improved = r0.loglik > f0.loglik + tol
        f0 = _merge(f0, r0)
        if not improved:
            progress.report(f"round {k + 1}: the restart did not improve the speed-only fit by more than {tol:g}, "
                            "so the fits are final")
            break
        progress.report(f"round {k + 1}: the speed-only fit improved to {f0.loglik:.2f}; the free fit restarts from it")
        r1 = fit_model(panel, spec, "free", x0=null_to(f0.x, spec, "free"), prep=prep,
                       labels=[f"null embedding, round {k + 1}"], **kw)
        f1 = _merge(f1, r1)
    if not f1.failed:
        progress.report(f"both fits done in {progress.duration(time.time() - t0)}: speed-only {f0.loglik:.2f}, "
                        f"free {f1.loglik:.2f}, likelihood ratio {max(2 * (f1.loglik - f0.loglik), 0.0):.2f}; "
                        f"{f0.n_eval + f1.n_eval} evaluations")
    return f0, f1


def hessian(fit: Fit, panel: Panel, prep: Prepared | None = None, h: float = 1e-5,
            chunk: int | None = None) -> np.ndarray:
    """Observed information (Hessian of minus the log-likelihood) by central differences of the
    exact gradient. Raises if the model cannot be evaluated next to the fit."""
    prep = prep or prepare(panel, fit.spec.meas_var, fit.spec.n_groups)
    n = fit.layout.size
    H = np.zeros((n, n))
    beat, t0 = progress.Heartbeat(), time.time()
    progress.report(f"information matrix of the {KIND_NAMES.get(fit.kind, fit.kind)}: {n} coordinates, "
                    f"{2 * n} evaluations")
    for j in range(n):
        e = np.zeros(n)
        e[j] = h
        vp, gp = value_and_grad(fit.x + e, fit.layout, prep, chunk)
        vm, gm = value_and_grad(fit.x - e, fit.layout, prep, chunk)
        if vp >= 1e10 or vm >= 1e10:
            raise RuntimeError(f"likelihood not evaluable next to the fit (coordinate {j})")
        H[:, j] = (gp - gm) / (2 * h)
        if beat.due():
            spent = time.time() - t0
            progress.report(f"information matrix: {j + 1} of {n} coordinates in {progress.duration(spent)}, "
                            f"about {progress.duration(spent / (j + 1) * (n - j - 1))} left")
    progress.report(f"information matrix done in {progress.duration(time.time() - t0)}")
    return 0.5 * (H + H.T)


def covariance(H: np.ndarray) -> np.ndarray:
    """Inverse of the observed information, with a pseudo-inverse if it is singular."""
    return covariance_checked(H)[0]


def param_names(lay: Layout) -> list:
    """A name for every coordinate of the parameter vector, e.g. 'log_tau[3]'."""
    out = []
    for name, shape in lay.blocks:
        out += [f"{name}[{i}]" for i in range(int(np.prod(shape)))]
    return out


def covariance_checked(H: np.ndarray, max_condition: float = 1e10, names: list | None = None) -> tuple[np.ndarray, dict]:
    """Inverse of the observed information and a report: the smallest eigenvalue, the condition
    number, whether the information was positive definite and well conditioned, and the
    coordinates along any nearly flat direction (typically a variance component at zero, such as
    a load spread estimated at 0). Directions with non-positive information are dropped
    (pseudo-inverse), which understates uncertainty along them, so the report must travel with
    any interval computed from it."""
    Hs = 0.5 * (H + H.T)
    w, V = np.linalg.eigh(Hs)
    pos = w > 0
    cond = float(w.max() / w[pos].min()) if pos.any() else float("inf")
    info = {"min_eigenvalue": float(w.min()), "condition_number": cond,
            "positive_definite": bool(pos.all()), "well_conditioned": bool(pos.all() and cond < max_condition)}
    flat = []
    for k in np.where(w <= w.max() / max_condition)[0]:
        top = np.argsort(-np.abs(V[:, k]))[:3]
        flat.append({"eigenvalue": float(w[k]),
                     "coordinates": [(names[i] if names else int(i), round(float(V[i, k]), 3)) for i in top]})
    info["flat_directions"] = flat
    inv_w = np.where(pos, 1.0 / np.where(pos, w, 1.0), 0.0)
    return (V * inv_w) @ V.T, info

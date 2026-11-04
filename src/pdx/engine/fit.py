"""Maximum-likelihood fitting of the engine (protocol v0.5, §7.1).

Start values come from yearly least-squares fits per phenotype mapped to continuous time.
The free model is started both at its own start and at the null fit written in its parameters,
so the likelihood ratio of H-P cannot come out negative from a poor local optimum.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import linalg, optimize

from ..data import Panel
from .likelihood import Prepared, prepare, value_and_grad
from .model import Layout, ModelSpec, pack_chol, pack_drift, unpack_numpy


@dataclass
class Fit:
    kind: str
    spec: ModelSpec
    layout: Layout
    x: np.ndarray
    loglik: float
    success: bool
    message: str
    n_eval: int
    params: dict = field(default_factory=dict)
    start_logliks: list = field(default_factory=list)   # final log-likelihood from each start

    @property
    def n_params(self) -> int:
        return self.layout.size


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


# ---------------------------------------------------------------- fitting

def fit_model(panel: Panel, spec: ModelSpec, kind: str, x0: np.ndarray | None = None,
              extra_starts: list | None = None, prep: Prepared | None = None,
              maxiter: int = 3000, chunk: int | None = None, penalty=None, ftol: float = 1e-10,
              gtol: float = 1e-5, bounds=None, use_default_start: bool = True, maxcor: int = 50) -> Fit:
    """Fit one kind of model; the best of the starts by likelihood is kept.

    x0 replaces the default start (for example a warm start in a bootstrap replicate);
    extra_starts are tried as well.
    """
    lay = Layout(spec, kind)
    prep = prep or prepare(panel, spec.meas_var, spec.n_groups)
    starts = []
    if x0 is not None:
        starts.append(x0)
    elif use_default_start:
        starts.append(start_values(panel, spec, kind))
    starts += list(extra_starts or [])
    count = {"n": 0}

    def f(v):
        count["n"] += 1
        return value_and_grad(v, lay, prep, chunk, penalty)

    best, finals = None, []
    for s0 in starts:
        r = optimize.minimize(f, np.asarray(s0, float), jac=True, method="L-BFGS-B", bounds=bounds,
                              options={"maxiter": maxiter, "maxfun": 4 * maxiter, "ftol": ftol, "gtol": gtol,
                                       "maxcor": maxcor})
        finals.append(-float(r.fun))
        if best is None or r.fun < best.fun:
            best = r
    return Fit(kind, spec, lay, best.x, -float(best.fun), bool(best.success), str(best.message),
               count["n"], unpack_numpy(best.x, lay), finals)


def fit_pair(panel: Panel, spec: ModelSpec, prep: Prepared | None = None, x0_null: np.ndarray | None = None,
             own_free_start: bool = True, **kw) -> tuple[Fit, Fit]:
    """The null and free fits of H-P (§7.1).

    The free fit starts at the null solution written in its parameters, so its likelihood is at
    least the null's and the likelihood ratio is never negative; as the protocol specifies, the
    free model's own default start is tried as well (own_free_start; the engine gate measures
    whether it ever changes the ratio).
    """
    prep = prep or prepare(panel, spec.meas_var, spec.n_groups)
    f0 = fit_model(panel, spec, "null", x0=x0_null, prep=prep, **kw)
    f1 = fit_model(panel, spec, "free", x0=null_to(f0.x, spec, "free"), prep=prep,
                   extra_starts=[start_values(panel, spec, "free")] if own_free_start else None, **kw)
    return f0, f1


def hessian(fit: Fit, panel: Panel, prep: Prepared | None = None, h: float = 1e-5,
            chunk: int | None = None) -> np.ndarray:
    """Observed information (Hessian of minus the log-likelihood) by central differences of the
    exact gradient. Raises if the model cannot be evaluated next to the fit."""
    prep = prep or prepare(panel, fit.spec.meas_var, fit.spec.n_groups)
    n = fit.layout.size
    H = np.zeros((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = h
        vp, gp = value_and_grad(fit.x + e, fit.layout, prep, chunk)
        vm, gm = value_and_grad(fit.x - e, fit.layout, prep, chunk)
        if vp >= 1e10 or vm >= 1e10:
            raise RuntimeError(f"likelihood not evaluable next to the fit (coordinate {j})")
        H[:, j] = (gp - gm) / (2 * h)
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

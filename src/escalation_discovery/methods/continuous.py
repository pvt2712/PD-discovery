"""continuous — continuous-optimisation controls and the sortability baselines (PROTOCOL.md §5.1, controls C1-C4).

- NOTEARS-linear (Zheng et al., NeurIPS 2018): the reference algorithm (L1-penalised least squares under the trace-
  exponential acyclicity constraint, augmented Lagrangian, L-BFGS-B), written out here so that knowledge can enter as
  bounds (a forbidden entry is fixed at zero) and no extra framework is needed.
- DAGMA-linear (Bello, Aragam & Ravikumar, NeurIPS 2022): the `dagma` package 1.1.1.
- sortnregress and R²-sortnregress (Reisach, Seiler & Weichwald, NeurIPS 2021; Reisach et al., NeurIPS 2023): order
  the variables by marginal variance (raw units, by definition) or by R² on all others, then regress each on its
  predecessors with BIC-selected lasso. They need no structure learning at all: if they score like NOTEARS, the
  learner is reading the sortability of the data, not its causal structure.
All run on the shared imputed table; NOTEARS and DAGMA on z-scores (raw units in the units stress setting).
"""
import contextlib
import io

import numpy as np
import scipy.linalg as sla
import scipy.optimize as sopt

from ..graphs import from_weighted
from .base import Method, forbidden_matrix


def notears_linear(X, lambda1=0.05, w_threshold=0.3, max_iter=100, h_tol=1e-8, rho_max=1e16, forbidden=None):
    """NOTEARS with the L2 loss (reference implementation of Zheng et al. 2018, with optional zero bounds)."""
    n, d = X.shape

    def _adj(w):
        return (w[: d * d] - w[d * d:]).reshape(d, d)

    def _loss(W):
        R = X - X @ W
        return 0.5 / n * (R ** 2).sum(), -1.0 / n * X.T @ R

    def _h(W):
        Em = sla.expm(W * W)
        return np.trace(Em) - d, Em.T * W * 2

    def _func(w):
        W = _adj(w)
        loss, g_loss = _loss(W)
        h, g_h = _h(W)
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum()
        g_smooth = g_loss + (rho * h + alpha) * g_h
        return obj, np.concatenate((g_smooth + lambda1, -g_smooth + lambda1), axis=None)

    F = np.zeros((d, d), dtype=bool) if forbidden is None else forbidden
    bnds = [(0, 0) if (i == j or F[i, j]) else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
    w_est, rho, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf
    for _ in range(max_iter):
        w_new, h_new = None, None
        while rho < rho_max:
            sol = sopt.minimize(_func, w_est, method="L-BFGS-B", jac=True, bounds=bnds)
            w_new = sol.x
            h_new, _ = _h(_adj(w_new))
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        w_est, h = w_new, h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break
    W = _adj(w_est)
    W[np.abs(W) < w_threshold] = 0
    return W


class NOTEARS(Method):
    name = "notears"
    family = "continuous"
    output = "DAG"
    accepts_knowledge = True          # as zero bounds; run without knowledge as the design's control
    missing = "impute"
    scale_sensitive = True
    version = "reference algorithm (Zheng et al. 2018), this repository"

    def __init__(self, use_knowledge=False, **cfg):
        super().__init__(**cfg)
        self.use_knowledge = use_knowledge
        self.name = "notears_tiered" if use_knowledge else "notears"

    def defaults(self):
        return dict(lambda1=0.05, w_threshold=0.3)

    def grid(self):
        return [dict(lambda1=l, w_threshold=t) for l in (0.01, 0.05, 0.1) for t in (0.1, 0.3)]

    def _fit(self, data):
        X = data.complete_input(self.scale_sensitive)
        F = forbidden_matrix(data) if (self.use_knowledge and data.knowledge) else None
        W = notears_linear(X, self.cfg["lambda1"], self.cfg["w_threshold"], forbidden=F)
        if not data.standardize:                      # unit-free threshold on raw data: standardised weights
            sd = X.std(0)
            W = W * sd[:, None] / np.where(sd[None, :] > 0, sd[None, :], 1)
        return from_weighted(W, data.names, 0.0)


class DAGMA(Method):
    name = "dagma"
    family = "continuous"
    output = "DAG"
    accepts_knowledge = False
    missing = "impute"
    scale_sensitive = True
    version = "dagma 1.1.1"

    def defaults(self):
        return dict(lambda1=0.02, w_threshold=0.3)

    def grid(self):
        return [dict(lambda1=l, w_threshold=t) for l in (0.01, 0.02, 0.05) for t in (0.1, 0.3)]

    def _fit(self, data):
        from dagma.linear import DagmaLinear
        X = data.complete_input(self.scale_sensitive)
        with contextlib.redirect_stderr(io.StringIO()):          # dagma draws a progress bar on stderr
            W = DagmaLinear(loss_type="l2", verbose=False).fit(X, lambda1=self.cfg["lambda1"], w_threshold=self.cfg["w_threshold"])
        return from_weighted(W, data.names, 0.0)


def _r2(X):
    out = np.empty(X.shape[1])
    for j in range(X.shape[1]):
        A = np.column_stack([np.delete(X, j, axis=1), np.ones(len(X))])
        coef, *_ = np.linalg.lstsq(A, X[:, j], rcond=None)
        out[j] = 1 - np.var(X[:, j] - A @ coef) / max(np.var(X[:, j]), 1e-12)
    return out


def sortnregress(X, key):
    """Reisach et al.'s baseline: regress each variable on all variables earlier in the order, adaptive-lasso style."""
    from sklearn.linear_model import LassoLarsIC, LinearRegression
    d = X.shape[1]
    W = np.zeros((d, d))
    order = np.argsort(key, kind="stable")
    for k in range(1, d):
        cov, target = order[:k], order[k]
        lr = LinearRegression().fit(X[:, cov], X[:, target])
        weight = np.abs(lr.coef_)
        ll = LassoLarsIC(criterion="bic").fit(X[:, cov] * weight, X[:, target])
        W[cov, target] = ll.coef_ * weight
    return W


class SortNRegress(Method):
    family = "continuous"
    output = "DAG"
    accepts_knowledge = False
    missing = "impute"
    version = "Reisach et al. 2021/2023, this repository"

    def __init__(self, key="var", **cfg):
        super().__init__(**cfg)
        self.key = key
        self.name = "sortnregress" if key == "var" else "r2sortnregress"
        self.scale_sensitive = key == "var"

    def _fit(self, data):
        X = data.imputed()                     # var-sortnregress reads raw units by definition
        if self.key == "r2":
            sd = X.std(0)
            X = (X - X.mean(0)) / np.where(sd > 0, sd, 1)
            key = _r2(X)
        else:
            key = X.var(0)
        W = sortnregress(X, key)
        sd = X.std(0)
        W = W * sd[:, None] / np.where(sd[None, :] > 0, sd[None, :], 1)     # standardised weights
        return from_weighted(W, data.names, 1e-8)


def varsortability(X, names, tiers, stat="var"):
    """Share of (earlier, later) cross-tier pairs whose earlier variable has the smaller variance (or R²): 0.5 = no
    cue, 1 = the cue encodes the time order (ppmi-causal/src/ppmi_causal/ml_checks.py:sortability)."""
    tof = {n: k for k, t in enumerate(tiers) for n in t}
    s = np.round(X.var(0) if stat == "var" else _r2((X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1)), 8)
    pairs = [(i, j) for i in range(len(names)) for j in range(len(names))
             if names[i] in tof and names[j] in tof and tof[names[i]] < tof[names[j]]]
    return float(np.mean([1.0 if s[i] < s[j] else 0.5 if s[i] == s[j] else 0.0 for i, j in pairs])) if pairs else np.nan

"""Multiplicity tools for the hierarchy of protocol v0.5, §2.6.

H-P at 5%; then Holm across the confirmatory links (gatekeeping); simultaneous intervals for
"similar" and "structurally different" labels; an intersection-union test for the overall
"speed and load only" verdict; Benjamini-Yekutieli control for the exploratory families.
"""
from __future__ import annotations

import numpy as np


def holm(p, alpha: float = 0.05) -> np.ndarray:
    """Rejections under Holm's step-down procedure."""
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    rej = np.zeros(m, bool)
    for rank, k in enumerate(order):
        if p[k] <= alpha / (m - rank):
            rej[k] = True
        else:
            break
    return rej


def holm_adjusted(p) -> np.ndarray:
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for rank, k in enumerate(order):
        run = max(run, min(1.0, (m - rank) * p[k]))
        adj[k] = run
    return adj


def bh(p, alpha: float = 0.05) -> np.ndarray:
    """Rejections under Benjamini-Hochberg."""
    p = np.asarray(p, float)
    m = len(p)
    if m == 0:
        return np.zeros(0, bool)
    order = np.argsort(p)
    thresh = alpha * np.arange(1, m + 1) / m
    ok = p[order] <= thresh
    rej = np.zeros(m, bool)
    if ok.any():
        kmax = np.max(np.where(ok)[0])
        rej[order[:kmax + 1]] = True
    return rej


def by(p, alpha: float = 0.05) -> np.ndarray:
    """Rejections under Benjamini-Yekutieli (valid under any dependence)."""
    m = len(p)
    c = np.sum(1.0 / np.arange(1, m + 1)) if m else 1.0
    return bh(p, alpha / c)


def simultaneous_intervals(est: np.ndarray, draws: np.ndarray, level: float = 0.90):
    """Simultaneous intervals from Monte Carlo draws of the estimates (max-|t| over the family)."""
    est = np.asarray(est, float)
    sd = draws.std(0, ddof=1)
    sd = np.where(sd > 0, sd, np.inf)
    tmax = np.max(np.abs(draws - draws.mean(0)) / sd, axis=1)
    c = np.quantile(tmax, level)
    return est - c * sd, est + c * sd


def iut_similar(lower, upper, eps: float) -> bool:
    """The overall verdict "speed and load only": every interval inside (-eps, eps)."""
    lower, upper = np.asarray(lower), np.asarray(upper)
    return bool(np.all((lower > -eps) & (upper < eps)))

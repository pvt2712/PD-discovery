"""Simulation from a fitted engine: the conditional parametric bootstrap of H-P (§7.3.1).

Each participant keeps their phenotype, baseline covariates, landmark scores, visit dates,
dose path, examination states and pattern of missing values. Their speed, load and true
landmark state are drawn from the fitted model given those, and the later visits are simulated
exactly over the actual intervals.
"""
from __future__ import annotations

import numpy as np

from ..data import Panel
from .fit import Fit
from .likelihood import prepare
from .transitions import transition_np


def _treat(par: dict, spec, dose, expo, on, g):
    h = np.zeros((len(dose), spec.d))
    if not spec.n_treat:
        return h
    fd = dose / (spec.ed50 + dose) if spec.treatment == "emax" else dose
    fe = expo / (spec.ec50_ldr + expo) if spec.treatment == "emax" else expo
    tm = np.where(spec.treat_mask)[0]
    state = np.repeat(np.asarray(on, float)[:, None], len(tm), 1)
    state[:, spec.on_when_treated[tm]] = (np.asarray(dose) > 0)[:, None]      # daily function (D24)
    h[:, tm] = -(par["kappa"][g][None, :] * fd[:, None] * state + par["kappa_ldr"][g][None, :] * fe[:, None])
    return h


def simulate_from_fit(fit: Fit, panel: Panel, rng: np.random.Generator) -> Panel:
    spec, par = fit.spec, fit.params
    d = spec.d
    prep = prepare(panel, spec.meas_var, spec.n_groups)
    mu1, S1, y1c = prep.mu1.numpy(), prep.S1.numpy(), prep.y1c.numpy()
    R = np.diag(spec.meas_var)
    sdR = np.sqrt(spec.meas_var)
    new_y = []
    for i in range(panel.n):
        cache = {}
        g = int(panel.group[i])
        yi, ti = panel.y[i], panel.t[i]
        h = _treat(par, spec, panel.dose[i], panel.exposure[i], panel.on[i], g)
        u = rng.normal()
        eta = par["log_speed"][g] + panel.z[i] @ par["beta"] + y1c[i] @ par["theta"]
        s = float(np.exp(eta + par["sigma"][g] * u))
        mu_l = par["gam_y"][g] * y1c[i] + par["gam_z"][g] @ panel.z[i] + par["lam"][g] * u
        l = mu_l + par["tau"][g] * rng.normal(size=d)
        q_era = np.exp(par["omega"] * panel.z[i, spec.noise_era_col]) if spec.noise_era_col is not None else 1.0
        o = ~np.isnan(yi[0])
        m, P = mu1[g].copy(), S1[g].copy()
        if o.any():
            Ho = np.eye(d)[o]
            Kg = P @ Ho.T @ np.linalg.inv(Ho @ P @ Ho.T + R[np.ix_(o, o)])
            m = m + Kg @ (yi[0, o] - m[o] - h[0, o])
            P = P - Kg @ Ho @ P
        w, V = np.linalg.eigh(0.5 * (P + P.T))
        X = m + V @ (np.sqrt(np.maximum(w, 0.0)) * rng.normal(size=d))
        out = np.full_like(yi, np.nan)
        out[0] = yi[0]
        A, Q, b = par["A"][g], par["Q"][g], par["b"][g]
        for k in range(1, len(ti)):
            dt = ti[k] - ti[k - 1]
            key = round(dt, 12)
            if key not in cache:
                cache[key] = transition_np(A, q_era * Q, s, dt)
            E, M, S = cache[key]
            ws, Vs = np.linalg.eigh(S)
            X = E @ X + M @ (b + l) + Vs @ (np.sqrt(np.maximum(ws, 0.0)) * rng.normal(size=d))
            yk = X + h[k] + sdR * rng.normal(size=d)
            miss = np.isnan(yi[k])
            yk[miss] = np.nan
            out[k] = yk
        new_y.append(out)
    return panel.with_y(new_y)

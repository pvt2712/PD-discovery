"""An independent, slow implementation of the engine's log-likelihood, for checks only.

It loops over participants, quadrature nodes and visits, uses scipy's matrix exponential with
Van Loan's blocks (not the engine's transition code), and selects the observed components
explicitly instead of masking. The engine gate (§16) requires
the torch likelihood and this one to agree.
"""
from __future__ import annotations

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.linalg import expm, solve_continuous_lyapunov
from scipy.special import logsumexp

from ..data import Panel
from .model import Layout, unpack_numpy


def _van_loan(A: np.ndarray, Q: np.ndarray, c: float):
    """E and M over an interval of length c from a block matrix exponential (Van Loan 1978), and S
    from the stationary covariance (S = Sigma - E Sigma E', scipy's Lyapunov solver) when A is
    stable, otherwise from Van Loan's block. Written independently of engine/transitions.py."""
    d = A.shape[0]
    B1 = np.zeros((2 * d, 2 * d))
    B1[:d, :d], B1[:d, d:] = c * A, c * np.eye(d)
    F1 = expm(B1)
    E = F1[:d, :d]
    if np.max(np.linalg.eigvals(A).real) < -1e-6:
        Sig = solve_continuous_lyapunov(A, -Q)
        S = Sig - E @ Sig @ E.T
    else:
        B2 = np.zeros((2 * d, 2 * d))
        B2[:d, :d], B2[:d, d:], B2[d:, d:] = -c * A, c * Q, c * A.T
        F2 = expm(B2)
        S = F2[d:, d:].T @ F2[:d, d:]
    return E, F1[:d, d:], 0.5 * (S + S.T)


def reference_loglik(x: np.ndarray, lay: Layout, panel: Panel, mu1: np.ndarray, S1: np.ndarray,
                     y1c: np.ndarray) -> float:
    spec = lay.spec
    d, K = spec.d, spec.n_quad
    par = unpack_numpy(x, lay)
    xs, ws = hermgauss(K)
    U, logw = np.sqrt(2.0) * xs, np.log(ws / np.sqrt(np.pi))
    R = np.diag(spec.meas_var)
    tm = np.where(spec.treat_mask)[0]
    total = 0.0
    for i in range(panel.n):
        g = panel.group[i]
        A, Q, b = par["A"][g], par["Q"][g], par["b"][g]
        eta = par["log_speed"][g] + panel.z[i] @ par["beta"] + y1c[i] @ par["theta"]
        mu_l = par["gam_y"][g] * y1c[i] + par["gam_z"][g] @ panel.z[i]
        q_era = np.exp(par["omega"] * panel.z[i, spec.noise_era_col]) if spec.noise_era_col is not None else 1.0
        yi, ti = panel.y[i], panel.t[i]
        hi = np.zeros_like(yi)
        if spec.n_treat:
            dose, expo, on = panel.dose[i], panel.exposure[i], panel.on[i]
            fd = dose / (spec.ed50 + dose) if spec.treatment == "emax" else dose
            fe = expo / (spec.ec50_ldr + expo) if spec.treatment == "emax" else expo
            hi[:, tm] = -(par["kappa"][g][None, :] * (fd * on)[:, None] + par["kappa_ldr"][g][None, :] * fe[:, None])
        # landmark posterior from the observed components only
        o1 = ~np.isnan(yi[0])
        m1, P1 = mu1[g].copy(), S1[g].copy()
        if o1.any():
            Ho = np.eye(d)[o1]
            Sv = Ho @ P1 @ Ho.T + R[np.ix_(o1, o1)]
            Kg = P1 @ Ho.T @ np.linalg.inv(Sv)
            m1 = m1 + Kg @ (yi[0, o1] - m1[o1] - hi[0, o1])
            P1 = P1 - Kg @ Ho @ P1
        lls = np.zeros(K)
        for k in range(K):
            s = np.exp(eta + par["sigma"][g] * U[k])
            m = np.concatenate([m1, mu_l + par["lam"][g] * U[k]])
            P = np.zeros((2 * d, 2 * d))
            P[:d, :d], P[d:, d:] = P1, np.diag(par["tau"][g] ** 2)
            ll = 0.0
            for t in range(1, len(ti)):
                E, M, S = _van_loan(A, q_era * Q, s * (ti[t] - ti[t - 1]))
                F = np.eye(2 * d)
                F[:d, :d], F[:d, d:] = E, M
                c = np.concatenate([M @ b, np.zeros(d)])
                m = F @ m + c
                P = F @ P @ F.T
                P[:d, :d] += S
                o = ~np.isnan(yi[t])
                if not o.any():
                    continue
                H = np.hstack([np.eye(d)[o], np.zeros((o.sum(), d))])
                v = yi[t, o] - H @ m - hi[t, o]
                Sv = H @ P @ H.T + R[np.ix_(o, o)]
                sign, logdet = np.linalg.slogdet(Sv)
                Si = np.linalg.inv(Sv)
                ll += -0.5 * (o.sum() * np.log(2 * np.pi) + logdet + v @ Si @ v)
                Kg = P @ H.T @ Si
                m = m + Kg @ v
                P = P - Kg @ H @ P
            lls[k] = ll
        total += logsumexp(lls + logw)
    return float(total)

"""Toy simulation (synthetic data only, no PPMI): speed varies between patients
within each phenotype. Does the group-level speed-only test still hold, and does a
person-level random-speed model?

Truth: dX = s_i (A X + b) dt + sqrt(s_i) G dW, sampled yearly; log s_i = log s_g + 0.4 u_i,
s_TD = 1, s_PIGD = 2. Scenario "speed": shared A. Scenario "speed+mech": PIGD has an
extra link Axial -> Independence.

Tests (each against chi-square with p^2 - 1 = 8 df):
  pooled: group-level matrix-power test B_PIGD = B_TD^s (free intercepts and covariances)
  random: person-level random speed; H1 phenotype acts on log-speed only (A shared,
          load b free per phenotype) vs H3 phenotype-specific A (load free, same speed spread)
"""
import json
import sys

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy import linalg, optimize, stats

sys.path.insert(0, ".")
import toy_speed_vs_pathway as T  # noqa: E402  (reuses mpow, logdet_resid, centred)

P = 3  # Axial, Function, Independence
GH_X, GH_W = hermgauss(12)
U_NODES = np.sqrt(2) * GH_X
U_W = GH_W / np.sqrt(np.pi)


def drift(extra=False, fast=False):
    """'slow' yearly dynamics resemble persistent clinical scales; 'fast' ones (decay 0.3,
    effects 0.4 per year) make second-order terms of the matrix exponential large."""
    d, e1, e2, ex = (0.30, 0.40, 0.40, 0.20) if fast else (0.20, 0.20, 0.25, 0.12)
    A = -d * np.eye(P)
    A[1, 0] = e1  # Axial -> Function
    A[2, 1] = e2  # Function -> Independence
    if extra:
        A[2, 0] = ex  # PIGD-only Axial -> Independence
    return A


Q = 0.10 * np.eye(P)


def transition(A, b, Qm, s):
    """Exact yearly transition for speed s: X' = E X + a + w, w ~ N(0, S)."""
    M1 = np.zeros((P + 1, P + 1))
    M1[:P, :P] = A
    M1[:P, P] = b
    F1 = linalg.expm(M1 * s)
    E, a = F1[:P, :P], F1[:P, P]
    M2 = np.zeros((2 * P, 2 * P))
    M2[:P, :P] = -A
    M2[:P, P:] = Qm
    M2[P:, P:] = A.T
    F2 = linalg.expm(M2 * s)
    S = F2[P:, P:].T @ F2[:P, P:]
    return E, a, (S + S.T) / 2


def simulate(rng, n, Tn, A, s_med, sd=0.4):
    b = -A @ np.ones(P)
    s = s_med * np.exp(sd * rng.normal(size=n))
    X = np.zeros((n, Tn + 1, P))
    X[:, 0] = rng.normal(0, 0.3, size=(n, P))
    for i in range(n):
        E, a, S = transition(A, b, Q, s[i])
        L = np.linalg.cholesky(S)
        for t in range(Tn):
            X[i, t + 1] = E @ X[i, t] + a + L @ rng.normal(size=P)
    return X


def unpack(theta, model):
    k = 0
    if model == "H1":
        A1 = theta[k:k + P * P].reshape(P, P); k += P * P
        A2 = A1
    else:
        A1 = theta[k:k + P * P].reshape(P, P); k += P * P
        A2 = theta[k:k + P * P].reshape(P, P); k += P * P
    b1 = theta[k:k + P]; k += P
    b2 = theta[k:k + P]; k += P
    Qs = []
    for _ in range(2):  # noise free per phenotype in both models, so the test concerns drift only
        Lq = np.zeros((P, P))
        Lq[np.tril_indices(P)] = theta[k:k + P * (P + 1) // 2]; k += P * (P + 1) // 2
        Lq[np.diag_indices(P)] = np.exp(Lq[np.diag_indices(P)])
        Qs.append(Lq @ Lq.T)
    gamma = theta[k] if model == "H1" else 0.0
    k += 1 if model == "H1" else 0
    sigma = np.exp(theta[k])
    return A1, A2, b1, b2, Qs[0], Qs[1], gamma, sigma


def group_ll(X, A, b, Qm, logs_med, sigma):
    """Marginal log-likelihood of one phenotype's patients, integrating log-speed."""
    n, Tn1, _ = X.shape
    Xt, Yt = X[:, :-1], X[:, 1:]
    lls = np.zeros((n, len(U_NODES)))
    for k, u in enumerate(U_NODES):
        s = np.exp(logs_med + sigma * u)
        E, a, S = transition(A, b, Qm, s)
        sign, ld = np.linalg.slogdet(S)
        if sign <= 0 or not np.all(np.isfinite(E)):
            return -1e12
        Si = np.linalg.inv(S)
        R = Yt - Xt @ E.T - a
        q = np.einsum("ntj,jk,ntk->n", R, Si, R)
        lls[:, k] = -0.5 * ((Tn1 - 1) * (P * np.log(2 * np.pi) + ld) + q)
    mx = lls.max(1, keepdims=True)
    return float(np.sum(mx[:, 0] + np.log(np.exp(lls - mx) @ U_W)))


def negll(theta, model, X1, X2):
    try:
        A1, A2, b1, b2, Q1, Q2, gamma, sigma = unpack(theta, model)
        v = group_ll(X1, A1, b1, Q1, 0.0, sigma) + group_ll(X2, A2, b2, Q2, gamma, sigma)
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        return 1e12
    return -v if np.isfinite(v) else 1e12


def start_values(X1, X2, model):
    def ols_B(X):
        Xt = X[:, :-1].reshape(-1, P)
        Yt = X[:, 1:].reshape(-1, P)
        B, _ = T.ols(Xt, Yt)
        Z = np.column_stack([np.ones(len(Xt)), Xt])
        coef, *_ = np.linalg.lstsq(Z, Yt, rcond=None)
        R = Yt - Z @ coef
        return B, coef[0], np.cov(R.T)
    B1, a1, S1 = ols_B(X1)
    B2, a2, S2 = ols_B(X2)
    A1 = np.real(linalg.logm(B1))
    A2 = np.real(linalg.logm(B2))
    b1 = A1 @ np.linalg.solve(B1 - np.eye(P), a1)
    b2 = A1 @ np.linalg.solve(B2 - np.eye(P), a2) if model == "H1" else A2 @ np.linalg.solve(B2 - np.eye(P), a2)
    lqs = []
    for S in (S1, S2):
        L = np.linalg.cholesky(S)
        lq = L.copy()
        lq[np.diag_indices(P)] = np.log(np.diag(L))
        lqs.append(lq[np.tril_indices(P)])
    parts = [A1.ravel()] + ([] if model == "H1" else [A2.ravel()]) + [b1, b2] + lqs
    if model == "H1":
        parts.append([np.log(2.0)])
    parts.append([np.log(0.3)])
    return np.concatenate([np.ravel(p) for p in parts])


def fit(X1, X2, model, th0=None):
    th0 = start_values(X1, X2, model) if th0 is None else th0
    res = optimize.minimize(negll, th0, args=(model, X1, X2), method="L-BFGS-B",
                            options={"maxiter": 3000, "maxfun": 300000})
    return res


def h1_to_h3(th):
    """Map an H1 solution to the equivalent H3 parameters (A2 = e^g A, b2 = e^g b2, Q2 = e^g Q2)."""
    A, A2, b1, b2, Q1, Q2, g, sigma = unpack(th, "H1")
    eg = np.exp(g)
    k = P * P
    parts = [A.ravel(), (eg * A).ravel(), b1, eg * b2]
    for Qm in (Q1, eg * Q2):
        L = np.linalg.cholesky(Qm)
        lq = L.copy()
        lq[np.diag_indices(P)] = np.log(np.diag(L))
        parts.append(lq[np.tril_indices(P)])
    parts.append([np.log(sigma)])
    return np.concatenate([np.ravel(x) for x in parts])


def pooled_test(X1, X2):
    g1 = T.centred(X1[:, :-1].reshape(-1, P), X1[:, 1:].reshape(-1, P))
    g2 = T.centred(X2[:, :-1].reshape(-1, P), X2[:, 1:].reshape(-1, P))
    (Xa, Ya), (Xb, Yb) = g1, g2
    n1, n2 = len(Xa), len(Xb)
    B1o, _ = T.ols(Xa, Ya)
    B2o, _ = T.ols(Xb, Yb)
    free = 0.5 * (n1 * T.logdet_resid(Xa, Ya, B1o) + n2 * T.logdet_resid(Xb, Yb, B2o))

    def nll(th):
        B1 = th[:-1].reshape(P, P)
        B2 = T.mpow(B1, np.exp(th[-1]))
        if B2 is None:
            return 1e12
        v = 0.5 * (n1 * T.logdet_resid(Xa, Ya, B1) + n2 * T.logdet_resid(Xb, Yb, B2))
        return v if np.isfinite(v) else 1e12

    grid = np.linspace(np.log(0.5), np.log(4), 41)
    g0 = grid[np.argmin([nll(np.append(B1o.ravel(), g)) for g in grid])]
    r = optimize.minimize(nll, np.append(B1o.ravel(), g0), method="L-BFGS-B")
    lrt = 2 * (r.fun - free)
    return float(lrt), float(np.exp(r.x[-1]))


def one_rep(rng, scenario, fast, n1=400, n2=200, Tn=5):
    X1 = simulate(rng, n1, Tn, drift(False, fast), 1.0)
    X2 = simulate(rng, n2, Tn, drift(scenario == "speed+mech", fast), 2.0)
    r1 = fit(X1, X2, "H1")
    r3a = fit(X1, X2, "H3")
    r3b = fit(X1, X2, "H3", h1_to_h3(r1.x))  # start from the nested solution, so H3 >= H1
    r3 = r3a if r3a.fun < r3b.fun else r3b
    lrt_rand = 2 * (r1.fun - r3.fun)
    df = P * P - 1
    gamma = r1.x[P * P + 2 * P + P * (P + 1)]
    lrt_pool, s_pool = pooled_test(X1, X2)
    return {"lrt_random": float(lrt_rand), "p_random": float(stats.chi2.sf(max(lrt_rand, 0), df)),
            "s_random": float(np.exp(gamma)), "lrt_pooled": lrt_pool,
            "p_pooled": float(stats.chi2.sf(max(lrt_pool, 0), df)), "s_pooled": s_pool,
            "conv": bool(r1.success and r3.success), "lrt_negative": bool(lrt_rand < -1e-3)}


def main(R, seed):
    out = {}
    cells = [(fast, sc) for fast in (False, True) for sc in ("speed", "speed+mech")]
    for fast, scenario in cells:
        rng = np.random.default_rng(seed + 10 * fast + (0 if scenario == "speed" else 1))
        Rc = R if scenario == "speed" else max(R // 2, 1)
        reps = [one_rep(rng, scenario, fast) for _ in range(Rc)]
        R_ = Rc
        rr = np.array([r["p_random"] < 0.05 for r in reps])
        rp = np.array([r["p_pooled"] < 0.05 for r in reps])
        key = ("fast_" if fast else "slow_") + scenario
        out[key] = {
            "R": R_,
            "random_speed_reject": float(rr.mean()), "random_mcse": float(np.sqrt(rr.mean() * (1 - rr.mean()) / R_)),
            "pooled_reject": float(rp.mean()), "pooled_mcse": float(np.sqrt(rp.mean() * (1 - rp.mean()) / R_)),
            "lrt_random_mean": float(np.mean([r["lrt_random"] for r in reps])),
            "lrt_pooled_mean": float(np.mean([r["lrt_pooled"] for r in reps])),
            "speed_ratio_random_median": float(np.median([r["s_random"] for r in reps])),
            "speed_ratio_pooled_median": float(np.median([r["s_pooled"] for r in reps])),
            "converged": float(np.mean([r["conv"] for r in reps])),
            "lrt_negative": float(np.mean([r["lrt_negative"] for r in reps])),
        }
        print(key, json.dumps(out[key]), flush=True)
    with open("toy_random_speed_results.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 100, 31)

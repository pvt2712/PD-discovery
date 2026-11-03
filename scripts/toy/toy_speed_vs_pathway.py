"""Toy simulation (synthetic data only, no PPMI): does a pure speed difference
look like a pathway difference in annual discrete-time analyses?

Shared continuous-time system dX = s(AX + c)dt + sqrt(s) G dW, sampled yearly.
Group 1 ("TD") s = 1; group 2 ("PIGD") s = 2.
Scenario "speed": identical A, only speed differs.
Scenario "speed+mech": PIGD also has one extra direct link Axial -> Independence.

Per replicate:
  1. per-group OLS of X(t+1) on X(t) (exact discrete model is linear),
  2. naive split: link significant in one group only,
  3. link-wise difference tests (Welch-type Wald, BH 5%),
  4. speed-only test: B2 = B1^s (matrix power) vs free B1, B2, likelihood ratio.
"""
import json
import sys

import numpy as np
from scipy import linalg, optimize, stats

NAMES = ["Brady", "Axial", "Cognition", "Mood", "Function", "Independence"]
P = len(NAMES)


def drift(extra=False):
    A = -0.20 * np.eye(P)
    A[1, 0] = 0.15  # Brady -> Axial
    A[4, 0] = 0.10  # Brady -> Function
    A[4, 1] = 0.20  # Axial -> Function
    A[5, 4] = 0.25  # Function -> Independence
    A[5, 2] = 0.15  # Cognition -> Independence
    A[2, 3] = 0.10  # Mood -> Cognition
    A[3, 1] = 0.08  # Axial -> Mood
    if extra:
        A[5, 1] = 0.12  # PIGD-only direct Axial -> Independence
    return A


Q = 0.10 * np.eye(P)
M_EQ = np.ones(P)  # equilibrium (severe) mean


def discrete(A, s):
    """Exact annual transition for speed s: X' = B X + a + w, w ~ N(0, S)."""
    B = linalg.expm(s * A)
    c = -A @ M_EQ
    a = (B - np.eye(P)) @ np.linalg.solve(A, c)
    sig_inf = linalg.solve_continuous_lyapunov(A, -Q)
    S = sig_inf - B @ sig_inf @ B.T
    return B, a, (S + S.T) / 2


def simulate(rng, n, T, A, s):
    B, a, S = discrete(A, s)
    L = np.linalg.cholesky(S)
    X = np.zeros((n, T + 1, P))
    X[:, 0] = rng.normal(0.0, 0.3, size=(n, P))
    for t in range(T):
        X[:, t + 1] = X[:, t] @ B.T + a + rng.normal(size=(n, P)) @ L.T
    Xt = X[:, :-1].reshape(-1, P)
    Yt = X[:, 1:].reshape(-1, P)
    return Xt, Yt


def ols(X, Y):
    n = X.shape[0]
    Z = np.column_stack([np.ones(n), X])
    coef, *_ = np.linalg.lstsq(Z, Y, rcond=None)
    R = Y - Z @ coef
    sigma2 = (R ** 2).sum(0) / (n - Z.shape[1])
    ZtZi = np.linalg.inv(Z.T @ Z)
    se = np.sqrt(np.outer(np.diag(ZtZi)[1:], sigma2))  # (P predictors, P outcomes)
    B = coef[1:].T  # B[i, j]: effect of X_j(t) on X_i(t+1)
    return B, se.T


def centred(X, Y):
    return X - X.mean(0), Y - Y.mean(0)


def logdet_resid(Xc, Yc, B):
    R = Yc - Xc @ B.T
    S = R.T @ R / Xc.shape[0]
    sign, ld = np.linalg.slogdet(S)
    return ld if sign > 0 else np.inf


def mpow(B, s):
    w, V = np.linalg.eig(B)
    if np.any((np.abs(w.imag) < 1e-12) & (w.real <= 0)):
        return None
    return (V @ np.diag(w.astype(complex) ** s) @ np.linalg.inv(V)).real


def speed_fit(g1, g2):
    (X1, Y1), (X2, Y2) = g1, g2
    n1, n2 = X1.shape[0], X2.shape[0]
    B1o, _ = ols(X1, Y1)
    B2o, _ = ols(X2, Y2)
    nll_free = 0.5 * (n1 * logdet_resid(X1, Y1, B1o) + n2 * logdet_resid(X2, Y2, B2o))

    def nll(theta):
        B1 = theta[:-1].reshape(P, P)
        B2 = mpow(B1, np.exp(theta[-1]))
        if B2 is None:
            return 1e12
        v = 0.5 * (n1 * logdet_resid(X1, Y1, B1) + n2 * logdet_resid(X2, Y2, B2))
        return v if np.isfinite(v) else 1e12

    # start: s from a 1-D search with B1 fixed at its OLS value
    grid = np.linspace(np.log(0.5), np.log(4.0), 41)
    ls0 = grid[np.argmin([nll(np.append(B1o.ravel(), g)) for g in grid])]
    res = optimize.minimize(nll, np.append(B1o.ravel(), ls0), method="L-BFGS-B",
                            options={"maxiter": 2000, "maxfun": 200000})
    lrt = 2 * (res.fun - nll_free)
    df = P * P - 1
    return {"s_hat": float(np.exp(res.x[-1])), "lrt": float(lrt),
            "p": float(stats.chi2.sf(max(lrt, 0.0), df)), "B1": res.x[:-1].reshape(P, P),
            "B1o": B1o, "B2o": B2o}


def bh(p, q=0.05):
    p = np.asarray(p)
    m = len(p)
    order = np.argsort(p)
    thresh = q * np.arange(1, m + 1) / m
    passed = p[order] <= thresh
    k = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
    out = np.zeros(m, bool)
    out[order[:k]] = True
    return out


OFF = [(i, j) for i in range(P) for j in range(P) if i != j]


def one_rep(rng, scenario, n1=800, n2=250, T=5):
    A1 = drift(False)
    A2 = drift(scenario == "speed+mech")
    X1, Y1 = simulate(rng, n1, T, A1, 1.0)
    X2, Y2 = simulate(rng, n2, T, A2, 2.0)
    B1, se1 = ols(X1, Y1)
    B2, se2 = ols(X2, Y2)
    p1 = np.array([2 * stats.norm.sf(abs(B1[i, j] / se1[i, j])) for i, j in OFF])
    p2 = np.array([2 * stats.norm.sf(abs(B2[i, j] / se2[i, j])) for i, j in OFF])
    sig1, sig2 = p1 < 0.05, p2 < 0.05
    naive_struct = int(np.sum(sig1 != sig2))
    d = np.array([B2[i, j] - B1[i, j] for i, j in OFF])
    dse = np.array([np.hypot(se1[i, j], se2[i, j]) for i, j in OFF])
    pd = 2 * stats.norm.sf(np.abs(d / dse))
    diff = bh(pd)
    stronger = int(np.sum(diff & sig1 & sig2 & (np.sign([B1[i, j] for i, j in OFF]) == np.sign([B2[i, j] for i, j in OFF]))))
    struct_like = int(np.sum(diff & (sig1 != sig2)))
    fit = speed_fit(centred(X1, Y1), centred(X2, Y2))
    # speed-adjusted deviations: free PIGD estimate minus TD estimate raised to s_hat
    pw = mpow(fit["B1o"], fit["s_hat"])
    dev = np.array([(fit["B2o"][i, j] - pw[i, j]) / se2[i, j] for i, j in OFF])
    top = OFF[int(np.argmax(np.abs(dev)))]
    return {
        "naive_struct": naive_struct, "diff_bh": int(diff.sum()), "stronger": stronger,
        "struct_like": struct_like, "s_hat": fit["s_hat"], "speed_p": fit["p"],
        "top_dev": f"{NAMES[top[1]]}->{NAMES[top[0]]}", "top_dev_z": float(np.max(np.abs(dev))),
        "B1": B1.tolist(), "B2": B2.tolist(),
    }


def main(R, seed):
    out = {}
    for scenario in ["speed", "speed+mech"]:
        rng = np.random.default_rng(seed + (0 if scenario == "speed" else 1))
        reps = [one_rep(rng, scenario) for _ in range(R)]
        B1m = np.mean([r["B1"] for r in reps], 0)
        B2m = np.mean([r["B2"] for r in reps], 0)
        links = {}
        for (i, j) in [(1, 0), (4, 1), (5, 4), (5, 2), (5, 1), (5, 0), (2, 1), (5, 3)]:
            links[f"{NAMES[j]}->{NAMES[i]}"] = [round(B1m[i, j], 3), round(B2m[i, j], 3)]
        rej = np.array([r["speed_p"] < 0.05 for r in reps])
        out[scenario] = {
            "R": R,
            "mean_links_TD_PIGD": links,
            "true_B_TD": discrete(drift(False), 1.0)[0].round(3).tolist(),
            "naive_struct_mean": float(np.mean([r["naive_struct"] for r in reps])),
            "diff_bh_mean": float(np.mean([r["diff_bh"] for r in reps])),
            "diff_bh_any": float(np.mean([r["diff_bh"] > 0 for r in reps])),
            "stronger_mean": float(np.mean([r["stronger"] for r in reps])),
            "struct_like_mean": float(np.mean([r["struct_like"] for r in reps])),
            "s_hat_mean": float(np.mean([r["s_hat"] for r in reps])),
            "s_hat_sd": float(np.std([r["s_hat"] for r in reps])),
            "speed_test_reject": float(rej.mean()),
            "speed_test_reject_mcse": float(np.sqrt(rej.mean() * (1 - rej.mean()) / R)),
            "top_dev_counts": {k: int(sum(r["top_dev"] == k for r in reps)) for k in set(r["top_dev"] for r in reps)},
        }
        print(scenario, json.dumps({k: v for k, v in out[scenario].items() if k != "true_B_TD"}, indent=1), flush=True)
    with open("toy_results.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200, 20261102)

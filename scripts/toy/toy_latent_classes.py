"""Toy simulation (synthetic data only, no PPMI): can latent *mechanism* classes be
told apart from latent *speed* classes with PPMI-like follow-up?

Two latent classes, patient-level membership, class-specific transition models.
Truth "speed":  class 2 (40%) runs the shared system at speed 2 (no mechanism change).
Truth "mech":   class 2 also has an extra direct link Axial -> Independence.
Follow-up: N = 430 with T = 7 transitions (PPMI 1.0-like) or N = 950 with T = 2 (PPMI 2.0-like).

Fitted models (EM, patient-level responsibilities):
  free:  B1, B2 unrestricted (class-specific intercepts and covariances in both models)
  speed: B2 = B1^s
Test: LRT (free vs speed), compared with chi-square(35); class recovery by adjusted Rand index.
"""
import json
import sys

import numpy as np
from scipy import optimize, stats

sys.path.insert(0, ".")
import toy_speed_vs_pathway as T  # noqa: E402

P = T.P


def simulate_classes(rng, n, Tn, truth, frac2=0.4):
    z = (rng.random(n) < frac2).astype(int)
    A1 = T.drift(False)
    A2 = T.drift(truth == "mech")
    Xs, Ys = [], []
    for k, (A, s) in enumerate([(A1, 1.0), (A2, 2.0)]):
        idx = np.where(z == k)[0]
        X, Y = T.simulate(rng, len(idx), Tn, A, s)
        Xs.append((idx, X, Y))
    Xall = np.zeros((n * Tn, P))
    Yall = np.zeros((n * Tn, P))
    pid = np.repeat(np.arange(n), Tn)
    for idx, X, Y in Xs:
        rows = (idx[:, None] * Tn + np.arange(Tn)).ravel()
        Xall[rows] = X
        Yall[rows] = Y
    return Xall, Yall, pid, z


def class_loglik(X, Y, B, a, S):
    R = Y - X @ B.T - a
    Si = np.linalg.inv(S)
    sign, ld = np.linalg.slogdet(S)
    return -0.5 * (P * np.log(2 * np.pi) + ld + np.einsum("ij,jk,ik->i", R, Si, R))


def wls(X, Y, w):
    Z = np.column_stack([np.ones(len(X)), X])
    ZW = Z * w[:, None]
    coef = np.linalg.solve(Z.T @ ZW + 1e-9 * np.eye(P + 1), ZW.T @ Y)
    R = Y - Z @ coef
    S = (R * w[:, None]).T @ R / w.sum()
    return coef[1:].T, coef[0], (S + S.T) / 2


def wstats(X, Y, w, B):
    R = Y - X @ B.T
    a = (R * w[:, None]).sum(0) / w.sum()
    Rc = R - a
    S = (Rc * w[:, None]).T @ Rc / w.sum()
    return a, (S + S.T) / 2


def em(X, Y, pid, n, resp0, model, max_iter=300, tol=1e-7):
    resp = resp0.copy()
    ll_old = -np.inf
    theta = None
    for it in range(max_iter):
        w = resp[pid]  # (M, 2)
        pi = resp.mean(0)
        if model == "free":
            params = [wls(X, Y, w[:, k]) for k in range(2)]
        else:
            W = w.sum(0)

            def obj(th):
                B1 = th[:-1].reshape(P, P)
                B2 = T.mpow(B1, np.exp(th[-1]))
                if B2 is None:
                    return 1e12
                v = 0.0
                for k, Bk in enumerate([B1, B2]):
                    _, S = wstats(X, Y, w[:, k], Bk)
                    sign, ld = np.linalg.slogdet(S)
                    if sign <= 0:
                        return 1e12
                    v += 0.5 * W[k] * ld
                return v

            if theta is None:
                B1f, _, _ = wls(X, Y, w[:, 0])
                B2f, _, _ = wls(X, Y, w[:, 1])
                grid = np.linspace(np.log(0.3), np.log(4.0), 25)
                g0 = grid[np.argmin([obj(np.append(B1f.ravel(), g)) for g in grid])]
                theta = np.append(B1f.ravel(), g0)
            res = optimize.minimize(obj, theta, method="L-BFGS-B", options={"maxiter": 60})
            theta = res.x
            B1 = theta[:-1].reshape(P, P)
            B2 = T.mpow(B1, np.exp(theta[-1]))
            params = []
            for k, Bk in enumerate([B1, B2]):
                a, S = wstats(X, Y, w[:, k], Bk)
                params.append((Bk, a, S))
        L = np.zeros((n, 2))
        for k, (B, a, S) in enumerate(params):
            L[:, k] = np.bincount(pid, weights=class_loglik(X, Y, B, a, S), minlength=n)
        L += np.log(np.clip(pi, 1e-12, None))
        mx = L.max(1, keepdims=True)
        ll = float((mx[:, 0] + np.log(np.exp(L - mx).sum(1))).sum())
        resp = np.exp(L - mx)
        resp /= resp.sum(1, keepdims=True)
        if ll - ll_old < tol * abs(ll):
            break
        ll_old = ll
    out = {"ll": ll, "resp": resp, "pi": resp.mean(0)}
    if model == "speed":
        out["s"] = float(np.exp(theta[-1]))
    return out


def ari(a, b):
    from itertools import product
    n = len(a)
    ct = np.zeros((2, 2))
    for i, j in product(range(2), range(2)):
        ct[i, j] = np.sum((a == i) & (b == j))
    comb = lambda x: x * (x - 1) / 2  # noqa: E731
    sum_ij = comb(ct).sum()
    sa, sb = comb(ct.sum(1)).sum(), comb(ct.sum(0)).sum()
    exp_ = sa * sb / comb(n)
    mx = (sa + sb) / 2
    return float((sum_ij - exp_) / (mx - exp_)) if mx != exp_ else 0.0


def informed_start(X, Y, pid, n):
    # patients with larger average absolute change start in class 2
    ch = np.bincount(pid, weights=np.abs(Y - X).sum(1), minlength=n) / np.bincount(pid, minlength=n)
    r2 = (ch > np.quantile(ch, 0.6)).astype(float) * 0.8 + 0.1
    return np.column_stack([1 - r2, r2])


def one_rep(rng, n, Tn, truth, starts=4):
    X, Y, pid, z = simulate_classes(rng, n, Tn, truth)
    inits = [informed_start(X, Y, pid, n)] + [rng.dirichlet([1, 1], size=n) for _ in range(starts - 1)]
    fits = [em(X, Y, pid, n, r0, "free") for r0 in inits]
    best = max(fits, key=lambda f: f["ll"])
    sp_inits = [best["resp"], informed_start(X, Y, pid, n)]
    sfits = [em(X, Y, pid, n, r0, "speed", max_iter=80) for r0 in sp_inits]
    sbest = max(sfits, key=lambda f: f["ll"])
    lrt = 2 * (best["ll"] - sbest["ll"])
    zhat = best["resp"].argmax(1)
    return {"lrt": float(lrt), "p": float(stats.chi2.sf(max(lrt, 0), P * P - 1)),
            "ari_free": ari(z, zhat), "ari_speed": ari(z, sbest["resp"].argmax(1)),
            "s_hat": max(sbest["s"], 1 / sbest["s"])}


def main(R, seed):
    out = {}
    cells = [(430, 7), (950, 2)]
    for truth in ["speed", "mech"]:
        for (n, Tn) in cells:
            rng = np.random.default_rng(seed + {"speed": 0, "mech": 100}[truth] + n + Tn)
            reps = [one_rep(rng, n, Tn, truth) for _ in range(R)]
            rej = np.array([r["p"] < 0.05 for r in reps])
            key = f"{truth}_N{n}_T{Tn}"
            out[key] = {
                "R": R,
                "reject_rate": float(rej.mean()),
                "mcse": float(np.sqrt(rej.mean() * (1 - rej.mean()) / R)),
                "lrt_mean": float(np.mean([r["lrt"] for r in reps])),
                "ari_free_median": float(np.median([r["ari_free"] for r in reps])),
                "ari_speed_median": float(np.median([r["ari_speed"] for r in reps])),
                "s_hat_median": float(np.median([r["s_hat"] for r in reps])),
            }
            print(key, json.dumps(out[key]), flush=True)
    with open("toy_latent_results.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 100, 7)

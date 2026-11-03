"""Validation of the DYNOTEARS re-implementation (dynotears.py) against the paper's own benchmark
(Pamfil et al. 2020, Fig. 3, panels a and c: Gaussian noise, Erdos-Renyi intra-slice graphs of
mean degree 2 and Erdos-Renyi inter-slice graphs with k = 2, p = 1). Synthetic data only, no PPMI.

The generator follows the paper's Appendix B.4 and the authors' causalnex generator:
- intra-slice DAG: lower-triangular Bernoulli(k / (d - 1)) entries, rows and columns permuted;
  weights +-U(0.5, 2.0);
- inter-slice graph: Bernoulli(k / d) for every lag-1 pair, self-lags included; weights
  +-U(0.3, 0.5) (decay alpha = 1 at p = 1);
- X_t = X_t W + X_{t-1} A + Z_t with N(0, 1) noise, one series of n samples after a burn-in of
  100 + max(n // 10, 50); a draw is rejected if any value is non-finite or at least 1e3, or if
  either graph is empty (the causalnex stationarity heuristic).
Hyperparameters as in the paper's Appendix B.3: n = 500, lambda = 0.05, tau_W = 0.3, tau_A = 0.1;
n = 50, lambda = 0.2, tau_W = 0.3, tau_A = 0.2. F1 is computed on directed edges, separately for
the intra-slice and inter-slice matrices.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dynotears as DY  # noqa: E402

DIMS = [5, 10, 20, 50]
SETTINGS = {500: (0.05, 0.3, 0.1), 50: (0.2, 0.3, 0.2)}  # lambda, tau_W, tau_A
REPORTED = {  # DYNOTEARS F1 read from the paper's Fig. 3 (ER2 intra x ER2 inter), d = 5, 10, 20, 50
    500: {"intra": [1.00, 0.99, 0.99, 0.99], "inter": [1.00, 0.98, 0.99, 0.99]},
    50: {"intra": [0.80, 0.90, 0.90, 0.89], "inter": [0.65, 0.76, 0.77, 0.73]},
}


def signed(rng, size, lo, hi):
    return rng.uniform(lo, hi, size) * rng.choice([-1.0, 1.0], size)


def make_data(rng, d, n, k=2.0):
    burn = 100 + max(n // 10, 50)
    for _ in range(1000):
        B = np.tril(rng.random((d, d)) < k / (d - 1), -1)
        perm = rng.permutation(d)
        B = B[np.ix_(perm, perm)]
        W = B * signed(rng, (d, d), 0.5, 2.0)
        A = (rng.random((d, d)) < k / d) * signed(rng, (d, d), 0.3, 0.5)
        if not W.any() or not A.any():
            continue
        Minv = np.linalg.inv(np.eye(d) - W)
        X = np.zeros((n + 1 + burn, d))
        with np.errstate(over="ignore", invalid="ignore"):
            for t in range(1, len(X)):
                X[t] = (X[t - 1] @ A + rng.normal(size=d)) @ Minv
        X = X[burn:]
        if np.all(np.isfinite(X)) and np.abs(X).max() < 1e3:
            return W, A, X
    raise RuntimeError("no stationary draw")


def f1(true, est):
    t, e = true != 0, est != 0
    tp = np.sum(t & e)
    if t.sum() + e.sum() == 0:
        return 1.0
    return float(2 * tp / (t.sum() + e.sum()))


def one(args):
    n, d, rep = args
    rng = np.random.default_rng(20261103 + 100000 * d + 1000 * (n == 50) + rep)
    W, A, X = make_data(rng, d, n)
    lam, tw, ta = SETTINGS[n]
    t0 = time.time()
    We, Ae, info = DY.dynotears(X[1:], X[:-1], lam, lam, tw, ta)
    return {"n": n, "d": d, "rep": rep, "intra": f1(W, We), "inter": f1(A, Ae),
            "converged": info["converged"], "secs": time.time() - t0}


def main(reps, workers):
    plan = [(n, d, r) for n in (500, 50) for d in DIMS for r in range(reps)]
    with Pool(workers) as pool:
        res = pool.map(one, plan, chunksize=1)
    out = {"gradient_check_max_rel_error": max(DY.gradient_check(0), DY.gradient_check(1, d=5, p=2))}
    diffs = []
    for n in (500, 50):
        out[str(n)] = {}
        for kind in ("intra", "inter"):
            row = {}
            for d in DIMS:
                v = np.array([r[kind] for r in res if r["n"] == n and r["d"] == d])
                rep_val = REPORTED[n][kind][DIMS.index(d)]
                row[str(d)] = {"mean": float(v.mean()), "se": float(v.std(ddof=1) / np.sqrt(len(v))),
                               "reported": rep_val}
                diffs.append(abs(v.mean() - rep_val))
            out[str(n)][kind] = row
    out["mean_abs_diff"] = float(np.mean(diffs))
    out["max_abs_diff"] = float(np.max(diffs))
    out["converged"] = float(np.mean([r["converged"] for r in res]))
    out["secs_median_by_d"] = {str(d): float(np.median([r["secs"] for r in res if r["d"] == d])) for d in DIMS}
    out["reps"] = reps
    with open("toy_dynotears_validation_results.json", "w") as f:
        json.dump(out, f, indent=1)
    for n in (500, 50):
        for kind in ("intra", "inter"):
            print(n, kind, {d: (round(x["mean"], 2), round(x["se"], 2), x["reported"])
                            for d, x in out[str(n)][kind].items()}, flush=True)
    print("mean/max abs diff", round(out["mean_abs_diff"], 3), round(out["max_abs_diff"], 3),
          "converged", out["converged"], "secs", out["secs_median_by_d"])


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20, int(sys.argv[2]) if len(sys.argv) > 2 else 8)

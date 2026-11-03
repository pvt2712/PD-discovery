"""Validation of the tsLDiffPC / tsDCI re-implementation (ts_ldiffpc.py) against the paper's own
benchmark (Ruer et al. 2026, Fig. 4). Synthetic data only, no PPMI.

The generator follows the authors' code (setting2_lag1): d series; lag-1 edges with probability
0.3 per ordered pair plus a self-lag for every series; same-time edges with probability 0.3 per
forward pair of a random order (at least one); coefficients +-U(0.2, 0.8), with each target's
same-time and lagged coefficients each capped at a total of 0.95; N(0, 1) noise; two independent
series of 1,000 after a burn-in of 200. Only same-time coefficients change in the anomalous regime:
one edge, all same-time parents of a node, or all of a node with at least two, each redrawn as
+-U(0.2, 1.0) with a change of at least 0.5. The truth is the one changed series, and F1 =
2/(1 + |P|) when it is in the predicted set P (strict: heads of directed edges).
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
import ts_ldiffpc as TS  # noqa: E402

REPORTED = {  # strict F1 read from the paper's Fig. 4 (d = 3, 5, 7, 9)
    "tsdci": {"single": [0.83, 0.60, 0.73, 0.80], "all": [0.83, 0.67, 0.79, 0.90], "min2": [0.65, 0.87, 0.69, 0.77]},
    "tsldiffpc": {"single": [0.17, 0.00, 0.03, 0.00], "all": [0.17, 0.17, 0.33, 0.50], "min2": [0.72, 0.77, 0.83, 0.70]},
}


def coef(rng, lo, hi):
    return rng.choice([-1.0, 1.0]) * rng.uniform(lo, hi)


def make_system(rng, d, model, p_edge=0.3):
    for _ in range(500):
        A1 = np.zeros((d, d))  # A1[j, i]: X_i(t-1) -> X_j(t)
        for i in range(d):
            for j in range(d):
                if i == j or rng.random() < p_edge:
                    A1[j, i] = coef(rng, 0.2, 0.8)
        order = rng.permutation(d)
        A0 = np.zeros((d, d))  # A0[j, i]: X_i(t) -> X_j(t)
        for a in range(d):
            for b in range(a + 1, d):
                if rng.random() < p_edge:
                    A0[order[b], order[a]] = coef(rng, 0.2, 0.8)
        if not A0.any():
            continue
        if model == "min2" and not ((A0 != 0).sum(1) >= 2).any():
            continue
        break
    for M in (A0, A1):
        for j in range(d):
            s = np.abs(M[j]).sum()
            if s > 0.95:
                M[j] *= 0.95 / s
    rows, cols = np.nonzero(A0)
    if model == "single":
        k = rng.integers(len(rows))
        target, changed = rows[k], [(rows[k], cols[k])]
    else:
        if model == "all":
            target = rows[rng.integers(len(rows))]
        else:
            cand = np.where((A0 != 0).sum(1) >= 2)[0]
            target = cand[rng.integers(len(cand))]
        changed = [(target, i) for i in np.nonzero(A0[target])[0]]
    A0b = A0.copy()
    for j, i in changed:
        for _ in range(1000):
            new = coef(rng, 0.2, 1.0)
            if abs(new - A0[j, i]) >= 0.5:
                break
        A0b[j, i] = new
    return A0, A0b, A1, int(target)


def simulate(rng, A0, A1, n=1000, burn=200):
    d = len(A0)
    M = np.linalg.inv(np.eye(d) - A0)
    X = np.zeros((n + burn + 2, d))
    for t in range(1, len(X)):
        X[t] = M @ (A1 @ X[t - 1] + rng.normal(size=d))
    X = X[-n:]
    return np.column_stack([X[:-1], X[1:]])


def f1(truth, P):
    return 2 / (1 + len(P)) if truth in P else 0.0


def one(args):
    d, model, rep = args
    seed = 12000 + d * 100000 + rep + {"single": 0, "all": 1, "min2": 2}[model] * 10 ** 7
    rng = np.random.default_rng(seed)
    A0, A0b, A1, target = make_system(rng, d, model)
    Z1 = simulate(np.random.default_rng(seed + 1), A0, A1)
    Z2 = simulate(np.random.default_rng(seed + 2), A0b, A1)
    out = {"d": d, "model": model, "rep": rep}
    for variant in ("tsdci", "tsldiffpc"):
        t0 = time.time()
        edges, sepset = TS.skeleton_with_sepsets(Z1, Z2, d, variant=variant)
        directed, undirected = TS.orient(edges, sepset, Z1, Z2, d, variant=variant)
        heads, pairs = TS.root_causes(directed, undirected, d)
        out[variant] = {"strict": f1(target, heads), "relaxed": f1(target, heads | pairs),
                        "n_edges": len(edges), "secs": time.time() - t0}
    return out


def main(reps, workers, dims):
    plan = [(d, m, r) for d in dims for m in ("single", "all", "min2") for r in range(reps)]
    with Pool(workers) as pool:
        res = pool.map(one, plan, chunksize=1)
    out = {}
    for variant in ("tsdci", "tsldiffpc"):
        out[variant] = {}
        for m in ("single", "all", "min2"):
            row = {}
            for d in dims:
                v = np.array([r[variant]["strict"] for r in res if r["d"] == d and r["model"] == m])
                w = np.array([r[variant]["relaxed"] for r in res if r["d"] == d and r["model"] == m])
                row[str(d)] = {"strict_mean": float(v.mean()), "strict_se": float(v.std(ddof=1) / np.sqrt(len(v))),
                               "relaxed_mean": float(w.mean()),
                               "reported_strict": REPORTED[variant][m][[3, 5, 7, 9].index(d)]}
            out[variant][m] = row
    out["secs_median"] = {v: float(np.median([r[v]["secs"] for r in res])) for v in ("tsdci", "tsldiffpc")}
    with open("toy_tsldiffpc_validation_results.json", "w") as f:
        json.dump(out, f, indent=1)
    for variant in ("tsdci", "tsldiffpc"):
        for m in ("single", "all", "min2"):
            print(variant, m, {d: (round(x["strict_mean"], 2), round(x["strict_se"], 2), x["reported_strict"])
                               for d, x in out[variant][m].items()}, flush=True)
    print("secs", out["secs_median"])


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30, int(sys.argv[2]) if len(sys.argv) > 2 else 8,
         [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [3, 5, 7])

"""Toy simulation (synthetic data only, no PPMI): does the person-level random-speed test
stay calibrated as cohorts grow, where the group-level test did not?

Same speed-only cohorts as toy_random_speed_scale.py (identical seeds and draw order, so
the two tests are compared on the same data): shared A, log s_i = log s_g + 0.4 u_i,
s_TD = 1, s_PIGD = 2, five transitions each, sizes 400/200, x4 and x16. The person-level
model of toy_random_speed.py (H1 speed only vs H3 phenotype-specific drift, 8 df) is fitted
with exact gradients from torch instead of numerical ones.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import sys  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy import optimize, stats  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import toy_random_speed as RS  # noqa: E402

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

P = RS.P
NT = P * (P + 1) // 2
TRI = torch.tril_indices(P, P)
U_NODES = torch.tensor(RS.U_NODES)
LOG_W = torch.tensor(np.log(RS.U_W))
LOG2PI = float(np.log(2 * np.pi))


def chol(v):
    L = torch.zeros(P, P).index_put((TRI[0], TRI[1]), v)
    d = torch.diagonal(L)
    return L - torch.diag(d) + torch.diag(torch.exp(d))


def group_nll(X, Y, A, b, L, mlog, sigma):
    s = torch.exp(mlog + sigma * U_NODES)
    K = s.shape[0]
    n, Tn, _ = X.shape
    M1 = torch.cat([torch.cat([A, b[:, None]], 1), torch.zeros(1, P + 1)], 0)
    F1 = torch.linalg.matrix_exp(s[:, None, None] * M1)
    E, a = F1[:, :P, :P], F1[:, :P, P]
    M2 = torch.cat([torch.cat([-A, L @ L.T], 1), torch.cat([torch.zeros(P, P), A.T], 1)], 0)
    F2 = torch.linalg.matrix_exp(s[:, None, None] * M2)
    S = F2[:, P:, P:].transpose(1, 2) @ F2[:, :P, P:]
    Ls = torch.linalg.cholesky(0.5 * (S + S.transpose(1, 2)))
    logdet = 2 * torch.log(torch.diagonal(Ls, dim1=1, dim2=2)).sum(-1)
    R = Y[None] - torch.einsum("ntp,kqp->kntq", X, E) - a[:, None, None, :]
    Zs = torch.linalg.solve_triangular(Ls, R.reshape(K, -1, P).transpose(1, 2), upper=False)
    q = (Zs ** 2).sum(1).reshape(K, n, Tn).sum(-1)
    ll = -0.5 * (Tn * (P * LOG2PI + logdet)[:, None] + q)
    return -torch.logsumexp(ll + LOG_W[:, None], 0).sum()


def nll(th, model, data):
    """Parameter layout of toy_random_speed.unpack."""
    k = 0

    def take(m):
        nonlocal k
        v = th[k:k + m]
        k += m
        return v

    A1 = take(P * P).reshape(P, P)
    A2 = take(P * P).reshape(P, P) if model == "H3" else A1
    b1, b2 = take(P), take(P)
    L1, L2 = chol(take(NT)), chol(take(NT))
    gamma = take(1)[0] if model == "H1" else torch.zeros(())
    sigma = torch.exp(take(1)[0])
    (X1, Y1), (X2, Y2) = data
    return group_nll(X1, Y1, A1, b1, L1, torch.zeros(()), sigma) + group_nll(X2, Y2, A2, b2, L2, gamma, sigma)


def fit(model, data, x0):
    def f(x):
        xt = torch.tensor(x, requires_grad=True)
        try:
            val = nll(xt, model, data)
            if not torch.isfinite(val):
                return 1e10, np.zeros_like(x)
            val.backward()
            return float(val.detach()), xt.grad.numpy().copy()
        except RuntimeError:
            return 1e10, np.zeros_like(x)

    return optimize.minimize(f, x0, jac=True, method="L-BFGS-B", options={"maxiter": 5000, "maxfun": 20000})


def person_test(args):
    X1, X2 = args
    data = [(torch.tensor(X[:, :-1]), torch.tensor(X[:, 1:])) for X in (X1, X2)]
    r1 = fit("H1", data, RS.start_values(X1, X2, "H1"))
    r3 = min([fit("H3", data, RS.start_values(X1, X2, "H3")), fit("H3", data, RS.h1_to_h3(r1.x))],
             key=lambda r: r.fun)
    lrt = 2 * (r1.fun - r3.fun)
    return float(lrt), float(np.exp(r1.x[P * P + 2 * P + 2 * NT])), bool(r1.success and r3.success)


def main(seed, workers):
    out = {}
    cells = [(fast, m, R) for fast in (False, True) for m, R in ((1, 200), (4, 200), (16, 100))]
    with Pool(workers) as pool:
        for fast, m, R in cells:
            rng = np.random.default_rng(seed + 10 * fast + m)
            cohorts = []
            for _ in range(R):
                X1 = RS.simulate(rng, 400 * m, 5, RS.drift(False, fast), 1.0)
                X2 = RS.simulate(rng, 200 * m, 5, RS.drift(False, fast), 2.0)
                cohorts.append((X1, X2))
            res = pool.map(person_test, cohorts, chunksize=1)
            lrts = np.array([r[0] for r in res])
            rej = stats.chi2.sf(np.maximum(lrts, 0), P * P - 1) < 0.05
            key = ("fast" if fast else "slow") + f"_x{m}"
            out[key] = {"R": R, "n_TD": 400 * m, "n_PIGD": 200 * m,
                        "person_reject": float(rej.mean()),
                        "person_mcse": float(np.sqrt(rej.mean() * (1 - rej.mean()) / R)),
                        "lrt_person_mean": float(lrts.mean()),
                        "speed_ratio_median": float(np.median([r[1] for r in res])),
                        "converged": float(np.mean([r[2] for r in res])),
                        "lrt_negative": float(np.mean(lrts < -1e-3))}
            print(key, json.dumps(out[key]), flush=True)
    with open("toy_random_speed_scale_person_results.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        # torch likelihood equals the numpy one of toy_random_speed at the start values
        rng = np.random.default_rng(1)
        X1 = RS.simulate(rng, 100, 5, RS.drift(False, False), 1.0)
        X2 = RS.simulate(rng, 50, 5, RS.drift(False, False), 2.0)
        data = [(torch.tensor(X[:, :-1]), torch.tensor(X[:, 1:])) for X in (X1, X2)]
        for model in ("H1", "H3"):
            th = RS.start_values(X1, X2, model)
            print(model, float(nll(torch.tensor(th), model, data)), RS.negll(th, model, X1, X2))
    else:
        main(53, int(sys.argv[1]) if len(sys.argv) > 1 else 8)

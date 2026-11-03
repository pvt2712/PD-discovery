"""Toy simulation 5 (synthetic data only, no PPMI): tsLDiffPC and tsDCI (Ruer et al. 2026) on the
cohorts of toy 4 (same seeds, so the same cohorts).

  T1  naive use: TD and PIGD transitions as the two "regimes" (the H-M1 comparator, run as
      published: alpha = 0.05, every conditioning set of the other window nodes); T1c the same
      with conditioning sets of at most 3 nodes, the cap needed at ten domains
  T2  re-targeted use: PIGD's real transitions against a reference sample of TD's mechanism run at
      each PIGD patient's own speed. TD's continuous-time links are fitted on TD patients only;
      PIGD's speed distribution, load and noise are then fitted on PIGD patients with those links
      held fixed. Each patient's speed is drawn from its posterior given the whole trajectory, so
      the reference keeps the link between speed and disease state. Four reference copies per real
      transition.

The skeleton is shared by tsLDiffPC and tsDCI (ts_ldiffpc.py), so one run gives both. Root causes
are the current-block targets of retained lagged difference edges (self-lags included).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import toy_comparators as C  # noqa: E402
import ts_ldiffpc as TS  # noqa: E402

P = C.P
COPIES = 4


def two_slice(X):
    return np.concatenate([X[:, :-1].reshape(-1, P), X[:, 1:].reshape(-1, P)], axis=1)


def node_params(A, b, L, mlog, sigma):
    s = np.exp(mlog + sigma * C.U_NODES.numpy())
    return [C.transition_np(A, b, L @ L.T, sk) for sk in s]


def posterior_nodes(X, params):
    """Posterior probability of each speed node for each patient, given the whole trajectory."""
    Xt, Yt = X[:, :-1], X[:, 1:]
    ll = []
    for E, a, S in params:
        R = Yt - Xt @ E.T - a
        _, ld = np.linalg.slogdet(S)
        q = np.einsum("ntj,jk,ntk->n", R, np.linalg.inv(S), R)
        ll.append(-0.5 * (Xt.shape[1] * (P * C.LOG2PI + ld) + q))
    lw = np.array(ll) + C.LOG_W.numpy()[:, None]
    lw -= lw.max(0)
    w = np.exp(lw)
    return w / w.sum(0)


def reference_rows(X, params, post, rng, copies=COPIES):
    n = X.shape[0]
    Xt = X[:, :-1]
    rows = []
    for _ in range(copies):
        u = rng.random(n)
        k = (np.cumsum(post, 0) < u).sum(0).clip(0, len(params) - 1)
        Y = np.empty_like(Xt)
        for kk in np.unique(k):
            m = k == kk
            E, a, S = params[kk]
            Y[m] = Xt[m] @ E.T + a + rng.normal(size=Xt[m].shape) @ np.linalg.cholesky(S).T
        rows.append(np.concatenate([Xt.reshape(-1, P), Y.reshape(-1, P)], axis=1))
    return np.concatenate(rows)


def fit(f, x0):
    """L-BFGS-B with exact gradients for a torch objective f(z)."""
    def obj(x):
        xt = torch.tensor(x, requires_grad=True)
        try:
            val = f(xt)
            if not torch.isfinite(val):
                return 1e10, np.zeros_like(x)
            val.backward()
            return float(val.detach()), xt.grad.numpy().copy()
        except RuntimeError:
            return 1e10, np.zeros_like(x)
    return C.optimize.minimize(obj, x0, jac=True, method="L-BFGS-B", options={"maxiter": 5000, "maxfun": 20000})


def speed_matched_reference(groups, rng):
    """TD's mechanism, fitted on TD patients only; PIGD's speed distribution, load and noise
    fitted on PIGD patients with that mechanism held fixed."""
    NS, NT = C.NS, C.NT
    (XT, YT), (XP, YP) = [(torch.tensor(X[:, :-1]), torch.tensor(X[:, 1:])) for X in groups]
    h0 = C.start_values(groups)[0]  # [A, bT, bP, lqT, lqP, gamma, log sigma]
    zero = torch.zeros(())

    def drift(z):
        return torch.zeros(P, P).index_put((C.I_SUP, C.J_SUP), z)

    fT = lambda z: C.group_nll(XT, YT, drift(z[:NS]), z[NS:NS + P], C.chol(z[NS + P:NS + P + NT]), zero,
                               torch.exp(z[-1]))
    rT = fit(fT, np.concatenate([h0[:NS], h0[NS:NS + P], h0[NS + 2 * P:NS + 2 * P + NT], h0[-1:]]))
    A = drift(torch.tensor(rT.x[:NS])).detach()
    fP = lambda z: C.group_nll(XP, YP, A, z[:P], C.chol(z[P:P + NT]), z[P + NT], torch.exp(z[-1]))
    rP = fit(fP, np.concatenate([h0[NS + P:NS + 2 * P], h0[NS + 2 * P + NT:NS + 2 * P + 2 * NT], h0[-2:]]))
    z = rP.x
    LP = C.chol(torch.tensor(z[P:P + NT])).detach().numpy()
    params = node_params(A.numpy(), z[:P], LP, float(z[P + NT]), float(np.exp(z[-1])))
    post = posterior_nodes(groups[1], params)
    return reference_rows(groups[1], params, post, rng), float(np.exp(z[P + NT]))


def record(kept, name):
    F = TS.lag_matrix(kept, P)
    return {f"{name}": C.flags_off(F), f"{name}_self": F.diagonal().tolist(),
            f"{name}_roots": F.any(1).tolist(), f"{name}_contemp": len(TS.contemporaneous_pairs(kept, P))}


def one_rep(args):
    scenario, seed = args
    rng = np.random.default_rng(seed)
    groups = C.simulate(rng, scenario)
    rec = {"scenario": scenario, "seed": seed}
    t0 = time.time()
    kept, nt = TS.skeleton(two_slice(groups[0]), two_slice(groups[1]), P)
    rec.update(record(kept, "T1"))
    rec["T1_tests"], rec["T1_secs"] = nt, time.time() - t0
    t0 = time.time()
    kept, nt = TS.skeleton(two_slice(groups[0]), two_slice(groups[1]), P, max_s=3)
    rec.update(record(kept, "T1c"))
    rec["T1c_tests"], rec["T1c_secs"] = nt, time.time() - t0
    t0 = time.time()
    ref, s_hat = speed_matched_reference(groups, np.random.default_rng(seed + 13))
    kept, nt = TS.skeleton(two_slice(groups[1]), ref, P)
    rec.update(record(kept, "T2"))
    rec["T2_tests"], rec["T2_secs"], rec["s_hat"] = nt, time.time() - t0, s_hat
    return rec


def footprints(s=2.0, tol=1e-3):
    from scipy import linalg
    out = {}
    for sc, kind in (("S2", "extra"), ("S3", "strong")):
        D = linalg.expm(s * C.drift(kind)) - linalg.expm(s * C.drift(None))
        out[sc] = {f"{C.NAMES[j]}->{C.NAMES[i]}" for i, j in C.OFF if abs(D[i, j]) > tol}
    return out


def summarise(reps, scenario, foot):
    true = {f"{C.NAMES[j]}->{C.NAMES[i]}" for i, j in C.TRUE_DIFF[scenario]}
    shared_map = [k for k in C.MAP_KEYS if k not in true]
    s = {"R": len(reps)}
    for m in ("T1", "T1c", "T2"):
        shared_all = [k for k in reps[0][m] if k not in true]
        fl_map = np.array([[r[m][k] for k in shared_map] for r in reps])
        fl_all = np.array([[r[m][k] for k in shared_all] for r in reps])
        roots = np.array([r[f"{m}_roots"] for r in reps])
        others = [i for i in range(P) if not (true and i == C.INDEPENDENCE)]
        d = {"any_off_flag": float(np.mean([any(r[m].values()) for r in reps])),
             "false_rate_map_links": float(fl_map.mean()), "false_rate_all_off": float(fl_all.mean()),
             "false_map_links_mean": float(fl_map.sum(1).mean()),
             "self_lag_flag_rate": float(np.mean([r[f"{m}_self"] for r in reps])),
             "false_root_rate": float(roots[:, others].mean()),
             "contemporaneous_mean": float(np.mean([r[f"{m}_contemp"] for r in reps])),
             "tests_median": float(np.median([r[f"{m}_tests"] for r in reps])),
             "secs_median": float(np.median([r[f"{m}_secs"] for r in reps]))}
        if true:
            (tk,) = true
            d["power_true_link"] = float(np.mean([r[m][tk] for r in reps]))
            d["exactly_true_link"] = float(np.mean([r[m][tk] and not any(r[m][k] for k in shared_all)
                                                    for r in reps]))
            outside = [k for k in shared_all if k not in foot[scenario]]
            d["false_rate_outside_footprint"] = float(np.mean([[r[m][k] for k in outside] for r in reps]))
            d["footprint_hit_rate"] = float(np.mean([[r[m][k] for k in foot[scenario]] for r in reps]))
            d["independence_root"] = float(roots[:, C.INDEPENDENCE].mean())
            d["only_independence_root"] = float(np.mean(roots[:, C.INDEPENDENCE] & ~roots[:, others].any(1)))
        s[m] = d
    s["s_hat_median"] = float(np.median([r["s_hat"] for r in reps]))
    return s


def main(R, seed, workers):
    plan = [(sc, seed + 1000 * k + i) for k, sc in enumerate(C.SCEN)
            for i in range(R if sc in ("S0", "S1") else max(R // 2, 1))]
    with Pool(workers) as pool:
        reps = pool.map(one_rep, plan, chunksize=1)
    foot = footprints()
    out = {sc: summarise([r for r in reps if r["scenario"] == sc], sc, foot) for sc in C.SCEN}
    with open("toy_tsldiffpc_results.json", "w") as f:
        json.dump(out, f, indent=1)
    for sc in C.SCEN:
        print(sc, json.dumps(out[sc]), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        for sc, sd in (("S1", 20261103 + 1000), ("S2", 20261103 + 2000)):
            r = one_rep((sc, sd))
            print(sc, "T1", round(r["T1_secs"], 1), "s", r["T1_tests"], "tests;",
                  [k for k, v in r["T1"].items() if v], "self", r["T1_self"], "contemp", r["T1_contemp"])
            print(sc, "T2", round(r["T2_secs"], 1), "s", r["T2_tests"], "tests;",
                  [k for k, v in r["T2"].items() if v], "self", r["T2_self"], "contemp", r["T2_contemp"],
                  "s_hat", round(r["s_hat"], 3), flush=True)
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else 200, 20261103,
             int(sys.argv[2]) if len(sys.argv) > 2 else 8)

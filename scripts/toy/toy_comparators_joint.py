"""Toy simulation 4, joint per-link tests on the same cohorts (synthetic data only, no PPMI).

  O3j  per-link Wald tests of D = 0 with every cross-link deviation of the map freed at once
       (PIGD drift = A + D, D on the 8 off-diagonal map links; each domain's own decay is
       shared, so it anchors the speed), person-level random speed, BH 5% over the 8 links.
       This is the coefficient-equality test of LDiffPC / DCI conditioned on all other
       parents, re-targeted to the speed-scaled reference. O3 in toy_comparators.py tests
       one link at a time at the speed-only fit instead.
  O3m  anchor-free per-link tests: free TD and PIGD drifts, speed by the minimal-L1 rule (the
       weighted median of the drift ratios), Monte Carlo p-values (Chen et al. 2023 for item
       bias; Bowden et al. 2016 for Mendelian randomisation), BH 5% over the 8 links.
  Per-domain versions of O3j and O3m test all deviations into one domain at once.

Seeds and draw order match toy_comparators.py, so the cohorts are identical.
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
import toy_comparators as C  # noqa: E402
import toy_speed_vs_pathway as T  # noqa: E402

P, NS, SUP_IDX = C.P, C.NS, C.SUP_IDX
FREE = np.array(C.SUP_OFF)  # D entries freed: the off-diagonal map links


def joint_tests(groups):
    data = [(torch.tensor(X[:, :-1]), torch.tensor(X[:, 1:])) for X in groups]
    h0s, _ = C.start_values(groups)
    r0 = C.minimize(C.make_obj("H0", data), h0s)
    n0 = len(r0.x)
    idx = torch.tensor(FREE)

    def f_red(z):
        D = torch.zeros(NS, dtype=z.dtype).index_put((idx,), z[n0:])
        return C.nll(torch.cat([z[:n0], D]), "J", data)

    def obj(x):
        xt = torch.tensor(x, requires_grad=True)
        try:
            val = f_red(xt)
            if not torch.isfinite(val):
                return 1e10, np.zeros_like(x)
            val.backward()
            return float(val.detach()), xt.grad.numpy().copy()
        except RuntimeError:
            return 1e10, np.zeros_like(x)

    r = optimize.minimize(obj, np.concatenate([r0.x, np.zeros(len(FREE))]), jac=True, method="L-BFGS-B",
                          options={"maxiter": 5000, "maxfun": 20000})
    H = torch.autograd.functional.hessian(f_red, torch.tensor(r.x)).numpy()
    V = np.linalg.inv(H)
    d = r.x[n0:]
    se = np.sqrt(np.clip(np.diag(V)[n0:], 1e-300, None))
    p = 2 * stats.norm.sf(np.abs(d / se))
    lrt = 2 * (r0.fun - r.fun)
    # per-domain Wald test: all free deviations into one domain at once
    Vd = V[n0:, n0:]
    p_dom = np.ones(P)
    for i in range(P):
        L = [k for k, l in enumerate(FREE) if SUP_IDX[l][0] == i]
        if L:
            p_dom[i] = stats.chi2.sf(float(d[L] @ np.linalg.solve(Vd[np.ix_(L, L)], d[L])), len(L))
    return p, d, float(lrt), bool(r.success and r0.success), p_dom


def minimal_l1_tests(groups, rng, M=2000):
    """O3m: anchor-free per-link tests (Chen et al. 2023 for DIF; Bowden 2016 for MR).
    Fit TD and PIGD drifts freely (HA, person-level random speed), map them to speed and
    deviations by the minimal-L1 rule (speed = weighted median of the entry ratios,
    weights |A_TD|; D = A_PIGD / speed - A_TD), and get p-values from Monte Carlo draws of
    the fitted drifts (centred), BH over the 8 off-diagonal map links."""
    data = [(torch.tensor(X[:, :-1]), torch.tensor(X[:, 1:])) for X in groups]
    h0s, has = C.start_values(groups)
    r0 = C.minimize(C.make_obj("H0", data), h0s)
    fa = C.make_obj("HA", data)
    ra = min([C.minimize(fa, C.h0_to_ha(r0.x)), C.minimize(fa, has)], key=lambda r: r.fun)
    H = torch.autograd.functional.hessian(lambda z: C.nll(z, "HA", data), torch.tensor(ra.x)).numpy()
    V = np.linalg.inv(H)[:2 * NS, :2 * NS]

    def to_delta(a, a2):
        safe = np.where(np.abs(a) > 1e-10, a, 1e-10)
        r = a2 / safe
        o = np.argsort(r, axis=-1)
        w = np.take_along_axis(np.abs(a), o, -1)
        c = np.cumsum(w, -1)
        k = (c < 0.5 * c[..., -1:]).sum(-1)
        s = np.take_along_axis(np.take_along_axis(r, o, -1), k[..., None], -1)
        return s[..., 0], a2 / s - a

    s_hat, d_hat = to_delta(ra.x[:NS], ra.x[NS:2 * NS])
    L = np.linalg.cholesky(0.5 * (V + V.T))
    draws = ra.x[:2 * NS] + rng.normal(size=(M, 2 * NS)) @ L.T
    _, d_star = to_delta(draws[:, :NS], draws[:, NS:])
    p = (1 + (np.abs(d_star - d_hat) >= np.abs(d_hat)).sum(0)) / (M + 1)
    # per-domain test: all deviations into one domain (own rate included), Monte Carlo reference
    p_dom = np.ones(P)
    for i in range(P):
        L = [l for l in range(NS) if SUP_IDX[l][0] == i]
        Ds = d_star[:, L] - d_hat[L]
        Si = np.linalg.pinv(np.atleast_2d(np.cov(Ds.T)))
        t_obs = d_hat[L] @ Si @ d_hat[L]
        p_dom[i] = (1 + (np.einsum("mi,ij,mj->m", Ds, Si, Ds) >= t_obs).sum()) / (M + 1)
    return p, float(s_hat), p_dom


def one_rep(args):
    scenario, seed = args
    rng = np.random.default_rng(seed)
    groups = C.simulate(rng, scenario)
    p, d, lrt, conv, pj_dom = joint_tests(groups)
    F = np.zeros((P, P), bool)
    flags = T.bh(p)
    for l, fl in zip(FREE, flags):
        F[tuple(SUP_IDX[l])] = fl
    pm, s_l1, pm_dom = minimal_l1_tests(groups, np.random.default_rng(seed + 11))
    testable = [i for i in range(P) if any(SUP_IDX[l][0] == i for l in FREE)]
    fj_dom = np.zeros(P, bool)
    fj_dom[testable] = T.bh(pj_dom[testable])
    Fm = np.zeros((P, P), bool)
    for l, fl in zip(FREE, T.bh(pm[FREE])):
        Fm[tuple(SUP_IDX[l])] = fl
    diag = [l for l in range(NS) if SUP_IDX[l][0] == SUP_IDX[l][1]]
    return {"scenario": scenario, "seed": seed, "O3j": C.flags_off(F), "lrt8": lrt,
            "p": p.tolist(), "d": d.tolist(), "conv": conv,
            "O3m": C.flags_off(Fm), "O3m_diag": int(T.bh(pm[diag]).sum()), "s_l1": s_l1,
            "p_m": pm.tolist(), "O3j_dom": fj_dom.tolist(), "O3m_dom": T.bh(pm_dom).tolist()}


def summarise(reps, scenario):
    true = {f"{C.NAMES[j]}->{C.NAMES[i]}" for i, j in C.TRUE_DIFF[scenario]}
    shared_map = [k for k in C.MAP_KEYS if k not in true]
    s = {"R": len(reps), "lrt8_mean": float(np.mean([r["lrt8"] for r in reps])),
         "lrt8_reject": float(np.mean([stats.chi2.sf(max(r["lrt8"], 0), len(FREE)) < 0.05 for r in reps])),
         "converged": float(np.mean([r["conv"] for r in reps])),
         "s_l1_median": float(np.median([r["s_l1"] for r in reps])),
         "O3m_diag_mean": float(np.mean([r["O3m_diag"] for r in reps]))}
    for m in ("O3j", "O3m"):
        fl = np.array([[r[m][k] for k in shared_map] for r in reps])
        d = {"any_off_flag": float(np.mean([any(r[m].values()) for r in reps])),
             "false_rate_map_links": float(fl.mean()), "false_map_links_mean": float(fl.sum(1).mean())}
        if true:
            (tk,) = true
            d["power_true_link"] = float(np.mean([r[m][tk] for r in reps]))
            d["exactly_true_link"] = float(np.mean([r[m][tk] and not any(r[m][k] for k in shared_map)
                                                    for r in reps]))
        s[m] = d
    for m in ("O3j_dom", "O3m_dom"):
        nodes = np.array([r[m] for r in reps])
        others = [i for i in range(P) if not (true and i == C.INDEPENDENCE)]
        d = {"any_node": float(nodes[:, others].any(1).mean()), "false_node_rate": float(nodes[:, others].mean())}
        if true:
            d["independence_flagged"] = float(nodes[:, C.INDEPENDENCE].mean())
            d["only_independence"] = float(np.mean(nodes[:, C.INDEPENDENCE] & ~nodes[:, others].any(1)))
        s[m] = d
    return s


def main(R, seed, workers):
    plan = [(sc, seed + 1000 * k + i) for k, sc in enumerate(C.SCEN)
            for i in range(R if sc in ("S0", "S1") else max(R // 2, 1))]
    with Pool(workers) as pool:
        reps = pool.map(one_rep, plan, chunksize=1)
    out = {sc: summarise([r for r in reps if r["scenario"] == sc], sc) for sc in C.SCEN}
    with open("toy_comparators_joint_results.json", "w") as f:
        json.dump(out, f, indent=1)
    for sc in C.SCEN:
        print(sc, json.dumps(out[sc]), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        for sc, sd in (("S1", 20261103 + 1000), ("S3", 20261103 + 3000)):
            r = one_rep((sc, sd))
            print(sc, "O3j", [k for k, v in r["O3j"].items() if v], "O3m", [k for k, v in r["O3m"].items() if v],
                  "domains", r["O3j_dom"], r["O3m_dom"], round(r["s_l1"], 3))
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else 200, 20261103,
             int(sys.argv[2]) if len(sys.argv) > 2 else 8)

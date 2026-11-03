"""Toy simulation 4, extra checks on the same cohorts (synthetic data only, no PPMI).

  N1s  per-link tests that free one group interaction at a time from the invariant model,
       BH 5% (the per-edge step of multigroup RI-CLPM score tests, psychonetrics
       equalityScoreTest and IVPP partial pruning)
  O4   node-level speed-adjusted score test: all D entries into one domain at once (the
       FANS / CD-NOD question asked against the speed-scaled reference), BH over domains
  O5   model-light speed-family check: do the two yearly matrices commute, as they must if
       they share eigenvectors (Xu, Duker & Matteson 2024)? Centred person bootstrap of the
       commutator norm

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
from scipy import linalg, stats  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import toy_comparators as C  # noqa: E402
import toy_speed_vs_pathway as T  # noqa: E402

P, NS, SUP_IDX = C.P, C.NS, C.SUP_IDX


def one_at_a_time(Xt, Yt, g, pid):
    """N1s: for each link k -> j, OLS of X_j(t+1) on [1, G, X(t), G*X_k(t)], person-clustered."""
    n = len(Xt)
    npers = int(pid.max()) + 1
    Tn = n // npers
    p = np.zeros((P, P))
    for k in range(P):
        Z = np.column_stack([np.ones(n), g, Xt, g * Xt[:, k]])
        q = Z.shape[1]
        ZtZi = np.linalg.inv(Z.T @ Z)
        Cf = ZtZi @ Z.T @ Yt
        E = Yt - Z @ Cf
        for j in range(P):
            S = (Z * E[:, [j]]).reshape(npers, Tn, q).sum(1)
            V = ZtZi @ (S.T @ S) @ ZtZi * (npers / (npers - 1)) * ((n - 1) / (n - q))
            p[j, k] = 2 * stats.norm.sf(abs(Cf[-1, j]) / np.sqrt(V[-1, -1]))
    return p


def node_score_tests(groups):
    """O4: score test of D[i, :] = 0 (support entries into domain i) at the speed-only fit."""
    data = [(torch.tensor(X[:, :-1]), torch.tensor(X[:, 1:])) for X in groups]
    h0s, _ = C.start_values(groups)
    r0 = C.minimize(C.make_obj("H0", data), h0s)
    xt = torch.tensor(np.concatenate([r0.x, np.zeros(NS)]))
    H = torch.autograd.functional.hessian(lambda z: C.nll(z, "J", data), xt).numpy()
    gr = torch.autograd.functional.jacobian(lambda z: C.nll(z, "J", data), xt).numpy()
    n0 = len(r0.x)
    Hth = H[:n0, :n0]
    p = np.ones(P)
    for i in range(P):
        L = [n0 + l for l in range(NS) if SUP_IDX[l][0] == i]
        Hl = H[np.ix_(L, range(n0))]
        sol = np.linalg.solve(Hth, Hl.T)
        u = gr[L] - sol.T @ gr[:n0]
        info = H[np.ix_(L, L)] - Hl @ sol
        p[i] = stats.chi2.sf(float(u @ np.linalg.solve(info, u)), len(L))
    return p


def commutator_test(grams, rng, nboot=200):
    """O5: H0 B_TD B_PIGD = B_PIGD B_TD, from the centred bootstrap of the Frobenius norm."""
    B = [C.B_from(G, H) for G, H in grams]
    Cm = B[0] @ B[1] - B[1] @ B[0]
    obs = np.sqrt((Cm ** 2).sum())
    boot = []
    for _ in range(nboot):
        Bb = []
        for G, H in grams:
            idx = rng.integers(0, len(G), len(G))
            Bb.append(C.B_from(G[idx], H[idx]))
        Cb = Bb[0] @ Bb[1] - Bb[1] @ Bb[0]
        boot.append(np.sqrt(((Cb - Cm) ** 2).sum()))
    return (1 + np.sum(np.array(boot) >= obs)) / (nboot + 1)


def one_rep(args):
    scenario, seed = args
    rng = np.random.default_rng(seed)
    groups = C.simulate(rng, scenario)
    Xt, Yt, g, pid = C.stack(groups)
    rec = {"scenario": scenario, "seed": seed}
    rec["N1s"] = C.flags_off(T.bh(one_at_a_time(Xt, Yt, g, pid).ravel()).reshape(P, P))
    rec["O4_nodes"] = T.bh(node_score_tests(groups)).tolist()
    rec["O5"] = bool(commutator_test(C.person_grams(groups), np.random.default_rng(seed + 7)) < 0.05)
    return rec


def footprints(s=2.0, tol=1e-3):
    """Off-diagonal yearly entries that one continuous-time link difference moves."""
    out = {}
    for sc, kind in (("S2", "extra"), ("S3", "strong")):
        D = linalg.expm(s * C.drift(kind)) - linalg.expm(s * C.drift(None))
        out[sc] = {f"{C.NAMES[j]}->{C.NAMES[i]}" for i, j in C.OFF if abs(D[i, j]) > tol}
    return out


def summarise(reps, extra, scenario, foot):
    true = {f"{C.NAMES[j]}->{C.NAMES[i]}" for i, j in C.TRUE_DIFF[scenario]}
    s = {"R": len(extra)}
    shared_map = [k for k in C.MAP_KEYS if k not in true]
    fl = np.array([[r["N1s"][k] for k in shared_map] for r in extra])
    d = {"any_off_flag": float(np.mean([any(r["N1s"].values()) for r in extra])),
         "false_rate_map_links": float(fl.mean())}
    if true:
        (tk,) = true
        shared_all = [k for k in extra[0]["N1s"] if k not in true]
        d["power_true_link"] = float(np.mean([r["N1s"][tk] for r in extra]))
        d["exactly_true_link"] = float(np.mean([r["N1s"][tk] and not any(r["N1s"][k] for k in shared_all)
                                                for r in extra]))
    s["N1s"] = d
    nodes = np.array([r["O4_nodes"] for r in extra])
    others = [i for i in range(P) if not (true and i == C.INDEPENDENCE)]
    d = {"any_node": float(nodes[:, others].any(1).mean()), "false_node_rate": float(nodes[:, others].mean())}
    if true:
        d["independence_flagged"] = float(nodes[:, C.INDEPENDENCE].mean())
        d["only_independence"] = float(np.mean(nodes[:, C.INDEPENDENCE] & ~nodes[:, others].any(1)))
    s["O4_nodes"] = d
    v = np.array([r["O5"] for r in extra])
    s["O5"] = {"reject": float(v.mean()), "mcse": float(np.sqrt(v.mean() * (1 - v.mean()) / len(v)))}
    if true and reps:
        # footprint-aware view: flags outside the yearly footprint of the real difference
        outside_all = [k for k in reps[0]["N1"] if k not in foot[scenario]]
        outside_map = [k for k in C.MAP_KEYS if k not in foot[scenario]]
        for name, keys in (("all_off", outside_all), ("map_links", outside_map)):
            s[f"false_rate_outside_footprint_{name}"] = {
                m: float(np.mean([[r[m][k] for k in keys] for r in (extra if m == "N1s" else reps)]))
                for m in C.LINK_METHODS + ["N1s"]}
        s["footprint"] = sorted(foot[scenario])
    return s


def main(R, seed, workers):
    plan = [(sc, seed + 1000 * k + i) for k, sc in enumerate(C.SCEN)
            for i in range(R if sc in ("S0", "S1") else max(R // 2, 1))]
    with Pool(workers) as pool:
        extra = pool.map(one_rep, plan, chunksize=1)
    try:
        with open("toy_comparators_reps.json") as f:
            reps = json.load(f)
    except FileNotFoundError:
        reps = []
    foot = footprints()
    out = {sc: summarise([r for r in reps if r["scenario"] == sc], [r for r in extra if r["scenario"] == sc],
                         sc, foot) for sc in C.SCEN}
    with open("toy_comparators_extra_results.json", "w") as f:
        json.dump(out, f, indent=1)
    for sc in C.SCEN:
        print(sc, json.dumps(out[sc]), flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200, 20261103,
         int(sys.argv[2]) if len(sys.argv) > 2 else 8)

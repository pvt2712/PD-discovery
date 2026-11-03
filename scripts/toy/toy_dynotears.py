"""Toy simulation 6 (synthetic data only, no PPMI): DYNOTEARS (Pamfil et al. 2020), the
gradient-based structure learner, as named comparator C12 on the cohorts of toy 4 (same seeds, so
the same cohorts).

Use as an analyst would run it unmodified: one graph per phenotype from the pooled two-slice
tables (every patient's yearly transitions stacked, as causalnex does for a list of series), on
domain scores standardised over the whole cohort. The graphs are then compared link by link: a
link present in one phenotype's graph and absent from the other's is read as a mechanism
difference. Two settings:
  C12   the paper's simulation settings for n >= 500: lambda_W = lambda_A = 0.05, tau_W = 0.3,
        tau_A = 0.1 (Pamfil 2020, Appendix B.3); the pre-specified comparator
  C12d  the causalnex defaults: lambda = 0.1, no threshold
Lagged links are scored as in toy 4 (off-diagonal links; self-lags reported separately).
Same-time (intra-slice) edges are compared as directed edges and reported separately.
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
import toy_comparators as C  # noqa: E402

P = C.P
SETTINGS = {"C12": (0.05, 0.3, 0.1), "C12d": (0.1, 0.0, 0.0)}  # lambda, tau_W, tau_A


def one_rep(args):
    scenario, seed = args
    rng = np.random.default_rng(seed)
    groups = C.simulate(rng, scenario)
    allv = np.concatenate([X.reshape(-1, P) for X in groups])
    mu, sd = allv.mean(0), allv.std(0)
    rec = {"scenario": scenario, "seed": seed}
    for name, (lam, tw, ta) in SETTINGS.items():
        lag, same, t0 = [], [], time.time()
        for X in groups:
            Z = (X - mu) / sd
            W, A, info = DY.dynotears(Z[:, 1:].reshape(-1, P), Z[:, :-1].reshape(-1, P), lam, lam, tw, ta)
            lag.append((A != 0).T)  # toy orientation: [to, from]
            same.append(W != 0)
        F = lag[0] ^ lag[1]
        Sd = same[0] ^ same[1]
        off = ~np.eye(P, dtype=bool)
        rec[name] = C.flags_off(F)
        rec[name + "_self"] = [bool(F[i, i]) for i in range(P)]
        rec[name + "_same_n"] = int(Sd.sum())
        rec[name + "_edges"] = [int((L & off).sum()) for L in lag]
        rec[name + "_same_edges"] = [int(S.sum()) for S in same]
        rec[name + "_secs"] = (time.time() - t0) / 2
    return rec


def summarise(reps, scenario, name):
    true = {f"{C.NAMES[j]}->{C.NAMES[i]}" for i, j in C.TRUE_DIFF[scenario]}
    shared_map = [k for k in C.MAP_KEYS if k not in true]
    shared_all = [k for k in reps[0][name] if k not in true]
    fl_map = np.array([[r[name][k] for k in shared_map] for r in reps])
    fl_all = np.array([[r[name][k] for k in shared_all] for r in reps])
    self_ = np.array([r[name + "_self"] for r in reps])
    same_n = np.array([r[name + "_same_n"] for r in reps])
    any_off = np.array([any(r[name][k] for k in shared_all) for r in reps])
    s = {"R": len(reps),
         "any_off_flag": float(np.mean([any(r[name].values()) for r in reps])),
         "any_false_off_flag": float(any_off.mean()),
         "false_rate_map_links": float(fl_map.mean()),
         "false_rate_all_off": float(fl_all.mean()),
         "false_map_links_mean": float(fl_map.sum(1).mean()),
         "self_lag_flag_rate": float(self_.mean()),
         "same_time_any": float((same_n > 0).mean()),
         "same_time_mean": float(same_n.mean()),
         "any_flag_any_kind": float(np.mean(any_off | self_.any(1) | (same_n > 0))),
         "lagged_edges_median": [float(np.median([r[name + "_edges"][g] for r in reps])) for g in (0, 1)],
         "same_edges_median": [float(np.median([r[name + "_same_edges"][g] for r in reps])) for g in (0, 1)],
         "secs_per_fit_median": float(np.median([r[name + "_secs"] for r in reps]))}
    if true:
        (tk,) = true
        hit = np.array([r[name][tk] for r in reps])
        s["power_true_link"] = float(hit.mean())
        s["exactly_true_link"] = float(np.mean(hit & ~any_off))
    return s


def main(R, seed, workers):
    plan = [(sc, seed + 1000 * k + i) for k, sc in enumerate(C.SCEN)
            for i in range(R if sc in ("S0", "S1") else max(R // 2, 1))]
    with Pool(workers) as pool:
        reps = pool.map(one_rep, plan, chunksize=1)
    out = {name: {sc: summarise([r for r in reps if r["scenario"] == sc], sc, name) for sc in C.SCEN}
           for name in SETTINGS}
    with open("toy_dynotears_results.json", "w") as f:
        json.dump(out, f, indent=1)
    for name in SETTINGS:
        for sc in C.SCEN:
            print(name, sc, json.dumps(out[name][sc]), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        for sc in C.SCEN:
            t0 = time.time()
            r = one_rep((sc, 20261103 + 7))
            print(sc, round(time.time() - t0, 1), "s", "C12", [k for k, v in r["C12"].items() if v],
                  "self", r["C12_self"], "same", r["C12_same_n"], "edges", r["C12_edges"], r["C12_same_edges"],
                  "| C12d", sum(r["C12d"].values()), r["C12d_edges"], flush=True)
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else 200, 20261103,
             int(sys.argv[2]) if len(sys.argv) > 2 else 8)

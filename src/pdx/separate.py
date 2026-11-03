"""Separate graphs, the transparent reference (protocol v0.5, §7.3.5; v0.3's Analysis B).

Each phenotype group is subsampled n_sub times to the same size (90% of the smaller group); the
Step 1 learner in its fast configuration runs on every subsample, giving each lagged link's
selection frequency p(e, k). The statistic is D(e) = p(e, TD) - p(e, PIGD), for adjacencies and
for directed edges separately. References: permuted labels (by participant), and optionally the
definition-matched generator's datasets. Each link's D(e) is centred and scaled by its null mean
and spread (with a floor of 1/n_sub, so a link never seen in the null still counts); candidate
links exceed the family-wise threshold, the 95th percentile of the maximum over links in each
null dataset (Westfall & Young 1993), and have |D(e)| >= 0.25 (provisional, D10). Never a
mechanism verdict: a pure speed difference also moves yearly graphs.
"""
from __future__ import annotations

import numpy as np

from .data import Panel
from .step1 import Step1Config, lagged_adjacency, lagged_directed, run_fci


def lagged_frequencies(panel: Panel, idx: np.ndarray, n_sub: int, size: int, rng: np.random.Generator,
                       cfg: Step1Config) -> tuple[np.ndarray, np.ndarray]:
    d = panel.d
    adj, dirn = np.zeros((d, d)), np.zeros((d, d))
    for _ in range(n_sub):
        sub = panel.subset(rng.choice(idx, size, replace=False))
        G, cols = run_fci(sub, cfg)[:2]
        adj += lagged_adjacency(G, cols, d)
        dirn += lagged_directed(G, cols, d)
    return adj / n_sub, dirn / n_sub


def _difference(panel: Panel, group: np.ndarray, n_sub: int, frac: float, rng, cfg):
    g0, g1 = np.where(group == 0)[0], np.where(group == 1)[0]
    size = max(int(frac * min(len(g0), len(g1))), 2)
    a0, d0 = lagged_frequencies(panel, g0, n_sub, size, rng, cfg)
    a1, d1 = lagged_frequencies(panel, g1, n_sub, size, rng, cfg)
    return a0 - a1, d0 - d1


def separate_graph_reference(panel: Panel, n_sub: int = 25, frac: float = 0.9, n_perm: int = 50,
                             rng: np.random.Generator | None = None, cfg: Step1Config | None = None,
                             threshold: float = 0.25, alpha_fw: float = 0.05,
                             null_panels: list | None = None) -> dict:
    rng = rng or np.random.default_rng()
    cfg = cfg or Step1Config()
    D_adj, D_dir = _difference(panel, panel.group, n_sub, frac, rng, cfg)
    out = {"D_adjacency": D_adj.tolist(), "D_directed": D_dir.tolist(), "n_sub": n_sub,
           "threshold_abs": threshold}

    def decide(null_adj, null_dir, label):
        res = {}
        for name, D, N in (("adjacency", D_adj, null_adj), ("directed", D_dir, null_dir)):
            mu = N.mean(0)
            sd = np.maximum(N.std(0, ddof=1), 1.0 / n_sub)     # a floor, so links never seen in the null count
            tmax = (np.abs(N - mu) / sd).reshape(len(N), -1).max(1)
            c = float(np.quantile(tmax, 1 - alpha_fw))
            cand = (np.abs(D - mu) / sd > c) & (np.abs(D) >= threshold)
            res[name] = {"fw_threshold_standardised": c, "candidates": cand.astype(int).tolist()}
        out[label] = res

    perm_adj, perm_dir = [], []
    for _ in range(n_perm):
        a, b = _difference(panel, rng.permutation(panel.group), n_sub, frac, rng, cfg)
        perm_adj.append(a)
        perm_dir.append(b)
    decide(np.array(perm_adj), np.array(perm_dir), "permutation_reference")
    if null_panels:
        gen_adj, gen_dir = [], []
        for p in null_panels:
            a, b = _difference(p, p.group, n_sub, frac, rng, cfg)
            gen_adj.append(a)
            gen_dir.append(b)
        decide(np.array(gen_adj), np.array(gen_dir), "definition_matched_reference")
    return out

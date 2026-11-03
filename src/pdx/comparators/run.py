"""Wrappers that run the comparators implemented in this package on a Panel (protocol v0.5, §8.4).

Each returns the links it would report as differing between the phenotypes, in the [to, from]
convention of the engine, so the benchmark can score them against the true speed-adjusted
classes. Self-lags (persistence) are reported separately from cross-links.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from ..data import Panel
from ..multiplicity import bh
from ..step1 import Step1Config, lagged_adjacency, run_fci
from . import dynotears as DY
from . import ts_ldiffpc as TS


def _pairs(panel: Panel, g: int | None = None, max_dt: float = 1.5):
    xs, x1s, who = [], [], []
    for i in range(panel.n):
        if g is not None and panel.group[i] != g:
            continue
        yi, ti = panel.y[i], panel.t[i]
        for k in range(len(yi) - 1):
            if ti[k + 1] - ti[k] <= max_dt and not (np.isnan(yi[k]).any() or np.isnan(yi[k + 1]).any()):
                xs.append(yi[k])
                x1s.append(yi[k + 1])
                who.append(i)
    return np.array(xs), np.array(x1s), np.array(who, int)


def _standardiser(panel: Panel):
    X, X1, _ = _pairs(panel)
    allv = np.vstack([X, X1])
    mu, sd = allv.mean(0), allv.std(0)
    return lambda A: (A - mu) / np.where(sd > 0, sd, 1.0)


def _split(M: np.ndarray) -> dict:
    off = ~np.eye(len(M), dtype=bool)
    return {"cross_links": (M & off).astype(int).tolist(), "self_lags": np.diag(M).astype(int).tolist()}


# ---------------------------------------------------------------- C12 DYNOTEARS

def c12_dynotears(panel: Panel, lam: float = 0.05, tau_w: float = 0.3, tau_a: float = 0.1) -> dict:
    """One graph per phenotype from the standardised two-slice tables (the paper's large-sample
    settings), compared link by link."""
    std = _standardiser(panel)
    lagged, same = [], []
    for g in range(2):
        X, X1, _ = _pairs(panel, g)
        W, A, _ = DY.dynotears(std(X1), std(X), lam, lam, tau_w, tau_a)
        lagged.append(A.T != 0)          # A[from, to] -> [to, from]
        same.append(W.T != 0)
    disc = lagged[0] != lagged[1]
    return {**_split(disc), "same_time_discordant": int((same[0] != same[1]).sum())}


# ---------------------------------------------------------------- C5 tsLDiffPC / tsDCI

def c5_tsldiffpc(panel: Panel, alpha: float = 0.05, max_s: int | None = 3, variant: str = "tsdci") -> dict:
    """Difference graph between the phenotypes' two-slice tables (re-implementation validated in
    the tsLDiffPC/tsDCI note); root causes are the current-block heads of retained edges."""
    std = _standardiser(panel)
    Z = []
    for g in range(2):
        X, X1, _ = _pairs(panel, g)
        Z.append(np.column_stack([std(X), std(X1)]))
    d = panel.d
    edges, sep = TS.skeleton_with_sepsets(Z[0], Z[1], d, alpha, max_s, variant)
    directed, undirected = TS.orient(edges, sep, Z[0], Z[1], d, alpha, variant)
    lag = np.zeros((d, d), bool)
    for a, b in directed:
        if a < d <= b:
            lag[b - d, a] = True
    for e in edges:
        a, b = sorted(e)
        if a < d <= b:
            lag[b - d, a] = True
    heads, pairs = TS.root_causes(directed, undirected, d)
    return {**_split(lag), "root_causes": sorted(int(v) for v in heads),
            "root_cause_pairs": sorted(int(v) for v in pairs)}


# ---------------------------------------------------------------- C3 NCT-style permutation test

def _yearly(X, X1):
    Z = np.column_stack([np.ones(len(X)), X])
    C, *_ = np.linalg.lstsq(Z, X1, rcond=None)
    return C[1:].T                        # [to, from]


def c3_nct(panel: Panel, n_perm: int = 2000, rng: np.random.Generator | None = None, alpha: float = 0.05) -> dict:
    """Network comparison test on the yearly (temporal) networks: global strength and maximum
    edge difference, labels permuted by participant. Per-edge decisions use the permutation
    distribution of the maximum standardised edge difference (Westfall & Young), which, unlike
    Holm on raw permutation p-values, can reject at ten domains."""
    rng = rng or np.random.default_rng()
    X, X1, who = _pairs(panel)
    g_of = panel.group[who]
    d = panel.d
    off = ~np.eye(d, dtype=bool)

    def stats_for(lab):
        B0 = _yearly(X[lab == 0], X1[lab == 0])
        B1 = _yearly(X[lab == 1], X1[lab == 1])
        D = B0 - B1
        return abs(np.abs(B0[off]).sum() - np.abs(B1[off]).sum()), np.abs(D[off]).max(), np.abs(D)

    s_obs, m_obs, e_obs = stats_for(g_of)
    groups = panel.group.copy()
    cs, cm, E = 0, 0, []
    for _ in range(n_perm):
        perm = rng.permutation(groups)[who]
        s, m, e = stats_for(perm)
        cs += s >= s_obs
        cm += m >= m_obs
        E.append(e)
    E = np.array(E)
    pe = ((E >= e_obs).sum(0) + 1) / (n_perm + 1)
    sd = np.maximum(E.std(0, ddof=1), 1e-12)
    tmax = (E / sd).reshape(n_perm, -1).max(1)
    p_fw = ((tmax[:, None, None] >= (e_obs / sd)[None]).sum(0) + 1) / (n_perm + 1)
    rej = p_fw <= alpha
    return {"p_global_strength": float((cs + 1) / (n_perm + 1)), "p_max_edge": float((cm + 1) / (n_perm + 1)),
            **_split(rej), "p_edges": pe.tolist(), "p_edges_familywise": p_fw.tolist()}


# ---------------------------------------------------------------- C11 naive separate graphs and v0.3 interactions

def c11_naive(panel: Panel, cfg: Step1Config | None = None, alpha: float = 0.05) -> dict:
    """Separate graphs read naively (one fast Step 1 fit per phenotype, discordant lagged
    adjacencies), and v0.3's yearly link-wise interaction tests (pooled regression of each domain
    on the year-t domains, phenotype and phenotype x source; participant-clustered errors; BH)."""
    cfg = cfg or Step1Config()
    d = panel.d
    adj = []
    for g in range(2):
        sub = panel.subset(np.where(panel.group == g)[0])
        G, cols = run_fci(sub, cfg)[:2]
        adj.append(lagged_adjacency(G, cols, d))
    X, X1, who = _pairs(panel)
    grp = panel.group[who].astype(float)
    _, cid = np.unique(who, return_inverse=True)
    nc = cid.max() + 1
    p = np.ones((d, d))
    for j in range(d):
        for k in range(d):
            Z = np.column_stack([np.ones(len(X)), X, grp, grp * X[:, k]])
            ZtZi = np.linalg.pinv(Z.T @ Z)
            beta = ZtZi @ Z.T @ X1[:, j]
            u = X1[:, j] - Z @ beta
            sc = np.zeros((nc, Z.shape[1]))
            np.add.at(sc, cid, Z * u[:, None])
            V = ZtZi @ (sc.T @ sc) @ ZtZi * nc / max(nc - 1, 1)
            p[j, k] = 2 * stats.norm.sf(abs(beta[-1]) / np.sqrt(max(V[-1, -1], 1e-300)))
    rej = bh(p.ravel(), alpha).reshape(d, d)
    return {"separate_graphs": _split(adj[0] != adj[1]), "interaction_tests": _split(rej),
            "p_interaction": p.tolist()}

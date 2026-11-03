"""Step 1: one shared causal map from all participants (protocol v0.5, §7.2), and the support of
Steps 2 to 5 (§6.3).

The learner is FCI with tiered background knowledge (causal-learn; Spirtes 2000; Zhang 2008),
run on the pooled two-slice table. The phenotype labels are not in the graph. Tiers, in order:
era | other baseline variables | year t | treatment at t+1 | the rest of year t+1; nothing but
era causes age at onset or sex. Tiered knowledge only says that a later node cannot be an
ancestor of an earlier one, so it puts an arrowhead at the later node and leaves the mark at the
earlier node to the data (causal-learn 0.1.4.8 sets a tail there instead, which would hide
possible hidden confounding; fci_graph replaces that step).

Tests, both with test-wise deletion of missing values (§7.2):
- "fisherz": the uncorrected Fisher-z test on the normal scores, the fast configuration used in
  every bootstrap replicate, subsample and simulated dataset;
- "cluster_fisherz": the partial correlation tested with a participant-clustered sandwich
  variance, which also allows unequal residual variance (one of the WP3 candidates, D9).
Any other causal-learn test name runs on complete rows.

A drift entry X -> Y enters the support when the map has an adjacency from X in year t to Y in
year t+1 (any marks, so links that may reflect hidden confounding stay in), or when the expert
map proposes it. Same-year adjacencies add no drift entry. When the interval rule of §6.3 calls
for it, the interval is partialled out of every column, which conditions every test on it.
"""
from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass

import numpy as np
from scipy import stats

from .data import Panel, two_slice_table

MARK = {-1: "-", 1: ">", 2: "o"}
_REGISTERED = False
_TESTS = {"fisherz": "pdx_fisherz", "cluster_fisherz": "pdx_cluster_fisherz"}


@dataclass
class Step1Config:
    alpha: float = 0.05
    depth: int = 3                    # separating-set cap (-1: none)
    test: str = "fisherz"
    era_col: int | None = 0           # column of z holding era: its own first tier
    exogenous_cols: tuple = (1, 2)    # columns of z that nothing but era causes (age at onset, sex)
    add_interval: bool = False        # condition every test on the interval (§6.3 rule)
    include_dose: bool = False        # dose at t (year-t tier) and at t+1 (its own tier)
    max_dt: float = 1.5               # used only when the panel has no grid years
    max_path_length: int = -1


def _register_tests() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    from causallearn.utils.cit import CIT_Base, register_ci_test

    class PartialCorrelation(CIT_Base):
        """Partial correlation with test-wise deletion: Fisher-z, or (robust=True) a Wald test
        with a participant-clustered, heteroscedasticity-robust variance."""
        robust = False
        label = "pdx_fisherz"

        def __init__(self, data, clusters=None, **kwargs):
            super().__init__(data, **kwargs)
            self.check_cache_method_consistent(self.label, "test-wise deletion")
            self.assert_input_data_is_valid(allow_nan=True)
            cl = np.arange(len(data)) if clusters is None else np.asarray(clusters)
            _, self.cid = np.unique(cl, return_inverse=True)
            self.obs = ~np.isnan(data)
            mu, sd = np.nanmean(data, 0), np.nanstd(data, 0)
            self.Z = np.nan_to_num((data - mu) / np.where(sd > 0, sd, 1.0))

        def __call__(self, X, Y, condition_set=None):
            Xs, Ys, S, key = self.get_formatted_XYZ_and_cachekey(X, Y, condition_set)
            if key in self.pvalue_cache:
                return self.pvalue_cache[key]
            rows = self.obs[:, [Xs[0], Ys[0]] + list(S)].all(1)
            n = int(rows.sum())
            p = 1.0
            if n > len(S) + 3:
                xy = self.Z[np.ix_(rows, [Xs[0], Ys[0]])]
                if S:
                    C = np.column_stack([np.ones(n), self.Z[np.ix_(rows, list(S))]])
                    coef, *_ = np.linalg.lstsq(C, xy, rcond=None)
                    xy = xy - C @ coef
                ex, ey = xy[:, 0] - xy[:, 0].mean(), xy[:, 1] - xy[:, 1].mean()
                sxx, syy = float(ex @ ex), float(ey @ ey)
                if sxx > 0 and syy > 0 and not self.robust:
                    r = float(np.clip(ex @ ey / np.sqrt(sxx * syy), -1 + 1e-12, 1 - 1e-12))
                    p = float(2 * stats.norm.sf(np.sqrt(n - len(S) - 3) * abs(np.arctanh(r))))
                elif sxx > 0 and syy > 0:
                    _, cid = np.unique(self.cid[rows], return_inverse=True)
                    ncl = int(cid.max()) + 1
                    beta = float(ex @ ey) / sxx
                    score = np.bincount(cid, weights=ex * (ey - beta * ex), minlength=ncl)
                    V = float(score @ score) / sxx ** 2 * ncl / max(ncl - 1, 1)
                    p = float(2 * stats.norm.sf(abs(beta) / np.sqrt(V))) if V > 0 else 1.0
            self.pvalue_cache[key] = p
            return p

    class ClusterPartialCorrelation(PartialCorrelation):
        robust = True
        label = "pdx_cluster_fisherz"

    register_ci_test("pdx_fisherz", PartialCorrelation)
    register_ci_test("pdx_cluster_fisherz", ClusterPartialCorrelation)
    _REGISTERED = True


def _orient_tiers(bk, graph) -> None:
    """Replacement for causal-learn's fci_orient_bk: a forbidden X -> Y means X is not an
    ancestor of Y, which in a PAG is an arrowhead at X; the other mark stays as it is."""
    if bk is None:
        return
    from causallearn.graph.Edge import Edge
    from causallearn.graph.Endpoint import Endpoint
    for edge in graph.get_graph_edges():
        n1, n2 = edge.get_node1(), edge.get_node2()
        if bk.is_forbidden(n2, n1):
            e1 = edge.get_proximal_endpoint(n1)
            graph.remove_edge(edge)
            graph.add_edge(Edge(n1, n2, e1, Endpoint.ARROW))
        elif bk.is_forbidden(n1, n2):
            e2 = edge.get_proximal_endpoint(n2)
            graph.remove_edge(edge)
            graph.add_edge(Edge(n1, n2, Endpoint.ARROW, e2))


def fci_graph(table: np.ndarray, tiers, alpha: float = 0.05, depth: int = -1, test: str = "fisherz",
              clusters: np.ndarray | None = None, exogenous=(), exempt=(), max_path_length: int = -1) -> np.ndarray:
    """FCI with tiers; returns causal-learn's endpoint matrix: G[i, j] is the mark at node i on the
    edge i - j (-1 tail, 1 arrowhead, 2 circle; 0 no edge). Columns in `exogenous` can be caused
    by no column outside `exempt` (themselves and era)."""
    import causallearn.search.ConstraintBased.FCI as FCI
    from causallearn.graph.GraphNode import GraphNode
    from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

    _register_tests()
    name = _TESTS.get(test, test)
    kw = {"clusters": clusters} if name in _TESTS.values() else {}
    if name not in _TESTS.values() and np.isnan(table).any():
        table = table[~np.isnan(table).any(1)]
    n = table.shape[1]
    nodes = [GraphNode(f"X{i + 1}") for i in range(n)]
    bk = BackgroundKnowledge()
    for i, t in enumerate(tiers):
        bk.add_node_to_tier(nodes[i], int(t))
    free = set(exogenous) | set(exempt)
    for e in exogenous:
        for j in range(n):
            if j not in free:
                bk.add_forbidden_by_node(nodes[j], nodes[e])
    original = FCI.fci_orient_bk
    FCI.fci_orient_bk = _orient_tiers
    try:
        with contextlib.redirect_stdout(io.StringIO()):   # causal-learn prints orientation steps regardless
            G, _ = FCI.fci(table, name, alpha, depth=depth, max_path_length=max_path_length,
                           background_knowledge=bk, verbose=False, show_progress=False, **kw)
    finally:
        FCI.fci_orient_bk = original
    return np.asarray(G.graph, int)


def step1_table(panel: Panel, cfg: Step1Config):
    """The two-slice table with the tiers of §6.2 (era first, then the other baseline variables).
    With add_interval the interval is partialled out of every other column and dropped, which
    conditions every test on it."""
    base = panel.z if panel.p else None
    X, tiers, who, cols = two_slice_table(panel, base, cfg.max_dt, cfg.add_interval, cfg.include_dose)
    tiers = tiers + 1
    if base is not None and cfg.era_col is not None:
        tiers[cfg.era_col] = 0
    if cfg.add_interval:
        j = cols.index("interval")
        iv = X[:, j]
        X = np.delete(X, j, 1)
        tiers = np.delete(tiers, j)
        cols = cols[:j] + cols[j + 1:]
        C = np.column_stack([np.ones(len(iv)), iv])
        for k in range(X.shape[1]):
            ok = ~np.isnan(X[:, k])
            if ok.sum() > 2:
                coef, *_ = np.linalg.lstsq(C[ok], X[ok, k], rcond=None)
                X[ok, k] = X[ok, k] - C[ok] @ coef + coef[0]
    exo = [c for c in cfg.exogenous_cols if base is not None and c < panel.p]
    return X, tiers, who, cols, exo


def _lagged_index(cols, d):
    return [cols.index(f"y{j}_t") for j in range(d)], [cols.index(f"y{j}_t1") for j in range(d)]


def lagged_adjacency(graph: np.ndarray, cols: list, d: int) -> np.ndarray:
    """(d, d) [to, from]: an adjacency between domain `from` in year t and `to` in year t+1."""
    it, it1 = _lagged_index(cols, d)
    return np.array([[graph[it[k], it1[j]] != 0 for k in range(d)] for j in range(d)])


def lagged_directed(graph: np.ndarray, cols: list, d: int) -> np.ndarray:
    """(d, d) [to, from]: a directed edge from year t to year t+1 (tail at t, arrowhead at t+1)."""
    it, it1 = _lagged_index(cols, d)
    return np.array([[graph[it[k], it1[j]] == -1 and graph[it1[j], it[k]] == 1 for k in range(d)]
                     for j in range(d)])


def run_fci(panel: Panel, cfg: Step1Config):
    X, tiers, who, cols, exo = step1_table(panel, cfg)
    exempt = [cfg.era_col] if (cfg.era_col is not None and panel.p) else []
    G = fci_graph(X, tiers, cfg.alpha, cfg.depth, cfg.test, clusters=who, exogenous=exo, exempt=exempt,
                  max_path_length=cfg.max_path_length)
    return G, cols, X.shape[0], len(np.unique(who)), int((~np.isnan(X).any(1)).sum())


def select_support(panel: Panel, expert: np.ndarray | None = None, cfg: Step1Config | None = None) -> np.ndarray:
    """The support of Steps 2 to 5: the expert map's links, plus the lagged adjacencies the fast
    configuration finds, plus every own rate."""
    cfg = cfg or Step1Config()
    G, cols = run_fci(panel, cfg)[:2]
    sup = lagged_adjacency(G, cols, panel.d)
    if expert is not None:
        sup = sup | np.asarray(expert, bool)
    return sup | np.eye(panel.d, dtype=bool)


def make_support_fn(expert: np.ndarray | None = None, cfg: Step1Config | None = None):
    """The selection rule re-applied in every bootstrap replicate of H-P (§7.3.1, step 3)."""
    cfg = cfg or Step1Config()
    return lambda panel: select_support(panel, expert, cfg)


def reported_map(panel: Panel, cfg: Step1Config | None = None, names: list | None = None) -> dict:
    """The reported map from the full fit, with its marks (aggregate output only). The default
    configuration is the clustered test with a separating-set cap of 4 (provisional; WP3 chooses
    at most three configurations per learner)."""
    cfg = cfg or Step1Config(test="cluster_fisherz", depth=4)
    G, cols, n_rows, n_part, n_complete = run_fci(panel, cfg)
    lag = lagged_adjacency(G, cols, panel.d)
    shown = [_rename(c, names) for c in cols] if names else cols
    edges = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if G[i, j] != 0:
                edges.append({"from": shown[i], "to": shown[j],
                              "edge": f"{MARK[G[i, j]].replace('>', '<')}-{MARK[G[j, i]]}"})
    return {"columns": shown, "edges": edges, "rows": int(n_rows), "complete_rows": n_complete,
            "participants": int(n_part), "lagged_adjacency": lag.astype(int).tolist(),
            "config": dict(cfg.__dict__)}


def _rename(col: str, names: list) -> str:
    if col.startswith("y") and "_" in col:
        j, slot = col[1:].split("_")
        return f"{names[int(j)]}_{slot}"
    return col


def sortability(table: np.ndarray, directed: np.ndarray, kind: str = "var") -> float:
    """Variance- or R^2-sortability (Reisach 2021, 2023) over the ancestor pairs of a set of
    directed edges between table columns (directed[i, j]: column i -> column j, the usual
    adjacency convention, unlike the [to, from] support matrices): the share of pairs whose score
    rises along the causal order (ties count one half). Complete rows only."""
    table = table[~np.isnan(table).any(1)]
    n = table.shape[1]
    R = np.asarray(directed, bool).copy()
    for k in range(n):
        R |= R[:, [k]] & R[[k], :]
    if kind == "var":
        score = table.var(0)
    else:
        score = np.empty(n)
        for j in range(n):
            Xo = np.column_stack([np.ones(len(table)), np.delete(table, j, 1)])
            coef, *_ = np.linalg.lstsq(Xo, table[:, j], rcond=None)
            res = table[:, j] - Xo @ coef
            score[j] = 1 - res.var() / table[:, j].var()
    pairs = np.argwhere(R & ~np.eye(n, dtype=bool))
    if len(pairs) == 0:
        return float("nan")
    a, b = score[pairs[:, 0]], score[pairs[:, 1]]
    return float(np.mean(np.where(a < b, 1.0, np.where(a == b, 0.5, 0.0))))

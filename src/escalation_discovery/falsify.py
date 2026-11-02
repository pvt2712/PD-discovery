"""falsify — test an expert DAG against data before any discovery runs (Arm B step 1, PROTOCOL.md §7.1).

Implied independences. For every pair of observed variables that the DAG separates even when the earlier years are
hidden (non-adjacent in the MAG of the DAG with one latent earlier year), one statement X _||_ Y | S, with S a
minimal separating set among the observed variables (networkx `find_minimal_d_separator`, restricted to observed
nodes). Pairs that only the hidden history connects carry no statement, so the hidden history cannot produce false
violations; the dagitty `localTests` idea (Ankan, Wortel & Textor 2021) on a latent-variable DAG.

Test: the cluster-robust GCM (citest.ClusterGCM: additive splines, cross-fitted over participants, CR1) on normal
scores, alpha 0.05; untestable statements are counted and left out. Statistic: the share of statements violated.
Null: the same DAG with its time-varying labels permuted within groups (doses among doses, the rest among the rest,
the same permutation in every year), so the null keeps the DAG's shape and timing and scrambles its clinical content
(the within-tier relabelling of ppmi-causal/expert.py); p = (1 + #null rates <= observed) / (1 + #null).
Edge support: each expert edge, in its year-t+1 pair, tested for dependence given the child's other observed parents.

In the pooled analysis the cohort context points into every node (saturated), so cohort differences are not tested
as if they were the expert's claims. A subset that fixes a context node removes it from the DAG (`drop`): a fixed
root node has no edges, so the subset's DAG is the DAG without it, not the DAG with it hidden.
"""
import itertools

import networkx as nx
import numpy as np
import pandas as pd

from . import expert as E
from . import truth as T
from .citest import ClusterGCM, copula


def saturated_context(spec):
    return tuple((c, n) for c in E.CONTEXT for n in list(spec.baseline) + list(spec.tv) + ["t"])


def statements(dag_hist: nx.DiGraph, observed):
    """(x, y, separating set) for every observed pair separable with the earlier years hidden."""
    O = set(observed)
    adj, sep, _ = T.mag_skeleton(dag_hist, observed)
    adjset = {frozenset(p) for p in adj}
    out = []
    for a, b in itertools.combinations(observed, 2):
        if frozenset((a, b)) in adjset:
            continue
        S = nx.find_minimal_d_separator(dag_hist, {a}, {b}, restricted=O - {a, b})
        out.append((a, b, tuple(sorted(S if S is not None else sep[(a, b)]))))
    return out


def relabel_maps(spec, n_perm, seed):
    """Within-group permutations of the time-varying labels (doses / the rest), the identity excluded."""
    rng = np.random.default_rng(seed)
    dose = [v for v in spec.tv if v in E.DOSE]
    rest = [v for v in spec.tv if v not in E.DOSE]
    maps = []
    while len(maps) < n_perm:
        mp = {}
        for group in (dose, rest):
            perm = list(rng.permutation(group))
            mp.update(dict(zip(group, perm)))
        if any(a != b for a, b in mp.items()):
            maps.append(mp)
    return maps


def relabel_dag(dag, mp):
    def ren(n):
        for sfx in ("_h1", "_0", "_1"):
            if n.endswith(sfx) and n[: -len(sfx)] in mp:
                return mp[n[: -len(sfx)]] + sfx
        if n.startswith("u_") and n[2:] in mp:
            return "u_" + mp[n[2:]]
        return n
    return nx.relabel_nodes(dag, {n: ren(n) for n in dag.nodes})


class Tester:
    """Cached clustered-GCM tests on one two-slice table."""

    def __init__(self, table: pd.DataFrame, names, seed=0):
        self.names = list(names)
        self.idx = {n: i for i, n in enumerate(self.names)}
        self.gcm = ClusterGCM(copula(table[self.names].to_numpy(dtype=float)), table.PATNO.to_numpy(), learner="splines",
                              cross_fit=5, seed=seed)

    def p(self, x, y, S=()):
        return self.gcm.test(self.idx[x], self.idx[y], [self.idx[s] for s in S])


def violation_rate(stmts, tester, alpha=0.05):
    ps = np.array([tester.p(x, y, S) for x, y, S in stmts], dtype=float)
    ok = ~np.isnan(ps)
    return (float(np.mean(ps[ok] < alpha)) if ok.any() else np.nan), int((~ok).sum()), ps


def test_expert(table: pd.DataFrame, spec: E.ExpertSpec | None = None, n_perm=100, alpha=0.05, seed=5, hidden=(),
                drop=()):
    """Falsification of the expert DAG on a pooled two-slice table. Returns (summary dict, statement table)."""
    spec = spec or E.load()
    ctx = saturated_context(spec)
    obs = [n for n in E.observed(spec, hidden=hidden) if n in table and n not in drop]
    dag = E.dynamic_dag(spec, L=1, person_effects=False, context_edges=ctx)
    dag.remove_nodes_from(drop)
    tester = Tester(table, obs, seed)
    stmts = statements(dag, obs)
    real, untestable, ps = violation_rate(stmts, tester, alpha)
    null = []
    for mp in relabel_maps(spec, n_perm, seed):
        st = statements(relabel_dag(dag, mp), obs)
        null.append(violation_rate(st, tester, alpha)[0])
    null = np.array(null, dtype=float)
    summary = dict(dag=spec.name, statements=len(stmts), untestable=untestable, alpha=alpha, violation_rate=round(real, 4),
                   null_mean=round(float(np.nanmean(null)), 4) if len(null) else np.nan,
                   null_5=round(float(np.nanpercentile(null, 5)), 4) if len(null) else np.nan,
                   null_95=round(float(np.nanpercentile(null, 95)), 4) if len(null) else np.nan,
                   p_value=round(float((1 + np.sum(null <= real)) / (1 + len(null))), 4) if len(null) else np.nan,
                   relabellings=len(null), rows=len(table), participants=int(table.PATNO.nunique()))
    st = pd.DataFrame([dict(x=x, y=y, given=" ".join(S), size=len(S), p=p, violated=bool(p < alpha) if not np.isnan(p) else None)
                       for (x, y, S), p in zip(stmts, ps)])
    return summary, st


def edge_support(table: pd.DataFrame, spec: E.ExpertSpec | None = None, alpha=0.05, seed=5, hidden=(), drop=()):
    """Each expert edge in its year-t+1 pair, tested given the child's other observed parents (and the child's own
    previous value, the time trend and the cohort context)."""
    spec = spec or E.load()
    obs = [n for n in E.observed(spec, hidden=hidden) if n in table and n not in drop]
    dag = E.dynamic_dag(spec, L=0, person_effects=False, context_edges=saturated_context(spec))
    dag.remove_nodes_from(drop)
    tester = Tester(table, obs, seed)
    rows = []
    for (a, b), kind in E.summary_pairs(spec):
        if a not in obs or b not in obs:
            continue
        others = [p for p in dag.predecessors(b) if p in obs and p != a]
        p = tester.p(a, b, others)
        rows.append(dict(source=a, target=b, kind=kind, grade=spec.grades.get((a.rsplit("_", 1)[0] if a[-2:] in ("_0", "_1") else a,
                                                                                b.rsplit("_", 1)[0]), ""),
                         conditioning=len(others), p=p, supported=bool(p < alpha) if not np.isnan(p) else None))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------- the core map on the decision-time rows
def core_map_dag(with_latents=True):
    """v2's core map as a DAG over its node ids; without latents, the latent nodes and their arrows are dropped (to
    count the implications the latent nodes remove)."""
    from . import coremap
    nd = coremap.nodes()
    g = nx.DiGraph()
    g.add_nodes_from(n for n, v in nd.items() if with_latents or v["measured"])
    for a in coremap.arrows():
        if with_latents or (nd[a["from"]]["measured"] and nd[a["to"]]["measured"]):
            g.add_edge(a["from"], a["to"])
    return g


def core_statements(g, measured):
    """Implied independences among the measured nodes with the latent nodes hidden: (x, y, minimal separator)."""
    adj, sep, _ = T.mag_skeleton(g, measured)
    adjset = {frozenset(p) for p in adj}
    out = []
    for a, b in itertools.combinations(measured, 2):
        if frozenset((a, b)) in adjset:
            continue
        S = nx.find_minimal_d_separator(g, {a}, {b}, restricted=set(measured) - {a, b})
        out.append((a, b, tuple(sorted(S if S is not None else sep[(a, b)]))))
    return out


def core_relabellings():
    """Every relabelling that permutes triggers among triggers and brakes among brakes (the identity excluded)."""
    from . import coremap
    nd = coremap.nodes()
    trig = [n for n, v in nd.items() if v["role"] == "trigger" and v["measured"]]
    brake = [n for n, v in nd.items() if v["role"] == "brake" and v["measured"]]
    out = []
    for pt in itertools.permutations(trig):
        for pb in itertools.permutations(brake):
            mp = dict(zip(trig, pt)) | dict(zip(brake, pb))
            if any(a != b for a, b in mp.items()):
                out.append(mp)
    return out


def block_p(gcm, cols_x, cols_y, cols_s):
    """Block test of node X (all its columns) against node Y: one block test per column of Y, Bonferroni over them."""
    ps = []
    for y in cols_y:
        r = gcm.block_test(cols_x, y, cols_s)
        ps.append(np.nan if r is None else r["p"])
    ps = np.array(ps, dtype=float)
    return float(min(1.0, np.nanmin(ps) * len(ps))) if np.isfinite(ps).any() else np.nan


def test_core_map(d: pd.DataFrame, alpha=0.05, seed=5):
    """The core map's implied independences on one cohort's decision-time rows (v2 DESIGN.md §6.2), against every
    trigger/brake relabelling (exact p). `d` is the trial table with PATNO, the arm column E and the map's columns."""
    from . import coremap
    nd = coremap.nodes()
    measured = [n for n, v in nd.items() if v["measured"]]
    cols = {n: [c for c in nd[n]["columns"] if c in d] for n in measured}
    measured = [n for n in measured if cols[n]]
    allc = sorted({c for n in measured for c in cols[n]})
    ci = {c: i for i, c in enumerate(allc)}
    gcm = ClusterGCM(copula(d[allc].to_numpy(dtype=float)), d.PATNO.to_numpy(), learner="splines", cross_fit=5, seed=seed)
    idx = lambda n: [ci[c] for c in cols[n]]

    def rate(stmts):
        ps = np.array([block_p(gcm, idx(x), idx(y), [i for s in S for i in idx(s)]) for x, y, S in stmts], dtype=float)
        ok = ~np.isnan(ps)
        return (float(np.mean(ps[ok] < alpha)) if ok.any() else np.nan), ps

    g = core_map_dag(True)
    stmts = core_statements(g, measured)
    real, ps = rate(stmts)
    null = np.array([rate(core_statements(nx.relabel_nodes(g, mp), measured))[0] for mp in core_relabellings()])
    n_without = len(core_statements(core_map_dag(False), measured))
    summary = dict(statements=len(stmts), statements_without_latents=n_without, violation_rate=real,
                   null_mean=float(np.nanmean(null)) if len(null) else np.nan, relabellings=len(null),
                   p_value=float((1 + np.sum(null <= real)) / (1 + len(null))) if len(null) else np.nan,
                   rows=len(d), participants=int(d.PATNO.nunique()))
    table = pd.DataFrame([dict(x=x, y=y, given=" ".join(S), p=p) for (x, y, S), p in zip(stmts, ps)])
    return summary, table


def core_map_main(argv=None):
    """uv run escd core-map: the core map's test in each cohort's decision-time rows (gated)."""
    import json
    from . import config as C
    C.require_ppmi_approval("the core-map falsification")
    out = C.results_dir("core_map")
    rows = []
    for cohort in C.COHORTS:
        d = pd.read_csv(C.derived_dir() / f"trials_dt_{cohort}.csv", low_memory=False)
        from .trials import dt_strategy
        arm = dt_strategy(d)
        d = d.assign(E=arm.map({1: 1.0, 0: 0.0}).astype(float))
        summ, table = test_core_map(d)
        table.to_csv(out / f"statements_{cohort}.csv", index=False)
        rows.append(dict(cohort=cohort, **summ))
        print(cohort, json.dumps(summ))
    pd.DataFrame(rows).to_csv(out / "core_map_falsification.csv", index=False)

"""Skeleton of tsLDiffPC and tsDCI (Ruer, Loranchet, Bystrova & Assaad 2026, arXiv:2608.21117),
re-implemented from the authors' pyCIPHOD code (MIT licence; read, not run).

Data are two-slice tables: columns 0..p-1 are the lag block X(t-1), p..2p-1 the current block X(t).
- Candidate edges: lag -> current (p^2, self-lags included) and current - current (p(p-1)/2).
- An edge is removed as soon as some conditioning set S of the other window nodes (all of them,
  lag and current, by increasing size up to max_s) gives p > alpha in a two-sample Wald z-test of
  the OLS coefficient (intercept included, plain OLS standard errors), in either regression
  direction (y on x + S, or x on y + S). No multiplicity correction.
- The admissible sets do not depend on the current graph, so each candidate edge is decided on
  its own, and tsLDiffPC and tsDCI return the same skeleton. They differ only in how same-time
  edges are oriented (colliders and Meek rules, or residual-variance invariance). Lagged edges are
  oriented by time in both, so the root causes found through lagged edges are the current-block
  targets of retained lagged edges.
"""
from itertools import combinations

import numpy as np
from scipy import stats


def gram(Z):
    W = np.column_stack([np.ones(len(Z)), Z])
    return W.T @ W, len(Z)


def coef_se(G, n, y, x, S):
    """OLS of column y on [1, x, S] from the Gram matrix of [1, Z]; column indices refer to Z."""
    idx = [0, 1 + x] + [1 + v for v in S]
    A = G[np.ix_(idx, idx)]
    b = G[idx, 1 + y]
    try:
        Ai = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        Ai = np.linalg.pinv(A)
    beta = Ai @ b
    s2 = (G[1 + y, 1 + y] - beta @ b) / (n - len(idx))
    return beta[1], np.sqrt(max(s2 * Ai[1, 1], 0.0))


def wald_p(g1, g2, y, x, S):
    """Two-sample Wald p-value; NaN when a fit fails (pyCIPHOD skips failed tests)."""
    try:
        b1, s1 = coef_se(*g1, y, x, S)
        b2, s2 = coef_se(*g2, y, x, S)
    except (np.linalg.LinAlgError, ValueError):
        return np.nan
    den = np.hypot(s1, s2)
    return 1.0 if den == 0 else 2 * stats.norm.sf(abs(b1 - b2) / den)


def skeleton(Z1, Z2, p, alpha=0.05, max_s=None):
    """Return {(x, y): retained} for every candidate edge, plus the number of tests run."""
    g1, g2 = gram(Z1), gram(Z2)
    nodes = list(range(2 * p))
    cands = [(k, p + j) for k in range(p) for j in range(p)]
    cands += [(p + j, p + m) for j in range(p) for m in range(j + 1, p)]
    max_s = 2 * p - 2 if max_s is None else max_s
    kept, ntest = {}, 0
    for x, y in cands:
        others = [v for v in nodes if v not in (x, y)]
        removed = False
        for s in range(max_s + 1):
            for S in combinations(others, s):
                for a, b in ((x, y), (y, x)):
                    ntest += 1
                    if wald_p(g1, g2, b, a, S) > alpha:
                        removed = True
                        break
                if removed:
                    break
            if removed:
                break
        kept[(x, y)] = not removed
    return kept, ntest


def lag_matrix(kept, p):
    """F[j, k] = True if the lagged difference edge k(t-1) -> j(t) is retained."""
    F = np.zeros((p, p), bool)
    for k in range(p):
        for j in range(p):
            F[j, k] = kept[(k, p + j)]
    return F


def contemporaneous_pairs(kept, p):
    return [(j, m) for j in range(p) for m in range(j + 1, p) if kept[(p + j, p + m)]]


# ---------------------------------------------------------------- full algorithms (orientation and
# root causes), for the validation against the paper's own benchmark. Nodes are ordered as in
# pyCIPHOD (by name, then time): lag block first within each series. Undirected-edge membership is
# treated as symmetric, which is the paper's intended semantics.

def _order(p):
    """Window nodes sorted as pyCIPHOD sorts DTimeVar(name, time): by series, then lag before current."""
    return sorted(range(2 * p), key=lambda v: (v % p, v // p))


def skeleton_with_sepsets(Z1, Z2, p, alpha=0.05, max_s=None, variant="tsdci"):
    """Skeleton plus the separating set recorded at removal, following each algorithm's loop order."""
    g1, g2 = gram(Z1), gram(Z2)
    order = _order(p)
    rank = {v: r for r, v in enumerate(order)}
    is_cand = lambda a, b: (a >= p or b >= p)  # at least one endpoint in the current block
    cands = [tuple(sorted(e, key=rank.get)) for e in combinations(order, 2) if is_cand(*e)]
    max_s = 2 * p - 2 if max_s is None else max_s
    adj = {v: set() for v in order}
    for a, b in cands:
        adj[a].add(b)
        adj[b].add(a)
    sepset = {}
    if variant == "tsdci":
        for x, y in cands:
            others = [v for v in order if v not in (x, y)]
            done = False
            for s in range(max_s + 1):
                for S in combinations(others, s):
                    for a, b in ((x, y), (y, x)):
                        if wald_p(g1, g2, b, a, S) > alpha:
                            adj[x].discard(y)
                            adj[y].discard(x)
                            sepset[(x, y)] = sepset[(y, x)] = S
                            done = True
                            break
                    if done:
                        break
                if done:
                    break
    else:  # tsldiffpc: level by level, adjacency snapshot per level, visits from both endpoints
        for s in range(max_s + 1):
            snap = {v: sorted(adj[v], key=rank.get) for v in order}
            for x in order:
                for y in snap[x]:
                    if not is_cand(x, y):
                        continue
                    others = [v for v in order if v not in (x, y)]
                    for S in combinations(others, s):
                        if wald_p(g1, g2, y, x, S) > alpha:
                            adj[x].discard(y)
                            adj[y].discard(x)
                            sepset[(x, y)] = sepset[(y, x)] = S
                            break
    edges = {frozenset((a, b)) for a in order for b in adj[a] if rank[a] < rank[b]}
    return edges, sepset


def resid_var(G, n, y, S):
    idx = [0] + [1 + v for v in S]
    A = G[np.ix_(idx, idx)]
    b = G[idx, 1 + y]
    beta = np.linalg.pinv(A) @ b  # pseudo-inverse, as in pyCIPHOD's residual-variance helper
    return (G[1 + y, 1 + y] - beta @ b) / (n - len(idx)), n - len(idx)


def orient(edges, sepset, Z1, Z2, p, alpha=0.05, variant="tsdci"):
    """Return (directed set of (tail, head), undirected set of frozensets)."""
    order = _order(p)
    rank = {v: r for r, v in enumerate(order)}
    directed, undirected = set(), set()
    for e in edges:
        a, b = sorted(e, key=rank.get)
        if (a < p) != (b < p):  # lagged edge: oriented by time
            directed.add((a, b) if a < p else (b, a))
        else:
            undirected.add(e)
    if variant == "tsdci":
        g1, g2 = gram(Z1), gram(Z2)

        def var_p(t, S):
            try:
                v1, d1 = resid_var(*g1, t, S)
                v2, d2 = resid_var(*g2, t, S)
            except (np.linalg.LinAlgError, ValueError):
                return None
            if d1 <= 0 or d2 <= 0 or v2 <= 0:
                return None
            pl = stats.f.cdf(v1 / v2, d1, d2)
            return 2 * min(pl, 1 - pl)

        def powerset(nodes):
            for s in range(len(nodes) + 1):
                yield from combinations(nodes, s)

        for e in sorted(list(undirected), key=lambda e: sorted(rank[v] for v in e)):
            i, j = sorted(e, key=rank.get)
            not_i = [v for v in order if v != i]
            not_j = [v for v in order if v != j]
            for m_i, m_j in zip(powerset(not_i), powerset(not_j)):
                pi = var_p(i, m_i)
                if pi is not None and pi > alpha:
                    src, tgt = (j, i) if j in m_i else (i, j)
                    if e in undirected:
                        undirected.discard(e)
                        directed.add((src, tgt))
                    break
                pj = var_p(j, m_j)
                if pj is not None and pj > alpha:
                    src, tgt = (i, j) if i in m_j else (j, i)
                    if e in undirected:
                        undirected.discard(e)
                        directed.add((src, tgt))
                    break
        return directed, undirected

    # tsldiffpc: unshielded colliders into current-block nodes, then time-restricted Meek rules
    adj = {v: set() for v in order}
    for e in edges:
        a, b = tuple(e)
        adj[a].add(b)
        adj[b].add(a)

    def set_dir(x, y):
        undirected.discard(frozenset((x, y)))
        directed.discard((y, x))
        directed.add((x, y))

    for x in order:
        for y in sorted(adj[x], key=rank.get):
            if y < p:
                continue
            for z in sorted(adj[y], key=rank.get):
                if z < p or z == x or z in adj[x]:
                    continue
                if y not in sepset.get((x, z), ()):
                    set_dir(x, y)
                    set_dir(z, y)
    same = lambda a, b: (a < p) == (b < p)
    und = lambda a, b: frozenset((a, b)) in undirected
    changed = True
    while changed:
        changed = False
        for x in order:
            for y in adj[x]:
                for z in adj[y] - adj[x] - {x}:
                    if same(y, z) and (x, y) in directed and und(y, z):  # rule 1
                        set_dir(y, z)
                        changed = True
                for z in adj[y] & adj[x]:
                    if same(x, z) and (x, y) in directed and (y, z) in directed and und(x, z):  # rule 2
                        set_dir(x, z)
                        changed = True
        for x in order:
            for y in adj[x]:
                if (x, y) not in directed:
                    continue
                for z in adj[y] - adj[x] - {x}:
                    if (z, y) not in directed:
                        continue
                    for w in adj[x] & adj[y] & adj[z]:
                        if same(w, y) and und(w, y) and und(w, x) and und(w, z):  # rule 3
                            set_dir(w, y)
                            changed = True
        for x in order:
            for y in adj[x]:
                if not und(x, y):
                    continue
                for w in adj[x] & adj[y]:
                    if (x, w) not in directed or not und(y, w):
                        continue
                    for z in adj[y] & adj[w]:
                        if same(y, z) and und(y, z) and (w, z) in directed:  # rule 4
                            set_dir(y, z)
                            changed = True
    return directed, undirected


def root_causes(directed, undirected, p):
    """C-hat: series heading a directed edge into the current block; pairs from undirected edges."""
    heads = {b % p for a, b in directed if b >= p}
    pairs = {v % p for e in undirected for v in e}
    return heads, pairs

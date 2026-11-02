"""graphs — one representation for every learner's output: a mark graph over named nodes.

An edge between a and b carries the mark at a and the mark at b: 'T' tail, 'A' arrowhead, 'C' circle (undetermined).
    a -> b  = (T, A)      a <-> b = (A, A)      a o-> b = (C, A)      a o-o b = (C, C)
A DAG has only (T, A); a CPDAG's undirected edge is stored as (C, C) (either orientation is possible); a PAG uses all
marks; (T, T) appears only when FCI's selection-bias rules fire. Edges are stored once, keyed by the pair in node order.
"""
import itertools
import json

import numpy as np

MARK = {-1: "T", 1: "A", 2: "C"}          # causal-learn endpoint codes
GLYPH = {("T", "A"): "-->", ("A", "T"): "<--", ("A", "A"): "<->", ("C", "A"): "o->", ("A", "C"): "<-o",
         ("C", "C"): "o-o", ("T", "T"): "---", ("C", "T"): "o--", ("T", "C"): "--o"}


class MarkGraph:
    def __init__(self, nodes, kind: str = "PAG"):
        self.nodes = list(nodes)
        self.index = {n: i for i, n in enumerate(self.nodes)}
        self.kind = kind
        self.edges = {}                     # (a, b) with index[a] < index[b] -> (mark at a, mark at b)

    # ---------------------------------------------------------------- construction
    def _key(self, a, b):
        return (a, b, False) if self.index[a] < self.index[b] else (b, a, True)

    def add(self, a, b, mark_a, mark_b):
        if a == b:
            raise ValueError(f"self-loop on {a}")
        x, y, flip = self._key(a, b)
        self.edges[(x, y)] = (mark_b, mark_a) if flip else (mark_a, mark_b)

    def directed(self, a, b):
        self.add(a, b, "T", "A")

    def remove(self, a, b):
        x, y, _ = self._key(a, b)
        self.edges.pop((x, y), None)

    def copy(self):
        g = MarkGraph(self.nodes, self.kind)
        g.edges = dict(self.edges)
        return g

    def restrict(self, nodes):
        keep = [n for n in self.nodes if n in set(nodes)]
        g = MarkGraph(keep, self.kind)
        for (a, b), m in self.edges.items():
            if a in g.index and b in g.index:
                g.add(a, b, *m)
        return g

    # ---------------------------------------------------------------- queries
    def marks(self, a, b):
        """(mark at a, mark at b), or None when a and b are not adjacent."""
        x, y, flip = self._key(a, b)
        m = self.edges.get((x, y))
        if m is None:
            return None
        return (m[1], m[0]) if flip else m

    def adjacent(self, a, b) -> bool:
        return self.marks(a, b) is not None

    def adjacencies(self) -> set:
        return {frozenset(k) for k in self.edges}

    def neighbours(self, n):
        return [m for m in self.nodes if m != n and self.adjacent(n, m)]

    def parents(self, n):
        """Definite parents: w -> n."""
        return [w for w in self.neighbours(n) if self.marks(w, n) == ("T", "A")]

    def possible_parents(self, n):
        """w -> n, w o-> n, w o-o n (CPDAG undirected edges included) and w --- n: no mark rules w out as a cause of n."""
        return [w for w in self.neighbours(n) if self.marks(w, n) in (("T", "A"), ("C", "A"), ("C", "C"), ("T", "T"))]

    def n_edges(self):
        return len(self.edges)

    # ---------------------------------------------------------------- conversions
    def to_json(self):
        return dict(kind=self.kind, nodes=self.nodes,
                    edges=[[a, b, ma, mb] for (a, b), (ma, mb) in sorted(self.edges.items())])

    @classmethod
    def from_json(cls, d):
        g = cls(d["nodes"], d.get("kind", "PAG"))
        for a, b, ma, mb in d["edges"]:
            g.add(a, b, ma, mb)
        return g

    def dumps(self):
        return json.dumps(self.to_json())

    def table(self):
        return [dict(a=a, b=b, edge=GLYPH.get(m, "".join(m)), mark_a=m[0], mark_b=m[1]) for (a, b), m in sorted(self.edges.items())]

    def __repr__(self):
        return f"MarkGraph({self.kind}, {len(self.nodes)} nodes, {len(self.edges)} edges)"


def from_causallearn(G, names, kind="PAG") -> MarkGraph:
    """causal-learn GeneralGraph (graph[i, j] = mark at i on edge i-j) -> MarkGraph. For a CPDAG (PC, BOSS, GES) an
    undirected edge (-1, -1) means either orientation: it is stored as (C, C)."""
    M = np.asarray(G.graph if hasattr(G, "graph") else G)
    g = MarkGraph(names, kind)
    for i, j in itertools.combinations(range(len(names)), 2):
        if M[i, j] != 0 or M[j, i] != 0:
            mi, mj = MARK.get(int(M[i, j]), "C"), MARK.get(int(M[j, i]), "C")
            if kind == "CPDAG" and (mi, mj) == ("T", "T"):
                mi, mj = "C", "C"
            g.add(names[i], names[j], mi, mj)
    return g


def from_weighted(W, names, threshold=0.0) -> MarkGraph:
    """Weighted adjacency W[i, j] (i -> j) -> DAG. An entry counts if |W| > threshold; a 2-cycle (both directions above
    the threshold, which acyclic learners should not return) keeps the larger weight."""
    W = np.asarray(W, dtype=float)
    A = np.abs(W) > threshold
    g = MarkGraph(names, "DAG")
    for i, j in itertools.combinations(range(len(names)), 2):
        if A[i, j] and A[j, i]:
            if abs(W[i, j]) >= abs(W[j, i]):
                g.directed(names[i], names[j])
            else:
                g.directed(names[j], names[i])
        elif A[i, j]:
            g.directed(names[i], names[j])
        elif A[j, i]:
            g.directed(names[j], names[i])
    return g


def from_nx(dag, names=None) -> MarkGraph:
    names = names or list(dag.nodes)
    g = MarkGraph(names, "DAG")
    for a, b in dag.edges:
        if a in g.index and b in g.index:
            g.directed(a, b)
    return g

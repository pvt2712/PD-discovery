"""metrics — scoring a learned graph against the Arm A truth (PROTOCOL.md §6.4).

Every metric takes a MarkGraph (DAG, CPDAG or PAG) and a truth.Truth, and is computed on the truth's scored pairs
(pairs with a year-t+1 node, no context node, history-induced and near-zero pairs removed) unless stated.

- adjacency precision / recall / F1 against the MAG skeleton (primary: F1);
- arrowhead precision / recall / F1 against the tiered PAG's invariant marks, on pairs adjacent in both graphs. An
  arrowhead where the PAG has a circle is an over-claim: counted separately, not as an error, because the truth class
  leaves it open; against the MAG (secondary) every mark is definite;
- SHD: pairs whose adjacency differs plus adjacent pairs whose marks differ from the PAG;
- SID (Peters & Bühlmann 2015) with latent variables: for each ordered pair (X, Y) with X a dose at t+1, the
  estimated parent set of X is used as an adjustment set and checked against the true generating DAG (latents
  included) with the generalised adjustment criterion (Perković et al. 2018); pairs with no valid observed
  adjustment set are excluded. CPDAG/PAG outputs give an interval (certain errors, possible errors) over the parent
  sets their undetermined edges allow;
- parent-set precision / recall / F1 for the key nodes (decision and outcomes): possible parents against the MAG
  parents;
- expert-edge recall: share of the expert's edges, in their year-t+1 pair, that the learner makes adjacent, by kind.
"""
import itertools

import networkx as nx
import numpy as np

from . import expert as E
from .graphs import MarkGraph


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else np.nan
    r = tp / (tp + fn) if tp + fn else np.nan
    f = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan     # an empty estimate scores F1 = 0
    return p, r, f


def adjacency(est: MarkGraph, truth, pairs=None):
    pairs = truth.scored() if pairs is None else pairs
    e = {p for p in est.adjacencies() if p in pairs}
    t = {p for p in truth.mag.adjacencies() if p in pairs}
    tp, fp, fn = len(e & t), len(e - t), len(t - e)
    p, r, f = _prf(tp, fp, fn)
    return dict(adj_tp=tp, adj_fp=fp, adj_fn=fn, adj_precision=p, adj_recall=r, adj_f1=f, est_edges=len(e))


def arrowheads(est: MarkGraph, ref: MarkGraph, pairs, prefix="ah"):
    """Arrowhead metrics on pairs adjacent in both graphs. With a PAG reference, only invariant marks (A or T) are
    scored and arrowheads at circle positions are reported as over-claims."""
    tp = fp = fn = over = 0
    for p in pairs:
        a, b = sorted(p, key=ref.index.get)
        me, mr = est.marks(a, b), ref.marks(a, b)
        if me is None or mr is None:
            continue
        for k in (0, 1):
            e_arrow, r_mark = me[k] == "A", mr[k]
            if r_mark == "C":
                over += e_arrow
                continue
            if e_arrow and r_mark == "A":
                tp += 1
            elif e_arrow:
                fp += 1
            elif r_mark == "A":
                fn += 1
    p, r, f = _prf(tp, fp, fn)
    return {f"{prefix}_precision": p, f"{prefix}_recall": r, f"{prefix}_f1": f, f"{prefix}_overclaims": over}


def shd(est: MarkGraph, ref: MarkGraph, pairs):
    d = 0
    for p in pairs:
        a, b = sorted(p, key=ref.index.get)
        me, mr = est.marks(a, b), ref.marks(a, b)
        if (me is None) != (mr is None) or (me is not None and me != mr):
            d += 1
    return d


def parent_sets(est: MarkGraph, truth, nodes=None):
    nodes = nodes or truth.key_nodes
    rows = {}
    for k in nodes:
        t = set(truth.mag.parents(k))
        e = set(est.possible_parents(k)) if k in est.index else set()
        p, r, f = _prf(len(e & t), len(e - t), len(t - e))
        rows[k] = dict(precision=p, recall=r, f1=f, exact=float(e == t))
    out = {f"pa_{k}_f1": v["f1"] for k, v in rows.items()}
    out["pa_mean_f1"] = float(np.nanmean([v["f1"] for v in rows.values()])) if rows else np.nan
    out["pa_decision_recall"] = rows.get("ldopa_1", {}).get("recall", np.nan)
    out["pa_decision_precision"] = rows.get("ldopa_1", {}).get("precision", np.nan)
    return out


def expert_recall(est: MarkGraph, truth):
    by = {}
    for (a, b), kind in truth.summary_pairs:
        if frozenset((a, b)) in truth.excluded_pairs:
            continue
        by.setdefault(kind, []).append(est.adjacent(a, b) if a in est.index and b in est.index else False)
    out = {f"expert_recall_{k}": float(np.mean(v)) for k, v in by.items()}
    allv = [x for k, v in by.items() if k != "context" for x in v]
    out["expert_recall"] = float(np.mean(allv)) if allv else np.nan
    return out


# ---------------------------------------------------------------------------- SID with latent variables
class AdjustmentOracle:
    """Validity of adjustment sets in the true generating DAG (latents included), cached across learned graphs."""

    def __init__(self, dag: nx.DiGraph, observed):
        self.dag, self.obs = dag, set(observed)
        self._de = {}
        self._an = {}
        self._cache = {}
        self._pbd = {}

    def de(self, n):
        if n not in self._de:
            self._de[n] = nx.descendants(self.dag, n) | {n}
        return self._de[n]

    def an(self, n):
        if n not in self._an:
            self._an[n] = nx.ancestors(self.dag, n) | {n}
        return self._an[n]

    def _parts(self, x, y):
        if (x, y) not in self._pbd:
            cn = (self.de(x) - {x}) & self.an(y)                      # nodes on causal paths x -> ... -> y
            forb = set().union(*[self.de(w) for w in cn]) | {x} if cn else {x}
            g = self.dag
            if cn:
                g = self.dag.copy()
                g.remove_edges_from([(x, w) for w in self.dag.successors(x) if w in cn])
            self._pbd[(x, y)] = (cn, forb, g)
        return self._pbd[(x, y)]

    def effect(self, x, y):
        return y in self.de(x) and y != x

    def valid(self, x, y, Z):
        key = (x, y, frozenset(Z))
        if key not in self._cache:
            cn, forb, g = self._parts(x, y)
            ok = not (set(Z) & forb) and nx.is_d_separator(g, {x}, {y}, set(Z))
            self._cache[key] = ok
        return self._cache[key]

    def identifiable(self, x, y):
        """A valid observed adjustment set exists iff the canonical one is valid (van der Zander, Liśkiewicz & Textor 2019)."""
        cn, forb, g = self._parts(x, y)
        canon = ((self.an(x) | self.an(y)) & self.obs) - forb - {x, y}
        return self.valid(x, y, canon)


def sid(est: MarkGraph, truth, oracle: AdjustmentOracle, sources=None, targets=None, max_subsets=256):
    """Latent-aware SID over (sources x targets): (certain errors, possible errors, pairs scored, pairs not identifiable)."""
    sources = sources or [f"{v}_1" for v in E.DOSE if f"{v}_1" in truth.observed]
    targets = targets or [n for n in truth.observed if E.slice_of(n) == 1]
    lo = hi = scored = unident = 0
    rng = np.random.default_rng(0)
    for x in sources:
        if x not in est.index:
            continue
        definite = set(est.parents(x))
        undecided = [w for w in est.possible_parents(x) if w not in definite]
        if len(undecided) <= 8:
            subsets = [set(c) for r in range(len(undecided) + 1) for c in itertools.combinations(undecided, r)]
        else:
            subsets = [set(), set(undecided)] + [{w for w in undecided if rng.random() < 0.5} for _ in range(max_subsets)]
        for y in targets:
            if y == x:
                continue
            if not oracle.identifiable(x, y):
                unident += 1
                continue
            scored += 1
            wrong = []
            for s in subsets:
                pa = definite | s
                if y in pa:
                    wrong.append(oracle.effect(x, y))
                else:
                    wrong.append(not oracle.valid(x, y, pa - {x, y}))
            lo += all(wrong)
            hi += any(wrong)
    return dict(sid_lower=lo, sid_upper=hi, sid_pairs=scored, sid_unidentifiable=unident)


def score_all(est: MarkGraph, truth, oracle: AdjustmentOracle | None = None):
    pairs = truth.scored()
    out = {}
    out.update(adjacency(est, truth, pairs))
    out.update(arrowheads(est, truth.pag, pairs, "ah"))
    out.update(arrowheads(est, truth.mag, pairs, "ahmag"))
    out["shd"] = shd(est, truth.pag, pairs)
    out.update(parent_sets(est, truth))
    out.update(expert_recall(est, truth))
    allpairs = {frozenset(p) for p in itertools.combinations(truth.observed, 2)} - {frozenset(p) for p in itertools.combinations(E.CONTEXT, 2)}
    full = adjacency(est, truth, allpairs)
    out["adj_f1_allpairs"] = full["adj_f1"]
    if oracle is not None:
        out.update(sid(est, truth, oracle))
    return out

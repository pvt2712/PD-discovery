"""The oracle MAG/PAG (truth.py) against brute-force FCI with a d-separation oracle, and the v5.1 layouts."""
import contextlib
import io
import itertools

import networkx as nx
import numpy as np
import pytest

from escalation_discovery import expert as E
from escalation_discovery import knowledge as K
from escalation_discovery import truth as T
from escalation_discovery.graphs import from_causallearn


def brute_force(dag, observed, tiers):
    fci = K.install((), ())
    bk, _ = K.background(observed, tiers)
    latent = [n for n in dag.nodes if n not in observed]
    gi = nx.relabel_nodes(dag, {n: i for i, n in enumerate(list(observed) + latent)})
    with contextlib.redirect_stdout(io.StringIO()):
        G, _ = fci(np.zeros((10, len(observed))), "d_separation", 0.05, true_dag=gi, background_knowledge=bk,
                   show_progress=False, verbose=False, node_names=list(observed))
    return from_causallearn(G, list(observed), "PAG")


def test_toy_latent_confounder():
    g = nx.DiGraph([("A", "B"), ("L", "B"), ("L", "C"), ("D", "C")])
    mag, pag = T.oracle(g, ["A", "B", "C", "D"], [["A", "B", "C", "D"]])
    assert mag.marks("B", "C") == ("A", "A")
    assert pag.marks("A", "B") == ("C", "A") and pag.marks("B", "C") == ("A", "A") and pag.marks("C", "D") == ("A", "C")
    assert not pag.adjacent("A", "C") and not pag.adjacent("A", "D")


@pytest.mark.parametrize("seed", range(6))
def test_fast_oracle_matches_brute_force_fci(seed):
    rng = np.random.default_rng(seed)
    n = 9
    order = list(range(n))
    g = nx.DiGraph()
    g.add_nodes_from([f"v{i}" for i in order])
    for i, j in itertools.combinations(order, 2):
        if rng.random() < 0.3:
            g.add_edge(f"v{i}", f"v{j}")
    latent = [f"v{i}" for i in rng.choice(n, 2, replace=False)]
    obs = [v for v in g.nodes if v not in latent]
    tiers = [obs[:3], obs[3:]]                       # some knowledge, as in the real use
    mag, pag = T.oracle(g, obs, tiers)
    ref = brute_force(g, obs, tiers)
    assert pag.adjacencies() == ref.adjacencies()
    assert pag.edges == ref.edges


def test_v51_layouts():
    spec = E.load()
    assert len(spec.tv) == 17 and len(spec.baseline) == 6 and spec.latent == ("trait",)
    obs = E.observed(spec)
    assert len(obs) == 43 and obs[:2] == ["era2", "genetic"]
    assert E.hide_rule(spec, 2) == ("motor_na", "fluct")
    assert all(n not in E.observed(spec, hidden=("motor_na",)) for n in ("motor_na_0", "motor_na_1"))
    for L in (0, 1, 2):
        assert nx.is_directed_acyclic_graph(E.dynamic_dag(spec, L=L))
    # a non-dose cause of a dose acts from the previous year
    d = E.dynamic_dag(spec, L=0)
    assert d.has_edge("motor_na_0", "ldopa_1") and not d.has_edge("motor_na_1", "ldopa_1")


def test_history_pairs_are_stable_in_depth():
    """Hidden earlier years add a fixed set of adjacencies on the scored pairs; more than one latent year adds none."""
    spec = E.load()
    obs = E.observed(spec)
    ev = T.evaluation_pairs(obs)
    adj = {}
    for L in (0, 1, 2):
        a, _, _ = T.mag_skeleton(E.dynamic_dag(spec, L=L, person_effects=False), obs)
        adj[L] = {frozenset(p) for p in a} & ev
    assert adj[0] <= adj[1] and adj[1] == adj[2]
    assert len(adj[1] - adj[0]) == 46

"""Metrics on hand-made graphs (metrics.py)."""
import networkx as nx
import numpy as np

from escalation_discovery import metrics as MT
from escalation_discovery.graphs import MarkGraph
from escalation_discovery.truth import Truth


def _truth(edges, observed, latent_edges=()):
    """A tiny Truth: MAG = PAG = the DAG over observed (no latents unless latent_edges)."""
    dag = nx.DiGraph(list(edges) + list(latent_edges))
    dag.add_nodes_from(observed)
    from escalation_discovery.truth import oracle
    mag, pag = oracle(dag, observed, [observed])
    pairs = {frozenset((a, b)) for i, a in enumerate(observed) for b in observed[i + 1:]}
    return Truth(name="toy", observed=observed, dag=dag, mag=mag, pag=pag, eval_pairs=pairs, key_nodes=["C"],
                 summary_pairs=[(("A", "B"), "same"), (("B", "C"), "same")])


def test_adjacency_and_shd():
    tr = _truth([("A", "B"), ("B", "C")], ["A", "B", "C"])
    est = MarkGraph(["A", "B", "C"], "DAG")
    est.directed("A", "B")
    est.directed("A", "C")
    s = MT.adjacency(est, tr)
    assert (s["adj_tp"], s["adj_fp"], s["adj_fn"]) == (1, 1, 1)
    assert np.isclose(s["adj_f1"], 0.5)
    assert MT.shd(est, tr.pag, tr.scored()) == 3       # A-B marks differ from the PAG's circles, A-C extra, B-C missing


def test_empty_estimate_scores_zero_f1():
    tr = _truth([("A", "B")], ["A", "B", "C"])
    assert MT.adjacency(MarkGraph(["A", "B", "C"]), tr)["adj_f1"] == 0.0


def test_arrowheads_against_pag_count_overclaims_separately():
    # collider A -> C <- B: the PAG has invariant arrowheads at C and circles at A and B
    tr = _truth([("A", "C"), ("B", "C")], ["A", "B", "C"])
    est = MarkGraph(["A", "B", "C"], "DAG")
    est.directed("A", "C")
    est.directed("C", "B")                               # wrong: arrowhead at B where the PAG has a circle
    s = MT.arrowheads(est, tr.pag, tr.scored())
    assert s["ah_precision"] == 1.0 and s["ah_recall"] == 0.5 and s["ah_overclaims"] == 1


def test_sid_zero_for_true_dag_and_counts_a_missing_confounder():
    edges = [("Z", "X"), ("Z", "Y"), ("X", "Y")]
    tr = _truth(edges, ["X", "Y", "Z"])
    oracle = MT.AdjustmentOracle(tr.dag, tr.observed)
    good = MarkGraph(["X", "Y", "Z"], "DAG")
    for a, b in edges:
        good.directed(a, b)
    assert MT.sid(good, tr, oracle, sources=["X"], targets=["Y", "Z"])["sid_upper"] == 0
    bad = MarkGraph(["X", "Y", "Z"], "DAG")
    bad.directed("X", "Y")
    bad.directed("Z", "Y")                               # Z not a parent of X: no adjustment for the confounder
    assert MT.sid(bad, tr, oracle, sources=["X"], targets=["Y"])["sid_lower"] == 1


def test_sid_skips_effects_no_observed_set_identifies():
    tr = _truth([("X", "Y")], ["X", "Y"], latent_edges=[("U", "X"), ("U", "Y")])
    oracle = MT.AdjustmentOracle(tr.dag, tr.observed)
    assert not oracle.identifiable("X", "Y")
    est = MarkGraph(["X", "Y"], "DAG")
    est.directed("X", "Y")
    s = MT.sid(est, tr, oracle, sources=["X"], targets=["Y"])
    assert s["sid_pairs"] == 0 and s["sid_unidentifiable"] == 1


def test_parent_sets_use_possible_parents():
    tr = _truth([("A", "C"), ("B", "C")], ["A", "B", "C"])
    est = MarkGraph(["A", "B", "C"], "PAG")
    est.add("A", "C", "C", "A")                          # A o-> C counts as a possible parent
    s = MT.parent_sets(est, tr, ["C"])
    assert np.isclose(s["pa_C_f1"], 2 / 3)

"""Adapters: output layout, knowledge semantics, the controls on simple systems, the ensemble."""
import numpy as np
import pytest

from escalation_discovery import arm_b
from escalation_discovery import expert as E
from escalation_discovery import generator as G
from escalation_discovery import missingness as M
from escalation_discovery.graphs import MarkGraph
from escalation_discovery.methods import ensemble
from escalation_discovery.methods.base import DiscoveryData, forbidden_matrix
from escalation_discovery.methods.classical import FCI, PC
from escalation_discovery.methods.continuous import NOTEARS, SortNRegress, notears_linear
from escalation_discovery.methods.tetrad import Tetrad


@pytest.fixture(scope="module")
def data():
    spec = E.load()
    gen = G.synthetic("G1")
    long, base = G.simulate(gen, 150, seed=21)
    long, base = M.PatternBank.synthetic(spec).apply(long, base, "mar", 22)
    ts = G.two_slice(long, base, spec)
    obs = E.observed(spec)
    return DiscoveryData(X=ts[obs].to_numpy(float), names=obs, groups=ts.PATNO.to_numpy(), tiers=E.tiers(spec), seed=3,
                         long=long, base=base)


@pytest.mark.parametrize("make", [lambda: FCI("mv_fisherz"), lambda: PC("mv_fisherz"), lambda: FCI("cgcm")])
def test_knowledge_is_respected(data, make):
    data.knowledge = True
    g = make().fit(data)
    assert g.nodes == data.names
    v = arm_b.violations(g, data.tiers)
    assert v["into_root"] == 0 and v["backwards"] == 0


def test_forbidden_matrix(data):
    F = forbidden_matrix(data)
    i = data.names.index
    assert F[i("adl_1"), i("adl_0")] and F[i("age"), i("male")] and F[i("adl_0"), i("age")]
    assert not F[i("era2"), i("age")] and not F[i("adl_0"), i("adl_1")]


def test_notears_recovers_a_chain():
    rng = np.random.default_rng(0)
    x = rng.normal(size=2000)
    y = 1.5 * x + rng.normal(size=2000)
    z = -1.2 * y + rng.normal(size=2000)
    X = np.column_stack([x, y, z])
    W = notears_linear((X - X.mean(0)) / X.std(0), 0.05, 0.3)
    A = np.abs(W) > 0
    assert A[0, 1] or A[1, 0]
    assert A[1, 2] or A[2, 1]
    assert not (A[0, 2] or A[2, 0])


def test_sortnregress_reads_variance_order():
    rng = np.random.default_rng(1)
    x = rng.normal(size=3000)
    y = 2.0 * x + rng.normal(size=3000)             # variance grows along the causal order
    d = DiscoveryData(X=np.column_stack([x, y]), names=["x", "y"], groups=np.arange(3000), tiers=[["x", "y"]])
    g = SortNRegress("var").fit(d)
    assert g.marks("x", "y") == ("T", "A")


def test_notears_tiered_has_no_forbidden_edges(data):
    data.knowledge = True
    g = NOTEARS(use_knowledge=True, lambda1=0.05, w_threshold=0.3).fit(data)
    assert arm_b.violations(g, data.tiers)["backwards"] == 0


def test_ensemble_majority():
    names = ["a", "b", "c"]
    g1, g2, g3 = MarkGraph(names), MarkGraph(names), MarkGraph(names)
    g1.add("a", "b", "T", "A")
    g2.add("a", "b", "T", "A")
    g3.add("b", "c", "T", "A")
    out = ensemble.majority({"m1": g1, "m2": g2, "m3": g3}, {"m1": "f1", "m2": "f2", "m3": "f3"}, names)
    assert out.adjacent("a", "b") and not out.adjacent("b", "c") and out.marks("a", "b") == ("T", "A")


@pytest.mark.slow
def test_tetrad_boss_with_knowledge(data):
    m = Tetrad("boss", penalty=4.0)
    ok, why = m.available()
    if not ok:
        pytest.skip(why)
    data.knowledge = True
    g = m.fit(data)
    assert g.n_edges() > 0 and arm_b.violations(g, data.tiers)["backwards"] == 0


def test_panel_is_the_plan_b_scope():
    """PROTOCOL.md §10.2: the dropped learners are out, the deep-learning learners run in R only, FCI is capped at 3."""
    from escalation_discovery.methods import registry
    entries = {e.name: e for e in registry.panel()}
    dropped = {"pc_cgcm", "deci", "jpcmciplus"} | {f"tetrad_{a}" for a in ("pc", "fci", "boss_fci", "gfci", "grasp_fci")}
    assert not dropped & {n.replace("_noknow", "") for n in entries}
    assert {n for n, e in entries.items() if e.realistic_only} == {"sdcd", "dibs", "avici", "cdfm", "arrow"}
    assert entries["fci_cgcm"].method.cfg["depth"] == 3 and entries["fci_fz"].method.cfg["depth"] == 3

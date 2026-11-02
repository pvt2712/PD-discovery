"""Arm B helpers: negative controls, frequencies, the blinded sheet, the selection rule, falsification statements."""
import json

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from escalation_discovery import arm_b
from escalation_discovery import expert as E
from escalation_discovery import falsify as F
from escalation_discovery.graphs import MarkGraph


def test_violations_counts():
    tiers = [["era2"], ["age"], ["x_0"], ["x_1"]]
    g = MarkGraph(["era2", "age", "x_0", "x_1"])
    g.add("x_1", "x_0", "T", "A")        # backwards in time
    g.add("x_0", "age", "T", "A")        # into an exogenous node from a system node (also backwards)
    g.add("era2", "age", "T", "A")       # context -> exogenous: allowed
    g.add("age", "x_1", "A", "A")        # bidirected: confounding, not a time violation (but an arrowhead into age)
    v = arm_b.violations(g, tiers)
    assert v["into_root"] == 2 and v["backwards"] == 2 and v["cross_tier_directed"] == 3


def test_frequencies_and_stable_graph():
    names = ["a", "b", "c"]
    fits = []
    for b in range(4):
        g = MarkGraph(names)
        g.add("a", "b", "T", "A")
        if b < 1:
            g.add("b", "c", "T", "A")
        fits.append(dict(b=b, graph=g.to_json()))
    fq = arm_b.frequencies(fits, names).set_index(["a", "b"])
    assert fq.loc[("a", "b"), "freq"] == 1.0 and fq.loc[("b", "c"), "freq"] == 0.25
    sg = arm_b.stable_graph(fq.reset_index(), None, names, 0.6)
    assert sg.adjacent("a", "b") and not sg.adjacent("b", "c")


def test_clinician_sheet_is_blinded():
    agree = pd.DataFrame([dict(a="motor_na_0", b="ldopa_1", source="expert", kind="same", status="found, compatible", marks="TA"),
                          dict(a="dep_1", b="adl_1", source="data only", kind="", status="data only", marks="TA"),
                          dict(a="cog_1", b="hall_1", source="expert", kind="same", status="not found", marks="")])
    sheet, key = arm_b.clinician_sheet(agree, seed=1)
    assert {"item", "statement", "choice", "choices", "comment"} == set(sheet.columns)
    assert not sheet.statement.str.contains("expert|data|stable|frequency").any()
    assert set(key.category) >= {"disagreement: data only", "disagreement: not found"}
    assert len(sheet) == len(key)


def test_selection_rule_prefers_anchor_without_a_winner():
    dec = pd.DataFrame([dict(method="fci_cgcm", role="anchor", primary=True, adj_f1_realistic=0.5),
                        dict(method="dagma", role="control", primary=True, adj_f1_realistic=0.2),
                        dict(method="avici", role="contender", primary=True, adj_f1_realistic=0.6, beats_best_anchor=False)])
    stab = pd.DataFrame([dict(method=m, median_freq=0.9) for m in ("fci_cgcm", "avici")])
    ctrl = pd.DataFrame([dict(method="fci_cgcm_noknow", backwards_rate_hi=0.2), dict(method="avici", backwards_rate_hi=0.3)])
    chosen, _ = arm_b.select(dec, stab, ctrl)
    assert chosen == "fci_cgcm"
    dec.loc[2, "beats_best_anchor"] = True
    assert arm_b.select(dec, stab, ctrl)[0] == "avici"


def test_statements_on_a_chain_and_the_relabelling_null():
    dag = nx.DiGraph([("A", "B"), ("B", "C")])
    st = F.statements(dag, ["A", "B", "C"])
    assert st == [("A", "C", ("B",))]
    spec = E.load()
    maps = F.relabel_maps(spec, 5, seed=1)
    assert len(maps) == 5 and all(set(m) == set(spec.tv) for m in maps)
    assert all(m[v] in E.DOSE for m in maps for v in E.DOSE)


def test_core_map_structure():
    """v2's core map: 3 triggers x 4 brakes give 3! * 4! - 1 relabellings; hiding the latent nodes removes implications."""
    from escalation_discovery import coremap
    maps = F.core_relabellings()
    assert len(maps) == 143
    measured = [n for n, v in coremap.nodes().items() if v["measured"]]
    with_lat = F.core_statements(F.core_map_dag(True), measured)
    without = F.core_statements(F.core_map_dag(False), measured)
    assert len(without) > len(with_lat) > 0
    assert all(x in measured and y in measured and set(S) <= set(measured) for x, y, S in with_lat)


def test_cohort_subsets():
    base = pd.DataFrame({"PATNO": [1, 2, 3, 4], "era2": [0, 1, 0, 1], "genetic": [0, 0, 1, 1]})
    long = pd.DataFrame({"PATNO": [1, 1, 2, 3, 4], "year": [0, 1, 0, 0, 0]})
    assert set(arm_b.subset(long, base, "ppmi1")[1].PATNO) == {1}
    assert set(arm_b.subset(long, base, "no_genetic")[1].PATNO) == {1, 2}
    assert set(arm_b.subset(long, base, "no_ppmi2")[1].PATNO) == {1, 3, 4}
    fixed = {s: arm_b.fixed_context(arm_b.subset(long, base, s)[1]) for s in ("all", "ppmi1", "no_genetic", "no_ppmi2")}
    assert fixed == {"all": (), "ppmi1": ("era2", "genetic"), "no_genetic": ("genetic",), "no_ppmi2": ()}


def test_fixed_context_nodes_leave_the_data():
    long, base = arm_b.subset(*arm_b.synthetic_panel(n=200, seed=3), "no_genetic")
    drop = arm_b.fixed_context(base)
    data, _ = arm_b.make_data(long, base, drop=drop)
    assert drop == ("genetic",) and data.X.shape[1] == len(data.names)
    assert "genetic" not in data.names + list(data.context) + sum(data.tiers, [])
    long[E.load().tv[0]] = 0.0                 # any other variable without variation stops the run
    with pytest.raises(ValueError, match="no variation"):
        arm_b.make_data(long, base, drop=drop)


def test_sensitivity_runs_take_the_main_selection(tmp_outputs):
    with pytest.raises(SystemExit):
        arm_b.selected_methods("synthetic")
    meta = tmp_outputs / "results" / "arm_b" / "synthetic" / "meta.json"
    meta.parent.mkdir(parents=True)
    meta.write_text(json.dumps({"selected": "fci_cgcm"}))
    assert arm_b.selected_methods("synthetic") == ["fci_cgcm"]
    meta.write_text(json.dumps({"selected": "ensemble"}))
    assert arm_b.selected_methods("synthetic") == []            # the ensemble needs its members: the whole panel

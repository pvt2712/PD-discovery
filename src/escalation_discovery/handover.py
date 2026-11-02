"""handover — what the discovery experiment hands to the inference experiment (PROTOCOL.md §9).

    uv run escd handover [--data synthetic|ppmi]

One aggregate YAML file, results/handover/<data>/to_inference.yaml:
1. the expert DAG's test: violation rate against the relabelled null, and the edges into the dose decision with
   their support;
2. the selected learner's stable graph: the possible parents of the next-year levodopa dose (ldopa_1), each mapped
   to the inference experiment's pre-decision covariate (escalation-study-v2 docs/core_map.yaml) and flagged when the
   43-covariate adjustment set lacks it ("data-revealed parent outside the set");
3. the members of the adjustment set the stable graph does not place before the decision (a possible descendant of
   ldopa_1 would be a reason to drop it; none is expected, as all are measured before the decision);
4. the latent-trait indicators the data support (for the Q4 sensitivity benchmarks).
The inference experiment decides what to do with it; discovery estimates no effect.
"""
import argparse
import json

import pandas as pd
import yaml

from . import config as C
from . import expert as E
from .graphs import MarkGraph

# two-slice node -> the inference experiment's covariate column(s) (v2 core_map.yaml analysis blocks)
TO_INFERENCE = {
    "motor_na_0": ["h_motor_na"], "axial_0": ["h_axial"], "adl_0": ["h_adl"], "se_0": ["h_se"], "fluct_0": ["h_fluct"],
    "dysk_0": ["h_dysk"], "ess_0": ["h_ess"], "icd_core_0": ["h_icd"], "cog_0": ["h_cog"], "dep_0": ["h_dep"],
    "hall_0": ["h_hall"], "ldopa_0": ["A0", "h_ldopa"], "da_0": ["h_da"], "maob_0": ["h_maob"], "ama_0": ["h_ama"],
    "t_ldopa_0": ["h_t_ldopa"], "cum_ldopa_0": ["h_cum_ldopa"], "age": ["age"], "male": ["male"], "educ": ["educ"],
    "apoe4": ["apoe4"], "dat0": ["dat0"], "dur0": ["dur0"], "t": ["t"],
    "era2": ["(cohort: v2 analyses each cohort separately)"], "genetic": ["(cohort: v2 analyses each cohort separately)"],
}


def adjustment_columns():
    """The 43 pre-decision covariates of escalation-study-v2 (its core_map.yaml, copied to docs/core_map.yaml)."""
    m = yaml.safe_load(open(C.docs_dir() / "core_map.yaml"))
    cols = []
    for node in m["nodes"]:
        if node.get("role") in ("trigger", "brake", "context"):
            cols += node.get("columns", [])
    for blk in m["off_guideline_blocks"].values():
        cols += blk["columns"]
    return list(dict.fromkeys(cols))


def build(data_label="synthetic"):
    out = C.results_dir(f"arm_b/{data_label}")
    meta = json.loads((out / "meta.json").read_text())
    chosen = meta.get("selected")
    fals = json.loads((out / "expert_falsification.json").read_text())
    sup = pd.read_csv(out / "expert_edge_support.csv")
    g = MarkGraph.from_json(json.loads((out / f"stable_{chosen}.json").read_text())) if chosen else None
    adj = set(adjustment_columns())
    parents = []
    if g is not None:
        for w in g.possible_parents("ldopa_1"):
            cols = TO_INFERENCE.get(w, [])
            parents.append(dict(node=w, marks="".join(g.marks(w, "ldopa_1")), definite=w in g.parents("ldopa_1"),
                                inference_columns=cols,
                                outside_adjustment_set=bool(cols) and not any(c in adj for c in cols) and not cols[0].startswith("(")))
    descendants = []
    if g is not None:
        for n in g.nodes:
            m = g.marks("ldopa_1", n) if n != "ldopa_1" else None
            if m and m[1] in ("A", "C") and m[0] == "T" and E.slice_of(n) == 0:
                descendants.append(n)
    dec = sup[sup.target == "ldopa_1"][["source", "kind", "p", "supported"]].to_dict("records")
    trait = [b for a, b, k in E.load().edges if a == "trait"]
    doc = dict(data=data_label, selected_learner=chosen, stability_tau=meta.get("tau"),
               expert_dag=dict(name="v51_escalation", **{k: fals[k] for k in ("statements", "violation_rate", "null_mean", "p_value")},
                               decision_edges=dec),
               decision_parents=parents,
               adjustment_set_members_after_decision=descendants,
               latent_trait_indicators=trait,
               note="Aggregate hand-over; discovery estimates no effect. Columns refer to escalation-study-v2.")
    hd = C.results_dir(f"handover/{data_label}")
    (hd / "to_inference.yaml").write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True))
    return doc


def main(argv=None):
    ap = argparse.ArgumentParser(prog="escd handover")
    ap.add_argument("--data", default="synthetic", choices=["synthetic", "ppmi"])
    a = ap.parse_args(argv)
    print(yaml.safe_dump(build(a.data), sort_keys=False, allow_unicode=True))

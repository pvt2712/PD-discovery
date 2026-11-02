"""expert — the expert DAG of the experiment (v5.1, docs/expert_dags.yaml) in the layouts both arms need.

The expert DAG is a summary graph over 17 time-varying and 6 baseline nodes plus a latent non-motor trait. Each edge
is tagged 'same' (X_t -> Y_t), 'lag' (X_t -> Y_t+1), 'both', or none (a baseline or latent source). The pooled
analysis (PROTOCOL.md §3) adds two cohort context nodes, era2 and genetic, which may point into any node and receive
none (joint causal inference, Mooij, Magliacane & Claassen, JMLR 2020).

- `two_slice_names()`: the observed variables of one two-slice row: context, exogenous and baseline nodes, then year
  t (suffix _0) and year t+1 (suffix _1).
- `tiers()`: the temporal background knowledge every method that accepts knowledge receives.
- `dynamic_dag()`: the data-generating DAG behind a two-slice row: L latent earlier years, a latent person effect per
  time-varying node, the latent trait and the cohort context. truth.py derives the oracle MAG/PAG over the observed
  variables from it. Earlier years are latent in every two-slice table, so year-t variables share hidden history;
  this is why the truth is a PAG and why scoring concentrates on pairs with a year-t+1 node (metrics.py).

Unrolling conventions follow ppmi-causal/src/ppmi_causal/expert.py:unroll: autoregression X_t -> X_t+1 for every
time-varying node; baseline and latent sources point into every year; a 'same' edge into a dose node from a non-dose
node becomes X_t -> dose_t+1 (the dose recorded at t+1 is a decision taken after visit t); the years-since-baseline
node t points into every time-varying node.
"""
from dataclasses import dataclass, field
from functools import lru_cache

import networkx as nx
import yaml

from . import config as C

CONTEXT = ("era2", "genetic")                       # cohort context (pooled analysis)
EXOGENOUS = ("age", "male", "educ", "apoe4", "t")   # nothing in the system can cause these
DOSE = ("ldopa", "da", "maob", "ama")

# type: binary | continuous | semicontinuous (a point mass at 0 plus a continuous part) | ordinal | count
NODE_INFO = {
    "era2": dict(type="binary", label="enrolled in the PPMI 2.0 era"),
    "genetic": dict(type="binary", label="enrolled in a genetic PD arm"),
    "age": dict(type="continuous", label="age at enrolment", unit="years"),
    "male": dict(type="binary", label="male sex"),
    "educ": dict(type="count", label="years of education", unit="years"),
    "apoe4": dict(type="binary", label="APOE e4 carrier"),
    "t": dict(type="count", label="years since enrolment", unit="years"),
    "dat0": dict(type="continuous", label="baseline putamen DaT binding (SBR)", unit="ratio"),
    "dur0": dict(type="continuous", label="disease duration at enrolment", unit="months"),
    "ldopa": dict(type="semicontinuous", label="levodopa dose (LEDD)", unit="mg/day"),
    "da": dict(type="semicontinuous", label="dopamine-agonist dose (LEDD)", unit="mg/day"),
    "maob": dict(type="semicontinuous", label="MAO-B inhibitor dose (LEDD)", unit="mg/day"),
    "ama": dict(type="semicontinuous", label="amantadine dose (LEDD)", unit="mg/day"),
    "t_ldopa": dict(type="semicontinuous", label="months on levodopa", unit="months"),
    "cum_ldopa": dict(type="semicontinuous", label="cumulative levodopa", unit="g-years"),
    "motor_na": dict(type="ordinal", label="OFF MDS-UPDRS III without axial items", unit="points"),
    "axial": dict(type="ordinal", label="OFF axial motor score", unit="points"),
    "adl": dict(type="ordinal", label="MDS-UPDRS II (motor experiences of daily living)", unit="points"),
    "se": dict(type="ordinal", label="Schwab & England ADL", unit="%"),
    "fluct": dict(type="ordinal", label="wearing-off (MDS-UPDRS 4.3-4.4)", unit="points"),
    "dysk": dict(type="ordinal", label="dyskinesia (MDS-UPDRS 4.1-4.2)", unit="points"),
    "ess": dict(type="ordinal", label="Epworth Sleepiness Scale", unit="points"),
    "icd_core": dict(type="count", label="impulse-control domains positive (QUIP core)", unit="domains"),
    "cog": dict(type="ordinal", label="MoCA", unit="points"),
    "dep": dict(type="ordinal", label="GDS-15", unit="points"),
    "hall": dict(type="ordinal", label="hallucinations (MDS-UPDRS 1.2)", unit="points"),
}


@dataclass(frozen=True)
class ExpertSpec:
    name: str
    tv: tuple                    # time-varying nodes
    baseline: tuple              # baseline nodes (exogenous and baseline disease measures)
    latent: tuple                # time-invariant latent nodes ('trait')
    edges: tuple                 # (from, to, kind), kind in {'same', 'lag', 'both', 'base'}
    grades: dict = field(default_factory=dict, compare=False, hash=False)

    @property
    def baseline_disease(self):
        return tuple(b for b in self.baseline if b not in EXOGENOUS)


@lru_cache(maxsize=8)
def load(name: str = "v51_escalation", path: str | None = None) -> ExpertSpec:
    path = path or str(C.docs_dir() / "expert_dags.yaml")
    spec = {g["name"]: g for g in yaml.safe_load(open(path))["dags"]}[name]
    nodes = list(spec["nodes"])
    baseline = tuple(n for n in nodes if n in ("age", "male", "educ", "apoe4", "dat0", "dur0"))
    tv = tuple(n for n in nodes if n not in baseline)
    latent = tuple(spec.get("latent", []))
    edges, grades = [], {}
    for e in spec["edges"]:
        a, b = e["from"], e["to"]
        kind = "base" if (a in baseline or a in latent) else (e.get("kind") or "both")
        edges.append((a, b, kind))
        grades[(a, b)] = e.get("grade")
    return ExpertSpec(name=name, tv=tv, baseline=baseline, latent=latent, edges=tuple(edges), grades=grades)


def two_slice_names(spec: ExpertSpec, context: bool = True) -> list:
    """Observed variables of one two-slice row, in tier order."""
    out = list(CONTEXT) if context else []
    out += [n for n in EXOGENOUS if n in spec.baseline or n == "t"]
    out += list(spec.baseline_disease)
    out += [f"{v}_0" for v in spec.tv]
    out += [f"{v}_1" for v in spec.tv if v in DOSE] + [f"{v}_1" for v in spec.tv if v not in DOSE]
    return out


def tiers(spec: ExpertSpec, context: bool = True, hidden=()) -> list:
    """Context | exogenous | baseline disease | year t | doses at t+1 | the rest of year t+1. Within a tier the data
    decide the direction; an edge from a later tier into an earlier one is forbidden."""
    hid = set(hidden)
    keep = lambda xs: [x for x in xs if x not in hid and _base(x) not in hid]
    out = [list(CONTEXT)] if context else []
    out += [keep([n for n in EXOGENOUS if n in spec.baseline or n == "t"]), keep(list(spec.baseline_disease)),
            keep([f"{v}_0" for v in spec.tv]), keep([f"{v}_1" for v in spec.tv if v in DOSE]),
            keep([f"{v}_1" for v in spec.tv if v not in DOSE])]
    return [t for t in out if t]


def _base(name: str) -> str:
    return name[:-2] if name[-2:] in ("_0", "_1") else name


def tier_of(spec: ExpertSpec, context: bool = True) -> dict:
    return {n: k for k, t in enumerate(tiers(spec, context)) for n in t}


def slice_of(name: str):
    """0 or 1 for a time-varying variable, None for context, exogenous and baseline nodes."""
    return {"_0": 0, "_1": 1}.get(name[-2:])


def dynamic_dag(spec: ExpertSpec, L: int = 2, person_effects: bool = True, context_edges=(),
                context: bool = True) -> nx.DiGraph:
    """The generating DAG of a two-slice row with L latent earlier years (slices h{L}..h1), the observed years
    0 (= t) and 1 (= t+1), a latent person effect u_<v> per time-varying node, the latent trait, and the cohort
    context edges `context_edges` ((context, target) pairs; a time-varying target is hit in every year).

    Years since enrolment (t) points into every year's time-varying nodes, including the latent ones, so the
    pooled table's mixture of enrolment rows (no history) and later rows is one DAG."""
    g = nx.DiGraph()
    slices = [f"h{k}" for k in range(L, 0, -1)] + ["0", "1"]
    tv, dose = set(spec.tv), set(DOSE)
    name = lambda v, s: f"{v}_{s}"
    g.add_nodes_from((list(CONTEXT) if context else []) + [n for n in EXOGENOUS if n in spec.baseline or n == "t"])
    g.add_nodes_from(list(spec.baseline_disease) + list(spec.latent))
    for i, s in enumerate(slices):
        prev = slices[i - 1] if i > 0 else None
        for v in spec.tv:
            g.add_node(name(v, s))
            g.add_edge("t", name(v, s))
            if person_effects:
                g.add_edge(f"u_{v}", name(v, s))
            if prev:
                g.add_edge(name(v, prev), name(v, s))
        for a, b, kind in spec.edges:
            if kind == "base":
                if b in tv:
                    g.add_edge(a, name(b, s))
                else:
                    g.add_edge(a, b)
                continue
            if a not in tv or b not in tv:
                continue
            if kind in ("lag", "both") and prev:
                g.add_edge(name(a, prev), name(b, s))
            if kind in ("same", "both"):
                if b in dose and a not in dose:
                    if prev:
                        g.add_edge(name(a, prev), name(b, s))
                else:
                    g.add_edge(name(a, s), name(b, s))
    if context:
        for c, b in context_edges:
            if b in tv:
                for s in slices:
                    g.add_edge(c, name(b, s))
            else:
                g.add_edge(c, b)
    if not nx.is_directed_acyclic_graph(g):
        raise ValueError(f"{spec.name}: not acyclic: {nx.find_cycle(g)}")
    return g


def observed(spec: ExpertSpec, context: bool = True, hidden=()) -> list:
    """Observed two-slice variables after hiding the time-varying or baseline nodes in `hidden` (both years)."""
    hid = set(hidden)
    return [n for n in two_slice_names(spec, context) if n not in hid and _base(n) not in hid]


def summary_pairs(spec: ExpertSpec, context_edges=()):
    """The expert's edges as the two-slice pair that represents them in year t+1, with their kind: same-year X -> Y as
    (X_1, Y_1), except into a dose (X_0, dose_1); lagged as (X_0, Y_1); baseline and context as (B, Y_1).
    Trait edges are latent and have no observed pair."""
    out = []
    for a, b, kind in spec.edges:
        if a in spec.latent:
            continue
        if kind == "base":
            out.append(((a, f"{b}_1" if b in spec.tv else b), "base"))
        elif kind in ("same", "both"):
            if b in DOSE and a not in DOSE:
                out.append(((f"{a}_0", f"{b}_1"), "same"))
            else:
                out.append(((f"{a}_1", f"{b}_1"), "same"))
            if kind == "both":
                out.append(((f"{a}_0", f"{b}_1"), "lag"))
        elif kind == "lag":
            out.append(((f"{a}_0", f"{b}_1"), "lag"))
    for c, b in context_edges:
        out.append(((c, f"{b}_1" if b in spec.tv else b), "context"))
    return out


def hide_rule(spec: ExpertSpec, k: int) -> tuple:
    """The k measured time-varying non-dose nodes to hide in the latent-confounding stress test (PROTOCOL.md §6.3):
    most children in the summary graph, ties broken by the number of dose children (the decision's causes), then by
    name. For v5.1: motor_na, then fluct."""
    kids, dose_kids = {}, {}
    for a, b, kind in spec.edges:
        if a in spec.tv and a not in DOSE and b != a:
            kids[a] = kids.get(a, 0) + 1
            dose_kids[a] = dose_kids.get(a, 0) + (b in DOSE)
    order = sorted(kids, key=lambda v: (-kids[v], -dose_kids[v], v))
    return tuple(order[:k])

"""registry — the pre-specified methods panel (PROTOCOL.md §5.1) as adapters, with and without knowledge.

`panel()` returns Entry objects: the adapter, whether it receives the tiers, and whether it is the variant that
represents its method in the ensemble and in the decision rule (the knowledge-using variant when one exists).

Scope of 29 October 2026 (Plan B, PROTOCOL.md §10.2, §12): adapters that stay in the repository but not in the
panel are PC with the clustered GCM, Tetrad's PC, FCI, BOSS-FCI, GFCI and GRaSP-FCI, J-PCMCI+ (`temporal.py`) and
DECI (`External("deci")`); the deep-learning learners run in Arm A's realistic settings only.
"""
from dataclasses import dataclass

from .base import Method
from .classical import BOSS, FCI, PC
from .continuous import DAGMA, NOTEARS, SortNRegress
from .external import External
from .tetrad import Tetrad


@dataclass
class Entry:
    name: str
    method: Method
    knowledge: bool
    primary: bool          # the variant used for the ensemble and the decision rule
    role: str              # anchor | control | contender | baseline
    realistic_only: bool = False   # Arm A: runs in the realistic settings R only (Plan B, PROTOCOL.md §6.3)


def _pair(method_factory, base_name, role):
    """The knowledge-using variant (primary) and the knowledge-free one (negative controls, Arm A comparison)."""
    a, b = method_factory(), method_factory()
    a.name, b.name = base_name, f"{base_name}_noknow"
    return [Entry(a.name, a, True, True, role), Entry(b.name, b, False, False, role)]


def panel(tetrad: bool = True, external: bool = True, knowledge_free: bool = True):
    out = []
    add = (lambda f, n, r: out.extend(_pair(f, n, r) if knowledge_free else _pair(f, n, r)[:1]))
    # classical anchors (causal-learn, with the paper's clustered test and with the textbook test)
    add(lambda: FCI("cgcm"), "fci_cgcm", "anchor")
    add(lambda: FCI("mv_fisherz"), "fci_fz", "anchor")
    add(lambda: PC("mv_fisherz"), "pc_fz", "anchor")
    out.append(Entry("boss_cl", BOSS(), False, True, "anchor"))
    if tetrad:
        add(lambda: Tetrad("boss"), "tetrad_boss", "anchor")
    # continuous-optimisation controls and the sortability baselines
    out.append(Entry("notears", NOTEARS(), False, True, "control"))
    out.append(Entry("notears_tiered", NOTEARS(use_knowledge=True), True, False, "control"))
    out.append(Entry("dagma", DAGMA(), False, True, "control"))
    out.append(Entry("sortnregress", SortNRegress("var"), False, True, "baseline"))
    out.append(Entry("r2sortnregress", SortNRegress("r2"), False, True, "baseline"))
    if external:
        for learner in ("sdcd", "dibs", "avici", "cdfm", "arrow"):
            out.append(Entry(learner, External(learner), False, True, "contender", realistic_only=True))
    return out


def available(entries):
    ok, missing = [], {}
    for e in entries:
        a, why = e.method.available()
        (ok.append(e) if a else missing.__setitem__(e.name, why))
    return ok, missing

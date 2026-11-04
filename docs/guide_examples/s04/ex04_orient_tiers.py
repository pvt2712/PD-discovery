"""Section 4.7: what replacing causal-learn's fci_orient_bk changes, and what it does not.
Synthetic linear Gaussian data with known graphs; L is a hidden cause (never in the table)."""
import contextlib
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import causallearn.search.ConstraintBased.FCI as FCI  # noqa: E402
import numpy as np  # noqa: E402
from causallearn.graph.Endpoint import Endpoint  # noqa: E402
from causallearn.graph.GraphNode import GraphNode  # noqa: E402
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge  # noqa: E402

from pdx.step1 import MARK, fci_graph  # noqa: E402


def show(G, names):
    return "; ".join(f"{names[i]} {MARK[G[i, j]].replace('>', '<')}-{MARK[G[j, i]]} {names[j]}"
                     for i in range(len(names)) for j in range(i + 1, len(names)) if G[i, j])


def causal_learn_as_is(T, tiers):
    nodes = [GraphNode(f"X{i + 1}") for i in range(T.shape[1])]
    bk = BackgroundKnowledge()
    for i, t in enumerate(tiers):
        bk.add_node_to_tier(nodes[i], t)
    with contextlib.redirect_stdout(io.StringIO()):
        G, _ = FCI.fci(T, "fisherz", 0.05, background_knowledge=bk, verbose=False, show_progress=False)
    return np.asarray(G.graph, int)


rng = np.random.default_rng(3)
n = 5000
e = lambda: rng.normal(size=n)  # noqa: E731
cases = {}
L = e()
cases["1. a_t <- L -> b_t1 (hidden cause only)"] = (np.column_stack([L + e(), L + e()]), ["a_t", "b_t1"])
a = e()
cases["2. a_t -> b_t1 (direct effect)"] = (np.column_stack([a, 0.7 * a + e()]), ["a_t", "b_t1"])
B = e()
A = 0.8 * B + e()
cases["3. B -> A_t -> C_t1 (chain)"] = (np.column_stack([B, A, 0.8 * A + e()]), ["B", "A_t", "C_t1"])
L, B = e(), e()
cases["4. B -> A_t <- L -> C_t1 (hidden cause into a collider)"] = (
    np.column_stack([B, 0.8 * B + L + e(), L + e()]), ["B", "A_t", "C_t1"])

for title, (T, names) in cases.items():
    tiers = list(range(T.shape[1]))
    print(title)
    print("   pdx fci_graph       :", show(fci_graph(T, tiers, 0.05, -1), names))
    print("   causal-learn as is  :", show(causal_learn_as_is(T, tiers), names))

# case 4 again, with an arrowhead check that looks only at the mark already there
original = FCI.is_arrow_point_allowed
FCI.is_arrow_point_allowed = lambda x, y, graph, knowledge: graph.get_endpoint(x, y) in (Endpoint.ARROW,
                                                                                         Endpoint.CIRCLE)
try:
    T, names = cases["4. B -> A_t <- L -> C_t1 (hidden cause into a collider)"]
    print("case 4, causal-learn's is_arrow_point_allowed replaced by a mark-only check:",
          show(fci_graph(T, [0, 1, 2], 0.05, -1), names))
finally:
    FCI.is_arrow_point_allowed = original

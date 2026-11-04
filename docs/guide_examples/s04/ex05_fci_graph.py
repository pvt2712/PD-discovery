"""Section 4.8: fci_graph on a hand-made two-slice table with the tiers of protocol §6.2 and the
"nothing but era causes age at onset or sex" rule. Synthetic linear Gaussian data, known graph."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import numpy as np  # noqa: E402

from pdx.step1 import MARK, fci_graph, lagged_adjacency  # noqa: E402

rng = np.random.default_rng(5)
n = 3000
e = lambda: rng.normal(size=n)  # noqa: E731
era, age, sex = e(), e(), e()
duration = 0.5 * age + e()                      # age at onset -> duration (synthetic)
y0_t = 0.5 * duration + e()
y1_t = 0.4 * age + e()
y0_t1 = 0.6 * y0_t + e()
y1_t1 = 0.6 * y1_t + 0.3 * y0_t + e()
T = np.column_stack([era, age, sex, duration, y0_t, y1_t, y0_t1, y1_t1])
cols = ["b0", "b1", "b2", "b3", "y0_t", "y1_t", "y0_t1", "y1_t1"]
names = ["era", "age", "sex", "duration", "y0_t", "y1_t", "y0_t1", "y1_t1"]
tiers = [0, 1, 1, 1, 2, 2, 3, 3]               # era | other baseline | year t | year t+1


def show(G):
    return "\n".join(f"   {names[i]:8s} {MARK[G[i, j]].replace('>', '<')}-{MARK[G[j, i]]} {names[j]}"
                     for i in range(len(names)) for j in range(i + 1, len(names)) if G[i, j])


G = fci_graph(T, tiers, 0.05, 3, "fisherz", exogenous=[1, 2], exempt=[0])
print("endpoint matrix G[i, j] = mark at i (-1 tail, 1 arrowhead, 2 circle):")
print(G)
print("edges, with the exogenous rule:")
print(show(G))
print("lagged adjacency [to, from]:\n", lagged_adjacency(G, cols, 2).astype(int))
print("edges, without the exogenous rule:")
print(show(fci_graph(T, tiers, 0.05, 3, "fisherz")))

"""Sections 4.10-4.11: reading lagged adjacencies and directed lagged edges off a hand-made
endpoint matrix, in the [to, from] convention of the support. No data."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import numpy as np  # noqa: E402

from pdx.step1 import lagged_adjacency, lagged_directed  # noqa: E402

cols = ["b0", "y0_t", "y1_t", "y0_t1", "y1_t1"]
G = np.zeros((5, 5), int)


def edge(a, b, mark_a, mark_b):                 # mark at a, mark at b: -1 tail, 1 arrowhead, 2 circle
    i, j = cols.index(a), cols.index(b)
    G[i, j], G[j, i] = mark_a, mark_b


edge("y0_t", "y0_t1", 2, 1)                     # y0_t  o-> y0_t1  (persistence of domain 0)
edge("y0_t", "y1_t1", -1, 1)                    # y0_t  --> y1_t1  (domain 0 drives domain 1, oriented)
edge("y1_t", "y1_t1", 2, 1)                     # y1_t  o-> y1_t1
edge("y0_t1", "y1_t1", 2, 2)                    # same-year o-o: adds no drift entry
edge("b0", "y1_t", 2, 1)                        # baseline o-> year t: adds no drift entry
print("lagged_adjacency [to, from]:\n", lagged_adjacency(G, cols, 2).astype(int))
print("lagged_directed  [to, from]:\n", lagged_directed(G, cols, 2).astype(int))

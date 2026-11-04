"""Notes 4.N: Step1Config.depth caps FCI's adjacency search only. Counts the tests FCI requests, by
size of the conditioning set, on a synthetic six-domain table. Synthetic data only."""
import collections
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import numpy as np  # noqa: E402
from causallearn.utils import cit as CITmod  # noqa: E402

from pdx import step1  # noqa: E402
from pdx.sim import scenarios  # noqa: E402
from pdx.step1 import Step1Config, fci_graph, step1_table  # noqa: E402

panel, _ = scenarios.generate("S1", 400, np.random.default_rng(7), d=6)
X, tiers, who, cols, exo = step1_table(panel, Step1Config())
step1._register_tests()
cls = CITmod._custom_ci_tests["pdx_fisherz"]
original = cls.__call__
for depth in (1, 3):
    sizes = collections.Counter()

    def counting(self, a, b, condition_set=None):
        sizes[0 if condition_set is None else len(condition_set)] += 1
        return original(self, a, b, condition_set)

    cls.__call__ = counting
    try:
        t0 = time.time()
        fci_graph(X, tiers, 0.05, depth, "fisherz", clusters=who, exogenous=exo, exempt=[0])
    finally:
        cls.__call__ = original
    print(f"depth={depth}: {X.shape[0]} rows x {X.shape[1]} columns, {time.time() - t0:.1f} s; "
          f"test calls by |S|: {dict(sorted(sizes.items()))}")

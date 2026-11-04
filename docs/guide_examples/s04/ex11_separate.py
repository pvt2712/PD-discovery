"""Sections 4.18-4.21: the separate-graphs reference. Part 1 feeds made-up frequency differences
into separate_graph_reference to show the decision rule by hand; part 2 runs it for real on a
small synthetic panel. Synthetic data only."""
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import numpy as np  # noqa: E402

from pdx import separate  # noqa: E402
from pdx.sim import scenarios  # noqa: E402
from pdx.step1 import Step1Config  # noqa: E402

np.set_printoptions(precision=3, suppress=True)

# ---- part 1: the decision rule on made-up numbers (d = 2, n_sub = 10, 20 permutations)
rng = np.random.default_rng(0)
observed = np.array([[0.0, 0.1], [0.5, 0.0]])          # D(e) = p(e, TD) - p(e, PIGD), [to, from]
null = [np.clip(np.round(rng.normal(0, [[0.02, 0.1], [0.15, 0.02]]), 1), -1, 1) for _ in range(20)]
null[3][0, 0] = 0.1                                      # the self-lags almost never differ in the null
queue = [(observed, observed)] + [(m, m) for m in null]
real_difference = separate._difference
separate._difference = lambda *args: queue.pop(0)        # replaces the FCI runs by the numbers above
try:
    dummy = SimpleNamespace(group=np.array([0, 0, 1, 1]))   # only .group is read once _difference is replaced
    out = separate.separate_graph_reference(dummy, n_sub=10, n_perm=20, rng=rng)
finally:
    separate._difference = real_difference
N = np.array(null)
mu, sd = N.mean(0), np.maximum(N.std(0, ddof=1), 1 / 10)
tmax = (np.abs(N - mu) / sd).reshape(20, -1).max(1)
print("null mean:\n", mu, "\nnull spread (floored at 1/n_sub = 0.1):\n", sd)
print("standardised observed |D - mu| / sd:\n", np.abs(observed - mu) / sd)
print("max over links in each null dataset:", np.sort(tmax).round(2))
print(f"95th percentile by hand: {np.quantile(tmax, 0.95):.3f}; code: "
      f"{out['permutation_reference']['adjacency']['fw_threshold_standardised']:.3f}")
print("candidates (beyond the threshold and |D| >= 0.25):", out["permutation_reference"]["adjacency"]["candidates"])

# ---- part 2: the real thing, small (n_sub = 3 subsamples, 4 permutations, separating sets up to 2)
panel, _ = scenarios.generate("S1", 300, np.random.default_rng(4), d=4)
t0 = time.time()
res = separate.separate_graph_reference(panel, n_sub=3, n_perm=4, rng=np.random.default_rng(1),
                                        cfg=Step1Config(depth=2))
print(f"\nreal run, {panel.n} participants (TD {int((panel.group == 0).sum())}, PIGD {int((panel.group == 1).sum())}), "
      f"{2 * 3 * (1 + 4)} FCI fits, {time.time() - t0:.1f} s")
print("D_adjacency [to, from]:\n", np.array(res["D_adjacency"]))
print("D_directed  [to, from]:\n", np.array(res["D_directed"]))
for kind in ("adjacency", "directed"):
    r = res["permutation_reference"][kind]
    print(f"{kind}: threshold {r['fw_threshold_standardised']:.2f}, candidates {r['candidates']}")

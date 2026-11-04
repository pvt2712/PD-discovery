"""Guide §6.3-6.4: the TD/PIGD rule on simulated year-0 items (synthetic only).

    OMP_NUM_THREADS=1 ../ppmi-causal/.venv/bin/python docs/guide_examples/s06/ex_phenotype_rule.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from pdx.sim import family_a  # noqa: E402
from pdx.sim.family_a import _items, default_truth, phenotype_rule  # noqa: E402

name = {0: "TD", 1: "PIGD", -1: "indeterminate"}
# 1. hand-made item rows: (tremor items, PIGD items) -> mean ratio -> class
rows = [([2, 2, 2, 2], [1, 1, 1, 1, 1]), ([0, 0, 0, 0], [2, 2, 2, 2, 2]), ([1, 1, 1, 1], [1, 1, 1, 1, 1]),
        ([0, 0, 0, 0], [0, 0, 0, 0, 0]), ([1, 0, 0, 0], [0, 0, 0, 0, 0]),
        ([2, 2, 2, 1], [2, 2, 2, 2, 2]), ([2, 2, 2, 2], [2, 2, 2, 2, 1]), ([3, 2, 2, 2], [2, 2, 2, 2, 1])]
trem = np.array([r[0] for r in rows], float)
pigd = np.array([r[1] for r in rows], float)
out = phenotype_rule(trem, pigd)
for (a, b), o in zip(rows, out):
    tm, pm = np.mean(a), np.mean(b)
    ratio = tm / pm if pm > 0 else float("inf")
    print(f"tremor mean {tm:.2f}  PIGD mean {pm:.2f}  ratio {ratio:6.3f}  -> {name[int(o)]}")

# 2. simulated items: level -> mean item score (1.2 + level, rounded, clipped to 0..4)
rng = np.random.default_rng(0)
for level in (-1.5, 0.0, 1.0, 3.0):
    it = _items(np.full(20000, level), 5, rng)
    print(f"level {level:+.1f}: mean item {it.mean():.3f} (1.2 + level = {1.2 + level:.1f}); share at 0: {np.mean(it == 0):.3f}, at 4: {np.mean(it == 4):.3f}")

# 3. phenotype shares produced by simulate() from year-0 states
panel, info = family_a.simulate(default_truth(4), 5000, np.random.default_rng(1))
n_all = info["n_TD"] + info["n_PIGD"] + info["n_excluded_indeterminate"]
print(f"simulated 5000: TD {info['n_TD']}, PIGD {info['n_PIGD']}, indeterminate (dropped) {info['n_excluded_indeterminate']} "
      f"({info['n_excluded_indeterminate'] / n_all:.1%})")
y1 = np.array([y[0] for y in panel.y])
for g in (0, 1):
    print(f"  landmark mean score of {name[g]:4s}: tremor {np.nanmean(y1[panel.group == g, 0]):+.3f}, axial {np.nanmean(y1[panel.group == g, 1]):+.3f}")
p2, i2 = family_a.simulate(default_truth(4), 5000, np.random.default_rng(1), keep_indeterminate=True)
print("keep_indeterminate=True: TD", i2["n_TD"], "PIGD", i2["n_PIGD"], "excluded", i2["n_excluded_indeterminate"])

"""Section 1.31: screening_retest, ICC(2,1) from a screening-baseline repeat. Simulated participants:
a stable true score, measured twice before treatment with error, and followed for five more years on
the grid (scores rise by 3 points a year)."""
import numpy as np
import pandas as pd
from _setup import TMP  # noqa: F401  (import paths)

from pdx import ppmi
from pdx.transforms import NormalScores

rng = np.random.default_rng(7)
n = 300
true = rng.normal(0, 3, n)                                   # true tremor score, raw points around 10
obs = lambda shift: np.clip(np.round(10 + true + shift + rng.normal(0, 1.5, n)), 0, 40)
sc, bl = obs(0), obs(0)
ids = np.arange(1, n + 1)
v = pd.concat([pd.DataFrame({"PATNO": ids, "EVENT_ID": "SC", "date": pd.Timestamp("2015-01-15"), "motor_state": "untreated",
                             "tremor": sc, "axial": np.where(ids <= 40, 2.0, np.nan)}),
               pd.DataFrame({"PATNO": ids, "EVENT_ID": "BL", "date": pd.Timestamp("2015-02-15"), "motor_state": "untreated",
                             "tremor": bl, "axial": np.where(ids <= 40, 3.0, np.nan)})])
grid = pd.concat([pd.DataFrame({"PATNO": ids, "year": 0, "tremor": bl, "axial": rng.integers(0, 6, n).astype(float)})] +
                 [pd.DataFrame({"PATNO": ids, "year": y, "tremor": obs(3 * y), "axial": np.nan}) for y in range(1, 6)])
part = pd.DataFrame({"PATNO": ids, "primary": True})
res = ppmi.screening_retest(v, part, grid, domains=("tremor", "axial"))
print(res)

# the same numbers by hand, and what 1 - ICC is in units of the pooled scale
ns = NormalScores().fit(grid[["tremor"]].to_numpy(float))
a, b = ns.transform(sc[:, None])[:, 0], ns.transform(bl[:, None])[:, 0]
Y = np.column_stack([a, b])
grand = Y.mean()
msr = 2 * np.sum((Y.mean(1) - grand) ** 2) / (n - 1)
msc = n * np.sum((Y.mean(0) - grand) ** 2)
mse = (np.sum((Y - grand) ** 2) - msr * (n - 1) - msc) / (n - 1)
icc = (msr - mse) / (msr + mse + 2 * (msc - mse) / n)
V = Y.var(ddof=1)
print(f"MSR {msr:.4f}  MSC {msc:.4f}  MSE {mse:.4f}  ICC {icc:.4f}")
print(f"variance of the retest scores on the pooled scale: {V:.4f}  (the pooled grid scores: "
      f"{np.nanvar(ns.transform(grid[['tremor']].to_numpy(float))):.4f})")
print(f"1 - ICC = {1 - icc:.4f}   (1 - ICC) x V = {(1 - icc) * V:.4f}   MSE = {mse:.4f}   "
      f"MSE + (MSC - MSE)/n = {mse + (msc - mse) / n:.4f}")

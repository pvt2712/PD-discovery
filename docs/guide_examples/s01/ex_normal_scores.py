"""Sections 1.7-1.9: orient and the frozen normal-scores transform. Made-up scores."""
import numpy as np
from _setup import TMP  # noqa: F401  (import paths)
from scipy import stats

from pdx.transforms import REVERSED, NormalScores, orient

print("== 1.7 orient ==")
raw = np.array([[26.0, 90.0, 3.0], [21.0, 70.0, 5.0]])
print(orient(raw, ["MoCA", "schwab_england", "tremor"]))
print("REVERSED:", REVERSED)
already = np.array([[4.0, 10.0]])          # ppmi.py output: cognition = 30 - MoCA, independence = 100 - S&E
print("orient applied to ppmi.py's already-oriented columns:", orient(already, ["cognition", "independence"]))

print("\n== 1.8 fit: mid-ranks and Phi^-1((r - 1/2) / n) ==")
x = np.array([[0.0], [0.0], [0.0], [1.0], [2.0], [2.0], [5.0], [np.nan]])
ns = NormalScores().fit(x)
u, sc = ns.tables[0]
counts = np.array([3, 1, 2, 1])
upper = np.cumsum(counts)
mid = upper - (counts - 1) / 2
print("unique values u:", u, "| counts:", counts, "| upper ranks:", upper, "| mid-ranks:", mid)
print("p = (mid - 0.5)/n, n = 7:", np.round((mid - 0.5) / 7, 4))
print("scores:", np.round(sc, 4), "| by hand:", np.round(stats.norm.ppf((mid - 0.5) / 7), 4))
z = ns.transform(x)[:, 0]
print("transformed sample:", np.round(z, 4))
print("mean, variance of the transformed sample:", round(float(np.nanmean(z)), 4), round(float(np.nanvar(z)), 4))

print("\n== 1.9 transform (frozen) and inverse ==")
new = np.array([[-3.0], [0.5], [1.5], [3.5], [9.0], [np.nan]])
print("new values:     ", new[:, 0])
print("normal scores:  ", np.round(ns.transform(new)[:, 0], 4))
print("back (inverse): ", np.round(ns.inverse(ns.transform(new))[:, 0], 4))
print("inverse of z = -2, 0, 2:", np.round(ns.inverse(np.array([[-2.0], [0.0], [2.0]]))[:, 0], 4))

print("\n== reversing by -x or by 30 - x gives minus the same scores ==")
rng = np.random.default_rng(1)
moca = rng.integers(18, 31, size=200).astype(float)[:, None]
a = NormalScores().fit(moca).transform(moca)
b = NormalScores().fit(-moca).transform(-moca)
c = NormalScores().fit(30 - moca).transform(30 - moca)
print("max |NS(-x) + NS(x)| =", float(np.abs(a + b).max()), "| max |NS(30 - x) - NS(-x)| =", float(np.abs(c - b).max()))

print("\n== a floor effect: 70% zeros ==")
hall = np.r_[np.zeros(700), rng.integers(1, 5, size=300)].astype(float)[:, None]
zh = NormalScores().fit(hall).transform(hall)[:, 0]
print("distinct scores:", np.unique(np.round(zh, 4)), "| mean", round(float(zh.mean()), 4), "| variance", round(float(zh.var()), 4))
print("no ties, n = 1000, variance of the fitted sample's scores:",
      round(float(NormalScores().fit(np.arange(1000.0)[:, None]).transform(np.arange(1000.0)[:, None]).var()), 4))

print("\n== a column with no observed value ==")
empty = NormalScores().fit(np.array([[1.0, np.nan], [2.0, np.nan]]))
try:
    empty.transform(np.array([[1.5, 3.0]]))
except ValueError as e:
    print("transform -> ValueError:", e)

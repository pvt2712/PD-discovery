"""Sections 4.3-4.5: the Fisher-z test of partial correlation by hand, by pdx_fisherz and by
causal-learn's own fisherz; then test-wise deletion. Synthetic data only."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import numpy as np  # noqa: E402
from causallearn.utils.cit import CIT  # noqa: E402
from scipy import stats  # noqa: E402

from pdx import step1  # noqa: E402

rng = np.random.default_rng(0)
n = 400
s1, s2 = rng.normal(size=n), rng.normal(size=n)
x = 0.7 * s1 + rng.normal(size=n)
y = 0.5 * s1 - 0.4 * s2 + 0.12 * x + rng.normal(size=n)
data = np.column_stack([s1, s2, x, y])          # columns 0, 1 = S; 2 = X; 3 = Y

# 1. by hand: residualise X and Y on [1, S], correlate, Fisher's z
C = np.column_stack([np.ones(n), s1, s2])
ex = x - C @ np.linalg.lstsq(C, x, rcond=None)[0]
ey = y - C @ np.linalg.lstsq(C, y, rcond=None)[0]
r = ex @ ey / np.sqrt((ex @ ex) * (ey @ ey))
z = 0.5 * np.log((1 + r) / (1 - r))
stat = np.sqrt(n - 2 - 3) * abs(z)
p_hand = 2 * stats.norm.sf(stat)
print(f"by hand:      r = {r:.6f}, z = {z:.6f}, sqrt(n-|S|-3)|z| = {stat:.4f}, p = {p_hand:.6g}")

# 2. the registered pdx test, as FCI calls it: cit(X, Y, S)
step1._register_tests()
cit = CIT(data, "pdx_fisherz")
print(f"pdx_fisherz:  p = {cit(2, 3, [0, 1]):.6g}   (cache key '{list(cit.pvalue_cache)[-1]}')")
# 3. causal-learn's own Fisher-z (complete data only)
print(f"fisherz:      p = {CIT(data, 'fisherz')(2, 3, [0, 1]):.6g}")

# 4. test-wise deletion: 10% of each column missing at random
miss = data.copy()
miss[rng.random(miss.shape) < 0.10] = np.nan
cit = CIT(miss, "pdx_fisherz")
for S in ([], [0], [0, 1]):
    rows = ~np.isnan(miss[:, [2, 3] + S]).any(1)
    sub = miss[rows]
    Cs = np.column_stack([np.ones(rows.sum())] + [sub[:, k] for k in S])
    rx = sub[:, 2] - Cs @ np.linalg.lstsq(Cs, sub[:, 2], rcond=None)[0]
    ry = sub[:, 3] - Cs @ np.linalg.lstsq(Cs, sub[:, 3], rcond=None)[0]
    rr = rx @ ry / np.sqrt((rx @ rx) * (ry @ ry))
    ph = 2 * stats.norm.sf(np.sqrt(rows.sum() - len(S) - 3) * abs(np.arctanh(rr)))
    print(f"S = {str(S):7s} rows used = {rows.sum()},  pdx p = {cit(2, 3, S):.6g},  by hand on those rows = {ph:.6g}")
print("rows complete in all four columns:", int((~np.isnan(miss).any(1)).sum()))

# 5. too few rows: n <= |S| + 3 gives p = 1, which FCI reads as independence
tiny = data[:5]
print("5 rows, |S| = 2: p =", CIT(tiny, "pdx_fisherz")(2, 3, [0, 1]))

"""Sections 4.37-4.47: the tsLDiffPC/tsDCI re-implementation on two regimes of three linear
series with a known difference: (a) a new lagged link, (b) a new same-time link, (c) a
stronger same-time link.
Synthetic linear Gaussian data only."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import numpy as np  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from scipy import stats  # noqa: E402

from pdx.comparators import ts_ldiffpc as TS  # noqa: E402

rng = np.random.default_rng(0)
p, n = 3, 2000


def series(B, c=0.0):
    """x_j(t) = sum_k B[j, k] x_k(t-1) + c * x_1(t) [for j = 2] + noise; returns the window [x(t-1), x(t)]."""
    x = np.zeros((n + 101, p))
    for t in range(1, n + 101):
        x[t] = B @ x[t - 1] + rng.normal(size=p)
        x[t, 2] += c * x[t, 1]
    return np.column_stack([x[100:-1], x[101:]])     # columns 0..2 = lag block, 3..5 = current block


B1 = np.array([[0.5, 0.0, 0.0], [0.3, 0.5, 0.0], [0.0, 0.3, 0.5]])
B2 = B1.copy()
B2[2, 0] = 0.4                                         # (a) new link x0(t-1) -> x2(t) in regime 2
Z1, Z2 = series(B1), series(B2)

# one regression from the Gram matrix, against statsmodels; then the two-sample Wald test
G1, G2 = TS.gram(Z1), TS.gram(Z2)
b, se = TS.coef_se(*G1, 5, 0, [4])                    # x2(t) on [1, x0(t-1), x1(t)]
ols = sm.OLS(Z1[:, 5], sm.add_constant(Z1[:, [0, 4]])).fit()
print(f"coef_se: b = {b:.6f}, se = {se:.6f};  statsmodels: b = {ols.params[1]:.6f}, se = {ols.bse[1]:.6f}")
b2, se2 = TS.coef_se(*G2, 5, 0, [4])
print(f"wald_p: {TS.wald_p(G1, G2, 5, 0, [4]):.3g};  by hand: "
      f"{2 * stats.norm.sf(abs(b - b2) / np.hypot(se, se2)):.3g}  (b1 = {b:.3f}, b2 = {b2:.3f})")

print("pyCIPHOD's node order for d = 3:", TS._order(p))
t0 = time.time()
kept, ntest = TS.skeleton(Z1, Z2, p)
print(f"\n(a) skeleton: {ntest} tests, {time.time() - t0:.2f} s")
print("    retained lagged difference edges F[to, from]:\n", TS.lag_matrix(kept, p).astype(int))
print("    retained same-time pairs:", TS.contemporaneous_pairs(kept, p))
for variant in ("tsdci", "tsldiffpc"):
    edges, sep = TS.skeleton_with_sepsets(Z1, Z2, p, variant=variant)
    directed, undirected = TS.orient(edges, sep, Z1, Z2, p, variant=variant)
    print(f"    {variant:9s}: directed {sorted(directed)}, undirected {[sorted(e) for e in undirected]}, "
          f"root causes {TS.root_causes(directed, undirected, p)}")

for label, (c1, c2) in (("(b) a new same-time link x1(t) -> x2(t): 0 in regime 1, 0.8 in regime 2", (0.0, 0.8)),
                        ("(c) a stronger same-time link x1(t) -> x2(t): 0.5 in regime 1, 1.0 in regime 2", (0.5, 1.0))):
    Z1, Z2 = series(B1, c=c1), series(B1, c=c2)
    print(f"\n{label} (window nodes 4 -> 5):")
    for variant in ("tsdci", "tsldiffpc"):
        edges, sep = TS.skeleton_with_sepsets(Z1, Z2, p, variant=variant)
        directed, undirected = TS.orient(edges, sep, Z1, Z2, p, variant=variant)
        print(f"    {variant:9s}: edges {[sorted(e) for e in edges]}, directed {sorted(directed)}, "
              f"undirected {[sorted(e) for e in undirected]}, (heads, pairs) {TS.root_causes(directed, undirected, p)}")
    G1, G2 = TS.gram(Z1), TS.gram(Z2)
    S = list(sep.get((4, 5), ()))                      # the set that removed the edge (tsLDiffPC's run), if any
    print(f"    separating set recorded for (4, 5): {S if (4, 5) in sep else 'none (edge kept)'}")
    print(f"    with S = {S}: x2(t) on x1(t): b1 = {TS.coef_se(*G1, 5, 4, S)[0]:.3f}, b2 = {TS.coef_se(*G2, 5, 4, S)[0]:.3f}, "
          f"p = {TS.wald_p(G1, G2, 5, 4, S):.2g};  x1(t) on x2(t): b1 = {TS.coef_se(*G1, 4, 5, S)[0]:.3f}, "
          f"b2 = {TS.coef_se(*G2, 4, 5, S)[0]:.3f}, p = {TS.wald_p(G1, G2, 4, 5, S):.2g}")

"""Section 4.6: the participant-clustered test (pdx_cluster_fisherz) by hand and against
statsmodels, then the size of both tests under three null designs. Synthetic data only."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import numpy as np  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from causallearn.utils.cit import CIT  # noqa: E402
from scipy import stats  # noqa: E402

from pdx import step1  # noqa: E402

step1._register_tests()
rng = np.random.default_rng(1)


def clustered(G=60, m=5, icc=0.8):
    """X and Y independent given S, but each has a participant effect (G participants x m rows)."""
    who = np.repeat(np.arange(G), m)
    s = rng.normal(size=G * m)
    a, b = rng.normal(size=G)[who], rng.normal(size=G)[who]
    x = 0.5 * s + np.sqrt(icc) * a + np.sqrt(1 - icc) * rng.normal(size=G * m)
    y = 0.5 * s + np.sqrt(icc) * b + np.sqrt(1 - icc) * rng.normal(size=G * m)
    return np.column_stack([s, x, y]), who


# 1. one dataset: the code, by hand, and statsmodels' cluster-robust OLS
data, who = clustered()
p_code = CIT(data, "pdx_cluster_fisherz", clusters=who)(1, 2, [0])
Z = (data - data.mean(0)) / data.std(0)
C = np.column_stack([np.ones(len(Z)), Z[:, 0]])
ex = Z[:, 1] - C @ np.linalg.lstsq(C, Z[:, 1], rcond=None)[0]
ey = Z[:, 2] - C @ np.linalg.lstsq(C, Z[:, 2], rcond=None)[0]
beta = ex @ ey / (ex @ ex)
score = np.bincount(who, weights=ex * (ey - beta * ex))      # one score sum per participant
G = len(score)
V = score @ score / (ex @ ex) ** 2 * G / (G - 1)
print(f"code:        p = {p_code:.6f}")
print(f"by hand:     beta = {beta:.6f}, se = {np.sqrt(V):.6f}, z = {beta / np.sqrt(V):.4f}, "
      f"p = {2 * stats.norm.sf(abs(beta) / np.sqrt(V)):.6f}")
ols = sm.OLS(Z[:, 2], np.column_stack([np.ones(len(Z)), Z[:, 1], Z[:, 0]])).fit(
    cov_type="cluster", cov_kwds={"groups": who})
N, K = len(Z), 3
print(f"statsmodels: beta = {ols.params[1]:.6f}, se = {ols.bse[1]:.6f} (CR1 with (N-1)/(N-K)); "
      f"se x sqrt((N-K)/(N-1)) = {ols.bse[1] * np.sqrt((N - K) / (N - 1)):.6f}")
print(f"plain Fisher-z on the same data: p = {CIT(data, 'pdx_fisherz')(1, 2, [0]):.6f}")
p_swap = CIT(data[:, [0, 2, 1]], "pdx_cluster_fisherz", clusters=who)(1, 2, [0])
print(f"robust test with X and Y swapped (regression of X on Y): p = {p_swap:.6f}")

# 2. size at the 5% level, 400 datasets per design
t0 = time.time()
designs = {}
for name, k in (("independent rows", 0.0), ("participant clusters", 0.0),
                ("shared heteroscedasticity 0.4", 0.4), ("shared heteroscedasticity 0.8", 0.8)):
    rej = np.zeros(2)
    for _ in range(400):
        if name == "participant clusters":
            data, who = clustered()
        else:
            n = 300
            s = rng.normal(size=n)
            scale = np.exp(k * s)                  # residual SD of X and Y both grow with S
            data = np.column_stack([s, 0.5 * s + scale * rng.normal(size=n), 0.5 * s + scale * rng.normal(size=n)])
            who = np.arange(n)
        rej += [CIT(data, "pdx_fisherz")(1, 2, [0]) < 0.05,
                CIT(data, "pdx_cluster_fisherz", clusters=who)(1, 2, [0]) < 0.05]
    designs[name] = rej / 400
print(f"\nrejection rate under the null (400 datasets each, {time.time() - t0:.1f} s):")
print(f"{'design':28s} {'pdx_fisherz':>12s} {'pdx_cluster_fisherz':>20s}")
for name, (a, b) in designs.items():
    print(f"{name:28s} {a:12.3f} {b:20.3f}")

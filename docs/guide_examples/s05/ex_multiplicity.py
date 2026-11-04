"""Sections 5.43-5.46: Holm, Benjamini-Hochberg and Benjamini-Yekutieli by hand and with the code;
max-|t| simultaneous intervals; the intersection-union verdict. Synthetic numbers only."""
import numpy as np

import _fixture  # noqa: F401  (import path)
from pdx.multiplicity import bh, by, holm, holm_adjusted, iut_similar, simultaneous_intervals

p = np.array([0.012, 0.0004, 0.049, 0.031, 0.20, 0.0089, 0.74, 0.033])
m, alpha = len(p), 0.05
o = np.argsort(p)
print("sorted p        :", p[o])
print("Holm thresholds :", np.round(alpha / (m - np.arange(m)), 5))
print("BH thresholds   :", np.round(alpha * np.arange(1, m + 1) / m, 5))
c = np.sum(1.0 / np.arange(1, m + 1))
print(f"BY constant c(m) = sum 1/j = {c:.4f}; BY thresholds:", np.round(alpha * np.arange(1, m + 1) / (m * c), 5))
print("rejected, in the original order:")
print("  p    :", p)
print("  holm :", holm(p).astype(int))
print("  bh   :", bh(p).astype(int))
print("  by   :", by(p).astype(int))
print("  Holm-adjusted p:", np.round(holm_adjusted(p), 4))

print("\nsimultaneous 90% intervals over 6 correlated estimates (max-|t|)")
rng = np.random.default_rng(1)
C = 0.5 * np.eye(6) + 0.5
draws = rng.multivariate_normal(np.zeros(6), 0.04 ** 2 * C, size=20000)
est = np.array([0.02, -0.01, 0.03, 0.00, 0.05, -0.04])
lo, hi = simultaneous_intervals(est, draws, 0.90)
sd = draws.std(0, ddof=1)
tmax = np.max(np.abs(draws - draws.mean(0)) / sd, axis=1)
print(f"  critical value c = {np.quantile(tmax, 0.90):.3f} (marginal 90% would use 1.645)")
print("  simultaneous lo:", np.round(lo, 3))
print("  simultaneous hi:", np.round(hi, 3))
print("  joint coverage of the centred draws:", np.mean(np.all(np.abs(draws) <= np.quantile(tmax, 0.90) * sd, axis=1)).round(3))
mlo, mhi = est - 1.645 * sd, est + 1.645 * sd
print("  marginal 90% lo:", np.round(mlo, 3))
print("  marginal 90% hi:", np.round(mhi, 3))
for eps in (0.10, 0.12, 0.15):
    print(f"  eps {eps:.2f}: iut_similar(marginal) = {iut_similar(mlo, mhi, eps)}, iut_similar(simultaneous) = {iut_similar(lo, hi, eps)}")

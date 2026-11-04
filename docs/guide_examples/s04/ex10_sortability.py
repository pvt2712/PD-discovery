"""Section 4.16: variance- and R^2-sortability of a known chain, raw and standardised.
Synthetic linear Gaussian data only."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import numpy as np  # noqa: E402

from pdx.step1 import sortability  # noqa: E402

rng = np.random.default_rng(2)
n = 5000
x0 = rng.normal(size=n)
x1 = 1.0 * x0 + rng.normal(size=n)
x2 = 1.0 * x1 + rng.normal(size=n)
raw = np.column_stack([x2, x0, x1])                # columns shuffled on purpose: 0 = x2, 1 = x0, 2 = x1
directed = np.zeros((3, 3), bool)
directed[1, 2] = directed[2, 0] = True             # x0 -> x1 -> x2 in the usual [from, to] convention
std = (raw - raw.mean(0)) / raw.std(0)
print("variances, raw:", raw.var(0).round(2))
print("variances, standardised (all 1 up to rounding):", [f"{v:.17f}" for v in std.var(0)])
for name, T in (("raw", raw), ("standardised", std)):
    print(f"{name:13s} var-sortability = {sortability(T, directed, 'var'):.3f}, "
          f"R2-sortability = {sortability(T, directed, 'r2'):.3f}")

"""Section 4.9: step1_table on a hand-made panel of three participants (two domains, four baseline
covariates, grid years with one missed year, one missing score). Synthetic numbers only."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import numpy as np  # noqa: E402

from pdx.data import Panel  # noqa: E402
from pdx.step1 import Step1Config, step1_table  # noqa: E402

np.set_printoptions(precision=3, suppress=True, linewidth=120)
y = [np.array([[0.1, 0.2], [0.3, 0.5], [0.6, 0.9]]),             # participant 0: grid years 1, 2, 3
     np.array([[1.0, 1.1], [1.2, np.nan], [1.9, 2.0]]),           # participant 1: years 1, 2, 4 (year 3 missed)
     np.array([[-0.5, -0.4], [-0.2, -0.3]])]                      # participant 2: years 1, 2
t = [np.array([0.0, 1.1, 2.0]), np.array([0.0, 0.9, 3.0]), np.array([0.0, 1.3])]
year = [np.array([1, 2, 3]), np.array([1, 2, 4]), np.array([1, 2])]
z = np.array([[-1.0, 0.2, 0.5, -0.1],                             # era, age at onset, sex, duration
              [0.0, -0.3, -0.5, 0.4],
              [1.0, 0.1, 0.0, -0.3]])
dose = [np.array([0.0, 2.0, 3.0]), np.array([1.0, 1.0, 4.0]), np.array([0.0, 0.0])]
panel = Panel(y, t, group=[0, 1, 0], z=z, dose=dose, year=year)

X, tiers, who, cols, exo = step1_table(panel, Step1Config())
print("columns:", cols)
print("tiers:  ", tiers.tolist(), "  exogenous columns:", exo)
print("row participant:", who.tolist())
print(X)

X, tiers, who, cols, exo = step1_table(panel, Step1Config(include_dose=True))
print("\nwith include_dose: columns", cols)
print("                   tiers  ", tiers.tolist())

raw = step1_table(panel, Step1Config())[0]
X, tiers, who, cols, exo = step1_table(panel, Step1Config(add_interval=True))
print("\nwith add_interval: columns", cols, "(the interval is partialled out, then dropped)")
iv = np.array([1.1, 0.9, 0.9, 1.3])                              # the four kept intervals
for k in (4, 6):
    ok = ~np.isnan(raw[:, k])
    slope = np.polyfit(iv[ok], raw[ok, k], 1)[0]
    print(f"  {cols[k]}: before {raw[:, k]}, after {X[:, k]}; slope on the interval {slope:.3f};",
          "after = before - slope x interval:", np.allclose(X[ok, k], raw[ok, k] - slope * iv[ok]))

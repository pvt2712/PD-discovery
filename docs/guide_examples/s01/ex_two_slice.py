"""Section 1.5: two_slice_table, the pooled year-to-year transitions of Step 1.
Two made-up participants, d = 2 domains, one baseline covariate."""
import numpy as np
from _setup import TMP  # noqa: F401  (import paths)

from pdx.data import Panel, two_slice_table

nan = np.nan
p = Panel(y=[np.array([[0.1, 0.2], [0.3, 0.4], [0.5, nan], [0.9, 1.0]]),   # years 1, 2, 3, 5 (year 4 missed)
             np.array([[1.0, 1.1], [1.2, 1.3]])],                         # years 1, 2
          t=[[1.0, 2.1, 2.9, 5.0], [1.1, 2.0]], group=[0, 1],
          dose=[[0, 0, 3, 4], [1, 1]], year=[[1, 2, 3, 5], [1, 2]])
base = np.array([[-0.5], [0.5]])

X, tiers, who, cols = two_slice_table(p, base)
print("columns:", cols)
print("tiers:  ", tiers.tolist())
print("rows from participant:", who.tolist())
print(X)

X, tiers, who, cols = two_slice_table(p, base, add_interval=True, include_dose=True, drop_incomplete=True)
print("\nwith interval and dose, complete rows only:")
print("columns:", cols)
print("tiers:  ", tiers.tolist())
print(X)

# without grid years, consecutive means an interval of at most max_dt = 1.5 years
p.year = None
X, _, who, _ = two_slice_table(p, base)
print("\nwithout grid years: rows from participant", who.tolist(), "(the 2.1-year step 2.9 -> 5.0 is dropped)")

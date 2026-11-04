"""Sections 1.32-1.35: disclosure control in wp0.py (made-up counts)."""
import itertools

import numpy as np
import pandas as pd
from _setup import TMP  # noqa: F401  (import paths)

from pdx import wp0

print("== 1.32 constants ==")
print("TD_CUT, PIGD_CUT:", wp0.TD_CUT, wp0.PIGD_CUT, "| QUANTILE_MIN_N:", wp0.QUANTILE_MIN_N, "| SITE_BANDS:", wp0.SITE_BANDS)
print("Columns() defaults:", wp0.Columns())

print("\n== 1.33 protect_1way ==")
print("_small(v, 5) for v = 0..6:", [wp0._small(v, 5) for v in range(7)])
for counts in ({"TD": 120, "PIGD": 64, "indeterminate": 3},
               {"TD": 120, "PIGD": 4, "indeterminate": 3},
               {"TD": 120, "PIGD": 64, "indeterminate": 0},
               {"1": 2, "2": 0, "3": 0}):
    print(f"  {counts} -> {wp0.protect_1way(counts, 5)}")

print("\n== 1.34 protect_2way ==")
t = pd.DataFrame([[40, 3, 12], [25, 0, 9], [30, 6, 1]], index=["A", "B", "C"], columns=["x", "y", "z"])
print(t.to_string())
res = wp0.protect_2way(t, 5)
for key, val in res.items():
    print(f"  {key}: {val}")
shown = sum(v for row in res["cells"].values() for v in row.values() if isinstance(v, int))
print("  what an outsider can say: the suppressed cells sum to", res["total"], "-", shown, "=", res["total"] - shown)


def intervals(res, k=5):
    """Every value each suppressed cell can take, given its label ('<k' or 'suppressed') and the
    published grand total (the only published constraint on the suppressed cells)."""
    labels = {(r, c): v for r, row in res["cells"].items() for c, v in row.items() if isinstance(v, str)}
    rest = res["total"] - sum(v for row in res["cells"].values() for v in row.values() if isinstance(v, int))
    feasible = {key: set() for key in labels}
    ranges = [range(1, k) if lab == f"<{k}" else range(k, rest + 1) for lab in labels.values()]
    for combo in itertools.product(*ranges):
        if sum(combo) == rest:
            for key, x in zip(labels, combo):
                feasible[key].add(x)
    return {f"{key} {labels[key]!r}": (min(xs), max(xs)) for key, xs in feasible.items()}


print("\n  interval left for each suppressed cell, given its label and the published total:")
print("  ", intervals(res))
print("\n  a one-row table:")
small = pd.DataFrame([[1, 5]], index=["PPMI 2.0"], columns=["PIGD", "TD"])
res_small = wp0.protect_2way(small, 5)
print("  ", res_small)
print("   intervals:", intervals(res_small))
print("\n  a one-way count published next to its total (as audit does for the transitions):")
print("  ", {"total": 3, "by_cohort": wp0.protect_1way({"PPMI 1.0": 3, "PPMI 2.0": 0}, 5)})

print("\n== 1.35 _share, _quantiles, _count_series ==")
print("_share(3, 40, 5) =", wp0._share(3, 40, 5), "| _share(1, 4, 5) =", wp0._share(1, 4, 5))
rng = np.random.default_rng(0)
print("_quantiles(49 values):", wp0._quantiles(pd.Series(rng.normal(size=49))))
q = wp0._quantiles(pd.Series(np.r_[rng.lognormal(size=60), np.inf, np.nan]))
print("_quantiles(60 values + inf + NaN):", {k: round(v, 3) for k, v in q.items()})
print("_count_series:", wp0._count_series(pd.Series(["OFF", "untreated", None, "OFF", "ON only"])))

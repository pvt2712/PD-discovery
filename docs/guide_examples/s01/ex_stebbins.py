"""Section 1.23: stebbins, the TD/PIGD rule of Stebbins et al. 2013. Hand-made item scores."""
from fractions import Fraction

import numpy as np
import pandas as pd
from _setup import TMP  # noqa: F401  (import paths)

from pdx import ppmi

T, P = ppmi.STEBBINS_TREMOR, ppmi.STEBBINS_PIGD          # 11 and 5 items


def row(tremor, pigd, state="untreated"):
    r = dict(zip(T, tremor))
    r.update(zip(P, pigd))
    r["motor_state"] = state
    return r


rows = {"A clear TD": row([2, 1] + [1] * 9, [0, 1, 0, 0, 0]),
        "B clear PIGD": row([0] * 11, [2, 1, 2, 1, 1]),
        "C between": row([1] * 11, [1, 1, 1, 1, 1]),
        "D on 1.15": row([3, 3, 3, 2, 2, 2, 2, 2, 2, 2, np.nan], [2, 2, 2, 2, 2]),
        "E on 0.90": row([1] * 9 + [0, np.nan], [1, 1, 1, 1, 1]),
        "F PIGD mean 0": row([1] + [0] * 10, [0] * 5),
        "G all zero": row([0] * 11, [0] * 5),
        "H 3 tremor blank": row([2] * 8 + [np.nan] * 3, [1] * 5),
        "I ON exam": row([2] * 11, [0] * 5, state="ON only")}
v = pd.DataFrame(rows.values(), index=rows.keys())
res = ppmi.stebbins(v)
res["tremor_mean"] = v[T].mean(axis=1).round(4)
res["pigd_mean"] = v[P].mean(axis=1).round(4)
print(res[["tremor_mean", "pigd_mean", "td_pigd_ratio", "phenotype"]].to_string())

# exact arithmetic against the code's floating point, over every possible item sum
mism, total, on_cut = 0, 0, 0
for nT in (9, 10, 11):
    for nP in (4, 5):
        for sT in range(4 * nT + 1):
            for sP in range(1, 4 * nP + 1):
                r = (sT / nT) / (sP / nP)
                R = Fraction(sT, nT) / Fraction(sP, nP)
                exact = "TD" if R >= Fraction(23, 20) else "PIGD" if R <= Fraction(9, 10) else "ind"
                flt = "TD" if r >= ppmi.TD_CUT else "PIGD" if r <= ppmi.PIGD_CUT else "ind"
                total += 1
                on_cut += R in (Fraction(23, 20), Fraction(9, 10))
                mism += exact != flt
print(f"\nexact vs floating point: {total} item-sum combinations, {on_cut} exactly on a cut-off, {mism} disagreements")

"""Section 1.24: clinical_subtype, the provisional Fereshtehnejad rule. Twelve made-up baseline rows."""
import numpy as np
import pandas as pd
from _setup import TMP  # noqa: F401  (import paths)

from pdx import ppmi

motor_total = [10, 12, 14, 16, 18, 20, 22, 24, 26, 30, 34, 40]
base = pd.DataFrame({"function": motor_total, "NP2TRMR": 0, "NP2WALK": 0, "NP2FREZ": 0, "tremor": 0,
                     "brady_rigidity": 0, "axial": 0,
                     "cognition": [1, 2, 3, 4, 5, 6, 2, 3, 9, 1, 2, 3],
                     "rbd": [2, 2, 3, 3, 3, 3, 4, 4, 2, 4, 6, 8],
                     "autonomic": [5, 6, 7, 8, 20, 21, 9, 10, 22, 4, 5, 6]},
                    index=pd.Index(range(101, 113), name="PATNO"))
base.loc[112, "tremor"] = np.nan               # one motor component missing: not classifiable
q = {c: base[c].quantile(0.75) for c in ["cognition", "rbd", "autonomic"]}
motor = base[["function", "NP2TRMR", "NP2WALK", "NP2FREZ", "tremor", "brady_rigidity", "axial"]].sum(axis=1, min_count=7)
print("75th percentiles: motor", motor.quantile(0.75), "| non-motor", q)
out = pd.DataFrame({"motor": motor, "cognition": base.cognition, "rbd": base.rbd, "autonomic": base.autonomic,
                    "subtype": ppmi.clinical_subtype(base)})
out["above"] = [("M" if m > motor.quantile(0.75) else "-") + "".join(
    k[0].upper() if base.loc[i, k] > q[k] else "-" for k in q) for i, m in motor.items()]
print(out.to_string())
print("rbd values at the 75th percentile (", q["rbd"], "): count =", int((base.rbd == q["rbd"]).sum()),
      "-> they are 'not above', so they can be mild motor-predominant")

"""Section 1.53: minimal reproductions for some of the notes for the author. Synthetic data only."""
import json

import numpy as np
import pandas as pd
import torch
from _setup import mock_download

from pdx import ppmi, wp0
from pdx.engine.model import Layout, ModelSpec, treat_effect

print("== note 5: with on = 0 the current-dose term is zero, whatever the dose ==")
spec = ModelSpec(d=2, support=np.eye(2, dtype=bool), treat_mask=np.array([True, True]), treatment="emax")
lay = Layout(spec, "null")
x = torch.zeros(lay.size, dtype=torch.float64)
x[lay.slices["kappa"][0]] = 1.0                       # short-duration effect 1 per domain (both groups)
dose = torch.tensor([[0.0, 3.0, 9.0]])                # 0, 300, 900 mg LEDD
expo = torch.zeros(1, 3)
for on in (0.0, 1.0):
    h = treat_effect(x, lay, dose, expo, torch.full((1, 3), on), torch.tensor([0]))
    print(f"  on = {on:.0f}: h for domain 0 at doses 0, 3, 9 = {h[0, :, 0].numpy().round(3) + 0.0}")

print("\n== note 17: the 12 +- 3 months rule on dates placed on the 15th ==")
for a, b in [("2015-02-15", "2015-11-15"), ("2015-03-15", "2015-12-15"), ("2015-01-15", "2016-04-15"),
             ("2015-05-15", "2016-08-15")]:
    gap = (pd.Timestamp(b) - pd.Timestamp(a)).days / 365.25
    print(f"  {a} -> {b}: {gap:.4f} years, inside [0.75, 1.25]: {0.75 <= gap <= 1.25}")

print("\n== note 21: counts that are published exactly ==")
rows = [{"participant": p, "cohort": "PPMI 1.0", "year": y, "date_years": float(y), "phenotype": "TD",
         "td_pigd_ratio": 2.0, "mode": "in person", "treated": 0, "dose_change": 0.0, "motor_state": "OFF", "tremor": 1.0}
        for p in range(60) for y in ([0, 1, 3] if p == 0 else [0, 1, 2, 3])]
rep = wp0.audit(pd.DataFrame(rows), wp0.Columns(domains=["tremor"]), min_cell=5)
print("  across_a_missed_year:", rep["transitions_after_landmark"]["across_a_missed_year"])
v = pd.DataFrame({"PATNO": [1, 2, 3] * 2, "EVENT_ID": ["SC"] * 3 + ["BL"] * 3, "motor_state": "untreated",
                  "date": [pd.Timestamp("2015-01-15")] * 3 + [pd.Timestamp("2015-02-15")] * 3,
                  "tremor": [3.0, 4, 5, 3, 5, 5], "brady_rigidity": 1.0, "axial": 1.0, "function": 1.0})
grid = pd.DataFrame({"PATNO": [1, 2, 3], "tremor": [3.0, 5, 5], "brady_rigidity": 1.0, "axial": 1.0, "function": 1.0})
part = pd.DataFrame({"PATNO": [1, 2, 3], "primary": True})
print("  screening_retest:", ppmi.screening_retest(v, part, grid))
print("  json.dumps of a non-finite ICC:", json.dumps({"icc_2_1": float("nan")}))

print("\n== note 29: two branches the unit tests never reach ==")
raw = mock_download()
u2p = raw / "Motor___MDS-UPDRS" / "MDS_UPDRS_Part_II__Patient_Questionnaire_01Jan2026.csv"
u2 = pd.read_csv(u2p)
u2[~((u2.PATNO == 2) & (u2.EVENT_ID == "BL"))].to_csv(u2p, index=False)      # participant 2: no Part II at BL
sep = raw / "Motor___MDS-UPDRS" / "Modified_Schwab___England_Activities_of_Daily_Living_01Jan2026.csv"
se = pd.read_csv(sep)
se.loc[(se.PATNO == 4) & (se.EVENT_ID == "V06"), "MSEADLG"] = 70                  # participant 4: S&E 70 at year 2
se.to_csv(sep, index=False)
out = ppmi.build(raw)
part = out["participants"].set_index("PATNO")
print("  participant 2:", part.loc[2, "exclusion"])
w = out["wp0"]
print("  participant 4, event_24m on the year-0 row:", w[(w.participant == "4") & (w.year == 0)].event_24m.iloc[0])

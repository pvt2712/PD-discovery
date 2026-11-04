"""Sections 1.29-1.30: analysis_panel on the repository's mock download (made-up participants), with
three edits: participant 12's year-2 examination is ON only, participant 3 has no diagnosis dates, and
ages at enrolment differ."""
import numpy as np
import pandas as pd
from _setup import mock_download

from pdx import ppmi

raw = mock_download()
u3p = raw / "Motor___MDS-UPDRS" / "MDS-UPDRS_Part_III_01Jan2026.csv"
u3 = pd.read_csv(u3p)
u3.loc[(u3.PATNO == 12) & (u3.EVENT_ID == "V06"), "PDSTATE"] = "ON"
u3.to_csv(u3p, index=False)
dxp = raw / "Medical_History" / "PD_Diagnosis_History_01Jan2026.csv"
dx = pd.read_csv(dxp)
dx.loc[dx.PATNO == 3, ["SXDT", "PDDXDT"]] = None
dx.to_csv(dxp, index=False)
stp = raw / "_Subject_Characteristics" / "Participant_Status_01Jan2026.csv"
st = pd.read_csv(stp)
st["ENROLL_AGE"] = st.PATNO.map({1: 70.0, 2: 55.0, 4: 66.0, 6: 58.0}).fillna(62.0)    # ages that differ
st.to_csv(stp, index=False)

out = ppmi.build(raw)
print("== 1.29 ==")
print("CORE_DOMAINS:", ppmi.CORE_DOMAINS, "\nTREATED_DOMAINS:", ppmi.TREATED_DOMAINS)

print("\n== 1.30 analysis_panel (default: OFF or untreated examinations only) ==")
panel, ns, info = ppmi.analysis_panel(out["participants"], out["grid"])
print("participants:", panel.n, "| d:", panel.d, "| p:", panel.p, "| group:", panel.group.tolist())
print("visits per participant:", [len(t) for t in panel.t], "| first grid year of each:", [int(y[0]) for y in panel.year])
print("t of participant 0 (years since enrolment):", panel.t[0])
print("z columns:", info["covariates"])
print(np.round(panel.z, 3))
print("column means of z:", np.round(panel.z.mean(0), 12))
i12 = panel.n - 1
print("participant 12: dose", panel.dose[i12], "| exposure", panel.exposure[i12], "| on", panel.on[i12])
print("participant 12, scores at year 2 (tremor, brady_rigidity, axial):", panel.y[i12][1, :3])
print("info:", {key: info[key] for key in ("treated_domains", "motor_on")}, "| counts:", info["counts"])

print("\n== with motor_on=True (the D5 option) ==")
panel_on, ns_on, _ = ppmi.analysis_panel(out["participants"], out["grid"], motor_on=True)
print("participant 12: on", panel_on.on[i12], "| year-2 scores (tremor, brady_rigidity, axial):", panel_on.y[i12][1, :3])
print("tremor table, default:", ns.tables[0][0], "| with motor_on:", ns_on.tables[0][0])

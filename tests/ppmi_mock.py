"""A tiny mock of the PPMI download (same folders, file names and columns the pipeline reads),
with made-up participants. For tests only; no PPMI data."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pdx import ppmi

STAMP = "_01Jan2026.csv"


def _write(root: Path, folder: str, name: str, rows: list, columns: list) -> None:
    (root / folder).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(root / folder / (name + STAMP), index=False)


def make(root: Path) -> None:
    gen = ["ENRLLRRK2", "ENRLGBA", "ENRLSNCA", "ENRLPRKN", "ENRLPINK1", "ENRLOTHGV"]
    status, demo, dx, concl = [], [], [], []
    u3_cols = ["PATNO", "EVENT_ID", "PAG_NAME", "INFODT", "PDSTATE", "PDTRTMNT"] + \
        list(dict.fromkeys(ppmi.TREMOR_III + ppmi.BRADY_RIGID + ppmi.AXIAL))
    u2_cols = ["PATNO", "EVENT_ID", "INFODT"] + ppmi.FUNCTION_II + ["NP2TRMR", "NP2WALK", "NP2FREZ"]
    u3, u2, u1, u4, se, moca, gds, scopa, ess, rbd = ([] for _ in range(10))
    gds_cols = ["PATNO", "EVENT_ID", "INFODT"] + ppmi.GDS_YES + ppmi.GDS_NO
    sc_cols = ["PATNO", "EVENT_ID", "INFODT"] + [f"SCAU{i}" for i in range(1, 22)]
    ess_cols = ["PATNO", "EVENT_ID", "INFODT"] + [f"ESS{i}" for i in range(1, 9)]
    rbd_cols = ["PATNO", "EVENT_ID", "INFODT"] + ppmi.RBDSQ_ITEMS + ppmi.RBDSQ_NEURO
    ledd = []
    rng = np.random.default_rng(0)

    def visit(p, ev, date, td, treated=0, state=None, pag="NUPDRS3"):
        tremor = 2 if td else 0
        pigd = 0 if td else 2
        row = {c: 1 for c in u3_cols[6:]}
        for c in ppmi.TREMOR_III:
            row[c] = tremor
        for c in ("NP3GAIT", "NP3FRZGT", "NP3PSTBL"):
            row[c] = pigd
        u3.append([p, ev, pag, date, state, treated] + [row[c] for c in u3_cols[6:]])
        r2 = {c: 1 for c in ppmi.FUNCTION_II}
        u2.append([p, ev, date] + [r2[c] for c in ppmi.FUNCTION_II] + [tremor, pigd, pigd])
        u1.append([p, ev, date, 0])
        u4.append([p, ev, date, 0 if not treated else 2])
        se.append([p, ev, date, 90])
        moca.append([p, ev, date, 27])
        gds.append([p, ev, date] + [0] * len(ppmi.GDS_YES) + [1] * len(ppmi.GDS_NO))
        scopa.append([p, ev, date] + [int(rng.integers(0, 2)) for _ in range(21)])
        ess.append([p, ev, date] + [1] * 8)
        rbd.append([p, ev, date] + [0] * len(ppmi.RBDSQ_ITEMS) + [0] * len(ppmi.RBDSQ_NEURO))

    for p in range(1, 13):
        era2 = p == 12
        enroll = "01/2021" if era2 else "01/2015"
        flags = [1 if (p == 9 and f == "ENRLLRRK2") else 0 for f in gen]
        status.append([p, 1, enroll, 62.0, "Enrolled"] + flags)
        demo.append([p, p % 2])
        dx.append([p, "06/2014" if not era2 else "06/2020", "09/2014" if not era2 else "09/2020"])
        td = p % 2 == 1
        y0, y1, y2 = ("01/2021", "01/2022", "01/2023") if era2 else ("01/2015", "01/2016", "01/2017")
        if p != 11:
            visit(p, "BL", y0, td, treated=1 if p == 10 else 0, state="OFF" if p == 10 else None)
        if p == 3:
            visit(p, "R04", "01/2016", td)                     # remote, on the anniversary
            visit(p, "V04", "03/2016", td)                     # in person, two months later: chosen
        elif p == 12:
            visit(p, "V04", y1, td, treated=1, state="OFF")
        else:
            visit(p, "V04", y1, td)
        if p != 5:
            visit(p, "V06", y2, td, treated=1 if p == 12 else 0, state="OFF" if p == 12 else None)
        if p == 5:
            concl.append([p, "06/2016", 1])
    status.append([13, 1, "01/2021", 60.0, "Screen failed"] + [0] * len(gen))     # never enrolled
    ledd.append([12, "LOG", "Sinemet", 100, 3, 100, "06/2021", None, 300])
    ledd.append([10, "LOG", "Sinemet", 100, 3, 100, "06/2014", None, 300])
    _write(root, "_Subject_Characteristics", "Participant_Status", status,
           ["PATNO", "COHORT", "ENROLL_DATE", "ENROLL_AGE", "ENROLL_STATUS"] + gen)
    _write(root, "_Subject_Characteristics", "Demographics", demo, ["PATNO", "SEX"])
    _write(root, "Medical_History", "PD_Diagnosis_History", dx, ["PATNO", "SXDT", "PDDXDT"])
    _write(root, "Study_Enrollment", "Conclusion_of_Study_Participation", concl, ["PATNO", "WDDT", "WDDEATH"])
    _write(root, "Motor___MDS-UPDRS", "MDS-UPDRS_Part_III", u3, u3_cols)
    _write(root, "Motor___MDS-UPDRS", "MDS_UPDRS_Part_II__Patient_Questionnaire", u2, u2_cols)
    _write(root, "Motor___MDS-UPDRS", "MDS-UPDRS_Part_I", u1, ["PATNO", "EVENT_ID", "INFODT", "NP1HALL"])
    _write(root, "Motor___MDS-UPDRS", "MDS-UPDRS_Part_IV__Motor_Complications", u4,
           ["PATNO", "EVENT_ID", "INFODT", "NP4TOT"])
    _write(root, "Motor___MDS-UPDRS", "Modified_Schwab___England_Activities_of_Daily_Living", se,
           ["PATNO", "EVENT_ID", "INFODT", "MSEADLG"])
    _write(root, "Non-motor_Assessments", "Montreal_Cognitive_Assessment__MoCA_", moca,
           ["PATNO", "EVENT_ID", "INFODT", "MCATOT"])
    _write(root, "Non-motor_Assessments", "Geriatric_Depression_Scale__Short_Version_", gds, gds_cols)
    _write(root, "Non-motor_Assessments", "SCOPA-AUT", scopa, sc_cols)
    _write(root, "Non-motor_Assessments", "Epworth_Sleepiness_Scale", ess, ess_cols)
    _write(root, "Non-motor_Assessments", "REM_Sleep_Behavior_Disorder_Screening_Questionnaire", rbd, rbd_cols)
    _write(root, "Medical_History", "LEDD_Concomitant_Medication_Log", ledd,
           ["PATNO", "EVENT_ID", "LEDTRT", "LEDDOSE", "LEDDOSFRQ", "LEDDSTRMG", "STARTDT", "STOPDT", "LEDD"])

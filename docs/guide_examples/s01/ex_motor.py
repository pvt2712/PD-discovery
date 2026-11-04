"""Section 1.18: _motor, the Part III examination of each visit. A hand-made Part III table of
made-up participants, written in the PPMI layout."""
import numpy as np
import pandas as pd
from _setup import TMP, write_table

from pdx import ppmi

allitems = list(dict.fromkeys(ppmi.TREMOR_III + ppmi.BRADY_RIGID + ppmi.AXIAL))


def rec(p, ev, pag, date, pdstate, pdtrtmnt, value=1, missing=()):
    r = {c: value for c in allitems}
    r.update({c: np.nan for c in missing})
    return {"PATNO": p, "EVENT_ID": ev, "PAG_NAME": pag, "INFODT": date, "PDSTATE": pdstate, "PDTRTMNT": pdtrtmnt, **r}


rows = [rec(1, "BL", "NUPDRS3", "01/2015", None, 0),                     # untreated
        rec(1, "V04", "NUPDRS3", "01/2016", "OFF", 1, value=2),          # OFF, recorded
        rec(1, "V04", "NUPDRS3A", "01/2016", None, 1),                   # page NUPDRS3A: read as ON
        rec(2, "BL", "NUPDRS3", "01/2015", None, None),                  # PDTRTMNT blank: counted as untreated
        rec(2, "V04", "NUPDRS3", "01/2016", None, 1),                    # treated, no state: unrecorded
        rec(3, "V04", "NUPDR3ON", "01/2016", None, 1, missing=ppmi.TREMOR_III[:3]),   # ON exam, 3 tremor items blank
        rec(4, "V04", "NUPDR3OF", "01/2016", None, 1, missing=allitems[:7]),           # OFF page with 24 of 31 items
        rec(5, "BL", "NUPDRS3", "01/2015", None, 0, missing=allitems[:6]),             # 25 of 31 items: an exam
        rec(5, "BL", "NUPDRS3", "02/2015", None, 0, value=0),                          # a second, complete record
        rec(6, "BL", "NUPDRS3", "01/2015", None, 0, value=101)]                        # every item 101 (unable to rate)
raw = TMP / "raw_motor"
write_table(raw, "Motor___MDS-UPDRS", "MDS-UPDRS_Part_III", pd.DataFrame(rows))
m = ppmi._motor(raw, {1, 2, 3, 4, 5, 6}).sort_values(["PATNO", "EVENT_ID"])
print(m[["PATNO", "EVENT_ID", "date", "tremor", "brady_rigidity", "axial", "tremor_on", "axial_on", "pdtrtmnt_any",
         "motor_state"]].to_string(index=False))
print("\nby hand, participant 1 V04 OFF: tremor 10 x 2 =", 10 * 2, "| brady 16 x 2 =", 16 * 2, "| axial 5 x 2 =", 5 * 2)
print("participant 5 BL keeps the complete record (31 items, all 0), not the earlier one with 25 items")

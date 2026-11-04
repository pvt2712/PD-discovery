"""Section 1.19: _ledd, the LEDD in force on each visit date. A hand-made medication log of one
made-up participant, written in the PPMI layout."""
import pandas as pd
from _setup import TMP, write_table

from pdx import ppmi

log = pd.DataFrame(
    [[1, "LOG", "Sinemet", 100, 3, 100, "01/2016", "03/2017", "300"],        # 300 mg/day until 03/2017
     [1, "LOG", "Sinemet", 100, 4, 100, "03/2017", None, "400"],             # switched to 400 mg/day in 03/2017
     [1, "LOG", "Entacapone", 200, 3, 200, "06/2017", None, "LD x 0.33"],    # COMT inhibitor: a multiplier
     [1, "LOG", "Pramipexole", 0.5, 3, 0.5, "06/2017", None, "150"],
     [1, "LOG", "Pramipexole", 0.5, 3, 0.5, "06/2017", None, "150"]],        # the same entry twice
    columns=["PATNO", "EVENT_ID", "LEDTRT", "LEDDOSE", "LEDDOSFRQ", "LEDDSTRMG", "STARTDT", "STOPDT", "LEDD"])
raw = TMP / "raw_ledd"
write_table(raw, "Medical_History", "LEDD_Concomitant_Medication_Log", log)
visits = pd.DataFrame({"PATNO": 1, "EVENT_ID": ["BL", "V04", "V05", "V06", "V07"],
                       "date": ppmi.mdate(pd.Series(["12/2015", "01/2017", "03/2017", "04/2017", "07/2017"]))})
out = ppmi._ledd(raw, visits)
print(visits.merge(out, on=["PATNO", "EVENT_ID"], how="left").to_string(index=False))
print("\nby hand, V07: 400 (levodopa) + 150 (pramipexole, once) + 0.33 x 400 (entacapone) =", 400 + 150 + 0.33 * 400)
print("V05 (03/2017) counts both Sinemet entries, the one stopped and the one started that month: 300 + 400 =", 300 + 400)

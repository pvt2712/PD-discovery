"""Section 1.17: participants, on the repository's mock download (made-up participants), with three
edits that exercise edge cases."""
import pandas as pd
from _setup import mock_download

from pdx import ppmi

raw = mock_download()
part = ppmi.participants(raw)
cols = ["PATNO", "enroll_date", "era", "genetic", "age_enroll", "male", "onset_date", "dx_date", "death_date",
        "died", "enrolled", "duration_years", "age_onset"]
print(part[cols].loc[part.PATNO.isin([1, 2, 5, 9, 12, 13])].to_string(index=False))
d, o = pd.Timestamp("2015-01-15"), pd.Timestamp("2014-09-15")
print("by hand, participant 1: duration =", (d - o).days, "/ 365.25 =", round((d - o).days / 365.25, 4),
      "| age at onset = 62 -", round((d - pd.Timestamp("2014-06-15")).days / 365.25, 4))

# three edits: a missing enrolment date, a death record without a date, a second demographics row
st = pd.read_csv(raw / "_Subject_Characteristics" / "Participant_Status_01Jan2026.csv")
st.loc[st.PATNO == 4, "ENROLL_DATE"] = None
st.to_csv(raw / "_Subject_Characteristics" / "Participant_Status_01Jan2026.csv", index=False)
cc = pd.read_csv(raw / "Study_Enrollment" / "Conclusion_of_Study_Participation_01Jan2026.csv")
cc = pd.concat([cc, pd.DataFrame({"PATNO": [6], "WDDT": [None], "WDDEATH": [1]})])
cc.to_csv(raw / "Study_Enrollment" / "Conclusion_of_Study_Participation_01Jan2026.csv", index=False)
dm = pd.read_csv(raw / "_Subject_Characteristics" / "Demographics_01Jan2026.csv")
dm = pd.concat([dm, pd.DataFrame({"PATNO": [1], "SEX": [0]})])
dm.to_csv(raw / "_Subject_Characteristics" / "Demographics_01Jan2026.csv", index=False)
p2 = ppmi.participants(raw).set_index("PATNO")
print("\nafter the edits:")
print(p2.loc[[1, 4, 6], ["enroll_date", "era", "male", "death_date", "died", "duration_years"]].to_string())

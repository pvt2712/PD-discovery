"""Sections 1.14-1.16: read, mdate, items_valid, item_sum and in_range. Synthetic files and values."""
import numpy as np
import pandas as pd
from _setup import TMP

from pdx import ppmi

print("== 1.14 read ==")
raw = TMP / "raw_read"
folder = raw / "_Subject_Characteristics"
folder.mkdir(parents=True)
for tag in ["05Sep2025", "13Oct2026", "02Nov2026"]:
    pd.DataFrame({"file": [f"Participant_Status_{tag}.csv"], "PATNO": [1]}).to_csv(
        folder / f"Participant_Status_{tag}.csv", index=False)
print("files present:", sorted(p.name for p in folder.iterdir()))
print("read picked:  ", ppmi.read(raw, "_Subject_Characteristics", "Participant_Status").file.iloc[0])
try:
    ppmi.read(raw, "_Subject_Characteristics", "Participant_Status", usecols=["PATNO", "COHORT"])
except KeyError as e:
    print("usecols check ->", e)
pd.DataFrame({"PATNO": [7]}).to_csv(folder / "Demographics.csv", index=False)
print("plain-name fallback:", ppmi.read(raw, "_Subject_Characteristics", "Demographics").to_dict("list"))

print("\n== 1.15 mdate ==")
s = pd.Series(["03/2015", "3/2015", "12/1999", None, "2015-03", "13/2015"])
print(pd.DataFrame({"raw": s, "mdate": ppmi.mdate(s)}).to_string(index=False))

print("\n== 1.16 items_valid, item_sum, in_range ==")
df = pd.DataFrame({"a": [1, 2, 101, 1, np.nan], "b": [0, 3, 2, -1, np.nan], "c": [4, "9", 1, 2, 1],
                   "d": [2, 2, 2, np.nan, 1], "e": [1, 1, 1, 1, 1]})
print(df.to_string())
print("items_valid (0-4):")
print(ppmi.items_valid(df, list("abcde")).to_string())
print("item_sum, 5 items, needs ceil(0.8 * 5) = 4:", ppmi.item_sum(df, list("abcde")).tolist())
print("in_range(0, 30):", ppmi.in_range(pd.Series([26, 31, -1, "x", 30]), 0, 30).tolist())

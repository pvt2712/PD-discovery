"""Sections 1.20-1.22: visits, on the repository's mock download (made-up participants) with item
values edited so that each scoring rule is exercised."""
import numpy as np
import pandas as pd
from _setup import mock_download

from pdx import ppmi

raw = mock_download()


def edit(folder, name, fn):
    path = raw / folder / f"{name}_01Jan2026.csv"
    df = pd.read_csv(path)
    df = fn(df)
    df.to_csv(path, index=False)


def at(df, p, ev="BL"):
    return (df.PATNO == p) & (df.EVENT_ID == ev)


def part2(df):
    df.loc[at(df, 1), ppmi.FUNCTION_II[:2]] = 101          # 2 of 10 items unable to rate: sum of 8
    df.loc[at(df, 2), ppmi.FUNCTION_II[:3]] = np.nan       # 3 of 10 blank: missing
    return df


def gds(df):
    df.loc[at(df, 1), ["GDSDROPD", "GDSBORED", "GDSMEMRY"]] = 1         # 3 depressive "yes"
    df.loc[at(df, 1), ["GDSGSPIR", "GDSENRGY"]] = 0                     # 2 depressive "no"
    df.loc[at(df, 1), ["GDSHOME", "GDSHAPPY", "GDSALIVE"]] = np.nan     # 12 of 15 answered
    df.loc[at(df, 2), ["GDSHOME", "GDSHAPPY", "GDSALIVE", "GDSWRTLS"]] = np.nan   # 11 answered
    df.loc[at(df, 4), "GDSDROPD"] = 9                                   # a code outside 0/1
    return df


def scopa(df):
    cols = [f"SCAU{i}" for i in range(1, 22)]
    df.loc[at(df, 1), cols] = [1, 0, 2, 1, 0, 0, 1, 9, 2, 1, 0, 0, 1, 0, 0, 1, np.nan, np.nan, np.nan, np.nan, 1]
    df.loc[at(df, 2), cols[:5]] = np.nan                                # 16 of 21 answered
    return df


def ess(df):
    df.loc[at(df, 1), [f"ESS{i}" for i in range(1, 9)]] = [0, 1, 2, 3, 0, 1, 2, 3]
    df.loc[at(df, 2), "ESS8"] = 5                                       # out of range: 7 items left
    return df


def rbdsq(df):
    df.loc[at(df, 1), ppmi.RBDSQ_ITEMS] = [1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0]
    df.loc[at(df, 1), "PARKISM"] = 1                                    # question 10: a nervous-system disease
    df.loc[at(df, 2), ppmi.RBDSQ_ITEMS[:3]] = np.nan                    # 9 of 12 items
    return df


def moca(df):
    df = df[~at(df, 7)]                                                 # participant 7: no MoCA at BL ...
    return pd.concat([df, pd.DataFrame({"PATNO": [7], "EVENT_ID": ["SC"], "INFODT": ["12/2014"], "MCATOT": [25]})])


def se(df):
    df.loc[at(df, 1), "MSEADLG"] = 85
    df.loc[at(df, 2), "MSEADLG"] = 105
    return df


def u1(df):
    df.loc[at(df, 1), "NP1HALL"] = 2
    df.loc[at(df, 2), "NP1HALL"] = 7
    return df


def u4(df):
    df.loc[at(df, 1), "NP4TOT"] = np.nan                                # untreated: blank becomes 0
    df.loc[at(df, 12, "V04"), "NP4TOT"] = np.nan                        # treated: blank stays missing
    return df


M, N = "Motor___MDS-UPDRS", "Non-motor_Assessments"
edit(M, "MDS_UPDRS_Part_II__Patient_Questionnaire", part2)
edit(N, "Geriatric_Depression_Scale__Short_Version_", gds)
edit(N, "SCOPA-AUT", scopa)
edit(N, "Epworth_Sleepiness_Scale", ess)
edit(N, "REM_Sleep_Behavior_Disorder_Screening_Questionnaire", rbdsq)
edit(N, "Montreal_Cognitive_Assessment__MoCA_", moca)
edit(M, "Modified_Schwab___England_Activities_of_Daily_Living", se)
edit(M, "MDS-UPDRS_Part_I", u1)
edit(M, "MDS-UPDRS_Part_IV__Motor_Complications", u4)

ids = set(range(1, 14))
v = ppmi.visits(raw, ids)
show = ["PATNO", "EVENT_ID", "date", "function", "mood", "autonomic", "sleepiness", "rbd", "MCATOT", "cognition",
        "independence", "hallucinations", "complications", "treated", "ledd", "remote", "motor_state"]
sel = v[(v.PATNO.isin([1, 2, 4, 7]) & (v.EVENT_ID == "BL")) | ((v.PATNO == 12) & (v.EVENT_ID == "V04"))]
print(sel[show].to_string(index=False))
print("\nrows:", len(v), "| events:", sorted(v.EVENT_ID.unique()), "| remote rows:", int(v.remote.sum()))
print("by hand, participant 1: function = 8 items x 1 =", 8, "| mood = 3 + 2 =", 5,
      "| autonomic = 1+0+2+1+0+0+1+3+2+1+0+0+1+0+0+1+1 =", 1 + 0 + 2 + 1 + 0 + 0 + 1 + 3 + 2 + 1 + 0 + 0 + 1 + 0 + 0 + 1 + 1,
      "| sleepiness = 12 | rbd = 5 + 1 = 6 | independence = 100 - 85 = 15")

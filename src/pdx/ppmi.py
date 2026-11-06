"""Study tables from the local PPMI download (protocol v0.5, §4, §5, §6.1).

Runs only on this machine, under the investigator's written approval (§14). Participant-level
outputs go to data/derived/ (git-ignored); nothing here prints or returns a participant-level row
to the caller's output. The parsing follows the PPMI file layout (folder/table_<date>.csv, dates
MM/YYYY set to the 15th) that the ppmi-causal build reads; the rules below are this protocol's.

Provisional choices, to be fixed in WP1 and recorded there:
- domains and items as in §6.1 (MDS-UPDRS III subscores from the OFF or untreated examination);
  a domain score needs at least 80% of its items; item codes outside an item's range (such as
  101, unable to rate) are missing; a Part III record with under 80% of its items is a state or
  dosing record, not an examination;
- MoCA, GDS, SCOPA-AUT, ESS and RBDSQ given at screening stand in for baseline when baseline is
  missing (PPMI 1.0 schedule);
- every domain oriented so that higher means worse (Schwab & England as 100 - score, MoCA as
  30 - score);
- a visit is remote when its event code starts with "R" (PPMI 2.0's remote schedule);
- unscheduled events (SC, ST, PW, TRANS, U01, RS1) are not eligible for the annual grid;
- PPMI's PD cohort table also lists people who were never enrolled (screen failed, excluded,
  declined, pending); only enrolled participants (including those who later withdrew or died) count;
- genetic enrolment (LRRK2, GBA, SNCA, PRKN, PINK1, other variant) defines genetic PD;
- treated at baseline: on dopaminergic treatment at the baseline examination, or a positive
  LEDD on the baseline date;
- an examination counts as untreated only if neither its form (PDTRTMNT) nor the medication log
  shows treatment on that date; otherwise, without a recorded OFF or ON state, it is "treated,
  state unrecorded" and its motor scores are not used (D5, D24; audit A6);
- the dose in force on a date comes from the medication log at month resolution: a record that
  stops in the date's month gives way to any record starting in that month (a dose change is
  entered as a stop and a new start; audit A4);
- the long-duration exposure is the levodopa dose (with a COMT inhibitor's multiplier) passed
  through a slow effect compartment over the preceding months, with a provisional half-life
  (D24, configs/protocol_v0.5.toml [treatment]);
- the Part II endpoint at 24 months: death by 24 months, or Schwab & England below 80 at the
  year-2 visit.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[3] / "PPMI" / "raw"
NOT_GRID = ("SC", "ST", "PW", "TRANS", "U01", "RS1")
ENROLLED = ("Enrolled", "Withdrew", "Complete", "Withdraw Deceased", "Baseline Withdraw", "Baseline")

TREMOR_III = ["NP3PTRMR", "NP3PTRML", "NP3KTRMR", "NP3KTRML", "NP3RTARU", "NP3RTALU", "NP3RTARL", "NP3RTALL",
              "NP3RTALJ", "NP3RTCON"]
BRADY_RIGID = ["NP3RIGN", "NP3RIGRU", "NP3RIGLU", "NP3RIGRL", "NP3RIGLL", "NP3FTAPR", "NP3FTAPL", "NP3HMOVR",
               "NP3HMOVL", "NP3PRSPR", "NP3PRSPL", "NP3TTAPR", "NP3TTAPL", "NP3LGAGR", "NP3LGAGL", "NP3BRADY"]
AXIAL = ["NP3RISNG", "NP3GAIT", "NP3FRZGT", "NP3PSTBL", "NP3POSTR"]
FUNCTION_II = ["NP2SPCH", "NP2SALV", "NP2SWAL", "NP2EAT", "NP2DRES", "NP2HYGN", "NP2HWRT", "NP2HOBB", "NP2TURN",
               "NP2RISE"]
# Stebbins et al. 2013: tremor = 2.10, 3.15-3.18 (11 items); PIGD = 2.12, 2.13, 3.10-3.12 (5 items)
STEBBINS_TREMOR = ["NP2TRMR"] + TREMOR_III
STEBBINS_PIGD = ["NP2WALK", "NP2FREZ", "NP3GAIT", "NP3FRZGT", "NP3PSTBL"]
TD_CUT, PIGD_CUT = 1.15, 0.90
GENETIC_FLAGS = ["ENRLLRRK2", "ENRLGBA", "ENRLSNCA", "ENRLPRKN", "ENRLPINK1", "ENRLOTHGV"]
GDS_YES = ["GDSDROPD", "GDSEMPTY", "GDSBORED", "GDSAFRAD", "GDSHLPLS", "GDSHOME", "GDSMEMRY", "GDSWRTLS", "GDSHOPLS",
           "GDSBETER"]
GDS_NO = ["GDSSATIS", "GDSGSPIR", "GDSHAPPY", "GDSALIVE", "GDSENRGY"]
RBDSQ_ITEMS = ["DRMVIVID", "DRMAGRAC", "DRMNOCTB", "SLPLMBMV", "SLPINJUR", "DRMVERBL", "DRMFIGHT", "DRMUMV", "DRMOBJFL",
               "MVAWAKEN", "DRMREMEM", "SLPDSTRB"]
RBDSQ_NEURO = ["STROKE", "HETRA", "PARKISM", "RLS", "NARCLPSY", "DEPRS", "EPILEPSY", "BRNINFM", "CNSOTH"]
# levodopa by name (audit A6: L-dopa, Duopa, co-careldopa, co-beneldopa and Dhivy added; carbidopa
# alone, such as Lodosyn, is not levodopa; \bl-dopa\b leaves out methyldopa)
LEVODOPA = (r"lev[a-z]*d[oa]p|\bl[- ]?dopa\b|careldopa|beneldopa|levocomp|levopar|nacom|isicom|sinemet|madopar|"
            r"rytary|numient|crexont|stalevo|corbilta|sastravi|duodopa|duopa|lecigon|vyalev|dhivy|inbrija|dopicar|"
            r"syndopa|prolopa|parcopa|atamet|kinson|benserazid")
LDR_HALF_LIFE_DAYS = 28.0      # provisional (D24); the run scripts pass configs/protocol_v0.5.toml's value
MOTOR_SCORES = ["tremor", "brady_rigidity", "axial"] + TREMOR_III + ["NP3GAIT", "NP3FRZGT", "NP3PSTBL"]
DOMAINS = ["tremor", "brady_rigidity", "axial", "function", "independence", "cognition", "mood", "autonomic",
           "hallucinations", "sleepiness", "rbd", "complications"]


# ---------------------------------------------------------------- reading

def read(raw: Path, folder: str, name: str, usecols=None) -> pd.DataFrame:
    paths = (glob.glob(os.path.join(raw, folder, name + "_[0-9][0-9][A-Z][a-z][a-z][0-9][0-9][0-9][0-9].csv"))
             or glob.glob(os.path.join(raw, folder, name + ".csv")))
    if not paths:
        raise FileNotFoundError(f"{folder}/{name} not found under {raw}")
    df = pd.read_csv(sorted(paths)[-1], low_memory=False)
    if usecols is not None:
        missing = [c for c in usecols if c not in df.columns]
        if missing:
            raise KeyError(f"{name}: columns not found: {missing}")
        df = df[list(usecols)]
    return df


def mdate(s: pd.Series) -> pd.Series:
    """MM/YYYY to the 15th of that month."""
    return pd.to_datetime("15/" + s.astype("string"), format="%d/%m/%Y", errors="coerce")


def items_valid(df: pd.DataFrame, items: list, lo: float = 0, hi: float = 4) -> pd.DataFrame:
    """Item scores with out-of-range codes (101 = unable to rate, 9 = unknown, ...) as missing."""
    x = df[items].apply(pd.to_numeric, errors="coerce")
    return x.where((x >= lo) & (x <= hi))


def item_sum(df: pd.DataFrame, items: list, min_frac: float = 0.8, lo: float = 0, hi: float = 4) -> pd.Series:
    return items_valid(df, items, lo, hi).sum(axis=1, min_count=int(np.ceil(min_frac * len(items))))


def in_range(s: pd.Series, lo: float, hi: float) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    return x.where((x >= lo) & (x <= hi))


# ---------------------------------------------------------------- participants

def participants(raw: Path = RAW) -> pd.DataFrame:
    ps = read(raw, "_Subject_Characteristics", "Participant_Status")
    pdc = ps[ps.COHORT == 1].copy()
    pdc["enroll_date"] = mdate(pdc.ENROLL_DATE)
    pdc["era"] = np.where(pdc.enroll_date.dt.year < 2020, "PPMI 1.0", "PPMI 2.0")
    flags = [c for c in GENETIC_FLAGS if c in pdc.columns]
    pdc["genetic"] = (pdc[flags].fillna(0).astype(float) == 1).any(axis=1)
    out = pdc[["PATNO", "enroll_date", "era", "genetic", "ENROLL_AGE", "ENROLL_STATUS"]].rename(
        columns={"ENROLL_AGE": "age_enroll"})
    dem = read(raw, "_Subject_Characteristics", "Demographics").drop_duplicates("PATNO", keep="last")
    out = out.merge(dem[["PATNO", "SEX"]].rename(columns={"SEX": "male"}), on="PATNO", how="left")
    dx = read(raw, "Medical_History", "PD_Diagnosis_History").drop_duplicates("PATNO", keep="last")
    dx["onset_date"], dx["dx_date"] = mdate(dx.SXDT), mdate(dx.PDDXDT)
    out = out.merge(dx[["PATNO", "onset_date", "dx_date"]], on="PATNO", how="left")
    concl = read(raw, "Study_Enrollment", "Conclusion_of_Study_Participation")
    concl["exit_date"] = mdate(concl.WDDT)
    dead = concl[concl.WDDEATH == 1].sort_values("exit_date").drop_duplicates("PATNO")
    out = out.merge(dead[["PATNO", "exit_date"]].rename(columns={"exit_date": "death_date"}), on="PATNO", how="left")
    out["died"] = out.death_date.notna() | (out.ENROLL_STATUS == "Withdraw Deceased")
    out["enrolled"] = out.ENROLL_STATUS.isin(ENROLLED)
    years = lambda a, b: (a - b).dt.days / 365.25
    out["duration_years"] = years(out.enroll_date, out.dx_date)
    out["age_onset"] = out.age_enroll - years(out.enroll_date, out.onset_date.fillna(out.dx_date))
    return out.drop(columns="ENROLL_STATUS")


# ---------------------------------------------------------------- visits

def _motor(raw: Path, ids: set) -> pd.DataFrame:
    """Part III per visit: the OFF or untreated examination (primary, D5) and the ON one.
    A treated participant's examination without a recorded state is not used (motor state
    "treated, state unrecorded")."""
    u3 = read(raw, "Motor___MDS-UPDRS", "MDS-UPDRS_Part_III")
    u3 = u3[u3.PATNO.isin(ids)].copy()
    state = np.select([u3.PDSTATE == "OFF", u3.PDSTATE == "ON", u3.PAG_NAME.isin(["NUPDRS3A", "NUPDR3ON"])],
                      ["OFF", "ON", "ON"], "untreated")
    state = np.where((state == "untreated") & (u3.PAG_NAME == "NUPDR3OF"), "OFF", state)
    state = np.where((state == "untreated") & (u3.PDTRTMNT == 1), "unrecorded", state)
    u3["state"] = state
    u3["date"] = mdate(u3.INFODT)
    allitems = list(dict.fromkeys(TREMOR_III + BRADY_RIGID + AXIAL))
    u3[allitems] = items_valid(u3, allitems)
    for name, items in (("tremor", TREMOR_III), ("brady_rigidity", BRADY_RIGID), ("axial", AXIAL)):
        u3[name] = item_sum(u3, items)
    # a record without most of its items is a state or dosing record, not an examination
    u3["n_items"] = u3[allitems].notna().sum(axis=1)
    u3["exam"] = u3.n_items >= 0.8 * len(allitems)
    key = ["PATNO", "EVENT_ID"]
    first = u3.sort_values("date").drop_duplicates(key)[key + ["date"]]
    prim = u3[u3.state.isin(["untreated", "OFF"]) & u3.exam].sort_values(["n_items", "date"])
    prim = prim.drop_duplicates(key, keep="last")
    prim = prim[key + ["tremor", "brady_rigidity", "axial", "state"] + TREMOR_III + ["NP3GAIT", "NP3FRZGT", "NP3PSTBL"]]
    on = u3[(u3.state == "ON") & u3.exam].sort_values(["n_items", "date"]).drop_duplicates(key, keep="last")
    on = on[key + ["tremor", "brady_rigidity", "axial"]].rename(
        columns={"tremor": "tremor_on", "brady_rigidity": "brady_rigidity_on", "axial": "axial_on"})
    u3["off_no_exam"] = (u3.state == "OFF") & ~u3.exam
    flags = u3.groupby(key).agg(pdtrtmnt_any=("PDTRTMNT", "max"),
                                unrecorded=("state", lambda st: (st == "unrecorded").any()),
                                off_no_exam=("off_no_exam", "any")).reset_index()
    m = first.merge(prim, on=key, how="left").merge(on, on=key, how="left").merge(flags, on=key, how="left")
    m["motor_state"] = np.select(
        [m.state.notna(), m.tremor_on.notna(), m.off_no_exam.fillna(False), m.unrecorded.fillna(False)],
        [m.state, "ON only", "OFF recorded, no examination", "treated, state unrecorded"], None)
    return m.drop(columns=["state", "unrecorded", "off_no_exam"])


def _in_force(ld: pd.DataFrame, at: pd.DataFrame) -> pd.DataFrame:
    """The medication records in force at each row of `at` (PATNO, date, and any other columns,
    which are kept): started on or before the date and not stopped before it. Dates are months, so
    a record that stops in the date's month gives way to any record of the same participant that
    starts in that month: a dose change entered as a stop and a new start counts once, at the new
    dose (audit A4), while a drug stopped without a replacement still counts in its last month."""
    act = at.merge(ld, on="PATNO", how="inner")
    act = act[(act.start <= act.date) & (act.stop.isna() | (act.stop >= act.date))]
    n_start = ld.groupby(["PATNO", "start"]).size().rename("n_start").reset_index().rename(columns={"start": "date"})
    act = act.merge(n_start, on=["PATNO", "date"], how="left")
    others_start = act.n_start.fillna(0) - (act.start == act.date).astype(int) > 0
    return act[~((act.stop == act.date) & others_start)].drop(columns="n_start")


def _ledd(raw: Path, visits: pd.DataFrame, half_life_days: float = LDR_HALF_LIFE_DAYS,
          months: int = 12) -> pd.DataFrame:
    """Total and levodopa LEDD in force on each visit date (from the LEDD medication log), the drug
    classes in use (1 levodopa, 2 other, 3 both) for detecting medication changes, and the
    long-duration exposure: the levodopa dose, times one plus any COMT inhibitor's multiplier, in
    force at the visit and at each of the `months` months before it, weighted by a slow effect
    compartment with the given half-life (month m before the visit gets e^(-k m D) - e^(-k (m+1) D),
    D a month, the oldest month the rest), so a dose held for months gives that dose (D24)."""
    ld = read(raw, "Medical_History", "LEDD_Concomitant_Medication_Log")
    ld = ld[ld.PATNO.isin(set(visits.PATNO))].copy()
    ld["start"], ld["stop"] = mdate(ld.STARTDT), mdate(ld.STOPDT)
    ld["ledd_num"] = pd.to_numeric(ld.LEDD, errors="coerce")
    ld["levodopa"] = ld.LEDTRT.astype("string").str.lower().str.contains(LEVODOPA, na=False)
    ld["ld_mult"] = ld.LEDD.astype("string").str.extract(r"LD x ([\d.]+)")[0].astype(float)   # COMT inhibitors
    ld = ld.drop_duplicates(["PATNO", "LEDTRT", "LEDDOSE", "LEDDOSFRQ", "LEDDSTRMG", "STARTDT", "STOPDT"])
    ld = ld.drop(columns=[c for c in ("EVENT_ID", "PAG_NAME", "REC_ID") if c in ld.columns])   # the log's own codes
    ld = ld[["PATNO", "start", "stop", "ledd_num", "levodopa", "ld_mult"]]
    vis = visits[["PATNO", "EVENT_ID", "date"]].drop_duplicates()
    act = _in_force(ld, vis)
    act["lev_ledd"] = act.ledd_num.where(act.levodopa, 0.0)
    act["other"] = ~act.levodopa
    out = act.groupby(["PATNO", "EVENT_ID"]).agg(ledd=("ledd_num", "sum"), ledd_levodopa=("lev_ledd", "sum"),
                                                comt_mult=("ld_mult", "max"), any_lev=("levodopa", "any"),
                                                any_other=("other", "any")).reset_index()
    out["ledd"] = out.ledd.fillna(0) + out.comt_mult.fillna(0) * out.ledd_levodopa
    out["classes"] = out.any_lev.astype(int) + 2 * out.any_other.astype(int)
    # the long-duration exposure
    k, month = np.log(2) / half_life_days, 30.4375
    w = np.exp(-k * month * np.arange(months)) - np.exp(-k * month * np.arange(1, months + 1))
    w[-1] = np.exp(-k * month * (months - 1))
    back = pd.concat([vis.assign(lag=m, date=vis.date - pd.DateOffset(months=m)) for m in range(months)])
    hist = _in_force(ld, back)
    hist = hist.assign(lev=hist.ledd_num.where(hist.levodopa, 0.0))
    per = hist.groupby(["PATNO", "EVENT_ID", "lag"]).agg(lev=("lev", "sum"), mult=("ld_mult", "max"))
    per = (per.lev.fillna(0) * (1 + per.mult.fillna(0))).rename("dose").reset_index()
    per["weighted"] = per.dose * w[per.lag.to_numpy()]
    expo = per.groupby(["PATNO", "EVENT_ID"]).weighted.sum().rename("exposure").reset_index()
    out = out.merge(expo, on=["PATNO", "EVENT_ID"], how="outer")
    return out.drop(columns=["comt_mult", "any_lev", "any_other"])


def visits(raw: Path, ids: set, ldr_half_life_days: float = LDR_HALF_LIFE_DAYS) -> pd.DataFrame:
    """One row per participant and event: date, mode, motor state, domain scores, Stebbins items,
    treatment, LEDD and the long-duration exposure."""
    v = _motor(raw, ids)

    def add(folder, name, cols, fn=None, sc_as_bl=False):
        nonlocal v
        df = read(raw, folder, name)
        df = df[df.PATNO.isin(ids)].copy()
        if sc_as_bl:      # PPMI 1.0 gave some instruments at screening; that value stands in for baseline
            has_bl = set(df.loc[df.EVENT_ID == "BL", "PATNO"])
            df.loc[(df.EVENT_ID == "SC") & ~df.PATNO.isin(has_bl), "EVENT_ID"] = "BL"
        if fn is not None:
            df = fn(df)
        df["date_"] = mdate(df.INFODT)
        df = df.sort_values("date_").drop_duplicates(["PATNO", "EVENT_ID"], keep="last")
        v = v.merge(df[["PATNO", "EVENT_ID", "date_"] + cols], on=["PATNO", "EVENT_ID"], how="outer")
        v["date"] = v.date.fillna(v.date_)
        v = v.drop(columns="date_")

    def part2(df):
        df["function"] = item_sum(df, FUNCTION_II)
        return df

    def gds(df):
        raw_items = items_valid(df, GDS_YES + GDS_NO, 0, 1)
        items = pd.concat([raw_items[GDS_YES] == 1, raw_items[GDS_NO] == 0], axis=1).astype(float)
        items[raw_items.isna().to_numpy()] = np.nan
        df["mood"] = items.sum(axis=1, min_count=12)
        return df

    def scopa(df):
        cols = [f"SCAU{i}" for i in range(1, 22)]
        x = df[cols].apply(pd.to_numeric, errors="coerce").replace(9, 3)      # 9 = uses a catheter, scored 3
        df["autonomic"] = x.where((x >= 0) & (x <= 3)).sum(axis=1, min_count=17)
        return df

    def ess(df):
        df["sleepiness"] = item_sum(df, [f"ESS{i}" for i in range(1, 9)], lo=0, hi=3)
        return df

    def rbdsq(df):
        neuro = items_valid(df, RBDSQ_NEURO, 0, 1)
        df["rbd"] = item_sum(df, RBDSQ_ITEMS, lo=0, hi=1) + (neuro.fillna(0).sum(axis=1) > 0).astype(float)
        return df

    add("Motor___MDS-UPDRS", "MDS_UPDRS_Part_II__Patient_Questionnaire",
        ["function", "NP2TRMR", "NP2WALK", "NP2FREZ"], part2)
    v[["NP2TRMR", "NP2WALK", "NP2FREZ"]] = items_valid(v, ["NP2TRMR", "NP2WALK", "NP2FREZ"])
    add("Motor___MDS-UPDRS", "MDS-UPDRS_Part_I", ["NP1HALL"])
    add("Motor___MDS-UPDRS", "MDS-UPDRS_Part_IV__Motor_Complications", ["NP4TOT"])
    add("Motor___MDS-UPDRS", "Modified_Schwab___England_Activities_of_Daily_Living", ["MSEADLG"])
    add("Non-motor_Assessments", "Montreal_Cognitive_Assessment__MoCA_", ["MCATOT"], sc_as_bl=True)
    add("Non-motor_Assessments", "Geriatric_Depression_Scale__Short_Version_", ["mood"], gds, sc_as_bl=True)
    add("Non-motor_Assessments", "SCOPA-AUT", ["autonomic"], scopa, sc_as_bl=True)
    add("Non-motor_Assessments", "Epworth_Sleepiness_Scale", ["sleepiness"], ess, sc_as_bl=True)
    add("Non-motor_Assessments", "REM_Sleep_Behavior_Disorder_Screening_Questionnaire", ["rbd"], rbdsq, sc_as_bl=True)
    v = v[v.date.notna()].copy()
    v["independence"] = 100 - in_range(v.MSEADLG, 0, 100)
    v["cognition"] = 30 - in_range(v.MCATOT, 0, 30)
    v["hallucinations"] = in_range(v.NP1HALL, 0, 4)
    v["complications"] = in_range(v.NP4TOT, 0, 24)
    v["remote"] = v.EVENT_ID.astype(str).str.startswith("R")
    led = _ledd(raw, v, ldr_half_life_days)
    v = v.merge(led, on=["PATNO", "EVENT_ID"], how="left")
    for c in ("ledd", "ledd_levodopa", "exposure"):
        v[c] = v[c].fillna(0)
    # an examination is untreated only if the medication log agrees (audit A6): with a dose in force
    # and no recorded state, it is "treated, state unrecorded" and its OFF/untreated scores go
    dosed = v.motor_state.eq("untreated") & (v.ledd > 0)
    v.loc[dosed, MOTOR_SCORES] = np.nan
    v.loc[dosed, "motor_state"] = "treated, state unrecorded"
    v["treated"] = ((v.ledd > 0) | (v.pdtrtmnt_any == 1)).astype(int)
    v["complications"] = v.complications.where(v.treated == 1, v.complications.fillna(0))   # zero before treatment
    return v.sort_values(["PATNO", "date"]).reset_index(drop=True)


# ---------------------------------------------------------------- phenotype and grid

def stebbins(v: pd.DataFrame) -> pd.DataFrame:
    """TD/PIGD from the OFF or untreated examination and Part II (Stebbins 2013): ratio of the mean
    tremor score to the mean PIGD score; TD >= 1.15, PIGD <= 0.90, otherwise indeterminate; a zero
    PIGD mean gives TD if the tremor mean is positive, indeterminate if both are zero."""
    tr = v[STEBBINS_TREMOR].apply(pd.to_numeric, errors="coerce")
    pg = v[STEBBINS_PIGD].apply(pd.to_numeric, errors="coerce")
    ok = tr.notna().sum(axis=1) >= 9
    ok &= pg.notna().sum(axis=1) >= 4
    tm, pm = tr.mean(axis=1), pg.mean(axis=1)
    ratio = np.where(pm > 0, tm / pm.where(pm > 0, 1.0), np.where(tm > 0, np.inf, np.nan))
    label = np.select([ratio >= TD_CUT, ratio <= PIGD_CUT], ["TD", "PIGD"], "indeterminate").astype(object)
    label[(~ok.to_numpy()) | (v.motor_state.isin(["untreated", "OFF"]) == False).to_numpy()] = None
    return pd.DataFrame({"td_pigd_ratio": np.where(ok, ratio, np.nan), "phenotype": label}, index=v.index)


def clinical_subtype(base: pd.DataFrame) -> pd.Series:
    """Provisional Fereshtehnejad subtype at baseline (Brain 2017), with percentiles taken in the
    population passed in: diffuse malignant if the motor score (MDS-UPDRS II + III items) is above
    the 75th percentile with at least one non-motor score above it (cognition, RBD symptoms,
    autonomic), or all three non-motor scores above it; mild motor-predominant if all four are
    at or below it; intermediate otherwise. The cognitive composite and the cut-offs are fixed in
    WP1 (D2); here cognition is 30 - MoCA."""
    motor = base[["function", "NP2TRMR", "NP2WALK", "NP2FREZ", "tremor", "brady_rigidity", "axial"]].sum(
        axis=1, min_count=7)
    nm = base[["cognition", "rbd", "autonomic"]]
    hi = lambda x: x > x.quantile(0.75)
    m_hi = hi(motor)
    n_hi = pd.concat([hi(nm[c]) for c in nm.columns], axis=1)
    ok = motor.notna() & nm.notna().all(axis=1)
    dm = (m_hi & n_hi.any(axis=1)) | n_hi.all(axis=1)
    mild = ~m_hi & ~n_hi.any(axis=1)
    out = pd.Series(np.where(dm, "diffuse malignant", np.where(mild, "mild motor-predominant", "intermediate")),
                    index=base.index, dtype=object)
    return out.where(ok)


def annual_grid(v: pd.DataFrame, anchor: pd.Series, max_year: int = 15) -> pd.DataFrame:
    """One visit per participant and year: the eligible visit within +-4 months of the enrolment
    anniversary, chosen in a fixed order that does not look at the scores (§4): in person before
    remote, then closest to the anniversary, then the motor state (untreated or OFF before ON
    only), and the completeness of the core battery only as the last tie-breaker. Year 0 is the
    baseline visit."""
    v = v[~v.EVENT_ID.isin(NOT_GRID)].copy()
    v["anchor"] = v.PATNO.map(anchor)
    v["months"] = (v.date - v.anchor).dt.days / 30.4375
    v["year"] = np.round(v.months / 12).astype("Int64")
    v["off_by"] = (v.months - 12 * v.year.astype(float)).abs()
    v = v[(v.off_by <= 4.0) & (v.year >= 0) & (v.year <= max_year)]
    v.loc[(v.year == 0) & (v.EVENT_ID != "BL"), "year"] = pd.NA       # baseline is the BL visit
    v = v[v.year.notna()]
    v["state_rank"] = v.motor_state.map({"untreated": 0, "OFF": 0, "ON only": 1}).fillna(2)
    v["n_core"] = v[DOMAINS].notna().sum(axis=1)
    v = v.sort_values(["PATNO", "year", "remote", "off_by", "state_rank", "n_core"],
                      ascending=[True, True, True, True, True, False])
    g = v.drop_duplicates(["PATNO", "year"], keep="first").copy()
    g["date_years"] = (g.date - g.anchor).dt.days / 365.25
    return g.drop(columns=["off_by", "state_rank", "n_core"]).reset_index(drop=True)


# ---------------------------------------------------------------- the study population and the WP0 table

def build(raw: Path = RAW, ldr_half_life_days: float = LDR_HALF_LIFE_DAYS) -> dict:
    """Participants, grid and the WP0 visit table (participant-level; keep inside data/)."""
    part = participants(raw)
    ids = set(part.PATNO)
    v = visits(raw, ids, ldr_half_life_days)
    anchor = part.set_index("PATNO").enroll_date
    bl_date = v[v.EVENT_ID == "BL"].groupby("PATNO").date.min()
    anchor = anchor.fillna(bl_date)
    grid = annual_grid(v, anchor)
    ph = stebbins(grid)
    grid = pd.concat([grid, ph], axis=1)
    base = grid[grid.year == 0].set_index("PATNO")
    part = part.set_index("PATNO")
    part["has_baseline"] = part.index.isin(base.index)
    part["treated_bl"] = base.treated.reindex(part.index).fillna(0).astype(bool)
    part["phenotype"] = base.phenotype.reindex(part.index)
    part["td_pigd_ratio"] = base.td_pigd_ratio.reindex(part.index)
    reached = set(grid.loc[grid.year == 1, "PATNO"])
    part["reached_y1"] = part.index.isin(reached)
    steps = [("not enrolled (screen failed, excluded, declined or pending)", ~part.enrolled),
             ("not sporadic (genetic enrolment)", part.genetic),
             ("no baseline visit", ~part.genetic & ~part.has_baseline),
             ("treated at baseline", ~part.genetic & part.has_baseline & part.treated_bl),
             ("no usable baseline phenotype", ~part.genetic & part.has_baseline & ~part.treated_bl
              & part.phenotype.isna())]
    part["exclusion"] = None
    for label, mask in steps:
        part.loc[mask & part.exclusion.isna(), "exclusion"] = label
    part["primary"] = part.exclusion.isna()
    sub = clinical_subtype(base.loc[base.index.intersection(part.index[part.primary])])
    part["clinical_subtype"] = sub.reindex(part.index)
    prim = part[part.primary]
    g = grid[grid.PATNO.isin(prim.index)].copy()
    g["cohort"] = g.PATNO.map(prim.era)
    g["phenotype_y0"] = g.PATNO.map(prim.phenotype)
    g["ratio_y0"] = g.PATNO.map(prim.td_pigd_ratio)
    for y in (1, 2):
        lab = g[g.year == y].set_index("PATNO").phenotype
        g[f"phenotype_y{y}"] = g.PATNO.map(lab)
    g = g.sort_values(["PATNO", "year"])
    prev = g.groupby("PATNO").ledd.shift()
    prev_cls = g.groupby("PATNO").classes.shift()
    changed = ((g.ledd - prev).abs() > 0) | (g.classes.fillna(0) != prev_cls.fillna(0))
    g["dose_change"] = changed.astype(float).where(prev.notna())
    se2 = g[g.year == 2].set_index("PATNO").independence
    dead24 = prim.death_date.notna() & ((prim.death_date - prim.enroll_date).dt.days <= 730.5)
    ev = pd.Series(np.nan, index=prim.index)
    ev[se2.reindex(prim.index).notna()] = (se2.reindex(prim.index) > 20).astype(float)   # S&E < 80
    ev[dead24] = 1.0
    g["event_24m"] = np.where(g.year == 0, g.PATNO.map(ev), np.nan)
    wp0 = pd.DataFrame({
        "participant": g.PATNO.astype(str), "cohort": g.cohort, "year": g.year.astype(int),
        "date_years": g.date_years, "phenotype": g.phenotype_y0, "td_pigd_ratio": g.ratio_y0,
        "clinical_subtype": g.PATNO.map(prim.clinical_subtype),
        "mode": np.where(g.remote, "remote", "in person"), "treated": g.treated, "dose_change": g.dose_change,
        "motor_state": g.motor_state.fillna("missing"), **{dname: g[dname] for dname in DOMAINS},
        "phenotype_y1": g.phenotype_y1, "phenotype_y2": g.phenotype_y2, "event_24m": g.event_24m})
    return {"participants": part.reset_index(), "grid": grid, "wp0": wp0, "visits": v}


def flow_table(part: pd.DataFrame) -> pd.DataFrame:
    """Participant flow by era: exclusions by reason and the primary population by phenotype
    (counts; protect them with wp0.protect_2way before sharing)."""
    rows = part.exclusion.fillna("primary: " + part.phenotype.fillna("?").astype(str))
    return pd.crosstab(rows, part.era)


# ---------------------------------------------------------------- the analysis panel (WP4)

CORE_DOMAINS = ["tremor", "brady_rigidity", "axial", "function", "independence", "cognition", "mood", "autonomic",
                "sleepiness", "rbd"]           # provisional: hallucinations and complications wait for D26
TREATED_DOMAINS = ("tremor", "brady_rigidity", "axial", "function")      # §6.1: motor domains and daily function


def analysis_panel(part: pd.DataFrame, grid: pd.DataFrame, domains=CORE_DOMAINS, motor_on: bool = False):
    """The H-P panel (§2.3, §4, §7.1): TD (0) and PIGD (1) participants of the primary population
    seen at the year-1 landmark, their grid visits from year 1 on, scores on one normal-scores
    transform fitted on the pooled primary population over all visits and groups and then frozen,
    centred covariates (era, age at onset, sex, disease duration), dose (100 mg LEDD), the
    long-duration exposure (levodopa through the slow effect compartment, 100 mg; D24) and the
    examination state. With motor_on (a D5 option), motor domains of ON-only visits enter with
    on = 1. Returns (Panel, NormalScores, info); everything here is participant-level except info,
    which also holds, per domain, the variance on the frozen scale at year 0 (the reference for
    converting a reliability into an error variance, audit A5) and at the landmark by phenotype."""
    from .data import Panel
    from .transforms import NormalScores

    part = part.set_index("PATNO") if "PATNO" in part.columns else part
    prim = part[part.primary]
    g = grid[grid.PATNO.isin(prim.index)].copy()
    if motor_on:
        for dname in ("tremor", "brady_rigidity", "axial"):
            on = g.motor_state.eq("ON only")
            g.loc[on, dname] = g.loc[on, dname + "_on"]
    ns = NormalScores().fit(g[domains].to_numpy(float))
    var_y0 = np.nanvar(ns.transform(g.loc[g.year == 0, domains].to_numpy(float)), axis=0, ddof=1)
    keep = prim[prim.phenotype.isin(["TD", "PIGD"]) & prim.reached_y1].index
    g = g[g.PATNO.isin(keep) & (g.year >= 1)].sort_values(["PATNO", "year"])
    g = g[g.PATNO.map(g.groupby("PATNO").year.min()) == 1]                      # starts at the landmark
    ids = list(dict.fromkeys(g.PATNO))
    z = pd.DataFrame({"era": (prim.loc[ids, "era"] == "PPMI 2.0").astype(float),
                      "age_onset": prim.loc[ids, "age_onset"], "male": prim.loc[ids, "male"].astype(float),
                      "duration": prim.loc[ids, "duration_years"]})
    z = z.fillna(z.mean())
    z = (z - z.mean()).to_numpy(float)
    y, t, dose, expo, on, yr = [], [], [], [], [], []
    for pid, rows in g.groupby("PATNO", sort=False):
        y.append(ns.transform(rows[domains].to_numpy(float)))
        t.append(rows.date_years.to_numpy(float))
        dose.append(rows.ledd.to_numpy(float) / 100.0)
        expo.append(rows.exposure.to_numpy(float) / 100.0)
        on.append((rows.motor_state.eq("ON only") & motor_on).to_numpy(float))
        yr.append(rows.year.to_numpy(int))
    group = (prim.loc[ids, "phenotype"] == "PIGD").astype(int).to_numpy()
    panel = Panel(y, t, group, z=z, dose=dose, exposure=expo, on=on, domains=list(domains), ids=np.asarray(ids),
                  meta={"source": "PPMI", "motor_on": motor_on}, year=yr)
    y1 = np.array([yi[0] for yi in y])
    var_y1 = {str(k): np.nanvar(y1[group == k], axis=0, ddof=1).tolist() for k in (0, 1)}
    info = {"domains": list(domains), "covariates": ["era (PPMI 2.0)", "age at onset", "male", "duration"],
            "treated_domains": [d for d in domains if d in TREATED_DOMAINS], "motor_on": motor_on,
            "counts": panel.aggregate_summary(), "variance_year0": dict(zip(domains, var_y0.tolist())),
            "variance_landmark_by_group": var_y1}
    return panel, ns, info


def screening_retest(v: pd.DataFrame, part: pd.DataFrame, grid: pd.DataFrame,
                     domains=("tremor", "brady_rigidity", "axial", "function"), max_gap_days: int = 120) -> dict:
    """Test-retest reliability from PPMI's own repeat: MDS-UPDRS Parts II and III given at
    screening and at baseline, before treatment (for D28). Pooled over the primary population
    (no phenotype split), on the frozen normal-scores scale fitted to the grid. Returns aggregate
    ICC(2,1), Pearson r, the pairs' variance on that scale, and the implied error variance
    (1 - ICC) times that variance (audit A5: the ICC is a share of the sample's own variance, which
    in early disease is well below the pooled scale's 1)."""
    from .transforms import NormalScores

    part = part.set_index("PATNO") if "PATNO" in part.columns else part
    ids = set(part.index[part.primary])
    ok = v.PATNO.isin(ids) & (v.motor_state == "untreated")
    sc = v[ok & (v.EVENT_ID == "SC")].drop_duplicates("PATNO").set_index("PATNO")
    bl = v[ok & (v.EVENT_ID == "BL")].drop_duplicates("PATNO").set_index("PATNO")
    both = sc.index.intersection(bl.index)
    gap = (bl.loc[both, "date"] - sc.loc[both, "date"]).dt.days.abs()
    both = gap.index[gap <= max_gap_days]
    g = grid[grid.PATNO.isin(ids)]
    ns = NormalScores().fit(g[list(domains)].to_numpy(float))
    x = ns.transform(sc.loc[both, list(domains)].to_numpy(float))
    y = ns.transform(bl.loc[both, list(domains)].to_numpy(float))
    out = {"pairs": int(len(both)), "median_gap_days": float(gap.loc[both].median()) if len(both) else None}
    for j, dname in enumerate(domains):
        m = ~(np.isnan(x[:, j]) | np.isnan(y[:, j]))
        n = int(m.sum())
        if n < 50:
            out[dname] = f"fewer than 50 pairs"
            continue
        a, b = x[m, j], y[m, j]
        Y = np.column_stack([a, b])
        grand = Y.mean()
        msr = 2 * np.sum((Y.mean(1) - grand) ** 2) / (n - 1)
        msc = n * np.sum((Y.mean(0) - grand) ** 2) / 1
        mse = (np.sum((Y - grand) ** 2) - msr * (n - 1) - msc) / (n - 1)
        icc = (msr - mse) / (msr + mse + 2 * (msc - mse) / n)
        v_ref = float(0.5 * (np.var(a, ddof=1) + np.var(b, ddof=1)))
        out[dname] = {"n": n, "icc_2_1": float(icc), "pearson_r": float(np.corrcoef(a, b)[0, 1]),
                      "sample_variance": v_ref, "error_variance": float((1 - icc) * v_ref),
                      "error_variance_unscaled": float(1 - icc)}
    return out

# Ported from escalation-study-v2/src/escalation_v2/build.py (29 October 2026). Changes are marked 'escalation-discovery:'.
"""build — raw PPMI download -> data/derived/{subjects,visits,medlog,grid}.csv (DESIGN.md §1).

    uv run escalation build

subjects.csv  one row per PD participant: cohort, baseline covariates, enrolment/exit/death/device dates
visits.csv    one row per participant x clinical visit: date, months since baseline, scores, LEDD by class
medlog.csv    one row per medication episode (LEDD log + PD drugs of the concomitant-medication log) with
              start/stop dates and LEDD — the table that dates an escalation
grid.csv      annual grid: for each participant and year the visit within +/-4 months of the anniversary with
              the most complete clinical battery; MDS-UPDRS IV set to 0 before any PD treatment
Aggregates (results/build_*.csv): counts per cohort, missingness, agreement of the visit treatment flag with
the dated log. PPMI dates have month resolution; every date is placed on the 15th.
"""
import re

import numpy as np
import pandas as pd

from . import config as C

DROP_EVENTS = {"SC", "PW", "TRANS"}  # screening / premature-withdrawal / transition visits. ST, the Symptomatic Therapy visit
# (PPMI code list), is a clinical assessment and is kept (corrected 28 October 2026; it was dropped as if it were screening)


# ---------------------------------------------------------------------------- scale scoring
def score_gds(df):
    """GDS-15: 10 items scored when answered yes, 5 reverse items when answered no (>= 13 answered)."""
    yes = ["GDSDROPD", "GDSEMPTY", "GDSBORED", "GDSAFRAD", "GDSHLPLS", "GDSHOME", "GDSMEMRY", "GDSWRTLS", "GDSHOPLS", "GDSBETER"]
    no = ["GDSSATIS", "GDSGSPIR", "GDSHAPPY", "GDSALIVE", "GDSENRGY"]
    items = pd.concat([df[yes] == 1, df[no] == 0], axis=1).astype(float)
    items[df[yes + no].isna().to_numpy()] = np.nan
    return items.sum(axis=1, min_count=13)


def score_scopa(df):
    """SCOPA-AUT items 1-21 (sexual items excluded); the catheter code 9 on the urinary items scores 3."""
    x = df[[f"SCAU{i}" for i in range(1, 22)]].astype(float).replace(9, 3)
    return x.sum(axis=1, min_count=19)


def score_rbdsq(df):
    """RBDSQ: 12 sleep items plus one point if any neurological condition is reported (>= 10 items)."""
    items = ["DRMVIVID", "DRMAGRAC", "DRMNOCTB", "SLPLMBMV", "SLPINJUR", "DRMVERBL", "DRMFIGHT", "DRMUMV", "DRMOBJFL",
             "MVAWAKEN", "DRMREMEM", "SLPDSTRB"]
    neuro = ["STROKE", "HETRA", "PARKISM", "RLS", "NARCLPSY", "DEPRS", "EPILEPSY", "BRNINFM", "CNSOTH"]
    return df[items].sum(axis=1, min_count=10) + (df[neuro].fillna(0).sum(axis=1) > 0).astype(float)


def score_quip(df):
    """QUIP-Current-Short: number of positive domains (gambling, sex, buying, eating, other behaviours; 0-5)
    and the number of positive core impulse-control domains (gambling, sex, buying; 0-3)."""
    dom = [["TMGAMBLE", "CNTRLGMB"], ["TMSEX", "CNTRLSEX"], ["TMBUY", "CNTRLBUY"], ["TMEAT", "CNTRLEAT"],
           ["TMTORACT", "TMTMTACT", "TMTRWD"]]
    answered = df[sum(dom, [])].notna().any(axis=1)
    total = sum((df[d].fillna(0).sum(axis=1) > 0).astype(float) for d in dom).where(answered)
    core = sum((df[d].fillna(0).sum(axis=1) > 0).astype(float) for d in dom[:3]).where(answered)
    return total, core


# ---------------------------------------------------------------------------- participants
def participants():
    """One row per participant of the PD cohort (COHORT == 1) with baseline and status variables."""
    ps = C.read_raw("_Subject_Characteristics", "Participant_Status")
    s = ps[(ps.COHORT == 1) & ~ps.ENROLL_STATUS.isin(["Screen failed", "Excluded", "Declined", "Screened", "Pending"])].copy()
    s["enroll_date"] = C.month_date(s.ENROLL_DATE)
    s["status_date"] = C.month_date(s.STATUS_DATE)
    genetic = (s.ENRLLRRK2 == 1) | (s.ENRLGBA == 1) | (s.ENRLSNCA == 1) | (s.ENRLPRKN == 1) | (s.ENRLPINK1 == 1)
    s["genetic_enrolled"] = genetic.astype(int)
    s["era"] = np.where(s.enroll_date.dt.year < 2020, "PPMI1", "PPMI2")
    s = s[["PATNO", "ENROLL_STATUS", "enroll_date", "status_date", "ENROLL_AGE", "genetic_enrolled", "era"]]
    s = s.rename(columns={"ENROLL_AGE": "age_enroll", "ENROLL_STATUS": "enroll_status"})

    dem = C.read_raw("_Subject_Characteristics", "Demographics").drop_duplicates("PATNO", keep="last")
    dem["male"] = (dem.SEX == 1).astype(float).where(dem.SEX.notna())
    dem["birth_date"] = C.month_date(dem.BIRTHDT)
    s = s.merge(dem[["PATNO", "male", "birth_date"]], on="PATNO", how="left")

    soc = C.read_raw("_Subject_Characteristics", "Socio-Economics").sort_values("EVENT_ID").drop_duplicates("PATNO")
    s = s.merge(soc[["PATNO", "EDUCYRS"]].rename(columns={"EDUCYRS": "educ_years"}), on="PATNO", how="left")

    gen = C.read_raw("_Subject_Characteristics", "iu_genetic_consensus_20251025")
    gen["apoe4"] = gen.APOE.astype("string").str.contains("E4").astype("float")
    gen.loc[gen.APOE.isna(), "apoe4"] = np.nan
    s = s.merge(gen[["PATNO", "apoe4", "LRRK2", "GBA", "SNCA", "PRKN", "PINK1"]], on="PATNO", how="left")

    dx = C.read_raw("Medical_History", "PD_Diagnosis_History").drop_duplicates("PATNO", keep="last")
    dx["pd_dx_date"] = C.month_date(dx.PDDXDT)
    dx["symptom_onset_date"] = C.month_date(dx.SXDT)
    s = s.merge(dx[["PATNO", "pd_dx_date", "symptom_onset_date"]], on="PATNO", how="left")

    concl = C.read_raw("Study_Enrollment", "Conclusion_of_Study_Participation")
    concl["exit_date"] = C.month_date(concl.WDDT)
    concl = concl.sort_values("exit_date").drop_duplicates("PATNO", keep="last")
    s = s.merge(concl[["PATNO", "exit_date", "WDDEATH"]], on="PATNO", how="left")
    s["died"] = ((s.WDDEATH == 1) | (s.enroll_status == "Withdraw Deceased")).astype(int)
    s["death_date"] = s.exit_date.where(s.died == 1).fillna(s.status_date.where(s.died == 1))
    s = s.drop(columns="WDDEATH")

    surg = C.read_raw("Medical_History", "Procedure_for_PD_Log")
    surg["device_date"] = C.month_date(surg.PDSURGDT)
    surg = surg.sort_values("device_date").drop_duplicates("PATNO")
    surg["device_type"] = surg.PDSURGTP.map({1: "DBS", 2: "LCIG", 3: "other", 4: "unknown", 5: "HIFU", 6: "focused US"})
    s = s.merge(surg[["PATNO", "device_date", "device_type"]], on="PATNO", how="left")
    return s


# ---------------------------------------------------------------------------- visits
def valid_items(df):
    """MDS-UPDRS items are scored 0-4; PPMI stores non-scores as codes (101 = unable to rate, 102-108 = other
    reasons). Codes become missing before any item is summed into a sub-score."""
    return df.apply(pd.to_numeric, errors="coerce").where(lambda x: x.ge(0) & x.le(4))


def visits(ids):
    """One row per participant x visit (EVENT_ID), anchored on the MDS-UPDRS III exam date."""
    u3 = C.read_raw("Motor___MDS-UPDRS", "MDS-UPDRS_Part_III")
    u3 = u3[u3.PATNO.isin(ids)].copy()
    u3["state"] = np.select([u3.PDSTATE == "OFF", u3.PDSTATE == "ON", u3.PAG_NAME.isin(["NUPDRS3A", "NUPDR3ON"])],
                            ["OFF", "ON", "ON"], "UNTREATED")
    u3.loc[(u3.state == "UNTREATED") & (u3.PAG_NAME == "NUPDR3OF"), "state"] = "OFF"
    u3["date"] = C.month_date(u3.INFODT)
    u3["NHY"] = u3.NHY.where(u3.NHY <= 5)
    axial_items = ["NP3SPCH", "NP3RISNG", "NP3GAIT", "NP3FRZGT", "NP3PSTBL", "NP3POSTR"]
    u3[axial_items] = valid_items(u3[axial_items])
    u3["axial"] = u3[axial_items].sum(axis=1, min_count=5)
    off = u3[u3.state.isin(["OFF", "UNTREATED"])].sort_values("NP3TOT", na_position="first")
    off = off.drop_duplicates(["PATNO", "EVENT_ID"], keep="last")
    off = off[["PATNO", "EVENT_ID", "date", "NP3TOT", "NHY", "axial", "PDTRTMNT", "state"]].rename(
        columns={"NP3TOT": "updrs3_off", "NHY": "hy_off", "axial": "axial_off", "state": "off_state"})
    on = u3[u3.state == "ON"].drop_duplicates(["PATNO", "EVENT_ID"], keep="last")
    on = on[["PATNO", "EVENT_ID", "date", "NP3TOT", "NHY", "PDTRTMNT"]].rename(
        columns={"NP3TOT": "updrs3_on", "NHY": "hy_on", "date": "date_on", "PDTRTMNT": "PDTRTMNT_on"})
    v = off.merge(on, on=["PATNO", "EVENT_ID"], how="outer")
    v["date"] = v.date.fillna(v.date_on)
    v["PDTRTMNT"] = v.PDTRTMNT.fillna(v.PDTRTMNT_on)
    v = v.drop(columns=["date_on", "PDTRTMNT_on"])

    def add(df, cols, rename=None, sc_as_bl=False, datecol="INFODT", how="left"):
        """Merge one form by (participant, visit). Only the MDS-UPDRS I/II and Schwab & England forms may create a
        visit the motor exam lacks (how="outer"); ancillary forms attach to existing visits only."""
        nonlocal v
        df = df[df.PATNO.isin(ids)].copy()
        if sc_as_bl:  # a screening-visit assessment stands in for baseline when baseline has none
            has_bl = set(df.loc[df.EVENT_ID == "BL", "PATNO"])
            df.loc[(df.EVENT_ID == "SC") & ~df.PATNO.isin(has_bl), "EVENT_ID"] = "BL"
        df["date_"] = C.month_date(df[datecol]) if datecol else pd.NaT
        df = df.sort_values("date_").drop_duplicates(["PATNO", "EVENT_ID"], keep="last")
        df = df[["PATNO", "EVENT_ID", "date_"] + cols].rename(columns=rename or {})
        v = v.merge(df, on=["PATNO", "EVENT_ID"], how=how)
        v["date"] = v.date.fillna(v.date_)
        v = v.drop(columns="date_")

    mot, nm = "Motor___MDS-UPDRS", "Non-motor_Assessments"
    u1 = C.read_raw(mot, "MDS-UPDRS_Part_I")
    u1[["NP1HALL", "NP1COG", "NP1DPRS"]] = valid_items(u1[["NP1HALL", "NP1COG", "NP1DPRS"]])
    add(u1, ["NP1HALL", "NP1COG", "NP1DPRS", "NP1RTOT"],
        {"NP1HALL": "np1_hall", "NP1COG": "np1_cog", "NP1DPRS": "np1_dprs", "NP1RTOT": "updrs1_rater"}, how="outer")
    add(C.read_raw(mot, "MDS-UPDRS_Part_I_Patient_Questionnaire"), ["NP1PTOT"], {"NP1PTOT": "updrs1_pt"}, how="outer")
    add(C.read_raw(mot, "MDS_UPDRS_Part_II__Patient_Questionnaire"), ["NP2PTOT"], {"NP2PTOT": "updrs2"}, how="outer")
    u4 = C.read_raw(mot, "MDS-UPDRS_Part_IV__Motor_Complications")
    u4[["NP4OFF", "NP4FLCTI", "NP4WDYSK", "NP4DYSKI"]] = valid_items(u4[["NP4OFF", "NP4FLCTI", "NP4WDYSK", "NP4DYSKI"]])
    u4["fluct"] = u4[["NP4OFF", "NP4FLCTI"]].sum(axis=1, min_count=1)       # wearing-off: time spent OFF, impact
    u4["dysk"] = u4[["NP4WDYSK", "NP4DYSKI"]].sum(axis=1, min_count=1)      # dyskinesia: time, impact
    add(u4, ["NP4TOT", "fluct", "dysk"], {"NP4TOT": "updrs4"})
    add(C.read_raw(mot, "Modified_Schwab___England_Activities_of_Daily_Living"), ["MSEADLG"], {"MSEADLG": "schwab_england"}, how="outer")
    vs = C.read_raw("Medical_History", "Vital_Signs")
    vs["bp_drop"] = vs.SYSSUP - vs.SYSSTND
    add(vs, ["bp_drop", "WGTKG"], {"WGTKG": "weight"}, sc_as_bl=True)
    add(C.read_raw(nm, "Montreal_Cognitive_Assessment__MoCA_"), ["MCATOT"], {"MCATOT": "moca"}, sc_as_bl=True)
    gds = C.read_raw(nm, "Geriatric_Depression_Scale__Short_Version_")
    gds["gds"] = score_gds(gds)
    add(gds, ["gds"], sc_as_bl=True)
    sc = C.read_raw(nm, "SCOPA-AUT")
    sc["scopa_aut"] = score_scopa(sc)
    add(sc, ["scopa_aut"], sc_as_bl=True)
    es = C.read_raw(nm, "Epworth_Sleepiness_Scale")
    es["ess"] = es[[f"ESS{i}" for i in range(1, 9)]].sum(axis=1, min_count=7)
    add(es, ["ess"], sc_as_bl=True)
    rb = C.read_raw(nm, "REM_Sleep_Behavior_Disorder_Screening_Questionnaire")
    rb["rbdsq"] = score_rbdsq(rb)
    add(rb, ["rbdsq"], sc_as_bl=True)
    qp = C.read_raw(nm, "QUIP-Current-Short")
    qp["quip"], qp["quip_core"] = score_quip(qp)
    add(qp, ["quip", "quip_core"], sc_as_bl=True)
    dat = C.read_raw("Imaging", "Xing_Core_Lab_-_Quant_SBR")
    dat = dat[dat.DATSCAN_ANALYZED == "Yes"].copy()
    add(dat, ["PUTAMEN_REF_CWM", "CAUDATE_REF_CWM"], {"PUTAMEN_REF_CWM": "dat_putamen", "CAUDATE_REF_CWM": "dat_caudate"},
        sc_as_bl=True, datecol=None)
    age = C.read_raw("_Subject_Characteristics", "Age_at_visit").groupby(["PATNO", "EVENT_ID"], as_index=False).AGE_AT_VISIT.min()
    v = v.merge(age.rename(columns={"AGE_AT_VISIT": "age"}), on=["PATNO", "EVENT_ID"], how="left")

    v = v[v.date.notna()].copy()
    v["motor_nonaxial_off"] = v.updrs3_off - v.axial_off
    v["ldopa_response"] = v.updrs3_off - v.updrs3_on   # observed ON benefit when both exams exist
    bl = v[v.EVENT_ID == "BL"].groupby("PATNO").date.min()
    first = v.groupby("PATNO").date.min()
    v["bl_date"] = v.PATNO.map(bl.reindex(first.index).fillna(first))
    v["months"] = ((v.date - v.bl_date).dt.days / C.MONTH).round().astype(int)
    v = v[~v.EVENT_ID.isin(DROP_EVENTS)]
    return v.sort_values(["PATNO", "date", "EVENT_ID"]).reset_index(drop=True)


# ---------------------------------------------------------------------------- medication log
CLASS_PATTERNS = {  # tolerant of the misspellings in the free-text logs
    "levodopa": r"lev[a-z]*d[oa]p|levocomp|levopar|nacom|isicom|sinemet|madopar|rytary|crexont|stalevo|duodopa|inbrija|"
                r"dopicar|syndopa|prolopa|parcopa|benserazid|carbidopa",
    # ropinirole and its misspellings start a word with "ropin" / "ropir"; a bare "ropi" also matched bupropion,
    # (fluticasone) propionate, ipratropium, tiotropium, atropine and benztropine (fixed 22 October 2026)
    "agonist": r"\bropi[nr]|pramip|rotig|mirapex|requip|neupro|nupro|apomorph|apokyn|piribedil|clarium|sifrol|kynmobi|bromocript|"
               r"cabergol|pergolid",
    "maob": r"rasag|azilect|seleg|eldepryl|zelapar|safinam|xadago|deprenyl",
    "comt": r"entcap|entacap|comtan|opicapon|ongentys|tolcapon|tasmar",
    "amantadine": r"amant|gocovri|osmolex|symmetrel|pk.?merz",
    "anticholinergic": r"trihex|benztrop|artane|cogentin|biperiden",
}


def classify(names):
    name = names.astype("string").str.lower()
    cls = pd.Series("other", index=names.index)
    for k, pat in CLASS_PATTERNS.items():
        cls[name.str.contains(pat, na=False) & (cls == "other")] = k
    cls[name.str.contains("stalevo", na=False)] = "levodopa"
    return cls


def medication_log(ids):
    """Medication episodes with dates: every LEDD-log row (with its LEDD) plus PD-drug rows of the general
    concomitant-medication log that have no LEDD-log counterpart (same participant, class and start month)."""
    ld = C.read_raw("Medical_History", "LEDD_Concomitant_Medication_Log")
    ld = ld[ld.PATNO.isin(ids)].copy()
    ld["drug_class"] = classify(ld.LEDTRT)
    ld["ledd"] = pd.to_numeric(ld.LEDD, errors="coerce")
    ld["ld_multiplier"] = ld.LEDD.astype("string").str.extract(r"LD x ([\d.]+)")[0].astype(float)
    ld.loc[ld.ld_multiplier.notna(), "drug_class"] = "comt"   # entacapone/opicapone recorded as a levodopa multiplier
    ld = ld.drop_duplicates(["PATNO", "LEDTRT", "LEDDOSE", "LEDDOSFRQ", "LEDDSTRMG", "STARTDT", "STOPDT"])
    ld = pd.DataFrame({"PATNO": ld.PATNO, "source": "LEDD", "drug": ld.LEDTRT, "drug_class": ld.drug_class, "ledd": ld.ledd,
                       "ld_multiplier": ld.ld_multiplier, "start": C.month_date(ld.STARTDT), "stop": C.month_date(ld.STOPDT)})
    cm = C.read_raw("Medical_History", "Concomitant_Medication_Log")
    cm = cm[cm.PATNO.isin(ids)].copy()
    cm["drug_class"] = classify(cm.CMTRT)
    cm = cm[cm.drug_class.isin(["levodopa", "agonist", "maob", "comt", "amantadine"])]
    cm = pd.DataFrame({"PATNO": cm.PATNO, "source": "CM", "drug": cm.CMTRT, "drug_class": cm.drug_class, "ledd": np.nan,
                       "ld_multiplier": np.nan, "start": C.month_date(cm.STARTDT), "stop": C.month_date(cm.STOPDT)})
    key = ["PATNO", "drug_class", "start"]
    cm = cm.merge(ld[key].drop_duplicates().assign(_in_ledd=1), on=key, how="left")
    cm = cm[cm._in_ledd.isna()].drop(columns="_in_ledd")
    out = pd.concat([ld, cm], ignore_index=True)
    out = out[out.start.notna()].sort_values(["PATNO", "start", "drug_class"]).reset_index(drop=True)
    return out


def ledd_at_visits(medlog, v):
    """LEDD by drug class in force at each visit date (an episode is active from its start month through its
    stop month; open episodes stay active). COMT multipliers ("LD x 0.33") add that fraction of the levodopa LEDD."""
    ld = medlog[(medlog.source == "LEDD")]
    vd = v[["PATNO", "EVENT_ID", "date"]].merge(ld, on="PATNO", how="inner")
    act = vd[(vd.start <= vd.date) & (vd.stop.isna() | (vd.stop >= vd.date))].copy()
    by = act.pivot_table(index=["PATNO", "EVENT_ID"], columns="drug_class", values="ledd", aggfunc="sum").fillna(0)
    by.columns = [f"ledd_{c}" for c in by.columns]
    for c in ["ledd_levodopa", "ledd_agonist", "ledd_maob", "ledd_comt", "ledd_amantadine", "ledd_anticholinergic", "ledd_other"]:
        if c not in by:
            by[c] = 0.0
    mult = act[act.ld_multiplier.notna()].groupby(["PATNO", "EVENT_ID"]).ld_multiplier.max()
    by["ledd_comt"] = by.ledd_comt + mult.reindex(by.index).fillna(0) * by.ledd_levodopa
    by["ledd_total"] = by[[c for c in by.columns if c.startswith("ledd_")]].sum(axis=1)
    by["drug_classes"] = act.groupby(["PATNO", "EVENT_ID"]).drug_class.agg(lambda s: ";".join(sorted(set(s) - {"other"})))
    return by.reset_index()


def levodopa_history_from_log(medlog, v):
    """Months on levodopa and cumulative levodopa (mg-years) at each visit date, integrated from the dated log."""
    ld = medlog[(medlog.source == "LEDD") & (medlog.drug_class == "levodopa") & medlog.ledd.notna()]
    first = ld.groupby("PATNO").start.min()
    v = v[["PATNO", "date"]].copy()
    v["months_on_ldopa_log"] = ((v.date - v.PATNO.map(first)).dt.days / C.MONTH).clip(lower=0).fillna(0)
    rows = v.merge(ld[["PATNO", "ledd", "start", "stop"]], on="PATNO", how="left")
    end = rows[["stop", "date"]].min(axis=1).fillna(rows.date)
    exposure = ((end - rows.start).dt.days / 365.25).clip(lower=0) * rows.ledd
    v["cum_ldopa_log"] = exposure.groupby([rows.PATNO, rows.date]).sum().reindex(pd.MultiIndex.from_frame(v[["PATNO", "date"]])).to_numpy()
    return v[["months_on_ldopa_log", "cum_ldopa_log"]].fillna(0)


# ---------------------------------------------------------------------------- grid
def annual_grid(v, max_year=14):
    """One visit per participant-year: within +/-4 months of the anniversary, the visit with the most complete
    core battery. MDS-UPDRS IV (and its wearing-off / dyskinesia parts) is 0 before any PD treatment."""
    g = v.copy()
    for c in ["updrs4", "fluct", "dysk"]:
        g[c] = g[c].where(g.on_pd_meds == 1, g[c].fillna(0))
    g["year"] = (g.months / 12).round().astype(int)
    g = g[(g.months - 12 * g.year).abs() <= 4]
    g["n_obs"] = g[[C.NODES[k] for k in C.CORE_NODES]].notna().sum(axis=1)
    g = g.sort_values(["PATNO", "year", "n_obs"], kind="stable").drop_duplicates(["PATNO", "year"], keep="last")
    g = g[g.year <= max_year]
    keep = list(dict.fromkeys(["PATNO", "year", "months", "EVENT_ID", "date", "age", "on_pd_meds", "ledd_total", "months_on_ldopa_log",
                               "cum_ldopa_log"] + list(C.NODES.values())))      # ledd_total is also a node: keep it once
    out = g[keep].rename(columns={c: k for k, c in C.NODES.items()})
    return out.sort_values(["PATNO", "year"]).reset_index(drop=True)


# ---------------------------------------------------------------------------- main
def build():
    out = C.derived_dir()
    subj = participants()
    ids = set(subj.PATNO)
    v = visits(ids)
    medlog = medication_log(ids)
    medlog.to_csv(out / "medlog.csv", index=False)

    v = v.merge(ledd_at_visits(medlog, v), on=["PATNO", "EVENT_ID"], how="left")
    lcols = [c for c in v.columns if c.startswith("ledd_")]
    v[lcols] = v[lcols].fillna(0)   # never in the LEDD log = untreated
    v["drug_classes"] = v.drug_classes.fillna("")
    v["on_pd_meds"] = ((v.ledd_total > 0) | (v.PDTRTMNT == 1)).astype(int)
    v[["months_on_ldopa_log", "cum_ldopa_log"]] = levodopa_history_from_log(medlog, v).to_numpy()
    grp = v.groupby("PATNO")
    v["visit_no"] = grp.cumcount()
    first_l = v[v.ledd_levodopa > 0].groupby("PATNO").months.min()   # visit-based history (as the first implementation)
    v["months_on_ldopa"] = (v.months - v.PATNO.map(first_l)).clip(lower=0).fillna(0)
    gap = grp.months.diff().fillna(0).clip(lower=0)
    v["cum_ldopa"] = (grp.ledd_levodopa.shift().fillna(0) * gap).groupby(v.PATNO).cumsum() / 12
    v.to_csv(out / "visits.csv", index=False)

    # participant-level summaries and cohort assignment
    fu = v.groupby("PATNO").agg(n_visits=("EVENT_ID", "size"), first_visit=("date", "min"), last_visit=("date", "max"),
                                fu_months=("months", "max"), bl_date=("bl_date", "first"))
    fu["treated_at_bl"] = v[v.visit_no == 0].set_index("PATNO").on_pd_meds
    blv = v[v.visit_no == 0].set_index("PATNO")
    fu["bl_dat_putamen"] = blv.dat_putamen
    fu["bl_dat_caudate"] = blv.dat_caudate
    subj = subj.merge(fu.reset_index(), on="PATNO", how="left")
    subj["disease_dur_months"] = ((subj.bl_date - subj.pd_dx_date).dt.days / C.MONTH).round(1)
    has = subj.n_visits.fillna(0) > 0
    sporadic = subj.genetic_enrolled == 0
    subj["cohort"] = np.select(
        [has & (subj.era == "PPMI1") & sporadic & (subj.treated_at_bl == 0), has & (subj.era == "PPMI2") & sporadic, has & ~sporadic],
        ["ppmi1", "ppmi2", "genetic"], "")
    subj.to_csv(out / "subjects.csv", index=False)

    grid = annual_grid(v)
    grid.to_csv(out / "grid.csv", index=False)

    # aggregates
    rows = []
    for c, label in C.COHORTS.items():
        pid = set(subj.loc[subj.cohort == c, "PATNO"])
        vc, gc = v[v.PATNO.isin(pid)], grid[grid.PATNO.isin(pid)]
        rows.append(dict(cohort=c, label=label, participants=len(pid), visits=len(vc), grid_rows=len(gc),
                         grid_rows_per_participant=round(len(gc) / max(1, gc.PATNO.nunique()), 2),
                         visits_on_levodopa=int((vc.ledd_levodopa > 0).sum()),
                         participants_ever_levodopa=int(vc[vc.ledd_levodopa > 0].PATNO.nunique()),
                         medlog_episodes=int(medlog.PATNO.isin(pid).sum()),
                         medlog_episodes_cm_only=int((medlog.PATNO.isin(pid) & (medlog.source == "CM")).sum())))
    counts = pd.DataFrame(rows)
    counts.to_csv(C.results_dir() / "build_counts.csv", index=False)
    miss = []
    for c in C.COHORTS:
        gc = grid[grid.PATNO.isin(set(subj.loc[subj.cohort == c, "PATNO"]))]
        for k in C.NODES:
            miss.append(dict(cohort=c, variable=k, missing_share=round(float(gc[k].isna().mean()), 3)))
    pd.DataFrame(miss).pivot(index="variable", columns="cohort", values="missing_share").to_csv(C.results_dir() / "build_missingness.csv")
    pd_v = v[v.PATNO.isin(set(subj.loc[subj.cohort != "", "PATNO"])) & v.PDTRTMNT.notna()]
    agree = pd.crosstab(pd_v.PDTRTMNT.map({0: "visit: untreated", 1: "visit: treated"}),
                        (pd_v.ledd_total > 0).map({False: "log: LEDD 0", True: "log: LEDD > 0"}))
    agree.to_csv(C.results_dir() / "build_ledd_agreement.csv")
    print(counts.to_string(index=False))
    print(f"\nagreement of the visit treatment flag with the dated medication log (visits):\n{agree.to_string()}")
    print(f"\nmedication log: {len(medlog)} episodes, {medlog.PATNO.nunique()} participants; by class:",
          medlog.drug_class.value_counts().to_dict())
    return subj, v, medlog, grid


def main(argv=None):
    C.require_ppmi_approval("data build")  # escalation-discovery: PROTOCOL.md §11 gate
    build()


if __name__ == "__main__":
    main()

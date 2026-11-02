# Ported from escalation-study-v2/src/escalation_v2/trials.py (29 October 2026). Changes are marked 'escalation-discovery:'.
"""trials — the decision-time target trial, the table of the paper (DESIGN.md §2); with --legacy also the visit-anchored
trial table of the first implementation (supplement only).

    uv run escalation trials [--legacy]

The decision-time table (`decision_time_table`; docs/reviewer/decision_time_design.md and
causal_pathway_and_optimal_dosing_design.md): one trial at every clinical assessment with the strategy read from the
dated log — escalate within the 3-month grace period, no increase within 6 months, or deviation — eligibility that uses
nothing after the visit, outcomes 3, 6, 12 and 24 months after it (the 3-month horizon is descriptive: about 15% of
assessments have a visit 1-5 months later), the pre-visit trajectory of the motor and functional scales (change since
the latest measurement 5-18 months earlier), the months since the last levodopa change (stable-dose run-in),
co-decisions inside the grace window (adjunct or amantadine started or increased), and the step size of the
escalation. `population="initiation"` builds the same table for assessments at which the participant has never been on
levodopa (start levodopa within 3 months vs no start within 6 months): the positive control on the same machinery.
Outputs: data/derived/trials_dt_<cohort>.csv, trials_dt_init_<cohort>.csv; results/trials_dt_counts.csv,
trials_dt_censoring.csv, trials_dt_init_counts.csv, trials_dt_init_censoring.csv. The two sensitivity tables of the
primary cell (outcome window ±6 months: trials_dt_<cohort>_w6.csv; history carry-forward ≤ 7 months:
trials_dt_<cohort>_c7.csv) are built by `uv run escalation estimate --sensitivities`.

The visit-anchored table (`--legacy`): one row per eligible person-visit t with levodopa LEDD > 0 at t, visit t+1 on the
annual grid with MDS-UPDRS II observed, no device-aided therapy by t+1 and no new PD drug class between t and t+1; the
decision A = levodopa dose recorded at t+1 with the medication-log dates of the change; outcomes at t+1, t+2 and twelve
months on the new dose; the initiation table as the first implementation's sanity anchor. Outputs:
data/derived/trials_<cohort>.csv, initiation_<cohort>.csv; results/trials_counts.csv, trials_dating.csv,
trials_censoring.csv.
"""
import numpy as np
import pandas as pd

from . import config as C
from .citest import copula

HIST = ["motor", "adl", "se", "cog", "dep", "aut", "ess", "rbd", "icd", "cx", "oh", "ldopa", "da", "maob", "ama", "cum_ldopa", "t_ldopa"]
EXTRA_HIST = ["motor_na", "axial", "fluct", "dysk", "hall", "icd_core", "comt", "hy", "motor_on", "weight", "response"]
OUTCOMES = {"adl": "MDS-UPDRS II", "cx": "MDS-UPDRS IV", "se": "Schwab & England", "motor": "OFF MDS-UPDRS III",
            "ess": "Epworth", "icd": "QUIP domains", "cog": "MoCA (expected null)"}
NEW_CLASSES = ["da", "maob", "ama", "comt"]
WINDOW = 4          # months around an anniversary that count as that year's visit
THRESHOLDS = (50, 100, 150)


def load():
    d = C.derived_dir()
    subj = pd.read_csv(d / "subjects.csv", low_memory=False, parse_dates=["enroll_date", "exit_date", "death_date", "device_date", "bl_date", "last_visit"])
    grid = pd.read_csv(d / "grid.csv", parse_dates=["date"])
    visits = pd.read_csv(d / "visits.csv", low_memory=False, parse_dates=["date"])
    medlog = pd.read_csv(d / "medlog.csv", parse_dates=["start", "stop"])
    return subj, grid, visits, medlog


def cohort_ids(subj, cohort):
    return set(subj.loc[subj.cohort == cohort, "PATNO"])


def trait_score(grid):
    """Baseline non-motor trait score: mean of the normal scores (within cohort) of RBD, autonomic, depression and
    sleepiness at year 0; one value per participant."""
    b = grid[grid.year == 0].set_index("PATNO")[["rbd", "aut", "dep", "ess"]]
    z = pd.DataFrame(copula(b.to_numpy()), index=b.index, columns=b.columns)
    return z.mean(axis=1, skipna=True).rename("trait0")


def ledd_steps(medlog, drug_class="levodopa"):
    """Change points of the LEDD in force for one drug class, per participant, from the dated LEDD log: one row per
    date at which the dose changed, with the level in force from that date on."""
    ld = medlog[(medlog.source == "LEDD") & (medlog.drug_class == drug_class) & medlog.ledd.notna()]
    up = pd.DataFrame({"PATNO": ld.PATNO, "date": ld.start, "chg": ld.ledd})
    down = pd.DataFrame({"PATNO": ld.PATNO, "date": ld.stop + pd.Timedelta(days=1), "chg": -ld.ledd}).dropna(subset=["date"])
    ev = pd.concat([up, down])
    ev["date"] = ev.date.dt.to_period("M").dt.to_timestamp() + pd.Timedelta(days=14)   # month resolution: the 15th
    ev = ev.groupby(["PATNO", "date"], as_index=False).chg.sum()     # a stop and a start in one month net out
    ev = ev[ev.chg.abs() > 1e-9].sort_values(["PATNO", "date"])
    ev["level"] = ev.groupby("PATNO").chg.cumsum().round(1)
    return ev.reset_index(drop=True)


def levodopa_steps(medlog):
    """Change points of the levodopa LEDD in force (see `ledd_steps`)."""
    return ledd_steps(medlog, "levodopa")


def date_changes(rows, steps):
    """For each trial row (row_id, PATNO, date_t, date_t1, A0): the dose changes in (t, t+1] from the log —
    their number, first and last date, and the first date at which the dose reached A0 + threshold."""
    m = rows[["row_id", "PATNO", "date_t", "date_t1", "A0"]].merge(steps, on="PATNO", how="inner")
    m = m[(m.date > m.date_t) & (m.date <= m.date_t1)]
    g = m.groupby("row_id")
    out = pd.DataFrame({"n_changes": g.size(), "first_change_date": g.date.min(), "last_change_date": g.date.max()})
    for thr in THRESHOLDS:
        hit = m[m.level >= m.A0 + thr].groupby("row_id").date.min()
        out[f"date_reach_{thr}"] = hit
    return out.reindex(rows.row_id)


def nearest_visit(rows, visits, target, after, cols, window=WINDOW):
    """Outcome values at the visit nearest to `target` (within +/-window months) and strictly after `after`."""
    cand = rows[["row_id", "PATNO", target, after]].merge(visits[["PATNO", "date"] + cols], on="PATNO")
    cand = cand[cand.date > cand[after]].copy()
    cand["dist"] = (cand.date - cand[target]).dt.days.abs() / C.MONTH
    cand = cand[cand.dist <= window].sort_values(["row_id", "dist"], kind="stable").drop_duplicates("row_id")
    return cand.set_index("row_id")[cols + ["date", "dist"]].reindex(rows.row_id)


def _wide(grid, cols):
    return {c: grid.pivot(index="PATNO", columns="year", values=c) for c in cols if c in grid.columns}


def _base(subj):
    sub = subj.set_index("PATNO")
    return pd.DataFrame({k: sub[c] for k, c in C.BASELINE.items()})


def _assemble(rows, w, keep, t, years, outcomes):
    d = pd.DataFrame(index=keep.index[keep])
    d["t"] = float(t)
    d["A"], d["A0"] = w["ldopa"][t + 1][keep], w["ldopa"][t][keep]
    d["delta"] = d.A - d.A0
    for c in HIST + EXTRA_HIST + ["months_on_ldopa_log", "cum_ldopa_log"]:
        if c in w:
            d[f"h_{c}"] = w[c][t][keep]
    for c in outcomes:
        d[f"y1_{c}"] = w[c][t + 1][keep]
        d[f"y2_{c}"] = w[c][t + 2][keep] if t + 2 in years else np.nan
    d["date_t"], d["date_t1"] = w["date"][t][keep], w["date"][t + 1][keep]
    d["date_t2"] = w["date"][t + 2][keep] if t + 2 in years else pd.NaT
    d["months_t1"] = w["months"][t + 1][keep]
    rows.append(d)


def _finish(rows, subj, grid, visits, steps, tr, outcomes, dated_clock=True):
    df = pd.concat(rows)
    df["PATNO"] = df.index.to_numpy()
    df = df.reset_index(drop=True)
    df["row_id"] = np.arange(len(df))
    df = df.merge(_base(subj), left_on="PATNO", right_index=True, how="left").merge(tr, left_on="PATNO", right_index=True, how="left")
    sub = subj.set_index("PATNO")
    for c in ["death_date", "device_date", "exit_date", "last_visit", "enroll_date"]:
        df[c] = df.PATNO.map(sub[c])
    df["enroll_year"] = df.enroll_date.dt.year
    # medication-log dating of the dose recorded at t+1
    ch = date_changes(df, steps)
    df = df.join(ch, on="row_id")
    df["n_changes"] = df.n_changes.fillna(0).astype(int)
    df["months_on_dose_t1"] = ((df.date_t1 - df.last_change_date).dt.days / C.MONTH).round(1)
    # clock for the twelve-months-on-dose outcome: the log date of the change; for an unchanged dose a start drawn
    # from the escalators' delay distribution, so both arms are measured at the same distance from visit t
    delays = (df.last_change_date - df.date_t).dt.days[df.n_changes > 0].to_numpy()
    rng = np.random.default_rng(0)
    drawn = pd.to_timedelta(rng.choice(delays, size=len(df)) if len(delays) else np.zeros(len(df)), unit="D")
    df["clock_start"] = df.last_change_date.where(df.n_changes > 0, df.date_t + drawn)
    df["clock_source"] = np.where(df.n_changes > 0, "medication log", "drawn from the escalation delay distribution")
    df["clock_delay_months"] = ((df.clock_start - df.date_t).dt.days / C.MONTH).round(1)
    df["target_12"] = df.clock_start + pd.Timedelta(days=round(12 * C.MONTH))
    # censoring of the t+2 outcome
    end2 = df.date_t1 + pd.Timedelta(days=round((12 + WINDOW) * C.MONTH))
    has_t2 = df.date_t2.notna()
    device_t2 = df.device_date.notna() & (df.device_date <= df.date_t2.fillna(end2))
    for c in outcomes:
        df.loc[device_t2, f"y2_{c}"] = np.nan
    df["censor2_reason"] = np.select(
        [has_t2 & ~device_t2 & df.y2_adl.notna(), device_t2, has_t2 & df.y2_adl.isna(),
         df.death_date.notna() & (df.death_date <= end2), df.exit_date.notna() & (df.exit_date <= end2),
         df.last_visit <= df.date_t1 + pd.Timedelta(days=round(WINDOW * C.MONTH))],
        ["observed", "device-aided therapy", "visit present, item missing", "death", "withdrawal", "no further visit (dropout or follow-up not yet reached)"],
        "visit outside the annual window")
    # twelve months on the dose recorded at t+1 (any clinical visit, not only grid visits)
    vcols = {c: C.NODES[c] for c in outcomes}
    vv = visits.copy()
    vv["updrs4"] = vv.updrs4.where(vv.on_pd_meds == 1, vv.updrs4.fillna(0))
    near = nearest_visit(df, vv, "target_12", "date_t", list(vcols.values()))
    for c, col in vcols.items():
        df[f"y12_{c}"] = near[col].to_numpy()
    df["date_12"], df["months_12_from_clock"] = near["date"].to_numpy(), ((near["date"] - df.clock_start.to_numpy()).dt.days / C.MONTH).round(1).to_numpy()
    device_12 = df.device_date.notna() & df.date_12.notna() & (df.device_date <= df.date_12)
    for c in outcomes:
        df.loc[device_12, f"y12_{c}"] = np.nan
    return df.drop(columns=["row_id", "target_12"])


def build_trials(cohort, subj, grid, visits, medlog):
    ids = cohort_ids(subj, cohort)
    g = grid[grid.PATNO.isin(ids)].sort_values(["PATNO", "year"])
    tr = trait_score(g)
    w = _wide(g, sorted(set(HIST + EXTRA_HIST + list(OUTCOMES) + ["date", "months", "months_on_ldopa_log", "cum_ldopa_log"])))
    dev = subj.set_index("PATNO").device_date
    years = sorted(w["ldopa"].columns)
    rows = []
    for t in years:
        if t + 1 not in years:
            continue
        on = (w["ldopa"][t] > 0) & w["ldopa"][t + 1].notna() & w["adl"][t + 1].notna()
        new_class = pd.Series(False, index=on.index)
        for c in NEW_CLASSES:
            if c in w:
                new_class |= (w[c][t].fillna(0) == 0) & (w[c][t + 1].fillna(0) > 0)
        devx = dev.reindex(on.index)
        device = devx.notna() & (devx <= w["date"][t + 1].reindex(on.index))
        keep = on & ~new_class & ~device
        if keep.sum():
            _assemble(rows, w, keep, t, years, OUTCOMES)
    df = _finish(rows, subj, g, visits, levodopa_steps(medlog), tr, OUTCOMES)
    df["cohort"] = cohort
    return df


def build_initiation(cohort, subj, grid, visits, medlog):
    """Person-visits not on levodopa at t with t+1 observed; E = levodopa started by t+1 (no device by t+1)."""
    ids = cohort_ids(subj, cohort)
    g = grid[grid.PATNO.isin(ids)].sort_values(["PATNO", "year"])
    tr = trait_score(g)
    w = _wide(g, sorted(set(HIST + EXTRA_HIST + list(OUTCOMES) + ["date", "months", "months_on_ldopa_log", "cum_ldopa_log"])))
    dev = subj.set_index("PATNO").device_date
    years = sorted(w["ldopa"].columns)
    rows = []
    for t in years:
        if t + 1 not in years:
            continue
        off = (w["ldopa"][t].fillna(0) == 0) & w["ldopa"][t + 1].notna() & w["adl"][t + 1].notna()
        devx = dev.reindex(off.index)
        device = devx.notna() & (devx <= w["date"][t + 1].reindex(off.index))
        keep = off & ~device
        if keep.sum():
            _assemble(rows, w, keep, t, years, OUTCOMES)
    df = _finish(rows, subj, g, visits, levodopa_steps(medlog), tr, OUTCOMES)
    df["E"] = (df.A > 0).astype(int)
    df["cohort"] = cohort
    return df


# ---------------------------------------------------------------------------- decision-time target trial
# docs/reviewer/decision_time_design.md: one trial per clinical assessment at which the participant is on levodopa by
# the log; the strategy is read from the dated log (a change logged in the visit's own month counts as the decision
# made at that visit); eligibility uses nothing after the visit. Windows are counted in the log's month resolution:
# the 90-day grace period is 3 months, the 180-day comparator horizon 6 months, the 30-day sensitivity 1 month.
GRACE, HORIZON, GRACE_SHORT = 3, 6, 1
CARRY_MONTHS = 13       # the annual scales are carried forward to the six-monthly assessments within this window
DT_HORIZONS = {"y3": (3, 2), "y6": (6, 2), "y12": (12, WINDOW), "y24": (24, WINDOW)}   # horizon -> (months, window); y3 descriptive
DT_CLASSES = {"da": "agonist", "maob": "maob", "ama": "amantadine", "comt": "comt"}
ADJUNCTS = ["da", "maob", "comt"]
STRATEGY = {1: "escalate", 0: "no increase", 2: "deviation"}
# outcomes of the decision-time table: the seven of the first implementation plus the pathway nodes of the new design
DT_OUTCOMES = {**OUTCOMES, "motor_on": "ON MDS-UPDRS III", "motor_na": "OFF non-axial MDS-UPDRS III",
               "fluct": "MDS-UPDRS IV wearing-off (4.3-4.4)", "dysk": "MDS-UPDRS IV dyskinesia (4.1-4.2)",
               "hall": "hallucinations (MDS-UPDRS 1.2)", "oh": "orthostatic systolic drop (mmHg)",
               "ldopa": "levodopa LEDD in force (mg/day)", "ledd_total": "total LEDD in force (mg/day)"}
VELOCITY = ["adl", "motor", "motor_na", "motor_on", "se", "fluct", "dysk"]    # pre-visit trajectory: h_d_<k>
VEL_GAP = (5, 18)       # months between the reference measurement and the latest one
STEP_ARMS = {0: "hold (no increase >= 50 mg within 6 months)", 1: "+50-100 mg", 2: "+150-200 mg", 3: "> +200 mg"}
TREATMENT_ARMS = {0: "hold (no levodopa increase, no adjunct or amantadine)", 1: "levodopa only (ESCALATE)", 2: "adjunct only (agonist, MAO-B or COMT inhibitor)"}


def level_before(rows, steps, datecol="date_t", strict=True):
    """Dose in force at each row's date from a step function: the level of the last change strictly before the
    date (`strict`; the visit's own month is a decision at the visit) or on/before it."""
    r = rows[["row_id", "PATNO", datecol]].sort_values(datecol)
    s = steps.rename(columns={"date": "_sdate"})[["PATNO", "_sdate", "level"]].sort_values("_sdate")
    m = pd.merge_asof(r, s, left_on=datecol, right_on="_sdate", by="PATNO", allow_exact_matches=not strict)
    return m.set_index("row_id").level.reindex(rows.row_id).fillna(0).to_numpy()


def comt_multiplier_before(rows, medlog):
    """Largest COMT-inhibitor levodopa multiplier ("LD x 0.33") in force strictly before each row's month (0 if none)."""
    ep = medlog[(medlog.drug_class == "comt") & medlog.ld_multiplier.notna()]
    m = rows[["row_id", "PATNO", "date_t"]].merge(ep[["PATNO", "ld_multiplier", "start", "stop"]], on="PATNO")
    m = m[(m.start < m.date_t) & (m.stop.isna() | (m.stop >= m.date_t))]
    return m.groupby("row_id").ld_multiplier.max().reindex(rows.row_id).fillna(0).to_numpy()


def first_reach(rows, steps, thr):
    """Days from the index visit to the first logged date at which the levodopa level reaches A0 + thr (the visit's
    own month counts as day 0), and the level reached; NaN when the dose never gets there."""
    m = rows[["row_id", "PATNO", "date_t", "A0"]].merge(steps, on="PATNO")
    m = m[(m.date >= m.date_t) & (m.level >= m.A0 + thr)].sort_values(["row_id", "date"]).drop_duplicates("row_id")
    m = m.set_index("row_id").reindex(rows.row_id)
    days = (m.date.to_numpy() - rows.date_t.to_numpy()) / np.timedelta64(1, "D")
    return days, m.level.to_numpy()


def dt_strategy(d, thr=100, grace=GRACE, horizon=HORIZON, isolated=False, confirm=False):
    """Arm code per row of a decision-time table: 1 = the dose reaches A0 + thr within the grace period (months),
    0 = it does not reach A0 + thr within the horizon, 2 = deviation (reached after the grace period but within the
    horizon; with `isolated`, also any new PD drug class or device therapy before the 12-month outcome, i.e. the
    'isolated levodopa change' sensitivity in which co-interventions are not permitted; with `confirm`, also a
    comparator whose 'no increase' status the log cannot confirm because the participant was not seen again within
    the horizon — a logged change appears only when the participant returns, so an unconfirmed comparator is censored
    for both arms instead of being counted as 'no increase', design_review.md §A)."""
    mo = d[f"months_reach_{thr}"].to_numpy(dtype=float)
    arm = np.where(mo <= grace, 1, np.where(mo <= horizon, 2, 0))      # NaN (never reached) compares False -> 0
    if isolated:
        co = (d.new_class_before_y12.fillna(False) | d.device_before_y12.fillna(False)).to_numpy(bool)
        arm = np.where(co, 2, arm)
    if confirm:
        unconfirmed = (arm == 0) & (d.followup_months.to_numpy(dtype=float) < horizon)
        arm = np.where(unconfirmed, 2, arm)
    return pd.Series(arm, index=d.index)


def dt_step_arm(d):
    """Step size of the escalation (DESIGN.md Pillar 3.3): 0 = hold (the dose does not reach A0 + 50 within 6 months),
    1 = +50-100 mg, 2 = +150-200 mg, 3 = more than +200 mg (the increase reaches A0 + 50 within the grace period and the
    dose in force at the end of the grace period is that far above A0); -1 = neither (an increase starting in months
    4-6, a step of 101-149 mg, or a level that fell back within the window) — censored for every arm."""
    mo = d.months_reach_50.to_numpy(dtype=float)
    dl = d.delta.to_numpy(dtype=float)
    early = mo <= GRACE
    hold = np.isnan(mo) | (mo > HORIZON)
    arm = np.full(len(d), -1)
    arm[hold] = 0
    arm[early & (dl >= 50) & (dl <= 100)] = 1
    arm[early & (dl >= 150) & (dl <= 200)] = 2
    arm[early & (dl > 200)] = 3
    return pd.Series(arm, index=d.index)


def dt_treatment_arm(d, thr=100):
    """Treatment strategy at the assessment (DESIGN.md §5, the secondary treatment-strategy family): 0 = hold (no
    increase to A0 + thr within 6 months, and no adjunct or amantadine started or increased in the grace period),
    1 = levodopa only (the escalation, with no adjunct or amantadine in the grace period), 2 = adjunct only (an
    agonist, MAO-B or COMT inhibitor started or increased in the grace period, no levodopa increase, no amantadine);
    -1 = none of them (levodopa deviations, amantadine, the combinations) — censored for every arm."""
    E = dt_strategy(d, thr=thr).to_numpy()
    adj = d.adj_up_grace.fillna(False).astype(bool).to_numpy()
    ama = d.ama_up_grace.fillna(False).astype(bool).to_numpy()
    arm = np.full(len(d), -1)
    arm[(E == 0) & ~adj & ~ama] = 0
    arm[(E == 1) & ~adj & ~ama] = 1
    arm[(E == 0) & adj & ~ama] = 2
    return pd.Series(arm, index=d.index)


def _index_visits(cohort, subj, visits, steps, population="escalation"):
    """Eligible index visits: any visit with MDS-UPDRS II, no device-aided therapy before the visit, one visit per
    participant-month (the most complete); escalation population = on levodopa by the log (dose in force before the
    visit's month > 0); initiation population = never on levodopa before the visit's month (no levodopa episode
    started before it)."""
    ids = cohort_ids(subj, cohort)
    v = visits[visits.PATNO.isin(ids) & visits.updrs2.notna() & visits.date.notna()].copy()
    v["n_obs"] = v[[C.NODES[k] for k in C.CORE_NODES]].notna().sum(axis=1)
    v = v.sort_values(["PATNO", "date", "n_obs"], kind="stable").drop_duplicates(["PATNO", "date"], keep="last")
    v = v.rename(columns={"date": "date_t"}).reset_index(drop=True)
    v["row_id"] = np.arange(len(v))
    v["A0"] = level_before(v, steps)
    dev = v.PATNO.map(subj.set_index("PATNO").device_date)
    v["device_before_visit"] = dev.notna() & (dev < v.date_t)
    if population == "initiation":
        first = steps[steps.chg > 0].groupby("PATNO").date.min()
        started = v.PATNO.map(first)
        keep = (started.isna() | (started >= v.date_t)) & (v.A0 == 0)
    else:
        keep = v.A0 > 0
    return v[keep & ~v.device_before_visit].drop(columns=["n_obs", "device_before_visit"]).reset_index(drop=True)


def latest_before_dated(rows, visits, col, datecol="date_t", tolerance_months=CARRY_MONTHS):
    """Latest observed value of a visit column at or before each row's `datecol` (within `tolerance_months`) and the
    date of that measurement; NaN / NaT when nothing was measured in the window."""
    r = rows[["row_id", "PATNO", datecol]].dropna(subset=[datecol]).sort_values(datecol)
    v = visits.loc[visits[col].notna(), ["PATNO", "date", col]].sort_values("date").rename(columns={"date": "_vdate"})
    m = pd.merge_asof(r, v, left_on=datecol, right_on="_vdate", by="PATNO", tolerance=pd.Timedelta(days=round(tolerance_months * C.MONTH)))
    m = m.set_index("row_id").reindex(rows.row_id)
    return m[col].to_numpy(), m["_vdate"].to_numpy()


def velocity(rows, visits, col, carry_months=CARRY_MONTHS, gap=VEL_GAP):
    """Pre-visit trajectory of one scale: the latest value at or before the visit (the history value) minus the latest
    value measured at least gap[0] and at most gap[1] months before it; returns (change, months between the two)."""
    cur, cur_date = latest_before_dated(rows, visits, col, "date_t", carry_months)
    r = rows[["row_id", "PATNO"]].copy()
    r["_cur"] = pd.to_datetime(cur_date) - pd.Timedelta(days=round(gap[0] * C.MONTH))
    ref, ref_date = latest_before_dated(r, visits, col, "_cur", gap[1] - gap[0])
    months = (pd.to_datetime(cur_date) - pd.to_datetime(ref_date)).days / C.MONTH
    return cur - ref, np.round(np.asarray(months, dtype=float), 1)


def months_since_change(rows, steps):
    """Months from the last change of the levodopa level strictly before the visit's month to the visit (the
    stable-dose run-in of design_review.md §C); NaN when the log has no earlier change."""
    r = rows[["row_id", "PATNO", "date_t"]].sort_values("date_t")
    s = steps.rename(columns={"date": "_sdate"})[["PATNO", "_sdate"]].sort_values("_sdate")
    m = pd.merge_asof(r, s, left_on="date_t", right_on="_sdate", by="PATNO", allow_exact_matches=False)
    m = m.set_index("row_id").reindex(rows.row_id)
    return np.round(((m.date_t - m._sdate).dt.days / C.MONTH).to_numpy(dtype=float), 1)


def _co_interventions(df, medlog):
    """First start of a PD drug class not in force before the index visit's month (restarts count), and device
    therapy: dates and months from the index visit."""
    ep = medlog[medlog.drug_class.isin(DT_CLASSES.values()) & medlog.start.notna()]
    m = df[["row_id", "PATNO", "date_t"]].merge(ep[["PATNO", "drug_class", "start", "stop"]], on="PATNO")
    before = m[(m.start < m.date_t) & (m.stop.isna() | (m.stop >= m.date_t))][["row_id", "drug_class"]].drop_duplicates()
    new = m[m.start >= m.date_t].merge(before.assign(_old=1), on=["row_id", "drug_class"], how="left")
    new = new[new._old.isna()].sort_values(["row_id", "start"]).drop_duplicates("row_id").set_index("row_id").reindex(df.row_id)
    out = pd.DataFrame(index=df.index)
    out["new_class_date"] = new.start.to_numpy()
    out["new_class_first"] = new.drug_class.to_numpy()
    out["new_class_months"] = ((out.new_class_date - df.date_t).dt.days / C.MONTH).round()
    out["device_months"] = ((df.device_date - df.date_t).dt.days / C.MONTH).round()
    return out


def latest_before(rows, visits, col, tolerance_months=CARRY_MONTHS):
    """Latest observed value of a visit column at or before each row's index date (within `tolerance_months`): the
    annual scales are carried forward to the six-monthly assessments, so the history is everything measured before
    the decision; NaN when nothing was measured in the window."""
    r = rows[["row_id", "PATNO", "date_t"]].sort_values("date_t")
    v = visits.loc[visits[col].notna(), ["PATNO", "date", col]].sort_values("date").rename(columns={"date": "_vdate"})
    m = pd.merge_asof(r, v, left_on="date_t", right_on="_vdate", by="PATNO", tolerance=pd.Timedelta(days=round(tolerance_months * C.MONTH)))
    return m.set_index("row_id")[col].reindex(rows.row_id).to_numpy()


def _dt_outcomes(df, visits, outcomes, window=WINDOW, horizons=DT_HORIZONS):
    """Outcomes at the visit nearest 3, 6, 12 and 24 months after the index visit (any visit at which the outcome was
    measured, +/- the horizon's window: 2 months at 3 and 6 months, `window` (4; 6 in the sensitivity table) at 12 and
    24 months; the MDS-UPDRS II visit dates the outcome) with the reason an outcome is missing; co-interventions before
    the outcome visit are flagged, not censored. MDS-UPDRS IV and its wearing-off and dyskinesia parts are 0 at a visit
    without PD treatment (as in the grid)."""
    vcols = {c: C.NODES[c] for c in outcomes}
    vv = visits.copy()
    for col in ("updrs4", "fluct", "dysk"):
        vv[col] = vv[col].where(vv.on_pd_meds == 1, vv[col].fillna(0))
    for tim, (months, w) in horizons.items():
        w = window if months >= 12 else w
        h = tim[1:]
        tgt = f"_target_{tim}"
        df[tgt] = df.date_t + pd.Timedelta(days=round(months * C.MONTH))
        any_visit = nearest_visit(df, vv, tgt, "date_t", [], w)
        new = {}
        for c, col in vcols.items():
            near = nearest_visit(df, vv[vv[col].notna()], tgt, "date_t", [col], w)
            new[f"{tim}_{c}"] = near[col].to_numpy()
            if c == "adl":
                new[f"date_{h}"] = near["date"].to_numpy()
        df = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)
        df[f"months_{h}"] = ((df[f"date_{h}"].to_numpy() - df.date_t.to_numpy()) / np.timedelta64(1, "D") / C.MONTH).round(1)
        end = df.date_t + pd.Timedelta(days=round((months + w) * C.MONTH))
        has = any_visit["date"].notna().to_numpy()
        df[f"censor{h}_reason"] = np.select(
            [df[f"{tim}_adl"].notna(), has, df.death_date.notna() & (df.death_date <= end), df.exit_date.notna() & (df.exit_date <= end),
             df.device_date.notna() & (df.device_date <= end), df.last_visit <= df.date_t + pd.Timedelta(days=round((months - w) * C.MONTH))],
            ["observed", "visit present, item missing", "death", "withdrawal", "device-aided therapy (no assessment in the window)",
             "no further visit (dropout or follow-up not yet reached)"], "visit outside the window")
        ref = df[f"date_{h}"].fillna(end)
        df[f"new_class_before_{tim}"] = df.new_class_date.notna() & (df.new_class_date <= ref)
        df[f"device_before_{tim}"] = df.device_date.notna() & (df.device_date <= ref)
        df = df.drop(columns=[tgt]).copy()
    return df


def decision_time_table(cohort, subj=None, grid=None, visits=None, medlog=None, window=WINDOW, carry_months=CARRY_MONTHS, population="escalation"):
    """One row per eligible clinical visit (time zero) with the strategy read from the dated medication log.
    Escalation population: 1 = a logged increase that brings the levodopa LEDD to >= A0 + 100 mg/day starting within
    the 90-day grace period, 0 = no such increase within 180 days, deviation = the increase starts in days 91-180.
    Initiation population (`population="initiation"`): never on levodopa before the visit; 1 = levodopa started within
    90 days, 0 = not started within 180 days, deviation = started in days 91-180. History columns h_* at the index
    visit (the class doses from the log in force before the visit's month; the annual scales carried forward within
    `carry_months`), the pre-visit trajectory h_d_<scale> (change since the latest measurement 5-18 months earlier),
    h_months_since_ldopa_change, A0 = log dose at the index date, A = dose in force at the end of the grace period,
    co-decisions in the grace window (adjunct / amantadine started or increased), the step-size arm E_step, outcomes
    y3_* / y6_* / y12_* / y24_* at the visit nearest 3 / 6 / 12 / 24 months after the visit. New drug classes and
    device therapy are flagged with their timing (permitted co-interventions in the primary analysis). `window` =
    months around the 12- and 24-month targets within which the outcome visit may fall (4; the sensitivity table uses
    6); `carry_months` = how far the history nodes are carried forward from an earlier visit (13; the sensitivity
    table uses 7)."""
    if subj is None:
        subj, grid, visits, medlog = load()
    steps = levodopa_steps(medlog)
    df = _index_visits(cohort, subj, visits, steps, population)
    sub = subj.set_index("PATNO")
    for c in ["death_date", "device_date", "exit_date", "last_visit", "enroll_date"]:
        df[c] = df.PATNO.map(sub[c])
    df["enroll_year"] = df.enroll_date.dt.year
    df["t"] = (df.months / 12).round(2)
    # history: the latest measurement of every clinical node at or before the index visit (the annual scales are
    # carried forward to six-monthly assessments, within 13 months); the class doses from the log in force before
    # the visit's month; months on levodopa and cumulative levodopa from the log at the visit date
    cv = visits[visits.PATNO.isin(cohort_ids(subj, cohort))]
    for k in HIST + EXTRA_HIST:
        if k in C.NODES and C.NODES[k] in df and k not in DT_CLASSES and k != "ldopa":
            df[f"h_{k}"] = latest_before(df, cv, C.NODES[k], carry_months)
            df[f"h_{k}_carried"] = df[f"h_{k}"].notna() & df[C.NODES[k]].isna()
    df = df.copy()      # de-fragment after the column-by-column history
    # the pre-visit trajectory (velocity) of the motor and functional scales
    vel = {}
    for k in VELOCITY:
        dv, gap = velocity(df, cv, C.NODES[k], carry_months)
        vel[f"h_d_{k}"] = dv
        if k == "adl":
            vel["h_d_months"] = gap
    df = pd.concat([df, pd.DataFrame(vel, index=df.index)], axis=1)
    df["h_months_since_ldopa_change"] = months_since_change(df, steps)
    df["h_ldopa"] = df.A0
    df["_grace_end"] = df.date_t + pd.Timedelta(days=round(GRACE * C.MONTH) + 2)
    for k, cls in DT_CLASSES.items():
        st = ledd_steps(medlog, cls)
        df[f"h_{k}"] = level_before(df, st)
        df[f"_g_{k}"] = level_before(df, st, "_grace_end", strict=False)
    df["adj_change_grace"] = df[[f"_g_{k}" for k in ADJUNCTS]].sum(axis=1) - df[[f"h_{k}" for k in ADJUNCTS]].sum(axis=1)
    df["ama_change_grace"] = df._g_ama - df.h_ama
    df = df.drop(columns=[f"_g_{k}" for k in DT_CLASSES])
    df["h_comt"] = df.h_comt + comt_multiplier_before(df, medlog) * df.A0     # entacapone / opicapone are logged as a levodopa multiplier
    df["h_months_on_ldopa_log"], df["h_cum_ldopa_log"] = df.months_on_ldopa_log, df.cum_ldopa_log
    # the decision read from the log (threshold 1 mg = any levodopa, the initiation strategy)
    for thr in (1,) + THRESHOLDS:
        days, level = first_reach(df, steps, thr)
        df[f"days_reach_{thr}"], df[f"months_reach_{thr}"] = days, np.round(days / C.MONTH)
        if thr == 100:
            df["A_reach"] = level
    df["A"] = level_before(df, steps, "_grace_end", strict=False)
    df["delta"] = df.A - df.A0
    df = df.drop(columns=["_grace_end"])
    thr = 1 if population == "initiation" else 100
    df["gap_days"] = df[f"days_reach_{thr}"]
    df = pd.concat([df, _co_interventions(df, medlog)], axis=1)
    early = df.new_class_months <= GRACE
    df["adj_up_grace"] = (df.adj_change_grace > 0) | (early & df.new_class_first.isin(["agonist", "maob", "comt"]))
    df["ama_up_grace"] = (df.ama_change_grace > 0) | (early & (df.new_class_first == "amantadine"))
    df = _dt_outcomes(df, visits, DT_OUTCOMES, window).copy()
    df["followup_months"] = ((df.last_visit - df.date_t).dt.days / C.MONTH).round(1)
    df["E"] = dt_strategy(df, thr=thr).to_numpy()
    df["strategy"] = df.E.map(STRATEGY)
    df["E_grace_30d"] = dt_strategy(df, thr=thr, grace=GRACE_SHORT).to_numpy()
    df["E_isolated"] = dt_strategy(df, thr=thr, isolated=True).to_numpy()
    df["E_confirmed"] = dt_strategy(df, thr=thr, confirm=True).to_numpy()
    df["E_step"] = dt_step_arm(df).to_numpy() if population == "escalation" else -1
    df = df.drop(columns=[c for c in C.BASELINE if c in df]).merge(_base(subj), left_on="PATNO", right_index=True, how="left")   # baseline age, not age at the visit
    df = df.merge(trait_score(grid[grid.PATNO.isin(cohort_ids(subj, cohort))]), left_on="PATNO", right_index=True, how="left")
    df["dur_t"] = (df.dur0 / 12 + df.t).round(2)      # disease duration at the visit (years)
    df["cohort"], df["population"] = cohort, population
    df = df.rename(columns={"EVENT_ID": "event_id"})
    carried = df[[c for c in df.columns if c.endswith("_carried")]]
    df["h_carried_share"] = carried.mean(axis=1).round(3)      # share of the history nodes carried forward from an earlier visit
    df = df.drop(columns=carried.columns)
    keep = ["PATNO", "cohort", "population", "event_id", "date_t", "t", "months", "strategy", "E", "E_grace_30d", "E_isolated", "E_confirmed", "E_step",
            "A0", "A", "A_reach", "delta", "gap_days", "adj_change_grace", "ama_change_grace", "adj_up_grace", "ama_up_grace"]
    keep += [c for c in df.columns if c.startswith(("h_", "days_reach_", "months_reach_", "y3_", "y6_", "y12_", "y24_", "new_class_", "device_", "censor"))]
    keep += ["date_3", "months_3", "date_6", "months_6", "date_12", "months_12", "date_24", "months_24", "followup_months", "dur_t"] + list(C.BASELINE)
    keep += ["trait0", "enroll_year", "death_date", "exit_date", "last_visit"]
    return df[list(dict.fromkeys(keep))].sort_values(["PATNO", "date_t"]).reset_index(drop=True)


def summarise_dt(df, cohort):
    """Counts of a decision-time table (aggregates): strategies under every definition, step arms, co-decisions,
    outcome availability per horizon and strategy, the pre-visit trajectory and the run-in."""
    e = df.E
    by = lambda mask: {STRATEGY[k]: int((mask & (e == k)).sum()) for k in (1, 0, 2)}
    both = df.groupby("PATNO").E.agg(lambda s: {0, 1} <= set(s)).sum()
    esc = df[e == 1]
    counts = dict(cohort=cohort, population=df.population.iloc[0] if len(df) else "", eligible_visits=len(df), participants=df.PATNO.nunique(),
                  visits_per_participant=round(len(df) / df.PATNO.nunique(), 2),
                  escalate=int((e == 1).sum()), no_increase=int((e == 0).sum()), deviation=int((e == 2).sum()),
                  participants_escalating=esc.PATNO.nunique(), participants_in_both_arms=int(both),
                  escalate_grace_30d=int((df.E_grace_30d == 1).sum()), deviation_grace_30d=int((df.E_grace_30d == 2).sum()),
                  escalate_ge50=int((df.months_reach_50 <= GRACE).sum()), escalate_ge150=int((df.months_reach_150 <= GRACE).sum()),
                  deviation_isolated=int((df.E_isolated == 2).sum()), escalate_isolated=int((df.E_isolated == 1).sum()), no_increase_isolated=int((df.E_isolated == 0).sum()),
                  no_increase_confirmed=int((df.E_confirmed == 0).sum()), no_increase_unconfirmed=int(((e == 0) & (df.E_confirmed == 2)).sum()),
                  stable_dose_runin_3m=int((df.h_months_since_ldopa_change >= GRACE).sum()),
                  **{f"step_{k}": int((df.E_step == k).sum()) for k in STEP_ARMS}, step_neither=int((df.E_step == -1).sum()),
                  adjunct_up_in_grace_escalate=int(esc.adj_up_grace.sum()), adjunct_up_in_grace_no_increase=int(df[e == 0].adj_up_grace.sum()),
                  amantadine_up_in_grace_escalate=int(esc.ama_up_grace.sum()),
                  gap_days_median_escalations=float(esc.gap_days.median()) if len(esc) else np.nan,
                  escalations_same_month_as_visit=int((esc.gap_days == 0).sum()),
                  delta_median_escalations=float(esc.delta.median()) if len(esc) else np.nan,
                  history_carried_forward_share=round(float(df.h_carried_share.mean()), 3),
                  h_d_adl_available_share=round(float(df.h_d_adl.notna().mean()), 3), h_d_months_median=float(df.h_d_months.median()),
                  h_cog_missing_share=round(float(df.h_cog.isna().mean()), 3), h_motor_missing_share=round(float(df.h_motor.isna().mean()), 3),
                  h_motor_on_missing_share=round(float(df.h_motor_on.isna().mean()), 3),
                  **{f"new_class_before_y12_{k}": v for k, v in by(df.new_class_before_y12).items()},
                  **{f"device_before_y12_{k}": v for k, v in by(df.device_before_y12).items()},
                  **{f"{tim}_adl_observed_{k}": v for tim in DT_HORIZONS for k, v in by(df[f"{tim}_adl"].notna()).items()},
                  **{f"followed_6_months_{k}": v for k, v in by(df.followup_months >= 6).items()},
                  **{f"months_to_{tim}_visit_mean": round(float(df[f"months_{tim[1:]}"].mean()), 2) for tim in DT_HORIZONS})
    cens = []
    for tim in DT_HORIZONS:
        c = df.groupby(["strategy", f"censor{tim[1:]}_reason"]).size().rename("rows").reset_index().rename(columns={f"censor{tim[1:]}_reason": "reason"})
        c["share_within_strategy"] = (c.rows / c.strategy.map(df.strategy.value_counts())).round(3)
        c.insert(0, "timing", tim); c.insert(0, "cohort", cohort); c.insert(0, "population", counts["population"])
        cens.append(c)
    return counts, pd.concat(cens, ignore_index=True)


def summarise(df, init, cohort):
    arm = np.where(df.delta >= 100, "esc", np.where(df.delta.abs() <= 50, "maint", "other"))
    both = df.assign(arm=arm).groupby("PATNO").arm.agg(lambda s: {"esc", "maint"} <= set(s)).sum()
    esc = df[df.delta >= 100]
    dated = esc.n_changes > 0
    counts = dict(cohort=cohort, eligible_person_visits=len(df), participants=df.PATNO.nunique(),
                  trials_per_participant=round(len(df) / df.PATNO.nunique(), 2),
                  escalations_ge100=int((df.delta >= 100).sum()), maintenance_pm50=int((df.delta.abs() <= 50).sum()),
                  escalations_ge50=int((df.delta >= 50).sum()), escalations_ge150=int((df.delta >= 150).sum()),
                  reductions_le_minus100=int((df.delta <= -100).sum()),
                  participants_in_both_arms=int(both), participants_with_arm=int(df.assign(arm=arm).query("arm != 'other'").PATNO.nunique()),
                  y2_adl_observed=int(df.y2_adl.notna().sum()), y12_adl_observed=int(df.y12_adl.notna().sum()),
                  escalations_with_log_date=int(dated.sum()), escalations_without_log_date=int((~dated).sum()),
                  months_on_new_dose_at_t1_median=round(float(esc.months_on_dose_t1.median()), 1),
                  months_on_new_dose_at_t1_q25=round(float(esc.months_on_dose_t1.quantile(0.25)), 1),
                  months_on_new_dose_at_t1_q75=round(float(esc.months_on_dose_t1.quantile(0.75)), 1),
                  escalations_ge6_months_on_dose_at_t1=int((esc.months_on_dose_t1 >= 6).sum()),
                  escalations_changed_in_more_than_one_step=int((esc.n_changes > 1).sum()),
                  initiation_rows=len(init), initiations=int(init.E.sum()), initiation_comparators=int((init.E == 0).sum()),
                  initiation_participants=init.PATNO.nunique())
    dating = esc.months_on_dose_t1.dropna()
    bins = pd.cut(dating, [-0.1, 2, 4, 6, 8, 10, 12, 17], labels=["0-2", "2-4", "4-6", "6-8", "8-10", "10-12", "12-16"])
    dating_tab = bins.value_counts().sort_index().rename_axis("months_on_new_dose_at_t1").reset_index(name="escalations")
    dating_tab.insert(0, "cohort", cohort)
    cens = df.censor2_reason.value_counts().rename_axis("reason").reset_index(name="rows")
    cens.insert(0, "cohort", cohort)
    cens["share"] = (cens.rows / len(df)).round(3)
    dd = df.assign(arm=arm, months_t_to_y12=(pd.to_datetime(df.date_12) - pd.to_datetime(df.date_t)).dt.days / C.MONTH,
                   months_t_to_y1=(pd.to_datetime(df.date_t1) - pd.to_datetime(df.date_t)).dt.days / C.MONTH)
    timing = dd.groupby("arm").agg(rows=("arm", "size"), months_visit_t_to_y1_visit=("months_t_to_y1", "mean"),
                                   months_visit_t_to_y12_visit=("months_t_to_y12", "mean"), months_on_dose_at_y12=("months_12_from_clock", "mean"),
                                   clock_delay_months=("clock_delay_months", "mean"), y12_adl_available=("y12_adl", lambda s: s.notna().mean())).round(2).reset_index()
    timing.insert(0, "cohort", cohort)
    return counts, dating_tab, cens, timing


def main(argv=None):
    C.require_ppmi_approval("trial emulation")  # escalation-discovery: PROTOCOL.md §11 gate
    args = argv or []
    subj, grid, visits, medlog = load()
    out = C.derived_dir()
    pd.set_option("display.width", 250)
    if "--legacy" in args:          # the visit-anchored tables of the first implementation (supplement only)
        counts, dating, cens, timing = [], [], [], []
        for cohort in C.COHORTS:
            df = build_trials(cohort, subj, grid, visits, medlog)
            init = build_initiation(cohort, subj, grid, visits, medlog)
            df.to_csv(out / f"trials_{cohort}.csv", index=False)
            init.to_csv(out / f"initiation_{cohort}.csv", index=False)
            a, b, c, tm = summarise(df, init, cohort)
            counts.append(a); dating.append(b); cens.append(c); timing.append(tm)
        pd.DataFrame(counts).to_csv(C.results_dir() / "trials_counts.csv", index=False)
        pd.concat(dating).to_csv(C.results_dir() / "trials_dating.csv", index=False)
        pd.concat(cens).to_csv(C.results_dir() / "trials_censoring.csv", index=False)
        pd.concat(timing).to_csv(C.results_dir() / "trials_y12_timing.csv", index=False)
        print(pd.DataFrame(counts).T.to_string())
    # the decision-time tables: escalation (the paper) and initiation (the positive control on the same machinery)
    for population, fname, cname in (("escalation", "trials_dt_{c}.csv", "trials_dt_counts.csv"), ("initiation", "trials_dt_init_{c}.csv", "trials_dt_init_counts.csv")):
        dcounts, dcens = [], []
        for cohort in C.COHORTS:
            dt = decision_time_table(cohort, subj, grid, visits, medlog, population=population)
            dt.to_csv(out / fname.format(c=cohort), index=False)
            a, c = summarise_dt(dt, cohort)
            dcounts.append(a); dcens.append(c)
        dcounts = pd.DataFrame(dcounts)
        dcounts.to_csv(C.results_dir() / cname, index=False)
        cens_name = "trials_dt_censoring.csv" if population == "escalation" else "trials_dt_init_censoring.csv"
        pd.concat(dcens, ignore_index=True).to_csv(C.results_dir() / cens_name, index=False)
        print(f"\ndecision-time table, {population} population (one trial per clinical assessment; strategy from the dated log):")
        print(dcounts.T.to_string())
        c12 = pd.concat(dcens)
        print(f"\nwhy the 12-month MDS-UPDRS II is missing, by strategy ({population}):")
        print(c12[c12.timing == "y12"].pivot_table(index=["reason"], columns=["cohort", "strategy"], values="rows", fill_value=0).astype(int).to_string())


if __name__ == "__main__":
    main()

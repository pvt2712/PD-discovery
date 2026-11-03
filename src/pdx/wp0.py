"""WP0 feasibility audit: aggregate counts only (protocol v0.5, §3, §4, §5.4, §6.3).

The audit reads one analysis-ready visit table, already built on the annual grid, and returns
aggregate tables for the clinical panel. It never returns, prints or writes a participant-level
row. Disclosure control (threshold `min_cell`, 5 by default; confirm against PPMI's guidance):
- counts from 1 to min_cell - 1 are suppressed, and within every table a second cell is
  suppressed wherever a single suppressed cell could be recovered from the table's total or a
  row or column margin (complementary suppression);
- shares are suppressed when their denominator is below min_cell;
- site sizes are reported in bands, and quantiles only from at least 50 values.
It runs on PPMI data only with the investigator's written approval (§14); scripts/run_wp0_audit.py
enforces that.

Input schema (one row per participant and analysed visit; column names configurable):
  participant   identifier (never output)
  cohort        "PPMI 1.0" / "PPMI 2.0"
  year          visit year on the annual grid (0 = baseline)
  date_years    visit date in years since enrolment (actual date, for intervals)
  phenotype     "TD", "PIGD" or "indeterminate" at year 0 (repeated on every row)
  td_pigd_ratio continuous tremor/PIGD ratio at year 0 (for the overlap diagnostic)
  site, mode    site identifier and visit mode ("in person", "remote")
  treated       1 if on dopaminergic treatment at the visit
  dose_change   1 if the medication changed within the interval before the visit
  motor_state   "OFF", "ON" or missing
  domain columns (the progression domains), NaN when missing
  phenotype_y1, phenotype_y2   optional reassessments for the stability table
  event_24m     optional Part II endpoint indicator, on the year-0 row
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Stebbins 2013 cut-offs, for the overlap diagnostic
TD_CUT, PIGD_CUT = 1.15, 0.90
QUANTILE_MIN_N = 50
SITE_BANDS = ((1, 4), (5, 19), (20, 49), (50, None))


@dataclass
class Columns:
    participant: str = "participant"
    cohort: str = "cohort"
    year: str = "year"
    date_years: str = "date_years"
    phenotype: str = "phenotype"
    ratio: str = "td_pigd_ratio"
    site: str = "site"
    mode: str = "mode"
    treated: str = "treated"
    dose_change: str = "dose_change"
    motor_state: str = "motor_state"
    domains: list = field(default_factory=list)
    phenotype_y1: str = "phenotype_y1"
    phenotype_y2: str = "phenotype_y2"
    event: str = "event_24m"


# ---------------------------------------------------------------- disclosure control

def _small(v: int, k: int) -> bool:
    return 0 < v < k


def protect_1way(counts: dict, k: int) -> dict:
    """Primary suppression of small cells; if exactly one cell is suppressed, the smallest other
    non-zero cell is suppressed too, so the total cannot give it back."""
    keys = list(counts)
    supp = {key for key in keys if _small(counts[key], k)}
    if len(supp) == 1:
        rest = sorted((counts[key], key) for key in keys if key not in supp and counts[key] > 0)
        if rest:
            supp.add(rest[0][1])
    return {str(key): (f"<{k}" if key in supp and _small(counts[key], k) else
                       "suppressed" if key in supp else int(counts[key])) for key in keys}


def protect_2way(table: pd.DataFrame, k: int) -> dict:
    """A cross-table published with its own margins. Primary suppression, then complementary
    suppression until no row or column has exactly one suppressed cell among its non-zero cells;
    a margin is shown only if none of its cells is suppressed, and the grand total always. So the
    only constraint on the suppressed cells is the grand total, which binds at least two of them.
    Publish the margins from here rather than as separate tables."""
    T = table.to_numpy(int)
    S = (T > 0) & (T < k)
    changed = True
    while changed:
        changed = False
        for M, Sx in ((T, S), (T.T, S.T)):
            for r in range(M.shape[0]):
                if Sx[r].sum() == 1:
                    cand = [(M[r, c], c) for c in range(M.shape[1]) if not Sx[r, c] and M[r, c] > 0]
                    if cand:
                        Sx[r, min(cand)[1]] = True
                        changed = True
    if S.sum() == 1:                                       # one suppressed cell, all else zero or shown
        cand = [(T[i, j], i, j) for i in range(T.shape[0]) for j in range(T.shape[1]) if not S[i, j] and T[i, j] > 0]
        if cand:
            _, i, j = min(cand)
            S[i, j] = True
    cell = lambda v, sup: (f"<{k}" if sup and _small(v, k) else "suppressed" if sup else int(v))
    cells = {str(ri): {str(cj): cell(T[i, j], S[i, j]) for j, cj in enumerate(table.columns)}
             for i, ri in enumerate(table.index)}
    rows = {str(ri): (int(T[i].sum()) if not S[i].any() else "suppressed") for i, ri in enumerate(table.index)}
    cols = {str(cj): (int(T[:, j].sum()) if not S[:, j].any() else "suppressed") for j, cj in enumerate(table.columns)}
    return {"cells": cells, "row_totals": rows, "column_totals": cols, "total": int(T.sum())}


def _share(num: float, den: int, k: int):
    return float(num / den) if den >= k else f"denominator <{k}"


def _quantiles(x: pd.Series, qs=(0.1, 0.25, 0.5, 0.75, 0.9)):
    x = x.replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < QUANTILE_MIN_N:
        return f"fewer than {QUANTILE_MIN_N} values"
    return {str(q): float(x.quantile(q)) for q in qs}


def _count_series(s: pd.Series) -> dict:
    vc = s.value_counts(dropna=False).sort_index()
    return {("missing" if pd.isna(k) else k): int(v) for k, v in vc.items()}


# ---------------------------------------------------------------- the audit

def audit(df: pd.DataFrame, cols: Columns, min_cell: int = 5, band: float = 0.10) -> dict:
    c, k = cols, min_cell
    out: dict = {"min_cell": k}
    base = df[df[c.year] == 0].drop_duplicates(c.participant)
    # cohort x phenotype at baseline; its margins are the counts by cohort and by phenotype
    out["participant_flow"] = protect_2way(pd.crosstab(base[c.cohort], base[c.phenotype]), k)
    y1 = set(df.loc[df[c.year] == 1, c.participant])
    reached = base[c.participant].isin(y1).map({True: "reached year 1", False: "not reached"})
    # phenotype x landmark; its column totals are the overall counts reaching the landmark
    out["landmark"] = protect_2way(pd.crosstab(base[c.phenotype], reached), k)
    post = df[df[c.year] >= 1].sort_values([c.participant, c.year])
    step = post.groupby(c.participant)[c.year].diff()
    gap = post.groupby(c.participant)[c.date_years].diff()
    consec = post.assign(_gap=gap)[step == 1]
    out["transitions_after_landmark"] = {
        "total": int(len(consec)),
        "by_phenotype": protect_1way(consec.groupby(c.phenotype).size().to_dict(), k),
        "by_cohort": protect_1way(consec.groupby(c.cohort).size().to_dict(), k),
        "across_a_missed_year": int((step > 1).sum()),
    }
    g = consec["_gap"]
    out["intervals"] = {
        "share_within_12_pm_3_months": _share(((g >= 0.75) & (g <= 1.25)).sum(), len(g), k),
        "quantiles_years": _quantiles(g),
        "share_with_dose_change": (_share(consec[c.dose_change].sum(), len(consec), k)
                                   if c.dose_change in consec else None),
    }
    share = out["intervals"]["share_within_12_pm_3_months"]
    out["intervals"]["step1_annual_tables_ok"] = bool(share >= 0.90) if isinstance(share, float) else None
    vpp = post.groupby(c.participant).size()
    out["visits_per_participant"] = protect_1way(vpp.value_counts().sort_index().to_dict(), k)
    for name in (c.mode, c.motor_state):
        if name in df:
            out[name] = protect_1way(_count_series(df[name]), k)
    miss = {}
    for dname in c.domains:
        entry = {"overall": _share(df[dname].isna().sum(), len(df), k), "by_phenotype": {}}
        for key, grp in df.groupby(c.phenotype):
            entry["by_phenotype"][str(key)] = _share(grp[dname].isna().sum(), len(grp), k)
        if c.mode in df:
            entry["by_mode"] = {str(key): _share(grp[dname].isna().sum(), len(grp), k)
                                for key, grp in df.groupby(c.mode)}
        miss[dname] = entry
    out["missing_share"] = miss
    if c.site in df:
        sizes = base.groupby(c.site).size()
        out["sites"] = {"n_sites": int(len(sizes)), "sites_by_participants": {
            (f"{lo}-{hi}" if hi else f"{lo}+"): int(((sizes >= lo) & ((sizes <= hi) if hi else True)).sum())
            for lo, hi in SITE_BANDS}}
    if c.ratio in base:
        r = base[c.ratio].replace([np.inf, -np.inf], np.nan).dropna()
        near = int((((r - TD_CUT).abs() <= band) | ((r - PIGD_CUT).abs() <= band)).sum())
        out["overlap"] = {"near_cutoffs": f"<{k}" if _small(near, k) else near, "band": band,
                          "ratio_quantiles": _quantiles(r)}
    stab = {}
    for col in (c.phenotype_y1, c.phenotype_y2):
        if col in base:
            stab[col] = protect_2way(pd.crosstab(base[c.phenotype], base[col]), k)
    if stab:
        out["phenotype_stability"] = stab
    if c.event in base:
        ev = int(base[c.event].sum())
        out["part2_events"] = {"events": f"<{k}" if _small(ev, k) else ev, "at_risk": int(base[c.event].notna().sum())}
    return out

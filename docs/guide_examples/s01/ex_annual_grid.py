"""Section 1.25: annual_grid, one visit per participant and year. A hand-made visit table of made-up
participants."""
import numpy as np
import pandas as pd
from _setup import TMP  # noqa: F401  (import paths)

from pdx import ppmi


def visit(p, ev, date, state="untreated", n_dom=12):
    r = {"PATNO": p, "EVENT_ID": ev, "date": pd.Timestamp(date), "remote": ev.startswith("R"), "motor_state": state}
    r.update({dn: (1.0 if j < n_dom else np.nan) for j, dn in enumerate(ppmi.DOMAINS)})
    return r


v = pd.DataFrame([
    visit(1, "BL", "2015-01-15"), visit(1, "V03", "2015-10-15"), visit(1, "R04", "2016-01-15"),   # remote vs 9 months
    visit(2, "BL", "2015-01-15"), visit(2, "V04", "2016-02-15", "ON only"), visit(2, "U02", "2016-01-15", "OFF"),
    visit(2, "V06", "2017-05-15"),                                                                # 4 months late
    visit(3, "BL", "2015-06-15"), visit(3, "V06", "2017-10-15"),                                  # 4 months late
    visit(4, "SC", "2014-12-15"), visit(4, "V01", "2015-04-15"), visit(4, "V04", "2016-01-15"),   # no BL
    visit(5, "BL", "2015-01-15"), visit(5, "V04", "2016-01-15", "ON only"),
    visit(5, "V05", "2016-01-15", "untreated", n_dom=6), visit(5, "V06", "2016-01-15", "untreated", n_dom=10),
])
anchor = pd.Series(pd.to_datetime(["2015-01-15", "2015-01-15", "2015-06-15", "2015-01-15", "2015-01-15"]),
                   index=[1, 2, 3, 4, 5])
cand = v.assign(months=(v.date - v.PATNO.map(anchor)).dt.days / 30.4375)
cand["year"] = np.round(cand.months / 12)
cand["off_by"] = (cand.months - 12 * cand.year).abs()
print("candidates (before the filters):")
print(cand[["PATNO", "EVENT_ID", "date", "remote", "motor_state", "months", "year", "off_by"]].round({"months": 3, "off_by": 3}).to_string(index=False))
g = ppmi.annual_grid(v, anchor)
g["n_domains"] = g[ppmi.DOMAINS].notna().sum(axis=1)
print("\nthe grid:")
print(g[["PATNO", "year", "EVENT_ID", "date", "date_years", "remote", "motor_state", "n_domains"]].to_string(index=False))

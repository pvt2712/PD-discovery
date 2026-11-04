"""Sections 1.26-1.28: build and flow_table on the repository's mock download (13 made-up
participants; see tests/ppmi_mock.py for what each one exercises)."""
from _setup import mock_download

from pdx import ppmi

out = ppmi.build(mock_download())
part = out["participants"]
print("== 1.26 population and exclusions ==")
print(part[["PATNO", "era", "enrolled", "genetic", "has_baseline", "treated_bl", "phenotype", "td_pigd_ratio",
            "reached_y1", "exclusion", "primary", "clinical_subtype"]].to_string(index=False))

print("\n== 1.27 the WP0 visit table ==")
w = out["wp0"]
print("columns:", list(w.columns))
print(w[w.participant.isin(["1", "5", "12"])][["participant", "cohort", "year", "date_years", "phenotype", "mode",
                                               "treated", "dose_change", "motor_state", "event_24m", "phenotype_y1",
                                               "phenotype_y2"]].to_string(index=False))
g = out["grid"]
print("\nparticipant 10 (treated at baseline) on the grid:")
print(g[g.PATNO == 10][["year", "EVENT_ID", "motor_state", "treated", "ledd", "classes"]].to_string(index=False))

print("\n== 1.28 flow_table ==")
print(ppmi.flow_table(part).to_string())

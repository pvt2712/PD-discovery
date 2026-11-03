"""The PPMI pipeline on a mock download (made-up participants; no PPMI data)."""
import json
import tempfile
from pathlib import Path

import numpy as np

import ppmi_mock
from pdx import ppmi, wp0


def _build():
    tmp = Path(tempfile.mkdtemp())
    ppmi_mock.make(tmp)
    return ppmi.build(tmp)


def test_population_phenotypes_and_grid():
    out = _build()
    part = out["participants"].set_index("PATNO")
    assert part.loc[13, "exclusion"].startswith("not enrolled")
    assert part.loc[9, "exclusion"] == "not sporadic (genetic enrolment)"
    assert part.loc[11, "exclusion"] == "no baseline visit"
    assert part.loc[10, "exclusion"] == "treated at baseline"
    assert part.primary.sum() == 9
    assert part.loc[1, "phenotype"] == "TD" and part.loc[2, "phenotype"] == "PIGD"
    g = out["grid"]
    row = g[(g.PATNO == 3) & (g.year == 1)].iloc[0]
    assert row.EVENT_ID == "V04"                                   # in person before remote
    w = out["wp0"]
    p12 = w[w.participant == "12"].set_index("year")
    assert p12.loc[1, "treated"] == 1 and p12.loc[1, "dose_change"] == True
    assert p12.loc[0, "cohort"] == "PPMI 2.0"
    p5 = w[w.participant == "5"].set_index("year")
    assert p5.loc[0, "event_24m"] == 1.0                           # died within 24 months
    assert set(ppmi.DOMAINS) <= set(w.columns)
    assert np.allclose(w.loc[w.year == 0, "independence"], 10.0)   # 100 - S&E 90


def test_flow_and_audit_run_on_the_tables():
    out = _build()
    flow = ppmi.flow_table(out["participants"])
    assert flow.to_numpy().sum() == 13
    rep = wp0.audit(out["wp0"], wp0.Columns(domains=ppmi.DOMAINS), min_cell=5)
    text = json.dumps(rep, default=str)
    assert '"participant"' not in text
    assert rep["participant_flow"]["total"] == 9


def test_analysis_panel_from_the_mock():
    out = _build()
    panel, ns, info = ppmi.analysis_panel(out["participants"], out["grid"])
    assert set(np.unique(panel.group)) <= {0, 1}
    assert panel.d == len(ppmi.CORE_DOMAINS) and panel.year[0][0] == 1
    assert all(np.all(np.diff(t) > 0) for t in panel.t)
    assert abs(np.nanmean(np.vstack(panel.y))) < 1.0                  # normal-scores scale
    assert info["counts"]["0"]["participants"] + info["counts"]["1"]["participants"] == panel.n

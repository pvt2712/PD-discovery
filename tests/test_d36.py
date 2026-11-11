"""Decision D36 (11 November 2026): H-P's two groups by age at onset, sex or probable RBD, beside TD/PIGD.
Synthetic PPMI mock only."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import ppmi_mock

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def _mock():
    from pdx import ppmi
    tmp = Path(tempfile.mkdtemp())
    ppmi_mock.make(tmp)
    out = ppmi.build(tmp)
    return out["participants"], out["grid"]


def test_each_grouping_follows_its_rule_and_leaves_the_speed_covariates():
    from pdx import ppmi
    part, grid = _mock()
    four = ["tremor", "brady_rigidity", "axial", "function"]
    base, _, base_info = ppmi.analysis_panel(part, grid, four)
    assert base_info["grouping"]["by"] == "phenotype" and base_info["grouping"]["group_1"] == "PIGD"
    assert base_info["covariates"] == ["era (PPMI 2.0)", "age at onset", "male", "duration"] and base.p == 4
    p = part.set_index("PATNO")
    rbd0 = grid[grid.year == 0].drop_duplicates("PATNO").set_index("PATNO").rbd
    for by, dropped in (("age_onset", "age at onset"), ("sex", "male"), ("rbd", None)):
        panel, _, info = ppmi.analysis_panel(part, grid, four, group_by=by)
        gr = info["grouping"]
        assert gr["by"] == by and gr["participants"] == [int((panel.group == 0).sum()), int((panel.group == 1).sum())]
        ids = panel.ids
        if by == "age_onset":
            assert gr["cut"] == float(np.median(p.loc[ids, "age_onset"]))
            assert np.array_equal(panel.group, (p.loc[ids, "age_onset"] >= gr["cut"]).astype(int).to_numpy())
        elif by == "sex":
            assert np.array_equal(panel.group, (p.loc[ids, "male"].astype(float) == 0).astype(int).to_numpy())
        else:
            assert gr["cut"] == 6.0 and np.array_equal(panel.group, (rbd0.reindex(ids).to_numpy(float) >= 6).astype(int))
        assert (dropped not in info["covariates"]) if dropped else info["covariates"] == base_info["covariates"]
        assert panel.p == len(info["covariates"]) and info["covariates"][0] == "era (PPMI 2.0)"   # era stays column 0
        # any motor phenotype enters, not only TD and PIGD
        assert set(p.loc[ids, "phenotype"].dropna()) <= {"TD", "PIGD", "indeterminate"}
        assert len(ids) >= len(base.ids) or by == "rbd"


def test_the_age_cut_can_be_fixed_and_the_rbd_cut_changed():
    from pdx import ppmi
    part, grid = _mock()
    panel, _, info = ppmi.analysis_panel(part, grid, ["tremor", "function"], group_by="age_onset", age_cut=62.0)
    assert info["grouping"]["cut"] == 62.0 and info["grouping"]["group_1"] == "onset at 62.0 or later"
    _, _, info5 = ppmi.analysis_panel(part, grid, ["tremor", "function"], group_by="rbd", rbd_cut=5)
    assert info5["grouping"]["cut"] == 5.0


def test_a_grouping_other_than_phenotype_needs_a_tag():
    sys.path.insert(0, str(SCRIPTS))
    import build_analysis_panel
    old = sys.argv
    try:
        sys.argv = ["build_analysis_panel.py", "--group-by", "sex", "--approval", "test"]
        try:
            build_analysis_panel.main()
            raise AssertionError("a grouping without --tag was accepted")
        except SystemExit as e:
            assert e.code == 2
    finally:
        sys.argv = old

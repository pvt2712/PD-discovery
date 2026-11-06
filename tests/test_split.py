"""The keyed-hash split shared with Paper 2 (escalation-study-v2/PROTOCOL.md §3.2; D33), on synthetic identifiers."""
import numpy as np

from pdx import split

# the same vectors are tested in escalation-study-v2/tests/test_split_guard.py: both copies of the rule must agree
VECTORS = {3000: "training", 3001: "training", 3004: "heldout", 3006: "heldout", 3010: "heldout", 3011: "validation",
           4100: "validation", 50001: "training"}


def test_split_vectors_match_paper2():
    assert {p: split.set_of(p) for p in VECTORS} == VECTORS
    assert round(split.u_value(3011), 6) == 0.746418
    assert {split.set_of(f) for f in (3011, 3011.0, "3011", " 3011 ", np.int64(3011))} == {"validation"}


def test_split_shares_stability_and_named_groups():
    ids = np.arange(3000, 23000)
    s = split.assign(ids)
    shares = s.value_counts(normalize=True)
    assert abs(shares["training"] - 0.6) < 0.015 and abs(shares["heldout"] - 0.2) < 0.015
    assert split.assign(ids[:1000]).tolist() == s[:1000].tolist()
    dev = split.in_sets(ids[:200], "development")
    assert dev.tolist() == [split.set_of(p) != "heldout" for p in ids[:200]]
    assert split.in_sets(ids[:200], "all").all()
    assert split.counts([3000, 3001, 3004, 3011, 3011]) == {"training": 2, "validation": 1, "heldout": 1}
    assert split.assignment_sha256([3011, 3000]) == split.assignment_sha256([3000.0, "3011", 3000])


def test_the_panel_build_keeps_only_the_development_participants():
    """build_analysis_panel.py --set development on the mock download: the held-out participants (2 and 5 of the
    mock's 1-12) never enter the panel, its normal-scores tables or its error variances (D33)."""
    import json
    import sys
    import tempfile
    from pathlib import Path

    import ppmi_mock
    from pdx import ppmi
    from pdx.data import load_panel
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import _common
    import build_analysis_panel
    tmp = Path(tempfile.mkdtemp())
    ppmi_mock.make(tmp / "raw")
    out = ppmi.build(tmp / "raw")
    der = tmp / "data" / "derived"
    der.mkdir(parents=True)
    out["participants"].to_parquet(der / "participants.parquet")
    out["grid"].to_parquet(der / "grid.parquet")
    old, argv = _common.DATA, sys.argv
    _common.DATA = tmp / "data"
    try:
        for which in ("development", "all"):
            sys.argv = ["build_analysis_panel.py", "--approval", "synthetic mock data, unit test", "--set", which]
            build_analysis_panel.main()
    finally:
        _common.DATA, sys.argv = old, argv
    dev, full = load_panel(der / "panel_tdpigd_development.npz"), load_panel(der / "panel_tdpigd.npz")
    spec = json.loads((der / "panel_spec_development.json").read_text())
    assert spec["split"]["participant_set"] == "development" and spec["split"]["sets"] == ["training", "validation"]
    held = {p for p in full.ids if split.set_of(p) == "heldout"}
    assert held, "the mock should hold a held-out participant in the whole-cohort panel"
    assert not set(dev.ids) & held
    assert set(dev.ids) == set(full.ids) - held


def test_the_hp_run_refuses_a_panel_built_for_another_participant_set():
    import os
    import sys
    import tempfile
    from pathlib import Path

    import ppmi_mock
    from pdx import ppmi
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import _common
    import build_analysis_panel
    import run_hp_real
    tmp = Path(tempfile.mkdtemp())
    ppmi_mock.make(tmp / "raw")
    out = ppmi.build(tmp / "raw")
    der = tmp / "data" / "derived"
    der.mkdir(parents=True)
    out["participants"].to_parquet(der / "participants.parquet")
    out["grid"].to_parquet(der / "grid.parquet")
    old = (_common.DATA, _common.RESULTS, sys.argv, os.environ.get("PYTHONHASHSEED"))
    _common.DATA, _common.RESULTS = tmp / "data", tmp / "results"
    os.environ["PYTHONHASHSEED"] = "0"
    try:
        sys.argv = ["build_analysis_panel.py", "--approval", "synthetic mock data, unit test"]      # the default set
        build_analysis_panel.main()
        assert (der / "panel_tdpigd_development.npz").exists() and not (der / "panel_tdpigd.npz").exists()
        sys.argv = ["run_hp_real.py", "--expert", str(_common.ROOT / "configs" / "expert_map_provisional.json"),
                    "--approval", "synthetic mock data, unit test", "--set", "all",
                    "--panel", str(der / "panel_tdpigd_development.npz"), "--spec", str(der / "panel_spec_development.json"),
                    "--out", str(tmp / "results" / "wp4")]
        try:
            run_hp_real.main()
            raise AssertionError("a panel built for another participant set was accepted")
        except SystemExit as e:
            assert "built for the participant set 'development'" in str(e)
    finally:
        _common.DATA, _common.RESULTS, sys.argv = old[:3]
        if old[3] is None:
            os.environ.pop("PYTHONHASHSEED", None)
        else:
            os.environ["PYTHONHASHSEED"] = old[3]

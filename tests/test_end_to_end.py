"""Both arms end to end on synthetic data, small sizes (outputs go to a temporary folder)."""
import json

import pandas as pd
import pytest

from escalation_discovery import arm_a, arm_b, handover
from escalation_discovery import expert as E


@pytest.mark.slow
def test_arm_a_job_end_to_end(tmp_outputs):
    s = arm_a.Setting("G1-test", "G1", n=150, realistic=True)
    out = arm_a.source_dir("synthetic")
    part = arm_a._job(s, 0, "synthetic", ["pc_fz", "fci_fz", "notears", "dagma"], {}, out)
    ens = arm_a._ensemble_job(s, 0, "synthetic", out)
    df = pd.concat([pd.read_csv(part), pd.read_csv(ens)], ignore_index=True)
    assert set(df.method) >= {"pc_fz", "fci_fz", "notears", "dagma", "ensemble"}
    ok = df[df.error.isna() | (df.error == "")]
    assert len(ok) == len(df)
    for c in ("adj_f1", "ah_precision", "pa_mean_f1"):
        vals = ok[c].dropna()
        assert ((vals >= 0) & (vals <= 1)).all()
    assert (ok.sid_lower <= ok.sid_upper).all()
    table = arm_a.decision_table(df.assign(realistic=True))
    assert "fci_fz" in set(table.method) or "pc_fz" in set(table.method)


@pytest.mark.slow
def test_deep_learning_learners_run_in_r_only(tmp_outputs):
    s = arm_a.Setting("G1-test-diag", "G1", n=150)
    part = arm_a._job(s, 0, "synthetic", ["sdcd", "sortnregress"], {}, arm_a.source_dir("synthetic"))
    assert set(pd.read_csv(part).method) == {"sortnregress"}


@pytest.mark.slow
def test_arm_b_end_to_end(tmp_outputs):
    meta = arm_b.run("synthetic", B=3, n_perm=2, method_names=["pc_fz", "notears"], n_jobs=2, panel=_panel())
    out = tmp_outputs / "results" / "arm_b" / "synthetic"
    for f in ("expert_falsification.json", "expert_edge_support.csv", "stability.csv", "negative_controls.csv",
              "agreement.csv", "clinician_sheet.csv", "clinician_sheet_key.csv", "meta.json"):
        assert (out / f).exists(), f
    fals = json.loads((out / "expert_falsification.json").read_text())
    assert fals["statements"] > 0 and 0 <= fals["violation_rate"] <= 1
    nc = pd.read_csv(out / "negative_controls.csv")
    assert set(nc.method) == {"pc_fz_noknow", "notears", "ensemble_noknow"}
    stab = pd.read_csv(out / "stability.csv")
    assert {"pc_fz", "notears", "ensemble"} <= set(stab.method)
    doc = handover.build("synthetic")
    assert doc["selected_learner"] == meta["selected"]


@pytest.mark.slow
def test_arm_b_sensitivity_run_with_one_learner(tmp_outputs):
    meta = arm_b.run("synthetic", B=2, n_perm=2, method_names=["pc_fz"], n_jobs=2, panel=_panel(),
                     subset_name="no_genetic")
    out = tmp_outputs / "results" / "arm_b" / "synthetic_no_genetic"
    assert meta["subset"] == "no_genetic" and meta["dropped"] == ["genetic"]
    given = pd.read_csv(out / "expert_statements.csv").given.fillna("")
    assert not given.str.contains("genetic").any()                               # the fixed node left the DAG too
    stab = pd.read_csv(out / "stability.csv")
    assert set(stab.method) == {"pc_fz"} and (stab.failed_fits == 0).all()     # no ensemble of one member
    assert set(pd.read_csv(out / "negative_controls.csv").method) == {"pc_fz_noknow"}


def _panel():
    from escalation_discovery import generator as G, missingness as M
    spec = E.load()
    long, base = G.simulate(G.synthetic("G1"), 160, 31)
    return M.PatternBank.synthetic(spec).apply(long, base, "mar", 32)

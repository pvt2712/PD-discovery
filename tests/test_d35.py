"""Decision D35 (10 November 2026): the proof of concept on four domains with a fixed map, and its
simulation. No PPMI data."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def test_the_poc_map_is_the_provisional_maps_links_between_the_four_domains():
    poc = json.loads((ROOT / "configs" / "expert_map_poc_d4.json").read_text(encoding="utf-8"))
    full = json.loads((ROOT / "configs" / "expert_map_provisional.json").read_text(encoding="utf-8"))
    four = ["tremor", "brady_rigidity", "axial", "function"]
    assert poc["domains"] == four
    assert poc["links"] == [l for l in full["links"] if l[0] in four and l[1] in four]
    assert sorted(map(tuple, poc["links"] + poc["set_aside_from_the_provisional_map"])) == sorted(map(tuple, full["links"]))
    sys.path.insert(0, str(SCRIPTS))
    import run_hp_real
    sup, aside = run_hp_real.expert_support(str(ROOT / "configs" / "expert_map_poc_d4.json"), four)
    assert int(sup.sum()) == 3 and aside == []
    assert sup[3, :3].all()                       # every motor domain feeds function


def test_a_fixed_map_is_part_of_the_checkpoint_settings():
    sys.path.insert(0, str(SCRIPTS))
    import run_hp_real
    from pdx import config
    from pdx.step1 import Step1Config
    cfg = config.load()
    p = ROOT / "configs" / "expert_map_poc_d4.json"
    a = argparse.Namespace(set="development", seed=1, expert=str(p), fixed_map=True)
    fixed = run_hp_real.run_settings(a, [p, p], cfg, Step1Config())
    a.fixed_map = False
    selected = run_hp_real.run_settings(a, [p, p], cfg, Step1Config())
    assert fixed["step1"] is None and isinstance(selected["step1"], dict) and fixed != selected


def test_the_poc_simulation_mirrors_the_ppmi_analysis():
    sys.path.insert(0, str(SCRIPTS))
    import run_poc_simulation as P
    from pdx import config
    cfg = config.load()
    tasks = P.tasks(3, 2)
    assert [t["scenario"] for t in tasks] == ["S1"] * 3 + ["S2"] * 2
    for t in tasks:
        assert t["d"] == 4 and t["step1"] is False and t["n_quad"] == cfg["engine"]["n_quad"]
        assert t["load_on_speed"] == cfg["engine"]["load_on_speed"] and t["fit_kw"] == cfg["fit"]
        assert sum(t["sim_kw"]["target_counts"]) == P.N_DEVELOPMENT and len(t["sim_kw"]["p_missing"]) == 4
    assert all(t["replicate"] for t in tasks[:3]) and not any(t["replicate"] for t in tasks[3:])
    assert len({t["seed"] for t in tasks}) == 5
    rows = [{"scenario": "S1", "lr": 20.0, "lr_star": 5.0, "df": 7, "converged": True, "true_speed_ratio": 2.0,
             "sp1": 2.1, "seconds": 60.0},
            {"scenario": "S1", "lr": 3.0, "lr_star": 6.0, "df": 7, "converged": True, "true_speed_ratio": 2.0,
             "sp1": 1.9, "seconds": 60.0},
            {"scenario": "S1", "error": "boom"},
            {"scenario": "S2", "lr": 30.0, "lr_star": None, "df": 7, "converged": True, "true_speed_ratio": 2.0,
             "sp1": 2.5, "seconds": 30.0}]
    s = P.summarise(rows)
    assert s["S1_speed_only"]["failed"] == 1
    assert abs(s["S1_speed_only"]["false_positives_bootstrap"]["rate"] - 2 / 3) < 1e-12   # the failure counts
    assert s["S2_one_link_differs"]["power_chi2"]["rate"] == 1.0


def test_nearly_flat_directions_do_not_swamp_the_covariance():
    """A parameter with (almost) no information, like a short-duration treatment effect on domains
    examined only OFF, must not get a variance of 1e19 that leaks into every other direction."""
    import numpy as np
    from pdx.engine.fit import covariance_checked
    rng = np.random.default_rng(0)
    Q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    H = (Q * np.array([5e3, 40.0, 2.0, 2e-19])) @ Q.T           # one direction the data say nothing about
    V, info = covariance_checked(H, names=["a", "b", "c", "k"])
    assert info["dropped_directions"] == 1 and not info["well_conditioned"]
    assert np.abs(V).max() < 1.0                                  # not 1e18
    keep = Q[:, :3]                                               # on the identified directions, the usual inverse
    assert np.allclose(keep.T @ V @ keep, np.diag(1 / np.array([5e3, 40.0, 2.0])))

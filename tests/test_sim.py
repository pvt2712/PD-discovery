import numpy as np

from pdx.sim import scenarios
from pdx.sim.family_a import INDETERMINATE, PIGD, TD, phenotype_rule


def test_phenotype_rule():
    trem = np.array([[2, 2, 2, 2], [0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]], float)
    pigd = np.array([[1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0]], float)
    assert phenotype_rule(trem, pigd).tolist() == [TD, PIGD, INDETERMINATE, INDETERMINATE]


def test_every_scenario_generates():
    for sc in scenarios.NULL_SCENARIOS + scenarios.POWER_SCENARIOS:
        panel, info = scenarios.generate(sc, 120, np.random.default_rng(1))
        assert panel.n > 50 and set(np.unique(panel.group)) == {0, 1}
        assert info["support"].shape == (4, 4) and info["support"].diagonal().all()
    for sc in ("S0", "S1", "S3"):
        panel, info = scenarios.generate(sc, 120, np.random.default_rng(2), family="B")
        assert panel.n > 50 and info["support"].shape == (4, 4)


def test_mechanism_scenarios_follow_the_table():
    from pdx.sim.scenarios import EPS, expert_map, pooled_sd, truth_for
    t1, t2, t2b, t3 = (truth_for(s) for s in ("S1", "S2", "S2b", "S3"))
    ax, ind, fn, tr = (t1.domains.index(k) for k in ("axial", "independence", "function", "tremor"))
    sd = pooled_sd(4)
    dev = (t2.A_pigd[ind, fn] - t2.A[ind, fn]) * sd[fn] / sd[ind]
    assert abs(dev - 2 * EPS) < 1e-12                           # S2: one deviation of 2 epsilon
    assert abs(t2b.A_pigd[ind, ax] / t2b.A[ind, ax] - 4) < 1e-12
    assert t3.A[ind, ax] == 0 and t3.A_pigd[ind, ax] > 0        # S3: a PIGD-only link
    assert t3.A[fn, tr] > 0 and t3.A_pigd[fn, tr] == 0          # and a TD-only link
    maps = [expert_map(truth_for(s)) for s in ("S0", "S1", "S2", "S2b", "S3")]
    assert all((m == maps[0]).all() for m in maps)              # the same map, so the same df
    assert np.allclose(t1.A, t1.A_pigd)                         # nulls share the mechanism


def test_family_a_treatment_uses_the_engine_form():
    from pdx.engine.model import ModelSpec
    from pdx.sim.family_a import default_truth
    t = default_truth(4)
    spec = ModelSpec(d=4, support=np.eye(4, dtype=bool), treat_mask=t.treat_mask, treatment="emax")
    assert t.ed50 == spec.ed50 and t.ec50_ldr == spec.ec50_ldr

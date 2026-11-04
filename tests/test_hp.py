import numpy as np
from helpers import slow

from pdx.hp import sequential_p


def _draws(values, oks=None):
    it = iter(zip(values, oks or [True] * len(values)))
    return lambda: next(it)


def test_sequential_stops_at_h_exceedances():
    vals = [5.0, 1.0, 6.0, 7.0, 0.5] + [0.1] * 100
    r = sequential_p(_draws(vals), observed=4.0, n_max=99, h=3)
    assert r["stopped_early"] and r["n_drawn"] == 4 and r["p"] == 3 / 4


def test_sequential_runs_to_n_max():
    r = sequential_p(_draws([0.1] * 99), observed=4.0, n_max=99, h=20)
    assert not r["stopped_early"] and r["p"] == 1 / 100


def test_ties_and_failures_count_as_exceedances():
    r = sequential_p(_draws([4.0, 0.0, 0.0], [True, False, True]), observed=4.0, n_max=3, h=20)
    assert r["n_exceed"] == 2 and r["failures"] == 1 and r["p"] == 3 / 4


def test_null_p_values_are_uniform():
    rng = np.random.default_rng(0)
    ps = []
    for _ in range(2000):
        obs = rng.chisquare(3)
        ps.append(sequential_p(lambda: (rng.chisquare(3), True), obs, n_max=99, h=10)["p"])
    ps = np.array(ps)
    assert abs((ps <= 0.05).mean() - 0.05) < 0.015
    assert abs((ps <= 0.5).mean() - 0.5) < 0.04


@slow
def test_speed_bootstrap_resumed_midway_gives_the_same_intervals():
    from helpers import small_panel, spec_for
    from pdx.engine.fit import fit_pair
    from pdx.speed import bootstrap_speed
    panel, info = small_panel("S1", n=80, seed=4)
    spec = spec_for(panel, info)
    f0, f1 = fit_pair(panel, spec)
    seen = []
    full = bootstrap_speed(panel, spec, f0, f1, B=2, rng=np.random.default_rng(7),
                           on_sample=lambda done: seen.append({k: (list(v) if isinstance(v, list) else v)
                                                               for k, v in done.items()}))
    resumed = bootstrap_speed(panel, spec, f0, f1, B=2, rng=np.random.default_rng(7), done=seen[0])
    assert len(seen) == 2 and resumed["sp1_ci"] == full["sp1_ci"] and resumed["B_used"] == full["B_used"]

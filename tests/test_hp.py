import numpy as np

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

import os

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
    full = bootstrap_speed(panel, spec, f0, f1, B=2, seed=7,
                           on_sample=lambda done: seen.append({k: (list(v) if isinstance(v, list) else v)
                                                               for k, v in done.items()}))
    resumed = bootstrap_speed(panel, spec, f0, f1, B=2, seed=7, done=seen[0])
    parallel_run = bootstrap_speed(panel, spec, f0, f1, B=2, seed=7, jobs=2)
    assert len(seen) == 2 and resumed["sp1_ci"] == full["sp1_ci"] and resumed["B_used"] == full["B_used"]
    # workers are single-threaded with smaller batches, so their fits reach the stopping rule (a gain
    # under 0.01 over 100 iterations) at slightly other points: 0.7% here, 0.03% without the rule
    assert np.allclose(parallel_run["sp1_ci"], full["sp1_ci"], rtol=2e-2)


def test_parallel_runs_give_the_results_in_order_from_several_workers():
    from pdx import parallel
    one = list(parallel.ordered(parallel.check_square, range(6), 1, parallel.check_setup, (2.0,)))
    many = list(parallel.ordered(parallel.check_square, range(6), 3, parallel.check_setup, (2.0,)))
    assert [v for v, _ in one] == [v for v, _ in many] == [2.0 * x * x for x in range(6)]
    assert all(pid != os.getpid() for _, pid in many) and all(pid == os.getpid() for _, pid in one)
    assert list(parallel.ordered(parallel.check_square, [], 3, parallel.check_setup, (2.0,))) == []


@slow
def test_bootstrap_replicates_agree_however_many_run_at_once():
    from helpers import small_panel, spec_for
    from pdx import hp, parallel
    from pdx.step1 import Step1Config
    panel, info = small_panel("S1", n=80, seed=5)
    spec = spec_for(panel, info)
    spec_s, f0, f1 = hp.observed_fits(panel, spec)
    args = (f0, panel, spec, info["support"], Step1Config(), None, 11)
    one = list(parallel.ordered(hp.boot_replicate, range(3), 1, hp.boot_setup, args))
    two = list(parallel.ordered(hp.boot_replicate, range(3), 2, hp.boot_setup, args))
    assert len(one) == 3 and all(abs(a[0] - b[0]) < 1e-3 and a[1] == b[1] for a, b in zip(one, two))

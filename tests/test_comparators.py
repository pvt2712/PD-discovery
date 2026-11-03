import numpy as np

from pdx.comparators import REGISTRY
from pdx.comparators import run as comp
from pdx.sim import scenarios
from pdx.step1 import Step1Config


def test_comparators_run_and_return_link_matrices():
    panel, _ = scenarios.generate("S1", 200, np.random.default_rng(3))
    for out in (comp.c12_dynotears(panel), comp.c5_tsldiffpc(panel),
                comp.c3_nct(panel, n_perm=20, rng=np.random.default_rng(0)),
                comp.c11_naive(panel, Step1Config(depth=2))["interaction_tests"]):
        assert np.array(out["cross_links"]).shape == (4, 4) and len(out["self_lags"]) == 4
    assert set(REGISTRY) == {f"C{i}" for i in range(1, 13)}

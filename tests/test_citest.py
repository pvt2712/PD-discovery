"""The clustered GCM's block test, used by the core-map test (falsify.block_p, PROTOCOL.md §7.1)."""
import numpy as np

from escalation_discovery.citest import ClusterGCM


def test_block_with_a_constant_column_is_untestable_not_a_crash():
    rng = np.random.default_rng(0)
    n, groups = 400, np.repeat(np.arange(100), 4)
    z = rng.normal(size=n)
    data = np.column_stack([z + rng.normal(size=n), np.zeros(n), z + rng.normal(size=n), z])   # column 1 is constant
    gcm = ClusterGCM(data, groups, learner="linear", cross_fit=1)
    r = gcm.block_test([0, 1], 2, [3])
    assert np.isnan(r["p"]) and np.isnan(r["coef"]).all()
    r = gcm.block_test([0], 2, [3])                    # the same test without the constant column
    assert np.isfinite(r["p"]) and np.isfinite(r["coef"]).all()

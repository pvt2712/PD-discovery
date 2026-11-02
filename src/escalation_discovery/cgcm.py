"""cgcm — the cluster-robust GCM (citest.ClusterGCM, ported from v2 with the 28 October 2026 corrections) as a
causal-learn conditional-independence test named "cgcm", for PC and FCI (as escalation-study/dag.py:_register did).

Additive cubic splines on the conditioning set, linear features of X and Y, cross-fitted over participants (K = 5),
patient-summed residual products, CR1 Wald test. A test the guard cannot run (too few training rows per design
column, or a singular covariance) returns p = 0, so the adjacency is kept (v1's correction of 28 October 2026: an
untestable question is not evidence of independence); the count is kept in UNTESTABLE.
"""
import numpy as np
from causallearn.utils.cit import CIT_Base, register_ci_test

from .citest import ClusterGCM

UNTESTABLE = {"count": 0}
NAME = "cgcm"


class CGCMTest(CIT_Base):
    def __init__(self, data, groups=None, learner="splines", cross_fit=5, quadratic=False, seed=0, **kwargs):
        super().__init__(data, **kwargs)
        self.check_cache_method_consistent(NAME, f"{learner}-{cross_fit}-{quadratic}")
        self.assert_input_data_is_valid(allow_nan=True)
        if groups is None:
            raise ValueError("cgcm needs groups=<participant id per row>")
        self.gcm = ClusterGCM(data, groups, learner=learner, cross_fit=cross_fit, quadratic=quadratic, seed=seed)

    def __call__(self, X, Y, condition_set=None):
        Xs, Ys, cond, key = self.get_formatted_XYZ_and_cachekey(X, Y, condition_set)
        if key in self.pvalue_cache:
            return self.pvalue_cache[key]
        p = self.gcm.test(Xs[0], Ys[0], list(cond))
        if np.isnan(p):
            UNTESTABLE["count"] += 1
            p = 0.0
        self.pvalue_cache[key] = p
        return p


register_ci_test(NAME, CGCMTest)

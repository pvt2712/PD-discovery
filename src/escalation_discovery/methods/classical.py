"""classical — PC, FCI and BOSS from causal-learn 0.1.4.8 (PROTOCOL.md §5.1, anchors A1-A4).

PC and FCI run on Gaussian-copula normal scores with test-wise deletion. The test is either the cluster-robust
generalised covariance measure, linear features with additive-spline adjustment, cross-fitted over participants
(`cgcm`, cgcm.py: the paper's own test, valid for repeated measures), or causal-learn's Fisher-z with test-wise
deletion (`mv_fisherz`: the textbook i.i.d. test). With knowledge, FCI gets the tiered semantics of
knowledge.py and PC gets the tiers as forbidden directions. BOSS (BIC) takes no knowledge in causal-learn.
"""
import contextlib
import io

import numpy as np

from .. import knowledge as K
from ..graphs import from_causallearn
from .base import Method

CLUSTER_TESTS = ("cgcm",)


def _test_kwargs(test, data):
    if test in CLUSTER_TESTS:
        from .. import cgcm  # noqa: F401  (registers the clustered test with causal-learn)
        return {"groups": data.groups, "seed": int(data.seed)}
    return {}


class FCI(Method):
    family = "classical"
    output = "PAG"
    accepts_knowledge = True
    missing = "testwise"
    version = "causal-learn 0.1.4.8"

    def __init__(self, test="cgcm", name=None, **cfg):
        super().__init__(**cfg)
        self.test = test
        self.name = name or f"fci_{'cgcm' if test == 'cgcm' else 'fz'}"

    def defaults(self):
        return dict(alpha=0.01, depth=3)        # largest separating set, in both search stages (PROTOCOL.md §10.2)

    def grid(self):
        return [dict(alpha=a) for a in (0.001, 0.005, 0.01, 0.05, 0.1, 0.2)]

    def _fit(self, data):
        X = data.copula()
        test = self.test if self.test in CLUSTER_TESTS else "mv_fisherz"
        depth = int(self.cfg.get("depth", -1))
        if data.knowledge:
            fci = K.install(data.context, data.exogenous, pds_depth=depth)
            bk, _ = K.background(data.names, data.tiers, data.context, data.exogenous)
        else:
            fci = K.install((), (), pds_depth=depth)
            bk = None
        with contextlib.redirect_stdout(io.StringIO()):
            G, _ = fci(X, test, self.cfg["alpha"], depth=depth, background_knowledge=bk,
                       show_progress=False, verbose=False, node_names=list(data.names), **_test_kwargs(test, data))
        return from_causallearn(G, data.names, "PAG")


class PC(Method):
    family = "classical"
    output = "CPDAG"
    accepts_knowledge = True
    missing = "testwise"
    version = "causal-learn 0.1.4.8"

    def __init__(self, test="cgcm", name=None, **cfg):
        super().__init__(**cfg)
        self.test = test
        self.name = name or f"pc_{'cgcm' if test == 'cgcm' else 'fz'}"

    def defaults(self):
        return dict(alpha=0.01, depth=-1)

    def grid(self):
        return [dict(alpha=a) for a in (0.001, 0.005, 0.01, 0.05, 0.1, 0.2)]

    def _fit(self, data):
        from causallearn.graph.GraphNode import GraphNode
        from causallearn.search.ConstraintBased.PC import pc
        X = data.copula()
        test = self.test if self.test in CLUSTER_TESTS else "mv_fisherz"
        alias = {n: f"X{i + 1}" for i, n in enumerate(data.names)}          # pc names its nodes X1..Xp
        bk = None
        if data.knowledge:
            bk, _ = K.background(data.names, data.tiers, data.context, data.exogenous, node=lambda n: GraphNode(alias[n]))
        with contextlib.redirect_stdout(io.StringIO()):
            cg = pc(X, self.cfg["alpha"], test, stable=True, depth=int(self.cfg.get("depth", -1)), background_knowledge=bk,
                    show_progress=False, **_test_kwargs(test, data))
        return from_causallearn(cg.G, data.names, "CPDAG")


class BOSS(Method):
    """causal-learn BOSS with the linear-Gaussian BIC on the shared imputed table (no knowledge in causal-learn)."""
    name = "boss_cl"
    family = "classical"
    output = "CPDAG"
    accepts_knowledge = False
    missing = "impute"
    version = "causal-learn 0.1.4.8"

    def defaults(self):
        return dict(lambda_value=2.0)

    def grid(self):
        return [dict(lambda_value=v) for v in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)]

    def _fit(self, data):
        from causallearn.search.PermutationBased.BOSS import boss
        from .base import copula
        X = copula(data.imputed())
        X = (X - X.mean(0)) / X.std(0)
        with contextlib.redirect_stdout(io.StringIO()):
            G = boss(X, score_func="local_score_BIC_from_cov", parameters={"lambda_value": self.cfg["lambda_value"]},
                     verbose=False, node_names=list(data.names))
        return from_causallearn(G, data.names, "CPDAG")

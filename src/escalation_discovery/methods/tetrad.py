"""tetrad — Tetrad 7.6.10 learners through JPype (PROTOCOL.md §5.1, anchors A6-A10).

BOSS, BOSS-FCI, GFCI and GRaSP-FCI with the degenerate-Gaussian BIC score and likelihood-ratio test for mixed data
(DG-BIC, DG-LRT: Andrews, Ramsey & Cooper, PMLR 104:4-21, 2019), and PC/FCI with DG-LRT, from the algcomparison
layer of tools/tetrad-gui-7.6.10-launch.jar (Maven Central; SHA-1 d7d9dc08ad2209a8d5bca6e94ab58323b1f48007), run on
OpenJDK 21. Knowledge: the same tiers; the context and exogenous tiers forbid edges within themselves.

Tetrad's DG score needs complete rows: these learners get the shared imputed table. A variable with at most five
distinct values (sex, APOE e4, the cohort indicators, hallucinations, impulse-control domains) enters as discrete,
every other as continuous (PROTOCOL.md §5.3).
"""
from pathlib import Path

import numpy as np

from .. import config as C
from ..graphs import MarkGraph
from .base import Method

MAX_DISCRETE_LEVELS = 5
_MARK = {"TAIL": "T", "ARROW": "A", "CIRCLE": "C"}


def jvm_path():
    home = Path(C.java_home())
    for cand in (home / "libexec/openjdk.jdk/Contents/Home/lib/server/libjvm.dylib", home / "lib/server/libjvm.dylib"):
        if cand.exists():
            return cand
    return None


def start_jvm():
    """One JVM per process; heap from ESCD_JAVA_HEAP (default 6g: BOSS-FCI on the dense two-slice graphs needs >2g)."""
    import os
    import jpype
    import jpype.imports  # noqa: F401
    if not jpype.isJVMStarted():
        heap = os.environ.get("ESCD_JAVA_HEAP", "6g")
        jpype.startJVM(str(jvm_path()), f"-Xmx{heap}", classpath=[str(C.tetrad_jar())], convertStrings=True)


def _dataset(X, names):
    """Tetrad mixed data set: columns with <= 5 distinct values become discrete (codes 0..k-1)."""
    from java.util import ArrayList
    from jpype import JArray, JDouble, JInt
    from edu.cmu.tetrad.data import BoxDataSet, ContinuousVariable, DiscreteVariable, MixedDataBox
    n, d = X.shape
    variables = ArrayList()
    cont = [None] * d
    disc = [None] * d
    for j, name in enumerate(names):
        col = X[:, j]
        levels = np.unique(col)
        if levels.size <= MAX_DISCRETE_LEVELS:
            codes = np.searchsorted(levels, col).astype(np.int32)
            variables.add(DiscreteVariable(name, int(levels.size)))
            disc[j] = JArray(JInt)(codes.tolist())
        else:
            variables.add(ContinuousVariable(name))
            cont[j] = JArray(JDouble)(col.astype(float).tolist())
    box = MixedDataBox(variables, n, JArray(JArray(JDouble))(cont), JArray(JArray(JInt))(disc))
    return BoxDataSet(box, variables)


def snap_discrete(X_obs, X_imp):
    """Imputed values of a discrete variable (<= 5 observed levels) moved to the nearest observed level."""
    X = np.array(X_imp, dtype=float, copy=True)
    for j in range(X.shape[1]):
        lv = np.unique(X_obs[~np.isnan(X_obs[:, j]), j])
        if 0 < lv.size <= MAX_DISCRETE_LEVELS:
            X[:, j] = lv[np.abs(X[:, j][:, None] - lv[None, :]).argmin(1)]
    return X


def _knowledge(data, forbid_within=True):
    from edu.cmu.tetrad.data import Knowledge
    kn = Knowledge()
    ctx, exo = set(data.context), set(data.exogenous)
    for k, tier in enumerate(data.tiers):
        for name in tier:
            kn.addToTier(k, name)
        if forbid_within and tier and (all(n in ctx for n in tier) or all(n in exo for n in tier)):
            kn.setTierForbiddenWithin(k, True)
    return kn


def _graph(g, names, kind):
    out = MarkGraph(names, kind)
    for e in g.getEdges():
        a, b = str(e.getNode1().getName()), str(e.getNode2().getName())
        ma, mb = _MARK[str(e.getEndpoint1())], _MARK[str(e.getEndpoint2())]
        if kind == "CPDAG" and (ma, mb) == ("T", "T"):
            ma, mb = "C", "C"
        out.add(a, b, ma, mb)
    return out


class Tetrad(Method):
    family = "classical"
    accepts_knowledge = True
    missing = "impute"
    version = "Tetrad 7.6.10 (algcomparison), OpenJDK 21"

    ALGS = {"boss": ("cpdag", "Boss", "score"), "pc": ("cpdag", "Pc", "test"), "fci": ("pag", "Fci", "test"),
            "boss_fci": ("pag", "BossFci", "both"), "gfci": ("pag", "Gfci", "both"), "grasp_fci": ("pag", "GraspFci", "both")}

    def __init__(self, alg="boss_fci", **cfg):
        super().__init__(**cfg)
        self.alg = alg
        self.name = f"tetrad_{alg}"
        self.output = "CPDAG" if self.ALGS[alg][0] == "cpdag" else "PAG"

    def defaults(self):
        return dict(alpha=0.01, penalty=2.0, depth=-1)    # depth: largest conditioning set (-1 = no cap)

    def grid(self):
        uses = self.ALGS[self.alg][2]
        if uses == "test":
            return [dict(alpha=a, penalty=2.0) for a in (0.001, 0.005, 0.01, 0.05, 0.1, 0.2)]
        if uses == "score":
            return [dict(alpha=0.01, penalty=p) for p in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)]
        return [dict(alpha=a, penalty=p) for a in (0.01, 0.05) for p in (1.0, 2.0, 4.0)]

    def available(self):
        if jvm_path() is None:
            return False, f"no JVM under {C.java_home()} (brew install openjdk@21)"
        if not C.tetrad_jar().exists():
            return False, f"missing {C.tetrad_jar()}"
        return True, ""

    def _fit(self, data):
        start_jvm()
        import importlib
        from edu.cmu.tetrad.algcomparison.independence import DegenerateGaussianLrt
        from edu.cmu.tetrad.algcomparison.score import DegenerateGaussianBicScore
        from edu.cmu.tetrad.util import Parameters, Params
        kind, cls, uses = self.ALGS[self.alg]
        Alg = getattr(importlib.import_module(f"edu.cmu.tetrad.algcomparison.algorithm.oracle.{kind}"), cls)
        test, score = DegenerateGaussianLrt(), DegenerateGaussianBicScore()
        alg = {"test": lambda: Alg(test), "score": lambda: Alg(score), "both": lambda: Alg(test, score)}[uses]()
        if data.knowledge:
            alg.setKnowledge(_knowledge(data, bool(self.cfg.get("forbid_within", True))))
        p = Parameters()
        for key, val in [(Params.ALPHA, float(self.cfg["alpha"])), (Params.PENALTY_DISCOUNT, float(self.cfg["penalty"])),
                         (Params.STRUCTURE_PRIOR, 0.0), (Params.NUM_STARTS, 1), (Params.USE_BES, False),
                         (Params.DEPTH, int(self.cfg.get("depth", -1))),
                         (Params.MAX_DISCRIMINATING_PATH_LENGTH, int(self.cfg.get("max_disc_path", -1))),
                         (Params.COMPLETE_RULE_SET_USED, True),
                         (Params.NUM_THREADS, 1), (Params.SEED, int(data.seed)), (Params.VERBOSE, False),
                         (Params.PRECOMPUTE_COVARIANCES, True)]:
            p.set(key, val)
        X = snap_discrete(data.X, data.imputed())
        g = alg.search(_dataset(X, data.names), p)
        return _graph(g, data.names, self.output)

"""base — the adapter interface: every learner maps (data, tiers) to a MarkGraph (PROTOCOL.md §5).

Preprocessing is fixed per method family and identical across methods of a family (PROTOCOL.md §5.3):
- conditional-independence learners: Gaussian-copula normal scores, missing values handled test-wise;
- score-based, continuous-optimisation, neural and pretrained learners: one shared chained-equation imputation per
  dataset (the same imputed table for every method), then z-scores, or raw units for the scale-sensitive learners in
  the units stress setting;
- learners with native missing-data support use it.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm, rankdata

from .. import expert as E
from ..graphs import MarkGraph


@dataclass
class DiscoveryData:
    """One dataset as every learner sees it."""
    X: np.ndarray                        # rows x variables, clinical units, NaN = missing
    names: list
    groups: np.ndarray                   # participant id of each row (clusters)
    tiers: list                          # background knowledge (expert.tiers)
    context: tuple = E.CONTEXT
    exogenous: tuple = E.EXOGENOUS
    knowledge: bool = True               # give tiers to the learners that accept them
    standardize: bool = True             # z-score inputs of scale-sensitive learners (False = raw units stress)
    listwise: bool = False               # complete-row learners drop incomplete rows instead of imputing (sensitivity)
    long: object = None                  # long panel (PATNO, year, nodes) for panel learners (J-PCMCI+)
    base: object = None                  # one row per participant: context, exogenous and baseline nodes
    seed: int = 0
    cache: dict = field(default_factory=dict)

    @property
    def tier_of(self):
        return {n: k for k, t in enumerate(self.tiers) for n in t}

    def copula(self):
        if "copula" not in self.cache:
            self.cache["copula"] = copula(self.X)
        return self.cache["copula"]

    def imputed(self):
        """Chained-equation imputation (sklearn IterativeImputer, Bayesian ridge, 10 rounds), shared by the learners
        that need complete rows."""
        if "imputed" not in self.cache:
            X = self.X
            if self.listwise:
                X = X[~np.isnan(X).any(axis=1)]
            elif np.isnan(X).any():
                from sklearn.experimental import enable_iterative_imputer  # noqa: F401
                from sklearn.impute import IterativeImputer
                X = IterativeImputer(max_iter=10, random_state=self.seed, skip_complete=True).fit_transform(X)
            self.cache["imputed"] = X
        return self.cache["imputed"]

    def complete_input(self, scale_sensitive=True):
        """Imputed table, z-scored unless the units stress setting asks for raw units."""
        X = self.imputed()
        if self.standardize or not scale_sensitive:
            sd = X.std(0)
            X = (X - X.mean(0)) / np.where(sd > 0, sd, 1.0)
        return X


def copula(X):
    """Gaussian-copula (nonparanormal) normal scores column-wise, NaN kept."""
    X = np.asarray(X, dtype=float).copy()
    for j in range(X.shape[1]):
        m = ~np.isnan(X[:, j])
        n = m.sum()
        if n > 1:
            X[m, j] = norm.ppf(rankdata(X[m, j]) / (n + 1))
    return X


def forbidden_matrix(data: DiscoveryData):
    """F[i, j] = True when the knowledge forbids i -> j: into a context node, into an exogenous node from anything but
    context, between two context or two exogenous nodes, or from a later tier into an earlier one."""
    names, tof = data.names, data.tier_of
    ctx, exo = set(data.context), set(data.exogenous)
    d = len(names)
    F = np.zeros((d, d), dtype=bool)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i == j:
                continue
            if b in ctx or (b in exo and a not in ctx) or (a in ctx and b in ctx) or (a in exo and b in exo):
                F[i, j] = True
            elif a in tof and b in tof and tof[a] > tof[b]:
                F[i, j] = True
    return F


class Method:
    """An adapter. Subclasses set the class attributes and implement `_fit`."""
    name = "base"
    family = "classical"           # classical | continuous | neural | end-to-end | pretrained | temporal | ensemble
    output = "PAG"                 # DAG | CPDAG | PAG
    accepts_knowledge = False
    missing = "testwise"           # testwise | impute | native
    scale_sensitive = False
    version = ""

    def __init__(self, **cfg):
        self.cfg = dict(self.defaults(), **cfg)

    def defaults(self):
        return {}

    def grid(self):
        """The tuning grid (PROTOCOL.md §5.4: at most 6 configurations per method)."""
        return [{}]

    def available(self):
        return True, ""

    def fit(self, data: DiscoveryData) -> MarkGraph:
        g = self._fit(data)
        g.kind = self.output
        return g

    def _fit(self, data):
        raise NotImplementedError

    def __repr__(self):
        return f"{self.name}({self.cfg})"

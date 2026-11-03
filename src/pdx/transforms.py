"""Score orientation and the frozen normal-scores transform (protocol v0.5, §6.1, §7.1).

Every domain is oriented so that higher means worse (MoCA and Schwab & England are reversed).
Every score then enters on one normal-scores transform, computed once on the pooled primary
population over all visits and groups, with mid-ranks for ties, and frozen: bootstrap and
simulated datasets are generated on that scale and never re-transformed. Part II recomputes it
within each training fold. The fitted transform holds per-value lookup tables of the pooled data,
so it is saved only inside the git-ignored data folder.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

# instruments and the engine's domain names whose raw scores are higher-is-better
REVERSED = ("moca", "schwab_england", "se_adl", "cognition", "independence")


def orient(values: np.ndarray, domains: list, reversed_domains=REVERSED) -> np.ndarray:
    """Multiply the columns of reversed domains by -1 so that higher means worse. The list of
    reversed domains must match the WP1 domain definitions; pass it explicitly when they change."""
    out = np.array(values, float, copy=True)
    for j, nm in enumerate(domains):
        if nm.lower() in reversed_domains:
            out[..., j] = -out[..., j]
    return out


@dataclass
class NormalScores:
    """Per-column map from raw values to normal scores, Phi^-1((midrank - 0.5) / n)."""
    tables: list = field(default_factory=list)     # per column: (sorted unique values, their scores)

    def fit(self, X: np.ndarray) -> "NormalScores":
        X = np.asarray(X, float)
        self.tables = []
        for j in range(X.shape[1]):
            v = X[:, j]
            v = v[~np.isnan(v)]
            u, counts = np.unique(v, return_counts=True)
            upper = np.cumsum(counts)
            mid = upper - (counts - 1) / 2.0            # mid-ranks of tied values
            self.tables.append((u, stats.norm.ppf((mid - 0.5) / len(v))))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Frozen map; values between fitted values are interpolated, values outside are clamped."""
        X = np.asarray(X, float)
        out = np.full_like(X, np.nan)
        for j, (u, sc) in enumerate(self.tables):
            ok = ~np.isnan(X[..., j])
            out[..., j][ok] = np.interp(X[..., j][ok], u, sc)
        return out

    def inverse(self, Z: np.ndarray) -> np.ndarray:
        """Back to the raw scale (for clinical units); monotone interpolation."""
        Z = np.asarray(Z, float)
        out = np.full_like(Z, np.nan)
        for j, (u, sc) in enumerate(self.tables):
            ok = ~np.isnan(Z[..., j])
            out[..., j][ok] = np.interp(Z[..., j][ok], sc, u)
        return out

"""Fidelity dashboard summaries (protocol v0.5, §8.2): the same aggregate statistics computed on
the real panel and on simulated panels, so that the simulators can be compared with PPMI. Every
output is an aggregate (quantiles, correlations, shares); thresholds are fixed in WP2."""
from __future__ import annotations

import numpy as np

from .data import Panel


def panel_summaries(panel: Panel, quantiles=(0.01, 0.1, 0.5, 0.9, 0.99)) -> dict:
    d = panel.d
    allv = np.vstack(panel.y)
    X, X1 = [], []
    for yi, ti in zip(panel.y, panel.t):
        for k in range(len(yi) - 1):
            if ti[k + 1] - ti[k] <= 1.5:
                X.append(yi[k])
                X1.append(yi[k + 1])
    X, X1 = np.array(X), np.array(X1)
    ok = ~(np.isnan(X).any(1) | np.isnan(X1).any(1))
    X, X1 = X[ok], X1[ok]
    lag = np.corrcoef(np.hstack([X, X1]).T)[:d, d:] if len(X) > 2 else np.full((d, d), np.nan)
    same = np.corrcoef(X1.T) if len(X1) > 2 else np.full((d, d), np.nan)
    Z = np.column_stack([np.ones(len(X)), X])
    C, *_ = np.linalg.lstsq(Z, X1, rcond=None)
    resid = X1 - Z @ C
    within = np.nanmean([np.nanvar(yi, axis=0) for yi in panel.y if len(yi) > 1], axis=0)
    visits = np.array([len(t) for t in panel.t])
    return {"marginal_quantiles": np.nanquantile(allv, quantiles, axis=0).tolist(),
            "lag_correlations": lag.tolist(), "same_year_correlations": same.tolist(),
            "residual_sd": resid.std(0).tolist(), "within_person_variance": within.tolist(),
            "share_missing": np.isnan(allv).mean(0).tolist(),
            "phenotype_share": float(np.mean(panel.group == 1)),
            "visits_quantiles": np.quantile(visits, (0.1, 0.5, 0.9)).tolist(),
            "share_treated_last_visit": float(np.mean([dz[-1] > 0 for dz in panel.dose]))}


def compare(real: dict, simulated: list) -> dict:
    """Simulated summaries (one dict per dataset) against the real ones: the real value's position
    within the simulated spread, per statistic and entry."""
    out = {}
    for key, rv in real.items():
        rv = np.asarray(rv, float)
        sv = np.array([np.asarray(s[key], float) for s in simulated])
        lo, hi = np.nanquantile(sv, [0.025, 0.975], axis=0)
        out[key] = {"real": rv.tolist(), "sim_2.5": np.asarray(lo).tolist(), "sim_97.5": np.asarray(hi).tolist(),
                    "inside": np.asarray((rv >= lo) & (rv <= hi)).tolist()}
    return out

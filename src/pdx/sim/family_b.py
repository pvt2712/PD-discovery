"""Simulator family B, outside the engine's class and coded separately (protocol v0.5, §8.2).

Discrete-time latent states with non-linear dynamics, a burden that drifts in calendar time and
moves every domain, site effects, and a treatment effect that saturates with time on treatment
rather than with dose. Measurement differs by phenotype and era through a common scale factor
on all domains, which leaves the measured mechanism shared up to speed. Domain-specific loadings
that differ by phenotype (`noninvariant`) change the measured drift itself (it becomes
L A L^-1), so they are a named stress, not part of the null scenarios. Under the null scenarios
the dynamics are shared and PIGD only runs them faster (its yearly step is scaled); the burden
walk is not speed-scaled, a load-type difference that H-P's null allows. Phenotypes are
re-derived from simulated year-0 items, as in family A. Synthetic data only.
"""
from __future__ import annotations

import numpy as np

from ..data import Panel
from .family_a import INDETERMINATE, PIGD, TD, _items, phenotype_rule


def simulate(n: int, rng: np.random.Generator, d: int = 4, speed_ratio: float = 2.0, sigma: float = 0.4,
             mechanism_shift: float = 0.0, visits=(4, 7), n_sites: int = 20, p_missing: float = 0.04,
             load_shift: float = 0.05, noninvariant: float = 0.0):
    names = ["tremor", "axial", "function", "independence"] + [f"extra{k}" for k in range(d - 4)]
    W = np.zeros((d, d))
    W[2, 0], W[2, 1], W[3, 2] = 0.10, 0.20, 0.25
    decay = 0.25
    C0 = 0.3 * np.eye(d) + 0.1
    C0[0, 1] = C0[1, 0] = -0.15
    X0 = rng.multivariate_normal(np.zeros(d), C0, size=n)
    ph = phenotype_rule(_items(X0[:, 0], 4, rng), _items(X0[:, 1], 5, rng))
    keep = ph != INDETERMINATE
    X0, ph = X0[keep], ph[keep]
    m = len(ph)
    era = (rng.random(m) < 0.5).astype(int)
    site = rng.integers(n_sites, size=m)
    site_eff = rng.normal(scale=0.08, size=(n_sites, d))
    z = np.column_stack([era, rng.normal(size=m), (rng.random(m) < 0.4), rng.normal(size=m)]).astype(float)
    z -= z.mean(0)
    speed = np.where(ph == PIGD, speed_ratio, 1.0) * np.exp(sigma * rng.normal(size=m))
    load = 0.1 * rng.normal(size=(m, d)) + np.where(ph[:, None] == PIGD, load_shift, 0.0) * (np.arange(d) == 1)
    tmask = np.zeros(d, bool)
    tmask[[0, 1]] = True
    ys, ts, doses, expos, ons = [], [], [], [], []
    for i in range(m):
        Wi = W.copy()
        if ph[i] == PIGD and mechanism_shift:
            Wi[3, 1] = mechanism_shift
        T = rng.integers(visits[0], visits[1] + 1)
        times = np.cumsum(1.0 + rng.uniform(-0.15, 0.15, size=T))
        loading = (1.0 + 0.04 * ph[i] + 0.05 * era[i]
                   + noninvariant * (ph[i] - 0.5) * np.where(np.arange(d) % 2 == 0, 1, -1))
        X, burden, t_prev, t_on = X0[i].copy(), 0.0, 0.0, None
        rows, dr, er, orow = [], [], [], []
        for k in range(T):
            dt = times[k] - t_prev
            for _ in range(4):
                step = speed[i] * dt / 4
                X = X + step * (-decay * X + Wi @ np.tanh(X) + 0.1 + load[i] + burden) \
                    + np.sqrt(0.1 * step) * rng.normal(size=d)
                burden += 0.03 * np.sqrt(dt / 4) * rng.normal()
            t_prev = times[k]
            if k > 0 and t_on is None and rng.random() < 1 / (1 + np.exp(-(-2.2 + X[tmask].mean()))):
                t_on = times[k]
            time_on = 0.0 if t_on is None else times[k] - t_on
            dose = 0.0 if t_on is None else min(9.0, 3.0 + 1.5 * time_on)
            on = float(t_on is not None and rng.random() < 0.5)
            h = np.where(tmask, -0.3 * (1 - np.exp(-time_on / 0.5)) * (0.5 + 0.5 * on), 0.0)
            y = loading * X + site_eff[site[i]] + h + 0.15 * rng.normal(size=d)
            y[rng.random(d) < p_missing] = np.nan
            rows.append(y)
            dr.append(dose)
            er.append(dose)
            orow.append(on)
            if k > 0 and rng.random() < 1 / (1 + np.exp(-(-4.0 + 0.8 * X[3]))):
                break
        ys.append(np.array(rows))
        ts.append(times[:len(rows)])
        doses.append(np.array(dr))
        expos.append(np.array(er))
        ons.append(np.array(orow))
    keep2 = [i for i in range(m) if len(ys[i]) >= 2]
    panel = Panel([ys[i] for i in keep2], [ts[i] for i in keep2], ph[keep2], z=z[keep2],
                  dose=[doses[i] for i in keep2], exposure=[expos[i] for i in keep2],
                  on=[ons[i] for i in keep2], domains=names, meta={"family": "B"},
                  year=[np.arange(1, len(ys[i]) + 1) for i in keep2])
    support = np.eye(d, dtype=bool) | (W != 0)
    support[3, 1] = True                                   # the expert map's candidate link, in every scenario
    return panel, {"speed_ratio": speed_ratio, "support": support,
                   "n_TD": int((ph[keep2] == TD).sum()), "n_PIGD": int((ph[keep2] == PIGD).sum())}

"""The simulation scenarios of protocol v0.5, §8.2, for family A (and the S0/S1/S3 analogues of B).

Shared-mechanism scenarios (the calibration of H-M1): S0, S0L, S1, S1S, S1T, S1V, S4, S7.
Mechanism-difference scenarios (power and localisation): S2, S2b, S3.

Deviations are sized on the protocol's standardised scale: epsilon = 0.10 standard deviations of
the target domain per standard deviation of the source domain per year at the reference speed,
using the pooled standard deviations of the landmark scores. Every scenario of a family hands the
analysis the same expert map (the true links plus one candidate link that is zero in the null
scenarios), so the null and power scenarios have the same degrees of freedom.

Stresses available through generate(): spurious (one more zero link in the map), missing_link
(a true link left out of the map), noninvariant (family B: domain-specific loadings that differ
by phenotype), and family A's simulate() options (sample size, visits, missingness, dropout and
treatment-start strength). The others listed in §8.2 are still to add.
"""
from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

import numpy as np

from . import family_a, family_b
from .family_a import Truth, default_truth

NULL_SCENARIOS = ("S0", "S0L", "S1", "S1S", "S1T", "S1V", "S4", "S7")
POWER_SCENARIOS = ("S2", "S2b", "S3")
EPS = 0.10


def _load_truth(t: Truth) -> Truth:
    b_load = t.b.copy()
    b_load[1, t.domains.index("axial")] += 0.05            # extra axial push in PIGD
    return replace(t, b=b_load)


@lru_cache(maxsize=None)
def pooled_sd(d: int) -> tuple:
    """Pooled standard deviations of the simulated landmark scores (fixed seed), for sizing."""
    panel, _ = family_a.simulate(_load_truth(default_truth(d)), 3000, np.random.default_rng(12345))
    y1 = np.array([y[0] for y in panel.y])
    return tuple(np.nanstd(y1, axis=0))


def _std(d: int, to: int, frm: int) -> float:
    """One epsilon in raw drift units for the link frm -> to."""
    sd = pooled_sd(d)
    return EPS * sd[to] / sd[frm]


def truth_for(scenario: str, d: int = 4) -> Truth:
    t = default_truth(d)
    names = t.domains
    ax, ind, fn, tr = (names.index(k) for k in ("axial", "independence", "function", "tremor"))
    tl = _load_truth(t)
    if scenario == "S0":
        return replace(t, speed=(1.0, 1.0))
    if scenario == "S0L":
        return replace(tl, speed=(1.0, 1.0))
    if scenario == "S1":
        return tl
    if scenario == "S1S":
        return replace(tl, sigma=(0.4, 0.6))
    if scenario == "S1T":
        kap = t.kappa.copy()
        kap[1, ax] *= 0.5
        return replace(tl, kappa=kap)
    if scenario == "S1V":
        return replace(tl, tau=(0.10, 0.15))
    if scenario == "S4":
        return replace(tl, curvature=0.08)
    if scenario == "S7":
        return replace(tl, sigma=(0.6, 0.6))
    A, Ap = t.A.copy(), t.A.copy()
    if scenario == "S2":                                   # one link stronger beyond speed, by 2 epsilon
        Ap[ind, fn] += 2 * _std(d, ind, fn)
    elif scenario == "S2b":                                # near epsilon in TD, four times as strong in PIGD
        A[ind, ax] = _std(d, ind, ax)
        Ap[ind, ax] = 4 * _std(d, ind, ax)
    elif scenario == "S3":                                 # one link only in PIGD, another only in TD
        Ap[ind, ax] = 2 * _std(d, ind, ax)
        Ap[fn, tr] = 0.0
    else:
        raise ValueError(f"unknown scenario {scenario}")
    return replace(tl, A=A, A_pigd=Ap)


def expert_map(truth: Truth) -> np.ndarray:
    """The map every scenario hands the analysis: the default links plus the candidate
    independence <- axial link (zero except in S2b and S3)."""
    base = default_truth(truth.d)
    sup = family_a.true_support(base) | family_a.true_support(truth)
    sup[truth.domains.index("independence"), truth.domains.index("axial")] = True
    return sup


def generate(scenario: str, n: int, rng: np.random.Generator, family: str = "A", d: int = 4, **kw):
    """(panel, info) with the support the analysis should use."""
    spurious = kw.pop("spurious", False)
    missing_link = kw.pop("missing_link", False)
    if family == "B":
        if scenario not in ("S0", "S1", "S3"):
            raise ValueError("family B has S0, S1 and S3 analogues")
        panel, info = family_b.simulate(n, rng, d=d, speed_ratio=1.0 if scenario == "S0" else 2.0,
                                        mechanism_shift=0.12 if scenario == "S3" else 0.0,
                                        load_shift=0.0 if scenario == "S0" else 0.05,
                                        noninvariant=kw.pop("noninvariant", 0.0))
        info["scenario"] = scenario
        return panel, info
    truth = truth_for(scenario, d)
    panel, info = family_a.simulate(truth, n, rng, **kw)
    sup = expert_map(truth)
    if spurious:
        sup[truth.domains.index("axial"), truth.domains.index("tremor")] = True
    if missing_link:
        sup[truth.domains.index("function"), truth.domains.index("axial")] = False
    info.update({"scenario": scenario, "support": sup, "truth": truth})
    return panel, info

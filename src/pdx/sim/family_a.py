"""Simulator family A, the engine's class (protocol v0.5, §8.2).

Participants are generated from year 0. The phenotype is re-derived from simulated year-0
MDS-UPDRS items by the TD/PIGD rule (Stebbins 2013 cut-offs), so the simulated phenotypes carry
the artefacts of defining groups from graph variables. Year 0 is then propagated to the year-1
landmark, from which the panel starts. Treatment starts with a hazard that rises with motor
severity (confounding by indication) and acts on the observed scores through the engine's Emax
forms (short-duration dose effect in the ON state, long-duration effect of exposure), dropout
rises with dependence, and some scores go missing. Synthetic data only.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..data import Panel
from ..engine.transitions import transition_np

TD, PIGD, INDETERMINATE = 0, 1, -1


@dataclass
class Truth:
    domains: list
    A: np.ndarray                 # (d, d) reference (TD) drift, A[to, from]
    A_pigd: np.ndarray            # PIGD drift at the reference speed (= A without a mechanism difference)
    speed: tuple = (1.0, 2.0)     # median speed of TD, PIGD
    sigma: tuple = (0.4, 0.4)     # spread of log-speed within TD, PIGD
    b: np.ndarray = None          # (2, d) loads
    q: float = 0.10               # diffusion variance per unit time (diagonal)
    tau: tuple = (0.10, 0.10)     # load spread within TD, PIGD
    lam: float = 0.0              # how load moves with speed
    meas_var: np.ndarray = None   # (d,) measurement-error variances
    treat_mask: np.ndarray = None  # (d,) treated domains
    kappa: np.ndarray = None      # (2, d) maximal short-duration effect (Emax, ON state)
    kappa_ldr: np.ndarray = None  # (2, d) maximal long-duration effect (Emax of exposure)
    ed50: float = 3.0             # half-maximal dose, 100 mg LEDD (as the engine's default)
    ec50_ldr: float = 3.0
    curvature: float = 0.0        # shared non-linear drift term c * tanh(X) (scenario S4)
    beta_age: float = 0.0         # effect of age at onset on log-speed
    meta: dict = field(default_factory=dict)

    @property
    def d(self) -> int:
        return len(self.domains)

    @property
    def speed_ratio(self) -> float:
        return self.speed[1] / self.speed[0]


def default_truth(d: int = 4) -> Truth:
    """A small shared mechanism: tremor and axial feed daily function, which feeds independence."""
    if d == 4:
        names = ["tremor", "axial", "function", "independence"]
        links = {(2, 0): 0.10, (2, 1): 0.20, (3, 2): 0.25}
    elif d == 6:
        names = ["tremor", "axial", "brady", "function", "independence", "cognition"]
        links = {(3, 0): 0.10, (3, 1): 0.15, (3, 2): 0.20, (4, 3): 0.25, (4, 5): 0.15, (5, 1): 0.08}
    elif d == 10:
        names = ["tremor", "axial", "brady", "function", "independence", "cognition", "mood", "autonomic",
                 "hallucinations", "sleepiness"]
        links = {(3, 0): 0.10, (3, 1): 0.15, (3, 2): 0.20, (4, 3): 0.25, (4, 5): 0.15, (5, 1): 0.08,
                 (6, 5): 0.10, (8, 5): 0.12, (4, 7): 0.08, (9, 7): 0.10, (6, 9): 0.08}
    else:
        raise ValueError("default_truth has presets for d = 4, 6 and 10")
    A = -0.25 * np.eye(d)
    for (i, j), v in links.items():
        A[i, j] = v
    b = np.tile(-A @ np.ones(d) * 0.5, (2, 1))
    tm = np.zeros(d, bool)
    tm[[names.index("tremor"), names.index("axial")]] = True
    if "brady" in names:
        tm[names.index("brady")] = True
    kap = np.where(tm, 0.5, 0.0)
    return Truth(names, A, A.copy(), b=b, meas_var=np.full(d, 0.02), treat_mask=tm,
                 kappa=np.tile(kap, (2, 1)), kappa_ldr=np.tile(0.5 * kap, (2, 1)))


def phenotype_rule(tremor_items: np.ndarray, pigd_items: np.ndarray) -> np.ndarray:
    """TD/PIGD from item means: TD if the ratio is at least 1.15 (or PIGD items are all 0 with
    tremor present), PIGD if at most 0.90, otherwise indeterminate (Stebbins 2013)."""
    tm, pm = tremor_items.mean(1), pigd_items.mean(1)
    out = np.full(len(tm), INDETERMINATE)
    pos = pm > 0
    ratio = np.where(pos, tm / np.where(pos, pm, 1.0), np.inf)
    out[(ratio >= 1.15) & ~((tm == 0) & (pm == 0))] = TD
    out[pos & (ratio <= 0.90)] = PIGD
    return out


def _items(level: np.ndarray, k: int, rng) -> np.ndarray:
    raw = 1.2 + level[:, None] + rng.normal(scale=0.6, size=(len(level), k))
    return np.clip(np.round(raw), 0, 4)


def _step(X, A, b, l, s, q, dt, c, rng, substeps=10):
    """Euler-Maruyama step for the curved model (only used when curvature != 0)."""
    h = dt / substeps
    for _ in range(substeps):
        X = X + s * (A @ X + b + l + c * np.tanh(X)) * h + np.sqrt(s * q * h) * rng.normal(size=len(X))
    return X


def simulate(truth: Truth, n: int, rng: np.random.Generator, visits=(4, 7), jitter: float = 0.15,
             p_missing=0.04, start_hazard=(-2.2, 1.0), dropout=(-4.0, 0.8),
             keep_indeterminate: bool = False, visit_probs: dict | None = None,
             target_counts: tuple | None = None) -> tuple[Panel, dict]:
    """Simulate n participants from year 0 (fewer remain after the phenotype rule).

    visit_probs maps a number of visits from the landmark on (the landmark included) to its
    probability, for example WP0's observed distribution, and replaces the uniform `visits` range; p_missing may be one
    share or one per domain; target_counts = (n_TD, n_PIGD) subsamples the simulated phenotypes
    to the given sizes (n must be large enough), so that the group sizes match real counts.
    """
    d = truth.d
    p_missing = np.broadcast_to(np.asarray(p_missing, float), (d,))
    if visit_probs:
        vk = np.array(sorted(visit_probs), int)
        vp = np.array([visit_probs[k] for k in vk], float)
        vp = vp / vp.sum()
    names = truth.domains
    i_trem, i_ax = names.index("tremor"), names.index("axial")
    i_ind = names.index("independence")
    # year 0: tremor and axial pull against each other, which drives the phenotype rule
    C0 = 0.3 * np.eye(d) + 0.1
    C0[i_trem, i_ax] = C0[i_ax, i_trem] = -0.15
    X0 = rng.multivariate_normal(np.zeros(d), C0, size=n)
    ph = phenotype_rule(_items(X0[:, i_trem], 4, rng), _items(X0[:, i_ax], 5, rng))
    keep = np.ones(n, bool) if keep_indeterminate else ph != INDETERMINATE
    X0, ph = X0[keep], ph[keep]
    ph = np.where(ph == INDETERMINATE, PIGD, ph)
    m = len(ph)
    era = (rng.random(m) < 0.5).astype(float)
    age = rng.normal(size=m)
    sex = (rng.random(m) < 0.4).astype(float)
    dur = rng.normal(size=m)
    z = np.column_stack([era, age, sex, dur])
    z = z - z.mean(0)
    u = rng.normal(size=m)
    s = np.array(truth.speed)[ph] * np.exp(np.array(truth.sigma)[ph] * u + truth.beta_age * age)
    load = truth.lam * u[:, None] + np.array(truth.tau)[ph][:, None] * rng.normal(size=(m, d))
    Q = truth.q * np.eye(d)
    ys, ts, doses, expos, ons = [], [], [], [], []
    cache = {}
    for i in range(m):
        g = ph[i]
        Ag = truth.A if g == TD else truth.A_pigd
        bg = truth.b[g]
        T = int(rng.choice(vk, p=vp)) if visit_probs else rng.integers(visits[0], visits[1] + 1)
        gaps = 1.0 + rng.uniform(-jitter, jitter, size=T)        # gap 0 is year 0 -> landmark
        times = np.cumsum(gaps)
        X = X0[i].copy()
        t_prev = 0.0
        treated, dose, expo = False, 0.0, 0.0
        y_rows, d_rows, e_rows, o_rows = [], [], [], []
        for k in range(T):
            dt = times[k] - t_prev
            if truth.curvature:
                X = _step(X, Ag, bg, load[i], s[i], truth.q, dt, truth.curvature, rng)
            else:
                E, M, S = transition_np(Ag, Q, s[i], dt)
                w, V = np.linalg.eigh(S)
                X = E @ X + M @ (bg + load[i]) + V @ (np.sqrt(np.maximum(w, 0)) * rng.normal(size=d))
            t_prev = times[k]
            if k > 0 and not treated:
                motor = X[truth.treat_mask].mean()
                treated = rng.random() < 1 / (1 + np.exp(-(start_hazard[0] + start_hazard[1] * motor)))
            if treated:
                dose = min(9.0, 3.0 if dose == 0 else dose + 1.5)
                expo = 0.5 * expo + 0.5 * dose
            on = float(treated and rng.random() < 0.5)
            h = -(truth.kappa[g] * dose / (truth.ed50 + dose) * on
                  + truth.kappa_ldr[g] * expo / (truth.ec50_ldr + expo))
            yk = X + h + np.sqrt(truth.meas_var) * rng.normal(size=d)
            yk[rng.random(d) < p_missing] = np.nan
            y_rows.append(yk)
            d_rows.append(dose)
            e_rows.append(expo)
            o_rows.append(on)
            if k > 0 and rng.random() < 1 / (1 + np.exp(-(dropout[0] + dropout[1] * X[i_ind]))):
                break
        ys.append(np.array(y_rows))
        ts.append(times[:len(y_rows)])
        doses.append(np.array(d_rows))
        expos.append(np.array(e_rows))
        ons.append(np.array(o_rows))
    keep2 = [i for i in range(m) if len(ys[i]) >= 2]
    if target_counts is not None:
        sel = []
        for g, k in zip((TD, PIGD), target_counts):
            idx = [i for i in keep2 if ph[i] == g]
            if len(idx) < k:
                raise ValueError(f"only {len(idx)} participants of phenotype {g} for a target of {k}; raise n")
            sel += list(rng.choice(idx, size=k, replace=False))
        keep2 = sorted(sel)
    panel = Panel([ys[i] for i in keep2], [ts[i] for i in keep2], ph[keep2], z=z[keep2],
                  dose=[doses[i] for i in keep2], exposure=[expos[i] for i in keep2],
                  on=[ons[i] for i in keep2], domains=list(names), meta={"family": "A"},
                  year=[np.arange(1, len(ys[i]) + 1) for i in keep2])
    info = {"speed_ratio": truth.speed_ratio, "n_TD": int((ph[keep2] == TD).sum()),
            "n_PIGD": int((ph[keep2] == PIGD).sum()), "n_excluded_indeterminate": int((~keep).sum())}
    return panel, info


def true_support(truth: Truth) -> np.ndarray:
    return (np.abs(truth.A) > 0) | (np.abs(truth.A_pigd) > 0)

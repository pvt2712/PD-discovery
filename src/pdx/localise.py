"""Localisation: which links differ beyond speed (protocol v0.5, §7.1, §7.3.2 to §7.3.4, §7.3.7).

Everything here except the shared-eigenvector check needs an identifying assumption, because
in the free model a phenotype's speed and its deviations trade off exactly.

- The minimal-L1 rule. Speed is the weighted median of the ratios of the two phenotypes'
  continuous-time entries, weighted by the size of the reference entries (Chen Y 2023;
  Bowden 2016). Entries whose reference value is near zero are left out of the ratios, and a
  sign change always counts as a deviation (it does not vote on the speed). It recovers the
  speed while the entries that deviate in one direction carry less than half of the weight.
- Diagnostics: the speed from all entries, from own rates only, from cross-links only, and the
  mode of the ratios (plurality rule; Guo 2018). Monte Carlo draws of the free fit, together
  with the diagnostics, give the set of speeds the data cannot tell apart; a localised verdict is
  reported only if it holds across that set, otherwise "speed and mechanism not separable".

Deviations are on the reference-speed scale: Delta = A_PIGD / s - A_TD, the free fits' drifts
being on the observed time scale.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, stats

from .data import Panel
from .engine.fit import Fit, _yearly_ols, covariance_checked, fit_model, hessian, null_to, param_names
from .engine.likelihood import Prepared, prepare, value_and_grad
from .engine.model import Layout, ModelSpec, pack_chol
from .multiplicity import bh, by, holm, simultaneous_intervals

CLASSES = ("similar", "different strength beyond speed", "structurally different",
           "different, type unresolved", "unresolved")


# ---------------------------------------------------------------- the minimal-L1 rule

def weighted_median(v, w) -> float:
    v, w = np.asarray(v, float), np.asarray(w, float)
    if len(v) == 0 or w.sum() <= 0:
        return float("nan")
    o = np.argsort(v)
    c = np.cumsum(w[o])
    k = int(np.searchsorted(c, 0.5 * c[-1]))
    return float(v[o][min(k, len(v) - 1)])


def _usable(a0, a1, near_zero):
    return (np.abs(a0) > near_zero) & (a1 / np.where(a0 == 0, 1.0, a0) > 0)


def speed_rule(a0, a1, near_zero: float = 0.05, mask=None) -> float:
    """Minimal-L1 speed from two vectors of support entries (reference a0, other phenotype a1)."""
    a0, a1 = np.asarray(a0, float), np.asarray(a1, float)
    ok = _usable(a0, a1, near_zero)
    if mask is not None:
        ok &= np.asarray(mask, bool)
    return weighted_median(a1[ok] / a0[ok], np.abs(a0[ok]))


def speed_rule_batch(a0, a1, near_zero: float = 0.05) -> np.ndarray:
    """Vectorised speed_rule over rows (Monte Carlo draws): a0, a1 of shape (M, k)."""
    ok = (np.abs(a0) > near_zero) & (a1 * a0 > 0)
    r = np.where(ok, a1 / np.where(ok, a0, 1.0), np.inf)
    w = np.where(ok, np.abs(a0), 0.0)
    o = np.argsort(r, axis=1)
    rs, ws = np.take_along_axis(r, o, 1), np.take_along_axis(w, o, 1)
    c = np.cumsum(ws, 1)
    k = (c < 0.5 * c[:, -1:]).sum(1)
    s = np.take_along_axis(rs, np.minimum(k, r.shape[1] - 1)[:, None], 1)[:, 0]
    return np.where(c[:, -1] > 0, s, np.nan)


def speed_mode(a0, a1, near_zero: float = 0.05) -> float:
    """Mode of the weighted log-ratios (normal kernel, modified Silverman bandwidth; Hartwig 2017)."""
    a0, a1 = np.asarray(a0, float), np.asarray(a1, float)
    ok = _usable(a0, a1, near_zero)
    if ok.sum() == 0:
        return float("nan")
    lr, w = np.log(a1[ok] / a0[ok]), np.abs(a0[ok])
    if ok.sum() == 1:
        return float(np.exp(lr[0]))
    w = w / w.sum()
    sd = np.sqrt(np.sum(w * (lr - np.sum(w * lr)) ** 2))
    mad = stats.median_abs_deviation(lr, scale="normal")
    bw = 0.9 * max(min(sd, mad) if min(sd, mad) > 0 else max(sd, mad), 1e-3) * len(lr) ** -0.2
    grid = np.linspace(lr.min() - 3 * bw, lr.max() + 3 * bw, 2001)
    dens = (w[None, :] * np.exp(-0.5 * ((grid[:, None] - lr[None, :]) / bw) ** 2)).sum(1)
    return float(np.exp(grid[np.argmax(dens)]))


def speed_ls(a0, a1) -> float:
    a0, a1 = np.asarray(a0, float), np.asarray(a1, float)
    return float(a1 @ a0 / max(a0 @ a0, 1e-12))


def speed_diagnostics(a0, a1, spec: ModelSpec, near_zero: float = 0.05) -> dict:
    si = spec.sup_idx
    own = si[:, 0] == si[:, 1]
    return {"all": speed_rule(a0, a1, near_zero), "own_rates": speed_rule(a0, a1, near_zero, own),
            "cross_links": speed_rule(a0, a1, near_zero, ~own), "mode": speed_mode(a0, a1, near_zero),
            "least_squares": speed_ls(a0, a1)}


# ---------------------------------------------------------------- Monte Carlo draws of the free fit

def _natural(raw: np.ndarray, spec: ModelSpec) -> np.ndarray:
    v = np.array(raw, float, copy=True)
    if spec.negative_own_rates:
        v[..., spec.diag_pos] = -np.exp(v[..., spec.diag_pos])
    return v


@dataclass
class FreeDraws:
    a0: np.ndarray        # (k,) estimated reference entries on the support
    a1: np.ndarray        # (k,) estimated entries of the other phenotype (observed time scale)
    d0: np.ndarray        # (M, k) draws of a0
    d1: np.ndarray        # (M, k) draws of a1
    cov: np.ndarray | None = None
    cov_info: dict = field(default_factory=dict)


def free_draws(free_fit: Fit, panel: Panel, M: int = 2000, rng: np.random.Generator | None = None,
               prep: Prepared | None = None, cov: np.ndarray | None = None, group: int = 1) -> FreeDraws:
    """Draws of both phenotypes' drift entries from the free fit's asymptotic normal distribution
    (raw parameters drawn, own rates mapped through -exp, so the draws respect the constraint)."""
    rng = rng or np.random.default_rng()
    lay, spec = free_fit.layout, free_fit.spec
    info = {}
    if cov is None:
        cov, info = covariance_checked(hessian(free_fit, panel, prep), names=param_names(lay))
    s0, s1 = lay.slices["A"][0], lay.slices[f"A{group}"][0]
    idx = np.r_[np.arange(s0.start, s0.stop), np.arange(s1.start, s1.stop)]
    V = 0.5 * (cov[np.ix_(idx, idx)] + cov[np.ix_(idx, idx)].T)
    w, U = np.linalg.eigh(V)
    L = U * np.sqrt(np.maximum(w, 0.0))
    raw = free_fit.x[idx] + rng.normal(size=(M, len(idx))) @ L.T
    k = spec.n_sup
    return FreeDraws(_natural(free_fit.x[s0], spec), _natural(free_fit.x[s1], spec),
                     _natural(raw[:, :k], spec), _natural(raw[:, k:], spec), cov, info)


def _mc_p(est: np.ndarray, draws: np.ndarray) -> np.ndarray:
    """Two-sided Monte Carlo p-values from draws centred on the estimate (as in the toy O3m
    check); NaN where the estimate is undefined."""
    est = np.asarray(est, float)
    M = len(draws)
    p = (1 + (np.abs(draws - est) >= np.abs(est)).sum(0)) / (M + 1)
    return np.where(np.isfinite(est), p, np.nan)


def _basic_ci(est, draws, level):
    a = (1 - level) / 2
    c = draws - est
    return est - np.quantile(c, 1 - a, axis=0), est - np.quantile(c, a, axis=0)


def _opt_kw(kw: dict | None) -> dict:
    return {k: v for k, v in (kw or {}).items() if k in ("maxiter", "chunk", "ftol", "gtol")}


# ---------------------------------------------------------------- per-link and per-domain tests

@dataclass
class SpeedView:
    """Everything a verdict needs at one speed (or with the speed's uncertainty in the draws)."""
    speed: float
    delta: np.ndarray        # (k,) speed-adjusted deviations
    d_draws: np.ndarray      # (M, k) their draws
    a1s: np.ndarray          # (k,) the other phenotype's entries at the common speed
    a1s_draws: np.ndarray
    p: np.ndarray            # (k,) Monte Carlo p-values


@dataclass
class LinkResults:
    links: list                    # (to, from) domain indices of each support entry
    own: np.ndarray                # True for own rates
    s_hat: float
    speed_set: tuple
    diagnostics: dict
    main: SpeedView                # at the minimal-L1 speed, with its uncertainty
    probes: list                   # SpeedView at fixed speeds across the speed set
    a0: np.ndarray
    a0_draws: np.ndarray
    family: np.ndarray             # entries of the multiplicity family (default: cross-links)
    draws: FreeDraws | None = None
    speed_identified: bool = True
    extra: dict = field(default_factory=dict)

    @property
    def delta(self):
        return self.main.delta

    @property
    def p(self):
        return self.main.p

    @property
    def probe_speeds(self):
        return np.array([v.speed for v in self.probes])

    @property
    def ci90(self):
        return _basic_ci(self.main.delta, self.main.d_draws, 0.90)


def per_link_tests(free_fit: Fit, panel: Panel, M: int = 2000, rng: np.random.Generator | None = None,
                   near_zero: float = 0.05, prep: Prepared | None = None, cov: np.ndarray | None = None,
                   family: np.ndarray | None = None, n_probe: int = 7, set_level: float = 0.95,
                   include_diagnostics: bool = True, draws: FreeDraws | None = None) -> LinkResults:
    """Anchor-free per-link tests (§7.3.3): minimal-L1 speed, Monte Carlo p-values (Chen Y 2023).

    The speed set is the central `set_level` range of the Monte Carlo minimal-L1 speeds, widened
    to include the diagnostic speeds; every verdict is re-derived at `n_probe` fixed speeds
    across it (classify_links, per_domain_tests). The rule works on the other phenotype's time
    scale (it minimises sum |a1 - s a0|), as the protocol states it; deviations are then reported
    on the reference scale, Delta = a1 / s - a0. If no reference entry can vote (all near zero or
    sign-changed), the speed is not identified and every p-value is NaN.
    """
    spec = free_fit.spec
    dr = draws or free_draws(free_fit, panel, M, rng, prep, cov)
    si = spec.sup_idx
    own = si[:, 0] == si[:, 1]
    family = ~own if family is None else np.asarray(family, bool)
    s_hat = speed_rule(dr.a0, dr.a1, near_zero)
    diag = speed_diagnostics(dr.a0, dr.a1, spec, near_zero)
    links = [tuple(map(int, r)) for r in si]
    if not np.isfinite(s_hat):
        k = len(dr.a0)
        nan = np.full(k, np.nan)
        view = SpeedView(float("nan"), nan, np.full((1, k), np.nan), nan, np.full((1, k), np.nan), nan)
        return LinkResults(links, own, float("nan"), (float("nan"), float("nan")), diag, view, [], dr.a0,
                           dr.d0, family, dr, speed_identified=False)
    s_star = speed_rule_batch(dr.d0, dr.d1, near_zero)
    ok = np.isfinite(s_star)
    d0, d1, ss = dr.d0[ok], dr.d1[ok], s_star[ok]
    delta = dr.a1 / s_hat - dr.a0
    d_star = d1 / ss[:, None] - d0
    main = SpeedView(float(s_hat), delta, d_star, dr.a1 / s_hat, d1 / ss[:, None], _mc_p(delta, d_star))
    a = (1 - set_level) / 2
    lo, hi = np.quantile(ss, [a, 1 - a])
    if include_diagnostics:
        cand = [v for key, v in diag.items() if key != "least_squares" and np.isfinite(v) and v > 0]
        lo, hi = min([lo] + cand), max([hi] + cand)
    probes = []
    for s in np.unique(np.r_[np.geomspace(lo, hi, n_probe), s_hat]):
        est, ds = dr.a1 / s - dr.a0, dr.d1 / s - dr.d0
        probes.append(SpeedView(float(s), est, ds, dr.a1 / s, dr.d1 / s, _mc_p(est, ds)))
    return LinkResults(links, own, float(s_hat), (float(lo), float(hi)), diag, main, probes, dr.a0, dr.d0[ok],
                       family, dr, extra={"draws_used": int(ok.sum()), "cov_info": dr.cov_info})


def _reject(p, method: str, alpha: float):
    if method == "holm":
        return holm(p, alpha)
    if method == "bh":
        return bh(p, alpha)
    if method == "by":
        return by(p, alpha)
    if method == "none":
        return np.asarray(p) <= alpha
    raise ValueError(method)


def _nonzero(lo, hi):
    return (lo > 0) | (hi < 0)


def _within(lo, hi, eps):
    return (lo > -eps) & (hi < eps)


def _classes_at(v: SpeedView, a0, a0_draws, family, eps, method, alpha) -> list:
    """The link classes of §2.3 at one speed view."""
    k = len(v.delta)
    rej = np.zeros(k, bool)
    rej[family] = _reject(v.p[family], method, alpha)
    sim_lo = np.full(k, np.nan)
    sim_hi = np.full(k, np.nan)
    if family.any():
        sim_lo[family], sim_hi[family] = simultaneous_intervals(v.delta[family], v.d_draws[:, family], 0.90)
    a0l, a0h = _basic_ci(a0, a0_draws, 0.95)
    a1l, a1h = _basic_ci(v.a1s, v.a1s_draws, 0.95)
    e0l, e0h = _basic_ci(a0, a0_draws, 0.90)
    e1l, e1h = _basic_ci(v.a1s, v.a1s_draws, 0.90)
    nz0, nz1 = _nonzero(a0l, a0h), _nonzero(a1l, a1h)
    eq0, eq1 = _within(e0l, e0h, eps), _within(e1l, e1h, eps)
    sign0, sign1 = np.sign(a0l + a0h), np.sign(a1l + a1h)
    out = []
    for i in range(k):
        if not family[i]:
            out.append("outside the family")
        elif rej[i]:
            same = nz0[i] and nz1[i] and sign0[i] == sign1[i]
            opposite = nz0[i] and nz1[i] and sign0[i] != sign1[i]
            if same:
                out.append("different strength beyond speed")
            elif (nz0[i] and eq1[i]) or (nz1[i] and eq0[i]) or opposite:
                out.append("structurally different")
            else:
                out.append("different, type unresolved")
        elif np.isfinite(sim_lo[i]) and sim_lo[i] > -eps and sim_hi[i] < eps:
            out.append("similar")
        else:
            out.append("unresolved")
    return out


def classify_links(res: LinkResults, eps: float, method: str = "by", alpha: float = 0.05,
                   family: np.ndarray | None = None) -> list[dict]:
    """The link classes of §2.3, with the speed-set rule of §7.1.

    Rejections are made within the family (default: the cross-links) by `method`; "similar" for
    an individual link uses its simultaneous 90% interval over the family; the equivalence part
    of "structurally different" uses 90% intervals, the non-zero parts 95% intervals. A class
    other than "unresolved" is reported only if the same class holds at every probe speed across
    the speed set; otherwise the verdict is "speed and mechanism not separable".
    """
    family = res.family if family is None else np.asarray(family, bool)
    k = len(res.links)
    if not res.speed_identified:
        return [{"link": res.links[i], "own_rate": bool(res.own[i]), "class": "speed not identified",
                 "stable_across_speeds": False, "verdict": "speed not identified", "delta": float("nan"),
                 "p": float("nan"), "ci90": [float("nan")] * 2, "max_deviation_allowed": float("nan")}
                for i in range(k)]
    main = _classes_at(res.main, res.a0, res.a0_draws, family, eps, method, alpha)
    at_probes = [_classes_at(v, res.a0, res.draws.d0, family, eps, method, alpha) for v in res.probes]
    lo, hi = res.ci90
    out = []
    for i in range(k):
        cls = main[i]
        claim = cls not in ("unresolved", "outside the family")
        stable = (not claim) or all(c[i] == cls for c in at_probes)
        if claim and cls != "similar":
            signs = {np.sign(v.delta[i]) for v in res.probes} | {np.sign(res.main.delta[i])}
            stable = stable and len(signs) == 1
        out.append({"link": res.links[i], "own_rate": bool(res.own[i]), "class": cls,
                    "stable_across_speeds": bool(stable),
                    "verdict": cls if stable else "speed and mechanism not separable",
                    "delta": float(res.main.delta[i]), "p": float(res.main.p[i]),
                    "ci90": [float(lo[i]), float(hi[i])],
                    "max_deviation_allowed": float(max(abs(lo[i]), abs(hi[i])))})
    return out


def speed_and_load_only(res: LinkResults, eps: float) -> bool:
    """The intersection-union verdict of §2.3: every support entry's 90% interval inside +-eps,
    at the minimal-L1 speed and at every probe speed across the speed set."""
    if not res.speed_identified:
        return False
    views = [res.main] + list(res.probes)
    return bool(all(np.all(_within(*_basic_ci(v.delta, v.d_draws, 0.90), eps)) for v in views))


def _domain_p(v: SpeedView, spec: ModelSpec) -> np.ndarray:
    si = spec.sup_idx
    pv = np.ones(spec.d)
    for j in range(spec.d):
        L = np.where(si[:, 0] == j)[0]
        D = v.d_draws[:, L] - v.delta[L]
        Si = np.linalg.pinv(np.atleast_2d(np.cov(D.T)))
        t_obs = v.delta[L] @ Si @ v.delta[L]
        pv[j] = (1 + (np.einsum("mi,ij,mj->m", D, Si, D) >= t_obs).sum()) / (len(D) + 1)
    return pv


def per_domain_tests(res: LinkResults, spec: ModelSpec, alpha: float = 0.05) -> dict:
    """H-D (§7.3.4): all deviations into one domain at once (own rate included), with a Monte
    Carlo reference for the Mahalanobis statistic, and Benjamini-Yekutieli across domains. A
    domain is reported as differing only if it is rejected at the minimal-L1 speed and at every
    probe speed across the speed set."""
    if not res.speed_identified:
        return {"p": [float("nan")] * spec.d, "verdict": ["speed not identified"] * spec.d}
    p_main = _domain_p(res.main, spec)
    rej_main = by(p_main, alpha)
    rej_all = rej_main.copy()
    for v in res.probes:
        rej_all &= by(_domain_p(v, spec), alpha)
    verdict = ["differs beyond speed" if r else ("speed and mechanism not separable" if m else
                                                  "no difference detected")
               for r, m in zip(rej_all, rej_main)]
    return {"p": p_main.tolist(), "reject_by": rej_main.tolist(), "verdict": verdict}


def recalibrate(p_obs: np.ndarray, p_null: np.ndarray) -> np.ndarray:
    """Recalibrate observed per-link p-values against their null distribution from the H-P
    bootstrap replicates (rows = replicates), as §7.3.3 requires when those are not uniform."""
    p_null = np.asarray(p_null, float)
    return (1 + (p_null <= np.asarray(p_obs)[None, :]).sum(0)) / (len(p_null) + 1)


# ---------------------------------------------------------------- joint layout helpers

def free_to_joint(free_fit: Fit, s: float, group: int = 1) -> np.ndarray:
    """The free fit written in the joint model's parameters at speed s for phenotype `group`."""
    spec = free_fit.spec
    lf, lj = free_fit.layout, Layout(spec, "joint")
    par = free_fit.params
    x = np.zeros(lj.size)
    for name, _ in lj.blocks:
        if name in lf.slices and lf.has(name):
            lj.put(x, name, lf.get(free_fit.x, name))
    gam = np.zeros(spec.n_groups - 1)
    gam[group - 1] = np.log(s)
    lj.put(x, "gamma", gam)
    a0 = par["A"][0][spec.support]
    for g in range(1, spec.n_groups):
        sg = s if g == group else 1.0
        lj.put(x, f"D{g}", par["A"][g][spec.support] / sg - a0)
    b, L = par["b"].copy(), lf.get(free_fit.x, "L").copy()
    b[group] /= s
    L[group] = pack_chol(par["Q"][group] / s)
    lj.put(x, "b", b)
    lj.put(x, "L", L)
    for name in ("gam_y", "gam_z", "lam"):
        if lj.has(name):
            v = lf.get(free_fit.x, name).copy()
            v[group] /= s
            lj.put(x, name, v)
    lt = lf.get(free_fit.x, "log_tau").copy()
    lt[group] -= np.log(s)
    lj.put(x, "log_tau", lt)
    return x


def _d_slice(lay: Layout, group: int = 1) -> slice:
    return lay.slices[f"D{group}"][0]


# ---------------------------------------------------------------- anchored per-link tests

def anchored_tests(panel: Panel, spec: ModelSpec, free_fit: Fit, prep: Prepared | None = None,
                   near_zero: float = 0.05, fit_kw: dict | None = None, alpha: float = 0.05) -> dict:
    """Sensitivity (§7.3.3): Wald tests with every cross-link deviation free and each domain's own
    rate shared (D_jj = 0), which anchors the speed to the own rates."""
    fit_kw = _opt_kw(fit_kw)
    prep = prep or prepare(panel, spec.meas_var, spec.n_groups)
    lay = Layout(spec, "joint")
    si = spec.sup_idx
    own = si[:, 0] == si[:, 1]
    a0, a1 = free_fit.params["A"][0][spec.support], free_fit.params["A"][1][spec.support]
    s0 = speed_rule(a0, a1, near_zero, own)
    x0 = free_to_joint(free_fit, s0 if np.isfinite(s0) else max(speed_ls(a0, a1), 0.1))
    dsl = _d_slice(lay)
    x0[dsl.start + np.where(own)[0]] = 0.0
    bounds = [(None, None)] * lay.size
    for k in np.where(own)[0]:
        bounds[dsl.start + k] = (0.0, 0.0)
    f = fit_model(panel, spec, "joint", x0=x0, prep=prep, bounds=bounds, **fit_kw)
    H = hessian(f, panel, prep)
    keep = np.array([b != (0.0, 0.0) for b in bounds])
    V = np.full((lay.size, lay.size), np.nan)
    V[np.ix_(keep, keep)], cov_info = covariance_checked(H[np.ix_(keep, keep)],
                                                          names=[n for n, k in zip(param_names(lay), keep) if k])
    D = f.x[dsl]
    se = np.sqrt(np.clip(np.diag(V)[dsl], 0.0, None))
    cross = ~own
    z = np.where(cross, D / np.where(se > 0, se, np.nan), np.nan)
    p = np.where(cross, 2 * stats.norm.sf(np.abs(z)), np.nan)
    Vd = V[dsl, dsl]
    p_dom = np.ones(spec.d)
    for j in range(spec.d):
        L = np.where((si[:, 0] == j) & cross)[0]
        if len(L):
            p_dom[j] = stats.chi2.sf(float(D[L] @ np.linalg.solve(Vd[np.ix_(L, L)], D[L])), len(L))
    rej = np.zeros(len(p), bool)
    rej[cross] = by(p[cross], alpha)
    return {"speed": float(np.exp(f.params["log_speed"][1])), "delta": D.tolist(), "se": se.tolist(),
            "p": p.tolist(), "reject_by": rej.tolist(), "p_domain": p_dom.tolist(),
            "converged": f.success, "loglik": f.loglik, "information": cov_info}


# ---------------------------------------------------------------- speed-adjusted joint discovery

def _penalised_fit(lay: Layout, prep: Prepared, z0: np.ndarray, lam: float, w: np.ndarray,
                   fixed_zero: np.ndarray | None = None, maxiter: int = 3000, chunk: int = 512,
                   ftol: float = 1e-10, gtol: float = 1e-5):
    """Adaptive-lasso fit of the joint model with D = u - v (u, v >= 0)."""
    n = lay.size
    dsl = _d_slice(lay)
    k = dsl.stop - dsl.start

    def to_x(z):
        x = z[:n].copy()
        x[dsl] = z[dsl] - z[n:]
        return x

    def f(z):
        val, g = value_and_grad(to_x(z), lay, prep, chunk)
        gz = np.concatenate([g, -g[dsl]])
        gz[dsl] = g[dsl] + lam * w
        gz[n:] += lam * w
        return val + lam * float(w @ (z[dsl] + z[n:])), gz

    bounds = [(None, None)] * n + [(0.0, None)] * k
    for j in range(k):
        bounds[dsl.start + j] = (0.0, None)
    if fixed_zero is not None:
        for j in np.where(fixed_zero)[0]:
            bounds[dsl.start + j] = (0.0, 0.0)
            bounds[n + j] = (0.0, 0.0)
    r = optimize.minimize(f, z0, jac=True, method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": maxiter, "maxfun": 4 * maxiter, "ftol": ftol, "gtol": gtol,
                                   "maxcor": 50})
    x = to_x(r.x)
    nll = value_and_grad(x, lay, prep, chunk)[0]
    return r, x, -nll


def _to_z(x: np.ndarray, lay: Layout) -> np.ndarray:
    dsl = _d_slice(lay)
    D = x[dsl]
    z = np.concatenate([x, np.maximum(-D, 0.0)])
    z[dsl] = np.maximum(D, 0.0)
    return z


def joint_discovery(panel: Panel, spec: ModelSpec, free_fit: Fit, null_fit: Fit,
                    prep: Prepared | None = None, n_lambda: int = 12, ratio: float = 1e-3,
                    near_zero: float = 0.05, n_units: str = "participants", refit: bool = True,
                    fit_kw: dict | None = None, tol: float = 1e-6) -> dict:
    """Speed-adjusted joint discovery (§7.3.2): adaptive lasso on Delta (own rates included,
    speed unpenalised), weights from the free fits at the minimal-L1 speed, the lambda path from
    the smallest value that zeroes every deviation, BIC with the number of participants (or of
    transitions), and an unpenalised refit of the selected support. Two phenotypes (D1 only)."""
    fit_kw = _opt_kw(fit_kw)
    prep = prep or prepare(panel, spec.meas_var, spec.n_groups)
    lay = Layout(spec, "joint")
    dsl = _d_slice(lay)
    a0, a1 = free_fit.params["A"][0][spec.support], free_fit.params["A"][1][spec.support]
    s = speed_rule(a0, a1, near_zero)
    if not np.isfinite(s):
        s = max(speed_ls(a0, a1), 0.1)
    w = 1.0 / np.maximum(np.abs(a1 / s - a0), 1e-4)
    xn = null_to(null_fit.x, spec, "joint")
    g = value_and_grad(xn, lay, prep)[1][dsl]
    lam_max = float(np.max(np.abs(g) / w)) * 1.01
    lams = lam_max * np.geomspace(1.0, ratio, n_lambda)
    n_eff = panel.n if n_units == "participants" else panel.n_transitions()
    path, xs_path = [], []
    z = _to_z(xn, lay)
    for lam in lams:
        r, x, ll = _penalised_fit(lay, prep, z, lam, w, **fit_kw)
        z = r.x
        D = x[dsl]
        active = np.abs(D) > tol
        df = lay.size - spec.n_sup + int(active.sum())
        path.append({"lambda": float(lam), "loglik": ll, "active": active.tolist(), "D": D.tolist(),
                     "bic": float(-2 * ll + np.log(n_eff) * df), "converged": bool(r.success)})
        xs_path.append(x)
    best = int(np.argmin([p["bic"] for p in path]))
    sel = np.array(path[best]["active"])
    # the speed is identified in the refit only if some deviation fixed at zero sits on a
    # reference entry that is clearly non-zero
    anchored = (~sel) & (np.abs(a0) > near_zero)
    out = {"speed_weights": float(s), "lambda_max": lam_max, "path": path, "best": best,
           "selected": sel.tolist(), "identified": bool(anchored.any())}
    if refit and sel.any() and anchored.any():
        xs = xs_path[best].copy()
        xs[dsl] = np.where(sel, xs[dsl], 0.0)
        bounds = [(None, None)] * lay.size
        for j in np.where(~sel)[0]:
            bounds[dsl.start + j] = (0.0, 0.0)
        f = fit_model(panel, spec, "joint", x0=xs, prep=prep, bounds=bounds, **fit_kw)
        out["refit"] = {"D": f.x[dsl].tolist(), "speed": float(np.exp(f.params["log_speed"][1])),
                        "loglik": f.loglik, "converged": f.success}
    return out


def stability_selection(panel: Panel, spec: ModelSpec, free_fit: Fit, null_fit: Fit, B: int = 50,
                        rng: np.random.Generator | None = None, fit_kw: dict | None = None,
                        disc_kw: dict | None = None) -> dict:
    """Selection frequencies over complementary pairs of random halves (Shah & Samworth 2013),
    split within phenotype. Each half refits the free and null models from warm starts."""
    rng = rng or np.random.default_rng()
    fit_kw = _opt_kw(fit_kw)
    disc_kw = dict(disc_kw or {})
    counts = np.zeros(spec.n_sup)
    used = 0
    for _ in range(B // 2):
        perm = [rng.permutation(np.where(panel.group == g)[0]) for g in range(panel.n_groups)]
        halves = [np.concatenate([p[: len(p) // 2] for p in perm]),
                  np.concatenate([p[len(p) // 2:] for p in perm])]
        for idx in halves:
            sub = panel.subset(idx)
            pr = prepare(sub, spec.meas_var, spec.n_groups)
            try:
                f0 = fit_model(sub, spec, "null", x0=null_fit.x, prep=pr, **fit_kw)
                f1 = fit_model(sub, spec, "free", x0=free_fit.x, prep=pr, **fit_kw)
                res = joint_discovery(sub, spec, f1, f0, prep=pr, refit=False, fit_kw=fit_kw, **disc_kw)
            except Exception:
                continue
            counts += np.array(res["selected"], float)
            used += 1
    return {"frequency": (counts / max(used, 1)).tolist(), "halves_used": used,
            "links": [tuple(map(int, r)) for r in spec.sup_idx]}


# ---------------------------------------------------------------- shared-eigenvector check (§7.3.7)

def commutator_test(panel: Panel, B: int = 500, rng: np.random.Generator | None = None) -> dict:
    """Do the phenotypes' yearly matrices commute, as one mechanism at any speeds implies?
    A centred participant bootstrap of the commutator (no continuous-time model)."""
    rng = rng or np.random.default_rng()
    d = panel.d

    def yearly(p: Panel):
        return [_yearly_ols(p, g, d)[0] for g in range(p.n_groups)]

    B0, B1 = yearly(panel)[:2]
    C = B0 @ B1 - B1 @ B0
    t_obs = float(np.linalg.norm(C))
    groups = [np.where(panel.group == g)[0] for g in range(panel.n_groups)]
    ts = []
    for _ in range(B):
        idx = np.concatenate([rng.choice(gi, len(gi)) for gi in groups])
        b0, b1 = yearly(panel.subset(idx))[:2]
        ts.append(float(np.linalg.norm(b0 @ b1 - b1 @ b0 - C)))
    ts = np.array(ts)
    return {"statistic": t_obs, "p": float((1 + (ts >= t_obs).sum()) / (B + 1)), "B": B}


def embeddability_check(panel: Panel) -> dict:
    """§6.3: where a fitted yearly matrix has complex or negative eigenvalues, other drift
    matrices can give the same yearly transitions (Culver 1966); report when this happens."""
    out = {}
    for g in range(panel.n_groups):
        w = np.linalg.eigvals(_yearly_ols(panel, g, panel.d)[0])
        out[str(g)] = {"complex": bool(np.any(np.abs(w.imag) > 1e-8)),
                       "negative_real": bool(np.any((np.abs(w.imag) <= 1e-8) & (w.real <= 0))),
                       "min_real": float(w.real.min())}
    return out

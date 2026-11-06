"""Paper 2's speed modifier: each participant's speed of progression at each of their treatment
decisions, filtered from their own visits up to the decision (escalation-study-v2/PROTOCOL.md §8
and Appendix A, task 8; exploratory).

The model is the engine's speed-only (null) fit on the development participants (as for Sp-1).
Its participant speed is

    log s_i = gamma_g + beta' z_i + theta' y1c_i + sigma_g u_i,      u_i ~ N(0, 1).

For a decision at time tau on the panel's time scale, the speed is the posterior mean and SD of
log s_i given only the participant's panel visits strictly before tau, plus the year-1 landmark when
it is on or before tau. A later visit dated on the decision's own date is left out: its scores come
before the decision, but its dose term may already carry the change made at that visit, and a
modifier must not reflect the decision. The landmark adds no dose term, because it enters only
through its scores. With m visits used (m >= 1: the landmark is the first), the posterior weight of
Gauss-Hermite node u_k is

    w_k  proportional to  (GH weight_k) * p(y_i,2..m | y_i,1, u_k),

where the per-node likelihoods come from the engine's own Kalman filter
(likelihood.participant_loglik(..., per_node=True)) run on a pseudo-participant that holds only
those m visits (truncated_panel). No later visit can move the result. By default the K nodes are
then re-centred on each participant's posterior (adaptive Gauss-Hermite, posterior_u), because K
nodes placed on the prior misplace a posterior much narrower than the prior; adaptive=False keeps
the rule above.

How the year-1 landmark enters (the engine conditions on it; engine/likelihood.py):
- x(1) in the speed is NOT the filtered true state. It is y1c_i, the OBSERVED year-1 scores (on the
  frozen normal-scores scale, a missing domain filled by the phenotype's landmark mean mu1_g),
  centred on the fitted panel's mean of those scores. The same y1c_i moves the load's mean
  (gam_y). The filtered landmark state (_landmark_update of the plug-in prior N(mu1_g, S1_g) with
  y_i(1)) only starts the filter.
- The landmark scores add no likelihood term. A decision before the second visit (landmark only)
  therefore has the prior, N(prior_mean, prior_sd^2), as its posterior, exactly: the K-node rule
  integrates u and u^2 exactly, so the landmark visit informs the speed only through theta' y1c_i,
  which is already in prior_mean.

The plug-in landmark prior (mu1_g, S1_g) and the centre of y1 are fitted quantities of the panel the
engine was fitted on (LandmarkReference), as are the normal-scores tables, the error variances and
the centring of the covariates z: held-out (and truncated) participants are scored with the
development panel's (panel_with_frozen_transform, landmark_reference).

The time scale (pdx.ppmi.annual_grid): a visit's t is date_years = (visit date - anchor) in days /
365.25, where visit dates are PPMI's MM/YYYY set to the 15th, and the anchor is the participant's
enrolment date (ENROLL_DATE, also on the 15th), or the first baseline (BL) visit's date where that
is missing. The landmark (year-1 visit) is therefore at about t = 1. to_panel_time puts Paper 2's
decision dates on this scale.

Participant-level throughout: nothing here prints, and results go only inside the git-ignored data
folder.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import torch

from . import split
from .data import Panel
from .engine.likelihood import LOG_SPEED_BOUND, chunks, gh_nodes, participant_loglik, prepare
from .engine.model import Layout, ModelSpec, unpack_numpy
from .transforms import NormalScores

TIME_TOL = 1e-6          # years (about 30 s): visits within it of the decision count as on its date
OUTPUT = ("n_visits_used", "log_speed_mean", "log_speed_sd", "prior_mean", "prior_sd")


# ---------------------------------------------------------------- the fitted panel's landmark reference

@dataclass
class LandmarkReference:
    """What the engine takes from the panel it was fitted on besides the parameters: the plug-in
    prior of the true landmark state per phenotype, and the centre of the landmark scores."""
    mu1: np.ndarray          # (G, d)
    S1: np.ndarray           # (G, d, d)
    centre: np.ndarray       # (d,): y1c = landmark scores (missing -> mu1_g) - centre

    def as_tuple(self) -> tuple:
        return self.mu1, self.S1, self.centre


def landmark_reference(panel: Panel, meas_var, n_groups: int | None = None) -> LandmarkReference:
    """The landmark reference of the panel a fit was made on, as prepare() builds it for that fit."""
    prep = prepare(panel, meas_var, n_groups)
    mu1 = prep.mu1.numpy()
    y1 = np.array([yi[0] for yi in panel.y])
    y1f = np.where(np.isnan(y1), mu1[panel.group], y1)
    centre = np.nanmean(y1f, 0)                                   # prepare()'s centring, same operations
    if not np.allclose(y1f - centre, prep.y1c.numpy(), rtol=0.0, atol=1e-12):
        raise RuntimeError("prepare() no longer centres the landmark scores on their mean; update landmark_reference")
    return LandmarkReference(mu1, prep.S1.numpy(), centre)


# ---------------------------------------------------------------- truncation and the per-node likelihood

def truncated_panel(panel: Panel, idx, n_visits) -> Panel:
    """Pseudo-participants: for each k, participant idx[k] of the panel with only their first
    n_visits[k] visits (1 = the landmark only); phenotype, covariates and id are the participant's."""
    idx, m = np.asarray(idx, int), np.asarray(n_visits, int)
    if idx.shape != m.shape:
        raise ValueError("idx and n_visits need one entry per pseudo-participant")
    if np.any(m < 1):
        raise ValueError("a pseudo-participant holds at least the landmark visit")
    sub = panel.subset(idx)
    cut = lambda L: [v[:k] for v, k in zip(L, m)]
    return replace(sub, y=cut(sub.y), t=cut(sub.t), dose=cut(sub.dose), exposure=cut(sub.exposure),
                   on=cut(sub.on), year=None if sub.year is None else cut(sub.year))


def _layout(fit, spec: ModelSpec, n_quad: int | None) -> Layout:
    spec = spec if n_quad is None else replace(spec, n_quad=int(n_quad))
    lay = Layout(spec, fit.kind)
    if lay.size != len(fit.x):
        raise ValueError(f"the {fit.kind} fit has {len(fit.x)} parameters, the model {lay.size}: pass the spec "
                         "it was fitted with (with its support)")
    if getattr(fit, "spec", None) is not None and not np.array_equal(fit.spec.support, spec.support):
        raise ValueError("the spec's support is not the fit's")
    return lay


def _per_node(x, lay: Layout, prep, chunk: int | None, nodes: np.ndarray | None = None,
              active: np.ndarray | None = None) -> np.ndarray:
    """(N, K) per-node log-likelihoods of the prepared participants, at the shared Gauss-Hermite nodes
    or at each participant's own `nodes` (N, K); only the `active` ones are evaluated, and
    landmark-only participants keep 0 at every node."""
    ll = torch.zeros(prep.N, lay.spec.n_quad, dtype=torch.float64)
    xt = torch.as_tensor(np.asarray(x, float))
    act = None if active is None else torch.as_tensor(active)
    nd = None if nodes is None else torch.as_tensor(np.array(nodes, float))
    with torch.no_grad():
        for idx in chunks(prep, chunk):
            if act is not None:
                idx = idx[act[idx]]
                if len(idx) == 0:
                    continue
            ll[idx] = participant_loglik(xt, lay, prep, idx, per_node=True, nodes=None if nd is None else nd[idx])
    return ll.numpy()


def node_logliks(fit, panel: Panel, reference: LandmarkReference | None = None, spec: ModelSpec | None = None,
                 n_quad: int | None = None, chunk: int | None = None):
    """log p(y_i,2..T_i | y_i,1, u_k) for every participant (or pseudo-participant) of `panel` at the
    K Gauss-Hermite nodes: returns (ll (N, K), nodes u_k (K,), log weights (K,)), u ~ N(0, 1).

    A participant seen only at the landmark has 0 at every node. `reference` is the
    LandmarkReference of the panel the fit was made on; the default, this panel's own, is right
    only when `panel` is that panel. `spec` defaults to fit.spec; n_quad changes the number of nodes
    (default: the spec's, as in the fit)."""
    spec = spec or fit.spec
    lay = _layout(fit, spec, n_quad)
    ref = reference or landmark_reference(panel, spec.meas_var, spec.n_groups)
    prep = prepare(panel, spec.meas_var, spec.n_groups, landmark=ref.as_tuple())
    U, logw = gh_nodes(lay.spec.n_quad)
    return _per_node(fit.x, lay, prep, chunk), U.numpy(), logw.numpy()


def _prior(x, lay: Layout, prep) -> tuple[np.ndarray, np.ndarray]:
    """Each participant's prior of log-speed given the landmark, as participant_loglik builds it:
    mean gamma_g + beta' z + theta' y1c, SD sigma_g."""
    par = unpack_numpy(x, lay)
    g = prep.g.numpy()
    eta = par["log_speed"][g] + prep.z.numpy() @ par["beta"] + prep.y1c.numpy() @ par["theta"]
    return eta, par["sigma"][g]


def _moments(log_weight: np.ndarray, u: np.ndarray):
    """Mean and SD of u under the normalised weights exp(log_weight), row by row."""
    a = log_weight - log_weight.max(1, keepdims=True)
    w = np.exp(a)
    w /= w.sum(1, keepdims=True)
    m = np.sum(w * u, 1)
    return m, np.sqrt(np.sum(w * (u - m[:, None]) ** 2, 1)), w


def posterior_u(x, lay: Layout, prep, chunk: int | None = None, adaptive: bool = True, tol: float = 1e-4,
                max_rounds: int = 10, first_scale: float = 0.25, min_scale: float = 0.01) -> tuple:
    """Posterior mean and SD of each prepared participant's u (log s = prior_mean + prior_sd * u).

    First the Gauss-Hermite rule on the prior: weights (GH weight_k) * exp(ll_k) at the K nodes.
    With `adaptive`, the K nodes are then re-centred on each participant's current posterior,
    u_k = m + c U_k with c its SD (adaptive Gauss-Hermite; weights GH weight_k * phi(u_k) /
    phi(U_k) * exp(ll(u_k))), until neither moment moves by more than `tol`. The prior rule alone
    resolves a posterior only while it stays wide relative to the node spacing: in a synthetic
    check at 10 domains (6-10 visits), its 20 nodes were exact to 0.01 while the posterior SD stayed
    above about 0.4 of the prior's, but off by up to 0.18-0.39 in log-speed (mean and SD) when it
    fell to 0.1-0.3 of it; re-centred, the same 20 nodes agreed with a dense grid to 0.01
    (tests/test_speed_at.py repeats the check at 4 domains). The first re-centred rule uses a scale
    of at least first_scale (the prior rule cannot place a narrow posterior to better than its node
    spacing), later ones at least min_scale. Landmark-only participants keep the prior (0, 1)
    exactly. Returns (m, sd, info)."""
    K = lay.spec.n_quad
    U, logw = (a.numpy() for a in gh_nodes(K))
    ll = _per_node(x, lay, prep, chunk)
    m, s, w = _moments(ll + logw[None, :], np.broadcast_to(U, ll.shape))
    nodes = np.broadcast_to(U, ll.shape)
    info = {"adaptive": bool(adaptive), "rounds": 0, "not_converged": 0}
    if adaptive:
        active = prep.Ti.numpy() >= 2
        nodes = nodes.copy()
        for r in range(max_rounds):
            if not active.any():
                break
            info["rounds"] = r + 1
            c = np.maximum(s, first_scale if r == 0 else min_scale)
            un = m[:, None] + c[:, None] * U[None, :]
            lw = logw[None, :] + 0.5 * U[None, :] ** 2 - 0.5 * un ** 2
            lln = _per_node(x, lay, prep, chunk, nodes=un, active=active)
            a = np.where(active)[0]
            m2, s2, w2 = _moments(lw[a] + lln[a], un[a])
            moved = np.maximum(np.abs(m2 - m[a]), np.abs(s2 - s[a]))
            m[a], s[a], w[a], nodes[a] = m2, s2, w2, un[a]
            active[a[moved < tol]] = False
        info["not_converged"] = int(active.sum())
    info["weights"], info["nodes"] = w, nodes
    return m, s, info


# ---------------------------------------------------------------- the speed at the decision

def filtered_log_speed(fit, spec: ModelSpec, panel: Panel, cutoffs: pd.DataFrame,
                       reference: LandmarkReference | None = None, n_quad: int | None = None,
                       adaptive: bool = True, chunk: int | None = None) -> pd.DataFrame:
    """The posterior of each participant's log-speed given only their visits up to each cutoff.

    fit: the engine fit (the speed-only fit for Paper 2), with `spec` the model it was fitted with
    (its support included; fit_from_record(rec, spec) takes the same). panel: the participants to
    score, with their PATNO in panel.ids. cutoffs: PATNO and tau, on the panel's time scale
    (to_panel_time); further columns (such as date_t) are carried through. reference: the
    LandmarkReference of the panel the fit was made on (default: `panel`'s own, right only when
    `panel` is that panel; pass the development panel's for held-out participants). n_quad: the
    number of Gauss-Hermite nodes (default: the spec's, as in the fit). adaptive: re-centre the
    nodes on each posterior (posterior_u); False gives the plain rule on the prior.

    Returns the kept cutoffs in their order, with n_visits_used (visits strictly before tau, and the
    landmark when it is on or before tau), log_speed_mean and log_speed_sd, prior_mean (gamma_g + beta' z + theta' y1c) and
    prior_sd (sigma_g). Dropped, and counted in .attrs["counts"]: cutoffs of participants not in the
    panel ("not_in_panel"), without a time ("no_time") and before the landmark ("before_landmark").
    A cutoff before the second visit uses the landmark visit only (posterior = prior)."""
    if panel.ids is None:
        raise ValueError("the panel needs its participant numbers (Panel.ids)")
    if not {"PATNO", "tau"} <= set(cutoffs.columns):
        raise ValueError("cutoffs needs the columns PATNO and tau")
    keys = [split.key(p) for p in panel.ids]
    where = {k: i for i, k in enumerate(keys)}
    if len(where) != len(keys):
        raise ValueError("the panel holds a participant number twice")
    lay = _layout(fit, spec, n_quad)
    ref = reference or landmark_reference(panel, spec.meas_var, spec.n_groups)
    cut = cutoffs.reset_index(drop=True)
    i_of = np.array([where.get(split.key(p), -1) for p in cut.PATNO], int)
    tau = pd.to_numeric(cut.tau, errors="coerce").to_numpy(float)
    found, timed = i_of >= 0, np.isfinite(tau)
    m = np.zeros(len(cut), int)
    for r in np.where(found & timed)[0]:
        ti = panel.t[i_of[r]]
        before = int(np.searchsorted(ti, tau[r] - TIME_TOL, side="left"))        # visits strictly before the decision
        m[r] = max(before, int(ti[0] <= tau[r] + TIME_TOL))                     # the landmark on the decision's date stays
    keep = found & timed & (m >= 1)
    counts = {"cutoffs": int(len(cut)), "not_in_panel": int((~found).sum()), "no_time": int((found & ~timed).sum()),
              "before_landmark": int((found & timed & (m == 0)).sum()), "scored": int(keep.sum()),
              "landmark_only": int((keep & (m == 1)).sum()), "nodes": int(lay.spec.n_quad)}
    rest = [c for c in cut.columns if c not in ("PATNO", "tau") + OUTPUT]
    out = cut.loc[keep, ["PATNO", "tau"] + rest].reset_index(drop=True)
    out["n_visits_used"] = m[keep]
    if keep.any():
        # one pseudo-participant per distinct (participant, visits used): decisions between the same
        # two visits share it
        base = max(len(t) for t in panel.t) + 1
        uniq, inv = np.unique(i_of[keep].astype(np.int64) * base + m[keep], return_inverse=True)
        pp = truncated_panel(panel, uniq // base, uniq % base)
        prep = prepare(pp, spec.meas_var, spec.n_groups, landmark=ref.as_tuple())
        eta, sig = _prior(fit.x, lay, prep)
        mu, su, info = posterior_u(fit.x, lay, prep, chunk, adaptive)
        ls = eta[:, None] + sig[:, None] * info["nodes"]
        if np.any((info["weights"] > 1e-12) & (np.abs(ls) > LOG_SPEED_BOUND)):
            warnings.warn(f"posterior weight at a node beyond the engine's log-speed bound (+-{LOG_SPEED_BOUND}), "
                          "where the likelihood is computed at the bound")
        inv = np.asarray(inv).reshape(-1)
        out["log_speed_mean"], out["log_speed_sd"] = (eta + sig * mu)[inv], (sig * su)[inv]
        out["prior_mean"], out["prior_sd"] = eta[inv], sig[inv]
        counts.update(pseudo_participants=int(len(uniq)), adaptive_rounds=info["rounds"],
                      not_converged=info["not_converged"])
    else:
        for c in OUTPUT[1:]:
            out[c] = np.zeros(0)
        counts.update(pseudo_participants=0, adaptive_rounds=0, not_converged=0)
    out.attrs["counts"] = counts
    return out


# ---------------------------------------------------------------- the panel of other participants, frozen transforms

def load_normal_scores(path) -> NormalScores:
    """The frozen normal-scores tables as scripts/build_analysis_panel.py saves them
    (normal_scores_<set>.npz: values_<j> and scores_<j> for domain j of the spec's domain list)."""
    with np.load(path, allow_pickle=False) as f:
        n = sum(k.startswith("values_") for k in f.files)
        if n == 0 or any(f"values_{j}" not in f.files or f"scores_{j}" not in f.files for j in range(n)):
            raise ValueError(f"{path}: not a normal-scores file (values_<j> and scores_<j> for j = 0, 1, ...)")
        return NormalScores([(f[f"values_{j}"], f[f"scores_{j}"]) for j in range(n)])


def baseline_covariates(part: pd.DataFrame, ids) -> pd.DataFrame:
    """The engine's covariates before centring, as ppmi.analysis_panel builds them: era (1 = PPMI
    2.0), age at onset, male and disease duration, one row per participant of `ids` (NaN kept)."""
    p = part.set_index("PATNO") if "PATNO" in part.columns else part
    return pd.DataFrame({"era": (p.loc[ids, "era"] == "PPMI 2.0").astype(float), "age_onset": p.loc[ids, "age_onset"],
                         "male": p.loc[ids, "male"].astype(float), "duration": p.loc[ids, "duration_years"]})


def panel_with_frozen_transform(part: pd.DataFrame, grid: pd.DataFrame, domains, ns: NormalScores,
                                motor_on: bool = False, z_mean=None) -> Panel:
    """ppmi.analysis_panel for the participants of these tables, with a given normal-scores transform
    (the development panel's, load_normal_scores) instead of one fitted here, and the ids set.

    The same participants enter (primary, TD or PIGD at year 0, seen at the year-1 landmark), with
    the same visits, dose, exposure, examination state and grid years. The covariates are centred on
    z_mean (era, age at onset, male, duration; the development panel's raw means,
    baseline_covariates(...).mean()), a missing value becoming that mean; without z_mean, on these
    participants' own means, exactly as analysis_panel. The error variances are not computed here:
    they are the development spec's."""
    part = part.set_index("PATNO") if "PATNO" in part.columns else part
    domains = list(domains)
    if len(ns.tables) != len(domains):
        raise ValueError(f"the transform has {len(ns.tables)} tables for {len(domains)} domains")
    prim = part[part.primary]
    g = grid[grid.PATNO.isin(prim.index)].copy()
    if motor_on:
        for dname in ("tremor", "brady_rigidity", "axial"):
            on = g.motor_state.eq("ON only")
            g.loc[on, dname] = g.loc[on, dname + "_on"]
    keep = prim[prim.phenotype.isin(["TD", "PIGD"]) & prim.reached_y1].index
    g = g[g.PATNO.isin(keep) & (g.year >= 1)].sort_values(["PATNO", "year"])
    g = g[g.PATNO.map(g.groupby("PATNO").year.min()) == 1]                      # starts at the landmark
    ids = list(dict.fromkeys(g.PATNO))
    if not ids:
        raise ValueError("no participant of these tables enters the panel (primary, TD or PIGD, seen at year 1)")
    z = baseline_covariates(prim, ids)
    if z_mean is None:                        # the panel's own centring, with analysis_panel's operations
        z = z.fillna(z.mean())
        z = (z - z.mean()).to_numpy(float)
    else:
        mean = pd.Series(np.asarray(z_mean, float).reshape(-1), index=z.columns)
        z = (z.fillna(mean) - mean).to_numpy(float)
    y, t, dose, expo, on, yr = [], [], [], [], [], []
    for _, rows in g.groupby("PATNO", sort=False):
        y.append(ns.transform(rows[domains].to_numpy(float)))
        t.append(rows.date_years.to_numpy(float))
        dose.append(rows.ledd.to_numpy(float) / 100.0)
        expo.append(rows.exposure.to_numpy(float) / 100.0)
        on.append((rows.motor_state.eq("ON only") & motor_on).to_numpy(float))
        yr.append(rows.year.to_numpy(int))
    group = (prim.loc[ids, "phenotype"] == "PIGD").astype(int).to_numpy()
    return Panel(y, t, group, z=z, dose=dose, exposure=expo, on=on, domains=domains, ids=np.asarray(ids),
                 meta={"source": "PPMI", "motor_on": motor_on, "normal_scores": "frozen"}, year=yr)


def same_panel(a: Panel, b: Panel, atol: float = 1e-12) -> bool:
    """Whether two panels hold the same participants in the same order (ids compared when both have
    them) with the same phenotypes, covariates, visit times, scores, doses, exposures and states."""
    if a.n != b.n or a.d != b.d or a.z.shape != b.z.shape or not np.array_equal(a.group, b.group):
        return False
    if a.ids is not None and b.ids is not None and [split.key(p) for p in a.ids] != [split.key(p) for p in b.ids]:
        return False
    if not np.allclose(a.z, b.z, rtol=0.0, atol=atol):
        return False
    for name in ("y", "t", "dose", "exposure", "on"):
        for u, v in zip(getattr(a, name), getattr(b, name)):
            if u.shape != v.shape or not np.allclose(u, v, rtol=0.0, atol=atol, equal_nan=True):
                return False
    return True


# ---------------------------------------------------------------- calendar dates on the panel's time scale

def panel_anchor(participants: pd.DataFrame, grid: pd.DataFrame | None = None) -> pd.Series:
    """The date each participant's panel time counts from (keyed by split.key(PATNO)), as ppmi.build
    sets it: the enrolment date, or the first baseline visit's date where that is missing. The grid
    stores this anchor on its rows; when a grid is given, its anchor comes first."""
    part = participants if "PATNO" in participants.columns else participants.reset_index()
    anchor = pd.to_datetime(part.drop_duplicates("PATNO").set_index("PATNO")["enroll_date"])
    if grid is not None and "anchor" in grid.columns:
        ga = grid.dropna(subset=["anchor"]).drop_duplicates("PATNO").set_index("PATNO")["anchor"]
        anchor = pd.to_datetime(ga).combine_first(anchor)
    anchor.index = [split.key(p) for p in anchor.index]
    return anchor


def to_panel_time(dates: pd.DataFrame, participants: pd.DataFrame, grid: pd.DataFrame | None = None) -> pd.Series:
    """Paper 2's decision dates (PATNO and date_t, pandas datetimes; monthly, on the 15th) on the
    panel's time scale: (date_t - anchor) in days / 365.25, as ppmi.annual_grid computes a visit's
    date_years. NaN for a participant without an anchor. Aligned with `dates`' index."""
    anchor = panel_anchor(participants, grid)
    a = pd.to_datetime(pd.Series([anchor.get(split.key(p), pd.NaT) for p in dates.PATNO], index=dates.index))
    return ((pd.to_datetime(dates.date_t) - a).dt.days / 365.25).rename("tau")

"""Marginal likelihood of the engine (protocol v0.5, §7.1).

For each participant the likelihood of the visits after the landmark, conditional on the
landmark (year-1) scores, is

    L_i = sum_k w_k  p(y_i,2..T | y_i,1, s_ik)        (Gauss-Hermite over log-speed)

and p(. | s) is computed exactly by a Kalman filter on the state augmented with the participant
load, [X; l], which integrates out the load, the measurement error and any missing values.
The true year-1 state has a plug-in prior per phenotype (the moments of the year-1 scores, less
the measurement error) and is updated with the year-1 scores, which are conditioned on and do
not enter the likelihood. Missing components are handled with the masked-innovation form, which
gives exactly the likelihood of the observed components.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.polynomial.hermite import hermgauss

from ..data import Panel
from .linalg import bmm
from .model import Layout, drifts, log_speeds, noise_cov, treat_effect
from .transitions import transitions_grouped

LOG2PI = float(np.log(2 * np.pi))
LOG_SPEED_BOUND = 15.0     # |log-speed| at a quadrature node; reached only far from any fitted value
CHUNK = 128                # participants per batch in a process with several threads
CHUNK_SINGLE = 32          # and in a single-threaded worker (one of several parallel jobs)


@dataclass
class Prepared:
    Y: torch.Tensor      # (N, T, d) scores, missing set to 0
    O: torch.Tensor      # (N, T, d) 1 where observed
    dt: torch.Tensor     # (N, T-1) interval to the next visit, 0 for padding
    dose: torch.Tensor   # (N, T)
    expo: torch.Tensor   # (N, T)
    on: torch.Tensor     # (N, T)
    z: torch.Tensor      # (N, p)
    y1c: torch.Tensor    # (N, d) centred landmark scores, missing filled by the phenotype mean
    g: torch.Tensor      # (N,) phenotype
    mu1: torch.Tensor    # (G, d) prior mean of the true landmark state
    S1: torch.Tensor     # (G, d, d) prior covariance of the true landmark state
    R: torch.Tensor      # (d,) measurement-error variances
    N: int
    T: int
    Ti: torch.Tensor     # (N,) visits per participant (chunks are formed by length to skip padding)


def gh_nodes(K: int):
    x, w = hermgauss(K)
    return torch.as_tensor(np.sqrt(2.0) * x), torch.as_tensor(np.log(w / np.sqrt(np.pi)))


def _pd(S: np.ndarray, floor: float = 1e-3) -> np.ndarray:
    w, V = np.linalg.eigh(0.5 * (S + S.T))
    return (V * np.maximum(w, floor)) @ V.T


def prepare(panel: Panel, meas_var: np.ndarray, n_groups: int | None = None) -> Prepared:
    N, d = panel.n, panel.d
    G = n_groups or panel.n_groups
    T = max(len(ti) for ti in panel.t)
    Y = np.zeros((N, T, d))
    O = np.zeros((N, T, d))
    dt = np.zeros((N, T - 1)) if T > 1 else np.zeros((N, 0))
    dose, expo, on = (np.zeros((N, T)) for _ in range(3))
    for i in range(N):
        Ti = len(panel.t[i])
        yi = panel.y[i]
        Y[i, :Ti] = np.nan_to_num(yi)
        O[i, :Ti] = ~np.isnan(yi)
        if Ti > 1:
            dt[i, :Ti - 1] = np.diff(panel.t[i])
        dose[i, :Ti], expo[i, :Ti], on[i, :Ti] = panel.dose[i], panel.exposure[i], panel.on[i]
    y1 = np.array([panel.y[i][0] for i in range(N)])
    mu1 = np.zeros((G, d))
    S1 = np.zeros((G, d, d))
    for g in range(G):
        rows = y1[panel.group == g]
        never = np.where(~np.isfinite(rows).any(0))[0] if len(rows) else np.array([], int)
        if len(never):
            names = getattr(panel, "domains", None) or [f"domain {j}" for j in range(d)]
            raise ValueError(f"phenotype {g}: {', '.join(names[j] for j in never)} never observed at the "
                             f"landmark, so its landmark prior is undefined; drop the domain or merge groups")
        mu1[g] = np.nanmean(rows, 0) if len(rows) else 0.0
        cov = np.ma.cov(np.ma.masked_invalid(rows), rowvar=False).filled(0.0) if len(rows) > 2 else np.eye(d)
        S1[g] = _pd(np.atleast_2d(cov) - np.diag(meas_var))
    y1f = np.where(np.isnan(y1), mu1[panel.group], y1)
    y1c = y1f - np.nanmean(y1f, 0)
    t = lambda a: torch.as_tensor(np.asarray(a, float))
    Ti = torch.as_tensor([len(ti) for ti in panel.t], dtype=torch.long)
    return Prepared(t(Y), t(O), t(dt), t(dose), t(expo), t(on), t(panel.z), t(y1c),
                    torch.as_tensor(panel.group, dtype=torch.long), t(mu1), t(S1), t(meas_var), N, T, Ti)


def default_chunk() -> int:
    """Participants per batch, by this process's threads: small batches stay in the processor's
    cache. On the M4 at 10 domains (scripts/bench_threads.py, 4 November 2026), one 4-thread process
    ran 1.7 times faster with 128 than with 512, and 4 single-threaded jobs about 1.9 times faster
    with 32. The batch size changes only the order of summation."""
    return CHUNK_SINGLE if torch.get_num_threads() == 1 else CHUNK


def chunks(prep: Prepared, size: int | None = None) -> list:
    """Participant index sets of at most `size` (default_chunk() if not given), grouped by number of
    visits. Participants seen only at the landmark are left out: their likelihood is exactly 1 (the
    quadrature weights sum to one), and a batch holding only them would have nothing to
    differentiate. They still inform the landmark prior and the centring in prepare()."""
    size = size or default_chunk()
    order = torch.argsort(prep.Ti, stable=True)
    order = order[prep.Ti[order] >= 2]
    return [order[a:a + size] for a in range(0, len(order), size)]


def _landmark_update(mu, S, y, o, h, R):
    """Posterior of the true landmark state given the landmark scores (no likelihood term)."""
    n, d = y.shape
    I = torch.eye(d, dtype=y.dtype)
    Mo = torch.diag_embed(o)
    v = (y - mu - h) * o
    Sv = Mo @ (S + torch.diag(R)) @ Mo + (I - Mo)
    K = torch.linalg.solve(Sv, (S @ Mo).transpose(-1, -2)).transpose(-1, -2)
    m = mu + (K @ v[..., None])[..., 0]
    P = S - K @ (Mo @ S)
    return m, 0.5 * (P + P.transpose(-1, -2))


def participant_loglik(x: torch.Tensor, lay: Layout, prep: Prepared, idx: torch.Tensor) -> torch.Tensor:
    """(n,) log-likelihood contributions of the participants in idx."""
    spec = lay.spec
    d, K = spec.d, spec.n_quad
    U, logw = gh_nodes(K)
    g = prep.g[idx]
    n = len(idx)
    AG, QG = drifts(x, lay), noise_cov(x, lay)
    b = lay.get(x, "b")[g]
    sig = torch.exp(lay.get(x, "log_sigma"))[g]
    eta = log_speeds(x, lay)[g]
    z, y1c = prep.z[idx], prep.y1c[idx]
    if spec.p:
        eta = eta + z @ lay.get(x, "beta")
    if lay.has("theta"):
        eta = eta + y1c @ lay.get(x, "theta")
    # log-speed at the nodes, bounded so that a wild trial point in a line search stays cheap
    s = torch.exp(torch.clamp(eta[:, None] + sig[:, None] * U[None, :], -LOG_SPEED_BOUND, LOG_SPEED_BOUND))
    mu_l = torch.zeros(n, d, dtype=x.dtype)
    if lay.has("gam_y"):
        mu_l = mu_l + y1c * lay.get(x, "gam_y")[g]
    if lay.has("gam_z"):
        mu_l = mu_l + torch.einsum("np,ndp->nd", z, lay.get(x, "gam_z")[g])
    lam = lay.get(x, "lam")[g] if lay.has("lam") else torch.zeros(n, d, dtype=x.dtype)   # (n, d)
    tau2 = torch.exp(2 * lay.get(x, "log_tau"))[g]                               # (n, d)
    R = prep.R
    Y, O = prep.Y[idx], prep.O[idx]
    h = treat_effect(x, lay, prep.dose[idx], prep.expo[idx], prep.on[idx], g)   # (n, T, d)

    m1, P1 = _landmark_update(prep.mu1[g], prep.S1[g], Y[:, 0], O[:, 0], h[:, 0], R)
    # filter state: X mean, load mean, and the blocks of the joint covariance [[Pxx, Pxl], [Plx, Pll]]
    mX = m1[:, None, :].expand(n, K, d)
    ml = mu_l[:, None, :] + lam[:, None, :] * U[None, :, None]
    Pxx = P1[:, None].expand(n, K, d, d)
    Pxl = torch.zeros(n, K, d, d, dtype=x.dtype)
    Pll = torch.diag_embed(tau2)[:, None].expand(n, K, d, d)
    ll = torch.zeros(n, K, dtype=x.dtype)
    T = int(prep.Ti[idx].max())                            # no work on padding beyond this chunk's longest
    if T < 2:
        return torch.logsumexp(ll + logw[None, :], 1)
    E, M, S = transitions_grouped(AG, QG, g, s[:, :, None] * prep.dt[idx][:, None, :T - 1])  # (n, K, T-1, d, d)
    if lay.has("omega"):                                   # era scales the noise (S is linear in Q)
        S = S * torch.exp(lay.get(x, "omega")[0] * z[:, spec.noise_era_col])[:, None, None, None, None]
    Rm = torch.diag(R)
    tr = lambda a: a.transpose(-1, -2)
    for t in range(1, T):
        Et, Mt, St = E[:, :, t - 1], M[:, :, t - 1], S[:, :, t - 1]
        # predict: X' = E X + M (b + l) + noise; the load is constant
        mX = (Et @ mX[..., None])[..., 0] + (Mt @ (b[:, None, :] + ml)[..., None])[..., 0]
        D = bmm(Et, Pxl)
        C = D + bmm(Mt, Pll)
        Pxx = bmm(bmm(Et, Pxx), tr(Et)) + bmm(C, tr(Mt)) + bmm(Mt, tr(D)) + St
        Pxl = C
        # update with the observed components (masked innovation; exact for the observed part)
        o = O[:, t][:, None, :]                                               # (n, 1, d)
        v = (Y[:, t, None, :] - mX - h[:, t, None, :]) * o
        Sv = (Pxx + Rm) * (o[..., :, None] * o[..., None, :]) + torch.diag_embed(1.0 - o)
        Lc = torch.linalg.cholesky(Sv)
        sol = torch.cholesky_solve(v[..., None], Lc)[..., 0]
        logdet = 2 * torch.log(torch.diagonal(Lc, dim1=-2, dim2=-1)).sum(-1)
        ll = ll - 0.5 * (o.sum(-1) * LOG2PI + logdet + (v * sol).sum(-1))
        Bt = o[..., :, None] * torch.cat([Pxx, Pxl], -1)                      # H P: (n, K, d, 2d)
        KT = torch.cholesky_solve(Bt, Lc)                                     # [Kx'; Kl'] side by side
        Kx, Kl = tr(KT[..., :d]), tr(KT[..., d:])
        mX = mX + (Kx @ v[..., None])[..., 0]
        ml = ml + (Kl @ v[..., None])[..., 0]
        Pxx = Pxx - bmm(Kx, Bt[..., :d])
        Pll = Pll - bmm(Kl, Bt[..., d:])
        Pxl = Pxl - bmm(Kx, Bt[..., d:])
        Pxx = 0.5 * (Pxx + tr(Pxx))
        Pll = 0.5 * (Pll + tr(Pll))
    return torch.logsumexp(ll + logw[None, :], 1)


def loglik(x: torch.Tensor, lay: Layout, prep: Prepared, chunk: int | None = None) -> torch.Tensor:
    """Total log-likelihood (differentiable; holds the whole graph, so use for small data)."""
    tot = torch.zeros((), dtype=x.dtype)
    for idx in chunks(prep, chunk):
        tot = tot + participant_loglik(x, lay, prep, idx).sum()
    return tot


FAILED = 1e10      # the objective value that marks a point where the model cannot be evaluated


def value_and_grad(xnp: np.ndarray, lay: Layout, prep: Prepared, chunk: int | None = None,
                   penalty=None, errors: list | None = None) -> tuple[float, np.ndarray]:
    """Negative log-likelihood and its gradient, chunk by chunk to bound memory.

    penalty(x) may add a differentiable term (used by the penalised localisation fits).
    Returns (FAILED, 0) if the model cannot be evaluated at xnp, and appends the reason to
    `errors` if given. The optimiser's line search backs away from such trial points; a fit whose
    start or end is such a point is marked failed by fit_model, never converged.
    """
    def fail(reason: str):
        if errors is not None:
            errors.append(reason)
        return FAILED, np.zeros(len(xnp))

    x = torch.tensor(np.asarray(xnp, float), requires_grad=True)
    total, grad = 0.0, np.zeros(len(xnp))
    try:
        for idx in chunks(prep, chunk):
            val = -participant_loglik(x, lay, prep, idx).sum()
            if not torch.isfinite(val):
                return fail("non-finite log-likelihood")
            total += float(val.detach())
            if val.requires_grad:              # a batch that does not depend on x adds no gradient
                gr, = torch.autograd.grad(val, x)
                grad += gr.numpy()
        if penalty is not None:
            val = penalty(x)
            gr, = torch.autograd.grad(val, x)
            total += float(val.detach())
            grad += gr.numpy()
    except (RuntimeError, ValueError, ArithmeticError) as e:
        return fail(f"{type(e).__name__}: {str(e)[:200]}")
    if not np.all(np.isfinite(grad)):
        return fail("non-finite gradient")
    return total, grad

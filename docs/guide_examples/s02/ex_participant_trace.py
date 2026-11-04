"""Sections 2.31-2.35: participant_loglik replayed line by line for one synthetic participant.

The lines between the [2.3x] markers are those of src/pdx/engine/likelihood.py (lines 125-194),
for a chunk holding one participant, with prints added. At the end the result is compared with
participant_loglik itself and with the independent reference implementation.
"""
import numpy as np
import torch

from _setup import s1_panel, with_gaps
from helpers import layout, random_point, spec_for
from pdx.engine.likelihood import (LOG2PI, LOG_SPEED_BOUND, _landmark_update, gh_nodes, participant_loglik,
                                   prepare)
from pdx.engine.linalg import bmm
from pdx.engine.model import drifts, log_speeds, noise_cov, treat_effect
from pdx.engine.reference import reference_loglik
from pdx.engine.transitions import transitions_grouped

panel, info = s1_panel(n=60, seed=1, d=4)
panel = with_gaps(panel)
spec = spec_for(panel, info)
prep = prepare(panel, spec.meas_var, spec.n_groups)
lay = layout(spec, "null")
x = torch.as_tensor(random_point(panel, spec, "null", seed=7))
i = next(i for i in range(panel.n)
         if panel.group[i] == 1 and len(panel.t[i]) >= 5 and np.isnan(panel.y[i][1:]).any())
idx = torch.tensor([i])
print(f"participant {i}: phenotype {panel.group[i]}, visit times {panel.t[i].round(3)}")
print("scores (NaN = missing):\n", panel.y[i].round(3))

print("\n[2.31] parameters, speed at the nodes, load prior, treatment, landmark")
d, K = spec.d, spec.n_quad
U, logw = gh_nodes(K)
g = prep.g[idx]
n = len(idx)
AG, QG = drifts(x, lay), noise_cov(x, lay)
b = lay.get(x, "b")[g]
sig = torch.exp(lay.get(x, "log_sigma"))[g]
eta = log_speeds(x, lay)[g]
z, y1c = prep.z[idx], prep.y1c[idx]
bz, ty = (z @ lay.get(x, "beta")).item(), (y1c @ lay.get(x, "theta")).item()
print(f"gamma_g = {eta.item():.4f}, beta'z = {bz:.4f}, theta'y1c = {ty:.4f}", end="")
eta = eta + z @ lay.get(x, "beta")
eta = eta + y1c @ lay.get(x, "theta")
print(f" -> eta = {eta.item():.4f}; sigma_g = {sig.item():.4f}")
s = torch.exp(torch.clamp(eta[:, None] + sig[:, None] * U[None, :], -LOG_SPEED_BOUND, LOG_SPEED_BOUND))
print("speeds s_m at the 12 nodes:", s[0].numpy().round(3))
mu_l = torch.zeros(n, d)
mu_l = mu_l + y1c * lay.get(x, "gam_y")[g]
mu_l = mu_l + torch.einsum("np,ndp->nd", z, lay.get(x, "gam_z")[g])
lam = lay.get(x, "lam")[g]
tau2 = torch.exp(2 * lay.get(x, "log_tau"))[g]
print("load prior: mean without the speed part", mu_l[0].numpy().round(4), "| lambda_g", lam[0].numpy().round(4),
      "| tau_g", tau2[0].sqrt().numpy().round(4))
R = prep.R
Y, O = prep.Y[idx], prep.O[idx]
h = treat_effect(x, lay, prep.dose[idx], prep.expo[idx], prep.on[idx], g)
Ti = int(prep.Ti[i])
print("dose", panel.dose[i].round(2), "| exposure", panel.exposure[i].round(2), "| ON", panel.on[i])
print("treatment terms h (visits x domains):\n", h[0, :Ti].numpy().round(4))
m1, P1 = _landmark_update(prep.mu1[g], prep.S1[g], Y[:, 0], O[:, 0], h[:, 0], R)
print("true landmark state: prior mean", prep.mu1[g][0].numpy().round(3), "-> posterior mean", m1[0].numpy().round(3))
print("                     prior sd  ", prep.S1[g][0].diagonal().sqrt().numpy().round(3), "-> posterior sd  ",
      P1[0].diagonal().sqrt().numpy().round(3))

print("\n[2.32] the filter state [X; l] at every node, and the transitions")
mX = m1[:, None, :].expand(n, K, d)
ml = mu_l[:, None, :] + lam[:, None, :] * U[None, :, None]
Pxx = P1[:, None].expand(n, K, d, d)
Pxl = torch.zeros(n, K, d, d)
Pll = torch.diag_embed(tau2)[:, None].expand(n, K, d, d)
ll = torch.zeros(n, K)
T = int(prep.Ti[idx].max())
c = s[:, :, None] * prep.dt[idx][:, None, :T - 1]
E, M, S = transitions_grouped(AG, QG, g, c)
era = torch.exp(lay.get(x, "omega")[0] * z[:, spec.noise_era_col])
S = S * era[:, None, None, None, None]
node = 6
print("intervals dt", prep.dt[i, :T - 1].numpy().round(3), "| c = s dt at node 6", c[0, node].numpy().round(3))
print("E, M, S:", tuple(E.shape), "| era factor exp(omega * era_i) =", round(era.item(), 4))
print("load prior mean at node 6 (mean + lambda u_6):", ml[0, node].numpy().round(4), f"(u_6 = {U[node]:.4f})")

print("\n[2.33-2.34] predict, then update with the observed components (node 6 shown)")
Rm = torch.diag(R)
tr = lambda a: a.transpose(-1, -2)               # noqa: E731
for t in range(1, T):
    Et, Mt, St = E[:, :, t - 1], M[:, :, t - 1], S[:, :, t - 1]
    m_full = torch.cat([mX, ml], -1)             # kept only for the check below
    P_full = torch.cat([torch.cat([Pxx, Pxl], -1), torch.cat([tr(Pxl), Pll], -1)], -2)
    mX = (Et @ mX[..., None])[..., 0] + (Mt @ (b[:, None, :] + ml)[..., None])[..., 0]
    D = bmm(Et, Pxl)
    C = D + bmm(Mt, Pll)
    Pxx = bmm(bmm(Et, Pxx), tr(Et)) + bmm(C, tr(Mt)) + bmm(Mt, tr(D)) + St
    Pxl = C
    if t == 1:                                   # the block formulas against the full augmented system
        F = torch.zeros(n, K, 2 * d, 2 * d)
        F[..., :d, :d], F[..., :d, d:], F[..., d:, d:] = Et, Mt, torch.eye(d)
        mf = (F @ m_full[..., None])[..., 0]
        mf[..., :d] += (Mt @ b[:, None, :, None])[..., 0]
        Pf = F @ P_full @ tr(F)
        Pf[..., :d, :d] += St
        print(" check against F P F' + [S 0; 0 0]: max |difference| in mean %.1e, Pxx %.1e, Pxl %.1e" % (
            float((mf[..., :d] - mX).abs().max()), float((Pf[..., :d, :d] - Pxx).abs().max()),
            float((Pf[..., :d, d:] - Pxl).abs().max())))
    o = O[:, t][:, None, :]
    v = (Y[:, t, None, :] - mX - h[:, t, None, :]) * o
    Sv = (Pxx + Rm) * (o[..., :, None] * o[..., None, :]) + torch.diag_embed(1.0 - o)
    Lc = torch.linalg.cholesky(Sv)
    sol = torch.cholesky_solve(v[..., None], Lc)[..., 0]
    logdet = 2 * torch.log(torch.diagonal(Lc, dim1=-2, dim2=-1)).sum(-1)
    term = -0.5 * (o.sum(-1) * LOG2PI + logdet + (v * sol).sum(-1))
    ll = ll + term
    print(f" visit {t}: {int(o.sum())} of {d} observed | innovation {v[0, node].numpy().round(3)} | "
          f"log-density term {term[0, node].item():8.4f}")
    Bt = o[..., :, None] * torch.cat([Pxx, Pxl], -1)
    KT = torch.cholesky_solve(Bt, Lc)
    Kx, Kl = tr(KT[..., :d]), tr(KT[..., d:])
    mX = mX + (Kx @ v[..., None])[..., 0]
    ml = ml + (Kl @ v[..., None])[..., 0]
    Pxx = Pxx - bmm(Kx, Bt[..., :d])
    Pll = Pll - bmm(Kl, Bt[..., d:])
    Pxl = Pxl - bmm(Kx, Bt[..., d:])
    Pxx = 0.5 * (Pxx + tr(Pxx))
    Pll = 0.5 * (Pll + tr(Pll))
print("load at node 6 after the last visit: mean", ml[0, node].numpy().round(4), "sd",
      Pll[0, node].diagonal().sqrt().numpy().round(4))

print("\n[2.35] Gauss-Hermite in log space")
print("log p(y_2..T | y_1, u_m):", ll[0].numpy().round(3))
print("posterior weight of each node, w_m p_m / sum:", torch.softmax(ll + logw[None, :], 1)[0].numpy().round(4))
total = torch.logsumexp(ll + logw[None, :], 1)
print(f"log L_i (logsumexp)        = {total.item():.10f}")
print(f"participant_loglik         = {participant_loglik(x, lay, prep, idx).item():.10f}")
ref = reference_loglik(x.numpy(), lay, panel.subset([i]), prep.mu1.numpy(), prep.S1.numpy(), prep.y1c.numpy()[[i]])
print(f"reference_loglik           = {ref:.10f}")
print(f"log of the plain sum of w_m exp(ll_m) = {np.log(np.sum(np.exp(logw.numpy() + ll[0].numpy()))):.10f}")
print(f"the plain sum if the terms were 50 times larger: {np.sum(np.exp(logw.numpy() + 50 * ll[0].numpy()))} ; "
      f"logsumexp: {torch.logsumexp(50 * ll + logw[None, :], 1).item():.4f}")

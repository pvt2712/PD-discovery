"""Exact transitions of the linear continuous-time model over an interval.

For speed s, interval dt, drift A and diffusion Q (per unit time at the reference speed), with
c = s * dt:

    E(c) = exp(c A)                                   transition matrix
    M(c) = integral_0^c exp(A w) dw                   multiplies the drift input (b + l)
    S(c) = integral_0^c exp(A w) Q exp(A' w) dw       noise covariance

All three depend on c only and compose exactly over consecutive intervals (they commute, being
functions of one A):

    E(a + b) = E(a) E(b),   M(a + b) = M(a) + E(a) M(b),   S(a + b) = S(a) + E(a) S(b) E(a)'.

``transitions_grouped`` computes them for many values of c per phenotype without one matrix
exponential per value: c is split as n_H * H + m * h + delta with H = 32 h and
h = 0.125 / ||A||_1; the n_H coarse steps are composed from binary powers (so the cost grows with
log c), the m fine steps from a table, and the remainder uses Taylor series of order 12
(truncation below 1e-18). For S there are two exact routes:
- when A is clearly stable (largest real eigenvalue below -1e-3 ||A||_1), the stationary
  covariance: S = Sigma - E Sigma E' with A Sigma + Sigma A' = -Q, accurate at any c there;
- otherwise composition of positive semi-definite pieces, which needs no stability of A.
``transitions`` (one matrix exponential per value, with Van Loan's block for S) is kept as the
check.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy import linalg

from .linalg import bmm

TAYLOR_ORDER = 12
STABLE_MARGIN = 1e-3
_FACT = [math.factorial(j) for j in range(TAYLOR_ORDER + 2)]


def _tr(a: torch.Tensor) -> torch.Tensor:
    return a.transpose(-1, -2)


def _taylor(A: torch.Tensor, Q: torch.Tensor | None, delta: torch.Tensor):
    """E, M and (if Q is given) S over the intervals delta (...,) by Taylor series; each (..., d, d)."""
    d = A.shape[-1]
    pows = [torch.eye(d, dtype=A.dtype)]
    for _ in range(TAYLOR_ORDER):
        pows.append(pows[-1] @ A)                          # A^k
    P = torch.stack(pows)                                  # (J, d, d)
    k = torch.arange(TAYLOR_ORDER + 1, dtype=A.dtype)
    fact = torch.tensor(_FACT, dtype=A.dtype)
    dk = delta[..., None] ** k                             # (..., J)
    w1 = dk * delta[..., None] / fact[1:]
    E = torch.einsum("...j,jab->...ab", dk / fact[:-1], P)
    M = torch.einsum("...j,jab->...ab", w1, P)
    if Q is None:
        return E, M, None
    lyap = [Q]
    for _ in range(TAYLOR_ORDER):
        lyap.append(A @ lyap[-1] + lyap[-1] @ A.T)         # L^k(Q), L(X) = A X + X A'
    S = torch.einsum("...j,jab->...ab", w1, torch.stack(lyap))
    return E, M, S


def _compose(x, y):
    """(E, M, S) over interval a, then over b (S may be None throughout)."""
    E1, M1, S1 = x
    E2, M2, S2 = y
    S = None if S1 is None else S1 + bmm(bmm(E1, S2), _tr(E1))
    return bmm(E1, E2), M1 + bmm(E1, M2), S


def _grid(A: torch.Tensor, Q: torch.Tensor | None, c: torch.Tensor):
    """E(c), M(c) and, if Q is given, S(c) for one drift A (d, d) and any shape of c >= 0."""
    d = A.shape[-1]
    norm = float(torch.linalg.matrix_norm(A.detach(), ord=1))
    h = 0.125 / max(norm, 1e-3)
    nh = 32
    H = nh * h
    I = torch.eye(d, dtype=A.dtype)
    Z = torch.zeros(d, d, dtype=A.dtype)
    withS = Q is not None
    step = _taylor(A, Q, torch.tensor(h, dtype=A.dtype))
    fine = [(I, Z, Z if withS else None)]
    for _ in range(nh):                                    # fine table: m h, m = 0..nh
        fine.append(_compose(fine[-1], step))
    big = fine[-1]                                         # interval H
    Pf = torch.stack([f[0] for f in fine[:nh]])
    Mf = torch.stack([f[1] for f in fine[:nh]])
    Sf = torch.stack([f[2] for f in fine[:nh]]) if withS else None
    cd = c.detach()
    cmax = float(cd.max()) if c.numel() else 0.0
    if not math.isfinite(cmax):
        raise ValueError("non-finite interval")
    iH = torch.floor(cd / H).long()
    rem = c - iH.to(c.dtype) * H
    ih = torch.clamp(torch.floor(rem.detach() / h).long(), 0, nh - 1)
    delta = rem - ih.to(c.dtype) * h
    shape = (*c.shape, d, d)
    state = (I.expand(shape), Z.expand(shape), Z.expand(shape) if withS else None)
    nbits = int(iH.max()).bit_length() if c.numel() else 0
    power = big
    for b in range(nbits):                                 # coarse steps from binary powers of H
        sel = (((iH >> b) & 1) == 1)[..., None, None]
        if bool(sel.any()):
            new = _compose(state, tuple(None if p is None else p.expand(shape) for p in power))
            state = tuple(None if n is None else torch.where(sel, n, o) for n, o in zip(new, state))
        if b + 1 < nbits:
            power = _compose(power, power)
    Ed, Md, Sd = _taylor(A, Q, delta)
    fine_part = _compose((Pf[ih], Mf[ih], Sf[ih] if withS else None), (Ed, Md, Sd))
    E, M, S = _compose(state, fine_part)
    return E, M, (None if S is None else 0.5 * (S + _tr(S)))


def lyapunov(A: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """Solve A X + X A' = -Q for X (one d x d system) via the Kronecker sum."""
    d = A.shape[-1]
    I = torch.eye(d, dtype=A.dtype)
    K = torch.kron(I, A) + torch.kron(A, I)
    X = torch.linalg.solve(K, -Q.T.reshape(d * d, 1))[:, 0].reshape(d, d).T
    return 0.5 * (X + X.T)


def is_clearly_stable(A: torch.Tensor) -> bool:
    w = torch.linalg.eigvals(A.detach())
    norm = float(torch.linalg.matrix_norm(A.detach(), ord=1))
    return float(w.real.max()) < -STABLE_MARGIN * max(norm, 1e-12)


def transitions_grouped(A: torch.Tensor, Q: torch.Tensor, g: torch.Tensor, c: torch.Tensor,
                        method: str = "auto"):
    """Exact transitions for participants in several phenotypes.

    A, Q: (G, d, d) per phenotype; g: (n,) phenotype of each participant; c: (n, K, T1) speed x
    interval. Returns E, M, S of shape (n, K, T1, d, d). method "auto" uses the stationary
    covariance for S when A is clearly stable and composition otherwise; "lyapunov" and
    "composition" force one route (for checks).
    """
    d = A.shape[-1]
    E = torch.zeros(*c.shape, d, d, dtype=A.dtype)
    M = torch.zeros_like(E)
    S = torch.zeros_like(E)
    for k in range(A.shape[0]):
        sel = (g == k).nonzero(as_tuple=True)[0]
        if len(sel) == 0:
            continue
        stable = method == "lyapunov" or (method == "auto" and is_clearly_stable(A[k]))
        if stable:
            Ek, Mk, _ = _grid(A[k], None, c[sel])
            Sig = lyapunov(A[k], Q[k])
            Sk = Sig - bmm(Ek @ Sig, _tr(Ek))
            Sk = 0.5 * (Sk + _tr(Sk))
        else:
            Ek, Mk, Sk = _grid(A[k], Q[k], c[sel])
        E = E.index_copy(0, sel, Ek)
        M = M.index_copy(0, sel, Mk)
        S = S.index_copy(0, sel, Sk)
    return E, M, S


def transitions(A: torch.Tensor, Q: torch.Tensor, sdt: torch.Tensor):
    """Reference version: matrix exponentials per value, S from Van Loan's block (accurate for
    moderate c). A, Q: (n, d, d); sdt: (n, K, T1)."""
    d = A.shape[-1]
    I = torch.eye(d, dtype=A.dtype)
    sc = sdt[..., None, None]
    Ab, Qb = A[:, None, None], Q[:, None, None]
    top = torch.cat([Ab * sc, I * sc], -1)
    F1 = torch.linalg.matrix_exp(torch.cat([top, torch.zeros_like(top)], -2))
    E, M = F1[..., :d, :d], F1[..., :d, d:]
    blk = torch.cat([torch.cat([-Ab * sc, Qb * sc], -1),
                     torch.cat([torch.zeros_like(top[..., :d]), _tr(Ab) * sc], -1)], -2)
    F2 = torch.linalg.matrix_exp(blk)
    S = _tr(F2[..., d:, d:]) @ F2[..., :d, d:]
    return E, M, 0.5 * (S + _tr(S))


def transition_np(A: np.ndarray, Q: np.ndarray, s: float, dt: float):
    """One speed and interval, in numpy (used by the simulators): E and M from one block matrix
    exponential, S by scaling and squaring of the composition rule, valid for any A."""
    d = A.shape[0]
    c = s * dt
    B1 = np.zeros((2 * d, 2 * d))
    B1[:d, :d], B1[:d, d:] = c * A, c * np.eye(d)
    F1 = linalg.expm(B1)
    E, M = F1[:d, :d], F1[:d, d:]
    k = max(0, int(np.ceil(np.log2(max(c * np.abs(A).sum(0).max() / 0.25, 1.0)))))
    a = c / 2 ** k
    Es = linalg.expm(a * A)
    Lk, Sa = Q.copy(), np.zeros((d, d))
    for j in range(TAYLOR_ORDER + 1):
        Sa += a ** (j + 1) / _FACT[j + 1] * Lk
        Lk = A @ Lk + Lk @ A.T
    for _ in range(k):
        Sa = Sa + Es @ Sa @ Es.T
        Es = Es @ Es
    return E, M, 0.5 * (Sa + Sa.T)

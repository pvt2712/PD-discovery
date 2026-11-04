"""Section 2.17: _grid splits c into coarse binary powers, a fine table and a Taylor remainder."""
import math
import time

import numpy as np
import torch
from scipy.linalg import expm, solve_continuous_lyapunov

import _setup                                    # noqa: F401
from pdx.engine.transitions import _grid

A = torch.tensor([[-0.25, 0.00], [0.20, -0.50]])
Q = torch.tensor([[0.10, 0.02], [0.02, 0.08]])
An, Qn = A.numpy(), Q.numpy()
Sig = solve_continuous_lyapunov(An, -Qn)                     # stationary covariance (scipy)
h = 0.125 / float(torch.linalg.matrix_norm(A, ord=1))
H = 32 * h
print(f"h = {h}, H = 32 h = {H}")
print("      c     iH  ih    delta  nbits   |E-expm|   |M-M_scipy|   |S-(Sig-E Sig E')|")
for c in (0.0, 0.1, 0.3, 7.5, 8.0, 60.0, 5000.0):
    iH = math.floor(c / H)
    rem = c - iH * H
    ih = min(max(math.floor(rem / h), 0), 31)
    delta = rem - ih * h
    E, M, S = _grid(A, Q, torch.tensor([c]))
    En = expm(c * An)
    B1 = np.zeros((4, 4)); B1[:2, :2], B1[:2, 2:] = c * An, c * np.eye(2)
    Mn = expm(B1)[:2, 2:]
    Sn = Sig - En @ Sig @ En.T
    print(f"{c:7.1f} {iH:6d} {ih:3d} {delta:8.4f} {iH.bit_length():5d}   {np.abs(E[0].numpy() - En).max():.1e}"
          f"     {np.abs(M[0].numpy() - Mn).max():.1e}       {np.abs(S[0].numpy() - Sn).max():.1e}")

# the derivative with respect to c flows only through delta, and is exact: dE/dc = A E(c)
c = torch.tensor([7.5], requires_grad=True)
E, _, _ = _grid(A, Q, c)
g, = torch.autograd.grad(E[0].sum(), c)
print(f"\nd/dc sum(E(c)) at c = 7.5: autograd {float(g):.12f}, A expm(cA) summed {float((An @ expm(7.5 * An)).sum()):.12f}")

# the cost grows with log c: 2000 values of c at each size
print("\n    c_max   nbits   time for 2000 values (E, M and S)")
for cmax in (1e1, 1e3, 1e5, 1e7):
    cs = torch.linspace(0.0, cmax, 2000)
    t0 = time.perf_counter()
    _grid(A, Q, cs)
    print(f"{cmax:9.0e} {int(cmax / H).bit_length():6d}   {time.perf_counter() - t0:.4f} s")

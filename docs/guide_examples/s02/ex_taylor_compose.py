"""Sections 2.13-2.16: E, M, S over a short interval by Taylor series, and the composition rule."""
import math

import numpy as np
import torch
from scipy.linalg import expm

import _setup                                    # noqa: F401
from pdx.engine.transitions import _FACT, TAYLOR_ORDER, _compose, _taylor

print("_FACT =", _FACT[:4], "...", _FACT[-1], f"({len(_FACT)} entries: 0! to {TAYLOR_ORDER + 1}!)")
print("first omitted term, E and M, delta ||A||_1 <= 0.125:   0.125^13 / 13! = %.2e" % (0.125 ** 13 / math.factorial(13)))
print("first omitted term, S, (||A||_1 + ||A||_inf) delta <= 0.25: 0.25^13 / 14! = %.2e" % (0.25 ** 13 / math.factorial(14)))
print("the same if ||A||_inf = 5 ||A||_1 (0.75):               0.75^13 / 14! = %.2e\n" % (0.75 ** 13 / math.factorial(14)))

A = torch.tensor([[-0.25, 0.00], [0.20, -0.50]])
Q = torch.tensor([[0.10, 0.02], [0.02, 0.08]])
An, Qn = A.numpy(), Q.numpy()


def scipy_EMS(c):
    """E and M from one block exponential, S from Van Loan's block (scipy), for moderate c."""
    B1 = np.zeros((4, 4)); B1[:2, :2], B1[:2, 2:] = c * An, c * np.eye(2)
    F1 = expm(B1)
    B2 = np.zeros((4, 4)); B2[:2, :2], B2[:2, 2:], B2[2:, 2:] = -c * An, c * Qn, c * An.T
    F2 = expm(B2)
    return F1[:2, :2], F1[:2, 2:], F2[2:, 2:].T @ F2[:2, 2:]


norm1 = float(torch.linalg.matrix_norm(A, ord=1))
h = 0.125 / norm1
print(f"||A||_1 = {norm1}, fine step h = 0.125 / ||A||_1 = {h}, Taylor order {TAYLOR_ORDER}")
E, M, S = _taylor(A, Q, torch.tensor(h))
Es, Ms, Ss = scipy_EMS(h)
print("E(h):\n", E.numpy(), "\nM(h):\n", M.numpy(), "\nS(h):\n", S.numpy())
print("max |Taylor - scipy|: E %.1e  M %.1e  S %.1e" % (np.abs(E.numpy() - Es).max(), np.abs(M.numpy() - Ms).max(),
                                                        np.abs(S.numpy() - Ss).max()))
print("first terms of S: delta*Q + delta^2/2 * (AQ + QA') =\n", (h * Qn + h ** 2 / 2 * (An @ Qn + Qn @ An.T)))

# composition: (E, M, S) over a, then over b, equals the triple over a + b (and the order does not matter)
a, b = 0.25, 0.10
ab = _compose(_taylor(A, Q, torch.tensor(a)), _taylor(A, Q, torch.tensor(b)))
ba = _compose(_taylor(A, Q, torch.tensor(b)), _taylor(A, Q, torch.tensor(a)))
direct = _taylor(A, Q, torch.tensor(a + b))
for name, u, v, w in zip("EMS", ab, ba, direct):
    print(f"{name}: |compose(a,b) - direct(a+b)| = {float((u - w).abs().max()):.1e}, "
          f"|compose(a,b) - compose(b,a)| = {float((u - v).abs().max()):.1e}")

# a drift with one heavily fed domain: column sums small, row sums large (||A||_inf = 5 ||A||_1)
from scipy.linalg import solve_continuous_lyapunov          # noqa: E402

from pdx.engine.transitions import transitions_grouped      # noqa: E402

d = 10
Ah = -0.25 * np.eye(d)
Ah[4, :] += 0.25
Ah[4, 4] = -0.25
Qh = 0.1 * np.eye(d)
print(f"\nrow-heavy drift: ||A||_1 = {np.abs(Ah).sum(0).max()}, ||A||_inf = {np.abs(Ah).sum(1).max()}")
Sig_h = solve_continuous_lyapunov(Ah, -Qh)
for c in (0.3, 1.0, 5.0, 20.0):
    _, _, Sc = transitions_grouped(torch.as_tensor(Ah)[None], torch.as_tensor(Qh)[None],
                                   torch.zeros(1, dtype=torch.long), torch.tensor([[[c]]]), method="composition")
    Ec = expm(c * Ah)
    print(f"  c = {c:4}: |S composition - (Sigma - E Sigma E')| = {np.abs(Sc[0, 0, 0].numpy() - (Sig_h - Ec @ Sig_h @ Ec.T)).max():.1e}")

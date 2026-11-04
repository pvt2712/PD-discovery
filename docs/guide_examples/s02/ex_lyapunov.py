"""Sections 2.18-2.19: the Kronecker Lyapunov solve, and the test that chooses the route for S."""
import numpy as np
import torch
from scipy.linalg import expm, solve_continuous_lyapunov

import _setup                                    # noqa: F401
from pdx.engine.transitions import STABLE_MARGIN, _grid, is_clearly_stable, lyapunov

A = torch.tensor([[-0.25, 0.00], [0.20, -0.50]])
Q = torch.tensor([[0.10, 0.02], [0.02, 0.08]])
X = lyapunov(A, Q)
print("Sigma (Kronecker solve):\n", X.numpy())
print("|Sigma - scipy|:", np.abs(X.numpy() - solve_continuous_lyapunov(A.numpy(), -Q.numpy())).max(),
      "| residual |A Sigma + Sigma A' + Q|:", float((A @ X + X @ A.T + Q).abs().max()))
I = torch.eye(2)
K = torch.kron(I, A) + torch.kron(A, I)
print("Kronecker sum K = I (x) A + A (x) I:\n", K.numpy())
print("eigenvalues of K (all sums of two eigenvalues of A):", np.sort(np.linalg.eigvals(K.numpy()).real))

print(f"\nis_clearly_stable: largest real eigenvalue < -{STABLE_MARGIN} * ||A||_1")
cases = {
    "stable (-0.25, -0.5)": [[-0.25, 0.0], [0.20, -0.50]],
    "max eig -6e-4, ||A||_1 = 0.5": [[-0.0006, 0.0], [0.30, -0.50]],
    "max eig -4e-4, ||A||_1 = 0.5": [[-0.0004, 0.0], [0.30, -0.50]],
    "marginal (0, -0.2)": [[-0.1, 0.1], [0.1, -0.1]],
    "unstable (+0.1, -0.5)": [[0.1, 0.0], [0.2, -0.5]],
}
for name, M in cases.items():
    Am = torch.tensor(M)
    w = np.linalg.eigvals(np.array(M)).real.max()
    Km = torch.kron(I, Am) + torch.kron(Am, I)
    cond = np.linalg.cond(Km.numpy())
    print(f"  {name:30s} max eig {w:+.4f}  threshold {-STABLE_MARGIN * float(torch.linalg.matrix_norm(Am, ord=1)):+.1e}"
          f"  clearly stable: {str(is_clearly_stable(Am)):5s}  cond(K) {cond:.1e}")

# just inside the margin, the stationary route still agrees with composition at small and large c
An = torch.tensor(cases["max eig -6e-4, ||A||_1 = 0.5"])
Sg = lyapunov(An, Q)
for c in (0.01, 1.0, 100.0):
    E, _, S = _grid(An, Q, torch.tensor([c]))
    S_stat = Sg - E[0] @ Sg @ E[0].T
    print(f"  c = {c:6}: |S stationary - S composition| = {float((S_stat - S[0]).abs().max()):.1e} (|S| = {float(S[0].abs().max()):.3g})")

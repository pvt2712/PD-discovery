"""Section 2.21: Van Loan's block exponential loses S at large c; the engine's routes do not."""
import numpy as np
import torch
from scipy.linalg import expm, solve_continuous_lyapunov

import _setup                                    # noqa: F401
from pdx.engine.transitions import transitions, transitions_grouped

rng = np.random.default_rng(0)
cases = {
    "dense 4 x 4": rng.normal(scale=0.3, size=(4, 4)) - 1.2 * np.eye(4),
    "triangular 2 x 2": np.array([[-0.25, 0.0], [0.20, -0.50]]),
}
for name, An in cases.items():
    d = len(An)
    Qn = 0.1 * np.eye(d)
    A, Q = torch.as_tensor(An)[None], torch.as_tensor(Qn)[None]
    Sig = solve_continuous_lyapunov(An, -Qn)
    w = np.linalg.eigvals(An)
    print(f"{name}: eigenvalues {np.round(w, 3)}")
    print("      c   |S| (true)   composition   stationary   Van Loan (torch matrix_exp)")
    for c in (1, 10, 30, 50, 100, 200, 600, 3000):
        cc = torch.tensor([[[float(c)]]])
        _, _, Sc = transitions_grouped(A, Q, torch.zeros(1, dtype=torch.long), cc, method="composition")
        _, _, Sl = transitions_grouped(A, Q, torch.zeros(1, dtype=torch.long), cc, method="lyapunov")
        _, _, Sv = transitions(A, Q, cc)
        En = expm(c * An)
        St = Sig - En @ Sig @ En.T
        err = lambda S: np.abs(S[0, 0, 0].numpy() - St).max()
        print(f"{c:7d}   {np.abs(St).max():.3e}     {err(Sc):.1e}      {err(Sl):.1e}     {err(Sv):.1e}")

"""Section 2.20: E, M and S for a 2 x 2 drift, against scipy's expm and solve_continuous_lyapunov."""
import numpy as np
import torch
from scipy.linalg import expm, solve_continuous_lyapunov

import _setup                                    # noqa: F401
from pdx.engine.transitions import transition_np, transitions, transitions_grouped

A = torch.tensor([[-0.25, 0.00], [0.20, -0.50]])
Q = torch.tensor([[0.10, 0.02], [0.02, 0.08]])
An, Qn = A.numpy(), Q.numpy()
s, dt = 1.5, 1.0                                 # speed 1.5, one year: c = 1.5
c = s * dt
E, M, S = transitions_grouped(A[None], Q[None], torch.zeros(1, dtype=torch.long), torch.tensor([[[c]]]))
E, M, S = E[0, 0, 0].numpy(), M[0, 0, 0].numpy(), S[0, 0, 0].numpy()
Sig = solve_continuous_lyapunov(An, -Qn)
En = expm(c * An)
B1 = np.zeros((4, 4)); B1[:2, :2], B1[:2, 2:] = c * An, c * np.eye(2)
Mn = expm(B1)[:2, 2:]
print("c = s * dt =", c)
print("E:\n", E, "\nexpm(cA):\n", En)
print("M:\n", M, "\nA^{-1}(E - I):\n", np.linalg.solve(An, En - np.eye(2)))
print("S:\n", S, "\nSigma - E Sigma E' (scipy):\n", Sig - En @ Sig @ En.T)
print("max |differences|: E %.1e, M %.1e (block expm), S %.1e" % (np.abs(E - En).max(), np.abs(M - Mn).max(),
                                                                  np.abs(S - (Sig - En @ Sig @ En.T)).max()))

# the shapes: three participants, two of phenotype 0 and one of phenotype 1; two nodes; two intervals
AG = torch.stack([A, 2.0 * A])                   # phenotype 1 runs the same drift twice as fast
QG = torch.stack([Q, 2.0 * Q])
g = torch.tensor([0, 1, 0])
cc = torch.tensor([[[0.5, 1.0], [1.5, 3.0]], [[0.5, 1.0], [1.5, 3.0]], [[0.25, 2.0], [7.5, 60.0]]])
E3, M3, S3 = transitions_grouped(AG, QG, g, cc)
print("\nshapes:", tuple(E3.shape), tuple(M3.shape), tuple(S3.shape))
print("participant 1 (phenotype 1, drift 2A) at c = 0.5 equals phenotype 0 at c = 1.0:",
      float((E3[1, 0, 0] - E3[0, 0, 1]).abs().max()))

# both routes for S, and a marginal drift (eigenvalues 0 and -0.2) that only composition can handle
E1, M1, S1 = transitions_grouped(AG, QG, g, cc, method="lyapunov")
E2, M2, S2 = transitions_grouped(AG, QG, g, cc, method="composition")
print("routes agree: max |S_lyapunov - S_composition| =", float((S1 - S2).abs().max()))
Am = torch.tensor([[[-0.1, 0.1], [0.1, -0.1]]])
cm = torch.tensor([[[0.3, 2.0, 7.5]]])
Ea, Ma, Sa = transitions_grouped(Am, Q[None], torch.zeros(1, dtype=torch.long), cm)
Eb, Mb, Sb = transitions(Am, Q[None], cm)
print("marginal drift, c = 0.3, 2, 7.5: max |grouped - Van Loan| =",
      max(float((u - v).abs().max()) for u, v in ((Ea, Eb), (Ma, Mb), (Sa, Sb))))
_, _, Sbig = transitions_grouped(Am, Q[None], torch.zeros(1, dtype=torch.long), torch.tensor([[[5e3]]]))
_, _, Snp = transition_np(Am[0].numpy(), Qn, 5e3, 1.0)
print("marginal drift, c = 5000: S =\n", Sbig[0, 0, 0].numpy(), "\n max |grouped - transition_np| =",
      np.abs(Sbig[0, 0, 0].numpy() - Snp).max())

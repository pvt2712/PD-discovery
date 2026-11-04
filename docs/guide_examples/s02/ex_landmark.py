"""Section 2.30: the masked landmark update equals the update on the observed components only."""
import numpy as np
import torch

import _setup                                    # noqa: F401
from pdx.engine.likelihood import _landmark_update

mu = torch.tensor([[0.2, 0.5, -0.1]])
S = torch.tensor([[[0.30, 0.10, 0.05], [0.10, 0.25, 0.02], [0.05, 0.02, 0.20]]])
R = torch.tensor([0.02, 0.05, 0.04])
y = torch.tensor([[0.6, 0.0, 0.3]])             # domain 1 missing (its 0 is a placeholder)
o = torch.tensor([[1.0, 0.0, 1.0]])
h = torch.tensor([[-0.1, 0.0, 0.0]])            # a treatment term on domain 0 at the landmark
m, P = _landmark_update(mu, S, y, o, h, R)
print("masked form:   m =", m[0].numpy(), "\n P =\n", P[0].numpy())

ob = np.array([0, 2])                            # the same update written with the observed components only
Sn, mun = S[0].numpy(), mu[0].numpy()
Ho = np.eye(3)[ob]
Sv = Ho @ Sn @ Ho.T + np.diag(R.numpy()[ob])
K = Sn @ Ho.T @ np.linalg.inv(Sv)
m2 = mun + K @ (y[0].numpy()[ob] - mun[ob] - h[0].numpy()[ob])
P2 = Sn - K @ Ho @ Sn
print("explicit form: m =", m2, "\n P =\n", P2)
print("max |differences|: m %.1e, P %.1e" % (np.abs(m[0].numpy() - m2).max(), np.abs(P[0].numpy() - P2).max()))
print("domain 1 was not observed, yet its mean moved from 0.5 to %.4f through its prior covariance" % m2[1])

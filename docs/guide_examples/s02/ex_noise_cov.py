"""Sections 2.7 and 2.11: the Cholesky parametrisation of Q (noise_cov) and its inverse (pack_chol)."""
import numpy as np
import torch

import _setup                                    # noqa: F401
from pdx.engine.model import Layout, ModelSpec, noise_cov, pack_chol

print("lower-triangle order, torch:", torch.tril_indices(3, 3).tolist())
print("lower-triangle order, numpy:", [list(map(int, a)) for a in np.tril_indices(3)])

spec = ModelSpec(d=3, support=np.zeros((3, 3)))
lay = Layout(spec, "free")
G = np.array([[0.30, 0.00, 0.00],
              [0.05, 0.20, 0.00],
              [-0.02, 0.03, 0.25]])
raw = G[np.tril_indices(3)].copy()
raw[[0, 2, 5]] = np.log(np.diag(G))             # diagonal entries are stored as logs
print("\nraw L block of phenotype 0:", raw.round(4))
x = torch.zeros(lay.size)
x[lay.slices["L"][0]] = torch.as_tensor(np.concatenate([raw, raw]))
Q = noise_cov(x, lay)
print("Q_0 = G G':\n", Q[0].numpy())
print("eigenvalues of Q_0:", np.linalg.eigvalsh(Q[0].numpy()).round(6))
print("pack_chol(Q_0) - raw:", np.abs(pack_chol(Q[0].numpy()) - raw).max())

# a nearly singular Q: pack_chol floors its eigenvalues at 1e-6 before the Cholesky factorisation
v = np.array([1.0, -1.0, 0.0]) / np.sqrt(2)
Qs = 0.1 * np.eye(3) - (0.1 - 1e-8) * np.outer(v, v)
print("\nnear-singular Q: smallest eigenvalue", np.linalg.eigvalsh(Qs).min())
x[lay.slices["L"][0]] = torch.as_tensor(np.concatenate([pack_chol(Qs), pack_chol(Qs)]))
Qr = noise_cov(x, lay)[0].numpy()
print("after the round trip: smallest eigenvalue", np.linalg.eigvalsh(Qr).min().round(10),
      "| max |change| in Q:", np.abs(Qr - Qs).max().round(10))

"""Section 2.22: the numpy transition used by the simulators, against the engine's."""
import numpy as np
import torch

import _setup                                    # noqa: F401
from pdx.engine.transitions import transition_np, transitions_grouped

Q = np.array([[0.10, 0.02], [0.02, 0.08]])
for name, A in (("stable", np.array([[-0.25, 0.0], [0.20, -0.50]])),
                ("marginal", np.array([[-0.1, 0.1], [0.1, -0.1]]))):
    for s, dt in ((0.7, 1.0), (2.0, 3.5), (5e3, 1.0)):
        c = s * dt
        k = max(0, int(np.ceil(np.log2(max(c * np.abs(A).sum(0).max() / 0.25, 1.0)))))
        E, M, S = transition_np(A, Q, s, dt)
        Eg, Mg, Sg = transitions_grouped(torch.as_tensor(A)[None], torch.as_tensor(Q)[None],
                                         torch.zeros(1, dtype=torch.long), torch.tensor([[[c]]]))
        dE = np.abs(E - Eg[0, 0, 0].numpy()).max()
        dM = np.abs(M - Mg[0, 0, 0].numpy()).max()
        dS = np.abs(S - Sg[0, 0, 0].numpy()).max()
        print(f"{name:8s} c = {c:7g}: doublings k = {k:2d} | max |numpy - engine|: E {dE:.1e}  M {dM:.1e}  S {dS:.1e}")

"""Notes for the author: transitions.py works in the dtype of its inputs, and only importing
pdx.engine.model sets torch's default to float64. This script deliberately does not use _setup."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src")]

import numpy as np                               # noqa: E402
import torch                                     # noqa: E402
from scipy.linalg import expm, solve_continuous_lyapunov  # noqa: E402

from pdx.engine.transitions import transitions_grouped    # noqa: E402

An = np.array([[-0.25, 0.0], [0.20, -0.50]])
Qn = np.array([[0.10, 0.02], [0.02, 0.08]])
Sig = solve_continuous_lyapunov(An, -Qn)
S_true = Sig - expm(10 * An) @ Sig @ expm(10 * An).T
for label in ("only transitions imported", "after importing pdx.engine.model"):
    if label.startswith("after"):
        import pdx.engine.model                  # noqa: F401
    A, Q = torch.tensor([An.tolist()]), torch.tensor([Qn.tolist()])
    _, _, S = transitions_grouped(A, Q, torch.zeros(1, dtype=torch.long), torch.tensor([[[10.0]]]))
    print(f"{label:34s}: dtype {S.dtype}, max |S - S_true| at c = 10: {np.abs(S[0, 0, 0].double().numpy() - S_true).max():.1e}")

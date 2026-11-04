"""Guide §6.5: the exact transition family A uses (transition_np) against the Euler-Maruyama step
used only for the curved scenario S4 (_step), over one interval (synthetic only).

    OMP_NUM_THREADS=1 ../ppmi-causal/.venv/bin/python docs/guide_examples/s06/ex_step_vs_exact.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from scipy.linalg import expm  # noqa: E402
from pdx.engine.transitions import transition_np  # noqa: E402
from pdx.sim.family_a import _step, default_truth  # noqa: E402

t = default_truth(4)
A, b, q = t.A, t.b[1], t.q
s, dt = 2.0, 1.0                                   # a PIGD participant at the median speed, one year
x0 = np.array([0.8, 1.2, 0.3, -0.2])
load = np.array([0.05, -0.1, 0.0, 0.1])
E, M, S = transition_np(A, q * np.eye(4), s, dt)
mean_exact = E @ x0 + M @ (b + load)
# check E and M against one block exponential, and S against Van Loan's block
d = 4
B = np.zeros((2 * d, 2 * d)); B[:d, :d], B[:d, d:] = s * dt * A, s * dt * np.eye(d)
F = expm(B)
V = np.zeros((2 * d, 2 * d)); V[:d, :d], V[:d, d:], V[d:, d:] = -s * dt * A, s * dt * q * np.eye(d), s * dt * A.T
G = expm(V)
S_vl = G[d:, d:].T @ G[:d, d:]
print("max |E - expm|, |M - block|, |S - Van Loan|:",
      f"{abs(E - F[:d, :d]).max():.1e}, {abs(M - F[:d, d:]).max():.1e}, {abs(S - S_vl).max():.1e}")

rng = np.random.default_rng(3)
R = 20000
draws = np.array([_step(x0.copy(), A, b, load, s, q, dt, 0.0, rng) for _ in range(R)])
print("exact mean       :", np.round(mean_exact, 4))
print("Euler (10 steps) :", np.round(draws.mean(0), 4), " (Monte Carlo SE about", np.round(draws.std(0).max() / np.sqrt(R), 4), ")")
print("exact sd         :", np.round(np.sqrt(np.diag(S)), 4))
print("Euler sd         :", np.round(draws.std(0), 4))
drawc = np.array([_step(x0.copy(), A, b, load, s, q, dt, 0.08, rng) for _ in range(R)])
print("with curvature c = 0.08, Euler mean:", np.round(drawc.mean(0), 4), "(the c tanh(X) push)")

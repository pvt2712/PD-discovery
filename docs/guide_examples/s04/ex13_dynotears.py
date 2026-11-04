"""Sections 4.30-4.36: DYNOTEARS on a three-variable series with a known same-time chain
x0 -> x1 -> x2 and known lagged effects. Synthetic linear Gaussian data only."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import numpy as np  # noqa: E402
import scipy.linalg as slin  # noqa: E402

from pdx.comparators import dynotears as DY  # noqa: E402

np.set_printoptions(precision=3, suppress=True)
d = 3
W = np.zeros((d, d))
W[0, 1] = W[1, 2] = 0.8                       # same-time: x0 -> x1 -> x2 (W[i, j]: i -> j)
A = 0.5 * np.eye(d)                           # lagged: each variable's own lag ...
A[0, 2] = 0.4                                 # ... and x0(t-1) -> x2(t)
rng = np.random.default_rng(0)
T, burn = 2000, 200
x = np.zeros((T + burn + 1, d))
IW = np.linalg.inv(np.eye(d) - W)
for t in range(1, T + burn + 1):              # x_t = x_t W + x_{t-1} A + z_t, solved for x_t
    x[t] = (x[t - 1] @ A + rng.normal(size=d)) @ IW
X, Y = x[burn + 1:], x[burn:-1]               # current slice, lag-1 slice

t0 = time.time()
W0, A0, info0 = DY.dynotears(X, Y, 0.05, 0.05)                      # no thresholds
W1, A1, info1 = DY.dynotears(X, Y, 0.05, 0.05, tau_w=0.3, tau_a=0.1)  # the paper's large-sample settings
print(f"{time.time() - t0:.2f} s for both fits; info: {info1}")
print("W estimated, no threshold:\n", W0)
print("W estimated, tau_w = 0.3 (truth: 0.8 at [0, 1] and [1, 2]):\n", W1)
print("A estimated, tau_a = 0.1 (truth: 0.5 on the diagonal, 0.4 at [0, 2]):\n", A1)
print(f"h(W) of the final estimate: {np.trace(slin.expm(W0 * W0)) - d:.2e}")

# the loss from cross-product matrices equals the authors' residual form
n = len(X)
v = np.abs(rng.normal(scale=0.3, size=2 * (d * d + d * d)))
Wv, Av = DY._unpack(v, d, 1)
S = (X.T @ X / n, X.T @ Y / n, Y.T @ Y / n)
f, _ = DY._objective(v, S, d, 1, 0.0, 0.0, 0.0, 0.0)              # rho = alpha = lambda = 0: the loss only
print(f"loss from cross-products {f:.10f}; residual form {0.5 / n * np.sum((X @ (np.eye(d) - Wv) - Y @ Av) ** 2):.10f}")
print(f"gradient check (largest relative error against central differences): {DY.gradient_check():.1e}")

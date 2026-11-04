"""Section 2.25: Gauss-Hermite nodes and weights for a standard normal u, and their accuracy."""
import math

import numpy as np
import torch

import _setup                                    # noqa: F401
from pdx.engine.likelihood import LOG_SPEED_BOUND, gh_nodes

U, logw = gh_nodes(12)
w = np.exp(logw.numpy())
print("K = 12 nodes u_m:", U.numpy().round(4))
print("weights w_m     :", np.array2string(w, precision=3))
print("sum of weights:", w.sum())
print("\n moment E[u^2j]   exact (2j-1)!!   12-node rule")
for j in (1, 2, 6, 11, 12):
    exact = math.prod(range(1, 2 * j, 2))
    print(f"   u^{2 * j:<2d}       {exact:16.6e}   {float((w * U.numpy() ** (2 * j)).sum()):16.6e}")
print("\n E[exp(sigma u)] = exp(sigma^2 / 2): relative error of the rule")
print(" sigma      K = 12      K = 20      K = 30")
for sigma in (0.4, 1.0, 2.0, 3.0):
    row = []
    for K in (12, 20, 30):
        Uk, lk = gh_nodes(K)
        approx = float(torch.logsumexp(lk + sigma * Uk, 0))      # log of sum_m w_m exp(sigma u_m)
        row.append(abs(math.expm1(approx - sigma ** 2 / 2)))
    print(f"  {sigma:3.1f}   " + "  ".join(f"{r:10.1e}" for r in row))
print(f"\nlargest node {float(U.max()):.3f}: the bound |log-speed| <= {LOG_SPEED_BOUND} binds at that node once "
      f"sigma > {LOG_SPEED_BOUND / float(U.max()):.2f} (for a mean log-speed of 0)")

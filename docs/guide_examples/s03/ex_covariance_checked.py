"""Section 3.12: covariance_checked on a 3 x 3 information matrix built from known eigenvalues:
one ordinary direction, one nearly flat and one with negative information. Toy numbers only."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

import numpy as np
from pdx.engine.fit import covariance, covariance_checked

np.set_printoptions(precision=4, suppress=False)
c, s = np.cos(0.3), np.sin(0.3)
V = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])   # a rotation in the (a, b) plane
w = np.array([50.0, 1e-12, -0.5])                              # ordinary, flat, negative
H = V @ np.diag(w) @ V.T
Cov, rep = covariance_checked(H, names=["a", "b", "c"])
print("eigenvalues of H:", np.linalg.eigvalsh(H))
for k in ("min_eigenvalue", "condition_number", "positive_definite", "well_conditioned"):
    print(f"  {k}: {rep[k]}")
for fd in rep["flat_directions"]:
    print("  flat direction:", fd)
print("covariance returned:\n", Cov)
print("it equals (1/50) v1 v1' + 1e12 v2 v2' to 0.1% (the negative direction is dropped):",
      np.allclose(Cov, np.outer(V[:, 0], V[:, 0]) / 50 + 1e12 * np.outer(V[:, 1], V[:, 1]), rtol=1e-3, atol=0))
print("variance of c (negative information): %.1f" % Cov[2, 2], "| covariance() gives the same matrix:",
      np.array_equal(covariance(H), Cov))

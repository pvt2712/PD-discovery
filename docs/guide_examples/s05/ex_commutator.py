"""Sections 5.41-5.42: the shared-eigenvector (commutator) check and the embeddability check.
Exact matrices first, then simulator family A panels. Synthetic data only."""
import time

import numpy as np
from scipy.linalg import expm

import _fixture  # noqa: F401  (import path)
from pdx.engine.fit import _yearly_ols
from pdx.localise import commutator_test, embeddability_check
from pdx.sim import scenarios

A = -0.25 * np.eye(4)
A[2, 0], A[2, 1], A[3, 2] = 0.10, 0.20, 0.25
B0, B1 = expm(A), expm(2.0 * A)
A1 = 2.0 * A
A1[2, 1] += 0.30                                  # axial -> function stronger beyond speed
B1x = expm(A1)
print("||B0 B1 - B1 B0||, same A at speeds 1 and 2:", f"{np.linalg.norm(B0 @ B1 - B1 @ B0):.2e}")
print("||B0 B1 - B1 B0||, one link stronger in PIGD :", f"{np.linalg.norm(B0 @ B1x - B1x @ B0):.4f}")

for sc in ("S0", "S1", "S2"):
    t0 = time.time()
    panel, info = scenarios.generate(sc, 600, np.random.default_rng(7), d=4)
    r = commutator_test(panel, B=200, rng=np.random.default_rng(8))
    print(f"{sc}: n = {panel.n}, statistic {r['statistic']:.4f}, p = {r['p']:.3f} ({time.time() - t0:.0f} s)")
print("embeddability (S2 panel):", embeddability_check(panel))
print("eigenvalues of the yearly least-squares matrix, TD:", np.round(np.linalg.eigvals(_yearly_ols(panel, 0, 4)[0]), 3))
print("eigenvalues of the true yearly matrix expm(A):    ", np.round(np.linalg.eigvals(B0), 3))

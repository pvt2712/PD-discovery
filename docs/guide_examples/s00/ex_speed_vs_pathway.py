"""Section 0.5, idea 1: one mechanism run at twice the speed changes the yearly links unevenly.
Synthetic numbers only."""
import numpy as np
from scipy.linalg import expm

A = np.array([[-0.50, 0.00],      # domain 1 (a motor sign): own rate
              [0.30, -0.40]])     # domain 2 (daily function): driven by domain 1, own rate
yearly_td = expm(A)               # TD runs the mechanism at speed 1
yearly_pigd = expm(2.0 * A)       # PIGD runs the same mechanism at speed 2
np.set_printoptions(precision=4, suppress=True)
print("yearly transition, TD:\n", yearly_td)
print("yearly transition, PIGD (same A, speed 2):\n", yearly_pigd)
print("entry-by-entry ratio PIGD / TD:\n", yearly_pigd / np.where(yearly_td == 0, np.nan, yearly_td))
print("continuous-time drift ratio (what the engine compares):", (2.0 * A)[1, 0] / A[1, 0])

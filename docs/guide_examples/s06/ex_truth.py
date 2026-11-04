"""Guide §6.1-6.2: the Truth record and the preset shared mechanism of family A (synthetic only).

    OMP_NUM_THREADS=1 ../ppmi-causal/.venv/bin/python docs/guide_examples/s06/ex_truth.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from pdx.sim.family_a import default_truth, true_support  # noqa: E402

np.set_printoptions(precision=3, suppress=True)
t = default_truth(4)
print("domains:", t.domains, "| d =", t.d)
print("A (row = to, column = from):\n", t.A + 0.0)          # + 0.0 turns -0.0 into 0.0 for printing
print("A_pigd equals A:", np.array_equal(t.A, t.A_pigd), "| speed (TD, PIGD) =", t.speed, "-> speed_ratio =", t.speed_ratio)
print("b (row 0 = TD, row 1 = PIGD):\n", t.b)
print("equilibrium -A^-1 b, per phenotype:", np.linalg.solve(-t.A, t.b[0]), np.linalg.solve(-t.A, t.b[1]))
print("eigenvalues of A:", np.round(np.linalg.eigvals(t.A).real, 3))
print("treat_mask:", t.treat_mask.astype(int), "| kappa:", t.kappa[0], "| kappa_ldr:", t.kappa_ldr[0])
print("q =", t.q, "| tau =", t.tau, "| sigma =", t.sigma, "| meas_var =", t.meas_var, "| lam =", t.lam)
print("true_support:\n", true_support(t).astype(int))
for d in (6, 10):
    td = default_truth(d)
    links = [(td.domains[j], td.domains[k], td.A[j, k]) for j, k in zip(*np.nonzero(td.A)) if j != k]
    print(f"d = {d}: {len(links)} links; largest real eigenvalue {np.linalg.eigvals(td.A).real.max():.3f}; "
          f"treated: {[n for n, m in zip(td.domains, td.treat_mask) if m]}")
    print("   ", "; ".join(f"{a} <- {b} {v:.2f}" for a, b, v in links))

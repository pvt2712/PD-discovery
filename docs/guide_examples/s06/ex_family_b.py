"""Guide §6.17-6.19: simulator family B - its S0, S1 and S3 analogues, the treatment that saturates
with time on treatment, and the measurement loadings (synthetic only).

    OMP_NUM_THREADS=1 ../ppmi-causal/.venv/bin/python docs/guide_examples/s06/ex_family_b.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from pdx.sim import family_b  # noqa: E402

np.set_printoptions(precision=3, suppress=True)
for label, kw in (("S0", dict(speed_ratio=1.0, load_shift=0.0)), ("S1", dict(speed_ratio=2.0, load_shift=0.05)),
                  ("S3", dict(speed_ratio=2.0, load_shift=0.05, mechanism_shift=0.12))):
    panel, info = family_b.simulate(600, np.random.default_rng(21), **kw)
    T = np.array([len(t) for t in panel.t])
    print(f"{label}: n_TD {info['n_TD']}, n_PIGD {info['n_PIGD']}, speed_ratio {info['speed_ratio']}, "
          f"median visits {np.median(T):.0f}, missing {np.isnan(np.vstack(panel.y)).mean():.3f}, "
          f"treated by last visit {np.mean([dz[-1] > 0 for dz in panel.dose]):.3f}")
print("support handed to the analysis:\n", info["support"].astype(int))
i = next(k for k in range(panel.n) if panel.dose[k][-1] > 0 and panel.dose[k][1] == 0)
print("a treated participant: t", panel.t[i], "\n   dose", panel.dose[i], "\n   exposure", panel.exposure[i],
      "(exposure is the dose)", "\n   on", panel.on[i])
tau = np.array([0.0, 0.25, 0.5, 1.0, 2.0])
print("treatment term at time on treatment", tau, ":\n   OFF", -0.3 * (1 - np.exp(-tau / 0.5)) * 0.5 + 0.0,
      "\n   ON ", -0.3 * (1 - np.exp(-tau / 0.5)) * 1.0 + 0.0)
# the loading of line 51, evaluated for each phenotype and era (d = 4)
for nonInv in (0.0, 0.10):
    print(f"noninvariant = {nonInv}:")
    for ph in (0, 1):
        for era in (0, 1):
            load = 1.0 + 0.04 * ph + 0.05 * era + nonInv * (ph - 0.5) * np.where(np.arange(4) % 2 == 0, 1, -1)
            print(f"   phenotype {ph}, era {era}: loading per domain {load}")
# linear part of the dynamics on the measured scale: L A L^-1 with A = -0.25 I + W
W = np.zeros((4, 4)); W[2, 0], W[2, 1], W[3, 2] = 0.10, 0.20, 0.25
A = -0.25 * np.eye(4) + W
for nonInv in (0.0, 0.10):
    L0 = np.diag(1.0 + nonInv * (0 - 0.5) * np.where(np.arange(4) % 2 == 0, 1, -1))
    L1 = np.diag(1.04 + nonInv * (1 - 0.5) * np.where(np.arange(4) % 2 == 0, 1, -1))
    M0, M1 = L0 @ A @ np.linalg.inv(L0), L1 @ A @ np.linalg.inv(L1)
    print(f"noninvariant = {nonInv}: measured linear drift, PIGD minus TD (era 0), largest |entry| "
          f"{np.abs(M1 - M0).max():.4f}")

"""Guide §6.10-6.14: the scenario truths of family A and the size of one epsilon (synthetic only).

    OMP_NUM_THREADS=1 ../ppmi-causal/.venv/bin/python docs/guide_examples/s06/ex_scenarios.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from pdx.sim import scenarios  # noqa: E402
from pdx.sim.family_a import default_truth  # noqa: E402

print("NULL_SCENARIOS:", scenarios.NULL_SCENARIOS, "| POWER_SCENARIOS:", scenarios.POWER_SCENARIOS, "| EPS =", scenarios.EPS)
t0 = time.time()
sd = scenarios.pooled_sd(4)
print(f"pooled_sd(4) = {np.round(sd, 4)}  (first call {time.time() - t0:.1f} s)")
t0 = time.time(); scenarios.pooled_sd(4)
print(f"second call {time.time() - t0:.4f} s; {scenarios.pooled_sd.cache_info()}")
names = default_truth(4).domains
ax, ind, fn, tr = (names.index(k) for k in ("axial", "independence", "function", "tremor"))
print(f"one epsilon in raw drift units: independence<-function {scenarios._std(4, ind, fn):.5f}, "
      f"independence<-axial {scenarios._std(4, ind, ax):.5f}, function<-tremor {scenarios._std(4, fn, tr):.5f}")
print(f"_load_truth: b[PIGD] - b[TD] = {np.round(scenarios._load_truth(default_truth(4)).b[1] - default_truth(4).b[0], 3)}")
print()
print("scenario  speed       sigma        tau          kappa[PIGD,axial]  curvature  b[PIGD,ax]-b[TD,ax]  changed drift entries (PIGD minus TD, in epsilon)")
for sc in scenarios.NULL_SCENARIOS + scenarios.POWER_SCENARIOS:
    t = scenarios.truth_for(sc, 4)
    db = t.b[1, ax] - t.b[0, ax]
    D = t.A_pigd - t.A
    changes = []
    for j, k in zip(*np.nonzero(np.abs(D) > 1e-12)):
        changes.append(f"{names[j]}<-{names[k]}: {D[j, k] / scenarios._std(4, j, k):+.2f}")
    base = default_truth(4).A
    td_changes = [f"TD {names[j]}<-{names[k]} = {t.A[j, k] / scenarios._std(4, j, k):.2f} eps"
                  for j, k in zip(*np.nonzero(np.abs(t.A - base) > 1e-12))]
    print(f"{sc:8s}  {str(t.speed):10s}  {str(t.sigma):11s}  {str(t.tau):11s}  {t.kappa[1, ax]:17.2f}  {t.curvature:9.2f}  "
          f"{db:19.2f}  {'; '.join(changes + td_changes) or '-'}")
t3 = scenarios.truth_for("S3", 4)
print(f"\nS3 raw values: A[function, tremor] TD {t3.A[fn, tr]:.3f} vs PIGD {t3.A_pigd[fn, tr]:.3f}; "
      f"A[independence, axial] TD {t3.A[ind, ax] + 0.0:.3f} vs PIGD {t3.A_pigd[ind, ax]:.3f}")
try:
    scenarios.truth_for("S5")
except ValueError as exc:
    print("truth_for('S5'):", exc)

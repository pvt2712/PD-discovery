"""Guide §6.47-6.48: recovery of known parameters - one dataset under S1 (speed-only fit, with
model-based intervals) and one under S2 (free fit), then the summaries (synthetic only, small n).

    OMP_NUM_THREADS=1 ../ppmi-causal/.venv/bin/python docs/guide_examples/s06/ex_gate_recovery.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from pdx import gate  # noqa: E402

for scenario, intervals in (("S1", True), ("S2", False)):
    t0 = time.time()
    shared = gate.recovery_setup(scenario, 60, 3, 4, None, intervals, None)
    row = gate.recovery_dataset(shared, 0)                 # dataset 0: seed 3 + 0
    print(f"{scenario} row ({time.time() - t0:.0f} s):")
    for k, v in row.items():
        numeric = isinstance(v, float) or (isinstance(v, list) and not isinstance(v[0], bool))
        print(f"   {k:22s}", np.round(v, 4) if numeric else v)
    out = gate.recovery(scenario, 60, 1, seed=3, d=4, with_intervals=intervals, rows=[row])   # nothing left to fit
    print(f"{scenario} summary:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()})

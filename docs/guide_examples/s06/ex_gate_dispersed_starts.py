"""Guide §6.42-6.43: both models fitted from their default start and from a dispersed start, with
the progress saved after each fit, then a run resumed half-way that reproduces the first
(synthetic only; the dataset of tests/test_gate.py).

    OMP_NUM_THREADS=1 ../ppmi-causal/.venv/bin/python docs/guide_examples/s06/ex_gate_dispersed_starts.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from pdx import gate  # noqa: E402
from pdx.qualification import spec_for  # noqa: E402
from pdx.sim import scenarios  # noqa: E402

panel, info = scenarios.generate("S1", 80, np.random.default_rng(6), d=4)
spec = spec_for(panel, info)
seen = []
t0 = time.time()
full = gate.check_dispersed_starts(panel, spec, np.random.default_rng(1), n_starts=1,
                                   on_fit=lambda done: seen.append({k: list(v) for k, v in done.items()}))
print(f"full run ({time.time() - t0:.0f} s):")
for k in ("null", "free"):
    print(f"   {k}: logliks {np.round(full[k]['logliks'], 6)}, spread {full[k]['spread']:.2e}")
print("   pass:", full["pass"])
print("progress saved after each fit:", [{k: len(v) for k, v in s.items()} for s in seen])
t0 = time.time()
resumed = gate.check_dispersed_starts(panel, spec, np.random.default_rng(1), n_starts=1, done=seen[1])
print(f"resumed from the second save ({time.time() - t0:.0f} s): identical to the full run: {resumed == full}")
shared = gate.start_setup(panel, spec, None)
print("start_setup keys:", sorted(shared), "| start_fit(null, default start):",
      round(gate.start_fit(shared, ("null", None)), 6))

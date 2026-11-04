"""Guide §6.38-6.39: the gate's first two checks on its small dataset - the engine's likelihood
against the independent implementation, and the exact gradient against central differences
(synthetic only; the same dataset and seeds as run_gate with seed 2026).

    OMP_NUM_THREADS=1 ../ppmi-causal/.venv/bin/python docs/guide_examples/s06/ex_gate_reference_gradient.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from pdx import gate  # noqa: E402
from pdx.engine.fit import start_values  # noqa: E402
from pdx.engine.likelihood import loglik, prepare  # noqa: E402
from pdx.engine.model import Layout  # noqa: E402
from pdx.engine.reference import reference_loglik  # noqa: E402
from pdx.qualification import spec_for  # noqa: E402
from pdx.sim import scenarios  # noqa: E402

rng = lambda k: np.random.default_rng([2026, k])          # as in run_gate
small, info = scenarios.generate("S1", 40, rng(0), d=4)
spec = spec_for(small, info)
print("small dataset:", small.n, "participants,", small.n_transitions(), "transitions")
t0 = time.time()
print("check_reference:", gate.check_reference(small, spec, rng(1)), f"({time.time() - t0:.1f} s)")
prep = prepare(small, spec.meas_var, spec.n_groups)
lay = Layout(spec, "null")
x = start_values(small, spec, "null") + 0.05 * np.random.default_rng(0).normal(size=lay.size)
a = float(loglik(torch.as_tensor(x), lay, prep))
b = reference_loglik(x, lay, small, prep.mu1.numpy(), prep.S1.numpy(), prep.y1c.numpy())
print(f"   one point, null model: engine {a:.10f}, reference {b:.10f}, difference {a - b:.1e}")
t0 = time.time()
print("check_gradient (h = 1e-6):", gate.check_gradient(small, spec, rng(2)), f"({time.time() - t0:.1f} s)")
for h in (1e-2, 1e-4, 1e-8, 1e-10):
    r = gate.check_gradient(small, spec, rng(2), h=h)
    print(f"   h = {h:.0e}: worst relative difference {r['max_relative_difference']:.1e}, pass {r['pass']}")

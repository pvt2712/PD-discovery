"""Guide §6.N, note on dispersed starts: on the small dataset of tests/test_gate.py, a dispersed
start of the speed-only model ends with a speed spread at zero, and a dispersed start of the free
model beats both of fit_pair's starts (synthetic only).

    OMP_NUM_THREADS=1 ../ppmi-causal/.venv/bin/python docs/guide_examples/s06/ex_note_starts.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from pdx.engine.fit import fit_model, fit_pair, param_names, start_values  # noqa: E402
from pdx.qualification import spec_for  # noqa: E402
from pdx.sim import scenarios  # noqa: E402

panel, info = scenarios.generate("S1", 80, np.random.default_rng(6), d=4)
spec = spec_for(panel, info)
rng = np.random.default_rng(1)                                    # as check_dispersed_starts draws them
x_null = start_values(panel, spec, "null")
start_null = x_null + 0.3 * rng.normal(size=len(x_null))
x_free = start_values(panel, spec, "free")
start_free = x_free + 0.3 * rng.normal(size=len(x_free))
fa = fit_model(panel, spec, "null")
fb = fit_model(panel, spec, "null", x0=start_null)
names = param_names(fa.layout)
j = names.index("log_sigma[0]")
print(f"null, default start:   loglik {fa.loglik:.4f}, log_sigma[0] = {fa.x[j]:.3f}")
print(f"null, dispersed start: loglik {fb.loglik:.4f}, log_sigma[0] = {fb.x[j]:.3f} ({fb.message})")
fc = fit_model(panel, spec, "null", x0=fb.x)
print(f"   restarted where it stopped: loglik {fc.loglik:.4f} after {fc.n_eval} evaluations")
f0, f1 = fit_pair(panel, spec)
fd = fit_model(panel, spec, "free", x0=start_free)
print(f"free, fit_pair's two starts: {np.round(f1.start_logliks, 4)}; a dispersed start: {fd.loglik:.4f}")
print(f"LR from fit_pair {2 * (f1.loglik - f0.loglik):.3f}; with the dispersed free optimum {2 * (fd.loglik - f0.loglik):.3f}")

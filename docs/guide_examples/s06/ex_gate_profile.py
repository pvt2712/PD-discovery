"""Guide §6.44-6.45: profile likelihoods over +- 4 standard errors, for the speed ratio and for a
load spread (synthetic only; the same small S1 dataset as ex_gate_fit_checks.py).

    OMP_NUM_THREADS=1 ../ppmi-causal/.venv/bin/python docs/guide_examples/s06/ex_gate_profile.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from pdx import gate  # noqa: E402
from pdx.engine.fit import covariance_checked, fit_model, hessian  # noqa: E402
from pdx.qualification import spec_for  # noqa: E402
from pdx.sim import scenarios  # noqa: E402

panel, info = scenarios.generate("S1", 120, np.random.default_rng(5), d=4)
spec = spec_for(panel, info)
f0 = fit_model(panel, spec, "null")
se = np.sqrt(np.clip(np.diag(covariance_checked(hessian(f0, panel))[0]), 1e-300, None))
lay = f0.layout
t0 = time.time()
shared = gate.profile_setup(f0, panel, 5, None)             # what each parallel worker holds
print("profile_setup keys:", sorted(shared))
for nm, ix, n_grid in (("gamma", 0, 5), ("log_tau", 0, 3)):
    j = lay.slices[nm][0].start + ix
    if nm == "gamma":                                         # through the worker function (n_grid from shared)
        p = gate.profile_job(shared, (nm, ix, float(se[j])))
    else:                                                     # directly, with a coarser grid to save time
        p = gate.profile(f0, panel, nm, ix, float(se[j]), n_grid=n_grid)
    drop = max(max(p["loglik"]), f0.loglik) - np.array(p["loglik"])
    print(f"profile of {p['parameter']} (estimate {f0.x[j]:.3f}, se {se[j]:.3f}): grid {np.round(p['grid'], 3)}")
    print(f"   drop from the top {np.round(drop, 3)} -> identified {p['identified']}")
print(f"({time.time() - t0:.0f} s for the two profiles)")

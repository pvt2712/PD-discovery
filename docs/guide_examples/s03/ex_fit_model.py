"""Section 3.7: fit_model with L-BFGS-B. What ftol and gtol do, and the best of several starts.
Synthetic data only (scenario S1, 80 simulated participants, 4 domains)."""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

import numpy as np
from helpers import small_panel, spec_for
from pdx.engine.fit import fit_model, start_values
from pdx.engine.likelihood import prepare, value_and_grad

panel, info = small_panel("S1", n=80, seed=1, d=4)
spec = spec_for(panel, info)
prep = prepare(panel, spec.meas_var, spec.n_groups)

t0 = time.time()
f = fit_model(panel, spec, "null", prep=prep)              # defaults: ftol 1e-10, gtol 1e-5, maxcor 50
nll, g = value_and_grad(f.x, f.layout, prep)
print("default settings: loglik %.6f, success %s, %d evaluations, %.0f s" % (f.loglik, f.success, f.n_eval, time.time() - t0))
print("  message:", f.message)
print("  largest |gradient| at the optimum: %.2e (gtol 1e-5)" % np.abs(g).max())
print("  ftol threshold on the last decrease: 1e-10 x max(|f|, 1) = %.1e" % (1e-10 * abs(f.loglik)))
print("  speed ratio exp(gamma) = %.3f (truth: median speed ratio 2)" % np.exp(f.layout.get(f.x, "gamma")[0]))

t0 = time.time()
f6 = fit_model(panel, spec, "null", prep=prep, ftol=1e-6)
print("ftol 1e-6: loglik %.6f (%.1e below), %d evaluations, %.0f s" % (f6.loglik, f.loglik - f6.loglik, f6.n_eval, time.time() - t0))

rng = np.random.default_rng(1)
far = start_values(panel, spec, "null") + 0.3 * rng.normal(size=f.layout.size)   # a dispersed start
t0 = time.time()
fm = fit_model(panel, spec, "null", prep=prep, extra_starts=[far])
print("two starts (default, dispersed): start_logliks", np.round(fm.start_logliks, 6),
      "| kept %.6f | %d evaluations in all, %.0f s" % (fm.loglik, fm.n_eval, time.time() - t0))

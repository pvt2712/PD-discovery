"""Section 3.N, note 1: a start at which the likelihood cannot be evaluated is reported as a
converged fit. value_and_grad returns (1e10, zero gradient) there, and L-BFGS-B stops at once
because the projected gradient is zero. Synthetic data only (scenario S1, 4 domains)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

import numpy as np
from helpers import small_panel, spec_for
from pdx.engine.fit import fit_pair, start_values
from pdx.engine.likelihood import prepare, value_and_grad
from pdx.engine.model import Layout
from pdx.hp import lr_stat

panel, info = small_panel("S1", n=80, seed=1, d=4)
spec = spec_for(panel, info)
lay = Layout(spec, "null")
bad = start_values(panel, spec, "null")
bad[lay.slices["log_tau"][0]] = 400.0                    # tau^2 = exp(800) overflows to inf
val, grad = value_and_grad(bad, lay, prepare(panel, spec.meas_var, spec.n_groups))
print("value_and_grad at the start: value %.0e, max |gradient| %.1f" % (val, np.abs(grad).max()))
f0, f1 = fit_pair(panel, spec, x0_null=bad)
print("null fit: loglik %.3g, success %s, %d evaluation(s), message: %s" % (f0.loglik, f0.success, f0.n_eval, f0.message))
print("free fit: loglik %.3f, start_logliks %s" % (f1.loglik, np.round(f1.start_logliks, 3)))
print("H-P statistic lr_stat = %.3g; HPResult.summary would show null_converged = %s" % (lr_stat(f0, f1), f0.success))

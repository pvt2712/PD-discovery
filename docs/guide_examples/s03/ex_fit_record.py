"""Section 3.1: the fields of a Fit record, from a deliberately short speed-only fit.
Synthetic data only (scenario S1, 80 simulated participants, 4 domains)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

import numpy as np
from helpers import small_panel, spec_for
from pdx.engine.fit import fit_model

panel, info = small_panel("S1", n=80, seed=1, d=4)
spec = spec_for(panel, info)
f = fit_model(panel, spec, "null", maxiter=40)          # stopped early on purpose
print("kind:", f.kind, "| n_params:", f.n_params, "| len(x):", len(f.x))
print("loglik: %.4f" % f.loglik, "| success:", f.success)
print("message:", f.message)
print("n_eval:", f.n_eval, "| start_logliks:", np.round(f.start_logliks, 4))
print("params keys:", sorted(f.params))
print("params['A'] shape:", f.params["A"].shape, "| log_speed:", np.round(f.params["log_speed"], 4))

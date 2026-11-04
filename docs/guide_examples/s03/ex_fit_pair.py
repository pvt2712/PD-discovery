"""Section 3.8: fit_pair, the speed-only (null) and free fits of H-P, and the likelihood ratio.
Synthetic data only (scenario S1: one shared mechanism, so H-P's null is true)."""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

import numpy as np
from helpers import small_panel, spec_for
from pdx.engine.fit import fit_pair
from pdx.hp import h_p_df

panel, info = small_panel("S1", n=80, seed=1, d=4)
spec = spec_for(panel, info)
t0 = time.time()
f0, f1 = fit_pair(panel, spec)
print("null: loglik %.6f, %d parameters, success %s, %d evaluations" % (f0.loglik, f0.n_params, f0.success, f0.n_eval))
print("free: loglik %.6f, %d parameters, success %s, %d evaluations" % (f1.loglik, f1.n_params, f1.success, f1.n_eval))
print("free start_logliks [from the null solution, from its own default]:", np.round(f1.start_logliks, 6))
lr = 2 * (f1.loglik - f0.loglik)
print("LR = 2 (l_free - l_null) = %.4f on df = %d (= n_params difference %d); %.0f s"
      % (lr, h_p_df(spec), f1.n_params - f0.n_params, time.time() - t0))
A0, A1 = f1.params["A"][0], f1.params["A"][1]
s_ls = float(A1.ravel() @ A0.ravel() / (A0.ravel() @ A0.ravel()))
print("speed-only speed exp(gamma) = %.3f; least-squares speed of the free drifts = %.3f" % (np.exp(f0.params["log_speed"][1]), s_ls))

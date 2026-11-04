"""Section 5.0: the examples' fixture recovers its true parameters on a larger panel (about 400
participants), so the noisy estimates of the small examples are sampling noise. Synthetic data
only; takes about a minute."""
import time

import numpy as np

from _fixture import engine_panel
from pdx import hp
from pdx.engine.fit import fit_pair
from pdx.localise import speed_rule

t0 = time.time()
panel, spec, truth = engine_panel("null", n=450, seed=9)
f0, f1 = fit_pair(panel, spec, own_free_start=False)
a0, a1 = f1.params["A"][0][spec.support], f1.params["A"][1][spec.support]
print(f"n = {panel.n} ({time.time() - t0:.0f} s): exp(gamma-hat) = {np.exp(f0.params['log_speed'][1]):.3f} (truth 2), "
      f"minimal-L1 speed {speed_rule(a0, a1):.3f}, LR {hp.lr_stat(f0, f1):.2f} on {hp.h_p_df(spec)} df")
print("speed-only fit, A-hat:\n", np.round(f0.params["A"][0], 3))
print("truth A:\n", truth["A"])
print("sigma-hat", np.round(f0.params["sigma"], 3), "(truth 0.3, 0.3)")

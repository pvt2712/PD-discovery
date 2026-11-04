"""Section 3.N, note 3: where the nesting of null_to stops being exact. Three edge cases, each
made by moving one parameter of a speed-only vector: a noise eigenvalue below pack_chol's floor
(1e-6), an own rate below pack_drift's floor (1e-4), and quadrature nodes at the +-15 log-speed
clamp of the likelihood. Synthetic data only (scenario S1, 4 domains)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

import numpy as np
import torch
from helpers import random_point, small_panel, spec_for
from pdx.engine.fit import null_to
from pdx.engine.likelihood import loglik, prepare
from pdx.engine.model import Layout, unpack_numpy

panel, info = small_panel("S1", n=80, seed=1, d=4)
spec = spec_for(panel, info)
prep = prepare(panel, spec.meas_var, spec.n_groups)
l0, l1 = Layout(spec, "null"), Layout(spec, "free")
base = random_point(panel, spec, "null", seed=5)
base[l0.slices["gamma"][0]] = np.log(1.8)
tri = spec.d * (spec.d + 1) // 2


def gap(x):
    a = float(loglik(torch.as_tensor(x), l0, prep))
    b = float(loglik(torch.as_tensor(null_to(x, spec, "free")), l1, prep))
    return a, b - a


x = base.copy()
x[l0.slices["L"][0].start + tri] = -9.0                  # PIGD's first noise variance e^-18 = 1.5e-8
q = np.linalg.eigvalsh(unpack_numpy(x, l0)["Q"][1]).min()
a, d1 = gap(x)
print("noise eigenvalue %.1e (x 1.8 = %.1e < 1e-6): free minus null log-likelihood %+.2e" % (q, 1.8 * q, d1))
x = base.copy()
x[l0.slices["A"][0].start + spec.diag_pos[0]] = -10.0    # own rate -e^-10 = -4.5e-5
a, d2 = gap(x)
print("own rate %.1e (x 1.8 = %.1e, floored at 1e-4): free minus null %+.2e" % (-np.exp(-10), -1.8 * np.exp(-10), d2))
x = base.copy()
x[l0.slices["log_sigma"][0].start + 1] = np.log(5.0)     # PIGD's speed spread sigma = 5
a, d3 = gap(x)
print("sigma = 5, so the outer nodes reach the +-15 clamp: free minus null %+.2e" % d3)
print("unchanged vector: free minus null %+.1e" % gap(base)[1])

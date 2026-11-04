"""Section 3.6: null_to writes a speed-only parameter vector in the free and joint layouts without
changing the log-likelihood. Synthetic data only (scenario S1, 4 domains)."""
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
l0 = Layout(spec, "null")
x0 = random_point(panel, spec, "null", seed=5)          # every block non-zero (load, era noise, ...)
x0[l0.slices["gamma"][0]] = np.log(1.8)                 # PIGD runs at 1.8 times the reference speed
ll0 = float(loglik(torch.as_tensor(x0), l0, prep))
print("speed-only log-likelihood: %.10f" % ll0)
for kind in ("free", "joint"):
    l1 = Layout(spec, kind)
    x1 = null_to(x0, spec, kind)
    ll1 = float(loglik(torch.as_tensor(x1), l1, prep))
    print(f"{kind:5s} layout ({l1.size} parameters): {ll1:.10f}  difference {ll1 - ll0:+.1e}")

p0 = unpack_numpy(x0, l0)
p1 = unpack_numpy(null_to(x0, spec, "free"), Layout(spec, "free"))
eg = np.exp(p0["log_speed"][1])
print("\nfree-layout quantities of PIGD against e^gamma times the speed-only ones (max |difference|):")
for name, a, b in [("A", p1["A"][1], eg * p0["A"][1]), ("b", p1["b"][1], eg * p0["b"][1]),
                   ("Q", p1["Q"][1], eg * p0["Q"][1]), ("tau", p1["tau"][1], eg * p0["tau"][1]),
                   ("lam", p1["lam"][1], eg * p0["lam"][1]), ("gam_y", p1["gam_y"][1], eg * p0["gam_y"][1]),
                   ("gam_z", p1["gam_z"][1], eg * p0["gam_z"][1])]:
    print(f"  {name:6s} {np.abs(a - b).max():.1e}")
print("unchanged: TD's blocks %.1e, sigma %.1e, beta %.1e, theta %.1e, kappa %.1e, omega %.1e" % (
    max(np.abs(p1[k][0] - p0[k][0]).max() for k in ("A", "b", "Q", "tau", "lam", "gam_y", "gam_z")),
    np.abs(p1["sigma"] - p0["sigma"]).max(), np.abs(p1["beta"] - p0["beta"]).max(),
    np.abs(p1["theta"] - p0["theta"]).max(), np.abs(p1["kappa"] - p0["kappa"]).max(), abs(p1["omega"] - p0["omega"])))
print("free log-speed offsets:", p1["log_speed"], "| PIGD own rates, speed-only vs free:",
      np.round(np.diag(p0["A"][1]), 4), np.round(np.diag(p1["A"][1]), 4))

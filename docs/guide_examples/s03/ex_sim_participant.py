"""Sections 3.14 and 3.15: one participant's speed, load and true landmark state as
simulate_from_fit draws them. The landmark posterior is computed here by replaying
simulate.py lines 48-54 and compared with the likelihood's own _landmark_update.
Synthetic data only (scenario S1, 4 domains)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

import numpy as np
import torch
from helpers import small_panel, spec_for
from pdx.engine.fit import fit_model
from pdx.engine.likelihood import _landmark_update, prepare
from pdx.engine.simulate import _treat

np.set_printoptions(precision=4, suppress=True)
panel, info = small_panel("S1", n=80, seed=1, d=4)
spec = spec_for(panel, info)
f0 = fit_model(panel, spec, "null")
par, prep = f0.params, prepare(panel, spec.meas_var, spec.n_groups)
mu1, S1, y1c = prep.mu1.numpy(), prep.S1.numpy(), prep.y1c.numpy()
R = np.diag(spec.meas_var)

i = next(k for k in range(panel.n) if np.isnan(panel.y[k][0]).any())   # a participant with a missing landmark score
g = int(panel.group[i])
eta = par["log_speed"][g] + panel.z[i] @ par["beta"] + y1c[i] @ par["theta"]
print(f"participant {i}, phenotype {g}: eta = {eta:.4f}, sigma_g = {par['sigma'][g]:.4f}")
print("  speed: median %.3f, 10th-90th percentile %.3f to %.3f" % (np.exp(eta), *np.exp(eta + np.array([-1.2816, 1.2816]) * par["sigma"][g])))
mu_u0 = par["gam_y"][g] * y1c[i] + par["gam_z"][g] @ panel.z[i]
print("  load mean at u = 0:", mu_u0, "| per unit of u:", par["lam"][g], "| spread tau_g:", par["tau"][g])

h = _treat(par, spec, panel.dose[i], panel.exposure[i], panel.on[i], g)
yi = panel.y[i]
o = ~np.isnan(yi[0])
m, P = mu1[g].copy(), S1[g].copy()                       # replay of simulate.py lines 48-54
Ho = np.eye(spec.d)[o]
Kg = P @ Ho.T @ np.linalg.inv(Ho @ P @ Ho.T + R[np.ix_(o, o)])
m = m + Kg @ (yi[0, o] - m[o] - h[0, o])
P = P - Kg @ Ho @ P
mt, Pt = _landmark_update(prep.mu1[g][None], prep.S1[g], torch.as_tensor(np.nan_to_num(yi[0]))[None],
                          torch.as_tensor(o.astype(float))[None], torch.as_tensor(h[0])[None], prep.R)
print("  landmark scores:", yi[0], "(observed:", o, ")")
print("  posterior mean of the true landmark state:", m)
print("  posterior sd:", np.sqrt(np.diag(P)), "| prior sd:", np.sqrt(np.diag(S1[g])))
print("  against the likelihood's _landmark_update: mean %.1e, covariance %.1e"
      % (np.abs(m - mt[0].numpy()).max(), np.abs(P - Pt[0].numpy()).max()))

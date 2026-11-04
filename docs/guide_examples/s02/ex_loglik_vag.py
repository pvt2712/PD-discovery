"""Sections 2.36-2.37: loglik, and value_and_grad against central finite differences."""
import numpy as np
import torch

from _setup import s1_panel
from helpers import layout, random_point, spec_for
from pdx.engine.fit import param_names
from pdx.engine.likelihood import loglik, participant_loglik, prepare, value_and_grad

panel, info = s1_panel(n=30, seed=1, d=4)
spec = spec_for(panel, info, treatment="emax")
prep = prepare(panel, spec.meas_var, spec.n_groups)
lay = layout(spec, "null")
x = random_point(panel, spec, "null", seed=11)
total = float(loglik(torch.as_tensor(x), lay, prep))
by_one = sum(float(participant_loglik(torch.as_tensor(x), lay, prep, torch.tensor([i]))) for i in range(prep.N))
print(f"loglik = {total:.10f}; sum of one-participant calls = {by_one:.10f}")
f0, g = value_and_grad(x, lay, prep)
print(f"value_and_grad: value = {f0:.10f} (= -loglik), gradient of length {len(g)}")
names = param_names(lay)
print("\n coordinate        autograd       central difference (h = 1e-6)   relative gap")
for name in ("A[2]", "A[7]", "gamma[0]", "beta[1]", "theta[0]", "log_sigma[1]", "b[3]", "L[0]", "L[13]",
             "gam_y[2]", "gam_z[5]", "lam[1]", "log_tau[4]", "omega[0]", "kappa[0]", "kappa_ldr[3]"):
    j = names.index(name)
    e = np.zeros(len(x))
    e[j] = 1e-6
    fd = (value_and_grad(x + e, lay, prep)[0] - value_and_grad(x - e, lay, prep)[0]) / 2e-6
    print(f" {name:14s} {g[j]:15.8f}   {fd:15.8f}                {abs(fd - g[j]) / max(1.0, abs(g[j])):.1e}")

pen = lambda v: 0.5 * (v ** 2).sum()             # noqa: E731
f1, g1 = value_and_grad(x, lay, prep, penalty=pen)
print(f"\nwith penalty 0.5 |x|^2: value rises by {f1 - f0:.6f} (0.5 |x|^2 = {0.5 * (x ** 2).sum():.6f}); "
      f"max |gradient change - x| = {np.abs((g1 - g) - x).max():.1e}")
xbad = x.copy()
xbad[lay.slices["A"][0].start + 2] = np.nan
fb, gb = value_and_grad(xbad, lay, prep)
print(f"at a point where the model cannot be evaluated: value {fb:.0e}, |gradient| {np.abs(gb).max()}")

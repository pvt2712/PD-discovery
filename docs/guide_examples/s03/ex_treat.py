"""Section 3.13: simulate._treat, the treatment terms h of the measurement equation, against the
likelihood's own version (model.treat_effect). Toy numbers only."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

import numpy as np
import torch
from pdx.engine.model import Layout, ModelSpec, treat_effect
from pdx.engine.simulate import _treat

np.set_printoptions(precision=4, suppress=True)
spec = ModelSpec(d=4, support=np.eye(4, dtype=bool), treat_mask=[True, True, False, False], treatment="emax")
par = {"kappa": np.array([[0.5, 0.5], [0.4, 0.3]]),           # (G, n_treat): short-duration maxima
       "kappa_ldr": np.array([[0.25, 0.25], [0.2, 0.1]])}     # long-duration maxima
dose = np.array([0.0, 3.0, 4.5, 6.0])                         # 100 mg LEDD at each visit
expo = np.array([0.0, 1.5, 3.0, 4.5])                         # long-duration exposure
on = np.array([0.0, 1.0, 0.0, 1.0])                           # examination in the ON state?
h = _treat(par, spec, dose, expo, on, g=0)
print("h for a TD participant (visits x domains):\n", h)
print("by hand, visit 2, domain 1: -(0.5 * 3/(3+3) * 1 + 0.25 * 1.5/(3+1.5)) = %.4f" % -(0.5 * 0.5 + 0.25 * 1.5 / 4.5))

lay = Layout(spec, "null")
x = np.zeros(lay.size)
lay.put(x, "kappa", par["kappa"])
lay.put(x, "kappa_ldr", par["kappa_ldr"])
t = lambda a: torch.as_tensor(a)[None, :]
for g in (0, 1):
    h_lik = treat_effect(torch.as_tensor(x), lay, t(dose), t(expo), t(on), torch.tensor([g]))[0].numpy()
    print(f"group {g}: max |_treat - model.treat_effect| = {np.abs(_treat(par, spec, dose, expo, on, g) - h_lik).max():.1e}")
spec_lin = ModelSpec(d=4, support=np.eye(4, dtype=bool), treat_mask=[True, True, False, False], treatment="linear")
print("linear form, visit 2, domain 1: %.4f (= -(0.5 * 3 + 0.25 * 1.5))" % _treat(par, spec_lin, dose, expo, on, 0)[1, 0])

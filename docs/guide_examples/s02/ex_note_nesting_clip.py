"""Notes for the author: pack_drift moves own rates smaller than 1e-4 in size to -1e-4, so the null
fit written in the free model's parameters (fit.null_to) is no longer the same model when a
phenotype's own rate, at its speed, is that small."""
import numpy as np
import torch

from _setup import s1_panel
from helpers import layout, random_point, spec_for
from pdx.engine.fit import null_to
from pdx.engine.likelihood import loglik, prepare

panel, info = s1_panel(n=60, seed=1, d=4)
spec = spec_for(panel, info)
prep = prepare(panel, spec.meas_var, spec.n_groups)
lay0, lay1 = layout(spec, "null"), layout(spec, "free")
x0 = random_point(panel, spec, "null", seed=5)
x0[lay0.slices["gamma"][0]] = np.log(1.8)        # PIGD speed 1.8
pos = lay0.slices["A"][0].start + spec.diag_pos[0]
print("own rate of domain 0 (reference speed) | x 1.8 (PIGD) | loglik null | loglik free from null | difference")
for raw in (np.log(0.25), -9.0, -10.0, -12.0):
    x = x0.copy()
    x[pos] = raw
    l0 = float(loglik(torch.as_tensor(x), lay0, prep))
    l1 = float(loglik(torch.as_tensor(null_to(x, spec, "free")), lay1, prep))
    print(f"   {-np.exp(raw):10.2e}                    {-1.8 * np.exp(raw):10.2e}   {l0:.8f}   {l1:.8f}   {l1 - l0:+.1e}")

"""Sections 2.1-2.3: the float64 default, ModelSpec and Layout."""
import torch

print("torch default dtype before importing the engine:", torch.get_default_dtype())
from _setup import s1_panel                 # noqa: E402  (imports pdx.engine.model)
print("torch default dtype after importing pdx.engine.model:", torch.get_default_dtype())

import numpy as np                          # noqa: E402

from helpers import spec_for                # noqa: E402
from pdx.engine.model import KINDS, Layout, ModelSpec  # noqa: E402

panel, info = s1_panel(n=60, seed=1, d=4)
names = info["truth"].domains
spec = spec_for(panel, info)
print("\nsupport [to, from] (1 = drift entry allowed), domains", names)
print(spec.support.astype(int))
print("sup_idx (to, from), row-major:", spec.sup_idx.tolist())
print("diag_pos:", spec.diag_pos.tolist(), "| n_sup:", spec.n_sup, "| treat_mask:", spec.treat_mask.astype(int).tolist(),
      "| n_treat:", spec.n_treat, "| noise_era_col:", spec.noise_era_col)

# __post_init__: the diagonal is always added; "emax" with no treated domain becomes "none"
s2 = ModelSpec(d=3, support=np.zeros((3, 3)), treatment="emax")
print("\nModelSpec(d=3, support=zeros, treatment='emax'): support diagonal =", np.diag(s2.support).tolist(),
      "| treatment =", s2.treatment, "| meas_var =", s2.meas_var.tolist())

for kind in KINDS:
    lay = Layout(spec, kind)
    print(f"\nLayout '{kind}': {lay.size} parameters")
    for name, shape in lay.blocks:
        sl = lay.slices[name][0]
        if sl.stop > sl.start:
            print(f"  {name:10s} [{sl.start:3d}:{sl.stop:3d}) shape {shape}")
        else:
            print(f"  {name:10s} (empty)  shape {shape}")

lay = Layout(spec, "null")
x = np.arange(lay.size, dtype=float)
print("\nget(x, 'b') on x = 0, 1, 2, ...:\n", lay.get(x, "b"))
lay.put(x, "log_sigma", [np.log(0.3), np.log(0.5)])
print("after put(x, 'log_sigma', ...):", lay.get(x, "log_sigma"), "| has('omega'):", lay.has("omega"),
      "| has('gamma'):", lay.has("gamma"))

"""Sections 2.9-2.10: unpack_numpy, and pack_drift as the inverse of _fill."""
import numpy as np

from _setup import s1_panel
from helpers import layout, random_point, spec_for
from pdx.engine.model import pack_drift, unpack_numpy

panel, info = s1_panel(n=60, seed=1, d=4)
spec = spec_for(panel, info)
for kind in ("null", "free", "joint"):
    lay = layout(spec, kind)
    par = unpack_numpy(random_point(panel, spec, kind, seed=7), lay)
    shapes = {k: (np.shape(v) if np.ndim(v) else "scalar") for k, v in par.items()}
    print(f"{kind}: {shapes}")

lay = layout(spec, "free")
x = random_point(panel, spec, "free", seed=7)
par = unpack_numpy(x, lay)
print("\nA of phenotype 0 (reference speed):\n", par["A"][0])
back = pack_drift(par["A"][0], spec)
print("pack_drift(A_0) - raw block 'A':", np.abs(back - lay.get(x, "A")).max())

A = par["A"][0].copy()
A[0, 0] = +0.02                                  # a positive own rate cannot be represented
v = pack_drift(A, spec)
print(f"own rate +0.02 -> raw {v[spec.diag_pos[0]]:.4f} -> own rate {-np.exp(v[spec.diag_pos[0]]):.1e}")
A[0, 0] = -4e-5                                  # an own rate smaller than 1e-4 in size is moved to -1e-4
v = pack_drift(A, spec)
print(f"own rate -4e-05 -> raw {v[spec.diag_pos[0]]:.4f} -> own rate {-np.exp(v[spec.diag_pos[0]]):.1e}")

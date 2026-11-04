"""Sections 2.4-2.6: _fill, drifts and log_speeds on a hand-made three-domain model."""
import numpy as np
import torch

import _setup                                    # noqa: F401
from pdx.engine.model import Layout, ModelSpec, _fill, drifts, log_speeds
from pdx.engine.transitions import is_clearly_stable

sup = np.zeros((3, 3), bool)
sup[1, 0] = sup[2, 1] = True                     # domain 0 -> 1 and 1 -> 2 (stored [to, from])
spec = ModelSpec(d=3, support=sup)
print("sup_idx (to, from):", spec.sup_idx.tolist(), " diag_pos:", spec.diag_pos.tolist())

# raw support values in sup_idx order: (0,0) (1,0) (1,1) (2,1) (2,2)
raw_A = torch.tensor([np.log(0.25), 0.20, np.log(0.50), 0.30, np.log(0.10)])
print("raw A values:", raw_A.numpy().round(4))
print("_fill(raw, constrain_diag=True):\n", _fill(raw_A, spec, True).numpy())
print("_fill(raw, constrain_diag=False) (how a deviation D_g is filled):\n", _fill(raw_A, spec, False).numpy())

for kind in ("null", "free", "joint"):
    lay = Layout(spec, kind)
    x = torch.zeros(lay.size)
    x[lay.slices["A"][0]] = raw_A
    if kind == "free":                           # PIGD's own drift: twice TD's, written in raw form
        x[lay.slices["A1"][0]] = torch.tensor([np.log(0.50), 0.40, np.log(1.00), 0.60, np.log(0.20)])
    if kind == "joint":                          # a deviation on one link and on one own rate
        x[lay.slices["D1"][0]] = torch.tensor([0.0, 0.0, 0.0, 0.15, 0.30])
    if kind in ("null", "joint"):
        x[lay.slices["gamma"][0]] = np.log(1.8)
    AG = drifts(x, lay)
    print(f"\n{kind}: log_speeds = {log_speeds(x, lay).numpy().round(4)}")
    print(" drift of phenotype 1 at the reference speed:\n", AG[1].numpy())
    eig = np.linalg.eigvals(AG[1].numpy()).real
    print(" its eigenvalues (real parts):", np.sort(eig).round(4), "| clearly stable:", is_clearly_stable(AG[1]))

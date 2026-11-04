"""Section 2.0: the engine's log-likelihood on a small synthetic panel, against the reference."""
import time

import numpy as np
import torch

from _setup import s1_panel
from helpers import layout, random_point, spec_for
from pdx.engine.likelihood import loglik, prepare
from pdx.engine.reference import reference_loglik

panel, info = s1_panel(n=60, seed=1, d=4)
print("synthetic S1 panel, domains:", info["truth"].domains)
for g, row in panel.aggregate_summary().items():
    print(f"  phenotype {g}: {row['participants']} participants, {row['visits']} visits, "
          f"{row['transitions']} transitions, {row['share_missing']:.3f} of scores missing")
spec = spec_for(panel, info)            # Emax treatment, era shifts the noise, 12 nodes
print(f"d = {spec.d}, p = {spec.p}, support entries = {spec.n_sup}, treated domains = {spec.n_treat}, "
      f"nodes = {spec.n_quad}, treatment = {spec.treatment}")
prep = prepare(panel, spec.meas_var, spec.n_groups)
print("Prepared tensors: Y", tuple(prep.Y.shape), " dt", tuple(prep.dt.shape),
      " mu1", tuple(prep.mu1.shape), " S1", tuple(prep.S1.shape))
print(f"\n{'kind':6s} {'params':>6s} {'engine':>16s} {'reference':>16s} {'|diff|':>9s} {'engine':>8s} {'reference':>9s}")
for kind in ("null", "free", "joint"):
    lay = layout(spec, kind)
    x = random_point(panel, spec, kind, seed=7)
    t0 = time.perf_counter()
    fast = float(loglik(torch.as_tensor(x), lay, prep))
    t1 = time.perf_counter()
    ref = reference_loglik(x, lay, panel, prep.mu1.numpy(), prep.S1.numpy(), prep.y1c.numpy())
    t2 = time.perf_counter()
    print(f"{kind:6s} {lay.size:6d} {fast:16.10f} {ref:16.10f} {abs(fast - ref):9.1e} {t1 - t0:7.3f}s {t2 - t1:8.2f}s")

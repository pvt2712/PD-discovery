"""Sections 2.38-2.39: the reference transitions (_van_loan) and reference_loglik against the engine."""
import time

import numpy as np
import torch

from _setup import s1_panel, with_gaps
from helpers import layout, random_point, spec_for
from pdx.engine.likelihood import loglik, prepare
from pdx.engine.reference import _van_loan, reference_loglik
from pdx.engine.transitions import transitions_grouped

Q = np.array([[0.10, 0.02], [0.02, 0.08]])
for name, A in (("stable", np.array([[-0.25, 0.0], [0.20, -0.50]])),
                ("marginal", np.array([[-0.1, 0.1], [0.1, -0.1]]))):
    route = "stationary" if np.max(np.linalg.eigvals(A).real) < -1e-6 else "Van Loan block"
    for c in (0.5, 5.0, 50.0):
        E, M, S = _van_loan(A, Q, c)
        Eg, Mg, Sg = transitions_grouped(torch.as_tensor(A)[None], torch.as_tensor(Q)[None],
                                         torch.zeros(1, dtype=torch.long), torch.tensor([[[c]]]))
        print(f"{name:8s} ({route:14s}) c = {c:4}: max |reference - engine|: E {np.abs(E - Eg[0, 0, 0].numpy()).max():.1e}"
              f"  M {np.abs(M - Mg[0, 0, 0].numpy()).max():.1e}  S {np.abs(S - Sg[0, 0, 0].numpy()).max():.1e}")

panel, info = s1_panel(n=40, seed=1, d=4)
panel = with_gaps(panel)
miss = np.mean([np.isnan(yi).mean() for yi in panel.y])
print(f"\nsynthetic panel with extra gaps: {panel.n} participants, {miss:.3f} of scores missing")
print("treatment  kind    engine            reference         |diff|     reference time")
for treatment in ("linear", "emax"):
    spec = spec_for(panel, info, treatment=treatment)
    prep = prepare(panel, spec.meas_var, spec.n_groups)
    for kind in ("null", "free", "joint"):
        lay = layout(spec, kind)
        x = random_point(panel, spec, kind, seed=7)
        fast = float(loglik(torch.as_tensor(x), lay, prep))
        t0 = time.perf_counter()
        ref = reference_loglik(x, lay, panel, prep.mu1.numpy(), prep.S1.numpy(), prep.y1c.numpy())
        print(f"{treatment:9s}  {kind:6s} {fast:16.10f}  {ref:16.10f}  {abs(fast - ref):.1e}    {time.perf_counter() - t0:.2f} s")

panel10, info10 = s1_panel(n=40, seed=4, d=10)
panel10 = panel10.subset(np.arange(14))
spec10 = spec_for(panel10, info10, treatment="emax")
prep10 = prepare(panel10, spec10.meas_var, spec10.n_groups)
lay10 = layout(spec10, "free")
x10 = random_point(panel10, spec10, "free", seed=3)
fast = float(loglik(torch.as_tensor(x10), lay10, prep10, chunk=5))
ref = reference_loglik(x10, lay10, panel10, prep10.mu1.numpy(), prep10.S1.numpy(), prep10.y1c.numpy())
print(f"\nten domains, 14 participants, free model, chunks of 5: engine {fast:.10f}, reference {ref:.10f}, |diff| {abs(fast - ref):.1e}")

# an unstable phenotype drift in the joint model (a deviation of +0.5 on one own rate), engine against
# reference; the reference then uses Van Loan's block, accurate at these moderate intervals
from pdx.engine.model import drifts                          # noqa: E402

spec = spec_for(panel, info, treatment="emax")
prep = prepare(panel, spec.meas_var, spec.n_groups)
lay = layout(spec, "joint")
x = random_point(panel, spec, "joint", seed=7)
x[lay.slices["D1"][0].start + spec.diag_pos[1]] = 0.5
A1 = drifts(torch.as_tensor(x), lay)[1].numpy()
fast = float(loglik(torch.as_tensor(x), lay, prep))
ref = reference_loglik(x, lay, panel, prep.mu1.numpy(), prep.S1.numpy(), prep.y1c.numpy())
print(f"\njoint model, PIGD drift with largest real eigenvalue {np.linalg.eigvals(A1).real.max():+.3f}: "
      f"engine {fast:.10f}, reference {ref:.10f}, |diff| {abs(fast - ref):.1e}")

"""Sections 3.14 to 3.16: simulate_from_fit, the conditional parametric bootstrap's simulator.
What it keeps, what it redraws, its exact transitions, and a score check: data simulated from
a fit give a mean log-likelihood gradient of zero at that fit. Synthetic data only (scenario S1)."""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

import numpy as np
from helpers import small_panel, spec_for
from pdx.engine.fit import fit_model
from pdx.engine.likelihood import prepare, value_and_grad
from pdx.engine.reference import _van_loan
from pdx.engine.simulate import simulate_from_fit
from pdx.engine.transitions import transition_np

panel, info = small_panel("S1", n=80, seed=1, d=4)
spec = spec_for(panel, info)
f0 = fit_model(panel, spec, "null")

# 1. what is kept
sim = simulate_from_fit(f0, panel, np.random.default_rng(0))
same_nan = all(np.array_equal(np.isnan(a), np.isnan(b)) for a, b in zip(sim.y, panel.y))
same_y1 = all(np.array_equal(a[0], b[0], equal_nan=True) for a, b in zip(sim.y, panel.y))
same_t = all(np.array_equal(a, b) for a, b in zip(sim.t, panel.t))
later = np.concatenate([a[1:].ravel() for a in sim.y]), np.concatenate([b[1:].ravel() for b in panel.y])
print("kept: landmark rows %s, visit times %s, missing pattern %s, groups %s, covariates %s, doses %s"
      % (same_y1, same_t, same_nan, np.array_equal(sim.group, panel.group), np.array_equal(sim.z, panel.z),
         all(np.array_equal(a, b) for a, b in zip(sim.dose, panel.dose))))
print("redrawn: later visits equal to the observed ones: %d of %d" % (np.sum(later[0] == later[1]), np.sum(~np.isnan(later[1]))))

# 2. the transition over one interval is exact: transition_np against the independent reference
par = f0.params
A, Q, b = par["A"][1], par["Q"][1], par["b"][1]
E, M, S = transition_np(A, Q, 1.3, 0.9)
E2, M2, S2 = _van_loan(A, Q, 1.3 * 0.9)
print("transition_np vs reference (speed 1.3, interval 0.9 y): max differences E %.1e, M %.1e, S %.1e"
      % (np.abs(E - E2).max(), np.abs(M - M2).max(), np.abs(S - S2).max()))

# 3. a summary statistic: mean yearly change from the landmark to the last visit, per phenotype
def yearly_change(p, g):
    rows = [(yi[-1] - yi[0]) / (ti[-1] - ti[0]) for yi, ti, gi in zip(p.y, p.t, p.group) if gi == g]
    return np.nanmean(np.array(rows), 0)

t0 = time.time()
R = 100
sims = [simulate_from_fit(f0, panel, np.random.default_rng([7, r])) for r in range(R)]
t_sim = (time.time() - t0) / R
for g, name in ((0, "TD"), (1, "PIGD")):
    s = np.array([yearly_change(p, g) for p in sims])
    print(f"{name:4s} yearly change, observed      :", np.round(yearly_change(panel, g), 3))
    print(f"{name:4s} yearly change, simulated mean:", np.round(s.mean(0), 3), " (sd over datasets", np.round(s.std(0), 3), ")")

# 4. the score check: E[gradient of the log-likelihood at the simulating parameters] = 0
t0 = time.time()
G = np.array([-value_and_grad(f0.x, f0.layout, prepare(p, spec.meas_var, spec.n_groups))[1] for p in sims])
sd = G.std(0, ddof=1)
use = sd > 1e-8 * np.abs(G).max()
z = G.mean(0)[use] / (sd[use] / np.sqrt(R))
print("score check over %d datasets, %d coordinates (of %d; the rest have no variation): mean z^2 = %.2f,"
      " share |z| > 1.96 = %.3f, max |z| = %.2f" % (R, use.sum(), len(use), np.mean(z ** 2), np.mean(np.abs(z) > 1.96), np.abs(z).max()))
print("observed data at the fit: max |gradient| = %.2e" % np.abs(value_and_grad(f0.x, f0.layout, prepare(panel, spec.meas_var, spec.n_groups))[1]).max())
print("%.2f s per simulated dataset, %.2f s per gradient" % (t_sim, (time.time() - t0) / R))

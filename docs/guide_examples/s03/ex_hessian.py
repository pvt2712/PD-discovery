"""Sections 3.9, 3.11 and 3.12: the observed information by central differences of the exact
gradient, its checks, and standard errors. Synthetic data only (scenario S1, 4 domains)."""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

import numpy as np
from helpers import small_panel, spec_for
from pdx.engine.fit import covariance_checked, fit_model, hessian, param_names
from pdx.engine.likelihood import prepare, value_and_grad

panel, info = small_panel("S1", n=80, seed=1, d=4)
spec = spec_for(panel, info)
prep = prepare(panel, spec.meas_var, spec.n_groups)
f0 = fit_model(panel, spec, "null", prep=prep)
lay, n = f0.layout, f0.layout.size

# the loop of hessian(), repeated here only to look at H before it is symmetrised
t0 = time.time()
Hraw = np.zeros((n, n))
for j in range(n):
    e = np.zeros(n)
    e[j] = 1e-5
    Hraw[:, j] = (value_and_grad(f0.x + e, lay, prep)[1] - value_and_grad(f0.x - e, lay, prep)[1]) / 2e-5
H = hessian(f0, panel, prep)
print("%d coordinates, %d gradient evaluations per Hessian, %.0f s per Hessian" % (n, 2 * n, (time.time() - t0) / 2))
print("hessian() equals the symmetrised loop: max difference %.1e" % np.abs(H - 0.5 * (Hraw + Hraw.T)).max())
print("asymmetry before symmetrising: max |H - H'| / max |H| = %.1e" % (np.abs(Hraw - Hraw.T).max() / np.abs(Hraw).max()))
H4 = hessian(f0, panel, prep, h=1e-4)
print("h = 1e-4 against h = 1e-5: max |difference| / max |H| = %.1e" % (np.abs(H4 - H).max() / np.abs(H).max()))

names = param_names(lay)
V, rep = covariance_checked(H, names=names)
print("\nreport: min eigenvalue %.3e, condition number %.3e, positive definite %s, well conditioned %s"
      % (rep["min_eigenvalue"], rep["condition_number"], rep["positive_definite"], rep["well_conditioned"]))
print("flat directions:", rep["flat_directions"])
w = np.linalg.eigvalsh(H)
print("four smallest eigenvalues:", ["%.1e" % v for v in w[:4]], "| largest: %.1f" % w[-1])
lo = np.argsort(np.diag(H))[:3]
print("smallest diagonal information:", [(names[i], "%.1e" % H[i, i], "tau = %.1e" % np.exp(f0.x[i])) for i in lo])
keep = [i for i, nm in enumerate(names) if not (nm.startswith(("log_tau", "log_sigma")) and f0.x[i] < np.log(0.01))]
_, rest = covariance_checked(H[np.ix_(keep, keep)])
print("without the %d spreads below 0.01 (as gate.identification does): condition number %.3e, well conditioned %s"
      % (n - len(keep), rest["condition_number"], rest["well_conditioned"]))

se = np.sqrt(np.diag(V))
for nm in ("gamma[0]", "log_sigma[0]", "log_sigma[1]", "log_tau[0]", "omega[0]"):
    j = names.index(nm)
    print(f"  {nm:13s} estimate {f0.x[j]:+.4f}  standard error {se[j]:.4f}")
g = value_and_grad(f0.x, lay, prep)[1]
print("distance to the exact optimum (Newton step), 0.5 g' I^-1 g = %.1e log-likelihood units" % (0.5 * g @ V @ g))
print("param_names examples:", names[0], names[8], names[29], names[100], "| total", len(names))

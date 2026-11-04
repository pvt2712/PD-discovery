"""Sections 5.1-5.4 and 5.8: H-P's statistic, its degrees of freedom and the nesting of the two
models, on one small panel simulated from the speed-only engine (so H-P's null is true).
Synthetic data only."""
import time

import numpy as np
from scipy import stats

from _fixture import engine_panel
from pdx import hp
from pdx.engine.fit import null_to
from pdx.engine.likelihood import prepare, value_and_grad
from pdx.engine.model import Layout

panel, spec, truth = engine_panel("null", n=80, seed=3)
print("participants by phenotype:", np.bincount(panel.group), "| transitions:", panel.n_transitions())
print("support (rows = to, columns = from):\n", spec.support.astype(int))
n_null, n_free = Layout(spec, "null").size, Layout(spec, "free").size
print(f"parameters: null {n_null}, free {n_free}, difference {n_free - n_null}; "
      f"(G - 1)(n_sup - 1) = (2 - 1)({spec.n_sup} - 1) = {hp.h_p_df(spec)}")

t0 = time.time()
spec_s, f0, f1 = hp.observed_fits(panel, spec, support_fn=None, fit_kw={"own_free_start": False})
print(f"observed fits: {time.time() - t0:.0f} s; converged: null {f0.success}, free {f1.success}")
prep = prepare(panel, spec.meas_var, spec.n_groups)
ll_at_null = -value_and_grad(null_to(f0.x, spec, "free"), Layout(spec, "free"), prep)[0]
print(f"log-likelihood: null fit {f0.loglik:.4f}; free model at the null solution {ll_at_null:.4f}; "
      f"free fit {f1.loglik:.4f}")
print(f"LR raw = 2 x ({f1.loglik:.4f} - ({f0.loglik:.4f})) = {2 * (f1.loglik - f0.loglik):.4f}; "
      f"lr_stat = {hp.lr_stat(f0, f1):.4f}")
print(f"chi-square({hp.h_p_df(spec_s)}) p-value, for orientation only: "
      f"{stats.chi2.sf(hp.lr_stat(f0, f1), hp.h_p_df(spec_s)):.3f}")
print("speed-only fit: PIGD speed exp(gamma) =", round(float(np.exp(f0.params["log_speed"][1])), 3), "(truth 2.0)")
print("free fit, TD drift A0:\n", np.round(f1.params["A"][0], 3))
print("free fit, PIGD drift A1 (observed time scale):\n", np.round(f1.params["A"][1], 3))

"""Notes for the author: value_and_grad's failure value (1e10 with a zero gradient) is read as
convergence by L-BFGS-B when it occurs at the start point. Two synthetic ways to get there."""
import warnings

import numpy as np

from _setup import s1_panel
from helpers import layout, random_point, spec_for
from pdx.engine.fit import fit_model
from pdx.engine.likelihood import prepare, value_and_grad

panel, info = s1_panel(n=60, seed=1, d=4)
spec = spec_for(panel, info)
prep = prepare(panel, spec.meas_var, spec.n_groups)
lay = layout(spec, "null")

x = random_point(panel, spec, "null", seed=5)
x[0] = np.nan                                    # (1) a start the model cannot evaluate
f = fit_model(panel, spec, "null", x0=x, prep=prep, maxiter=50)
print(f"(1) start with a NaN entry: success={f.success}, loglik={f.loglik:.0e}, evaluations={f.n_eval}")
print("    message:", f.message)

y = [yi.copy() for yi in panel.y]                # (2) one domain never observed at the landmark in PIGD
for i in np.where(panel.group == 1)[0]:
    y[i][0, 0] = np.nan
p2 = panel.with_y(y)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    prep2 = prepare(p2, spec.meas_var, spec.n_groups)
print(f"(2) mu1 of PIGD = {prep2.mu1[1].numpy().round(3)}; warnings: {[str(m.message) for m in w]}")
print("    value at a good start:", value_and_grad(random_point(p2, spec, "null", seed=5), lay, prep2)[0])
f2 = fit_model(p2, spec, "null", prep=prep2, maxiter=50)
print(f"    fit from the default start: success={f2.success}, loglik={f2.loglik:.0e}, evaluations={f2.n_eval}")

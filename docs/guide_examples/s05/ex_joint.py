"""Sections 5.36-5.39: the speed-deviation trade-off in the joint model (free_to_joint), the
anchored per-link tests and speed-adjusted joint discovery (adaptive lasso, BIC, refit). The
panel is simulated from the engine with PIGD at speed 2 and one link only PIGD has beyond speed
(axial -> independence, Delta = 0.30). Synthetic data only."""
import time

import numpy as np

from _fixture import engine_panel
from pdx.engine.fit import fit_pair
from pdx.engine.likelihood import prepare, value_and_grad
from pdx.engine.model import Layout
from pdx.localise import anchored_tests, free_to_joint, joint_discovery, speed_rule

panel, spec, truth = engine_panel("free", n=100, seed=4, extra={(3, 1): 0.30})
t0 = time.time()
f0, f1 = fit_pair(panel, spec, own_free_start=False)
prep = prepare(panel, spec.meas_var, spec.n_groups)
print(f"n = {panel.n}; fits {time.time() - t0:.0f} s; free log-likelihood {f1.loglik:.4f}")
lay = Layout(spec, "joint")
dsl = lay.slices["D1"][0]
links = [tuple(map(int, r)) for r in spec.sup_idx]
print("links (to, from):", links)
for s in (1.0, 1.5, 2.5):
    x = free_to_joint(f1, s)
    ll = -value_and_grad(x, lay, prep)[0]
    print(f"joint model at speed {s}: log-likelihood {ll:.4f}; D = {np.round(x[dsl], 3)}")

t0 = time.time()
anc = anchored_tests(panel, spec, f1, prep=prep)
print(f"\nanchored ({time.time() - t0:.0f} s): speed {anc['speed']:.3f}, converged {anc['converged']}")
for k, (d, se, p) in enumerate(zip(anc["delta"], anc["se"], anc["p"])):
    if links[k][0] != links[k][1]:
        print(f"  {links[k]}: D {d:7.3f}  se {se:.3f}  p {p:.4f}")
print("  per-domain Wald p:", np.round(anc["p_domain"], 4))

t0 = time.time()
a0, a1 = f1.params["A"][0][spec.support], f1.params["A"][1][spec.support]
s = speed_rule(a0, a1)
w = 1.0 / np.maximum(np.abs(a1 / s - a0), 1e-4)
print(f"\nadaptive weights at the minimal-L1 speed {s:.3f}:", np.round(w, 1))
jd = joint_discovery(panel, spec, f1, f0, prep=prep, n_lambda=4)
print(f"joint discovery ({time.time() - t0:.0f} s): lambda_max {jd['lambda_max']:.3f}")
for k, step in enumerate(jd["path"]):
    print(f"  lambda {step['lambda']:8.4f}  loglik {step['loglik']:9.3f}  active {np.array(step['active']).astype(int)}"
          f"  BIC {step['bic']:8.2f}{'  <- best' if k == jd['best'] else ''}")
print("  selected:", [links[k] for k in np.where(jd["selected"])[0]], "| identified:", jd["identified"])
if "refit" in jd:
    print(f"  refit: speed {jd['refit']['speed']:.3f}, D = {np.round(jd['refit']['D'], 3)}, converged {jd['refit']['converged']}")

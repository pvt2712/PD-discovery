"""Sections 5.25-5.34 on a real (small) free fit: Monte Carlo draws from the fit's asymptotic normal
distribution, the anchor-free per-link tests, the classes, the per-domain tests and the
intersection-union verdict. The panel is simulated from the engine with PIGD at speed 2 and one
link that only PIGD has beyond speed: axial -> independence, Delta = 0.30 (three epsilons).
Synthetic data only; the free model's second start is skipped to keep the example short."""
import time

import numpy as np

from _fixture import DOMAINS, engine_panel
from pdx.engine.fit import fit_pair
from pdx.engine.likelihood import prepare
from pdx.localise import (_basic_ci, classify_links, free_draws, per_domain_tests, per_link_tests,
                          speed_and_load_only)

panel, spec, truth = engine_panel("free", n=160, seed=4, extra={(3, 1): 0.30})
t0 = time.time()
f0, f1 = fit_pair(panel, spec, own_free_start=False)
prep = prepare(panel, spec.meas_var, spec.n_groups)
dr = free_draws(f1, panel, M=2000, rng=np.random.default_rng(1), prep=prep)
print(f"n = {panel.n}; fits and Hessian: {time.time() - t0:.0f} s; information: "
      f"positive definite {dr.cov_info['positive_definite']}, condition number {dr.cov_info['condition_number']:.3g}")
links = [tuple(map(int, r)) for r in spec.sup_idx]
print("links (to, from):", links)
print("a0 (TD)     :", np.round(dr.a0, 3))
print("a1 (PIGD)   :", np.round(dr.a1, 3))
own = spec.diag_pos
print("own rates: estimate", np.round(dr.a0[own], 3), "| mean of draws", np.round(dr.d0[:, own].mean(0), 3),
      "(draws are -exp(normal), so not centred on the estimate)")
lo_b, hi_b = _basic_ci(dr.a0, dr.d0, 0.95)
lo_p, hi_p = np.quantile(dr.d0, [0.025, 0.975], axis=0)
print(f"95% interval of TD's tremor own rate: basic [{lo_b[0]:.3f}, {hi_b[0]:.3f}], percentile [{lo_p[0]:.3f}, {hi_p[0]:.3f}]")

res = per_link_tests(f1, panel, draws=dr)
print(f"\nminimal-L1 speed {res.s_hat:.3f}; speed set ({res.speed_set[0]:.3f}, {res.speed_set[1]:.3f}); "
      f"{res.extra['draws_used']} of 2000 draws identify the speed")
print("diagnostics:", {k: round(v, 3) for k, v in res.diagnostics.items()})
true_delta = (truth["A1"] / truth["speed"] - truth["A"])[spec.support]
for c, td in zip(classify_links(res, eps=0.10, method="by"), true_delta):
    if c["class"] == "outside the family":
        continue
    frm, to = DOMAINS[c["link"][1]], DOMAINS[c["link"][0]]
    print(f"  {frm:>6} -> {to:<12} Delta {c['delta']:7.3f} (true {td:5.2f})  p {c['p']:.4f}  "
          f"ci90 [{c['ci90'][0]:6.3f},{c['ci90'][1]:6.3f}]  {c['verdict']}")
dom = per_domain_tests(res, spec)
print("per-domain p:", np.round(dom["p"], 4), dom["verdict"])
print("speed and load only:", speed_and_load_only(res, 0.10))

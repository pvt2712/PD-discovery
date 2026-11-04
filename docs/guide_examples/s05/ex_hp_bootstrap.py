"""Sections 5.5, 5.6 and 5.9: a tiny H-P with support re-selection and a 3-replicate conditional
parametric bootstrap (h = 2), run the way scripts/run_hp_real.py runs it (boot_setup and
boot_replicate through parallel.ordered). Synthetic data only; the free model's second start is
skipped (own_free_start=False) to keep the example short."""
import time

import numpy as np

from _fixture import engine_panel
from pdx import hp, parallel
from pdx.engine.simulate import simulate_from_fit
from pdx.step1 import Step1Config, make_support_fn

panel, spec, truth = engine_panel("null", n=80, seed=3)
cfg = Step1Config(depth=2)
expert = spec.support.copy()
support_fn = make_support_fn(expert, cfg)
fit_kw = {"own_free_start": False}
t0 = time.time()
fits = hp.observed_fits(panel, spec, support_fn, fit_kw)
spec_s, f0, f1 = fits
print(f"observed: selected support adds {int(spec_s.support.sum() - expert.sum())} link(s) to the map; "
      f"LR {hp.lr_stat(f0, f1):.3f} on {hp.h_p_df(spec_s)} df ({time.time() - t0:.0f} s)")
smaller = expert.copy()
smaller[3, 2] = False                                                        # a map without function -> independence
print("with a map lacking (3,2), the selection would add:",
      [tuple(map(int, e)) for e in np.argwhere(make_support_fn(smaller, cfg)(panel) & ~smaller)])

seed = 11
sim0 = simulate_from_fit(f0, panel, np.random.default_rng([seed, 3, 0]))     # replicate 0's dataset
print("replicate i = 0: landmark scores kept:", all(np.array_equal(a[0], b[0], equal_nan=True) for a, b in zip(panel.y, sim0.y)),
      "| missing pattern kept:", all(np.array_equal(np.isnan(a), np.isnan(b)) for a, b in zip(panel.y, sim0.y)),
      "| re-selected support adds", int(support_fn(sim0).sum() - expert.sum()), "link(s)")

t0 = time.time()
draws = parallel.ordered(hp.boot_replicate, range(3), 1, hp.boot_setup, (f0, panel, spec, expert, cfg, fit_kw, seed))
log = []
res = hp.run_hp(panel, spec, support_fn, n_max=3, h=2, fits=fits, draw=lambda: next(draws),
                callback=lambda drawn, exceed, stat, ok: log.append(f"replicate i = {drawn - 1}: LR* {stat:.3f}, "
                                                                     f"ok {ok}, exceedances {exceed}"))
draws.close()
print("\n".join(log), f"({time.time() - t0:.0f} s)")
print("result:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.summary().items()})

shared = hp.boot_setup(f0, panel, spec, expert, cfg, fit_kw, seed)
again = hp.boot_replicate(shared, 1)
print(f"replicate i = 1 run again on its own: LR* {again[0]:.6f} (in the run: {res.lr_star[1]:.6f})")

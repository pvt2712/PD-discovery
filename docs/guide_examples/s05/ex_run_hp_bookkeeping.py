"""Sections 5.1 and 5.9: what run_hp does with the replicates it is given, using stand-in fits
(only their log-likelihoods are read) and a scripted stream of replicate outcomes
(statistic, ok, converged). No model is fitted. Synthetic numbers only."""
from types import SimpleNamespace

import numpy as np

import _fixture  # noqa: F401  (import path)
from pdx import hp
from pdx.engine.model import ModelSpec

sup = np.eye(4, dtype=bool)
sup[2, 0] = sup[2, 1] = sup[3, 2] = sup[3, 1] = True
spec = ModelSpec(d=4, support=sup)
f0 = SimpleNamespace(loglik=-100.0, success=True)                        # stand-in null fit
f1 = SimpleNamespace(loglik=-95.0, success=True, start_logliks=[-95.0, -95.2])
script = [(12.0, True, True),       # above the observed LR of 10: exceedance 1
          (np.inf, False, False),   # a failed replicate: exceedance 2
          (3.0, True, False),       # below, and its fits stopped short of convergence
          (11.0, True, True),       # exceedance 3 = h: stop
          (0.5, True, True)]        # never drawn
it = iter(script)
seen = []
res = hp.run_hp(None, spec, fits=(spec, f0, f1), n_max=999, h=3, draw=lambda: next(it),
                callback=lambda drawn, exceed, stat, ok: seen.append((drawn, exceed, stat, ok)))
print("callback calls (drawn, exceed, statistic, ok):", seen)
print("summary:", res.summary())
print("lr_star:", res.lr_star)

it = iter(script[2:])
resumed = hp.run_hp(None, spec, fits=(spec, f0, f1), n_max=999, h=3, draw=lambda: next(it),
                    resume=[(12.0, True), (np.inf, False)])
print("resumed after two replicates: p", resumed.p, "| failures", resumed.failures,
      "| replicates_not_converged", resumed.non_converged)

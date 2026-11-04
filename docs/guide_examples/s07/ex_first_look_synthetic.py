"""Guide §7.12-7.13: run_hp_real.first_look and sp1_interval on one synthetic dataset (synthetic only).

The panel is family A's S1 scenario (shared mechanism, PIGD twice as fast) with 150 simulated
participants and four domains, so the truth is known: no mechanism difference and a speed ratio
of 2. main() is not called (it needs --approval and the PPMI panel); first_look is called directly.
Two shortcuts keep it to a few minutes, neither of which changes a number: the observed fits and
the speed-only model's Hessian are computed once and handed back when first_look asks for them
again, and the localisation uses 500 Monte Carlo draws instead of the configured 2,000.

    OMP_NUM_THREADS=1 ../ppmi-causal/.venv/bin/python docs/guide_examples/s07/ex_first_look_synthetic.py
"""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
import run_hp_real as R  # noqa: E402
from pdx import config, hp, speed  # noqa: E402
from pdx.engine.fit import covariance_checked, hessian  # noqa: E402
from pdx.qualification import spec_for  # noqa: E402
from pdx.sim import scenarios  # noqa: E402
from pdx.step1 import Step1Config, make_support_fn  # noqa: E402

t0 = time.time()
panel, info = scenarios.generate("S1", 150, np.random.default_rng(3), d=4)
spec = spec_for(panel, info)
names = list(panel.domains)
expert = info["support"].copy()
np.fill_diagonal(expert, False)                  # as expert_support returns it: links only
cfg = config.load()
cfg["localisation"]["mc_draws"] = 500            # shortcut (the configured value is 2,000)
s1 = Step1Config(alpha=cfg["step1"]["alpha"], depth=cfg["step1"]["depth_fast"], test=cfg["step1"]["test_fast"])
support_fn = make_support_fn(expert, s1)
print(f"synthetic panel: {panel.n} participants, groups {np.bincount(panel.group).tolist()}, "
      f"true speed ratio {info['speed_ratio']}")

fits = hp.observed_fits(panel, spec, support_fn)
hp.observed_fits = lambda *a, **k: fits          # shortcut: first_look's own call reuses the fits
H = hessian(fits[1], panel)
R.hessian = lambda fit, p: H                     # shortcut: sp1_interval's own call reuses the Hessian
out = R.first_look(panel, spec, support_fn, expert, names, cfg, np.random.default_rng(20261103),
                   lambda m: print("   log:", m))

print("\n1. report keys:", list(out))
print("2. hp:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out["hp"].items()})
print("3. support:", out["support"])
est = out["speed"]
print(f"4. Sp-1 = {est['sp1']:.4f}, model-based 95% interval "
      f"[{est['sp1_interval_model_based'][0]:.4f}, {est['sp1_interval_model_based'][1]:.4f}]")
print("   Sp-2 sigma:", np.round(est["sp2"]["sigma"], 4).tolist(), " Sp-3 share:", round(est["sp3"]["share"], 4))
print("5. link verdicts:", [(c["link"], c["verdict"]) for c in out["links_exploratory"]])
print("   speed and load only:", out["speed_and_load_only"], " domain verdicts:", out["domains"]["verdict"])

# ---------------------------------------------------------------- the delta method by hand
f0 = fits[1]
lay = f0.layout
V, report = covariance_checked(H)
jg = lay.slices["gamma"][0].start
th = lay.slices["theta"][0]
y1c, g = speed._y1c(panel, f0.spec), panel.group
dbar = y1c[g == 1].mean(0) - y1c[g == 0].mean(0)
log_sp1 = f0.params["log_speed"][1] + f0.params["theta"] @ dbar
var = V[jg, jg] + 2 * dbar @ V[th, jg] + dbar @ V[th, th] @ dbar
lo, hi = np.exp(log_sp1 - 1.959964 * np.sqrt(var)), np.exp(log_sp1 + 1.959964 * np.sqrt(var))
print("\n6. by hand: gamma =", round(float(f0.params["log_speed"][1]), 4), " theta =",
      np.round(f0.params["theta"], 4).tolist(), " mean year-1 difference =", np.round(dbar, 4).tolist())
print(f"   log Sp-1 = {log_sp1:.4f}; Var = V_gg + 2 d'V_tg + d'V_tt d = {V[jg, jg]:.5f} + "
      f"{2 * dbar @ V[th, jg]:.5f} + {dbar @ V[th, th] @ dbar:.5f} = {var:.5f}")
print(f"   interval exp(log Sp-1 -+ 1.96 se) = [{lo:.4f}, {hi:.4f}]")
print("   conditioning report that sp1_interval does not pass on:",
      {k: report[k] for k in ("min_eigenvalue", "condition_number", "positive_definite", "well_conditioned")})
print(f"\n{time.time() - t0:.0f} s in all")

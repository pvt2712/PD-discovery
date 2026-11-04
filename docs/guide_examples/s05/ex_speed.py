"""Sections 5.12-5.18: Sp-1, Sp-2 and Sp-3 from the fits, recomputed by hand, and a 3-sample
participant bootstrap. The panel is simulated from the speed-only engine with PIGD at speed 2 and
a landmark effect on log-speed (theta = 0.3 for tremor and axial). Synthetic data only; the free
model's second start is skipped (own_free_start=False) to keep the example short."""
import time

import numpy as np

from _fixture import engine_panel
from pdx import speed
from pdx.engine.fit import fit_pair
from pdx.localise import speed_rule
from pdx.step1 import Step1Config

panel, spec, truth = engine_panel("null", n=80, seed=3, theta=(0.3, 0.3, 0.0, 0.0))
t0 = time.time()
f0, f1 = fit_pair(panel, spec, own_free_start=False)
print(f"fits: {time.time() - t0:.0f} s")
par, g = f0.params, panel.group
y1c = speed._y1c(panel, spec)
dy = y1c[g == 1].mean(0) - y1c[g == 0].mean(0)
print("mean centred landmark state, PIGD - TD:", np.round(dy, 3))
print("theta-hat:", np.round(par["theta"], 3), "| gamma-hat:", round(float(par["log_speed"][1]), 3))
print(f"Sp-1 = exp(gamma + theta.dy) = exp({par['log_speed'][1]:.3f} + {par['theta'] @ dy:.3f}) = "
      f"{np.exp(par['log_speed'][1] + par['theta'] @ dy):.3f}; speed.sp1 = {speed.sp1(f0, panel):.3f}")
print(f"truth: exp(gamma) = 2.000, Sp-1 = exp(log 2 + theta.dy) = {2.0 * np.exp(truth['theta'] @ dy):.3f}")
a0, a1 = f1.params["A"][0][spec.support], f1.params["A"][1][spec.support]
print(f"free-fit speeds: minimal-L1 {speed_rule(a0, a1):.3f}, least squares {speed.sp3(f1)['s_ls']:.3f}")

s2 = speed.sp2(f0, panel)
eta = par["log_speed"][g] + panel.z @ par["beta"] + y1c @ par["theta"]
pi = np.bincount(g) / len(g)
within, between = float(np.sum(pi * par["sigma"] ** 2)), float(np.var(eta))
print("\nSp-2: sigma", np.round(s2["sigma"], 3), "| 90/10 ratio exp(2 x 1.2816 x sigma)", np.round(s2["ratio_90_10"], 3))
print(f"  between = var(eta) = {between:.4f}; within = sum pi_g sigma_g^2 = {within:.4f}; "
      f"share = {between / (between + within):.4f} (code {s2['share_explained']:.4f})")
print("  speed-load correlation lam/sqrt(lam^2 + tau^2), TD row:", np.round(s2["speed_load_corr"][0], 3))

A0, A1 = f1.params["A"][0], f1.params["A"][1]
s_ls = float(np.sum(A1 * A0) / np.sum(A0 * A0))
share = 1 - np.sum((A1 - s_ls * A0) ** 2) / np.sum((A1 - A0) ** 2)
print(f"\nSp-3 by hand: s_LS = <A1, A0>/<A0, A0> = {s_ls:.4f}; share = {share:.4f}; code {speed.sp3(f1)}")

flat = speed._flat(speed.speed_estimates(f0, f1, panel), f1)
print("\nbootstrapped vector (_flat):", len(flat), "entries = 1 + G + p + d + 1 + G*d + 1 =",
      1 + 2 + panel.p + 4 + 1 + 8 + 1)

t0 = time.time()
seen = []
out = speed.bootstrap_speed(panel, spec, f0, f1, B=3, seed=7, fit_kw={"own_free_start": False},
                            support=(spec.support, Step1Config(depth=2)),
                            on_sample=lambda done: seen.append({k: (list(v) if isinstance(v, list) else v)
                                                                for k, v in done.items()}))
print(f"bootstrap: {time.time() - t0:.0f} s; B_used {out['B_used']}, failures {out['failures']}")
for key in ("sp1_ci", "sigma_ci", "ratio_90_10_ci", "share_explained_ci", "sp3_bias_corrected", "sp3_lower_95",
            "sp3_test_statistic", "sp3_threshold", "sp3_reportable"):
    v = out[key]
    print(f"  {key}: {np.round(v, 3).tolist() if not isinstance(v, bool) else v}")
R = np.array(seen[-1]["rows"])
lo = 2 * np.log(speed.sp1(f0, panel)) - np.quantile(R[:, 0], 0.975)
hi = 2 * np.log(speed.sp1(f0, panel)) - np.quantile(R[:, 0], 0.025)
print(f"  Sp-1 basic interval by hand: exp(2 log Sp-1 - q*) = [{np.exp(lo):.3f}, {np.exp(hi):.3f}]")
Dm = np.array(seen[-1]["diffs"])
d_hat = (A1 - A0).ravel()
print(f"  Sp-3 test by hand: ||A1 - A0|| = {np.linalg.norm(d_hat):.3f}; centred bootstrap norms "
      f"{np.round(np.linalg.norm(Dm - d_hat, axis=1), 3)}")

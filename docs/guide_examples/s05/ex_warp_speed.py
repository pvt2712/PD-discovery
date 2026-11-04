"""Sections 5.10-5.11: the warp-speed calibration (one bootstrap replicate per simulated dataset)
on a toy test, then hp.warp_replicate and hp.reference_distribution on small engine panels.
Synthetic data only."""
import time

import numpy as np
from scipy import stats

from _fixture import engine_panel
from pdx import hp

print("1. toy: X_1..X_20 ~ N(0, 1); statistic T = n xbar^2 / s^2; parametric bootstrap from N(0, s0^2)")
rng = np.random.default_rng(0)
n, R, B = 20, 4000, 199


def stat(x):
    return len(x) * x.mean() ** 2 / x.var(ddof=1)


T, Tstar_one, rej_full = np.empty(R), np.empty(R), np.empty(R, bool)
for r in range(R):
    x = rng.normal(size=n)
    s0 = np.sqrt(np.mean(x ** 2))                       # the fitted null: mean 0, ML scale
    T[r] = stat(x)
    boot = np.array([stat(rng.normal(scale=s0, size=n)) for _ in range(B)])
    rej_full[r] = (1 + np.sum(boot >= T[r])) / (B + 1) <= 0.05
    Tstar_one[r] = boot[0]                              # warp speed keeps one replicate per dataset
crit = np.sort(Tstar_one)[int(np.ceil(0.95 * R)) - 1]
print(f"   full bootstrap ({B} per dataset): rejection rate {rej_full.mean():.4f}")
print(f"   warp speed (1 per dataset): critical value {crit:.3f} (exact F(1,19): {stats.f.ppf(0.95, 1, n - 1):.3f}), "
      f"rejection rate {np.mean(T > crit):.4f}")
print(f"   bootstrap fits needed: {R * B} against {R}")

print("2. hp.warp_replicate on one small engine panel (speed-only model true)")
t0 = time.time()
panel, spec, _ = engine_panel("null", n=60, seed=21)
w = hp.warp_replicate(panel, spec, None, np.random.default_rng(5), {"own_free_start": False})
print("  ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in w.items() if k not in ("null_fit", "free_fit")},
      f"({time.time() - t0:.0f} s)")

print("3. hp.reference_distribution on two datasets from a generator")
t0 = time.time()
ref = hp.reference_distribution(lambda g: engine_panel("null", n=60, seed=int(g.integers(1 << 30)))[0], 2, spec,
                                rng=np.random.default_rng(6), fit_kw={"own_free_start": False})
print("   statistics:", np.round(ref, 3), f"({time.time() - t0:.0f} s)")

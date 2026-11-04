"""Guide §6.6-6.8: family A's simulate() - visits, treatment, measurement, missingness, dropout,
visit_probs and target_counts (synthetic only; the visit shares and counts below are invented).

    OMP_NUM_THREADS=1 ../ppmi-causal/.venv/bin/python docs/guide_examples/s06/ex_simulate.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from pdx.sim import family_a, scenarios  # noqa: E402

np.set_printoptions(precision=3, suppress=True)
truth = scenarios.truth_for("S1", 4)
panel, info = family_a.simulate(truth, 400, np.random.default_rng(11))
print("1. info:", info, "| panel.n =", panel.n, "| p =", panel.p)
T = np.array([len(t) for t in panel.t])
print("   visits per participant (landmark included):", {int(k): int(c) for k, c in zip(*np.unique(T, return_counts=True))})
gaps = np.concatenate([np.diff(t) for t in panel.t])
print(f"   first visit time: {min(t[0] for t in panel.t):.3f} to {max(t[0] for t in panel.t):.3f}; "
      f"intervals: {gaps.min():.3f} to {gaps.max():.3f}")
allv = np.vstack(panel.y)
print("   share missing per domain:", np.round(np.isnan(allv).mean(0), 3))
print("   treated by the last visit:", np.mean([dz[-1] > 0 for dz in panel.dose]).round(3),
      "| treated at the landmark:", np.mean([dz[0] > 0 for dz in panel.dose]))
print("   z column means (era, age, sex, duration):", panel.z.mean(0).round(4) + 0.0)
i = next(k for k in range(panel.n) if panel.dose[k][-1] > 0 and len(panel.t[k]) >= 6)
print(f"2. participant {i} ({'PIGD' if panel.group[i] else 'TD'}):")
print("   t       :", panel.t[i])
print("   dose    :", panel.dose[i])
print("   exposure:", panel.exposure[i])
print("   on      :", panel.on[i])
print("   year    :", panel.year[i])
d = panel.dose[i]
e = np.zeros_like(d)
for k in range(len(d)):
    e[k] = 0.5 * (e[k - 1] if k else 0.0) + 0.5 * d[k]
print("   exposure by the recursion 0.5 e[k-1] + 0.5 dose[k]:", e)
kap, kl = truth.kappa[panel.group[i]], truth.kappa_ldr[panel.group[i]]
h = -(np.outer(d / (3 + d) * panel.on[i], kap) + np.outer(panel.exposure[i] / (3 + panel.exposure[i]), kl))
print("   treatment term h on (tremor, axial) at each visit:\n", h[:, :2].T + 0.0)

# 3. WP0-style options: invented visit shares, per-domain missingness and target counts
vp = {2: 0.2, 3: 0.3, 6: 0.5}
p3, i3 = family_a.simulate(truth, 1500, np.random.default_rng(12), visit_probs=vp,
                           p_missing=(0.01, 0.02, 0.10, 0.20), target_counts=(120, 90))
T3 = np.array([len(t) for t in p3.t])
print("3. target_counts (120, 90) ->", i3["n_TD"], i3["n_PIGD"], "| visits:", {int(k): int(c) for k, c in zip(*np.unique(T3, return_counts=True))})
print("   share missing per domain:", np.round(np.isnan(np.vstack(p3.y)).mean(0), 3))
print("   z column means after subsampling:", p3.z.mean(0).round(4))
try:
    family_a.simulate(truth, 200, np.random.default_rng(12), target_counts=(120, 90))
except ValueError as exc:
    print("4. n too small:", exc)

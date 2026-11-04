"""Guide §6.6 and §6.47: in family A the speed is drawn independently of the year-0 state, but the
panel starts at the year-1 landmark, whose scores already reflect the speed over the first gap.
What, then, is the truth for the engine's gamma (a contrast at equal centred landmark scores)?

The participants' speeds are not stored in the panel, so this script records them by wrapping
family_a.transition_np (each participant's calls share one speed) and regresses log-speed on
phenotype and the centred landmark scores, the engine's speed equation without its random part.
Synthetic only.

    OMP_NUM_THREADS=1 ../ppmi-causal/.venv/bin/python docs/guide_examples/s06/ex_speed_landmark.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from pdx.engine.likelihood import prepare  # noqa: E402
from pdx.sim import family_a, scenarios  # noqa: E402

for d in (4, 10):
    t0 = time.time()
    calls, orig = [], family_a.transition_np
    family_a.transition_np = lambda A, Q, s, dt: (calls.append(s), orig(A, Q, s, dt))[1]
    panel, info = scenarios.generate("S1", 20000, np.random.default_rng(7), d=d)
    family_a.transition_np = orig
    runs = []                                   # consecutive calls with one speed = one participant
    for s in calls:
        if runs and runs[-1][0] == s:
            runs[-1][1] += 1
        else:
            runs.append([s, 1])
    speeds = np.array([s for s, c in runs if c >= 2])      # simulate keeps participants with >= 2 visits
    assert len(speeds) == panel.n
    g = panel.group
    y1c = prepare(panel, info["truth"].meas_var).y1c.numpy()
    X = np.column_stack([np.ones(panel.n), g, y1c])
    coef, *_ = np.linalg.lstsq(X, np.log(speeds), rcond=None)
    marginal = np.log(speeds[g == 1]).mean() - np.log(speeds[g == 0]).mean()
    print(f"d = {d}: {panel.n} participants ({time.time() - t0:.0f} s)")
    print(f"   log(truth.speed_ratio) = {np.log(info['speed_ratio']):.4f}; sample difference of mean log-speeds = {marginal:.4f}")
    print(f"   regression on [1, PIGD, centred landmark scores]: PIGD coefficient {coef[1]:.4f}, "
          f"TD intercept {coef[0]:.4f}, landmark coefficients {np.round(coef[2:], 3)}")
    print(f"   PIGD coefficient + theta'(mean landmark PIGD - TD) = "
          f"{coef[1] + coef[2:] @ (y1c[g == 1].mean(0) - y1c[g == 0].mean(0)):.4f} (the Sp-1 construction)")

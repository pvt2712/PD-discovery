"""Toy simulation (synthetic data only, no PPMI): does the false-rejection excess of the
group-level speed-only test grow with sample size when speed varies within phenotypes?

Same truth as toy_random_speed.py, speed-only scenario (shared A, log s_i = log s_g + 0.4 u_i,
s_TD = 1, s_PIGD = 2). Only the group-level matrix-power test is run, because the
person-level random-speed model is too slow for large cohorts; it is correctly specified,
so its likelihood ratio is chi-square(8) in large samples.
Cohort sizes: 400/200 (as in toy_random_speed.py), x4 and x16, five transitions each.
"""
import json
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, ".")
import toy_random_speed as RS  # noqa: E402


def main(seed):
    out = {}
    cells = [(fast, m, R) for fast in (False, True) for m, R in ((1, 200), (4, 200), (16, 100))]
    for fast, m, R in cells:
        rng = np.random.default_rng(seed + 10 * fast + m)
        lrts = []
        for _ in range(R):
            X1 = RS.simulate(rng, 400 * m, 5, RS.drift(False, fast), 1.0)
            X2 = RS.simulate(rng, 200 * m, 5, RS.drift(False, fast), 2.0)
            lrt, _ = RS.pooled_test(X1, X2)
            lrts.append(lrt)
        lrts = np.array(lrts)
        rej = stats.chi2.sf(np.maximum(lrts, 0), RS.P * RS.P - 1) < 0.05
        key = ("fast" if fast else "slow") + f"_x{m}"
        out[key] = {"R": R, "n_TD": 400 * m, "n_PIGD": 200 * m,
                    "pooled_reject": float(rej.mean()),
                    "pooled_mcse": float(np.sqrt(rej.mean() * (1 - rej.mean()) / R)),
                    "lrt_pooled_mean": float(lrts.mean())}
        print(key, json.dumps(out[key]), flush=True)
    with open("toy_random_speed_scale_results.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main(53)

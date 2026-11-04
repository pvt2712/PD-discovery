"""Section 5.35: recalibrating per-link p-values against their null distribution from bootstrap
replicates (localise.recalibrate). Made-up p-values only."""
import numpy as np

import _fixture  # noqa: F401  (import path)
from pdx.localise import recalibrate

rng = np.random.default_rng(0)
B = 999
# null p-values of three links over B replicates: link 0 calibrated (uniform), link 1 too small
# (anti-conservative: Beta(0.5, 1)), link 2 too large (conservative: Beta(2, 1))
p_null = np.column_stack([rng.uniform(size=B), rng.beta(0.5, 1.0, size=B), rng.beta(2.0, 1.0, size=B)])
print("share of null p-values <= 0.05 per link:", np.round((p_null <= 0.05).mean(0), 3))
p_obs = np.array([0.03, 0.03, 0.03])
print("observed p      :", p_obs)
print("recalibrated p  :", np.round(recalibrate(p_obs, p_null), 4))
print("by hand, link 1 : (1 + #{p_null <= 0.03}) / (B + 1) =", (1 + np.sum(p_null[:, 1] <= 0.03)) / (B + 1))

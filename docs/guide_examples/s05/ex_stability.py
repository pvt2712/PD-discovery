"""Section 5.40: stability selection over one complementary pair of random halves (B = 2), split
within phenotype, on the panel of ex_joint.py. Synthetic data only."""
import time

import numpy as np

from _fixture import engine_panel
from pdx.engine.fit import fit_pair
from pdx.localise import stability_selection

panel, spec, truth = engine_panel("free", n=100, seed=4, extra={(3, 1): 0.30})
f0, f1 = fit_pair(panel, spec, own_free_start=False)
t0 = time.time()
st = stability_selection(panel, spec, f1, f0, B=2, rng=np.random.default_rng(3), disc_kw={"n_lambda": 3})
print(f"{st['halves_used']} halves used ({time.time() - t0:.0f} s)")
for link, fr in zip(st["links"], st["frequency"]):
    print(f"  {link}: selected in {fr:.0%} of halves")

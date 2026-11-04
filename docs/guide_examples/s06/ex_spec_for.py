"""Guide §6.21: spec_for, the analysis model of a simulated dataset (synthetic only).

    OMP_NUM_THREADS=1 ../ppmi-causal/.venv/bin/python docs/guide_examples/s06/ex_spec_for.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from pdx.engine.model import Layout  # noqa: E402
from pdx.hp import h_p_df  # noqa: E402
from pdx.qualification import spec_for  # noqa: E402
from pdx.sim import scenarios  # noqa: E402

for fam in ("A", "B"):
    panel, info = scenarios.generate("S1", 150, np.random.default_rng(2), family=fam)
    spec = spec_for(panel, info)
    print(f"family {fam}: d={spec.d} p={spec.p} treatment={spec.treatment} n_quad={spec.n_quad} "
          f"noise_era_col={spec.noise_era_col} treat_mask={spec.treat_mask.astype(int)} meas_var={spec.meas_var}")
    print(f"   n_sup={spec.n_sup}, H-P df={h_p_df(spec)}, parameters: null {Layout(spec, 'null').size}, "
          f"free {Layout(spec, 'free').size}, joint {Layout(spec, 'joint').size}")
spec_lin = spec_for(panel, info, treatment="linear", n_quad=20, era=False)
print("options: treatment", spec_lin.treatment, "| n_quad", spec_lin.n_quad, "| noise_era_col", spec_lin.noise_era_col)

"""Guide §6.33-6.34: the fidelity summaries and the comparison of one panel with simulated panels.
A synthetic S1 panel stands in for the real one (synthetic only; no PPMI data).

    OMP_NUM_THREADS=1 ../ppmi-causal/.venv/bin/python docs/guide_examples/s06/ex_fidelity.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from pdx import fidelity  # noqa: E402
from pdx.sim import scenarios  # noqa: E402

np.set_printoptions(precision=3, suppress=True)
real_panel, _ = scenarios.generate("S1", 600, np.random.default_rng(100))
real = fidelity.panel_summaries(real_panel)
print("statistics:", list(real))
print("lag correlations (row = domain at t, column = domain at t+1):\n", np.array(real["lag_correlations"]))
print("residual sd:", np.array(real["residual_sd"]), "| within-person variance:", np.array(real["within_person_variance"]))
print("phenotype share:", round(real["phenotype_share"], 3), "| visits quantiles:", real["visits_quantiles"],
      "| treated at last visit:", round(real["share_treated_last_visit"], 3))
for sc in ("S1", "S0"):
    sims = [fidelity.panel_summaries(scenarios.generate(sc, 600, np.random.default_rng(200 + r))[0]) for r in range(20)]
    cmp = fidelity.compare(real, sims)
    inside = {k: f"{np.mean(v['inside']):.2f}" for k, v in cmp.items()}
    print(f"20 simulated {sc} panels against the stand-in: share of entries inside the 2.5-97.5% envelope")
    print("  ", inside)
c = fidelity.compare(real, sims)["marginal_quantiles"]
print("S0 envelope for the 90% quantile of each domain:", np.array(c["sim_2.5"])[3], "to", np.array(c["sim_97.5"])[3],
      "| stand-in:", np.array(c["real"])[3])

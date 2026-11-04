"""Guide §6.15-6.16: the expert map every scenario hands the analysis, and generate() with its
stresses (synthetic only).

    OMP_NUM_THREADS=1 ../ppmi-causal/.venv/bin/python docs/guide_examples/s06/ex_generate.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from pdx.qualification import spec_for  # noqa: E402
from pdx.hp import h_p_df  # noqa: E402
from pdx.sim import scenarios  # noqa: E402
from pdx.sim.family_a import true_support  # noqa: E402

maps = {sc: scenarios.expert_map(scenarios.truth_for(sc)) for sc in scenarios.NULL_SCENARIOS + scenarios.POWER_SCENARIOS}
print("expert map (row = to, column = from), identical in all 11 scenarios:",
      all((m == maps["S0"]).all() for m in maps.values()))
print(maps["S0"].astype(int))
print("true links outside the map in S3:", (true_support(scenarios.truth_for("S3")) & ~maps["S3"]).sum(),
      "| map entries that are zero in S1's truth:", int((maps["S1"] & ~true_support(scenarios.truth_for("S1"))).sum()))

panel, info = scenarios.generate("S1", 200, np.random.default_rng(4))
print("generate('S1') info keys:", sorted(info))
print("   H-P df with this map:", h_p_df(spec_for(panel, info)))
_, i_sp = scenarios.generate("S1", 200, np.random.default_rng(4), spurious=True)
_, i_ml = scenarios.generate("S1", 200, np.random.default_rng(4), missing_link=True)
print("spurious adds:", [tuple(map(int, ix)) for ix in np.argwhere(i_sp["support"] & ~info["support"])],
      "| missing_link removes:", [tuple(map(int, ix)) for ix in np.argwhere(info["support"] & ~i_ml["support"])])
p_kw, i_kw = scenarios.generate("S1", 200, np.random.default_rng(4), p_missing=0.25, visits=(2, 3))
print("simulate() options pass through: share missing",
      round(float(np.isnan(np.vstack(p_kw.y)).mean()), 3), "| max visits", max(len(t) for t in p_kw.t))
pb, ib = scenarios.generate("S1", 200, np.random.default_rng(4), family="B")
print("family B info keys:", sorted(ib), "| support:\n", ib["support"].astype(int))
_, ib_sp = scenarios.generate("S1", 200, np.random.default_rng(4), family="B", spurious=True, target_counts=(10, 10))
print("family B ignores spurious and target_counts: same support", bool((ib_sp["support"] == ib["support"]).all()),
      "| n_TD, n_PIGD =", ib_sp["n_TD"], ib_sp["n_PIGD"])
try:
    scenarios.generate("S2", 200, np.random.default_rng(4), family="B")
except ValueError as exc:
    print("generate('S2', family='B'):", exc)

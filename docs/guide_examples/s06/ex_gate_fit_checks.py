"""Guide §6.40-6.41 and §6.46: the checks that need the speed-only fit - quadrature agreement at
12, 20 and 30 nodes (with H-P's statistic) and the conditioning of the observed information
(synthetic only; one small S1 dataset, as in tests/test_gate.py).

    OMP_NUM_THREADS=1 ../ppmi-causal/.venv/bin/python docs/guide_examples/s06/ex_gate_fit_checks.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from pdx import gate  # noqa: E402
from pdx.engine.fit import fit_model  # noqa: E402
from pdx.qualification import spec_for  # noqa: E402
from pdx.sim import scenarios  # noqa: E402

panel, info = scenarios.generate("S1", 120, np.random.default_rng(5), d=4)
spec = spec_for(panel, info)
t0 = time.time()
f0 = fit_model(panel, spec, "null")
print(f"speed-only fit: {panel.n} participants, loglik {f0.loglik:.4f}, {f0.n_eval} evaluations ({time.time() - t0:.0f} s)")

t0 = time.time()
q = gate.check_quadrature_lr(f0, panel)
print("null loglik by nodes:", {k: round(v, 4) for k, v in q["loglik"].items()}, "spread", round(q["spread"], 4), "pass", q["pass"])
print("free loglik by nodes:", {k: round(v, 4) for k, v in q["free_loglik"].items()}, "spread", round(q["free_spread"], 4))
print("LR by nodes:", {k: round(v, 4) for k, v in q["lr"].items()}, "spread", round(q["lr_spread"], 4), f"({time.time() - t0:.0f} s)")

t0 = time.time()
ident = gate.identification(f0, panel)
se = ident.pop("se")
print(f"identification ({time.time() - t0:.0f} s): condition number {ident['condition_number']:.3g}, "
      f"positive definite {ident['positive_definite']}, well conditioned {ident['well_conditioned']}")
print("   spreads at zero:", ident["spreads_at_zero"], "| without them:",
      f"{ident['condition_number_without_them']:.3g}", ident["well_conditioned_without_them"],
      "| max |correlation|", round(ident["max_abs_correlation"], 3))
lay = f0.layout
for nm, ix in (("gamma", 0), ("log_tau", 0)):
    j = lay.slices[nm][0].start + ix
    print(f"   {nm}[{ix}]: estimate {f0.x[j]:.4f}, se {se[j]:.4f}")
print("   flat directions (eigenvalue below max/1e10):", len(ident["flat_directions"]), "| first:", ident["flat_directions"][0])

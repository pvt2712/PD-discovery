"""Guide §6.31-6.32: power against the S1 critical value of the same configuration, and the bias
and RMSE of log Sp-1, on invented rows (synthetic only; nothing is fitted).

    OMP_NUM_THREADS=1 ../ppmi-causal/.venv/bin/python docs/guide_examples/s06/ex_power_speed.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from pdx import qualification as Q  # noqa: E402

rng = np.random.default_rng(8)
null = [{"family": "A", "scenario": "S1", "config": "k1", "lr": float(rng.chisquare(7)),
         "lr_star": float(rng.chisquare(7)), "df": 7, "converged": True, "replicate_ok": True,
         "replicate_converged": True, "seconds": 1.0, "sp1": float(np.exp(np.log(2) + 0.1 * rng.normal() + 0.03)),
         "true_speed_ratio": 2.0, "sp3_share": float(rng.uniform())} for _ in range(200)]
cal = Q.calibration_summary(null)
print("S1 critical value (k1):", round(cal["A:S1:k1"]["critical_value"], 3))
power_rows = [{"family": "A", "scenario": "S2", "config": "k1", "lr": float(rng.noncentral_chisquare(7, 12.0))}
              for _ in range(100)]
power_rows += [{"family": "A", "scenario": "S2", "config": "k1", "error": "RuntimeError('...')"}] * 10
power_rows += [{"family": "A", "scenario": "S2", "config": "k2", "lr": 50.0}] * 5   # no S1 run with key k2
for key, v in Q.power_summary(power_rows, cal).items():
    print(key, {k: (round(x, 4) if isinstance(x, float) else x) for k, x in v.items()})
ok = [r for r in power_rows if r["config"] == "k1" and "error" not in r]
print("power if the 10 failed datasets counted as non-rejections:",
      round(sum(r["lr"] > cal["A:S1:k1"]["critical_value"] for r in ok) / 110, 4))
for key, v in Q.speed_summary(null).items():
    print(key, {k: (round(x, 4) if isinstance(x, float) else x) for k, x in v.items()})

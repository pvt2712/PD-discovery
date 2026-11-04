"""Guide §7.4-7.6: run_power_check.wp0_settings and summarise on made-up inputs (synthetic only).

No PPMI data and no file under data/ or results/ is read: the WP0 audit below is invented and
written to a temporary folder, and summarise() receives invented dataset rows.

    OMP_NUM_THREADS=1 ../ppmi-causal/.venv/bin/python docs/guide_examples/s07/ex_power_summarise.py
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
import run_power_check as R  # noqa: E402

# ---------------------------------------------------------------- 1. wp0_settings on an invented audit
NAMES10 = ["tremor", "axial", "brady_rigidity", "function", "independence", "cognition", "mood", "autonomic",
           "hallucinations", "sleepiness"]
audit = {
    "landmark": {"cells": {"TD": {"reached year 1": 500, "not reached": 40},
                           "PIGD": {"reached year 1": 120, "not reached": "<5"}}},
    "visits_per_participant": {"1": 30, "2": 50, "3": "<5", "4": 200, "5": 300},
    "missing_share": {**{d: {"overall": 0.01 * (k + 1)} for k, d in enumerate(NAMES10)},
                      "rbd": {"overall": 0.5}},          # a domain outside the simulator's ten: ignored
}
tmp = Path(tempfile.mkdtemp())
(tmp / "audit.json").write_text(json.dumps(audit))
s = R.wp0_settings(tmp / "audit.json")
print("1. wp0_settings on an invented audit")
print("   target_counts:", s["target_counts"])
print("   visit_probs  :", s["visit_probs"])
print("   p_missing    :", s["p_missing"])
one = 30 / (30 + 50 + 2 + 200 + 300)
print(f"   by hand: share with one visit = 30/582 = {one:.5f}; "
      f"TD 500 x {1 - one:.5f} = {500 * (1 - one):.2f}; PIGD 120 x {1 - one:.5f} = {120 * (1 - one):.2f}")

# ---------------------------------------------------------------- 2. summarise on invented rows
df = [13, 13, 14, 13, 15, 14, 13, 13]
lr = [38.2, 25.0, 31.5, 19.8, 44.1, 28.7, 33.0, 22.4]
rows = [{"scenario": "S2", "df": d, "lr": v, "seconds": 3600.0 + 60 * k} for k, (d, v) in enumerate(zip(df, lr))]
rows.append({"scenario": "S2", "error": "RuntimeError('fit failed')"})       # a failed dataset
rows += [{"scenario": "S1", "df": 13, "lr": 15.0, "lr_star": v, "seconds": 7000.0} for v in (12.1, 18.5, 9.7)]
out = R.summarise(rows)
print("\n2. summarise on 8 invented S2 rows, 1 failed row and 3 invented S1 (null) rows")
for k, v in out.items():
    print(f"   {k}: {v}")
crit = stats.chi2.ppf(0.95, df)
print("   by hand: critical values", {int(d): round(float(c), 3) for d, c in zip(df, crit)})
print("   by hand: rejections", [int(a > c) for a, c in zip(lr, crit)])
dev = np.array(lr) - np.array(df)
print(f"   by hand: LR - df = {np.round(dev, 2).tolist()}, mean {dev.mean():.4f}, se {dev.std(ddof=1) / np.sqrt(len(dev)):.4f}")
p13 = stats.ncx2.sf(stats.chi2.ppf(0.95, 13), 13, dev.mean())
print(f"   by hand: noncentral power for one dataset with df 13 at lambda-hat: {p13:.4f}")

# ---------------------------------------------------------------- 3. the 20-dataset Monte Carlo check
print("\n3. the optional 20-dataset check: exact (Clopper-Pearson) one-sided 95% lower bounds")
for k in (20, 19, 18):
    lower = stats.beta.ppf(0.05, k, 20 - k + 1)
    print(f"   {k} of 20 reject: lower bound {lower:.4f}")
rows20 = [{"scenario": "S2", "df": 13, "lr": 60.0, "seconds": 1.0} for _ in range(20)]
o20 = R.summarise(rows20)
print("   summarise on 20 rows that all reject: power_mc_chi2 =", o20["power_mc_chi2"], "mcse_mc =", o20["mcse_mc"])
for p in (0.80, 0.90, 0.95, 0.99):
    print(f"   chance that all 20 reject when the true power is {p:.2f}: {p ** 20:.4f}")

# ---------------------------------------------------------------- 4. edge cases
print("\n4. edge cases")
print("   one S2 row:", {k: R.summarise(rows[:1])[k] for k in ("noncentrality_se", "power_noncentral_range",
                                                                 "gate_80_percent")})
low = [{"scenario": "S2", "df": 13, "lr": v, "seconds": 1.0} for v in (8.0, 10.0, 12.0)]
print("   statistics below their df (lambda-hat floored at 0):",
      {k: round(R.summarise(low)[k], 4) for k in ("noncentrality", "power_noncentral")})
print("   no S2 row:", R.summarise([{"scenario": "S1", "lr": 1.0, "lr_star": 2.0}]))

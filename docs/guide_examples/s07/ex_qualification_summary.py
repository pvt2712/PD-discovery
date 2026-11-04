"""Guide §7.16: what `run_qualification.py --summarise` computes, on invented dataset rows (synthetic only).

main() is not called (it would read and write results/qualification/). The task list it would
build, and the summary it would print, are reproduced here with invented rows.

    OMP_NUM_THREADS=1 ../ppmi-causal/.venv/bin/python docs/guide_examples/s07/ex_qualification_summary.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
import run_qualification  # noqa: E402,F401  (sets the import path, as the script does)
from pdx import qualification as Q  # noqa: E402

# ---------------------------------------------------------------- 1. the task list of the default command
tasks = []
for sc in ("S0", "S1"):
    tasks += Q.make_tasks(sc, 2, 1400, "A", d=4, step1=True, fit_kw={"maxiter": 3000}, sim_kw={})
print("1. tasks of `--datasets 2` with the defaults:")
for t in tasks:
    print("  ", t["scenario"], t["seed"], Q.config_key(t), Q._path(Path("datasets"), t).name)

# ---------------------------------------------------------------- 2. the summary on invented rows
rng = np.random.default_rng(0)
cfg = Q.config_key(tasks[0])


def row(sc, lr, star, df=6, sp1=2.0, true=2.0):
    return {"family": "A", "scenario": sc, "config": cfg, "lr": lr, "lr_star": star, "df": df, "converged": True,
            "replicate_ok": star is not None, "replicate_converged": True, "seconds": 60.0,
            "sp1": sp1, "true_speed_ratio": true, "sp3_share": 0.9, "task": {"scenario": sc}}


rows = [row("S0", float(rng.chisquare(6)), float(rng.chisquare(6)), sp1=float(np.exp(rng.normal(0, .1))), true=1.0)
        for _ in range(20)]
rows += [row("S1", float(rng.chisquare(6)), float(rng.chisquare(6)), sp1=float(2 * np.exp(rng.normal(0, .1))))
         for _ in range(20)]
rows[25]["lr_star"], rows[25]["replicate_ok"] = None, False          # one failed bootstrap replicate
rows += [row("S2", float(rng.noncentral_chisquare(6, 15)), None) for _ in range(10)]
null_rows = [r for r in rows if r["scenario"] in Q.NULL_A]
power_rows = [r for r in rows if r["scenario"] not in Q.NULL_A]
cal = Q.calibration_summary(null_rows)
print("\n2. calibration (one entry per family:scenario:config):")
for k, v in cal.items():
    print(f"   {k}: critical {v['critical_value']:.3f}, rejection {v['rejection_rate']:.3f} (MCSE {v['mcse']:.3f}),"
          f" chi-square rejection {v['rejection_rate_chi2']:.3f}, replicates failed {v['replicates_failed']},"
          f" passes Bradley {v['passes_bradley']}")
print("   H-M1:", Q.h_m1(cal))
print("   power:", {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in Q.power_summary(power_rows, cal).items()})
print("   speed:", {k: round(v["bias_log_sp1"], 4) for k, v in Q.speed_summary(rows).items()})
s1_stars = sorted(r["lr_star"] if r["lr_star"] is not None else np.inf for r in null_rows if r["scenario"] == "S1")
print(f"   by hand, S1: the ceil(0.95 x 20) = 19th smallest bootstrap statistic is {s1_stars[18]:.3f}")

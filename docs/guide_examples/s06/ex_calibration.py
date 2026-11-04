"""Guide §6.27-6.29: the warp-speed rejection rate, its Monte Carlo standard error and Bradley's
7.5% criterion, on invented rows whose statistics are chi-square draws (synthetic only; nothing
is fitted). The second part measures how often one cell passes when the procedure is exactly
calibrated (true rate 5%) or clearly not (true rate 10%), including the noise of the bootstrap
critical value.

    OMP_NUM_THREADS=1 ../ppmi-causal/.venv/bin/python docs/guide_examples/s06/ex_calibration.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402
from pdx import qualification as Q  # noqa: E402

print("_mcse(0.05, 500) =", round(Q._mcse(0.05, 500), 5), "| _mcse(0.05, 100) =", round(Q._mcse(0.05, 100), 5))
print("_rank_quantile([3, 1, 2, None], 0.5) =", Q._rank_quantile([3.0, 1.0, 2.0, None], 0.5),
      "| ([1..20], 0.95) =", Q._rank_quantile(list(range(1, 21)), 0.95),
      "| ([1..19, inf], 0.95) =", Q._rank_quantile(list(range(1, 20)) + [float("inf")], 0.95))


def rows(R, rng, inflate=1.0, df=7, fail=0, config="k1"):
    """R invented datasets: observed LR ~ inflate * chi2(df), one bootstrap LR ~ chi2(df)."""
    out = [{"family": "A", "scenario": "S1", "config": config, "lr": float(inflate * rng.chisquare(df)),
            "lr_star": float(rng.chisquare(df)), "df": df, "converged": True, "replicate_ok": True,
            "replicate_converged": True, "seconds": 1.0} for _ in range(R)]
    for r in out[:fail]:
        r.update({"lr_star": None, "replicate_ok": False})
    return out


rng = np.random.default_rng(5)
example = rows(500, rng, fail=3) + [{"family": "A", "scenario": "S1", "config": "k1", "seed": 1, "n": 1400,
                                     "error": "RuntimeError('...')"}]
cal = Q.calibration_summary(example)
v = cal["A:S1:k1"]
print({k: (round(v[k], 4) if isinstance(v[k], float) else v[k]) for k in
       ("datasets", "failed", "critical_value", "rejection_rate", "mcse", "passes_bradley", "rejection_rate_chi2",
        "replicates_failed")})
print("chi-square(7) 95% point for comparison:", round(stats.chi2.ppf(0.95, 7), 4))

t0 = time.time()
f10 = stats.chi2.ppf(0.95, 7) / stats.chi2.ppf(0.90, 7)          # inflation giving a 10% true rate
for R in (100, 300, 500):
    for label, inflate in (("true rate 5%", 1.0), ("true rate 10%", f10)):
        cells = [Q.calibration_summary(rows(R, rng, inflate))["A:S1:k1"] for _ in range(2000)]
        passes = [c["passes_bradley"] for c in cells]
        k = int(np.floor(0.075 * R + 1e-9))
        p_true = 0.05 if inflate == 1.0 else 0.10
        rates = np.array([c["rejection_rate"] for c in cells])
        print(f"R = {R}, {label}: mean rejection rate {rates.mean():.4f}, sd {rates.std():.4f} "
              f"(reported mcse about {Q._mcse(rates.mean(), R):.4f}); passes in {np.mean(passes):.3f} of 2000 cells "
              f"(binomial with a known critical value: {stats.binom.cdf(k, R, p_true):.3f})")
    kq = np.ceil(0.95 * R)
    p_bar = 1 - kq / (R + 1)
    sd_theory = np.sqrt(kq * (R - kq + 1) / ((R + 1) ** 2 * (R + 2)) + p_bar * (1 - p_bar) / R)
    print(f"   calibrated case in theory: mean 1 - ceil(0.95 R)/(R + 1) = {p_bar:.4f}, sd {sd_theory:.4f}")
print(f"({time.time() - t0:.0f} s)")

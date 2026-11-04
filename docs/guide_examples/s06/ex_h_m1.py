"""Guide §6.30: h_m1, the gate of protocol §8.7, on invented calibration tables, and the chance
that a perfectly calibrated procedure passes all ten cells (synthetic only; nothing is fitted).

    OMP_NUM_THREADS=1 ../ppmi-causal/.venv/bin/python docs/guide_examples/s06/ex_h_m1.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from scipy import stats  # noqa: E402
from pdx import qualification as Q  # noqa: E402


def table(fail=(), drop=(), extra=()):
    cal = {}
    cells = [("A", s) for s in Q.NULL_A] + [("B", s) for s in Q.NULL_B]
    for fam, sc in cells:
        if f"{fam}:{sc}" in drop:
            continue
        cal[f"{fam}:{sc}:k{fam}"] = {"family": fam, "scenario": sc, "config": f"k{fam}",
                                    "passes_bradley": f"{fam}:{sc}" not in fail}
    for fam, sc, cfg in extra:
        cal[f"{fam}:{sc}:{cfg}"] = {"family": fam, "scenario": sc, "config": cfg, "passes_bradley": True}
    return cal


print("cells required:", [f"A:{s}" for s in Q.NULL_A] + [f"B:{s}" for s in Q.NULL_B])
print("all pass        ->", Q.h_m1(table()))
print("only A:S4 fails ->", Q.h_m1(table(fail={"A:S4"})))
print("A:S4 and A:S1S  ->", Q.h_m1(table(fail={"A:S4", "A:S1S"})))
print("B:S1 missing    ->", Q.h_m1(table(drop={"B:S1"})))
two = table(extra=[("A", "S1", "kA2")])
print("two configs     ->", Q.h_m1(two))
print("   with config  ->", Q.h_m1(two, config={"A": "kA"}))

# operating characteristic of the whole gate (independent cells): binomial with a known critical
# value, and the per-cell pass rates measured with the warp-speed critical value (ex_calibration.py)
b = {R: stats.binom.cdf(int(0.075 * R + 1e-9), R, 0.05) for R in (500, 300, 100)}
print(f"P(all ten cells pass | every true rate 5%), binomial only: {b[500] ** 2 * b[300] ** 2 * b[100] ** 6:.3f}")
warp = {500: 0.938, 300: 0.888, 100: 0.728}
print(f"   with the warp-speed critical value's noise:          {warp[500] ** 2 * warp[300] ** 2 * warp[100] ** 6:.3f}")

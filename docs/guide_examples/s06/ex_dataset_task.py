"""Guide §6.22: dataset_task, one dataset of the warp-speed calibration - simulate, select the
support with Step 1, fit both models, draw one bootstrap replicate (synthetic only, small n).

    OMP_NUM_THREADS=1 ../ppmi-causal/.venv/bin/python docs/guide_examples/s06/ex_dataset_task.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from pdx import qualification as Q  # noqa: E402

task = Q.make_tasks("S1", 1, 80, "A", d=4)[0]
print("task:", task)
t0 = time.time()
res = Q.dataset_task(task)
print(f"wall time {time.time() - t0:.0f} s")
for k in ("config", "scenario", "family", "seed", "n", "lr", "lr_star", "df", "converged", "replicate_ok",
          "replicate_converged", "sp1", "sp3_share", "true_speed_ratio"):
    v = res[k]
    print(f"   {k:20s} {v:.4f}" if isinstance(v, float) else f"   {k:20s} {v}")
print("   counts:", res["counts"])

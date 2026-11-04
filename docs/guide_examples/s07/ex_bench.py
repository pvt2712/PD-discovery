"""Guide §7.20: the pieces of scripts/bench_threads.py that can run without a benchmark (synthetic only).

No job is started and no file is read or written.

    ../ppmi-causal/.venv/bin/python docs/guide_examples/s07/ex_bench.py
"""
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
import bench_threads as B  # noqa: E402

print("1. parse:", B.parse("4x1"), B.parse("1X10"))
try:
    B.parse("4")
except ValueError as e:
    print("   parse('4') fails:", e)

jobs, evals, wall = 4, 30, 150.0          # invented: 4 jobs of 30 evaluations each, 150 s from 'go' to the last result
print(f"2. evals_per_hour = 3600 x jobs x evals / wall = 3600 x {jobs} x {evals} / {wall} = {3600 * jobs * evals / wall:.0f}")
each = [4.6, 4.9, 5.0, 4.8]               # invented seconds_per_eval of each job
print(f"   if every job had run alone at its own pace: {sum(3600 / s for s in each):.0f} an hour")

proc = SimpleNamespace(stdout=iter(["torch: loading\n", "ready\n", "progress 50%\n", '{"seconds_per_eval": 4.6}\n']))
print("3. _read skips other lines:", B._read(proc, "ready"), "|", B._read(proc, "{"))
try:
    B._read(proc, "{")
except RuntimeError as e:
    print("   a job that ends early:", e)

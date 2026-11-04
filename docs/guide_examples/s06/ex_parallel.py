"""Guide §6.35-6.37: parallel.ordered - results in item order, single-threaded spawned workers,
per-piece seeds, the thread variables restored, and an early stop (synthetic only).

    ../ppmi-causal/.venv/bin/python docs/guide_examples/s06/ex_parallel.py

(run without OMP_NUM_THREADS in the shell, so that the parent's unset variable is seen to be restored).
The `if __name__ == "__main__":` guard is required: "spawn" starts each worker by importing this
file afresh, and without the guard every worker would start the pool again.
"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from pdx import parallel  # noqa: E402


def seed_setup(seed):                       # module level, so that the workers can find it
    return {"seed": seed}


def draw(shared, r):
    """Piece r: its own stream, so the value does not depend on which worker runs it."""
    return float(np.random.default_rng(shared["seed"] + r).normal())


def threads(shared, item):
    import torch
    return item, os.environ.get("OMP_NUM_THREADS"), torch.get_num_threads(), os.getpid()


if __name__ == "__main__":
    print("parent pid", os.getpid(), "| OMP_NUM_THREADS in the parent:", os.environ.get("OMP_NUM_THREADS"))
    t0 = time.time()
    print("jobs=2:", list(parallel.ordered(parallel.check_square, [1, 2, 3, 4, 5], 2, parallel.check_setup, (10.0,))),
          f"({time.time() - t0:.1f} s)")
    print("jobs=1:", list(parallel.ordered(parallel.check_square, [1, 2, 3], 1, parallel.check_setup, (10.0,))))
    print("workers see:", list(parallel.ordered(threads, ["a", "b"], 2, parallel.check_setup, (1.0,))))
    print("parent afterwards:", os.environ.get("OMP_NUM_THREADS"))
    runs = {j: list(parallel.ordered(draw, range(6), j, seed_setup, (100,))) for j in (1, 2, 3)}
    print("per-piece seeds, jobs 1/2/3 identical:", runs[1] == runs[2] == runs[3], np.round(runs[1], 4))
    gen = parallel.ordered(parallel.check_square, list(range(50)), 3, parallel.check_setup, (1.0,))
    print("first result:", next(gen))
    gen.close()                              # e.g. a sequential test that has stopped
    print("generator closed; the pool was terminated")
    print("empty items:", list(parallel.ordered(parallel.check_square, [], 4, parallel.check_setup, (1.0,))))

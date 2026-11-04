"""Guide §7.15: resuming H-P's sequential p-value with a lower cap than the saved replicates (synthetic only).

hp.sequential_p is given ten saved replicates, as run_hp_real.py rebuilds them from its checkpoint,
and no new draw (the draw function raises if called). No file is read or written.

    OMP_NUM_THREADS=1 ../ppmi-causal/.venv/bin/python docs/guide_examples/s07/ex_sequential_resume.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from pdx.hp import sequential_p  # noqa: E402


def no_draw():
    raise RuntimeError("no new draw expected")


saved = [(0.1, True)] * 10                     # ten replicates saved by an earlier run with --n-max 10
for n_max in (5, 10):
    r = sequential_p(no_draw, observed=4.0, n_max=n_max, h=20, resume=saved)
    print(f"resumed with --n-max {n_max}:", {k: r[k] for k in ("p", "n_drawn", "n_exceed", "stopped_early")})

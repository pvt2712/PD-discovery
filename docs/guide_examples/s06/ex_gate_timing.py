"""Guide §6.49: time_replicate - seconds for support selection, the two observed fits and one whole
bootstrap replicate; then the same call with the replicate's simulation made to fail, which is
timed without any flag (synthetic only, small n).

    OMP_NUM_THREADS=1 ../ppmi-causal/.venv/bin/python docs/guide_examples/s06/ex_gate_timing.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from pdx import gate, hp  # noqa: E402

r = gate.time_replicate(60, 4, seed=0)
print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})


def broken(*args, **kwargs):
    raise RuntimeError("simulation failed")


hp.simulate_from_fit = broken             # _replicate catches the error and returns (inf, False, False, None)
r2 = gate.time_replicate(60, 4, seed=0)
print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r2.items()})

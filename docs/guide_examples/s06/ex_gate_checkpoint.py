"""Guide §6.50-6.54: plain(), the Checkpoint, why every gate section has its own random stream,
and what run_gate does before any fit (settings, quick mode, seeds). Synthetic only; the
checkpoint files live in a temporary folder, and the target counts below are invented.

    OMP_NUM_THREADS=1 ../ppmi-causal/.venv/bin/python docs/guide_examples/s06/ex_gate_checkpoint.py
"""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from pdx import gate  # noqa: E402

# ---------------------------------------------------------------- 1. plain
print("1. plain:", gate.plain({"a": np.arange(3), "b": np.float64(1.5), "c": np.bool_(True), "d": (1, 2),
                                "e": {2: 0.5}, "f": np.int64(7), "g": float("nan"), "h": np.inf}))
try:
    gate.plain({"fit": object()})
except TypeError as exc:
    print("   plain of an object that is not JSON:", repr(exc))

# ---------------------------------------------------------------- 2. Checkpoint
path = Path(tempfile.mkdtemp()) / "gate.partial.json"
said, calls = [], []
ck = gate.Checkpoint(path, {"n": 10, "sim_kw": {"target_counts": (3, 2)}}, said.append)
print("2. file exists before the first section:", path.exists())
print("   section a:", ck.section("a", lambda: calls.append(1) or {"x": np.float64(2.0)}), "| work done", len(calls), "time(s)")
again = gate.Checkpoint(path, {"n": 10, "sim_kw": {"target_counts": [3, 2]}}, said.append)   # a tuple saved as a list
print("   after a restart:", again.section("a", lambda: calls.append(1) or {"x": 99.0}), "| work done", len(calls), "time(s)")
print("   log:", said)
try:
    gate.Checkpoint(path, {"n": 11, "sim_kw": {"target_counts": (3, 2)}})
except RuntimeError as exc:
    print("   other settings:", str(exc).replace(str(path.parent), "<tmp>"))
print("   saved file:", path.read_text())


# ---------------------------------------------------------------- 3. one stream per section
def toy_gate(ck, streams):
    """Two sections that draw random numbers, as the gate's sections do."""
    shared = np.random.default_rng(2026)
    rng = (lambda k: np.random.default_rng([2026, k])) if streams else (lambda k: shared)
    a = ck.section("first", lambda: rng(1).normal(size=2).round(4))
    b = ck.section("second", lambda: rng(2).normal(size=2).round(4))
    return a, b


class Crash(Exception):
    pass


for streams in (False, True):
    folder = Path(tempfile.mkdtemp())
    whole = toy_gate(gate.Checkpoint(folder / "whole.json", {}), streams)
    ck = gate.Checkpoint(folder / "broken.json", {})
    real_section = ck.section

    def section_then_crash(key, fn):
        if key == "second":
            raise Crash                        # the run stops after the first section was saved
        return real_section(key, fn)
    ck.section = section_then_crash
    try:
        toy_gate(ck, streams)
    except Crash:
        pass
    resumed = toy_gate(gate.Checkpoint(folder / "broken.json", {}), streams)
    print(f"3. {'one stream per section' if streams else 'one shared stream     '}: uninterrupted {whole}, "
          f"resumed {resumed} -> equal: {whole == resumed}")


# ---------------------------------------------------------------- 4. run_gate up to its checkpoint
class Stop(Exception):
    pass


class Peek(gate.Checkpoint):
    def __init__(self, path, settings, log=None):
        print("4. settings run_gate would save:", settings)
        raise Stop


real_checkpoint, gate.Checkpoint = gate.Checkpoint, Peek
for quick in (False, True):
    try:
        gate.run_gate(n=1400, d=10, n_datasets=20, quick=quick,
                      sim_kw={"target_counts": (700, 500), "visit_probs": {2: 0.4, 6: 0.6}})
    except Stop:
        pass
gate.Checkpoint = real_checkpoint
p2 = Path(tempfile.mkdtemp()) / "g.partial.json"
gate.Checkpoint(p2, {"n": 1}).save()
try:
    gate.run_gate(n=200, d=4, checkpoint=p2)
except RuntimeError as exc:
    print("   a checkpoint from another run:", str(exc).replace(str(p2.parent), "<tmp>"))
seed, n_datasets = 2026, 20
print("5. section streams: rng(k) = default_rng([seed, k]) for k = 0 small dataset, 1 reference, 2 gradient,"
      " 3 gate dataset, 4 dispersed starts")
print("   recovery seeds: S1", seed + 100, "to", seed + 100 + n_datasets - 1, "| S2", seed + 200, "to",
      seed + 200 + n_datasets - 1, "| timing", seed + 300)

# ---------------------------------------------------------------- 6. the coordinates run_gate profiles
from pdx.engine.model import Layout  # noqa: E402
from pdx.qualification import spec_for  # noqa: E402
from pdx.sim import scenarios  # noqa: E402

for d in (4, 10):
    panel, info = scenarios.generate("S1", 60, np.random.default_rng(0), d=d)
    spec = spec_for(panel, info)
    lay = Layout(spec, "null")
    names = [("gamma", 0), ("log_sigma", 1), ("b", spec.d + 1), ("L", 0), ("log_tau", 0), ("lam", 0)]
    names += [("kappa", 0)] if spec.n_treat else []
    dom, treated = panel.domains, [x for x, m in zip(panel.domains, spec.treat_mask) if m]
    rows, tri = [], np.tril_indices(spec.d)
    for nm, ix in names:
        shape = lay.slices[nm][1]
        g, k = np.unravel_index(ix, shape) if len(shape) == 2 else (ix if nm != "gamma" else 1, None)
        what = {"gamma": "log speed ratio of PIGD", "log_sigma": "log speed spread of PIGD",
                "b": f"load of group {g} on {dom[k] if k is not None else ''}",
                "L": f"group {g}, Cholesky entry ({tri[0][k] if k is not None else ''},{tri[1][k] if k is not None else ''}) on the log scale",
                "log_tau": f"log load spread of group {g} on {dom[k] if k is not None else ''}",
                "lam": f"speed-load link of group {g} on {dom[k] if k is not None else ''}",
                "kappa": f"short-duration effect of group {g} on {treated[k] if k is not None else ''}"}[nm]
        rows.append(f"{nm}[{ix}] = {what}")
    print(f"6. d = {d}, {lay.size} parameters; profiled:", "; ".join(rows))

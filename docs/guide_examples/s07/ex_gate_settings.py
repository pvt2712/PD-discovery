"""Guide §7.1-7.3: what scripts/run_engine_gate.py records and how --wp0 and --quick shape the gate (synthetic only).

code_version() runs read-only git commands. The rest uses invented counts; configs/wp0_sim_settings.json
is not read, and nothing under data/ or results/ is touched (the checkpoint lives in a temporary folder).

    OMP_NUM_THREADS=1 ../ppmi-causal/.venv/bin/python docs/guide_examples/s07/ex_gate_settings.py
"""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
import run_engine_gate as G  # noqa: E402
from pdx import gate  # noqa: E402
from pdx.sim.family_a import default_truth  # noqa: E402

# ---------------------------------------------------------------- 1. code_version
v = G.code_version()
print("1. code_version(): commit", (v["commit"] or "")[:7], "| a hash of uncommitted changes:",
      v["uncommitted_diff_sha256"] is not None, "| untracked files listed:", len(v["untracked"] or []))

# ---------------------------------------------------------------- 2. p_missing[:d] against the simulator's domains
NAMES10 = ["tremor", "axial", "brady_rigidity", "function", "independence", "cognition", "mood", "autonomic",
           "hallucinations", "sleepiness"]          # the order of run_power_check.wp0_settings (line 42)
print("\n2. which WP0 missing share each simulated domain receives with --wp0 (p_missing[:d])")
for d in (4, 6, 10):
    pairs = list(zip(default_truth(d).domains, NAMES10[:d]))
    wrong = [f"{sim} <- {src}" for sim, src in pairs if not src.startswith(sim)]
    print(f"   d = {d:>2}: mismatches {wrong if wrong else 'none'}")

# ---------------------------------------------------------------- 3. --quick with --wp0 rescales the counts
tc = (600, 150)                                    # invented WP0 counts
quick = tuple(max(2, round(200 * c / sum(tc))) for c in tc)       # gate.run_gate, lines 331-334
print(f"\n3. --quick --wp0 with invented counts {tc}: target_counts {quick}, pool n = {5 * sum(quick)}")

# ---------------------------------------------------------------- 4. one quick checkpoint name for every d
path = Path(tempfile.mkdtemp()) / "gate_quick.partial.json"
settings = {"n": 200, "d": 4, "n_datasets": 2, "seed": 2026, "quick": True, "sim_kw": {}, "fit_kw": {}}
said = []
gate.Checkpoint(path, settings, said.append).section("reference", lambda: {"pass": True})
print("\n4. a quick run with --d 4 saved:", said)
try:
    gate.Checkpoint(path, {**settings, "d": 10})
except RuntimeError as e:
    print("   the same command with --d 10 is refused:", str(e).split("/")[-1])

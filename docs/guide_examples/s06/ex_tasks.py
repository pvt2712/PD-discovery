"""Guide §6.23-6.26: tasks, seeds, configuration keys, one JSON per dataset, skip on rerun, and a
failed dataset logged rather than dropped. Nothing is fitted: one result file is written by hand
and one task fails at once (an unknown scenario). Files go to a temporary folder (synthetic only).

    OMP_NUM_THREADS=1 ../ppmi-causal/.venv/bin/python docs/guide_examples/s06/ex_tasks.py
"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from pdx import qualification as Q  # noqa: E402

a = Q.make_tasks("S1", 3, 1400, "A", d=10, fit_kw={"maxiter": 3000})
c = Q.make_tasks("S2", 2, 1400, "A", d=10, fit_kw={"maxiter": 3000})
b = Q.make_tasks("S1", 2, 1400, "B", d=10, fit_kw={"maxiter": 3000})
print("seeds  A:S1", [t["seed"] for t in a], " A:S2", [t["seed"] for t in c], " B:S1", [t["seed"] for t in b])
print("seed0=2 shifts them:", [t["seed"] for t in Q.make_tasks("S1", 3, 1400, "A", seed0=2)])
print("config keys  A:S1", Q.config_key(a[0]), " A:S2", Q.config_key(c[0]), " B:S1", Q.config_key(b[0]))
print("   replicate=False changes the key:", Q.config_key({**c[0], "replicate": False}))
vp_int = {2: 0.2, 10: 0.8}
vp_str = {"2": 0.2, "10": 0.8}
k_int = Q.config_key({**a[0], "sim_kw": {"visit_probs": vp_int}})
k_str = Q.config_key({**a[0], "sim_kw": {"visit_probs": vp_str}})
print(f"   visit_probs with int keys -> {k_int}, with str keys -> {k_str}")

out = Path(tempfile.mkdtemp())
done = dict(a[0])
p = Q._path(out, done)
print("file name:", p.name)
p.write_text(json.dumps({"task": done, "config": Q.config_key(done), "scenario": "S1", "family": "A",
                         "seed": done["seed"], "note": "written by hand"}))
bad = {"scenario": "S9", "family": "A", "n": 50, "seed": 7}
rows = Q.run_tasks([done, bad], out, jobs=1)
print("row 1 (existing file, skipped):", rows[0]["note"])
print("row 2 (failed, logged):", {k: rows[1][k] for k in ("scenario", "seed", "n", "family", "error")})
print("files now:", sorted(q.name for q in out.iterdir()))

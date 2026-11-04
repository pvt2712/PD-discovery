"""Guide §7.9-7.11 and §7.15: the small helpers of scripts/run_hp_real.py on invented inputs (synthetic only).

expert_support, write_atomic, name_links and named_support are run on an invented map and invented
verdicts. Nothing under data/ or results/ is read or written: the map and the checkpoint live in a
temporary folder.

    OMP_NUM_THREADS=1 ../ppmi-causal/.venv/bin/python docs/guide_examples/s07/ex_hp_real_helpers.py
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
import run_hp_real as R  # noqa: E402

tmp = Path(tempfile.mkdtemp())
names = ["tremor", "axial", "function", "independence"]          # an invented four-domain panel

# ---------------------------------------------------------------- 1. expert_support
links = [["tremor", "function"], ["axial", "function"], ["function", "independence"],
         ["treatment", "axial"],                 # treatment: a known name, not a panel domain
         ["mood", "cognition"]]                  # known domains that this panel leaves out
(tmp / "map.json").write_text(json.dumps({"status": "PROVISIONAL, invented for the guide", "links": links}))
sup, aside = R.expert_support(str(tmp / "map.json"), names)
print("1. expert_support")
print("   support (rows = to, columns = from):")
for i, row in enumerate(sup.astype(int)):
    print(f"   {names[i]:>12} {row.tolist()}")
print("   set aside:", aside)
(tmp / "bad.json").write_text(json.dumps({"links": [["tremor", "fuction"]]}))       # a typo
try:
    R.expert_support(str(tmp / "bad.json"), names)
except SystemExit as e:
    print("   a typo stops the run:", e)

# ---------------------------------------------------------------- 2. named_support
support = sup | np.eye(4, dtype=bool)
support[3, 1] = True                             # pretend Step 1 added axial -> independence
print("\n2. named_support")
for link in R.named_support(support, sup, names):
    print("  ", link)

# ---------------------------------------------------------------- 3. name_links
classes = [{"link": (2, 0), "verdict": "similar"}, {"link": (3, 1), "verdict": "structurally different"}]
print("\n3. name_links:", R.name_links(classes, names))
print("   the input list was changed in place:", classes[0])

# ---------------------------------------------------------------- 4. write_atomic
ck = tmp / "hp_tdpigd_checkpoint.json"
state = {"lr": 31.25, "replicates": [[12.5, True], [None, False]]}
R.write_atomic(ck, json.dumps(state))
print("\n4. write_atomic: files in the folder:", sorted(p.name for p in tmp.iterdir()))
print("   read back:", json.loads(ck.read_text()))
resume = [(np.inf if s is None else s, ok) for s, ok in json.loads(ck.read_text())["replicates"]]
print("   as run_hp_real rebuilds it for resume:", resume)

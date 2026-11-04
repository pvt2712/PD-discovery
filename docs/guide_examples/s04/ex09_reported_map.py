"""Section 4.15: reported_map, the map from the full fit with its marks (aggregate output only).
Synthetic family-A panel (scenario S1). Synthetic data only."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import numpy as np  # noqa: E402

from pdx.sim import scenarios  # noqa: E402
from pdx.step1 import reported_map  # noqa: E402

panel, _ = scenarios.generate("S1", 600, np.random.default_rng(1), d=4)
t0 = time.time()
out = reported_map(panel, names=panel.domains)
print(f"{time.time() - t0:.2f} s; keys: {list(out)}")
print("rows", out["rows"], "| complete rows", out["complete_rows"], "| participants", out["participants"])
print("config:", json.dumps(out["config"]))
print("edges between year t and year t+1, and within year t+1:")
for e in out["edges"]:
    if e["to"].endswith("_t1"):
        print(f"   {e['from']:16s} {e['edge']} {e['to']}")
print("lagged_adjacency [to, from]:", out["lagged_adjacency"])

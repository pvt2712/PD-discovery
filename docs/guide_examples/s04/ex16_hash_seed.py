"""Notes 4.N: the same Step 1 fit, run in two fresh Python processes that differ only in their
string-hash seed, can return different graphs (causal-learn 0.1.4.8 iterates over sets of nodes
in its Possible-D-SEP step). Synthetic data only."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CHILD = """
import sys; sys.path.insert(0, {src!r})
import numpy as np
from pdx.sim import scenarios
from pdx.step1 import MARK, Step1Config, lagged_adjacency, run_fci
panel, _ = scenarios.generate("S1", 400, np.random.default_rng(8), d=6)
G, cols = run_fci(panel, Step1Config(test="cluster_fisherz", depth=4))[:2]
print("lagged adjacency:", lagged_adjacency(G, cols, panel.d).astype(int).tolist())
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        if G[i, j]:
            print(cols[i], MARK[G[i, j]].replace(">", "<") + "-" + MARK[G[j, i]], cols[j])
""".format(src=str(ROOT / "src"))

runs = {}
for seed in ("0", "1"):
    env = dict(os.environ, PYTHONHASHSEED=seed, OMP_NUM_THREADS="1")
    runs[seed] = subprocess.run([sys.executable, "-c", CHILD], env=env, capture_output=True, text=True,
                                check=True).stdout.splitlines()
print("same lagged adjacency:", runs["0"][0] == runs["1"][0])
print("edges only with PYTHONHASHSEED=0:", sorted(set(runs["0"][1:]) - set(runs["1"][1:])))
print("edges only with PYTHONHASHSEED=1:", sorted(set(runs["1"][1:]) - set(runs["0"][1:])))

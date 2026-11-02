"""FCI (tiers) at the pooled base size with the conditioning-set cap applied to the adjacency search and to the
possible-d-separation stage (PROTOCOL.md §5.1, §10.1). Synthetic data only. Writes results/timing_fci_depth.csv."""
import sys
import time
import warnings

import pandas as pd

from escalation_discovery import arm_a, expert as E, metrics as MT
from escalation_discovery import config as C
from escalation_discovery.methods.classical import FCI

warnings.filterwarnings("ignore")
spec = E.load()
gens = arm_a.load_generators("synthetic")
truth = arm_a.get_truth("synthetic", gens["G1"], ())
oracle = MT.AdjustmentOracle(truth.dag, truth.observed)
data, ts = arm_a.make_data(gens["G1"], arm_a.load_bank("synthetic", spec), arm_a.Setting("G1-timing", "G1"), seed=123)
data.knowledge = True
plan = [(t, int(d)) for t in sys.argv[1].split(",") for d in sys.argv[2].split(",")]
rows = []
for test, depth in plan:
    t0 = time.time()
    g = FCI(test, depth=depth).fit(data)
    s = MT.score_all(g, truth, oracle)
    rows.append(dict(method=f"fci_{'cgcm' if test == 'cgcm' else 'fz'}", depth=depth, seconds=round(time.time() - t0, 1),
                     edges=g.n_edges(), adj_f1=s["adj_f1"], ah_f1=s["ah_f1"], pa_decision_recall=s["pa_decision_recall"]))
    print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(C.results_dir() / "timing_fci_depth.csv", index=False)

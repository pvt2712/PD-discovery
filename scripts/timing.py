"""Wall time of every available learner on one synthetic dataset of a given size (PROTOCOL.md §10.1).

    uv run python scripts/timing.py [participants] [generator] [group]

group: all (default) | light (causal-learn, continuous, temporal) | tetrad | external (side environments), or a
comma-separated list of learner names. Writes results/timing_<participants>_<generator>_<label>.csv (aggregates of
synthetic data only; label = the group, or "list" with a hash of the names).
"""
import sys
import time
import warnings

import pandas as pd

from escalation_discovery import arm_a, expert as E, metrics as MT
from escalation_discovery import config as C
from escalation_discovery.methods import registry
from escalation_discovery.methods.external import External

warnings.filterwarnings("ignore")
n = int(sys.argv[1]) if len(sys.argv) > 1 else arm_a.BASE_N
kind = sys.argv[2] if len(sys.argv) > 2 else "G1"
group = sys.argv[3] if len(sys.argv) > 3 else "all"
label = group if group in ("all", "light", "tetrad", "external") else "list" + str(abs(hash(group)) % 10000)
spec = E.load()
gens = arm_a.load_generators("synthetic")
bank = arm_a.load_bank("synthetic", spec)
setting = arm_a.Setting(f"{kind}-timing", kind, n=n)
truth = arm_a.get_truth("synthetic", gens[kind], ())
oracle = MT.AdjustmentOracle(truth.dag, truth.observed)
data, ts = arm_a.make_data(gens[kind], bank, setting, seed=123)
entries, missing = registry.available(registry.panel())


def in_group(e):
    """A group name, or a comma-separated list of learner names."""
    ext = isinstance(e.method, External)
    groups = {"all": True, "tetrad": arm_a.is_heavy(e.name), "external": ext,
              "light": not ext and not arm_a.is_heavy(e.name)}
    return groups[group] if group in groups else e.name in group.split(",")


rows = []
for e in [e for e in entries if in_group(e)]:
    data.knowledge = e.knowledge
    t0 = time.time()
    try:
        g, err = e.method.fit(data), ""
    except Exception as ex:
        g, err = None, repr(ex)[:300]
    secs = time.time() - t0
    s = MT.score_all(g, truth, oracle) if g is not None else {}
    rows.append(dict(method=e.name, seconds=round(secs, 1), edges=g.n_edges() if g else None, adj_f1=s.get("adj_f1"),
                     ah_f1=s.get("ah_f1"), error=err))
    print(rows[-1], flush=True)
    pd.DataFrame(rows).assign(rows=len(ts), participants=int(ts.PATNO.nunique())).to_csv(
        C.results_dir() / f"timing_{n}_{kind}_{label}.csv", index=False)

"""Which part of Tetrad's tiered BOSS-FCI exhausts memory at the pooled base size? Synthetic data only."""
import resource
import sys
import time
import warnings

from escalation_discovery import arm_a, expert as E
from escalation_discovery.methods.tetrad import Tetrad

warnings.filterwarnings("ignore")
spec = E.load()
gens = arm_a.load_generators("synthetic")
data, ts = arm_a.make_data(gens["G1"], arm_a.load_bank("synthetic", spec), arm_a.Setting("G1-timing", "G1"), seed=123)
for alg, knowledge, cfg in [("boss_fci", False, {}), ("boss_fci", True, {"forbid_within": False}), ("gfci", False, {})]:
    data.knowledge = knowledge
    t0 = time.time()
    try:
        g = Tetrad(alg, **cfg).fit(data)
        res = f"{g.n_edges()} edges"
    except Exception as ex:
        res = repr(ex)[:120]
    print(dict(alg=alg, knowledge=knowledge, cfg=cfg, seconds=round(time.time() - t0, 1), result=res,
               max_rss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 1)), flush=True)

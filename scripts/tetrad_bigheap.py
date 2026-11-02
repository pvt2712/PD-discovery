"""Tiered BOSS-FCI and GFCI at the pooled base size with a large JVM heap, one at a time (synthetic data only)."""
import resource
import time
import warnings

from escalation_discovery import arm_a, expert as E
from escalation_discovery.methods.tetrad import Tetrad

warnings.filterwarnings("ignore")
spec = E.load()
gens = arm_a.load_generators("synthetic")
data, ts = arm_a.make_data(gens["G1"], arm_a.load_bank("synthetic", spec), arm_a.Setting("G1-timing", "G1"), seed=123)
data.knowledge = True
for alg in ("boss_fci", "gfci"):
    t0 = time.time()
    try:
        g = Tetrad(alg).fit(data)
        res = f"{g.n_edges()} edges"
    except Exception as ex:
        res = repr(ex)[:120]
    print(dict(alg=alg, knowledge=True, heap="12g", seconds=round(time.time() - t0, 1), result=res,
               max_rss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 1)), flush=True)

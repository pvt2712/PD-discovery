"""How many conditional-independence tests PC and FCI run at the pooled base size, by conditioning-set size, with and
without a depth cap (PROTOCOL.md §10.1). causal-learn 0.1.4.8's PC has no depth argument (it is ignored); FCI has one. Uses the fast Fisher-z test to count; the clustered GCM's cost is the count
times its per-test time. Synthetic data only. Writes results/test_counts.csv."""
import collections
import contextlib
import io
import time
import warnings

import pandas as pd

from escalation_discovery import arm_a, expert as E
from escalation_discovery import config as C
from escalation_discovery import knowledge as K
from escalation_discovery.methods.base import copula

warnings.filterwarnings("ignore")
from causallearn.utils import cit as CIT_mod  # noqa: E402

counts = collections.Counter()
_orig = CIT_mod.MV_FisherZ.__call__


def counting_call(self, X, Y, condition_set=None):
    counts[len(condition_set or ())] += 1
    return _orig(self, X, Y, condition_set)


CIT_mod.MV_FisherZ.__call__ = counting_call
spec = E.load()
gens = arm_a.load_generators("synthetic")
data, ts = arm_a.make_data(gens["G1"], arm_a.load_bank("synthetic", spec), arm_a.Setting("G1-timing", "G1"), seed=123)
X = copula(data.X)
import sys
algs = sys.argv[1].split(",") if len(sys.argv) > 1 else ["pc", "fci"]
depths = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [-1, 3]
out_name = f"test_counts_{'_'.join(algs)}.csv"
rows = []
for alg in algs:
    for depth in depths:
        counts.clear()
        t0 = time.time()
        bk, _ = K.background(data.names, data.tiers, data.context, data.exogenous)
        with contextlib.redirect_stdout(io.StringIO()):
            if alg == "fci":
                fci = K.install(data.context, data.exogenous)
                G, _ = fci(X, "mv_fisherz", 0.01, depth=depth, background_knowledge=bk, show_progress=False, verbose=False,
                           node_names=list(data.names))
                edges = int((G.graph != 0).sum() // 2)
            else:
                from causallearn.graph.GraphNode import GraphNode
                from causallearn.search.ConstraintBased.PC import pc
                alias = {n: f"X{i + 1}" for i, n in enumerate(data.names)}
                bk, _ = K.background(data.names, data.tiers, data.context, data.exogenous, node=lambda n: GraphNode(alias[n]))
                cg = pc(X, 0.01, "mv_fisherz", stable=True, depth=depth, background_knowledge=bk, show_progress=False)
                edges = int((cg.G.graph != 0).sum() // 2)
        r = dict(alg=alg, depth=depth, tests=sum(counts.values()), seconds=round(time.time() - t0, 1), edges=edges,
                 **{f"size_{k}": v for k, v in sorted(counts.items())})
        rows.append(r)
        print(r, flush=True)
        pd.DataFrame(rows).assign(rows=len(ts)).to_csv(C.results_dir() / out_name, index=False)

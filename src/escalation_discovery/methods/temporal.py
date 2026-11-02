"""temporal — J-PCMCI+ on the long panel; not in the panel since 29 October 2026 (PROTOCOL.md §5.5).

J-PCMCI+ (Günther, Ninad & Runge, UAI 2023, PMLR 216:766-776; arXiv 2306.12896; tigramite 5.2.10.1) treats each
participant as one dataset on a common year axis, with the context, exogenous and baseline nodes as space-context
variables and a year one-hot vector as the time dummy. Participant one-hot dummies (the space dummy) are omitted:
with ~1,700 participants they would add as many columns, and in ppmi-causal's run they had no power
(docs/paper2_baselines.md). ParCorrMult on copula normal scores; tau_max = 1. Links are mapped onto the two-slice
names: a lag-1 link X(t-1) -> Y(t) is X_0 -> Y_1, a contemporaneous link X - Y is drawn in both years, a context
link C -> Y in both years. Tiers are not passed: time order is built into the method (lags point forward) and the
context variables cannot receive links.
"""
import numpy as np
import pandas as pd

from .. import expert as E
from ..graphs import MarkGraph
from .base import Method, copula

_LINK = {"-->": ("T", "A"), "<--": ("A", "T"), "o-o": ("C", "C"), "x-x": ("C", "C"), "o->": ("C", "A"), "<-o": ("A", "C"),
         "<->": ("A", "A")}


class JPCMCIPlus(Method):
    name = "jpcmciplus"
    family = "temporal"
    output = "CPDAG"
    accepts_knowledge = False
    missing = "native"
    version = "tigramite 5.2.10.1"

    def defaults(self):
        return dict(alpha=0.01)

    def grid(self):
        return [dict(alpha=a) for a in (0.001, 0.005, 0.01, 0.05, 0.1, 0.2)]

    def _fit(self, data):
        from tigramite import data_processing as pp
        from tigramite.independence_tests.parcorr_mult import ParCorrMult
        from tigramite.jpcmciplus import JPCMCIplus
        long, base = data.long, data.base
        if long is None or base is None:
            raise ValueError("J-PCMCI+ needs the long panel and the baseline table")
        tv = sorted({n[:-2] for n in data.names if E.slice_of(n) is not None}, key=lambda v: data.names.index(f"{v}_0"))
        ctx = [n for n in data.names if E.slice_of(n) is None and n != "t"]
        L = long.copy()
        L[tv] = copula(L[tv].to_numpy(dtype=float))
        B = base.set_index("PATNO")[ctx].astype(float)
        B = pd.DataFrame(copula(B.to_numpy()), index=B.index, columns=ctx).fillna(0.0)
        T = int(L.year.max()) + 1
        years = np.arange(T)
        pids = [p for p, d in L.groupby("PATNO") if len(d) >= 2]
        nsys, nctx = len(tv), len(ctx)
        frames = {}
        for k, p in enumerate(pids):
            d = L[L.PATNO == p].set_index("year").reindex(years)[tv].to_numpy()
            arr = np.hstack([d, np.repeat(B.loc[p].to_numpy()[None, :], T, axis=0), np.identity(T)])
            frames[k] = np.where(np.isnan(arr), 999.0, arr)
        sys_idx = list(range(nsys))
        ctx_idx = list(range(nsys, nsys + nctx))
        t_idx = nsys + nctx
        vector_vars = {i: [(i, 0)] for i in sys_idx + ctx_idx}
        vector_vars[t_idx] = [(i, 0) for i in range(nsys + nctx, nsys + nctx + T)]
        names = tv + ctx + ["year-dummy"]
        df = pp.DataFrame(data=frames, vector_vars=vector_vars, analysis_mode="multiple", var_names=names, missing_flag=999.0)
        nc = {i: "system" for i in sys_idx}
        nc.update({i: "space_context" for i in ctx_idx})
        nc[t_idx] = "time_dummy"
        model = JPCMCIplus(dataframe=df, cond_ind_test=ParCorrMult(significance="analytic"), node_classification=nc, verbosity=0)
        res = model.run_jpcmciplus(tau_min=0, tau_max=1, pc_alpha=self.cfg["alpha"])
        G = res["graph"]
        g = MarkGraph(data.names, "CPDAG")
        for i in range(len(names) - 1):
            for j in range(len(names) - 1):
                for tau in (0, 1):
                    link = G[i, j, tau]
                    if link in ("", None) or (tau == 0 and i >= j):
                        continue
                    mi, mj = _LINK.get(link, ("C", "C"))
                    a, b = names[i], names[j]
                    if tau == 1 and a in tv and b in tv:
                        g.add(f"{a}_0", f"{b}_1", "T", "A")
                    elif tau == 0 and a in tv and b in tv:
                        for s in ("_0", "_1"):
                            g.add(a + s, b + s, mi, mj)
                    elif tau == 0:
                        c, v = (a, b) if a in ctx else (b, a)
                        if v in tv:
                            for s in ("_0", "_1"):
                                g.add(c, v + s, "T", "A")
        return g

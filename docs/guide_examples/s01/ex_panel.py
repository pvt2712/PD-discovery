"""Sections 1.2-1.4: the Panel container, its checks, its helpers and its counts.
Three made-up participants with d = 2 domains."""
import numpy as np
from _setup import TMP  # noqa: F401  (import paths)

from pdx.data import Panel

nan = np.nan
y = [np.array([[0.1, -0.2], [0.4, nan], [0.9, 0.3]]),     # participant 0 (TD): 3 visits, one domain missing
     np.array([[1.0, 0.5], [1.6, 1.1]]),                   # participant 1 (PIGD): 2 visits
     np.array([[-0.3, nan]])]                              # participant 2 (PIGD): landmark only
t = [[1.0, 2.1, 3.0], [0.9, 2.0], [1.2]]
p = Panel(y, t, group=[0, 1, 1], z=[[-0.5, 0.2], [0.5, -0.1], [0.0, -0.1]], year=[[1, 2, 3], [1, 2], [1]])

print("== 1.2 defaults filled in by __post_init__ ==")
print("y[2] shape:", p.y[2].shape, "| t[0]:", p.t[0], "| group dtype:", p.group.dtype)
print("dose[0]:", p.dose[0], "| exposure[1]:", p.exposure[1], "| on[2]:", p.on[2])
print("z shape without covariates:", Panel(y, t, [0, 1, 1]).z.shape)

print("\n== 1.2 the checks ==")
for label, args in [("times not increasing", dict(y=y[:1], t=[[1.0, 1.0, 3.0]], group=[0])),
                    ("one time missing", dict(y=y[:1], t=[[1.0, 2.0]], group=[0])),
                    ("group too short", dict(y=y, t=t, group=[0, 1])),
                    ("one visit given as a flat vector", dict(y=[np.array([0.1, -0.2])], t=[[1.0]], group=[0]))]:
    try:
        Panel(**args)
        print(f"{label:34s} -> accepted")
    except ValueError as e:
        print(f"{label:34s} -> ValueError: {e}")
q = Panel([np.zeros((2, 2)), np.zeros((1, 3))], [[1.0, 2.0], [1.0]], [0, 1])
print("domains differ between participants  -> accepted; d =", q.d, "but y[1] has", q.y[1].shape[1], "columns")

print("\n== 1.3 properties and helpers ==")
print("n, d, p, n_groups, n_transitions:", p.n, p.d, p.p, p.n_groups, p.n_transitions())
s = p.subset([2, 0])
print("subset([2, 0]): groups", s.group, "| years", [a.tolist() for a in s.year], "| z rows", s.z.tolist())
w = p.with_y([v + 1.0 for v in p.y])
print("with_y: new y[0][0] =", w.y[0][0], "| old y[0][0] =", p.y[0][0], "| t shares memory:", np.shares_memory(w.t[0], p.t[0]))
print("only group 1 kept -> n_groups =", p.subset([1, 2]).n_groups, "| only group 0 kept -> n_groups =",
      p.subset([0]).n_groups)

print("\n== 1.4 aggregate_summary ==")
print(p.aggregate_summary())
miss = [np.isnan(v).mean() for v in p.y[1:]]
pooled = np.isnan(np.vstack(p.y[1:])).mean()
print("PIGD: mean of per-participant shares =", round(float(np.mean(miss)), 4), "| pooled share of cells =", round(float(pooled), 4))

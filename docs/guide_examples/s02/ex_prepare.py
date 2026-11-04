"""Sections 2.24, 2.26 and 2.27: prepare() on a hand-made panel of six participants and two domains."""
import numpy as np

import _setup                                    # noqa: F401
from pdx.data import Panel
from pdx.engine.likelihood import _pd, prepare

nan = np.nan
y = [np.array([[0.1, 0.3], [0.2, nan], [0.5, 0.4]]),              # TD, three visits, one score missing
     np.array([[-0.2, 0.1], [0.0, 0.2]]),                          # TD, two visits
     np.array([[0.4, -0.1]]),                                      # TD, landmark only
     np.array([[0.6, nan], [0.7, 0.9], [nan, nan], [1.1, 1.0]]),   # PIGD, landmark score and a whole visit missing
     np.array([[0.3, 0.5], [0.6, 0.4]]),                           # PIGD
     np.array([[0.9, 0.8], [1.0, 1.1]])]                           # PIGD
t = [np.array([0.0, 1.0, 2.1]), np.array([0.0, 1.2]), np.array([0.0]),
     np.array([0.0, 0.9, 2.0, 3.2]), np.array([0.0, 1.0]), np.array([0.0, 1.1])]
panel = Panel(y, t, group=[0, 0, 0, 1, 1, 1], dose=[np.zeros(len(a)) for a in t])
meas_var = np.array([0.02, 0.05])
prep = prepare(panel, meas_var, n_groups=2)
print("N, T:", prep.N, prep.T, "| Ti:", prep.Ti.tolist())
print("Y[3] (missing set to 0):\n", prep.Y[3].numpy(), "\nO[3]:\n", prep.O[3].numpy())
print("dt:\n", prep.dt.numpy())
print("mu1 (phenotype means of the landmark scores):\n", prep.mu1.numpy())
y1 = np.array([yi[0] for yi in y])
for g in (0, 1):
    raw = np.ma.cov(np.ma.masked_invalid(y1[panel.group == g]), rowvar=False).filled(0.0)
    print(f"phenotype {g}: pairwise covariance of the landmark scores\n", raw,
          f"\n  minus diag(meas_var), floored at 1e-3 (S1[{g}]):\n", prep.S1[g].numpy())
print("y1c (missing landmark score filled by its phenotype mean, then centred on the pooled mean):\n",
      prep.y1c.numpy())
print("\n_pd on an indefinite matrix:", _pd(np.array([[1.0, 2.0], [2.0, 1.0]])).round(4).tolist(),
      "eigenvalues", np.linalg.eigvalsh(_pd(np.array([[1.0, 2.0], [2.0, 1.0]]))).round(4))

# two small branches: a batch of landmark-only participants, and phenotypes with two or fewer participants
import torch                                     # noqa: E402

from pdx.engine.likelihood import participant_loglik   # noqa: E402
from pdx.engine.model import Layout, ModelSpec          # noqa: E402

spec = ModelSpec(d=2, support=np.eye(2), meas_var=meas_var)
lay = Layout(spec, "null")
x = torch.zeros(lay.size)
print("\nlandmark-only participant 2 in a batch of its own:", participant_loglik(x, lay, prep, torch.tensor([2])).tolist())
small = panel.subset([0, 1, 4])                  # two TD participants and one PIGD participant
print("S1 with two TD and one PIGD participant (identity minus meas_var):\n", prepare(small, meas_var, 2).S1.numpy())

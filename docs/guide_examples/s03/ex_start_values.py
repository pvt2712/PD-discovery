"""Sections 3.2 to 3.5: the start values, built from yearly least-squares fits per phenotype.
Synthetic data only (scenario S1: PIGD runs TD's mechanism at twice the median speed)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

import numpy as np
from scipy import integrate, linalg
from helpers import small_panel, spec_for
from pdx.engine.fit import _drift_from_yearly, _load_from_intercept, _yearly_ols, start_values
from pdx.engine.likelihood import prepare, value_and_grad
from pdx.engine.model import Layout, unpack_numpy

np.set_printoptions(precision=3, suppress=True)
panel, info = small_panel("S1", n=80, seed=1, d=4)
spec = spec_for(panel, info)
truth = info["truth"]
d = spec.d

print("== 3.2 _yearly_ols ==")
for g in (0, 1):
    pairs = kept = 0
    for i in np.where(panel.group == g)[0]:
        y, t = panel.y[i], panel.t[i]
        for k in range(len(t) - 1):
            pairs += 1
            kept += (0.5 <= t[k + 1] - t[k] <= 1.6) and not (np.isnan(y[k]).any() or np.isnan(y[k + 1]).any())
    B, a, Sig = _yearly_ols(panel, g, d)
    print(f"group {g}: {kept} of {pairs} consecutive pairs used")
    print(" B (yearly matrix):\n", B)
    print(" true yearly matrix at the median speed, expm(%.0f A):\n" % truth.speed[g], linalg.expm(truth.speed[g] * truth.A))
    print(" a:", a, "| diag(Sigma):", np.diag(Sig))

print("\n== 3.3 _drift_from_yearly ==")
A_true = truth.A
print("exact case: logm(expm(A)) recovers A on the support, max error %.1e"
      % np.abs(_drift_from_yearly(linalg.expm(A_true), spec) - A_true).max())
for g in (0, 1):
    B, _, _ = _yearly_ols(panel, g, d)
    print(f"group {g} start drift (own time scale):\n", _drift_from_yearly(B, spec))
Bneg = np.diag([0.9, -0.2, 0.7, 0.8])                   # a yearly matrix with a negative eigenvalue
print("negative eigenvalue: diag of the start drift =", np.diag(_drift_from_yearly(Bneg, spec)),
      "| real part of log(-0.2) =", round(float(np.log(0.2)), 3))

print("\n== 3.4 _load_from_intercept ==")
s, b_true = 2.0, truth.b[1]
M_block = linalg.expm(np.block([[s * A_true, s * np.eye(d)], [np.zeros((d, d)), np.zeros((d, d))]]))[:d, d:]
M_quad, _ = integrate.quad_vec(lambda w: linalg.expm(A_true * w), 0.0, s)
print("M from the block exponential vs numerical integral: max difference %.1e" % np.abs(M_block - M_quad).max())
a = M_block @ b_true
print("true b:", b_true, "| recovered from a = M b:", _load_from_intercept(A_true, a, s))

print("\n== 3.5 start_values ==")
prep = prepare(panel, spec.meas_var, spec.n_groups)
for kind in ("null", "free", "joint"):
    lay = Layout(spec, kind)
    x = start_values(panel, spec, kind)
    par = unpack_numpy(x, lay)
    nll, _ = value_and_grad(x, lay, prep)
    line = f"{kind:5s}: size {lay.size}, loglik at start {-nll:.3f}"
    if lay.has("gamma"):
        line += f", gamma start {lay.get(x, 'gamma')[0]:.3f} (speed {np.exp(lay.get(x, 'gamma')[0]):.3f}; truth 2)"
    print(line)
    print("   b start:", par["b"].round(3).tolist())
    print("   diag Q start:", np.array([np.diag(q) for q in par["Q"]]).round(3).tolist(),
          "| sigma:", par["sigma"].round(3), "| tau:", par["tau"][:, 0].round(3))
print("truth: b =", truth.b.round(3).tolist(), "| diag Q = %.2f (reference time scale)" % truth.q,
      "| sigma =", truth.sigma, "| tau =", truth.tau)

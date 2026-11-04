"""Guide §7.43-7.57: cheap checks of the toy simulations' building blocks (synthetic only).

No toy is run end to end (each takes hours). The pieces below take seconds, write nothing, and
read no file: they import the toys' own functions from scripts/toy/ and run them on fresh synthetic data.

    OMP_NUM_THREADS=1 ../ppmi-causal/.venv/bin/python docs/guide_examples/s07/ex_toys.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "toy"))
import dynotears as DY  # noqa: E402
import toy_random_speed as RS  # noqa: E402
import toy_speed_vs_pathway as T  # noqa: E402
import ts_ldiffpc as TS  # noqa: E402

np.set_printoptions(precision=4, suppress=True)

# ---------------------------------------------------------------- 1. speed is a matrix power
B1, a1, S1 = T.discrete(T.drift(False), 1.0)
B2, a2, S2 = T.discrete(T.drift(False), 2.0)
print("1. toy 1: yearly matrices of one drift at speeds 1 and 2")
print("   max |B(2) - B(1)^2| =", f"{np.abs(B2 - B1 @ B1).max():.2e}")
names = T.NAMES
for (i, j) in [(1, 0), (4, 1), (5, 4), (5, 1), (1, 1)]:
    print(f"   {names[j]:>9} -> {names[i]:<12} TD {B1[i, j]:.4f}  PIGD {B2[i, j]:.4f}  ratio {B2[i, j] / B1[i, j]:.3f}")
from scipy import linalg  # noqa: E402
print("   mpow on the true B(1), whose six eigenvalues are all e^-0.2: max |mpow(B(1), 2) - B(2)| =",
      f"{np.abs(T.mpow(B1, 2.0) - B2).max():.2e}")
rng = np.random.default_rng(0)
errs = []
for _ in range(20):
    Bo, _ = T.ols(*T.centred(*T.simulate(rng, 800, 5, T.drift(False), 1.0)))
    errs += [np.abs(T.mpow(Bo, s) - linalg.fractional_matrix_power(Bo, s).real).max() for s in (2.0, 1.7)]
print(f"   mpow on 20 OLS estimates (800 x 5 transitions), s = 2 and 1.7, against scipy: largest error {max(errs):.1e}")

# ---------------------------------------------------------------- 2. the yearly footprint of one link difference
D =linalg.expm(2.0 * T.drift(True)) - linalg.expm(2.0 * T.drift(False))
moved = [f"{names[j]}->{names[i]}" for i in range(6) for j in range(6) if i != j and abs(D[i, j]) > 1e-3]
print("\n2. yearly entries (off the diagonal) moved by the PIGD-only Axial -> Independence link at speed 2:", moved)

# ---------------------------------------------------------------- 3. the torch likelihood of toy 3b equals the numpy one
import torch  # noqa: E402
import toy_random_speed_scale_person as PP  # noqa: E402
rng = np.random.default_rng(1)
X1 = RS.simulate(rng, 100, 5, RS.drift(False, False), 1.0)
X2 = RS.simulate(rng, 50, 5, RS.drift(False, False), 2.0)
data = [(torch.tensor(X[:, :-1]), torch.tensor(X[:, 1:])) for X in (X1, X2)]
print("\n3. toy_random_speed_scale_person.py's check mode: torch and numpy negative log-likelihoods at the start values")
for model in ("H1", "H3"):
    th = RS.start_values(X1, X2, model)
    print(f"   {model}: torch {float(PP.nll(torch.tensor(th), model, data)):.6f}  numpy {RS.negll(th, model, X1, X2):.6f}")

# ---------------------------------------------------------------- 4. DYNOTEARS gradient check
print("\n4. dynotears.gradient_check(): largest relative error", f"{DY.gradient_check(0):.2e}",
      "| with d = 5, two lags:", f"{DY.gradient_check(1, d=5, p=2):.2e}")

# ---------------------------------------------------------------- 5. tsLDiffPC's skeleton on a speed-only difference
rng = np.random.default_rng(5)
Byr = np.array([[0.6, 0.0], [0.3, 0.6]])          # yearly matrix at speed 1 (X(t) = B X(t-1) + noise)


def two_slice(B, n):
    X = np.zeros((n + 101, 2))
    for t in range(1, len(X)):
        X[t] = B @ X[t - 1] + rng.normal(size=2)
    X = X[100:]
    return np.column_stack([X[:-1], X[1:]])


Z1, Z2 = two_slice(Byr, 4000), two_slice(Byr @ Byr, 4000)      # regime 2 = the same system, twice as fast
kept, ntest = TS.skeleton(Z1, Z2, 2)
print("\n5. ts_ldiffpc.skeleton, regime 2 = regime 1 run twice as fast (yearly matrix squared):")
print("   retained lagged difference edges F[to, from]:\n  ", TS.lag_matrix(kept, 2).astype(int).tolist(),
      "| same-time pairs:", TS.contemporaneous_pairs(kept, 2), "| tests run:", ntest)
print("   root causes (targets of retained lagged edges):", np.where(TS.lag_matrix(kept, 2).any(1))[0].tolist())

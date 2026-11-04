"""Shared set-up for the section 5 examples. Synthetic data only: nothing here reads data/ or
writes results/.

engine_panel() returns a small panel whose later visits are simulated from the engine itself,
with known parameters, so the model that is fitted is the model that generated the data:
- the design (phenotype labels, covariates, visit dates, landmark scores, missing values) comes
  from simulator family A (pdx.sim.scenarios, scenario S0);
- the later visits come from pdx.engine.simulate.simulate_from_fit, applied to a hand-built
  "fit" that holds the true parameters (the same routine the H-P bootstrap uses).

The specification is deliberately light, so that one pair of fits takes seconds: no treatment
terms, a participant load without covariate shifts, and six quadrature nodes.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT / "src", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

from pdx.engine.fit import Fit, start_values  # noqa: E402
from pdx.engine.model import Layout, ModelSpec, pack_chol, pack_drift, unpack_numpy  # noqa: E402
from pdx.engine.simulate import simulate_from_fit  # noqa: E402
from pdx.sim import scenarios  # noqa: E402

DOMAINS = ["tremor", "axial", "function", "independence"]


def light_spec(panel, support, meas_var) -> ModelSpec:
    return ModelSpec(d=panel.d, support=support, p=panel.p, treatment="none", meas_var=meas_var,
                     n_quad=6, load_on_z=False, load_on_y1=False)


def true_params(spec: ModelSpec, speed: float = 2.0, theta=(0.0, 0.0, 0.0, 0.0), extra: dict | None = None):
    """The true drift A (TD), PIGD's drift on the observed time scale, and the rest."""
    A = np.diag(np.full(4, -0.25))
    A[2, 0], A[2, 1], A[3, 2] = 0.10, 0.20, 0.25          # tremor, axial -> function -> independence
    A1 = speed * A
    for (to, frm), v in (extra or {}).items():            # deviations beyond speed (reference scale)
        A1[to, frm] += speed * v
    return {"A": A, "A1": A1, "speed": speed, "theta": np.asarray(theta, float)}


def _fake_fit(panel, spec, kind: str, tp: dict) -> Fit:
    lay = Layout(spec, kind)
    x = start_values(panel, spec, kind)
    lay.put(x, "A", pack_drift(tp["A"], spec))
    if kind == "null":
        lay.put(x, "gamma", [np.log(tp["speed"])])
        Q1, b1, tau1 = 0.10 * np.eye(4), np.full(4, 0.05), 0.10
    else:                                                  # the free model carries the speed in A1, b1, Q1, tau1
        lay.put(x, "A1", pack_drift(tp["A1"], spec))
        Q1, b1, tau1 = tp["speed"] * 0.10 * np.eye(4), tp["speed"] * np.full(4, 0.05), tp["speed"] * 0.10
    lay.put(x, "beta", np.zeros(spec.p))
    lay.put(x, "theta", tp["theta"])
    lay.put(x, "log_sigma", np.log([0.3, 0.3]))
    lay.put(x, "b", np.vstack([np.full(4, 0.05), b1]))
    lay.put(x, "L", np.vstack([pack_chol(0.10 * np.eye(4)), pack_chol(Q1)]))
    lay.put(x, "lam", np.zeros((2, 4)))
    lay.put(x, "log_tau", np.log(np.vstack([np.full(4, 0.10), np.full(4, tau1)])))
    return Fit(kind, spec, lay, x, float("nan"), True, "true parameters", 0, unpack_numpy(x, lay))


def engine_panel(kind: str = "null", n: int = 80, seed: int = 3, speed: float = 2.0,
                 theta=(0.0, 0.0, 0.0, 0.0), extra: dict | None = None):
    """(panel, spec, true parameters). kind "null": PIGD runs TD's drift at `speed`; kind "free":
    PIGD's drift is speed * (A + deviations in `extra`), a mechanism difference."""
    rng = np.random.default_rng(seed)
    design, info = scenarios.generate("S0", n, rng, d=4)
    spec = light_spec(design, info["support"], np.full(4, 0.02))
    tp = true_params(spec, speed, theta, extra)
    panel = simulate_from_fit(_fake_fit(design, spec, kind, tp), design, rng)
    return panel, spec, tp

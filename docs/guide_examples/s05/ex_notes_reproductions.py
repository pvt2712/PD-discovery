"""Section 5.N, Notes 2 and 3: minimal reproductions of two crashes. Synthetic data only; no model
is fitted (the second uses the fixture's true-parameter "fits")."""
import warnings
from types import SimpleNamespace

import numpy as np

from _fixture import _fake_fit, engine_panel
from pdx import speed
from pdx.engine.model import ModelSpec
from pdx.localise import FreeDraws, per_link_tests

warnings.simplefilter("ignore", RuntimeWarning)          # nanquantile of an empty array warns before it fails

# Note 2: the point speed is identified, but no Monte Carlo draw is
sup = np.eye(2, dtype=bool)
sup[1, 0] = True
spec2 = ModelSpec(d=2, support=sup)
a0, a1 = np.array([-0.04, -0.06, 0.03]), np.array([-0.05, -0.12, 0.05])
d0 = np.tile(a0, (50, 1))
d0[:, 1] = -0.04                                          # the only voting entry falls below 0.05 in every draw
try:
    per_link_tests(SimpleNamespace(spec=spec2), None, draws=FreeDraws(a0, a1, d0, np.tile(a1, (50, 1))))
except Exception as exc:
    print("Note 2: per_link_tests raised", f"{type(exc).__name__}: {exc}")

# Note 3: every speed-bootstrap sample fails (here because of an invalid fit option)
panel, spec, truth = engine_panel("null", n=80, seed=3)
f0, f1 = _fake_fit(panel, spec, "null", truth), _fake_fit(panel, spec, "free", truth)
try:
    speed.bootstrap_speed(panel, spec, f0, f1, B=2, fit_kw={"not_an_option": 1})
except Exception as exc:
    print("Note 3: bootstrap_speed raised", f"{type(exc).__name__}: {exc}")

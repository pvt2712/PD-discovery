"""Shared setup for the Section 2 examples (the engine: model and likelihood). Synthetic data only.

- Puts src/ and tests/ on the import path, so the examples run from the repository folder
  without installing the package.
- Imports pdx.engine.model first: that import sets torch's default dtype to float64, which the
  transition code relies on (see the guide, section 2.1).
- Offers the synthetic panels of the unit tests (simulator family A, scenario S1) and the
  helper that adds extra missing values, exactly as tests/test_engine.py does.

Nothing here reads data/ or results/.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

import numpy as np                      # noqa: E402
import torch                            # noqa: E402

import pdx.engine.model                 # noqa: E402,F401  (sets the float64 default)

np.set_printoptions(precision=4, suppress=True, linewidth=110)
torch.set_printoptions(precision=4, linewidth=110)


def s1_panel(n: int = 60, seed: int = 1, d: int = 4):
    """A synthetic panel from scenario S1 (PIGD runs the shared mechanism at speed 2) and its info."""
    from helpers import small_panel
    return small_panel("S1", n=n, seed=seed, d=d)


def with_gaps(panel, seed: int = 3, share: float = 0.15):
    """Extra missing values, including a whole missing visit (as in tests/test_engine.py)."""
    rng = np.random.default_rng(seed)
    y = []
    for yi in panel.y:
        yi = yi.copy()
        mask = rng.random(yi.shape) < share
        mask[0] = rng.random(yi.shape[1]) < share / 2
        mask[0, 0] = False
        yi[mask] = np.nan
        if len(yi) > 3:
            yi[2] = np.nan
        y.append(yi)
    return panel.with_y(y)

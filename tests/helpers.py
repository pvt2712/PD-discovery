"""Shared synthetic fixtures for the tests (synthetic data only)."""
from __future__ import annotations

import numpy as np

from pdx.engine.fit import start_values
from pdx.engine.model import Layout, ModelSpec
from pdx.sim import scenarios


def slow(fn):
    fn.slow = True
    return fn


def small_panel(scenario: str = "S1", n: int = 120, seed: int = 1, d: int = 4):
    rng = np.random.default_rng(seed)
    panel, info = scenarios.generate(scenario, n, rng, d=d)
    return panel, info


def spec_for(panel, info, treatment: str = "emax", era: bool = True, n_quad: int = 12) -> ModelSpec:
    truth = info["truth"]
    return ModelSpec(d=panel.d, support=info["support"], p=panel.p, treat_mask=truth.treat_mask,
                     treatment=treatment, meas_var=truth.meas_var, n_quad=n_quad,
                     noise_era_col=0 if era else None)


def random_point(panel, spec: ModelSpec, kind: str, seed: int = 0, scale: float = 0.05) -> np.ndarray:
    """A start value moved at random, so that every parameter (load, era noise, ...) is active."""
    rng = np.random.default_rng(seed)
    x = start_values(panel, spec, kind)
    return x + scale * rng.normal(size=len(x))


def layout(spec: ModelSpec, kind: str) -> Layout:
    return Layout(spec, kind)

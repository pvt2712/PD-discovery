from types import SimpleNamespace

import numpy as np

from pdx.engine.model import ModelSpec
from pdx.localise import (FreeDraws, classify_links, per_domain_tests, per_link_tests, speed_and_load_only,
                          speed_mode, speed_rule, speed_rule_batch, weighted_median)


def test_weighted_median():
    assert weighted_median([1, 2, 3], [1, 1, 1]) == 2
    assert weighted_median([1, 2, 3], [5, 1, 1]) == 1
    assert weighted_median([3, 1, 2], [1, 1, 5]) == 2


def test_speed_rule_recovers_pure_speed_and_resists_one_deviation():
    a0 = np.array([-0.25, -0.25, -0.25, -0.25, 0.10, 0.20, 0.25])
    assert abs(speed_rule(a0, 2.0 * a0) - 2.0) < 1e-12
    a1 = 2.0 * a0
    a1[5] = 0.9                                   # one link much stronger beyond speed
    assert abs(speed_rule(a0, a1) - 2.0) < 1e-12
    a1[4] = -0.2                                  # a sign change never votes on the speed
    assert abs(speed_rule(a0, a1) - 2.0) < 1e-12
    assert abs(speed_mode(a0, 2.0 * a0) - 2.0) < 1e-2


def test_batch_matches_scalar():
    rng = np.random.default_rng(0)
    a0 = rng.normal(size=(200, 7))
    a1 = 1.7 * a0 + 0.1 * rng.normal(size=(200, 7))
    b = speed_rule_batch(a0, a1, 0.05)
    s = np.array([speed_rule(a0[i], a1[i], 0.05) for i in range(200)])
    assert np.allclose(b, s, equal_nan=True)


def _setup(entries, d, noise=0.01, M=4000, seed=0, extra_noise=None):
    """entries: {(to, from): (a0, a1)}; draws a +- noise around the estimates."""
    sup = np.zeros((d, d), bool)
    for (i, j) in entries:
        sup[i, j] = True
    spec = ModelSpec(d=d, support=sup)
    order = [tuple(r) for r in spec.sup_idx]
    a0 = np.array([entries[k][0] for k in order])
    a1 = np.array([entries[k][1] for k in order])
    rng = np.random.default_rng(seed)
    sd = np.full(len(order), noise)
    for k, v in (extra_noise or {}).items():
        sd[order.index(k)] = v
    dr = FreeDraws(a0, a1, a0 + sd * rng.normal(size=(M, len(order))), a1 + sd * rng.normal(size=(M, len(order))))
    return SimpleNamespace(spec=spec), spec, dr, order


BASE = {(0, 0): (-0.25, -0.5), (1, 1): (-0.25, -0.5), (2, 2): (-0.25, -0.5), (3, 3): (-0.25, -0.5),
        (1, 0): (0.20, 0.64),                     # stronger beyond speed (speed 2): Delta = 0.12
        (2, 0): (0.00, 0.50),                     # present in PIGD only
        (2, 1): (0.15, 0.30), (3, 1): (0.15, 0.30), (3, 2): (0.15, 0.30)}   # similar


def test_link_classes():
    fit, spec, dr, order = _setup(BASE, 4)
    res = per_link_tests(fit, None, draws=dr)
    assert abs(res.s_hat - 2.0) < 0.05
    cls = {tuple(c["link"]): c["verdict"] for c in classify_links(res, eps=0.10, method="none")}
    assert cls[(1, 0)] == "different strength beyond speed"
    assert cls[(2, 0)] == "structurally different"
    assert cls[(2, 1)] == cls[(3, 1)] == cls[(3, 2)] == "similar"
    assert cls[(0, 0)] == "outside the family"
    assert not speed_and_load_only(res, 0.10)
    dom = per_domain_tests(res, spec)
    assert dom["verdict"][1] == "differs beyond speed" and dom["verdict"][2] == "differs beyond speed"


def test_wide_interval_is_unresolved():
    fit, spec, dr, order = _setup(BASE, 4, extra_noise={(3, 2): 0.3})
    cls = {tuple(c["link"]): c["verdict"] for c in classify_links(per_link_tests(fit, None, draws=dr), 0.10, "none")}
    assert cls[(3, 2)] == "unresolved"


def test_disagreeing_diagnostics_make_verdicts_not_separable():
    entries = {(0, 0): (-0.25, -0.5), (1, 1): (-0.25, -0.5), (2, 2): (-0.25, -0.5),
               (1, 0): (0.20, 0.64), (2, 1): (0.15, 0.30)}         # cross-links alone give speed 3.2
    fit, spec, dr, order = _setup(entries, 3)
    res = per_link_tests(fit, None, draws=dr)
    assert res.speed_set[1] > 3.0
    cls = {tuple(c["link"]): c["verdict"] for c in classify_links(res, 0.10, "none")}
    assert cls[(1, 0)] == "speed and mechanism not separable"


def test_unidentified_speed():
    entries = {(0, 0): (-0.04, -0.045), (1, 1): (-0.045, -0.04), (1, 0): (0.04, 0.04)}
    fit, spec, dr, order = _setup(entries, 2)
    res = per_link_tests(fit, None, draws=dr)
    assert not res.speed_identified
    assert all(c["verdict"] == "speed not identified" for c in classify_links(res, 0.10))
    assert per_domain_tests(res, spec)["verdict"] == ["speed not identified"] * 2

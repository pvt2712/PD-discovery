"""Engine checks of protocol v0.5, §16: the likelihood against an independent implementation,
exact gradients against finite differences, exact nesting of the null model, and the fast
transitions against one matrix exponential per interval."""
from __future__ import annotations

import numpy as np
import torch

from helpers import layout, random_point, small_panel, spec_for
from pdx.engine.fit import null_to
from pdx.engine.likelihood import loglik, prepare, value_and_grad
from pdx.engine.model import drifts, noise_cov
from pdx.engine.reference import reference_loglik
from pdx.engine.transitions import transitions, transitions_grouped


def _with_gaps(panel, seed=3, share=0.15):
    """Extra missing values, including a whole missing visit, to exercise the masked filter."""
    rng = np.random.default_rng(seed)
    y = []
    for yi in panel.y:
        yi = yi.copy()
        mask = rng.random(yi.shape) < share
        mask[0] = rng.random(yi.shape[1]) < share / 2          # some year-1 scores missing too
        mask[0, 0] = False                                      # never the whole landmark visit
        yi[mask] = np.nan
        if len(yi) > 3:
            yi[2] = np.nan
        y.append(yi)
    return panel.with_y(y)


def test_likelihood_matches_reference():
    panel, info = small_panel(n=40)
    panel = _with_gaps(panel)
    for treatment in ("linear", "emax"):
        spec = spec_for(panel, info, treatment=treatment)
        prep = prepare(panel, spec.meas_var, spec.n_groups)
        for kind in ("null", "free", "joint"):
            lay = layout(spec, kind)
            x = random_point(panel, spec, kind, seed=7)
            fast = float(loglik(torch.as_tensor(x), lay, prep))
            ref = reference_loglik(x, lay, panel, prep.mu1.numpy(), prep.S1.numpy(), prep.y1c.numpy())
            assert abs(fast - ref) < 1e-8 * max(1.0, abs(ref)), (treatment, kind, fast, ref)


def test_gradient_matches_finite_differences():
    panel, info = small_panel(n=30)
    spec = spec_for(panel, info, treatment="emax")
    prep = prepare(panel, spec.meas_var, spec.n_groups)
    rng = np.random.default_rng(0)
    for kind in ("null", "free"):
        lay = layout(spec, kind)
        x = random_point(panel, spec, kind, seed=11)
        f0, g = value_and_grad(x, lay, prep)
        for j in rng.choice(len(x), size=12, replace=False):
            e = np.zeros(len(x))
            e[j] = 1e-6
            fd = (value_and_grad(x + e, lay, prep)[0] - value_and_grad(x - e, lay, prep)[0]) / 2e-6
            assert abs(fd - g[j]) < 1e-4 * max(1.0, abs(g[j])), (kind, j, fd, g[j])


def test_null_is_nested_in_free_and_joint():
    panel, info = small_panel(n=60)
    spec = spec_for(panel, info)
    prep = prepare(panel, spec.meas_var, spec.n_groups)
    x0 = random_point(panel, spec, "null", seed=5)
    x0[layout(spec, "null").slices["gamma"][0]] = np.log(1.8)
    l0 = float(loglik(torch.as_tensor(x0), layout(spec, "null"), prep))
    for kind in ("free", "joint"):
        x1 = null_to(x0, spec, kind)
        l1 = float(loglik(torch.as_tensor(x1), layout(spec, kind), prep))
        assert abs(l1 - l0) < 1e-8 * abs(l0), (kind, l0, l1)


def test_grouped_transitions_match_matrix_exponential():
    panel, info = small_panel(n=20)
    spec = spec_for(panel, info)
    lay = layout(spec, "free")
    x = torch.as_tensor(random_point(panel, spec, "free", seed=2))
    A, Q = drifts(x, lay), noise_cov(x, lay)
    g = torch.as_tensor(np.arange(20) % 2)
    c = torch.as_tensor(np.random.default_rng(0).uniform(0.0, 10.0, size=(20, 3, 4)))
    E, M, S = transitions_grouped(A, Q, g, c)
    E2, M2, S2 = transitions(A[g], Q[g], c)          # Van Loan's block is accurate for moderate c
    for a, b in ((E, E2), (M, M2), (S, S2)):
        assert float((a - b).abs().max()) < 1e-10
    from scipy import linalg
    c = torch.as_tensor(np.random.default_rng(1).uniform(0.0, 60.0, size=(20, 3, 4)))
    E, _, S = transitions_grouped(A, Q, g, c)
    for i in range(20):                               # long intervals: the stationary-covariance identity
        k = int(g[i])
        Sig = linalg.solve_continuous_lyapunov(A[k].detach().numpy(), -Q[k].detach().numpy())
        for a in range(3):
            for b in range(4):
                Ei = E[i, a, b].detach().numpy()
                assert np.abs(S[i, a, b].detach().numpy() - (Sig - Ei @ Sig @ Ei.T)).max() < 1e-12


def test_ten_domain_likelihood_matches_reference():
    """Ten domains exercise the blocked matrix products (engine/linalg.py)."""
    panel, info = small_panel(n=40, d=10, seed=4)
    panel = panel.subset(np.arange(min(panel.n, 14)))
    spec = spec_for(panel, info, treatment="emax")
    prep = prepare(panel, spec.meas_var, spec.n_groups)
    for kind in ("null", "free"):
        lay = layout(spec, kind)
        x = random_point(panel, spec, kind, seed=3)
        fast = float(loglik(torch.as_tensor(x), lay, prep, chunk=5))
        ref = reference_loglik(x, lay, panel, prep.mu1.numpy(), prep.S1.numpy(), prep.y1c.numpy())
        assert abs(fast - ref) < 1e-8 * max(1.0, abs(ref)), (kind, fast, ref)


def test_transitions_for_marginal_drift_and_long_intervals():
    """S by composition needs no stability of A (eigenvalues summing to zero), stays accurate
    over long intervals, and costs grow only with log c."""
    import time
    from pdx.engine.transitions import transition_np
    A = torch.tensor([[[-0.1, 0.1], [0.1, -0.1]]], dtype=torch.float64)      # eigenvalues 0 and -0.2
    Q = torch.tensor([[[0.10, 0.02], [0.02, 0.08]]], dtype=torch.float64)
    c = torch.tensor([[[0.3, 2.0, 7.5]]], dtype=torch.float64)
    E, M, S = transitions_grouped(A, Q, torch.zeros(1, dtype=torch.long), c)
    E2, M2, S2 = transitions(A, Q, c)
    for a, b in ((E, E2), (M, M2), (S, S2)):
        assert float((a - b).abs().max()) < 1e-10
    big = torch.tensor([[[5e3]]], dtype=torch.float64)
    t0 = time.time()
    _, _, Sb = transitions_grouped(A, Q, torch.zeros(1, dtype=torch.long), big)
    assert time.time() - t0 < 1.0
    _, _, Snp = transition_np(A[0].numpy(), Q[0].numpy(), 5e3, 1.0)
    assert np.allclose(Sb[0, 0, 0].numpy(), Snp, rtol=1e-8, atol=1e-8)


def test_wild_speed_parameters_stay_cheap():
    import time
    panel, info = small_panel(n=30)
    spec = spec_for(panel, info)
    prep = prepare(panel, spec.meas_var, spec.n_groups)
    lay = layout(spec, "null")
    x = random_point(panel, spec, "null", seed=1)
    x[lay.slices["log_sigma"][0]] = 3.0                   # sigma = 20 at a line-search trial point
    t0 = time.time()
    val, g = value_and_grad(x, lay, prep)
    assert time.time() - t0 < 10 and np.isfinite(val)


def test_both_noise_routes_agree():
    """The stationary-covariance route (clearly stable A) and the composition route agree."""
    panel, info = small_panel(n=20)
    spec = spec_for(panel, info)
    lay = layout(spec, "free")
    x = torch.as_tensor(random_point(panel, spec, "free", seed=9))
    A, Q = drifts(x, lay), noise_cov(x, lay)
    g = torch.as_tensor(np.arange(20) % 2)
    c = torch.as_tensor(np.random.default_rng(3).uniform(0.0, 80.0, size=(20, 3, 4)))
    E1, M1, S1 = transitions_grouped(A, Q, g, c, method="lyapunov")
    E2, M2, S2 = transitions_grouped(A, Q, g, c, method="composition")
    for a, b in ((E1, E2), (M1, M2), (S1, S2)):
        assert float((a - b).abs().max()) < 1e-11

"""Recovery of a known speed and a non-negative likelihood ratio (protocol §16 unit tests)."""
from helpers import small_panel, spec_for
from pdx.engine.fit import fit_pair
from pdx.speed import sp1, sp3


def test_recovery_and_nonnegative_lr():
    panel, info = small_panel("S1", n=400, seed=11)
    spec = spec_for(panel, info)
    f0, f1 = fit_pair(panel, spec, maxiter=1500)
    assert f1.loglik >= f0.loglik - 1e-6
    s = sp1(f0, panel)
    assert 1.4 < s < 2.8, s
    assert sp3(f1)["s_ls"] > 1.0

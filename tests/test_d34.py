"""Decision D34 (7 November 2026): the PPMI run's capped bootstrap and the model-based intervals of
the speed estimates. Synthetic data only."""
import sys
from pathlib import Path

import numpy as np
from helpers import small_panel, spec_for

from pdx import config, speed
from pdx.engine.fit import covariance_checked, fit_pair, hessian, param_names

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def test_the_configuration_holds_the_d34_sizes():
    cfg = config.load()
    assert cfg["wp4"]["n_max"] == 39 and cfg["wp4"]["speed_bootstrap"] == 0 and cfg["wp4"]["model_draws"] > 0
    assert cfg["hp"]["n_max"] == 999          # the registered design is unchanged; D34 sizes the PPMI run


def test_the_p_value_at_the_cap_comes_from_the_first_replicates_in_order():
    sys.path.insert(0, str(SCRIPTS))
    import run_hp_real
    obs, h = 5.0, 3
    below, above = [1.0, True], [9.0, True]
    # 39 replicates, one at or above the observed: rejects at 5% (p = 2/40)
    r = run_hp_real.p_at_cap([below] * 20 + [above] + [below] * 18, obs, 39, h)
    assert r["p"] == 2 / 40 and r["n_drawn"] == 39 and not r["stopped_early"]
    # a larger run drawn later does not change the result at the cap
    later = [below] * 20 + [above] + [below] * 18 + [above, above, below]
    assert run_hp_real.p_at_cap(later, obs, 39, h) == r
    # h exceedances before the cap stop the sequence there: p = h / drawn
    early = [above, below, above, [None, False]]          # a failed replicate counts as at or above
    s = run_hp_real.p_at_cap(early, obs, 39, h)
    assert s["stopped_early"] and s["p"] == 3 / 4 and s["n_drawn"] == 4
    # still open: fewer than the cap and fewer than h exceedances
    assert run_hp_real.p_at_cap([below] * 10 + [above], obs, 39, h) is None


def test_model_based_intervals_have_the_bootstrap_layout_and_cover_the_estimates():
    panel, info = small_panel("S1", n=80, seed=11)
    spec = spec_for(panel, info)
    f0, f1 = fit_pair(panel, spec, maxiter=60)
    cov0 = covariance_checked(hessian(f0, panel), names=param_names(f0.layout))[0]
    cov1 = covariance_checked(hessian(f1, panel), names=param_names(f1.layout))[0]
    a = speed.model_based(f0, f1, panel, cov0, cov1, 0.95, 300, np.random.default_rng(1))
    b = speed.model_based(f0, f1, panel, cov0, cov1, 0.95, 300, np.random.default_rng(1))
    assert a["sp1_ci"] == b["sp1_ci"] and a["sp3_lower_95"] == b["sp3_lower_95"]       # reproducible
    for key in ("sp1_ci", "sigma_ci", "ratio_90_10_ci", "beta_ci", "theta_ci", "share_explained_ci",
                "speed_load_corr_ci", "sp3_bias_corrected", "sp3_lower_95", "sp3_reportable"):
        assert key in a
    est = a["estimates"]
    assert a["B"] == 0 and a["draws"] == 300 and "model-based" in a["method"]
    assert a["sp1_ci"][0] < est["sp1"] < a["sp1_ci"][1]
    for (lo, hi), s in zip(a["sigma_ci"], est["sp2"]["sigma"]):
        assert lo < s < hi
    # the estimates are those of the fits, with the landmark states computed once or every time
    assert est == speed.speed_estimates(f0, f1, panel)
    # without uncertainty the intervals shrink to the estimates
    z = speed.model_based(f0, f1, panel, 0 * cov0, 0 * cov1, 0.95, 50, np.random.default_rng(2))
    assert np.allclose(z["sp1_ci"], est["sp1"]) and np.allclose(np.ravel(z["sigma_ci"]), np.repeat(est["sp2"]["sigma"], 2))

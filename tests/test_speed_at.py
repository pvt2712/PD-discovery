"""Paper 2's speed at the decision (escalation-study-v2/PROTOCOL.md §8, Appendix A task 8): the filtered
log-speed posterior, its truncation, the frozen-transform panel, the time scale and the guarded script,
on synthetic data and the mock PPMI download only."""
from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

import ppmi_mock
from pdx import ppmi, speed_at, split
from pdx.engine.fit import Fit, start_values
from pdx.engine.likelihood import prepare
from pdx.engine.model import Layout, ModelSpec, unpack_numpy
from pdx.engine.simulate import simulate_from_fit
from pdx.sim import scenarios

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
OUT = list(speed_at.OUTPUT)


def _known_model(n: int = 300, seed: int = 4, sigma: float = 0.6, tau: float = 0.1, visits=(4, 7)):
    """A speed-only model with known parameters (phenotype speed, covariates, landmark scores and
    treatment all active), a panel simulated from it, and each participant's true log-speed."""
    panel, info = scenarios.generate("S1", n, np.random.default_rng(seed), d=4, visits=visits)
    truth = info["truth"]
    spec = ModelSpec(d=panel.d, support=info["support"], p=panel.p, treat_mask=truth.treat_mask, treatment="emax",
                     meas_var=truth.meas_var, n_quad=20, noise_era_col=0, load_on_speed=False)
    lay = Layout(spec, "null")
    x = start_values(panel, spec, "null")
    lay.put(x, "log_sigma", np.full(2, np.log(sigma)))
    lay.put(x, "gamma", [np.log(2.0)])
    lay.put(x, "beta", [0.1, 0.2, 0.0, -0.1])
    lay.put(x, "theta", [0.1, 0.0, 0.15, 0.0])
    lay.put(x, "log_tau", np.full((2, panel.d), np.log(tau)))
    lay.put(x, "kappa", np.full((2, 2), 0.3))
    lay.put(x, "kappa_ldr", np.full((2, 2), 0.1))
    fit = Fit("null", spec, lay, x, 0.0, True, "known parameters", 0, unpack_numpy(x, lay))
    speeds = []
    sim = simulate_from_fit(fit, panel, np.random.default_rng(seed + 1), speeds=speeds)
    return fit, spec, replace(sim, ids=np.arange(sim.n) + 3000), np.array(speeds)


def _every_visit(panel) -> pd.DataFrame:
    """A cutoff just after every visit of every participant (the first is the landmark)."""
    return pd.DataFrame([(panel.ids[i], t + 0.01) for i in range(panel.n) for t in panel.t[i]], columns=["PATNO", "tau"])


def test_per_node_likelihoods_are_the_engines_before_weighting():
    import torch
    from scipy.special import logsumexp

    from pdx.engine.likelihood import participant_loglik
    fit, spec, panel, _ = _known_model(n=80)
    prep = prepare(panel, spec.meas_var, spec.n_groups)
    ref = speed_at.landmark_reference(panel, spec.meas_var, spec.n_groups)
    again = prepare(panel, spec.meas_var, spec.n_groups, landmark=ref.as_tuple())     # the reference is prepare's own
    for name in ("mu1", "S1", "y1c"):
        assert torch.equal(getattr(prep, name), getattr(again, name)), name
    x, lay, idx = torch.as_tensor(fit.x), Layout(spec, "null"), torch.arange(panel.n)
    ll, U, logw = speed_at.node_logliks(fit, panel)
    assert ll.shape == (panel.n, spec.n_quad)
    assert np.allclose(logsumexp(ll + logw[None, :], 1), participant_loglik(x, lay, prep, idx).numpy(),
                       rtol=0.0, atol=1e-10)
    own = participant_loglik(x, lay, prep, idx, per_node=True, nodes=torch.as_tensor(np.tile(U, (panel.n, 1))))
    assert np.allclose(own.numpy(), ll, rtol=0.0, atol=1e-10)                 # own nodes = the shared ones here
    one = speed_at.truncated_panel(panel, np.arange(panel.n), np.ones(panel.n, int))
    assert [len(t) for t in one.t] == [1] * panel.n and list(one.ids) == list(panel.ids)
    assert not speed_at.node_logliks(fit, one, reference=ref)[0].any()       # the landmark alone: 0 at every node


def test_the_posterior_tracks_the_true_speed_and_narrows_with_visits():
    fit, spec, panel, true = _known_model()
    res = speed_at.filtered_log_speed(fit, spec, panel, _every_visit(panel))
    i = res.PATNO.to_numpy() - 3000
    T = np.array([len(t) for t in panel.t])[i]
    k = np.concatenate([np.arange(1, len(t) + 1) for t in panel.t])
    assert (res.n_visits_used.to_numpy() == k).all()                  # every visit before the cutoff counts
    last = res[res.n_visits_used.to_numpy() == T]
    li = last.PATNO.to_numpy() - 3000
    r_post, r_prior = (np.corrcoef(last[c], true[li])[0, 1] for c in ("log_speed_mean", "prior_mean"))
    assert r_post > 0.75 and r_post > r_prior + 0.1, (r_post, r_prior)
    # the SD narrows, weakly, as the cutoff takes in more visits (on average: one participant's can widen) ...
    sd = res[T >= 4].groupby("n_visits_used").log_speed_sd.mean().loc[1:4].to_numpy()
    assert np.all(np.diff(sd) <= 1e-12) and sd[-1] < 0.85 * sd[0], sd
    # ... and with all visits it is below the prior SD for (nearly) everyone
    assert np.mean(last.log_speed_sd <= last.prior_sd + 1e-12) >= 0.98


def test_the_landmark_alone_gives_the_prior_and_earlier_cutoffs_are_dropped():
    fit, spec, panel, _ = _known_model(n=120)
    t1, t2 = np.array([t[0] for t in panel.t]), np.array([t[1] for t in panel.t])
    cut = pd.concat([pd.DataFrame({"PATNO": panel.ids, "tau": t1 - 0.5, "kind": "before"}),
                     pd.DataFrame({"PATNO": panel.ids, "tau": t1, "kind": "at"}),
                     pd.DataFrame({"PATNO": panel.ids, "tau": 0.5 * (t1 + t2), "kind": "between"}),
                     pd.DataFrame({"PATNO": [99999], "tau": [2.0], "kind": "stranger"}),
                     pd.DataFrame({"PATNO": [panel.ids[0]], "tau": [np.nan], "kind": "no time"})])
    res = speed_at.filtered_log_speed(fit, spec, panel, cut)
    c = res.attrs["counts"]
    assert (c["before_landmark"], c["not_in_panel"], c["no_time"]) == (panel.n, 1, 1)
    assert c["scored"] == c["landmark_only"] == 2 * panel.n
    assert set(res.kind) == {"at", "between"} and (res.n_visits_used == 1).all()     # extra columns carried
    # the landmark scores add no likelihood term: the posterior is the prior, exactly
    assert np.allclose(res.log_speed_mean, res.prior_mean, rtol=0.0, atol=1e-12)
    assert np.allclose(res.log_speed_sd, res.prior_sd, rtol=1e-12, atol=0.0)
    # the prior is gamma_g + beta' z + theta' y1c with y1c the OBSERVED, centred landmark scores, SD sigma_g
    par = fit.params
    y1c = prepare(panel, spec.meas_var, spec.n_groups).y1c.numpy()
    at = res[res.kind == "at"]
    j = at.PATNO.to_numpy() - 3000
    eta = par["log_speed"][panel.group] + panel.z @ par["beta"] + y1c @ par["theta"]
    assert np.allclose(at.prior_mean, eta[j], rtol=0.0, atol=1e-12)
    assert np.allclose(at.prior_sd, par["sigma"][panel.group[j]], rtol=1e-12, atol=0.0)


def test_later_visits_never_move_the_speed_at_a_decision():
    fit, spec, panel, _ = _known_model(n=150)
    rows = [i for i in range(panel.n) if len(panel.t[i]) >= 4]
    cut = pd.DataFrame({"PATNO": panel.ids[rows], "tau": [panel.t[i][1] + 0.3 for i in rows]})     # two visits used
    every = pd.DataFrame({"PATNO": panel.ids[rows], "tau": 100.0})
    # after the cutoff: other scores, doses, exposures, states and dates, and the last visit gone
    later = lambda L, f: [np.concatenate([v[:2], f(v[2:-1])]) if i in rows else v for i, v in enumerate(L)]
    changed = replace(panel, y=later(panel.y, lambda v: v + 3.0), t=later(panel.t, lambda v: v + 0.2),
                      dose=later(panel.dose, lambda v: v + 2.0), exposure=later(panel.exposure, lambda v: v + 1.0),
                      on=later(panel.on, lambda v: 1.0 - v), year=later(panel.year, lambda v: v))
    for adaptive in (True, False):
        a = speed_at.filtered_log_speed(fit, spec, panel, cut, adaptive=adaptive)
        b = speed_at.filtered_log_speed(fit, spec, changed, cut, adaptive=adaptive)
        assert (a.n_visits_used == 2).all() and np.allclose(a[OUT], b[OUT], rtol=0.0, atol=1e-12)
    a, b = (speed_at.filtered_log_speed(fit, spec, p, every) for p in (panel, changed))
    assert np.abs(a.log_speed_mean - b.log_speed_mean).max() > 0.05        # while the later visits do count later


def test_the_adaptive_rule_matches_a_dense_grid_where_the_prior_rule_drifts():
    """With visits that pin the speed down far more tightly than the prior, 20 nodes placed on the prior
    misplace the posterior; re-centred on it, they agree with a dense grid through the same filter."""
    fit, spec, panel, _ = _known_model(n=200, sigma=1.5, tau=0.01, visits=(6, 10))
    cut = pd.DataFrame({"PATNO": panel.ids, "tau": [t[-1] + 0.01 for t in panel.t]})        # after the last visit: all used
    ada = speed_at.filtered_log_speed(fit, spec, panel, cut)
    plain = speed_at.filtered_log_speed(fit, spec, panel, cut, adaptive=False)
    grid = np.linspace(-8.0, 8.0, 641)                                       # trapezoid rule on u ~ N(0, 1)
    lay = Layout(replace(spec, n_quad=len(grid)), "null")
    prep = prepare(panel, spec.meas_var, spec.n_groups)
    ll = speed_at._per_node(fit.x, lay, prep, 32, nodes=np.broadcast_to(grid, (panel.n, len(grid))))
    m, s, _ = speed_at._moments(ll - 0.5 * grid[None, :] ** 2, np.broadcast_to(grid, ll.shape))
    eta, sig = speed_at._prior(fit.x, lay, prep)
    err = lambda r: max(np.abs(r.log_speed_mean - (eta + sig * m)).max(), np.abs(r.log_speed_sd - sig * s).max())
    assert err(ada) < 0.02 and err(plain) > 0.05, (err(ada), err(plain))
    assert ada.attrs["counts"]["not_converged"] == 0


def _mock_tables():
    tmp = Path(tempfile.mkdtemp())
    ppmi_mock.make(tmp / "raw")
    return tmp, ppmi.build(tmp / "raw")


def test_the_frozen_transform_panel_is_analysis_panels_given_its_own_tables():
    tmp, out = _mock_tables()
    part, grid = out["participants"], out["grid"]
    for motor_on in (False, True):
        panel, ns, _ = ppmi.analysis_panel(part, grid, motor_on=motor_on)
        # saved and loaded as scripts/build_analysis_panel.py saves the tables
        np.savez_compressed(tmp / "ns.npz", **{f"values_{j}": u for j, (u, _) in enumerate(ns.tables)},
                            **{f"scores_{j}": sc for j, (_, sc) in enumerate(ns.tables)})
        frozen = speed_at.panel_with_frozen_transform(part, grid, ppmi.CORE_DOMAINS,
                                                      speed_at.load_normal_scores(tmp / "ns.npz"), motor_on)
        assert list(frozen.ids) == list(panel.ids) and frozen.domains == panel.domains
        assert np.array_equal(frozen.group, panel.group) and np.array_equal(frozen.z, panel.z)
        for name in ("y", "t", "dose", "exposure", "on"):
            assert all(np.array_equal(u, v, equal_nan=True) for u, v in zip(getattr(frozen, name), getattr(panel, name)))
        assert all(np.array_equal(u, v) for u, v in zip(frozen.year, panel.year))
        assert speed_at.same_panel(frozen, panel, atol=0.0)
        # centred on given means instead (the development panel's, for held-out participants): here the same
        zm = speed_at.baseline_covariates(part, panel.ids).mean()
        again = speed_at.panel_with_frozen_transform(part, grid, ppmi.CORE_DOMAINS, ns, motor_on, z_mean=zm)
        assert np.allclose(again.z, panel.z, rtol=0.0, atol=1e-12)


def test_decision_dates_land_on_the_panel_time_scale():
    _, out = _mock_tables()
    part, grid = out["participants"], out["grid"]
    visits = grid[["PATNO", "date"]].rename(columns={"date": "date_t"})
    for g in (None, grid):                                   # the anchor from the enrolment date, or the grid's
        assert np.array_equal(speed_at.to_panel_time(visits, part, g).to_numpy(), grid.date_years.to_numpy())
    panel, _, _ = ppmi.analysis_panel(part, grid)            # the panel's visit times, back from their dates
    for i, p in enumerate(panel.ids):
        rows = grid[(grid.PATNO == p) & (grid.year >= 1)].sort_values("year")
        tau = speed_at.to_panel_time(pd.DataFrame({"PATNO": p, "date_t": rows.date.to_numpy()}), part)
        assert np.array_equal(tau.to_numpy(), panel.t[i])
    # and a month on the 15th, as Paper 2 dates decisions, returns from its panel time
    months = pd.DataFrame({"PATNO": 4, "date_t": pd.date_range("2015-01-15", periods=60, freq=pd.DateOffset(months=1))})
    tau = speed_at.to_panel_time(months, part)
    back = speed_at.panel_anchor(part, grid)[split.key(4)] + pd.to_timedelta(np.round(tau.to_numpy() * 365.25), unit="D")
    assert (back == months.date_t.to_numpy()).all()
    assert speed_at.to_panel_time(pd.DataFrame({"PATNO": [999], "date_t": [pd.Timestamp("2016-01-15")]}), part).isna().all()
    # without an enrolment date, ppmi.build anchors on the baseline visit; the grid keeps that anchor
    blank = part.assign(enroll_date=part.enroll_date.where(part.PATNO != 4))
    four = visits[visits.PATNO == 4]
    assert speed_at.to_panel_time(four, blank).isna().all()
    assert np.array_equal(speed_at.to_panel_time(four, blank, grid).to_numpy(), grid.date_years[grid.PATNO == 4].to_numpy())


# ---------------------------------------------------------------- the script, end to end on the mock

def _mock_run():
    """The mock tables, build_analysis_panel.py's development panel, a speed-only fit with every speed
    term active written as the H-P run's checkpoint, and a decisions file, in a temporary data folder."""
    sys.path.insert(0, str(SCRIPTS))
    import _common
    import build_analysis_panel
    import run_hp_real
    from pdx import config
    from pdx.data import load_panel
    from pdx.engine.fit import null_to
    tmp = Path(tempfile.mkdtemp())
    ppmi_mock.make(tmp / "raw")
    out = ppmi.build(tmp / "raw")
    der = tmp / "data" / "derived"
    der.mkdir(parents=True)
    out["participants"].to_parquet(der / "participants.parquet")
    out["grid"].to_parquet(der / "grid.parquet")
    old, argv = _common.DATA, sys.argv
    _common.DATA = tmp / "data"
    try:
        sys.argv = ["build_analysis_panel.py", "--approval", "synthetic mock data, unit test"]
        build_analysis_panel.main()
    finally:
        _common.DATA, sys.argv = old, argv
    cfg = config.load()
    pspec = json.loads((der / "panel_spec_development.json").read_text())
    dev = load_panel(der / "panel_tdpigd_development.npz")
    names = pspec["domains"]
    expert, _ = run_hp_real.expert_support(str(_common.ROOT / "configs" / "expert_map_provisional.json"), names)
    spec = run_hp_real.build_spec(dev, pspec, names, expert, cfg)
    lay = Layout(spec, "null")
    x = start_values(dev, spec, "null")
    lay.put(x, "log_sigma", np.full(2, np.log(0.5)))
    lay.put(x, "beta", [0.2, 0.1, -0.1, 0.1])
    lay.put(x, "theta", np.full(len(names), 0.05))
    rec = lambda kind, v: {"kind": kind, "x": [float(a) for a in v], "loglik": -1.0, "success": True,
                           "message": "synthetic", "n_eval": 0, "n_iter": 0, "failed": False, "starts": [],
                           "start_logliks": []}
    state = {"format": 2,
             "settings": {"participant_set": "development",
                          "panel_sha256": _common.file_sha256(der / "panel_tdpigd_development.npz"),
                          "panel_spec_sha256": _common.file_sha256(der / "panel_spec_development.json"),
                          "engine": cfg["engine"], "treatment": cfg["treatment"]},
             "observed": {"support": spec.support.astype(int).tolist(), "null": rec("null", x),
                          "free": rec("free", null_to(x, spec, "free"))}}
    ckpt = tmp / "results" / "wp4" / "hp_tdpigd_development_checkpoint.json"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_text(json.dumps(state))
    dates = ["2015-07-15", "2016-01-15", "2016-06-15", "2017-01-15", "2018-06-15", "2022-01-15", "2023-03-15"]
    pd.DataFrame([(p, d) for p in range(1, 14) for d in dates], columns=["PATNO", "date_t"]).to_csv(
        tmp / "data" / "decisions.csv", index=False)
    return tmp, der, out, ckpt, spec, x


def _run_script(tmp, *args):
    sys.path.insert(0, str(SCRIPTS))
    import _common
    import speed_at_decisions
    old = (_common.DATA, _common.RESULTS, sys.argv)
    _common.DATA, _common.RESULTS = tmp / "data", tmp / "results"
    try:
        sys.argv = ["speed_at_decisions.py", *args]
        speed_at_decisions.main()
    finally:
        _common.DATA, _common.RESULTS, sys.argv = old


def test_the_script_scores_held_out_participants_with_the_development_fit_and_scales():
    tmp, der, out, ckpt, spec, x = _mock_run()
    dec = str(tmp / "data" / "decisions.csv")
    manifest = tmp / "manifest.json"                         # stands in for Paper 2's freeze manifest
    manifest.write_text("{}")
    for which in ("heldout", "development"):
        _run_script(tmp, "--approval", "synthetic mock data, unit test", "--set", which, "--decisions", dec,
                    "--freeze-manifest", str(manifest))
    held = pd.read_csv(der / "speed_at_decisions_heldout.csv", parse_dates=["date_t"])
    devr = pd.read_csv(der / "speed_at_decisions_development.csv", parse_dates=["date_t"])
    assert list(held.columns) == ["PATNO", "date_t", "tau", *OUT]
    assert set(held.PATNO) == {2, 5} and not set(devr.PATNO) & {2, 5}        # the mock's held-out participants
    assert set(devr.PATNO) <= {1, 3, 4, 6, 7, 8, 12}                         # genetic, treated, no baseline: out
    assert not (held.date_t == "2015-07-15").any() and not ((devr.PATNO == 3) & (devr.date_t == "2016-01-15")).any()
    lm = pd.concat([held, devr]).query("n_visits_used == 1")                   # participant 3's landmark: March 2016
    assert len(lm) and np.allclose(lm.log_speed_mean, lm.prior_mean, atol=1e-12)
    p2 = held[held.PATNO == 2].set_index("date_t")
    day = pd.Timestamp
    # participant 2's second visit is on 15 January 2017: a decision that day does not use it (its dose may carry
    # the decision), a decision in June 2018 does
    assert p2.loc[day("2016-01-15"), "n_visits_used"] == 1 and p2.loc[day("2017-01-15"), "n_visits_used"] == 1
    assert p2.loc[day("2018-06-15"), "n_visits_used"] == 2
    assert np.allclose(p2.loc[day("2016-01-15"), OUT].to_numpy(float), p2.loc[day("2017-01-15"), OUT].to_numpy(float))
    assert (held[held.PATNO == 5].n_visits_used == 1).all()                    # died after year 1
    # held-out participants sit on the development panel's scales: its normal-scores tables, covariate
    # means and landmark centring (not on their own)
    from pdx.data import load_panel
    dev = load_panel(der / "panel_tdpigd_development.npz")
    ns = speed_at.load_normal_scores(der / "normal_scores_development.npz")
    names = json.loads((der / "panel_spec_development.json").read_text())["domains"]
    part, grid = out["participants"], out["grid"]
    dpart, hpart = (part[split.in_sets(part.PATNO, s)] for s in ("development", "heldout"))
    zm = speed_at.baseline_covariates(dpart, dev.ids).mean()
    hp = speed_at.panel_with_frozen_transform(hpart, grid[grid.PATNO.isin(set(hpart.PATNO))], names, ns, z_mean=zm)
    raw = grid[(grid.PATNO == 2) & (grid.year >= 1)].sort_values("year")[names].to_numpy(float)
    assert np.array_equal(hp.y[list(hp.ids).index(2)], ns.transform(raw), equal_nan=True)
    ref = speed_at.landmark_reference(dev, spec.meas_var, spec.n_groups)
    par = unpack_numpy(x, Layout(spec, "null"))
    y1 = np.array([v[0] for v in hp.y])
    y1c = np.where(np.isnan(y1), ref.mu1[hp.group], y1) - ref.centre
    zc = (speed_at.baseline_covariates(hpart, hp.ids).fillna(zm) - zm).to_numpy()
    eta = par["log_speed"][hp.group] + zc @ par["beta"] + y1c @ par["theta"]
    got = held.drop_duplicates("PATNO").set_index("PATNO").prior_mean
    assert np.allclose(got.loc[list(hp.ids)].to_numpy(), eta, atol=1e-12)


def test_the_script_refuses_a_checkpoint_from_another_panel_and_runs_only_with_approval():
    tmp, der, _, ckpt, _, _ = _mock_run()
    dec = str(tmp / "data" / "decisions.csv")
    manifest = tmp / "manifest.json"
    manifest.write_text("{}")
    fm = ["--freeze-manifest", str(manifest)]
    for args, message in ((["--set", "heldout", "--decisions", dec], "approval"),
                          (["--approval", "synthetic mock data, unit test", "--set", "heldout", "--decisions", dec,
                            "--freeze-manifest", str(tmp / "no_manifest.json")], "freeze manifest is missing"),
                          (["--approval", "synthetic mock data, unit test", "--set", "heldout",
                            "--decisions", str(tmp / "decisions.csv"), *fm], "must be inside")):
        try:
            _run_script(tmp, *args)
            raise AssertionError("the script ran")
        except SystemExit as e:
            assert message in str(e), str(e)
    state = json.loads(ckpt.read_text())
    state["settings"]["panel_sha256"] = "0" * 64
    ckpt.write_text(json.dumps(state))
    try:
        _run_script(tmp, "--approval", "synthetic mock data, unit test", "--set", "heldout", "--decisions", dec, *fm)
        raise AssertionError("a checkpoint fitted on another panel was accepted")
    except SystemExit as e:
        assert "another panel_tdpigd_development.npz" in str(e), str(e)
    assert not (der / "speed_at_decisions_heldout.csv").exists()


def test_a_visit_on_the_decision_date_is_left_out_except_the_landmark():
    """A later visit dated on the decision's date may carry the decision in its dose term, so it is left out; the
    landmark on that date stays (it adds no dose term)."""
    fit, spec, panel, _ = _known_model(n=60)
    rows = [i for i in range(panel.n) if len(panel.t[i]) >= 3]
    on_third = pd.DataFrame({"PATNO": panel.ids[rows], "tau": [panel.t[i][2] for i in rows]})
    on_landmark = pd.DataFrame({"PATNO": panel.ids[rows], "tau": [panel.t[i][0] for i in rows]})
    assert (speed_at.filtered_log_speed(fit, spec, panel, on_third).n_visits_used == 2).all()
    assert (speed_at.filtered_log_speed(fit, spec, panel, on_landmark).n_visits_used == 1).all()

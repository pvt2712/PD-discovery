"""Checks for the revision of 5 November 2026, after the PC run's bootstrap failed (synthetic and
mock data only): single-visit participants in small batches, failed fits, the bootstrap guard,
the alternating starts, warm starts across supports, lam = 0, the examination state of D24,
the landmark-prior check, the medication log (dose in force, exposure, levodopa names, treated
examinations) and the run script's checkpoint rules."""
import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from helpers import random_point, slow, small_panel, spec_for

from pdx import hp
from pdx.engine.fit import fit_model, fit_pair, free_to_null, remap_support
from pdx.engine.likelihood import chunks, loglik, participant_loglik, prepare, value_and_grad
from pdx.engine.model import Layout, drifts, treat_effect
from pdx.engine.reference import reference_loglik

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _with_single_visits(panel, k: int):
    """The panel with its first k participants cut back to the landmark visit only."""
    cut = lambda L: [v[:1] if i < k else v for i, v in enumerate(L)]
    return replace(panel, y=cut(panel.y), t=cut(panel.t), dose=cut(panel.dose), exposure=cut(panel.exposure),
                   on=cut(panel.on), year=None if panel.year is None else cut(panel.year))


def test_single_visit_participants_add_nothing_and_any_batch_size_works():
    panel, info = small_panel("S1", n=120, seed=3)
    panel = _with_single_visits(panel, 40)                # more than a batch of 32
    spec = spec_for(panel, info)
    prep = prepare(panel, spec.meas_var, spec.n_groups)
    lay = Layout(spec, "null")
    x = random_point(panel, spec, "null")
    v32, g32 = value_and_grad(x, lay, prep, 32)
    v128, g128 = value_and_grad(x, lay, prep, 128)
    assert v32 < 1e10 and np.isfinite(v32) and np.abs(g32).max() > 0
    assert abs(v32 - v128) < 1e-8 * max(1.0, abs(v128)) and np.allclose(g32, g128, atol=1e-8)
    single = torch.as_tensor(np.arange(40))
    assert torch.allclose(participant_loglik(torch.as_tensor(x), lay, prep, single), torch.zeros(40, dtype=torch.float64))
    assert sum(len(c) for c in chunks(prep, 32)) == panel.n - 40
    assert abs(float(loglik(torch.as_tensor(x), lay, prep, 32)) + v32) < 1e-8 * max(1.0, abs(v32))


def test_a_start_that_cannot_be_evaluated_is_a_failed_fit_not_a_converged_one():
    panel, info = small_panel("S1", n=60, seed=2)
    spec = spec_for(panel, info)
    bad = np.full(Layout(spec, "null").size, np.nan)
    f = fit_model(panel, spec, "null", x0=bad, maxiter=5)
    assert f.failed and not f.success and f.loglik == -np.inf and f.message.startswith("failed")
    assert f.starts[0]["stopped_by"] == "not evaluable at the start"
    good = fit_model(panel, spec, "null", maxiter=5, extra_starts=[bad])    # one bad start among two
    assert not good.failed and np.isfinite(good.loglik) and good.start_logliks[1] == -np.inf
    assert np.isnan(hp.lr_stat(f, good)) and np.isnan(hp.lr_stat(good, f))


def test_the_guard_names_a_broken_bootstrap():
    zero = (0.0, True, True, None, {"null_eval": 900, "free_eval": 1200, "null_starts": 1, "free_starts": 2})
    stuck = (0.4, True, True, None, {"null_eval": 2, "free_eval": 3, "null_starts": 1, "free_starts": 2})
    fail = (np.inf, False, False, None, {"error": "RuntimeError: element 0 of tensors does not require grad"})
    fine = (3.2, True, True, None, {"null_eval": 900, "free_eval": 1200, "null_starts": 1, "free_starts": 2})
    assert "exactly 0" in hp.degenerate([zero] * 5)
    assert "stopped at their start" in hp.degenerate([stuck] * 5)
    assert "failed" in hp.degenerate([fail] * 5) and "grad" in hp.degenerate([fail] * 5)
    assert hp.degenerate([zero] * 4 + [fine]) is None and hp.degenerate([fail] * 4 + [fine]) is None


def test_run_hp_stops_on_a_degenerate_bootstrap():
    panel, info = small_panel("S1", n=60, seed=2)
    spec = spec_for(panel, info)
    f0 = fit_model(panel, spec, "null", maxiter=3)
    f1 = fit_model(panel, spec, "free", maxiter=3)
    zero = (0.0, True, True, None, {"null_eval": 900, "free_eval": 1200})
    try:
        hp.run_hp(panel, spec, n_max=50, fits=(spec, f0, f1), draw=lambda: zero)
        raise AssertionError("the guard did not stop the bootstrap")
    except RuntimeError as e:
        assert "exactly 0" in str(e)
    resumed = hp.run_hp(panel, spec, n_max=8, fits=(spec, f0, f1), draw=lambda: zero, resume=[(0.0, True)] * 5)
    assert resumed.n_drawn == 8                       # a resumed run past the first replicates is not re-checked


def test_the_free_fit_projected_onto_the_null_inverts_the_embedding():
    from pdx.engine.fit import null_to
    panel, info = small_panel("S1", n=80, seed=4)
    for spec in (spec_for(panel, info), replace(spec_for(panel, info), load_on_speed=False)):
        x = random_point(panel, spec, "null", seed=3, scale=0.1)
        back = free_to_null(null_to(x, spec, "free"), spec)
        assert np.allclose(back, x, atol=1e-8)


def test_a_warm_start_is_written_for_another_support():
    panel, info = small_panel("S1", n=80, seed=4)
    spec = spec_for(panel, info)
    sup = spec.support.copy()
    off = np.argwhere(sup & ~np.eye(spec.d, dtype=bool))
    dropped, added = tuple(off[0]), (0, spec.d - 1) if not sup[0, spec.d - 1] else (1, spec.d - 1)
    new = sup.copy()
    new[dropped] = False
    new[added] = True
    spec_b = spec.with_support(new)
    x = random_point(panel, spec, "null", seed=5, scale=0.1)
    y = remap_support(x, spec, spec_b, "null")
    A, B = drifts(torch.as_tensor(x), Layout(spec, "null"))[0], drifts(torch.as_tensor(y), Layout(spec_b, "null"))[0]
    both = sup & new
    assert torch.allclose(A[torch.as_tensor(both)], B[torch.as_tensor(both)])
    assert B[added] == 0 and B[dropped] == 0
    la, lb = Layout(spec, "null"), Layout(spec_b, "null")
    for name in ("gamma", "b", "L", "log_tau", "lam", "log_sigma"):
        assert np.allclose(la.get(x, name), lb.get(y, name))


def test_fit_pair_alternates_and_records_its_starts():
    panel, info = small_panel("S1", n=60, seed=6)
    spec = spec_for(panel, info)
    f0, f1 = fit_pair(panel, spec, maxiter=150)
    labels0, labels1 = [s["label"] for s in f0.starts], [s["label"] for s in f1.starts]
    assert labels0[:2] == ["default", "sigma 0.6"] and "projection of the free fit, round 1" in labels0
    assert labels1[:2] == ["null embedding", "own default"]
    assert f1.loglik >= f0.loglik - 1e-6 and not (f0.failed or f1.failed)
    w0, w1 = fit_pair(panel, spec, x0_null=f0.x, maxiter=150)          # a replicate's rule: warm, no sigma start
    assert [s["label"] for s in w0.starts][0] == "warm" and "sigma 0.6" not in [s["label"] for s in w0.starts]


def test_lambda_zero_model_matches_the_reference_and_simulates():
    from pdx.engine.fit import Fit, start_values
    from pdx.engine.model import unpack_numpy
    from pdx.engine.simulate import simulate_from_fit
    small, info = small_panel("S1", n=30, seed=7)
    spec = replace(spec_for(small, info), load_on_speed=False)
    prep = prepare(small, spec.meas_var, spec.n_groups)
    for kind in ("null", "free"):
        lay = Layout(spec, kind)
        assert not lay.has("lam")
        x = random_point(small, spec, kind, seed=8)
        a = float(loglik(torch.as_tensor(x), lay, prep))
        b = reference_loglik(x, lay, small, prep.mu1.numpy(), prep.S1.numpy(), prep.y1c.numpy())
        assert abs(a - b) / max(1.0, abs(b)) < 1e-8
    lay = Layout(spec, "null")
    x = start_values(small, spec, "null")
    fit = Fit("null", spec, lay, x, 0.0, True, "", 0, unpack_numpy(x, lay))
    assert np.all(fit.params["lam"] == 0)
    sim = simulate_from_fit(fit, small, np.random.default_rng(1))
    assert sim.n == small.n and all(np.isfinite(v[1:][~np.isnan(small.y[i][1:])]).all() for i, v in enumerate(sim.y))


def test_daily_function_takes_the_dose_and_the_motor_domains_the_examination_state():
    from pdx.engine.simulate import _treat
    small, info = small_panel("S1", n=30, seed=7)
    tm = np.zeros(small.d, bool)
    tm[[0, 1, 2]] = True                       # tremor, axial and daily function in the d = 4 preset
    when = np.zeros(small.d, bool)
    when[2] = True
    spec = replace(spec_for(small, info), treat_mask=tm, on_when_treated=when)
    lay = Layout(spec, "null")
    x = random_point(small, spec, "null", seed=9)
    lay.put(x, "kappa", np.full((2, 3), 0.5))
    lay.put(x, "kappa_ldr", np.zeros((2, 3)))
    dose = torch.tensor([[0.0, 3.0, 3.0]])
    on = torch.tensor([[0.0, 0.0, 1.0]])
    h = treat_effect(torch.as_tensor(x), lay, dose, dose, on, torch.tensor([0]))[0].numpy()
    assert h[0].tolist() == [0.0] * small.d                       # no dose
    assert h[1, 0] == 0 and h[1, 1] == 0 and h[1, 2] < 0            # OFF examination, but function treated
    assert h[2, 0] < 0 and h[2, 1] < 0 and h[2, 2] < 0              # ON examination
    par = {"kappa": lay.get(x, "kappa"), "kappa_ldr": lay.get(x, "kappa_ldr")}
    assert np.allclose(_treat(par, spec, dose[0].numpy(), dose[0].numpy(), on[0].numpy(), 0), h)
    prep = prepare(small, spec.meas_var, spec.n_groups)
    a = float(loglik(torch.as_tensor(x), lay, prep))
    b = reference_loglik(x, lay, small, prep.mu1.numpy(), prep.S1.numpy(), prep.y1c.numpy())
    assert abs(a - b) / max(1.0, abs(b)) < 1e-8


def test_a_domain_never_seen_at_the_landmark_in_a_phenotype_stops_with_its_name():
    panel, info = small_panel("S1", n=60, seed=2)
    y = [v.copy() for v in panel.y]
    for i in np.where(panel.group == 1)[0]:
        y[i][0, 1] = np.nan
    bad = replace(panel.with_y(y), domains=["tremor", "axial", "function", "independence"])
    try:
        prepare(bad, np.full(4, 0.02), 2)
        raise AssertionError("prepare accepted an undefined landmark prior")
    except ValueError as e:
        assert "axial" in str(e) and "phenotype 1" in str(e)


@slow
def test_bootstrap_replicates_in_workers_move_with_single_visit_participants():
    """The PC run's failure, end to end: a panel with more single-visit participants than a
    worker's batch, replicates in two worker processes; the statistics must not all be 0."""
    from pdx import parallel
    from pdx.step1 import Step1Config
    panel, info = small_panel("S1", n=120, seed=3)
    panel = _with_single_visits(panel, 40)
    spec = spec_for(panel, info)
    spec_s, f0, f1 = hp.observed_fits(panel, spec, fit_kw={"maxiter": 400})
    args = (f0, panel, spec, info["support"], Step1Config(), {"maxiter": 400}, 11)
    reps = list(parallel.ordered(hp.boot_replicate, range(2), 2, hp.boot_setup, args))
    assert all(r[1] for r in reps) and any(r[0] > 0 for r in reps)
    assert all(r[4]["null_eval"] > 10 and r[4]["free_eval"] > 10 for r in reps)


# ---------------------------------------------------------------- the medication log (mock files)

def _log_dir(rows):
    import ppmi_mock
    tmp = Path(tempfile.mkdtemp())
    ppmi_mock._write(tmp, "Medical_History", "LEDD_Concomitant_Medication_Log", rows,
                     ["PATNO", "EVENT_ID", "LEDTRT", "LEDDOSE", "LEDDOSFRQ", "LEDDSTRMG", "STARTDT", "STOPDT", "LEDD"])
    return tmp


def _visits(pairs):
    import pandas as pd
    from pdx import ppmi
    return pd.DataFrame({"PATNO": [p for p, _ in pairs], "EVENT_ID": [f"E{k}" for k in range(len(pairs))],
                         "date": ppmi.mdate(pd.Series([d for _, d in pairs]))})


def test_a_dose_change_within_the_visit_month_counts_once_at_the_new_dose():
    from pdx import ppmi
    raw = _log_dir([[1, "LOG", "Sinemet", 100, 3, 100, "01/2015", "06/2016", 300],
                    [1, "LOG", "Sinemet", 100, 4, 100, "06/2016", None, 400],
                    [2, "LOG", "Pramipexole", 1, 1, 1, "01/2015", "06/2016", 100],
                    [3, "LOG", "Amantadine", 100, 1, 100, "01/2015", "06/2016", 100],
                    [3, "LOG", "Carbidopa/Levodopa", 100, 3, 100, "06/2016", None, 300]])
    v = _visits([(1, "05/2016"), (1, "06/2016"), (1, "07/2016"), (2, "06/2016"), (3, "06/2016")])
    out = ppmi._ledd(raw, v).set_index("EVENT_ID")
    assert out.loc["E0", "ledd"] == 300 and out.loc["E1", "ledd"] == 400 and out.loc["E2", "ledd"] == 400
    assert out.loc["E3", "ledd"] == 100                         # stopped without a replacement: its last month
    assert out.loc["E4", "ledd"] == 300 and out.loc["E4", "ledd_levodopa"] == 300


def test_the_exposure_follows_a_slow_compartment():
    from pdx import ppmi
    raw = _log_dir([[1, "LOG", "Sinemet", 100, 3, 100, "01/2015", "06/2016", 300],
                    [1, "LOG", "Sinemet", 100, 4, 100, "06/2016", None, 400],
                    [2, "LOG", "Stalevo", 100, 3, 100, "01/2015", None, 300],
                    [2, "LOG", "Entacapone", 200, 3, 200, "01/2015", None, "LD x 0.33"]])
    v = _visits([(1, "06/2016"), (1, "01/2017"), (2, "06/2016")])
    out = ppmi._ledd(raw, v, half_life_days=28.0).set_index("EVENT_ID")
    w0 = 1 - np.exp(-np.log(2) / 28.0 * 30.4375)
    assert abs(out.loc["E0", "exposure"] - (400 * w0 + 300 * (1 - w0))) < 1e-6
    assert 399.0 < out.loc["E1", "exposure"] < 400.0
    assert abs(out.loc["E2", "exposure"] - 300 * 1.33) < 1e-6        # held for months, with the COMT multiplier


def test_levodopa_names():
    import re
    from pdx import ppmi
    yes = ["L-Dopa", "L dopa", "Duopa", "co-careldopa", "Co-beneldopa", "Dhivy", "carbidopa/levodopa", "Stalevo",
           "Sinemet CR", "Madopar", "Rytary", "foslevodopa/foscarbidopa"]
    no = ["Carbidopa", "Lodosyn", "methyldopa", "Pramipexole", "Rasagiline", "Entacapone"]
    hit = lambda s: re.search(ppmi.LEVODOPA, s.lower()) is not None
    assert all(hit(s) for s in yes), [s for s in yes if not hit(s)]
    assert not any(hit(s) for s in no), [s for s in no if hit(s)]


def test_an_untreated_examination_with_a_dose_in_force_is_not_used():
    import pandas as pd
    import ppmi_mock
    from pdx import ppmi
    tmp = Path(tempfile.mkdtemp())
    ppmi_mock.make(tmp)
    path = next((tmp / "Medical_History").glob("LEDD_Concomitant_Medication_Log_*.csv"))
    log = pd.read_csv(path)
    log.loc[len(log)] = [1, "LOG", "Sinemet", 100, 3, 100, "06/2016", None, 300]    # participant 1 starts treatment
    log.to_csv(path, index=False)
    v = ppmi.build(tmp)["visits"].set_index(["PATNO", "EVENT_ID"])
    assert v.loc[(1, "V06"), "motor_state"] == "treated, state unrecorded" and np.isnan(v.loc[(1, "V06"), "tremor"])
    assert v.loc[(1, "V04"), "motor_state"] == "untreated" and not np.isnan(v.loc[(1, "V04"), "tremor"])
    assert v.loc[(1, "V06"), "exposure"] > 0 and v.loc[(1, "V04"), "exposure"] == 0


def test_the_panel_reports_the_variances_for_the_error_conversion():
    import ppmi_mock
    from pdx import ppmi
    tmp = Path(tempfile.mkdtemp())
    ppmi_mock.make(tmp)
    out = ppmi.build(tmp)
    panel, ns, info = ppmi.analysis_panel(out["participants"], out["grid"])
    assert set(info["variance_year0"]) == set(ppmi.CORE_DOMAINS)
    assert set(info["variance_landmark_by_group"]) == {"0", "1"}
    assert all(len(v) == len(ppmi.CORE_DOMAINS) for v in info["variance_landmark_by_group"].values())


# ---------------------------------------------------------------- the run script's checkpoint

def test_the_checkpoint_is_reused_only_for_the_same_run_and_never_with_a_lower_n_max():
    sys.path.insert(0, str(SCRIPTS))
    import run_hp_real
    out = Path(tempfile.mkdtemp())
    path = out / "hp_tdpigd_checkpoint.json"
    settings = {"seed": 1, "panel_sha256": "a", "engine": {"n_quad": 20}}
    log = []
    state = run_hp_real.load_checkpoint(path, settings, 199, log.append)
    assert state["replicates"] == [] and state["n_max"] == 199
    state["replicates"] = [[1.0, True]]
    path.write_text(json.dumps(state))
    again = run_hp_real.load_checkpoint(path, settings, 299, log.append)
    assert again["replicates"] == [[1.0, True]] and again["n_max"] == 299
    try:
        run_hp_real.load_checkpoint(path, settings, 99, log.append)
        raise AssertionError("a lower --n-max was accepted")
    except SystemExit as e:
        assert "--n-max 99" in str(e)
    other = run_hp_real.load_checkpoint(path, {**settings, "seed": 2}, 199, log.append)
    assert other["replicates"] == [] and len(list(out.glob("*.old.json"))) == 1 and "starting afresh" in log[-1]

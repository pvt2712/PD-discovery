"""H-P, the speed estimates and the localisation on the prepared PPMI panel (protocol v0.5, WP4).

Guarded: it reads participant-level data, so it refuses to run without the investigator's
written approval, reads only from data/ and writes aggregates only to results/.

    python scripts/run_hp_real.py --expert configs/expert_map.json --approval "<who, date, document>"
    python scripts/run_hp_real.py --expert configs/expert_map_provisional.json --first-look --approval "..."
    python scripts/run_hp_real.py --expert ... --approval "..." --jobs 6 --time-replicate   # then the full run

PYTHONHASHSEED must be 0 (Step 1's FCI visits sets in hash order; the script refuses otherwise).
--first-look stops after the observed fits (several hours): H-P's statistic with a chi-square
p-value instead of the conditional bootstrap's, the speed estimates with a model-based interval
for Sp-1, and the per-link localisation. It is an exploratory look, not the registered analysis.
--time-replicate stops after the observed fits and one batch of bootstrap replicates, and writes
hp_tdpigd_development_timing.json with the full run's estimated length; the same command without it then
continues from the checkpoint. Progress is printed and appended to <name>.log in the output folder.

Participant set (D33): by default the run uses Paper 2's development participants (training and
validation; escalation-study-v2/PROTOCOL.md §3.2), so the held-out participants stay unread. It
reads data/derived/panel_tdpigd_development.npz and panel_spec_development.json, and writes
hp_tdpigd_development.json, .log, _checkpoint.json and _timing.json. The run refuses a panel built
for another set. --set all reads and writes the earlier whole-cohort names.

The panel (TD and PIGD participants) and its specification come from
scripts/build_analysis_panel.py (data/derived/panel_tdpigd_development.npz and its spec). The expert
map is the frozen WP1 map, a JSON file of links by domain name (configs/expert_map_TEMPLATE.json
shows the format); it holds no participant data. The support is selected on this panel, so that every bootstrap
replicate repeats exactly the same selection; Step 1's reported map, from all participants, is a
separate run. The checkpoint holds the run's settings (input hashes, map, seed, protocol
constants), the observed fits with their parameters, and every bootstrap replicate and speed
sample. A restart with the same settings reuses all of them; a checkpoint from other settings is
renamed and kept, and a lower --n-max than the checkpoint's is refused (audit A8). Reports record
the settings, the code versions used and the fitted parameters.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import _common
import numpy as np
from scipy import stats

from pdx import config, hp, localise, parallel, speed, wp1
from pdx.data import load_panel
from pdx.engine.fit import covariance_checked, fit_from_record, hessian, param_names
from pdx.engine.likelihood import prepare
from pdx.engine.model import ModelSpec
from pdx.step1 import Step1Config, make_support_fn

FIRST_LOOK = ("EXPLORATORY FIRST LOOK, run on the investigator's instruction before WP1-WP3 and registration, with "
              "the map named below. The p-value is the chi-square approximation, not H-P's conditional bootstrap, "
              "and the Sp-1 interval is model-based, not the bootstrap's.")


def expert_support(path: str, names: list) -> tuple[np.ndarray, list]:
    """The support from the frozen map: its links between the panel's domains. A link naming any
    other known domain (treatment, or a floor-bound domain left out of H-P, D26) is set aside and
    reported; an unknown name stops the run."""
    sup, aside = np.zeros((len(names), len(names)), bool), []
    for frm, to in json.loads(open(path, encoding="utf-8").read())["links"]:
        unknown = [x for x in (frm, to) if x not in wp1.DISPLAY]
        if unknown:
            sys.exit(f"Unknown domain in the expert map: {', '.join(unknown)}")
        if frm in names and to in names:
            sup[names.index(to), names.index(frm)] = True
        else:
            aside.append([frm, to])
    return sup, aside


def write_atomic(path, text: str) -> None:
    """Write through a temporary file, so an interruption never leaves half a checkpoint. Windows
    refuses to replace a file another program has open, so a refusal is retried."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text)
    for attempt in range(20):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.5)


def name_links(classes: list, names: list) -> list:
    """Localisation verdicts with their (to, from) indices replaced by [from, to] names."""
    for c in classes:
        to, frm = (int(v) for v in c["link"])
        c["link"] = [names[frm], names[to]]
    return classes


def named_support(support: np.ndarray, expert: np.ndarray, names: list) -> list:
    """The support's links between domains by name, each marked by where it came from."""
    return [{"from": names[j], "to": names[i], "source": "map" if expert[i, j] else "Step 1"}
            for i, j in np.argwhere(support) if i != j]


def sp1_interval(f0, panel, level: float) -> list:
    """A model-based interval for Sp-1 (delta method on the speed-only fit), for the first look;
    the protocol's interval comes from the bootstrap (§2.5)."""
    lay = f0.layout
    V = covariance_checked(hessian(f0, panel))[0]
    a = np.zeros(len(f0.x))
    a[lay.slices["gamma"][0].start] = 1.0
    if lay.has("theta"):
        y1c, g = speed._y1c(panel, f0.spec), panel.group
        a[lay.slices["theta"][0]] = y1c[g == 1].mean(0) - y1c[g == 0].mean(0)
    est, se = np.log(speed.sp1(f0, panel)), float(np.sqrt(max(a @ V @ a, 0.0)))
    z = stats.norm.ppf(0.5 + level / 2)
    return [float(np.exp(est - z * se)), float(np.exp(est + z * se))]


def first_look(panel, spec, support_fn, expert, names, cfg, rng, log, fit_kw: dict | None = None) -> dict:
    """H-P's observed fits only: the statistic with a chi-square p-value, the speed estimates and
    the per-link localisation (the full run adds the bootstrap)."""
    t0 = time.time()
    log("support selection and the two observed fits: started (several hours)")
    spec_s, f0, f1 = hp.observed_fits(panel, spec, support_fn, fit_kw)
    lr, df = hp.lr_stat(f0, f1), hp.h_p_df(spec_s)
    p = float(stats.chi2.sf(lr, df))
    log(f"fits done ({(time.time() - t0) / 60:.0f} min): LR {lr:.2f} on {df} df, chi-square p = {p:.3g}")
    out = {"hp": {"lr": lr, "lr_raw": 2.0 * (f1.loglik - f0.loglik), "df": df, "p_chi2": p,
                  "chi2_critical": float(stats.chi2.ppf(1 - cfg["hp"]["alpha"], df)),
                  "null_loglik": f0.loglik, "free_loglik": f1.loglik, "null_converged": f0.success,
                  "free_converged": f1.success, "free_start_logliks": f1.start_logliks,
                  "null_fit": f0.summary(), "free_fit": f1.summary()},
           "observed_fit_parameters": {"names": param_names(f0.layout), "null": f0.x,
                                       "free_names": param_names(f1.layout), "free": f1.x},
           "support": named_support(spec_s.support, expert, names)}
    est = speed.speed_estimates(f0, f1, panel)
    est["sp1_interval_model_based"] = sp1_interval(f0, panel, cfg["speed"]["level"])
    out["speed"] = est
    log(f"speed estimates done: Sp-1 = {est['sp1']:.2f} (model-based 95% interval "
        f"{est['sp1_interval_model_based'][0]:.2f}-{est['sp1_interval_model_based'][1]:.2f})")
    prep = prepare(panel, spec_s.meas_var, spec_s.n_groups)
    links = localise.per_link_tests(f1, panel, cfg["localisation"]["mc_draws"], rng,
                                    cfg["localisation"]["near_zero"], prep)
    out["links_exploratory"] = name_links(localise.classify_links(links, cfg["localisation"]["eps"], "by"), names)
    out["speed_and_load_only"] = localise.speed_and_load_only(links, cfg["localisation"]["eps"])
    out["speed_set"] = links.speed_set
    out["speed_diagnostics"] = links.diagnostics
    out["domains"] = {"names": names, **localise.per_domain_tests(links, spec_s)}
    log(f"localisation done ({(time.time() - t0) / 60:.0f} min in all)")
    return out


def build_spec(panel, pspec: dict, names: list, expert: np.ndarray, cfg: dict) -> ModelSpec:
    """The analysis model: the map's support, the panel's treated domains and error variances, and
    the protocol constants (20 quadrature nodes and lam = 0 since D31-D32; D24's examination state)."""
    return ModelSpec(d=panel.d, support=expert, p=panel.p, treat_mask=np.isin(names, pspec["treated_domains"]),
                     meas_var=np.array(pspec["meas_var"], float), treatment=cfg["engine"]["treatment"],
                     ed50=cfg["engine"]["ed50"], ec50_ldr=cfg["engine"]["ec50_ldr"], n_quad=cfg["engine"]["n_quad"],
                     noise_era_col=cfg["engine"]["noise_era_col"], load_on_speed=cfg["engine"]["load_on_speed"],
                     on_when_treated=np.isin(names, cfg["treatment"]["on_when_treated"]))


def run_settings(a, paths, cfg: dict, s1: Step1Config) -> dict:
    """Everything a checkpoint must share with the run that resumes it (audit A8): the participant
    set, the inputs (by hash), the map, the seed and the protocol constants that change the fits or
    the replicates."""
    return {"participant_set": a.set, "seed": a.seed, "panel_sha256": _common.file_sha256(paths[0]),
            "panel_spec_sha256": _common.file_sha256(paths[1]), "map_sha256": _common.file_sha256(a.expert),
            "engine": cfg["engine"], "fit": cfg["fit"], "treatment": cfg["treatment"], "h": cfg["hp"]["h"],
            "step1": {"alpha": s1.alpha, "depth": s1.depth, "test": s1.test, "add_interval": s1.add_interval,
                      "include_dose": s1.include_dose}}


def load_checkpoint(path, settings: dict, n_max: int, log) -> dict:
    """The checkpoint if it holds this run (same settings), else a fresh state; a checkpoint from
    other settings is kept beside, renamed, never overwritten. A lower --n-max than the checkpoint's
    is refused: Besag-Clifford's p-value needs the n_max the replicates were drawn under."""
    fresh = {"format": 2, "settings": settings, "n_max": n_max, "code_versions": [], "replicates": [],
             "replicate_info": []}
    if not path.exists():
        return fresh
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("format") != 2 or state.get("settings") != json.loads(json.dumps(_common.to_json(settings))):
        old = path.with_name(f"{path.stem}.{time.strftime('%Y%m%d-%H%M%S')}.old.json")
        path.replace(old)
        log(f"the checkpoint holds another run (other settings, inputs or format); kept as {old.name}, starting afresh")
        return fresh
    if n_max < state["n_max"]:
        sys.exit(f"Refusing: --n-max {n_max} is below the checkpoint's {state['n_max']}; the p-value needs the "
                 "n_max the replicates were drawn under. Use at least that, or move the checkpoint away.")
    state["n_max"] = n_max
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["development", "all"], default="development",
                    help="participants: Paper 2's development sets (D33; the default) or everyone (the old run)")
    ap.add_argument("--panel", default=None, help="default: data/derived/panel_tdpigd_<set>.npz")
    ap.add_argument("--spec", default=None, help="default: data/derived/panel_spec_<set>.json")
    ap.add_argument("--expert", required=True, help="the frozen expert map: JSON {'links': [[from, to], ...]}")
    ap.add_argument("--interval-in-step1", action="store_true", help="the §6.3 interval rule failed in WP0")
    ap.add_argument("--dose-in-step1", action="store_true", help="dose nodes in the Step 1 table")
    ap.add_argument("--approval", default=None)
    ap.add_argument("--seed", type=int, default=20261103)
    ap.add_argument("--first-look", action="store_true",
                    help="observed fits only, with a chi-square p-value: an exploratory look before WP1-WP3")
    ap.add_argument("--time-replicate", action="store_true",
                    help="after the observed fits, run one batch of bootstrap replicates (--jobs of them, at least "
                         f"{hp.GUARD_REPLICATES}), report their time and the full run's estimate, and stop; the "
                         "replicates are kept for the full run")
    ap.add_argument("--n-max", type=int, default=None, help="H-P bootstrap replicates at most (default: the protocol's)")
    ap.add_argument("--speed-bootstrap", type=int, default=None, help="speed bootstrap samples (default: the protocol's)")
    ap.add_argument("--threads", type=int, default=None, help="threads for the engine (default: PyTorch's)")
    ap.add_argument("--jobs", type=int, default=1, help="bootstrap replicates and speed samples run this many at once")
    ap.add_argument("--out", default=str(_common.RESULTS / "wp4"))
    a = ap.parse_args()
    if a.threads:
        import torch
        torch.set_num_threads(a.threads)
    _common.require_approval(a.approval, "the WP4 analysis")
    _common.require_hash_seed()
    sfx = "" if a.set == "all" else f"_{a.set}"
    a.panel = a.panel or str(_common.DATA / "derived" / f"panel_tdpigd{sfx}.npz")
    a.spec = a.spec or str(_common.DATA / "derived" / f"panel_spec{sfx}.json")
    paths = [_common.require_inside(_common.Path(p), _common.DATA, "inputs") for p in (a.panel, a.spec)]
    out = _common.require_inside(_common.Path(a.out), _common.RESULTS, "the output")
    out.mkdir(parents=True, exist_ok=True)
    name = ("hp_first_look" if a.first_look else "hp_tdpigd") + sfx

    def log(message: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M')}  {message}"
        print(line, flush=True)
        with open(out / f"{name}.log", "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    cfg = config.load()
    panel = load_panel(paths[0])
    pspec = json.loads(paths[1].read_text(encoding="utf-8"))
    built_for = pspec.get("split", {}).get("participant_set", "all")
    if built_for != a.set:
        sys.exit(f"Refusing: the panel was built for the participant set '{built_for}', and this run asks for "
                 f"'{a.set}'. Rebuild it with scripts/build_analysis_panel.py --set {a.set}.")
    names = pspec["domains"]
    expert, aside = expert_support(a.expert, names)
    spec = build_spec(panel, pspec, names, expert, cfg)
    fit_kw = dict(cfg["fit"])
    rng = np.random.default_rng(a.seed)
    s1 = Step1Config(alpha=cfg["step1"]["alpha"], depth=cfg["step1"]["depth_fast"], test=cfg["step1"]["test_fast"],
                     add_interval=a.interval_in_step1, include_dose=a.dose_in_step1)
    support_fn = make_support_fn(expert, s1)
    settings = run_settings(a, paths, cfg, s1)
    code = _common.code_version()
    log(f"participant set: {a.set}; panel: {panel.n} participants, {panel.d} domains; map {a.expert} ({int(expert.sum())} links); "
        f"{spec.n_quad} quadrature nodes; load moves with speed: {spec.load_on_speed}; code {code['commit']}")

    map_status = json.loads(open(a.expert, encoding="utf-8").read()).get("status")
    if a.first_look:
        report = {"status": FIRST_LOOK, "approval": a.approval, "map": a.expert, "map_status": map_status,
                  "settings": settings, "code": code, "counts": panel.aggregate_summary(),
                  "map_links_set_aside": aside,
                  **first_look(panel, spec, support_fn, expert, names, cfg, rng, log, fit_kw)}
        (out / f"{name}.json").write_text(json.dumps(_common.to_json(report), indent=1))
        log(f"wrote {out / f'{name}.json'}")
        return

    n_max = a.n_max or cfg["hp"]["n_max"]
    n_speed = cfg["speed"]["bootstrap"] if a.speed_bootstrap is None else a.speed_bootstrap
    ckpt = out / f"{name}_checkpoint.json"
    state = load_checkpoint(ckpt, settings, n_max, log)
    if code not in state["code_versions"]:
        state["code_versions"].append(code)

    def save_state():
        write_atomic(ckpt, json.dumps(_common.to_json(state)))

    # the observed fits are kept in the checkpoint, so a restart with the same settings reuses them
    if "observed" in state:
        spec_s = spec.with_support(np.array(state["observed"]["support"], bool))
        fits = (spec_s, fit_from_record(state["observed"]["null"], spec_s),
                fit_from_record(state["observed"]["free"], spec_s))
        log(f"observed fits reused from the checkpoint; {len(state['replicates'])} H-P replicates done earlier")
    else:
        log("support selection and the two observed fits: started (several hours)")
        t0 = time.time()
        fits = hp.observed_fits(panel, spec, support_fn, fit_kw)
        state["observed"] = {"support": fits[0].support.astype(int).tolist(),
                             "null": fits[1].to_record(), "free": fits[2].to_record()}
        save_state()
        log(f"observed fits done ({(time.time() - t0) / 3600:.1f} h): speed-only {fits[1].message}, "
            f"{fits[1].n_eval} evaluations; free {fits[2].message}, {fits[2].n_eval} evaluations")
    lr_obs = hp.lr_stat(fits[1], fits[2])
    log(f"observed statistic: LR {lr_obs:.2f} on {hp.h_p_df(fits[0])} df")
    resume = [(np.inf if s is None else s, ok) for s, ok in state["replicates"]] or None

    def save(drawn, exceed, stat, ok, rest):
        info = rest[2] if len(rest) > 2 and rest[2] else {}
        state["replicates"].append([stat if np.isfinite(stat) else None, ok])
        state["replicate_info"].append(info)
        save_state()
        how = (f"failed ({info.get('error', 'unknown')})" if not ok else
               f"{info.get('null_eval', 0) + info.get('free_eval', 0)} evaluations"
               + (", support re-selected differently" if info.get("support_changed") else "")
               + (f", {info['seconds'] / 60:.0f} min" if "seconds" in info else ""))
        log(f"H-P bootstrap replicate {drawn} of at most {n_max}: statistic {stat:.2f} ({how}); "
            f"{exceed} of {cfg['hp']['h']} needed to stop are at or above the observed")

    # each replicate has its own seed, so `jobs` of them run at once and Besag-Clifford's rule is
    # applied in replicate order: any jobs >= 2 give identical results (one job, in this process,
    # agrees to the fits' stopping rule)
    done = len(state["replicates"])
    upto = min(n_max, done + max(a.jobs, hp.GUARD_REPLICATES)) if a.time_replicate else n_max
    draws = parallel.ordered(hp.boot_replicate, range(done, upto), a.jobs, hp.boot_setup,
                             (fits[1], panel, spec, expert, s1, fit_kw, a.seed))
    try:
        res = hp.run_hp(panel, spec, support_fn, upto, cfg["hp"]["h"], callback=save, resume=resume, fits=fits,
                        draw=lambda: next(draws))
    finally:
        draws.close()
    if a.time_replicate:
        secs = [i["seconds"] for i in state["replicate_info"][done:] if i and "seconds" in i]
        per = float(np.median(secs)) if secs else float("nan")
        est = per * np.ceil(max(n_max - len(state["replicates"]), 0) / max(a.jobs, 1)) / 3600
        timing = {"replicates_timed": len(secs), "median_minutes_per_replicate": per / 60, "jobs": a.jobs,
                  "remaining_replicates_at_most": n_max - len(state["replicates"]),
                  "hours_for_the_rest_at_most": est, "replicate_info": state["replicate_info"][done:],
                  "note": "Besag-Clifford stops early once h replicates reach the observed statistic, so the "
                          "full run can be much shorter than the estimate."}
        (out / f"{name}_timing.json").write_text(json.dumps(_common.to_json(timing), indent=1))
        log(f"timing: {per / 60:.0f} min per replicate in a worker; the remaining {timing['remaining_replicates_at_most']} "
            f"replicates take at most {est:.0f} h with {a.jobs} jobs; the replicates are kept for the full run")
        return
    log(f"H-P done: p = {res.p:.3g} from {res.n_drawn} bootstrap replicates")
    status = map_status if map_status and map_status.startswith(("PROVISIONAL", "TEMPLATE")) else None
    report = {"status": ("EXPLORATORY: " + status) if status else "map as given", "approval": a.approval,
              "map": a.expert, "map_status": map_status, "settings": settings, "code_versions": state["code_versions"],
              "counts": panel.aggregate_summary(),
              "hp": {**res.summary(), "n_max": n_max, "h": cfg["hp"]["h"]},
              "observed_fit_parameters": {"names": param_names(res.null_fit.layout), "null": res.null_fit.x,
                                          "free_names": param_names(res.free_fit.layout), "free": res.free_fit.x},
              "support": res.support.astype(int).tolist(), "support_links": named_support(res.support, expert, names),
              "map_links_set_aside": aside, "bootstrap_lr": res.lr_star}
    spec_s = spec.with_support(res.support)
    if n_speed:
        report["speed"] = speed.bootstrap_speed(panel, spec_s, res.null_fit, res.free_fit, n_speed, seed=a.seed,
                                                fit_kw=fit_kw, support=(expert, s1), done=state.get("speed"),
                                                on_sample=save_speed(state, save_state, n_speed, log), jobs=a.jobs)
    else:
        report["speed"] = {"estimates": speed.speed_estimates(res.null_fit, res.free_fit, panel), "B": 0}
    prep = prepare(panel, spec_s.meas_var, spec_s.n_groups)
    links = localise.per_link_tests(res.free_fit, panel, cfg["localisation"]["mc_draws"],
                                    np.random.default_rng([a.seed, 2]), cfg["localisation"]["near_zero"], prep)
    report["links_exploratory"] = name_links(localise.classify_links(links, cfg["localisation"]["eps"], "by"), names)
    report["speed_and_load_only"] = localise.speed_and_load_only(links, cfg["localisation"]["eps"])
    report["speed_diagnostics"] = links.diagnostics
    report["speed_set"] = links.speed_set
    report["domains"] = {"names": names, **localise.per_domain_tests(links, spec_s)}
    (out / f"{name}.json").write_text(json.dumps(_common.to_json(report), indent=1))
    log(f"wrote {out / f'{name}.json'}")


def save_speed(state: dict, save_state, n_speed: int, log):
    """The speed bootstrap's checkpoint callback."""
    def on_sample(done):
        state["speed"] = done
        save_state()
        log(f"speed bootstrap sample {len(done['rows']) + done['failures']} of {n_speed} done "
            f"({done['failures']} failed so far)")
    return on_sample


if __name__ == "__main__":
    main()

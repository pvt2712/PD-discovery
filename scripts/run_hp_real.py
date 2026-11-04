"""H-P, the speed estimates and the localisation on the prepared PPMI panel (protocol v0.5, WP4).

Guarded: it reads participant-level data, so it refuses to run without the investigator's
written approval, reads only from data/ and writes aggregates only to results/.

    python scripts/run_hp_real.py --expert configs/expert_map.json --approval "<who, date, document>"
    python scripts/run_hp_real.py --expert configs/expert_map_provisional.json --first-look --approval "..."

--first-look stops after the observed fits (about an hour): H-P's statistic with a chi-square
p-value instead of the conditional bootstrap's, the speed estimates with a model-based interval
for Sp-1, and the per-link localisation. It is an exploratory look, not the registered analysis.
Progress is printed and appended to <name>.log in the output folder.

The panel (TD and PIGD participants) and its specification come from
scripts/build_analysis_panel.py (data/derived/panel_tdpigd.npz and panel_spec.json). The expert
map is the frozen WP1 map, a JSON file of links by domain name (configs/expert_map_TEMPLATE.json
shows the format); it holds no participant data. The support is selected on this panel, so that every bootstrap
replicate repeats exactly the same selection; Step 1's reported map, from all participants, is a
separate run. The bootstrap is checkpointed after every replicate and resumes from the
checkpoint if the observed statistic is unchanged.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import _common
import numpy as np
from scipy import stats

from pdx import config, hp, localise, speed, wp1
from pdx.data import load_panel
from pdx.engine.fit import covariance_checked, hessian
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


def first_look(panel, spec, support_fn, expert, names, cfg, rng, log) -> dict:
    """H-P's observed fits only: the statistic with a chi-square p-value, the speed estimates and
    the per-link localisation (the full run adds the bootstrap)."""
    t0 = time.time()
    log("support selection and the two observed fits: started (about an hour)")
    spec_s, f0, f1 = hp.observed_fits(panel, spec, support_fn)
    lr, df = hp.lr_stat(f0, f1), hp.h_p_df(spec_s)
    p = float(stats.chi2.sf(lr, df))
    log(f"fits done ({(time.time() - t0) / 60:.0f} min): LR {lr:.2f} on {df} df, chi-square p = {p:.3g}")
    out = {"hp": {"lr": lr, "lr_raw": 2.0 * (f1.loglik - f0.loglik), "df": df, "p_chi2": p,
                  "chi2_critical": float(stats.chi2.ppf(1 - cfg["hp"]["alpha"], df)),
                  "null_loglik": f0.loglik, "free_loglik": f1.loglik, "null_converged": f0.success,
                  "free_converged": f1.success, "free_start_logliks": f1.start_logliks},
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=str(_common.DATA / "derived" / "panel_tdpigd.npz"))
    ap.add_argument("--spec", default=str(_common.DATA / "derived" / "panel_spec.json"))
    ap.add_argument("--expert", required=True, help="the frozen expert map: JSON {'links': [[from, to], ...]}")
    ap.add_argument("--interval-in-step1", action="store_true", help="the §6.3 interval rule failed in WP0")
    ap.add_argument("--dose-in-step1", action="store_true", help="dose nodes in the Step 1 table")
    ap.add_argument("--approval", default=None)
    ap.add_argument("--seed", type=int, default=20261103)
    ap.add_argument("--first-look", action="store_true",
                    help="observed fits only, with a chi-square p-value: an exploratory look before WP1-WP3")
    ap.add_argument("--n-max", type=int, default=None, help="H-P bootstrap replicates at most (default: the protocol's)")
    ap.add_argument("--speed-bootstrap", type=int, default=None, help="speed bootstrap samples (default: the protocol's)")
    ap.add_argument("--threads", type=int, default=None, help="threads for the engine (default: PyTorch's)")
    ap.add_argument("--out", default=str(_common.RESULTS / "wp4"))
    a = ap.parse_args()
    if a.threads:
        import torch
        torch.set_num_threads(a.threads)
    _common.require_approval(a.approval, "the WP4 analysis")
    paths = [_common.require_inside(_common.Path(p), _common.DATA, "inputs") for p in (a.panel, a.spec)]
    out = _common.require_inside(_common.Path(a.out), _common.RESULTS, "the output")
    out.mkdir(parents=True, exist_ok=True)
    name = "hp_first_look" if a.first_look else "hp_tdpigd"

    def log(message: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M')}  {message}"
        print(line, flush=True)
        with open(out / f"{name}.log", "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    cfg = config.load()
    panel = load_panel(paths[0])
    pspec = json.loads(paths[1].read_text(encoding="utf-8"))
    names = pspec["domains"]
    expert, aside = expert_support(a.expert, names)
    tm = np.isin(names, pspec["treated_domains"])
    mv = np.array(pspec["meas_var"], float)
    spec = ModelSpec(d=panel.d, support=expert, p=panel.p, treat_mask=tm, meas_var=mv,
                     treatment=cfg["engine"]["treatment"], ed50=cfg["engine"]["ed50"],
                     ec50_ldr=cfg["engine"]["ec50_ldr"], n_quad=cfg["engine"]["n_quad"],
                     noise_era_col=cfg["engine"]["noise_era_col"])
    rng = np.random.default_rng(a.seed)
    s1 = Step1Config(alpha=cfg["step1"]["alpha"], depth=cfg["step1"]["depth_fast"], test=cfg["step1"]["test_fast"],
                     add_interval=a.interval_in_step1, include_dose=a.dose_in_step1)
    support_fn = make_support_fn(expert, s1)
    log(f"panel: {panel.n} participants, {panel.d} domains; map {a.expert} ({int(expert.sum())} links)")

    map_status = json.loads(open(a.expert, encoding="utf-8").read()).get("status")
    if a.first_look:
        report = {"status": FIRST_LOOK, "approval": a.approval, "map": a.expert, "map_status": map_status,
                  "counts": panel.aggregate_summary(), "map_links_set_aside": aside,
                  **first_look(panel, spec, support_fn, expert, names, cfg, rng, log)}
        (out / f"{name}.json").write_text(json.dumps(_common.to_json(report), indent=1))
        log(f"wrote {out / f'{name}.json'}")
        return

    n_max = a.n_max or cfg["hp"]["n_max"]
    n_speed = cfg["speed"]["bootstrap"] if a.speed_bootstrap is None else a.speed_bootstrap
    ckpt = out / "hp_tdpigd_checkpoint.json"
    state = {"replicates": []}
    if ckpt.exists():
        state = json.loads(ckpt.read_text(encoding="utf-8"))

    def save(drawn, exceed, stat, ok):
        state["replicates"].append([stat if np.isfinite(stat) else None, ok])
        write_atomic(ckpt, json.dumps(state))
        log(f"H-P bootstrap replicate {drawn} of at most {n_max}: statistic {stat:.2f}; "
            f"{exceed} of {cfg['hp']['h']} needed to stop are at or above the observed")

    def save_speed(done):
        state["speed"] = done
        write_atomic(ckpt, json.dumps(state))
        log(f"speed bootstrap sample {len(done['rows']) + done['failures']} of {n_speed} done")

    # the observed fits are repeated on resume (deterministic); saved replicates are reused only if
    # the observed statistic matches the checkpoint
    log("support selection and the two observed fits: started (about an hour)")
    fits = hp.observed_fits(panel, spec, support_fn)
    lr_obs = hp.lr_stat(fits[1], fits[2])
    log(f"observed fits done: LR {lr_obs:.2f} on {hp.h_p_df(fits[0])} df")
    resume = None
    if state.get("lr") is not None and abs(state["lr"] - lr_obs) < 1e-6:
        resume = [(np.inf if s is None else s, ok) for s, ok in state["replicates"]]
        log(f"resuming: {len(resume)} H-P bootstrap replicates were done earlier")
    else:
        state = {"lr": lr_obs, "replicates": []}
        write_atomic(ckpt, json.dumps(state))
    rng = np.random.default_rng([a.seed, len(state["replicates"])])
    res = hp.run_hp(panel, spec, support_fn, n_max, cfg["hp"]["h"], rng, callback=save, resume=resume, fits=fits)
    log(f"H-P done: p = {res.p:.3g} from {res.n_drawn} bootstrap replicates")
    status = map_status if map_status and map_status.startswith(("PROVISIONAL", "TEMPLATE")) else None
    report = {"status": ("EXPLORATORY: " + status) if status else "map as given", "approval": a.approval,
              "map": a.expert, "map_status": map_status, "counts": panel.aggregate_summary(),
              "hp": {**res.summary(), "n_max": n_max, "h": cfg["hp"]["h"]},
              "support": res.support.astype(int).tolist(), "support_links": named_support(res.support, expert, names),
              "map_links_set_aside": aside, "bootstrap_lr": res.lr_star}
    spec_s = spec.with_support(res.support)
    if n_speed:
        report["speed"] = speed.bootstrap_speed(panel, spec_s, res.null_fit, res.free_fit, n_speed,
                                                np.random.default_rng([a.seed, 1]), support_fn=support_fn,
                                                done=state.get("speed"), on_sample=save_speed)
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
    (out / "hp_tdpigd.json").write_text(json.dumps(_common.to_json(report), indent=1))
    log(f"wrote {out / 'hp_tdpigd.json'}")


if __name__ == "__main__":
    main()

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

Size (protocol D34, [wp4] in the configuration): one pair of fits takes days on the real panel, not
the minute the protocol's budget assumed, so by default H-P's bootstrap stops at 39 replicates and
the speed estimates get model-based intervals (draws from the fits' asymptotic normal
distributions) instead of the participant bootstrap. --n-max and --speed-bootstrap give larger
runs, which are sensitivity analyses: the report always gives the p-value at D34's cap, from the
first replicates in order, and the chi-square p-value beside it.

Monitoring: every running fit, in this process or in a worker, writes a line when each start
begins and ends, and one every --progress-minutes (default 5; 0 turns them off) with its
iteration, evaluations, log-likelihood and gain over the stopping window. Each line is labelled
with its piece of work ("observed fits", "replicate 12", "speed sample 3", "localisation").
<name>_status.txt is a snapshot rewritten as the run goes: the step, the counts, the time left at
the current pace, and the latest line of everything still running. A stop (Ctrl+C or an error)
is written to the log with the reason.

Proof of concept (D35): --fixed-map keeps the map's links as the support and switches Step 1's
selection off, in the observed fits and in every bootstrap replicate, so the degrees of freedom
are fixed. --tag NAME reads the panel built with the same tag (build_analysis_panel.py --tag) and
adds _NAME to every output, so the 4-domain run (--tag d4) sits beside the 10-domain one:

    python scripts/run_hp_real.py --expert configs/expert_map_provisional.json --tag d4 --fixed-map \
        --jobs 6 --approval "..."

Groups (D36): the panel's spec names the two groups (build_analysis_panel.py --group-by: TD/PIGD,
age at onset, sex or probable RBD); the log and the report say which, and the speed ratio is group 1
against group 0.

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
renamed and kept. A lower --n-max than the checkpoint's is refused once a replicate has been drawn
(audit A8); before that it is accepted and recorded. Reports record the settings, the code versions
used and the fitted parameters.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import traceback

import _common
import numpy as np
from scipy import stats

from pdx import config, hp, localise, parallel, progress, speed, wp1
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


def first_look(panel, spec, support_fn, expert, names, cfg, rng, log, fit_kw: dict | None = None,
               headline=lambda text: None) -> dict:
    """H-P's observed fits only: the statistic with a chi-square p-value, the speed estimates and
    the per-link localisation (the full run adds the bootstrap)."""
    t0 = time.time()
    log("support selection and the two observed fits: started (several hours)")
    headline(f"Step 1 of 2: the observed fits (support selection, then the speed-only and free fits), started "
             f"{time.strftime('%H:%M')}; they take several hours.")
    with progress.piece("observed fits"):
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
    headline(f"Step 2 of 2: the speed estimates (with the speed-only fit's information matrix) and the "
             f"localisation (the free fit's information matrix), started {time.strftime('%H:%M')}.")
    with progress.piece("speed estimates"):
        est = speed.speed_estimates(f0, f1, panel)
        est["sp1_interval_model_based"] = sp1_interval(f0, panel, cfg["speed"]["level"])
    out["speed"] = est
    log(f"speed estimates done: Sp-1 = {est['sp1']:.2f} (model-based 95% interval "
        f"{est['sp1_interval_model_based'][0]:.2f}-{est['sp1_interval_model_based'][1]:.2f})")
    prep = prepare(panel, spec_s.meas_var, spec_s.n_groups)
    with progress.piece("localisation"):
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
            "step1": None if getattr(a, "fixed_map", False) else
            {"alpha": s1.alpha, "depth": s1.depth, "test": s1.test, "add_interval": s1.add_interval,
             "include_dose": s1.include_dose}}


def load_checkpoint(path, settings: dict, n_max: int, log) -> dict:
    """The checkpoint if it holds this run (same settings), else a fresh state; a checkpoint from
    other settings is kept beside, renamed, never overwritten. A lower --n-max than the checkpoint's
    is refused once a replicate has been drawn: Besag-Clifford's p-value needs the n_max the
    replicates were drawn under, and a cap chosen after seeing some of them could pick the p-value.
    Before the first replicate it is accepted (D34 lowered the cap after the observed fits had
    started) and recorded in the checkpoint."""
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
        if state["replicates"]:
            sys.exit(f"Refusing: --n-max {n_max} is below the checkpoint's {state['n_max']}, and "
                     f"{len(state['replicates'])} replicates were drawn under it; the p-value needs the n_max they "
                     "were drawn under. Use at least that, or move the checkpoint away.")
        state.setdefault("n_max_changes", []).append(
            {"from": state["n_max"], "to": n_max, "replicates_drawn": 0, "when": time.strftime("%Y-%m-%d %H:%M")})
        log(f"bootstrap cap lowered from {state['n_max']} to {n_max} before any replicate was drawn")
    state["n_max"] = n_max
    return state


def p_at_cap(replicates: list, observed: float, cap: int, h: int) -> dict | None:
    """Besag-Clifford's p-value at the cap of D34 from the first replicates in order: H-P's result
    whatever larger number is drawn later. None while it is still open."""
    first = [(np.inf if s is None else s, ok) for s, ok in replicates[:cap]]
    if len(first) < cap and sum(1 for s, ok in first if not ok or s >= observed) < h:
        return None
    r = hp.sequential_p(lambda: None, observed, cap, h, resume=first)
    return {"cap": cap, **{k: r[k] for k in ("p", "n_drawn", "n_exceed", "stopped_early")}}


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
    ap.add_argument("--n-max", type=int, default=None,
                    help="H-P bootstrap replicates at most (default: D34's 39; a larger cap is a sensitivity analysis)")
    ap.add_argument("--speed-bootstrap", type=int, default=None,
                    help="participant-bootstrap samples for the speed estimates (default: D34's 0, which gives "
                         "model-based intervals)")
    ap.add_argument("--threads", type=int, default=None, help="threads for the engine (default: PyTorch's)")
    ap.add_argument("--jobs", type=int, default=1, help="bootstrap replicates and speed samples run this many at once")
    ap.add_argument("--progress-minutes", type=float, default=5.0,
                    help="minutes between the progress lines of each running fit (0: none); it never changes a result")
    ap.add_argument("--fixed-map", action="store_true",
                    help="D35: the map's links only, no Step 1 selection (observed fits and every replicate)")
    ap.add_argument("--tag", default=None,
                    help="reads panel_tdpigd_<set>_<tag>.npz and its spec and adds _<tag> to the outputs (D35: d4)")
    ap.add_argument("--out", default=str(_common.RESULTS / "wp4"))
    a = ap.parse_args()
    if a.threads:
        import torch
        torch.set_num_threads(a.threads)
    _common.require_approval(a.approval, "the WP4 analysis")
    _common.require_hash_seed()
    sfx = ("" if a.set == "all" else f"_{a.set}") + (f"_{a.tag}" if a.tag else "")
    a.panel = a.panel or str(_common.DATA / "derived" / f"panel_tdpigd{sfx}.npz")
    a.spec = a.spec or str(_common.DATA / "derived" / f"panel_spec{sfx}.json")
    paths = [_common.require_inside(_common.Path(p), _common.DATA, "inputs") for p in (a.panel, a.spec)]
    out = _common.require_inside(_common.Path(a.out), _common.RESULTS, "the output")
    out.mkdir(parents=True, exist_ok=True)
    name = ("hp_first_look" if a.first_look else "hp_tdpigd") + sfx
    lock = threading.Lock()          # the workers' lines arrive on another thread

    def log(message: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M')}  {message}"
        with lock:
            print(line, flush=True)
            with open(out / f"{name}.log", "a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    board = progress.Board(out / f"{name}_status.txt", f"H-P run {name} (participant set: {a.set})")

    def on_progress(where: str, message: str, event: str) -> None:
        if event == "line":
            log(f"[{where}] {message}" if where else message)
        board.note(where, message, event)

    progress.configure(on_progress if a.progress_minutes > 0 else None, a.progress_minutes)
    try:
        analysis(a, paths, out, name, log, board)
    except KeyboardInterrupt:
        log("stopped by the user (Ctrl+C); the same command continues from the checkpoint")
        board.headline("Stopped by the user (Ctrl+C). The same command continues from the checkpoint.")
        raise
    except Exception:
        log("stopped by an error; once it is fixed, the same command continues from the checkpoint\n"
            + traceback.format_exc())
        board.headline("Stopped by an error: the log ends with the reason. Once it is fixed, the same command "
                       "continues from the checkpoint.")
        raise


def analysis(a, paths, out, name: str, log, board) -> None:
    """The run itself; main parses the arguments and sets up the log, the progress lines and the
    status file."""
    cfg = config.load()
    panel = load_panel(paths[0])
    pspec = json.loads(paths[1].read_text(encoding="utf-8"))
    built_for = pspec.get("split", {}).get("participant_set", "all")
    if built_for != a.set:
        sys.exit(f"Refusing: the panel was built for the participant set '{built_for}', and this run asks for "
                 f"'{a.set}'. Rebuild it with scripts/build_analysis_panel.py --set {a.set}.")
    names = pspec["domains"]
    grouping = pspec.get("grouping") or {"by": "phenotype", "group_0": "TD", "group_1": "PIGD"}
    expert, aside = expert_support(a.expert, names)
    spec = build_spec(panel, pspec, names, expert, cfg)
    fit_kw = dict(cfg["fit"])
    rng = np.random.default_rng(a.seed)
    s1 = Step1Config(alpha=cfg["step1"]["alpha"], depth=cfg["step1"]["depth_fast"], test=cfg["step1"]["test_fast"],
                     add_interval=a.interval_in_step1, include_dose=a.dose_in_step1)
    support_fn = None if a.fixed_map else make_support_fn(expert, s1)
    step1_cfg = None if a.fixed_map else s1        # what the bootstrap and speed samples re-select with
    settings = run_settings(a, paths, cfg, s1)
    code = _common.code_version()
    log(f"participant set: {a.set}; panel: {panel.n} participants, {panel.d} domains; map {a.expert} ({int(expert.sum())} links); "
        f"{spec.n_quad} quadrature nodes; load moves with speed: {spec.load_on_speed}; code {code['commit']}")
    log(f"groups ({grouping['by']}): 0 = {grouping['group_0']}, 1 = {grouping['group_1']}; the speed ratio is 1 against 0")
    if a.fixed_map:
        log(f"support: the map's {int(expert.sum())} links between these domains, fixed (D35); Step 1's selection is off")
    log((f"progress: every running fit adds a line every {a.progress_minutes:g} min" if a.progress_minutes > 0
         else "progress lines are off") + f"; {name}_status.txt holds a snapshot of the run")

    map_status = json.loads(open(a.expert, encoding="utf-8").read()).get("status")
    if a.first_look:
        report = {"status": FIRST_LOOK, "approval": a.approval, "map": a.expert, "map_status": map_status,
                  "settings": settings, "code": code, "grouping": grouping, "counts": panel.aggregate_summary(),
                  "map_links_set_aside": aside,
                  **first_look(panel, spec, support_fn, expert, names, cfg, rng, log, fit_kw, board.headline)}
        (out / f"{name}.json").write_text(json.dumps(_common.to_json(report), indent=1))
        log(f"wrote {out / f'{name}.json'}")
        board.headline(f"Finished at {time.strftime('%H:%M')}: the first look is in {name}.json.")
        return

    n_max = a.n_max or cfg["wp4"]["n_max"]                      # D34
    n_speed = cfg["wp4"]["speed_bootstrap"] if a.speed_bootstrap is None else a.speed_bootstrap
    ckpt = out / f"{name}_checkpoint.json"
    state = load_checkpoint(ckpt, settings, n_max, log)
    if code not in state["code_versions"]:
        state["code_versions"].append(code)

    def save_state():
        write_atomic(ckpt, json.dumps(_common.to_json(state)))

    steps = 2 if a.time_replicate else 4
    # the observed fits are kept in the checkpoint, so a restart with the same settings reuses them
    if "observed" in state:
        spec_s = spec.with_support(np.array(state["observed"]["support"], bool))
        fits = (spec_s, fit_from_record(state["observed"]["null"], spec_s),
                fit_from_record(state["observed"]["free"], spec_s))
        log(f"observed fits reused from the checkpoint; {len(state['replicates'])} H-P replicates done earlier")
    else:
        what = "the two observed fits" if a.fixed_map else "support selection and the two observed fits"
        log(f"{what}: started (several hours)")
        board.headline(f"Step 1 of {steps}: the observed fits ("
                       + ("the map fixed; " if a.fixed_map else "support selection, then ")
                       + f"the speed-only and free fits), started {time.strftime('%H:%M')}; they take several hours. "
                       f"They are saved to the checkpoint only when both are done.")
        t0 = time.time()
        with progress.piece("observed fits"):
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
        board.headline(boot_headline(drawn, exceed))

    # each replicate has its own seed, so `jobs` of them run at once and Besag-Clifford's rule is
    # applied in replicate order: any jobs >= 2 give identical results (one job, in this process,
    # agrees to the fits' stopping rule)
    done = len(state["replicates"])
    upto = min(n_max, done + max(a.jobs, hp.GUARD_REPLICATES)) if a.time_replicate else n_max
    h = cfg["hp"]["h"]

    def boot_headline(drawn: int, exceed: int) -> str:
        """The status file's headline during the bootstrap: counts, pace and the time left."""
        secs = [i["seconds"] for i in state["replicate_info"] if i and "seconds" in i]
        per = float(np.median(secs)) if secs else float("nan")
        left = max(upto - drawn, 0)
        rows = [f"Step 2 of {steps}: " + ("a timing batch of H-P bootstrap replicates" if a.time_replicate
                                          else "the H-P bootstrap") + f", {a.jobs} at once",
                f"  Replicates done: {drawn} of " + (f"{upto} in this batch" if a.time_replicate else f"at most {n_max}"),
                f"  At or above the observed statistic (LR {lr_obs:.2f}): {exceed}; the bootstrap stops once {h} are"]
        if secs:
            rows += [f"  One replicate takes {progress.duration(per)} in a worker (median of {len(secs)})",
                     f"  Time left: at most {progress.duration(per * np.ceil(left / max(a.jobs, 1)))}"
                     + ("" if a.time_replicate else f"; less if {h} reach the observed statistic first")]
        else:
            rows.append("  Time per replicate and time left: known once the first replicate finishes")
        return "\n".join(rows)

    log(f"H-P bootstrap: replicates {done + 1} to at most {upto}, {a.jobs} at once; each worker's fits report "
        f"under [replicate N]")
    board.headline(boot_headline(done, sum(1 for s, ok in state["replicates"] if not ok or s is None or s >= lr_obs)))
    draws = parallel.ordered(hp.boot_replicate, range(done, upto), a.jobs, hp.boot_setup,
                             (fits[1], panel, spec, expert, step1_cfg, fit_kw, a.seed))
    try:
        res = hp.run_hp(panel, spec, support_fn, upto, cfg["hp"]["h"], callback=save, resume=resume, fits=fits,
                        draw=lambda: next(draws))
    finally:
        draws.close()
        board.clear()                # the workers have stopped: nothing is running any more
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
        board.headline(f"Finished the timing batch at {time.strftime('%H:%M')}: {per / 60:.0f} min per replicate in a "
                       f"worker; the remaining {timing['remaining_replicates_at_most']} replicates take at most {est:.0f} h "
                       f"with {a.jobs} jobs. Send {name}_timing.json and {name}.log.")
        return
    df = hp.h_p_df(fits[0])
    p_chi2 = float(stats.chi2.sf(lr_obs, df))
    at_cap = p_at_cap(state["replicates"], lr_obs, cfg["wp4"]["n_max"], h)
    log(f"H-P done: p = {res.p:.3g} from {res.n_drawn} bootstrap replicates"
        + (f"; at D34's cap of {at_cap['cap']}: p = {at_cap['p']:.3g}" if at_cap and n_max != at_cap["cap"] else "")
        + f"; chi-square p = {p_chi2:.3g} on {df} df (sensitivity)")
    status = map_status if map_status and map_status.startswith(("PROVISIONAL", "TEMPLATE")) else None
    report = {"status": ("EXPLORATORY: " + status) if status else "map as given", "approval": a.approval,
              "map": a.expert, "map_status": map_status, "settings": settings, "code_versions": state["code_versions"],
              "grouping": grouping, "counts": panel.aggregate_summary(),
              "hp": {**res.summary(), "n_max": n_max, "h": h, "n_max_changes": state.get("n_max_changes", []),
                     "result_at_d34_cap": at_cap,
                     "p_chi2": p_chi2, "p_chi2_note": "chi-square reference on the selected support's df: a "
                                                     "sensitivity analysis beside the bootstrap (D34)"},
              "observed_fit_parameters": {"names": param_names(res.null_fit.layout), "null": res.null_fit.x,
                                          "free_names": param_names(res.free_fit.layout), "free": res.free_fit.x},
              "support": res.support.astype(int).tolist(), "support_links": named_support(res.support, expert, names),
              "map_links_set_aside": aside, "bootstrap_lr": res.lr_star}
    spec_s = spec.with_support(res.support)
    if n_speed:
        prior = state.get("speed") or {}
        k0 = len(prior.get("rows", [])) + prior.get("failures", 0)
        log(f"speed bootstrap: samples {k0 + 1} to {n_speed}, {a.jobs} at once; each worker's fits report under "
            f"[speed sample N]")
        board.headline(f"Step 3 of 4: the speed bootstrap, {a.jobs} at once\n  Samples done: {k0} of {n_speed}\n"
                       f"  Time left: known once the first samples finish")
        try:
            report["speed"] = speed.bootstrap_speed(panel, spec_s, res.null_fit, res.free_fit, n_speed, seed=a.seed,
                                                    fit_kw=fit_kw, support=None if a.fixed_map else (expert, s1),
                                                    done=state.get("speed"),
                                                    on_sample=save_speed(state, save_state, n_speed, log,
                                                                         board.headline, a.jobs, k0),
                                                    jobs=a.jobs)
        finally:
            board.clear()
    prep = prepare(panel, spec_s.meas_var, spec_s.n_groups)
    if not n_speed:
        draws = cfg["wp4"]["model_draws"]
        log(f"speed estimates: model-based intervals (D34) from both fits' information matrices and {draws} draws")
        board.headline(f"Step 3 of 4: model-based intervals for the speed estimates (D34): the information "
                       f"matrices of both fits, then {draws} draws; started {time.strftime('%H:%M')}")
        with progress.piece("speed intervals"):
            cov_null, info_null = covariance_checked(hessian(res.null_fit, panel, prep),
                                                     names=param_names(res.null_fit.layout))
            cov_free, info_free = covariance_checked(hessian(res.free_fit, panel, prep),
                                                     names=param_names(res.free_fit.layout))
            report["speed"] = {**speed.model_based(res.null_fit, res.free_fit, panel, cov_null, cov_free,
                                                   cfg["speed"]["level"], draws, np.random.default_rng([a.seed, 3])),
                               "information_null": info_null, "information_free": info_free}
        sp = report["speed"]
        log(f"speed estimates done: Sp-1 = {sp['estimates']['sp1']:.2f} (model-based 95% interval "
            f"{sp['sp1_ci'][0]:.2f}-{sp['sp1_ci'][1]:.2f})")
    board.headline(f"Step 4 of 4: the localisation (the free fit's information matrix, then "
                   f"{cfg['localisation']['mc_draws']} Monte Carlo draws), started {time.strftime('%H:%M')}")
    with progress.piece("localisation"):
        links = localise.per_link_tests(res.free_fit, panel, cfg["localisation"]["mc_draws"],
                                        np.random.default_rng([a.seed, 2]), cfg["localisation"]["near_zero"], prep)
    report["links_exploratory"] = name_links(localise.classify_links(links, cfg["localisation"]["eps"], "by"), names)
    report["speed_and_load_only"] = localise.speed_and_load_only(links, cfg["localisation"]["eps"])
    report["speed_diagnostics"] = links.diagnostics
    report["speed_set"] = links.speed_set
    report["domains"] = {"names": names, **localise.per_domain_tests(links, spec_s)}
    (out / f"{name}.json").write_text(json.dumps(_common.to_json(report), indent=1))
    log(f"wrote {out / f'{name}.json'}")
    board.headline(f"Finished at {time.strftime('%H:%M')}: H-P p = {res.p:.3g} from {res.n_drawn} replicates; the "
                   f"report is {name}.json.")


def save_speed(state: dict, save_state, n_speed: int, log, headline=lambda text: None, jobs: int = 1,
               k0: int = 0):
    """The speed bootstrap's checkpoint callback; it also refreshes the status file's headline,
    with the time left at this session's pace (k0 samples were done before it began)."""
    t0 = time.time()

    def on_sample(done):
        state["speed"] = done
        save_state()
        k = len(done["rows"]) + done["failures"]
        log(f"speed bootstrap sample {k} of {n_speed} done ({done['failures']} failed so far)")
        new = k - k0
        left = ("known once the first samples finish" if new < max(jobs, 1) else
                f"about {progress.duration((time.time() - t0) / new * max(n_speed - k, 0))} at this session's pace")
        headline(f"Step 3 of 4: the speed bootstrap, {jobs} at once\n  Samples done: {k} of {n_speed} "
                 f"({done['failures']} failed)\n  Time left: {left}")
    return on_sample


if __name__ == "__main__":
    main()

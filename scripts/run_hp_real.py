"""H-P, the speed estimates and the localisation on the prepared PPMI panel (protocol v0.5, WP4).

Guarded: it reads participant-level data, so it refuses to run without the investigator's
written approval, reads only from data/ and writes aggregates only to results/.

    python scripts/run_hp_real.py --expert configs/expert_map.json --approval "<who, date, document>"

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

import _common
import numpy as np

from pdx import config, hp, localise, speed
from pdx.data import load_panel
from pdx.engine.likelihood import prepare
from pdx.engine.model import ModelSpec
from pdx.step1 import Step1Config, make_support_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=str(_common.DATA / "derived" / "panel_tdpigd.npz"))
    ap.add_argument("--spec", default=str(_common.DATA / "derived" / "panel_spec.json"))
    ap.add_argument("--expert", required=True, help="the frozen expert map: JSON {'links': [[from, to], ...]}")
    ap.add_argument("--interval-in-step1", action="store_true", help="the §6.3 interval rule failed in WP0")
    ap.add_argument("--dose-in-step1", action="store_true", help="dose nodes in the Step 1 table")
    ap.add_argument("--approval", default=None)
    ap.add_argument("--seed", type=int, default=20261103)
    ap.add_argument("--out", default=str(_common.RESULTS / "wp4"))
    a = ap.parse_args()
    _common.require_approval(a.approval, "the WP4 analysis")
    paths = [_common.require_inside(_common.Path(p), _common.DATA, "inputs") for p in (a.panel, a.spec)]
    out = _common.require_inside(_common.Path(a.out), _common.RESULTS, "the output")
    out.mkdir(parents=True, exist_ok=True)
    cfg = config.load()
    panel = load_panel(paths[0])
    pspec = json.loads(paths[1].read_text())
    names = pspec["domains"]
    expert = np.zeros((panel.d, panel.d), bool)
    for frm, to in json.loads(open(a.expert).read())["links"]:
        expert[names.index(to), names.index(frm)] = True
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

    ckpt = out / "hp_tdpigd_checkpoint.json"
    state = {"replicates": []}
    if ckpt.exists():
        state = json.loads(ckpt.read_text())

    def save(drawn, exceed, stat, ok):
        state["replicates"].append([stat if np.isfinite(stat) else None, ok])
        ckpt.write_text(json.dumps(state))

    # the observed fits are repeated on resume (deterministic); saved replicates are reused only if
    # the observed statistic matches the checkpoint
    fits = hp.observed_fits(panel, spec, support_fn)
    lr_obs = hp.lr_stat(fits[1], fits[2])
    resume = None
    if state.get("lr") is not None and abs(state["lr"] - lr_obs) < 1e-6:
        resume = [(np.inf if s is None else s, ok) for s, ok in state["replicates"]]
    else:
        state = {"lr": lr_obs, "replicates": []}
        ckpt.write_text(json.dumps(state))
    rng = np.random.default_rng([a.seed, len(state["replicates"])])
    res = hp.run_hp(panel, spec, support_fn, cfg["hp"]["n_max"], cfg["hp"]["h"], rng, callback=save,
                    resume=resume, fits=fits)
    report = {"approval": a.approval, "counts": panel.aggregate_summary(), "hp": res.summary(),
              "support": res.support.astype(int).tolist(), "bootstrap_lr": res.lr_star}
    spec_s = spec.with_support(res.support)
    report["speed"] = speed.bootstrap_speed(panel, spec_s, res.null_fit, res.free_fit, cfg["speed"]["bootstrap"],
                                            rng, support_fn=support_fn)
    prep = prepare(panel, spec_s.meas_var, spec_s.n_groups)
    links = localise.per_link_tests(res.free_fit, panel, cfg["localisation"]["mc_draws"], rng,
                                    cfg["localisation"]["near_zero"], prep)
    report["links_exploratory"] = localise.classify_links(links, cfg["localisation"]["eps"], "by")
    report["speed_and_load_only"] = localise.speed_and_load_only(links, cfg["localisation"]["eps"])
    report["speed_diagnostics"] = links.diagnostics
    report["speed_set"] = links.speed_set
    report["domains"] = localise.per_domain_tests(links, spec_s)
    (out / "hp_tdpigd.json").write_text(json.dumps(_common.to_json(report), indent=1))
    print("wrote", out / "hp_tdpigd.json")


if __name__ == "__main__":
    main()

"""Paper 2's speed modifier: each participant's filtered log-speed at their treatment decisions
(escalation-study-v2/PROTOCOL.md §8 and Appendix A, task 8; exploratory; src/pdx/speed_at.py).

Guarded: it reads participant-level data, so it refuses to run without the investigator's written
approval. It reads
  results/wp4/hp_tdpigd_development_checkpoint.json   the H-P run's speed-only fit and support
                                                      (fitted on the development participants)
  data/derived/panel_tdpigd_development.npz           the panel that fit was made on (its landmark
                                                      prior and the centring of the landmark scores)
  data/derived/panel_spec_development.json            its domains and error variances
  data/derived/normal_scores_development.npz          its frozen normal-scores tables
  data/derived/participants.parquet, grid.parquet     the tables of build_ppmi_tables.py
  --decisions <CSV inside data/>                      Paper 2's decisions: PATNO and date_t (monthly
                                                      dates on the 15th)
and writes data/derived/speed_at_decisions_<set>.csv (participant-level, git-ignored) with PATNO,
date_t, tau (the decision on the panel's time scale), n_visits_used, log_speed_mean, log_speed_sd,
prior_mean and prior_sd. It prints counts only.

--set heldout scores Paper 2's held-out participants with the engine fitted on the development
participants, on the development panel's normal-scores tables, error variances, covariate centring
and landmark prior (all frozen); --set development scores the development participants. Before
anything is scored, the checkpoint must come from a development run on exactly these panel and spec
files (SHA-256 checked), and the development panel rebuilt from the tables with the frozen tables
must reproduce the saved one. Only visits before each decision enter its speed, but the held-out
participants' panel is built from all their grid visits. So --set heldout refuses to run until
Paper 2's freeze manifest exists (escalation-study-v2/docs/freeze/manifest.json; PROTOCOL.md §3.3,
rule 1); --freeze-manifest names it if the two repositories are not side by side.

    python scripts/speed_at_decisions.py --approval "<who, date, document>" --set heldout \
        --decisions data/derived/<decisions>.csv [--checkpoint results/wp4/...] [--n-quad 20] [--plain]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _common
import numpy as np
import pandas as pd

from pdx import speed_at, split
from pdx.data import load_panel
from pdx.engine.fit import fit_from_record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--approval", default=None)
    ap.add_argument("--set", choices=["heldout", "development"], required=True,
                    help="the participants scored (Paper 2's split, pdx.split)")
    ap.add_argument("--decisions", required=True, help="CSV inside data/ with PATNO and date_t")
    ap.add_argument("--checkpoint", default=None,
                    help="the development H-P checkpoint (default: results/wp4/hp_tdpigd_development_checkpoint.json)")
    ap.add_argument("--n-quad", type=int, default=None, help="Gauss-Hermite nodes (default: the fit's)")
    ap.add_argument("--freeze-manifest", default=str(_common.ROOT.parent / "escalation-study-v2" / "docs" / "freeze" / "manifest.json"),
                    help="Paper 2's freeze manifest; --set heldout runs only when it exists")
    ap.add_argument("--plain", action="store_true",
                    help="the plain Gauss-Hermite rule on the prior, without re-centring the nodes on each posterior")
    a = ap.parse_args()
    _common.require_approval(a.approval, "the speed-at-decision scoring")
    if a.set == "heldout" and not Path(a.freeze_manifest).is_file():
        sys.exit("Refusing: Paper 2's held-out set stays closed until its freeze (escalation-study-v2/PROTOCOL.md §3.3, "
                 f"rule 1), and the freeze manifest is missing: {a.freeze_manifest}")
    import run_hp_real                             # build_spec: the run's model, built the same way

    der = _common.DATA / "derived"
    ckpt = _common.require_inside(Path(a.checkpoint or _common.RESULTS / "wp4" / "hp_tdpigd_development_checkpoint.json"),
                                  _common.RESULTS, "the checkpoint")
    files = {"panel": "panel_tdpigd_development.npz", "spec": "panel_spec_development.json",
             "ns": "normal_scores_development.npz", "participants": "participants.parquet", "grid": "grid.parquet"}
    paths = {k: _common.require_inside(der / f, _common.DATA, "inputs") for k, f in files.items()}
    dec_path = _common.require_inside(Path(a.decisions), _common.DATA, "the decisions")
    for p in [ckpt, dec_path, *paths.values()]:
        if not p.exists():
            sys.exit(f"Missing input: {p}")

    # the model: the run's speed-only fit, on exactly the panel and spec it was fitted with
    state = json.loads(ckpt.read_text(encoding="utf-8"))
    settings = state.get("settings", {})
    if settings.get("participant_set") != "development":
        sys.exit("Refusing: the checkpoint is not from a run on the development participants (D33).")
    if "observed" not in state:
        sys.exit("Refusing: the checkpoint holds no observed fits yet.")
    for key, name in (("panel_sha256", "panel"), ("panel_spec_sha256", "spec")):
        if settings.get(key) != _common.file_sha256(paths[name]):
            sys.exit(f"Refusing: the checkpoint's fits were made on another {files[name]} (SHA-256 differs).")
    pspec = json.loads(paths["spec"].read_text(encoding="utf-8"))
    names = pspec["domains"]
    dev = load_panel(paths["panel"])
    support = np.array(state["observed"]["support"], bool)
    spec = run_hp_real.build_spec(dev, pspec, names, support,
                                  {"engine": settings["engine"], "treatment": settings["treatment"]})
    spec = spec.with_support(support)
    null = fit_from_record(state["observed"]["null"], spec)
    if null.failed:
        sys.exit("Refusing: the checkpoint's speed-only fit failed.")

    # the development panel again, from the tables with the frozen transform: it must be the fitted one
    part = pd.read_parquet(paths["participants"])
    grid = pd.read_parquet(paths["grid"])
    ns = speed_at.load_normal_scores(paths["ns"])
    motor_on = bool(pspec.get("motor_on", False))
    dev_part = part[split.in_sets(part.PATNO, "development")]
    rebuilt = speed_at.panel_with_frozen_transform(dev_part, grid[grid.PATNO.isin(set(dev_part.PATNO))], names, ns,
                                                   motor_on)
    if not speed_at.same_panel(rebuilt, dev):
        sys.exit("Refusing: the development panel rebuilt from the tables with the frozen normal-scores tables differs "
                 "from the one the fit was made on (were the tables rebuilt since?).")
    reference = speed_at.landmark_reference(dev, spec.meas_var, spec.n_groups)
    if a.set == "development":
        panel = rebuilt
    else:
        z_mean = speed_at.baseline_covariates(dev_part, rebuilt.ids).mean().to_numpy(float)
        held = part[split.in_sets(part.PATNO, "heldout")]
        panel = speed_at.panel_with_frozen_transform(held, grid[grid.PATNO.isin(set(held.PATNO))], names, ns, motor_on,
                                                     z_mean=z_mean)

    # the decisions on the panel's time scale, and their speeds
    dec = pd.read_csv(dec_path)
    if not {"PATNO", "date_t"} <= set(dec.columns):
        sys.exit("Refusing: the decisions need the columns PATNO and date_t.")
    n_file = len(dec)
    dec = dec.loc[split.in_sets(dec.PATNO, a.set), ["PATNO", "date_t"]].copy()
    dec["date_t"] = pd.to_datetime(dec.date_t, errors="coerce")
    dec["tau"] = speed_at.to_panel_time(dec, part, grid)
    res = speed_at.filtered_log_speed(null, spec, panel, dec, reference=reference, n_quad=a.n_quad,
                                      adaptive=not a.plain)
    out = der / f"speed_at_decisions_{a.set}.csv"
    res[["PATNO", "date_t", "tau", *speed_at.OUTPUT]].to_csv(out, index=False)

    c = res.attrs["counts"]
    by_group = {("TD", "PIGD")[g]: int((panel.group == g).sum()) for g in range(2)}
    print(f"participant set: {a.set}; engine panel: {panel.n} participants {json.dumps(by_group)}")
    print(f"decisions: {n_file} in the file, {len(dec)} of {a.set} participants; dropped: {c['not_in_panel']} of "
          f"participants outside the engine's panel, {c['no_time']} without a date or anchor, {c['before_landmark']} "
          f"before the year-1 landmark; scored: {c['scored']} ({c['landmark_only']} on the landmark visit only, "
          f"{c['pseudo_participants']} distinct visit sets)")
    print(f"quadrature: {c['nodes']} nodes, " + ("plain Gauss-Hermite on the prior" if a.plain else
          f"adaptive, {c['adaptive_rounds']} rounds, {c['not_converged']} not converged"))
    print(f"wrote data/derived/{out.name}")


if __name__ == "__main__":
    main()

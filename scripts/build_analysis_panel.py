"""Build the H-P analysis panel from the WP0 tables (protocol v0.5, §2.3, §7.1; for WP2-WP4).

Guarded: it reads participant-level tables, so it refuses to run without the investigator's
written approval. It reads data/derived/participants.parquet and grid.parquet (from
build_ppmi_tables.py) and writes, all inside data/derived/ (git-ignored):
  panel_tdpigd_development.npz       the panel (no pickling)
  panel_spec_development.json        domains, treated domains, fixed measurement-error variances (D28)
                                     and how they were derived, with each landmark variance's margin
                                     over them, and the participant set with its split record
  normal_scores_development.npz      the frozen normal-scores tables
It prints counts only.

Participant set (D33): by default only the development participants of Paper 2's split (training
and validation; escalation-study-v2/PROTOCOL.md §3.2) enter, so the held-out participants stay
unread. Their scores also never shape the normal-scores tables or the error variances. --set all
builds the earlier whole-cohort panel under the old file names (panel_tdpigd.npz and so on); no
analysis approved after 6 November 2026 uses it.

    python scripts/build_analysis_panel.py --approval "<who, date, document>" \
        [--set development|all] [--domains tremor brady_rigidity ...] \
        [--measurement configs/measurement_error.json] [--motor-on]
"""
from __future__ import annotations

import argparse
import json

import _common
import numpy as np
import pandas as pd

from pdx import ppmi, split
from pdx.data import save_panel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--approval", default=None)
    ap.add_argument("--domains", nargs="+", default=ppmi.CORE_DOMAINS)
    ap.add_argument("--measurement", default=str(_common.ROOT / "configs" / "measurement_error.json"),
                    help="JSON with error_variance = 1 - reliability per domain (D28)")
    ap.add_argument("--unscaled-errors", action="store_true",
                    help="sensitivity: use 1 - reliability as the error variance, not scaled to the year-0 variance")
    ap.add_argument("--motor-on", action="store_true", help="D5 option: ON-only examinations enter with on = 1")
    ap.add_argument("--set", choices=["development", "all"], default="development",
                    help="participants: Paper 2's development sets (D33; the default) or everyone (the old panel)")
    a = ap.parse_args()
    _common.require_approval(a.approval, "the analysis-panel build")
    d = _common.DATA / "derived"
    part = pd.read_parquet(d / "participants.parquet")
    grid = pd.read_parquet(d / "grid.parquet")
    every = split.counts(part.PATNO)
    if a.set != "all":
        part = part[split.in_sets(part.PATNO, a.set)]
        grid = grid[grid.PATNO.isin(set(part.PATNO))]
    sfx = "" if a.set == "all" else f"_{a.set}"
    split_record = {"participant_set": a.set, "sets": list(split.NAMED[a.set]), "rule": f"SHA-256('{split.SALT}:' + PATNO), "
                    f"first 15 hex digits / 16^15; training < {split.CUTS[0]}, validation < {split.CUTS[1]}",
                    "participants_by_set_in_the_tables": every,
                    "assignment_sha256_of_the_participants_used": split.assignment_sha256(part.PATNO)}
    print(f"participant set: {a.set} ({' + '.join(split.NAMED[a.set])}): {len(part)} of {sum(every.values())} participants "
          f"in the tables")
    panel, ns, info = ppmi.analysis_panel(part, grid, a.domains, a.motor_on)
    meas = json.loads(open(a.measurement, encoding="utf-8").read())
    unreliable = {dn: float(meas["error_variance"][dn]) for dn in a.domains}       # 1 - reliability
    # audit A5: a reliability is a share of its sample's variance; on the pooled scale (all visits,
    # variance 1) an early-disease sample varies less, so 1 - R overstates the error. The year-0
    # variance stands in for the reliability samples' (early PD), provisionally (D28).
    ref = {dn: 1.0 if a.unscaled_errors else float(info["variance_year0"][dn]) for dn in a.domains}
    mv = [unreliable[dn] * ref[dn] for dn in a.domains]
    margin = {grp: {dn: float(v - e) for dn, v, e in zip(a.domains, var, mv)}
              for grp, var in info["variance_landmark_by_group"].items()}
    thin = sorted({dn for m in margin.values() for dn, v in m.items() if v < 0.05})
    spec = {**info, "meas_var": mv, "measurement_source": meas.get("source"), "approval": a.approval, "split": split_record,
            "measurement": {"rule": ("1 - reliability" if a.unscaled_errors else
                                     "(1 - reliability) x the year-0 variance on the frozen scale (audit A5)"),
                            "one_minus_reliability": unreliable, "reference_variance": ref,
                            "landmark_margin_by_group": margin, "domains_with_margin_below_0.05": thin}}
    save_panel(panel, d / f"panel_tdpigd{sfx}.npz")
    np.savez_compressed(d / f"normal_scores{sfx}.npz", **{f"values_{j}": u for j, (u, _) in enumerate(ns.tables)},
                        **{f"scores_{j}": sc for j, (_, sc) in enumerate(ns.tables)})
    (d / f"panel_spec{sfx}.json").write_text(json.dumps(_common.to_json(spec), indent=1))
    print(f"wrote data/derived/panel_tdpigd{sfx}.npz, panel_spec{sfx}.json and normal_scores{sfx}.npz")
    print("panel:", json.dumps(info["counts"]))
    print("error variances:", json.dumps(dict(zip(a.domains, [round(v, 3) for v in mv]))))
    if thin:
        print("warning: the landmark variance exceeds the error variance by less than 0.05 for", ", ".join(thin))


if __name__ == "__main__":
    main()

"""Build the H-P analysis panel from the WP0 tables (protocol v0.5, §2.3, §7.1; for WP2-WP4).

Guarded: it reads participant-level tables, so it refuses to run without the investigator's
written approval. It reads data/derived/participants.parquet and grid.parquet (from
build_ppmi_tables.py) and writes, all inside data/derived/ (git-ignored):
  panel_tdpigd.npz       the panel (no pickling)
  panel_spec.json        domains, treated domains, fixed measurement-error variances (D28)
  normal_scores.npz      the frozen normal-scores tables
It prints counts only.

    python scripts/build_analysis_panel.py --approval "<who, date, document>" \
        [--domains tremor brady_rigidity ...] [--measurement configs/measurement_error.json] [--motor-on]
"""
from __future__ import annotations

import argparse
import json

import _common
import numpy as np
import pandas as pd

from pdx import ppmi
from pdx.data import save_panel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--approval", default=None)
    ap.add_argument("--domains", nargs="+", default=ppmi.CORE_DOMAINS)
    ap.add_argument("--measurement", default=str(_common.ROOT / "configs" / "measurement_error.json"),
                    help="JSON {domain: error variance on the normal-scores scale} (D28)")
    ap.add_argument("--motor-on", action="store_true", help="D5 option: ON-only examinations enter with on = 1")
    a = ap.parse_args()
    _common.require_approval(a.approval, "the analysis-panel build")
    d = _common.DATA / "derived"
    part = pd.read_parquet(d / "participants.parquet")
    grid = pd.read_parquet(d / "grid.parquet")
    panel, ns, info = ppmi.analysis_panel(part, grid, a.domains, a.motor_on)
    meas = json.loads(open(a.measurement, encoding="utf-8").read())
    mv = [float(meas["error_variance"][dn]) for dn in a.domains]
    spec = {**info, "meas_var": mv, "measurement_source": meas.get("source"), "approval": a.approval}
    save_panel(panel, d / "panel_tdpigd.npz")
    np.savez_compressed(d / "normal_scores.npz", **{f"values_{j}": u for j, (u, _) in enumerate(ns.tables)},
                        **{f"scores_{j}": sc for j, (_, sc) in enumerate(ns.tables)})
    (d / "panel_spec.json").write_text(json.dumps(_common.to_json(spec), indent=1))
    print("panel:", json.dumps(info["counts"]))


if __name__ == "__main__":
    main()

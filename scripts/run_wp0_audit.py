"""WP0 feasibility audit (protocol v0.5, §3): aggregate counts only, small cells suppressed.

Guarded: it reads participant-level data, so it refuses to run without the investigator's
written approval, reads only from data/ and writes only to results/wp0/.

    python scripts/run_wp0_audit.py --visits data/derived/wp0_visits.csv --domains tremor axial ... \
        --approval "<who, date, document>"
"""
from __future__ import annotations

import argparse
import json

import _common
import pandas as pd

from pdx import wp0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--visits", required=True)
    ap.add_argument("--domains", nargs="+", required=True)
    ap.add_argument("--min-cell", type=int, default=5)
    ap.add_argument("--approval", default=None)
    ap.add_argument("--out", default=str(_common.RESULTS / "wp0"))
    a = ap.parse_args()
    _common.require_approval(a.approval, "the WP0 audit")
    src = _common.require_inside(_common.Path(a.visits), _common.DATA, "the visit table")
    out = _common.require_inside(_common.Path(a.out), _common.RESULTS, "the output")
    df = pd.read_parquet(src) if src.suffix == ".parquet" else pd.read_csv(src)
    report = wp0.audit(df, wp0.Columns(domains=a.domains), min_cell=a.min_cell)
    report["approval"] = a.approval
    out.mkdir(parents=True, exist_ok=True)
    (out / "wp0_audit.json").write_text(json.dumps(report, indent=1))
    print("wrote", out / "wp0_audit.json")


if __name__ == "__main__":
    main()

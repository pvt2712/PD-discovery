"""The engine gate of protocol v0.5, §16, on synthetic data (no PPMI data).

    python scripts/run_engine_gate.py --quick          # the harness itself, in minutes
    python scripts/run_engine_gate.py --n 1400 --d 10  # the real size (hours; start from the Terminal app)
"""
from __future__ import annotations

import argparse
import json

import _common  # noqa: F401
from pdx.gate import run_gate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1400)
    ap.add_argument("--d", type=int, default=4, choices=(4, 6, 10))
    ap.add_argument("--datasets", type=int, default=20)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=str(_common.RESULTS / "engine_gate"))
    a = ap.parse_args()
    out = _common.require_inside(_common.Path(a.out), _common.RESULTS, "the output")
    out.mkdir(parents=True, exist_ok=True)
    report = run_gate(a.n, a.d, a.datasets, a.seed, a.quick)
    path = out / ("gate_quick.json" if a.quick else f"gate_n{a.n}_d{a.d}.json")
    path.write_text(json.dumps(_common.to_json(report), indent=1))
    print(json.dumps({k: report[k] for k in ("all_automatic_checks_pass", "timing")}, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()

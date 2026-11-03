"""WP3 method qualification on the simulators (protocol v0.5, §8), synthetic data only.

    python scripts/run_qualification.py --scenarios S0 S1 --family A --datasets 500 --n 1400 --jobs 4
    python scripts/run_qualification.py --summarise

One JSON per dataset in results/qualification/datasets/, skipped on rerun; --summarise pools
them into the calibration, H-M1, power and speed tables.
"""
from __future__ import annotations

import argparse
import json

import _common  # noqa: F401
from pdx import qualification as Q

OUT = _common.RESULTS / "qualification"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="*", default=["S0", "S1"])
    ap.add_argument("--family", default="A", choices=("A", "B"))
    ap.add_argument("--datasets", type=int, default=10)
    ap.add_argument("--n", type=int, default=1400)
    ap.add_argument("--d", type=int, default=4)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--no-step1", action="store_true", help="skip support re-selection (faster, not the protocol)")
    ap.add_argument("--maxiter", type=int, default=3000)
    ap.add_argument("--sim-kw", default="{}", help='stress options as JSON, e.g. \'{"spurious": true}\'')
    ap.add_argument("--config-a", default=None, help="configuration key of the protocol run, family A")
    ap.add_argument("--config-b", default=None, help="configuration key of the protocol run, family B")
    ap.add_argument("--summarise", action="store_true")
    a = ap.parse_args()
    ds = OUT / "datasets"
    if not a.summarise:
        tasks = []
        for sc in a.scenarios:
            tasks += Q.make_tasks(sc, a.datasets, a.n, a.family, d=a.d, step1=not a.no_step1,
                                  fit_kw={"maxiter": a.maxiter}, sim_kw=json.loads(a.sim_kw))
        Q.run_tasks(tasks, ds, a.jobs)
    rows = [json.loads(p.read_text()) for p in sorted(ds.glob("*.json"))]
    null_rows = [r for r in rows if r["scenario"] in Q.NULL_A]
    power_rows = [r for r in rows if r["scenario"] not in Q.NULL_A]
    cal = Q.calibration_summary(null_rows)
    cfg = {k: v for k, v in (("A", a.config_a), ("B", a.config_b)) if v}
    summary = {"calibration": cal, "h_m1": Q.h_m1(cal, config=cfg or None),
               "power": Q.power_summary(power_rows, cal), "speed": Q.speed_summary(rows)}
    (OUT / "summary.json").write_text(json.dumps(_common.to_json(summary), indent=1))
    print(json.dumps(_common.to_json(summary), indent=1))


if __name__ == "__main__":
    main()

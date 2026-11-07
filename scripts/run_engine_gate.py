"""The engine gate of protocol v0.5, §16, on synthetic data (no PPMI data).

    python scripts/run_engine_gate.py --quick                      # the harness itself, in minutes
    python scripts/run_engine_gate.py --wp0 --d 10 --datasets 20   # the real size at WP0's numbers (days)
    python scripts/run_engine_gate.py --wp0 --d 10 --datasets 20 --jobs 5   # the same, 5 pieces at once
    python scripts/run_engine_gate.py --n 1400 --d 10              # a chosen size, default visits

--wp0 gives every real-size dataset WP0's group sizes, visits and missingness, as the power check
does (configs/wp0_sim_settings.json, aggregate counts only). Progress is printed and appended to
<name>.log. Finished work is saved as it goes in <name>.partial.json and skipped when the same
command is run again; the file is removed when the report is written.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
import traceback

import _common
import run_power_check
from _common import code_version
from pdx import config, progress
from pdx.gate import run_gate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1400)
    ap.add_argument("--d", type=int, default=4, choices=(4, 6, 10))
    ap.add_argument("--datasets", type=int, default=20)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--wp0", action="store_true", help="WP0's group sizes, visits and missingness")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--threads", type=int, default=None, help="threads for the engine (default: PyTorch's)")
    ap.add_argument("--jobs", type=int, default=1,
                    help="dispersed starts, profiles and recovery datasets run this many at once")
    ap.add_argument("--progress-minutes", type=float, default=5.0,
                    help="minutes between the progress lines of each running fit (0: none); it never changes a result")
    ap.add_argument("--out", default=str(_common.RESULTS / "engine_gate"))
    a = ap.parse_args()
    if a.threads:
        import torch
        torch.set_num_threads(a.threads)
    out = _common.require_inside(_common.Path(a.out), _common.RESULTS, "the output")
    out.mkdir(parents=True, exist_ok=True)
    n, sim_kw = a.n, None
    if a.wp0:
        s = run_power_check.wp0_sim_settings()
        sim_kw = {"target_counts": s["target_counts"], "p_missing": s["p_missing"][:a.d],
                  "visit_probs": {int(k): v for k, v in s["visit_probs"].items()}}
        n = 5 * sum(s["target_counts"])
    name = f"gate_wp0_d{a.d}" if a.wp0 else f"gate_n{a.n}_d{a.d}"
    name = ("gate_quick_wp0" if a.wp0 else "gate_quick") if a.quick else name
    partial = out / f"{name}.partial.json"

    lock = threading.Lock()          # the workers' progress lines arrive on another thread

    def log(message: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M')}  {message}"
        with lock:
            print(line, flush=True)
            with open(out / f"{name}.log", "a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def on_progress(where: str, message: str, event: str) -> None:
        if event == "line":
            log(f"[{where}] {message}" if where else message)

    progress.configure(on_progress if a.progress_minutes > 0 else None, a.progress_minutes)
    log(f"started; finished work is kept in {partial.name}, so the same command resumes"
        + (f"; every running fit adds a progress line every {a.progress_minutes:g} min" if a.progress_minutes > 0 else ""))
    try:
        cfg = config.load()
        engine = {"n_quad": cfg["engine"]["n_quad"], "load_on_speed": cfg["engine"]["load_on_speed"]}
        report = {"code": code_version(),
                  **run_gate(n, a.d, a.datasets, a.seed, a.quick, fit_kw=dict(cfg["fit"]), sim_kw=sim_kw,
                             checkpoint=partial, log=log, jobs=a.jobs, engine=engine)}
    except KeyboardInterrupt:
        log("stopped by the user; run the same command to resume")
        raise
    except Exception:
        log("stopped by an error; run the same command to resume once it is fixed\n" + traceback.format_exc())
        raise
    path = out / f"{name}.json"
    path.write_text(json.dumps(_common.to_json(report), indent=1))
    partial.unlink(missing_ok=True)
    log(f"finished; all automatic checks pass: {report['all_automatic_checks_pass']}; report in {path.name}")
    print(json.dumps({k: report[k] for k in ("all_automatic_checks_pass", "timing")}, indent=1))


if __name__ == "__main__":
    main()

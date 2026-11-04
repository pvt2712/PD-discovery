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
import hashlib
import json
import subprocess
import time
import traceback

import _common
import run_power_check
from pdx.gate import run_gate


def code_version() -> dict:
    """The code the gate runs: the commit, a hash of uncommitted changes to tracked files, and any
    untracked files (so the report can be tied to a later commit)."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=_common.ROOT, capture_output=True, text=True).stdout
    try:
        diff = git("diff", "HEAD")
    except OSError:                                    # git not installed (a downloaded copy)
        return {"commit": None, "uncommitted_diff_sha256": None, "untracked": None}
    return {"commit": git("rev-parse", "HEAD").strip(),
            "uncommitted_diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff else None,
            "untracked": git("ls-files", "--others", "--exclude-standard").split()}


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

    def log(message: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M')}  {message}"
        print(line, flush=True)
        with open(out / f"{name}.log", "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    log(f"started; finished work is kept in {partial.name}, so the same command resumes")
    try:
        report = {"code": code_version(),
                  **run_gate(n, a.d, a.datasets, a.seed, a.quick, sim_kw=sim_kw, checkpoint=partial, log=log,
                             jobs=a.jobs)}
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

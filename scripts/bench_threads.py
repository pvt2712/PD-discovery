"""How best to spread the engine over this machine's cores (protocol v0.5, D14). Synthetic data only.

One synthetic dataset at WP0's numbers (the power pilot's settings) is evaluated by J jobs of T
threads each, all at once. A fit needs the same number of likelihood-and-gradient evaluations
whatever the threading, so the machine's evaluations per hour set the time of the WP2-WP3 runs.
A full fit pair of one pilot dataset then checks the gain against the pilot's own time.

    python scripts/bench_threads.py --configs 4x1 2x2 1x4 1x10 1x1 [--evals 30] [--chunk 128]
    python scripts/bench_threads.py --fit 1x4 --seeds 1080003      # J seeds at once, T threads each
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import _common

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
OUT = _common.RESULTS / "bench"
PILOT = _common.RESULTS / "wp0" / "power" / "datasets"


def eval_child(threads: int, evals: int, seed: int, chunk: int | None = None) -> None:
    """One job: build the dataset and warm up, wait for the parent's go, then time the evaluations."""
    import numpy as np
    import torch
    torch.set_num_threads(threads)
    import run_power_check as R
    from pdx import qualification as Q
    from pdx.engine.fit import start_values
    from pdx.engine.likelihood import prepare, value_and_grad
    from pdx.engine.model import Layout
    from pdx.sim import scenarios
    s = R.wp0_sim_settings()
    kw = {"target_counts": s["target_counts"], "p_missing": s["p_missing"],
          "visit_probs": {int(k): v for k, v in s["visit_probs"].items()}}
    panel, info = scenarios.generate("S2", 5 * sum(s["target_counts"]), np.random.default_rng(seed), family="A",
                                     d=10, **kw)
    spec = Q.spec_for(panel, info)
    prep = prepare(panel, spec.meas_var, spec.n_groups)
    lay = Layout(spec, "free")
    x = start_values(panel, spec, "free")
    for _ in range(2):
        value_and_grad(x, lay, prep, chunk)
    print("ready", flush=True)
    sys.stdin.readline()
    t0 = time.perf_counter()
    for _ in range(evals):
        value_and_grad(x, lay, prep, chunk)
    print(json.dumps({"seconds_per_eval": (time.perf_counter() - t0) / evals, "n": int(panel.n)}), flush=True)


def fit_child(threads: int, seed: int) -> None:
    """One job: the pilot's own task for this seed (Step 1 and the fit pair), timed as in the pilot."""
    import torch
    torch.set_num_threads(threads)
    from pdx import qualification as Q
    task = next(json.loads(p.read_text(encoding="utf-8")) for p in PILOT.glob(f"A_S2_*_seed{seed}.json"))["task"]
    task["sim_kw"]["visit_probs"] = {int(k): v for k, v in task["sim_kw"]["visit_probs"].items()}
    print("ready", flush=True)
    sys.stdin.readline()
    r = Q.dataset_task(task)
    print(json.dumps({"seed": seed, "seconds": r["seconds"], "lr": r["lr"], "df": r["df"]}), flush=True)


def _read(p, start: str) -> str:
    """The job's next output line starting with `start`; anything else (a library's progress) is skipped."""
    for line in p.stdout:
        if line.startswith(start):
            return line.strip()
    raise RuntimeError("a job ended early")


def run_jobs(jobs: int, threads: int, args: list) -> tuple[list, float]:
    """Start the jobs, let them set up, start them together and return their results and the wall time."""
    env = {**os.environ, **{v: str(threads) for v in THREAD_VARS}}
    procs = [subprocess.Popen([sys.executable, __file__, "--child", str(threads)] + a, env=env, text=True,
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE) for a in args[:jobs]]
    for p in procs:
        _read(p, "ready")
    t0 = time.perf_counter()
    for p in procs:
        p.stdin.write("go\n")
        p.stdin.flush()
    res = [json.loads(_read(p, "{")) for p in procs]
    wall = time.perf_counter() - t0
    for p in procs:
        p.wait()
    return res, wall


def parse(cfg: str) -> tuple[int, int]:
    jobs, threads = cfg.lower().split("x")
    return int(jobs), int(threads)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=[])
    ap.add_argument("--evals", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1080003)
    ap.add_argument("--chunk", type=int, default=None,
                    help="participants per likelihood batch (default: the engine's, which follows the threads)")
    ap.add_argument("--fit", default=None, help="JxT: J pilot seeds fitted at once with T threads each")
    ap.add_argument("--seeds", type=int, nargs="*", default=[1080003])
    ap.add_argument("--child", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--fit-seed", type=int, default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.child is not None:
        if a.fit_seed is not None:
            fit_child(a.child, a.fit_seed)
        else:
            eval_child(a.child, a.evals, a.seed, a.chunk)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "threads.json"
    report = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for cfg in a.configs:
        jobs, threads = parse(cfg)
        child = ["--evals", str(a.evals), "--seed", str(a.seed)] + (["--chunk", str(a.chunk)] if a.chunk else [])
        res, wall = run_jobs(jobs, threads, [child] * jobs)
        cfg = f"{cfg}_chunk{a.chunk}" if a.chunk else f"{cfg}_default_chunk"
        report[cfg] = {"chunk": a.chunk, "jobs": jobs, "threads": threads, "evals_each": a.evals,
                       "seconds_per_eval_each": [r["seconds_per_eval"] for r in res],
                       "evals_per_hour": 3600 * jobs * a.evals / wall, "participants": res[0]["n"]}
        print(cfg, json.dumps(report[cfg]), flush=True)
    if a.fit:
        jobs, threads = parse(a.fit)
        res, wall = run_jobs(jobs, threads, [["--fit-seed", str(s)] for s in a.seeds])
        pilot = {r["seed"]: next(json.loads(p.read_text(encoding="utf-8")) for p in PILOT.glob(f"A_S2_*_seed{r['seed']}.json"))
                 for r in res}
        report[f"fit_{a.fit}"] = {"jobs": len(res), "threads": threads, "wall_hours": wall / 3600,
                                  "fit_pairs_per_hour": 3600 * len(res) / wall,
                                  "datasets": [{**r, "pilot_seconds": pilot[r["seed"]]["seconds"],
                                                "pilot_lr": pilot[r["seed"]]["lr"]} for r in res]}
        print(f"fit_{a.fit}", json.dumps(report[f"fit_{a.fit}"]), flush=True)
    path.write_text(json.dumps(_common.to_json(report), indent=1))


if __name__ == "__main__":
    main()

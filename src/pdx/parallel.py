"""Independent pieces of work (bootstrap replicates, simulated datasets, profiles) run at once in
single-threaded worker processes. The engine is limited by memory traffic, so several
single-threaded jobs finish more work per hour than one process with many threads
(scripts/bench_threads.py). Workers are started with "spawn", which also works on Windows.

Every piece carries its own seed, so the results do not depend on how many jobs run them: any
number of workers gives identical results. One job runs in the calling process with its threads,
which changes the order of summation, so its fits reach the stopping rule (a gain under 0.01 over
100 iterations) at slightly other points: in a test, H-P's statistic agreed to 1e-3 and Sp-1's
bootstrap bounds to 0.7% (0.03% with the rule switched off), not bit for bit.

When the calling process reports progress (pdx.progress), the workers report too: their lines
reach the caller's sink through a queue, so one log holds every worker's lines.
"""
from __future__ import annotations

import multiprocessing as mp
import os

from . import progress

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
_shared: dict = {}


def _start(setup, args, report=None):
    import torch
    torch.set_num_threads(1)
    if report is not None:                   # (queue, heartbeat seconds): report like the parent
        progress.configure(progress.queue_sink(report[0]), report[1] / 60.0)
    _shared.clear()
    _shared.update(setup(*args))


def _run(job):
    fn, item = job
    return fn(_shared, item)


def ordered(fn, items, jobs: int, setup, args: tuple = ()):
    """Yield fn(shared, item) for each item, in the order of the items; shared = setup(*args),
    built once in each worker. fn and setup must be module-level functions. With jobs > 1 the
    items run `jobs` at a time in worker processes; closing the generator early (a sequential
    test that has stopped) ends the workers at once. jobs <= 1 runs everything in this process."""
    items = list(items)
    if not items:
        return
    if jobs <= 1 or len(items) == 1:
        shared = setup(*args)
        for item in items:
            yield fn(shared, item)
        return
    saved = {v: os.environ.get(v) for v in THREAD_VARS}
    os.environ.update({v: "1" for v in THREAD_VARS})          # inherited by the workers as they start
    ctx = mp.get_context("spawn")
    fwd = progress.Forwarder(ctx) if progress.enabled() else None
    try:
        pool = ctx.Pool(min(jobs, len(items)), initializer=_start,
                        initargs=(setup, args, (fwd.q, progress.every()) if fwd else None))
    except BaseException:
        if fwd:
            fwd.close()
        raise
    finally:
        for v, old in saved.items():
            if old is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = old
    try:
        yield from pool.imap(_run, [(fn, item) for item in items])
    finally:
        pool.terminate()
        pool.join()
        if fwd:
            fwd.close()


def check_setup(k: float) -> dict:
    """For the tests: a trivial shared state."""
    return {"k": k, "pid": os.getpid()}


def check_square(shared: dict, x: float) -> tuple:
    """For the tests: k x squared, with the worker's process id."""
    return shared["k"] * x * x, shared["pid"]


def check_report(shared: dict, x: float) -> int:
    """For the tests: report a line inside a labelled piece of work, return the process id."""
    with progress.piece(f"item {x}"):
        progress.report(f"square {shared['k'] * x * x:g}")
    return shared["pid"]

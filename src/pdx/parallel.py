"""Independent pieces of work (bootstrap replicates, simulated datasets, profiles) run at once in
single-threaded worker processes. The engine is limited by memory traffic, so several
single-threaded jobs finish more work per hour than one process with many threads
(scripts/bench_threads.py). Workers are started with "spawn", which also works on Windows.

Every piece carries its own seed, so the results do not depend on how many jobs run them: any
number of workers gives identical results. One job runs in the calling process with its threads,
which changes the order of summation, so its fits agree with the workers' to their numerical
tolerance (below 1e-4 in a test statistic), not bit for bit.
"""
from __future__ import annotations

import multiprocessing as mp
import os

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
_shared: dict = {}


def _start(setup, args):
    import torch
    torch.set_num_threads(1)
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
    try:
        pool = mp.get_context("spawn").Pool(min(jobs, len(items)), initializer=_start, initargs=(setup, args))
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


def check_setup(k: float) -> dict:
    """For the tests: a trivial shared state."""
    return {"k": k, "pid": os.getpid()}


def check_square(shared: dict, x: float) -> tuple:
    """For the tests: k x squared, with the worker's process id."""
    return shared["k"] * x * x, shared["pid"]

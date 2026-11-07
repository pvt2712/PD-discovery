"""Progress lines for monitoring long runs (pdx.progress): what they say, where they go, and that they
never change a result. Synthetic data only."""
import os
import tempfile
import threading
from pathlib import Path

import numpy as np
from helpers import small_panel, spec_for

from pdx import parallel, progress
from pdx.engine.fit import fit_model, fit_pair, hessian


def _collect():
    lines, lock = [], threading.Lock()

    def sink(where, message, event):
        with lock:
            lines.append((where, message, event))
    return lines, sink


def test_fits_report_starts_heartbeats_and_ends_and_their_results_do_not_change():
    panel, info = small_panel("S1", n=60, seed=3)
    spec = spec_for(panel, info)
    quiet = fit_model(panel, spec, "null", maxiter=40)
    lines, sink = _collect()
    progress.configure(sink, every_minutes=0)           # a heartbeat at every iteration
    try:
        with progress.piece("test fit"):
            loud = fit_model(panel, spec, "null", maxiter=40)
            H = hessian(loud, panel)
    finally:
        progress.configure(None, 5)
    assert np.array_equal(quiet.x, loud.x) and quiet.loglik == loud.loglik and quiet.n_eval == loud.n_eval
    assert np.all(np.isfinite(H))
    msgs = [m for _, m, e in lines if e == "line"]
    assert {w for w, _, _ in lines} == {"test fit"} and lines[-1] == ("test fit", "", "end")
    assert any("speed-only fit, start 'start 1' (1 of 1): begins at log-likelihood" in m for m in msgs)
    assert any(": iteration " in m and "gained" in m and "s per evaluation" in m for m in msgs)
    assert any(": ended (" in m and "evaluations in" in m for m in msgs)
    assert any(m.startswith("information matrix: 1 of") for m in msgs) and msgs[-1].startswith("information matrix done")


def test_a_fit_pair_reports_the_ratio_as_it_develops():
    panel, info = small_panel("S1", n=60, seed=4)
    spec = spec_for(panel, info)
    lines, sink = _collect()
    progress.configure(sink, every_minutes=60)
    try:
        f0, f1 = fit_pair(panel, spec, maxiter=40)
    finally:
        progress.configure(None, 5)
    msgs = [m for _, m, e in lines if e == "line"]
    assert any("likelihood ratio so far" in m for m in msgs)
    assert msgs[-1].startswith("both fits done in") and f"{max(2 * (f1.loglik - f0.loglik), 0.0):.2f}" in msgs[-1]
    assert not any(": iteration " in m for m in msgs)    # no heartbeat inside an hour


def test_worker_lines_reach_the_parent_in_one_log():
    lines, sink = _collect()
    progress.configure(sink, every_minutes=5)
    try:
        pids = list(parallel.ordered(parallel.check_report, range(4), 2, parallel.check_setup, (2.0,)))
        here = list(parallel.ordered(parallel.check_report, range(2), 1, parallel.check_setup, (2.0,)))
    finally:
        progress.configure(None, 5)
    assert all(p != os.getpid() for p in pids) and all(p == os.getpid() for p in here)
    got = sorted((w, m) for w, m, e in lines if e == "line")
    want = sorted([(f"item {x}", f"square {2.0 * x * x:g}") for x in range(4)]
                  + [(f"item {x}", f"square {2.0 * x * x:g}") for x in range(2)])
    assert got == want
    assert sorted(w for w, _, e in lines if e == "end") == sorted([f"item {x}" for x in range(4)] + ["item 0", "item 1"])


def test_without_a_sink_nothing_is_reported_and_a_broken_sink_never_stops_a_fit():
    assert not progress.enabled()
    panel, info = small_panel("S1", n=60, seed=5)
    spec = spec_for(panel, info)

    def broken(*args):
        raise OSError("the disk is full")
    progress.configure(broken, every_minutes=0)
    try:
        f = fit_model(panel, spec, "null", maxiter=10)
    finally:
        progress.configure(None, 5)
    assert np.isfinite(f.loglik)


def test_the_status_board_shows_the_headline_and_what_is_running():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "status.txt"
        b = progress.Board(path, "Test run")
        b.headline("Step 2 of 4: the H-P bootstrap")
        b.note("replicate 3", "free fit, start 'warm' (1 of 1): iteration 40", "line")
        b.note("replicate 4", "speed-only fit, start 'warm' (1 of 1): begins", "line")
        text = path.read_text(encoding="utf-8")
        assert "Test run" in text and "Step 2 of 4" in text and "replicate 3, for" in text and "iteration 40" in text
        b.note("replicate 3", "", "end")
        text = path.read_text(encoding="utf-8")
        assert "replicate 3" not in text and "replicate 4" in text
        b.clear()
        assert "Working on" not in path.read_text(encoding="utf-8")


def test_durations_read_naturally():
    assert progress.duration(40) == "40 s" and progress.duration(600) == "10 min"
    assert progress.duration(7200) == "2.0 h" and progress.duration(float("nan")) == "unknown"

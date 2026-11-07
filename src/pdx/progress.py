"""Progress lines for long runs (the PPMI analysis, the engine gate): what each fit is doing while
it runs.

The fitting code reports through this module, and a script decides where the lines go
(`configure`). In the worker processes of `parallel.ordered` the lines travel to the parent
through a queue and the parent writes them, so every line lands in one log, in order of arrival,
whatever the number of jobs. With nothing configured nothing is reported (the tests, library
use), and reporting never changes a result.

Each line carries the label of the piece of work it belongs to ("observed fits", "replicate 12",
"speed sample 3"), set with `piece`. A fit reports when each start begins, with the time one
evaluation takes; every `every()` seconds while it runs (the iteration, the evaluations, the
log-likelihood and its gain over the stopping window); and when each start ends. `Board` keeps a
snapshot of the run in a small text file.
"""
from __future__ import annotations

import contextlib
import re
import threading
import time

_state: dict = {"sink": None, "every": 300.0, "labels": []}


def configure(sink=None, every_minutes: float | None = None) -> None:
    """Send progress to sink(label, message, event), where event is "line", or "end" when a piece
    of work finishes; sink None switches reporting off. every_minutes sets the interval between a
    running fit's heartbeat lines (0 gives one at every iteration, for tests)."""
    _state["sink"] = sink
    if every_minutes is not None:
        _state["every"] = max(float(every_minutes), 0.0) * 60.0


def enabled() -> bool:
    return _state["sink"] is not None


def every() -> float:
    """Seconds between the heartbeat lines of a running fit."""
    return _state["every"]


def label() -> str:
    return " / ".join(_state["labels"])


def _emit(where: str, message: str, event: str) -> None:
    sink = _state["sink"]
    if sink is None:
        return
    try:
        sink(where, message, event)
    except Exception:              # a reporting problem never stops the analysis
        pass


def report(message: str) -> None:
    """One progress line, labelled with the current piece of work."""
    _emit(label(), message, "line")


@contextlib.contextmanager
def piece(name: str):
    """Label the lines reported inside the block with `name`; the sink hears "end" on exit."""
    _state["labels"].append(name)
    full = label()
    try:
        yield
    finally:
        _state["labels"].pop()
        _emit(full, "", "end")


class Heartbeat:
    """due() is True once every `every()` seconds, and never while reporting is off."""

    def __init__(self):
        self.last = time.time()

    def due(self) -> bool:
        if not enabled():
            return False
        now = time.time()
        if now - self.last >= every():
            self.last = now
            return True
        return False


def duration(seconds: float) -> str:
    """A duration for a person: '40 s', '12 min' or '3.4 h'."""
    if seconds != seconds or seconds in (float("inf"), float("-inf")):
        return "unknown"
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 90 * 60:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


# ---------------------------------------------------------------- worker processes

def queue_sink(q):
    """In a worker process: a sink that sends every event to the parent through q."""
    def sink(where, message, event):
        q.put((where, message, event))
    return sink


class Forwarder:
    """In the parent: a queue for the workers and a thread that hands their events to this
    process's sink. A SimpleQueue writes synchronously, so a worker's last lines are in the pipe
    before its result reaches the parent, and they survive the pool's termination."""

    def __init__(self, ctx):
        self.q = ctx.SimpleQueue()
        self.sink = _state["sink"]
        self.thread = threading.Thread(target=self._run, name="progress-forwarder", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while True:
            try:
                item = self.q.get()
            except Exception:
                return
            if item is None:
                return
            try:
                self.sink(*item)
            except Exception:
                pass

    def close(self, timeout: float = 5.0) -> None:
        try:
            self.q.put(None)
        except Exception:
            pass
        self.thread.join(timeout)


# ---------------------------------------------------------------- the snapshot file

def _natural(text: str) -> list:
    """A sort key that puts 'replicate 2' before 'replicate 10'."""
    return [(0, int(t), "") if t.isdigit() else (1, 0, t) for t in re.split(r"(\d+)", text)]


class Board:
    """A snapshot of a run in a small text file, rewritten whenever something changes: a headline
    set by the script, and the latest line of every piece of work still running. Writing never
    stops the run: if the file cannot be written (held by another program, say), the next change
    tries again."""

    def __init__(self, path, title: str):
        self.path, self.title, self.t0 = path, title, time.time()
        self.lock = threading.Lock()
        self.head, self.pieces = "", {}

    def headline(self, text: str) -> None:
        with self.lock:
            self.head = text
            self._write()

    def clear(self) -> None:
        """Forget the running pieces (after a pool of workers has been stopped)."""
        with self.lock:
            self.pieces.clear()
            self._write()

    def note(self, where: str, message: str, event: str) -> None:
        """Take one progress event (the sink's arguments)."""
        with self.lock:
            now = time.time()
            if event == "end":
                for k in [k for k in self.pieces if k == where or k.startswith(where + " / ")]:
                    del self.pieces[k]
            else:
                first = self.pieces.get(where, (now, now, ""))[0]
                self.pieces[where] = (first, now, message)
            self._write()

    def text(self) -> str:
        now = time.time()
        rows = [self.title, f"Updated {time.strftime('%Y-%m-%d %H:%M:%S')}; this session started "
                            f"{duration(now - self.t0)} ago.", ""]
        if self.head:
            rows += [self.head, ""]
        if self.pieces:
            rows.append("Working on:")
            for k, (first, last, msg) in sorted(self.pieces.items(), key=lambda kv: (int(kv[1][0]), _natural(kv[0]))):
                rows.append(f"  - {k or 'main'}, for {duration(now - first)} (last line {duration(now - last)} ago): {msg}")
        return "\n".join(rows) + "\n"

    def _write(self) -> None:
        try:
            with open(self.path, "w", encoding="utf-8") as fh:
                fh.write(self.text())
        except OSError:
            pass

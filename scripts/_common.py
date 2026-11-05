"""Shared helpers for the run scripts: the import path, the data-use guards and what a report
records about the code and inputs that produced it."""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DATA = ROOT / "data"
RESULTS = ROOT / "results"


def require_inside(path: Path, folder: Path, what: str) -> Path:
    path = Path(path).resolve()
    if folder.resolve() not in path.parents:
        sys.exit(f"Refusing: {what} must be inside {folder} (got {path}).")
    return path


def require_approval(reference: str | None, what: str) -> None:
    """Commands that read PPMI participant data wait for approval (protocol §14)."""
    if not reference or len(reference.strip()) < 8:
        sys.exit(f"Refusing to run {what}: it reads PPMI participant data, which needs the investigator's "
                 "written approval (protocol §14). Pass --approval '<who, date, document>' once it is given.")


def to_json(obj):
    """JSON-safe values: numpy arrays and scalars to lists and floats, non-finite numbers to null."""
    import math

    import numpy as np
    if isinstance(obj, dict):
        return {str(k): to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_json(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        return float(obj) if math.isfinite(obj) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def code_version() -> dict:
    """The code a run uses: the commit, a hash of uncommitted changes to tracked files, and any
    untracked files (so a report can be tied to a later commit)."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout
    try:
        diff = git("diff", "HEAD")
    except OSError:                                    # git not installed (a downloaded copy)
        return {"commit": None, "uncommitted_diff_sha256": None, "untracked": None}
    return {"commit": git("rev-parse", "HEAD").strip(),
            "uncommitted_diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff else None,
            "untracked": git("ls-files", "--others", "--exclude-standard").split()}


def file_sha256(path) -> str:
    """A file's SHA-256, to tie a report or checkpoint to its exact inputs (the hash reveals nothing
    about the contents)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def require_hash_seed() -> None:
    """Step 1's FCI visits sets in hash order, so a run is reproducible only with a fixed hash seed
    (audit A8); the parallel workers inherit it from the environment."""
    if os.environ.get("PYTHONHASHSEED") != "0":
        sys.exit("Refusing: set PYTHONHASHSEED=0 first, so that Step 1's support selection is reproducible "
                 "(macOS/Linux: export PYTHONHASHSEED=0; Windows PowerShell: $env:PYTHONHASHSEED = \"0\").")

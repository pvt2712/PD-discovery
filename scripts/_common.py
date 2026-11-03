"""Shared helpers for the run scripts: the import path and the data-use guards."""
from __future__ import annotations

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

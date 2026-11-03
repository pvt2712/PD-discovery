"""Loads the pre-specified constants (configs/protocol_v0.5.toml)."""
from __future__ import annotations

import tomllib
from pathlib import Path

DEFAULT = Path(__file__).resolve().parents[2] / "configs" / "protocol_v0.5.toml"


def load(path: str | Path | None = None) -> dict:
    with open(path or DEFAULT, "rb") as fh:
        return tomllib.load(fh)

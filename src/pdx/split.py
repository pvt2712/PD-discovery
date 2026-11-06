"""The training, validation and held-out sets of Paper 2's protocol (escalation-study-v2/PROTOCOL.md §3.2; decision
D33 here): H-P and its speed estimands run on the development participants only (training and validation).

The rule reads no data. For each PPMI participant number (PATNO):

    u = first 15 hexadecimal digits of SHA-256("ppmi-paper2-split-v1:" + PATNO) / 16**15
    training if u < 0.60, validation if 0.60 <= u < 0.80, held-out otherwise.

It is a copy of escalation-study-v2/src/escalation_v2/split.py (the PC holds only this repository). Both copies are
tested against the same vectors (tests/test_split.py), so they cannot drift apart unnoticed.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

SALT = "ppmi-paper2-split-v1"
CUTS = (0.60, 0.80)
SETS = ("training", "validation", "heldout")
DEVELOPMENT = ("training", "validation")
NAMED = {"development": DEVELOPMENT, "all": SETS, "training": ("training",), "validation": ("validation",),
         "heldout": ("heldout",)}


def key(patno) -> str:
    """The participant number as the rule hashes it: an integer without a decimal part, or the stripped string."""
    if isinstance(patno, (int, np.integer)):
        return str(int(patno))
    s = str(patno).strip()
    try:
        f = float(s)
    except ValueError:
        return s
    return str(int(f)) if np.isfinite(f) and f == int(f) else s


def u_value(patno) -> float:
    return int(hashlib.sha256(f"{SALT}:{key(patno)}".encode()).hexdigest()[:15], 16) / 16 ** 15


def set_of(patno) -> str:
    u = u_value(patno)
    return SETS[0] if u < CUTS[0] else SETS[1] if u < CUTS[1] else SETS[2]


def assign(patnos) -> pd.Series:
    s = pd.Series(list(patnos))
    lut = {p: set_of(p) for p in s.dropna().unique()}
    return s.map(lut)


def in_sets(patnos, sets) -> np.ndarray:
    """Boolean mask of the entries in the named sets; `sets` may be a group name of NAMED ('development', 'all')."""
    if isinstance(sets, str):
        sets = NAMED.get(sets, (sets,))
    unknown = set(sets) - set(SETS)
    if unknown:
        raise ValueError(f"unknown set(s) {sorted(unknown)}; the sets are {SETS}")
    return assign(patnos).isin(tuple(sets)).to_numpy()


def assignment_sha256(patnos) -> str:
    """SHA-256 of the sorted 'PATNO,set' lines (the hash reveals no identifier)."""
    ids = sorted({key(p) for p in pd.Series(list(patnos)).dropna()}, key=lambda k: k.zfill(12))
    return hashlib.sha256("".join(f"{k},{set_of(k)}\n" for k in ids).encode()).hexdigest()


def counts(patnos) -> dict:
    """How many distinct participants fall in each set."""
    s = assign(pd.Series(list(patnos)).dropna().unique())
    return {k: int((s == k).sum()) for k in SETS}

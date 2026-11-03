"""Panel data for the engine: one sequence of visits per participant, from the year-1 landmark on.

Scores must already be oriented (higher means worse) and on the frozen normal-scores scale
(transforms.py). Missing values are NaN. Nothing here prints or writes participant-level data;
``aggregate_summary`` returns counts only.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np


@dataclass
class Panel:
    y: list                      # per participant: (T_i, d) scores, NaN = missing; row 0 = landmark visit
    t: list                      # per participant: (T_i,) visit times in years
    group: np.ndarray            # (N,) int phenotype, 0 = reference (TD)
    z: np.ndarray | None = None  # (N, p) centred baseline covariates (era, age at onset, sex, duration)
    dose: list | None = None     # per participant: (T_i,) current dose at each visit (100 mg LEDD)
    exposure: list | None = None  # per participant: (T_i,) long-duration exposure at each visit
    on: list | None = None       # per participant: (T_i,) 1 if the examination was in the ON state
    ids: np.ndarray | None = None
    domains: list | None = None
    meta: dict = field(default_factory=dict)
    year: list | None = None     # per participant: (T_i,) visit year on the annual grid (1 = landmark)

    def __post_init__(self):
        self.group = np.asarray(self.group, int)
        n = len(self.y)
        if len(self.t) != n or len(self.group) != n:
            raise ValueError("y, t and group must have one entry per participant")
        self.y = [np.asarray(v, float).reshape(len(v), -1) for v in self.y]
        self.t = [np.asarray(v, float).reshape(-1) for v in self.t]
        for yi, ti in zip(self.y, self.t):
            if len(yi) != len(ti):
                raise ValueError("each participant needs one time per visit")
            if len(ti) > 1 and np.any(np.diff(ti) <= 0):
                raise ValueError("visit times must increase")
        if self.z is None:
            self.z = np.zeros((n, 0))
        self.z = np.asarray(self.z, float).reshape(n, -1)
        for name in ("dose", "exposure", "on"):
            v = getattr(self, name)
            if v is None:
                setattr(self, name, [np.zeros(len(ti)) for ti in self.t])
            else:
                setattr(self, name, [np.asarray(a, float).reshape(-1) for a in v])
        if self.year is not None:
            self.year = [np.asarray(a, int).reshape(-1) for a in self.year]
            if any(len(a) != len(ti) for a, ti in zip(self.year, self.t)):
                raise ValueError("each participant needs one grid year per visit")

    @property
    def n(self) -> int:
        return len(self.y)

    @property
    def d(self) -> int:
        return self.y[0].shape[1]

    @property
    def p(self) -> int:
        return self.z.shape[1]

    @property
    def n_groups(self) -> int:
        return int(self.group.max()) + 1

    def subset(self, idx) -> "Panel":
        idx = np.asarray(idx, int)
        pick = lambda L: [L[i] for i in idx]
        return replace(self, y=pick(self.y), t=pick(self.t), group=self.group[idx], z=self.z[idx],
                       dose=pick(self.dose), exposure=pick(self.exposure), on=pick(self.on),
                       ids=None if self.ids is None else np.asarray(self.ids)[idx], meta=dict(self.meta),
                       year=None if self.year is None else pick(self.year))

    def with_y(self, y: list) -> "Panel":
        return replace(self, y=[np.asarray(v, float) for v in y], meta=dict(self.meta))

    def n_transitions(self) -> int:
        return int(sum(len(ti) - 1 for ti in self.t))

    def aggregate_summary(self) -> dict:
        """Counts only, by phenotype."""
        out = {}
        for g in range(self.n_groups):
            idx = np.where(self.group == g)[0]
            T = np.array([len(self.t[i]) for i in idx])
            miss = np.mean([np.isnan(self.y[i]).mean() for i in idx]) if len(idx) else np.nan
            out[str(g)] = {"participants": int(len(idx)), "visits": int(T.sum()),
                           "transitions": int((T - 1).sum()), "median_visits": float(np.median(T)) if len(T) else 0.0,
                           "share_missing": float(miss)}
        return out


def two_slice_table(panel: Panel, baseline: np.ndarray | None = None, max_dt: float = 1.5,
                    add_interval: bool = False, include_dose: bool = False, drop_incomplete: bool = False):
    """Pooled year-to-year transitions for Step 1 (protocol v0.5, §6.2, §6.3).

    Rows are (baseline, [interval], x_t, [dose_t], [dose_t+1], x_t+1) for consecutive grid years
    (a step across a missed year is not used). With grid years in the panel, consecutive means a
    year difference of exactly 1; without them, an interval of at most max_dt years. Returns
    (table, tiers, rows_participant, columns):
    - tiers gives each column's temporal tier: baseline 0, year t 1, treatment at t+1 2 (if
      included), the rest of year t+1 last; the caller may split the baseline tier further;
    - rows_participant gives the participant index of each row (the resampling unit);
    - columns names each column ("b0", ..., "interval", "y<j>_t", "dose_t", "dose_t1", "y<j>_t1").
    Rows with missing values are kept for test-wise deletion unless drop_incomplete.
    """
    b = 0 if baseline is None else baseline.shape[1]
    d = panel.d
    rows, who = [], []
    for i in range(panel.n):
        yi, ti = panel.y[i], panel.t[i]
        yr = None if panel.year is None else panel.year[i]
        for k in range(len(yi) - 1):
            dt = ti[k + 1] - ti[k]
            consecutive = (yr[k + 1] - yr[k] == 1) if yr is not None else dt <= max_dt
            if not consecutive:
                continue
            r = [baseline[i] if b else np.zeros(0)]
            if add_interval:
                r.append([dt])
            r.append(yi[k])
            if include_dose:
                r.append([panel.dose[i][k], panel.dose[i][k + 1]])
            r.append(yi[k + 1])
            rows.append(np.concatenate(r))
            who.append(i)
    cols = [f"b{j}" for j in range(b)] + (["interval"] if add_interval else [])
    tiers = [0] * len(cols)
    cols += [f"y{j}_t" for j in range(d)]
    tiers += [1] * d
    if include_dose:
        cols += ["dose_t", "dose_t1"]
        tiers += [1, 2]
    last = max(tiers) + 1
    cols += [f"y{j}_t1" for j in range(d)]
    tiers += [last] * d
    X = np.array(rows, float).reshape(len(rows), len(cols))
    who = np.array(who, int)
    if drop_incomplete:
        keep = ~np.isnan(X).any(1)
        X, who = X[keep], who[keep]
    return X, np.array(tiers), who, cols


def save_panel(panel: Panel, path) -> None:
    """Save a panel without pickling (flat arrays plus offsets). Participant-level: only ever
    inside the git-ignored data folder."""
    lens = np.array([len(t) for t in panel.t], int)
    extra = {} if panel.year is None else {"year": np.concatenate(panel.year)}
    np.savez_compressed(path, lens=lens, y=np.concatenate(panel.y), t=np.concatenate(panel.t),
                        group=panel.group, z=panel.z, dose=np.concatenate(panel.dose),
                        exposure=np.concatenate(panel.exposure), on=np.concatenate(panel.on),
                        domains=np.array(panel.domains or [], dtype=str), **extra)


def load_panel(path) -> Panel:
    f = np.load(path, allow_pickle=False)
    cuts = np.cumsum(f["lens"])[:-1]
    split = lambda a: np.split(a, cuts)
    doms = [str(v) for v in f["domains"]] or None
    return Panel(split(f["y"]), split(f["t"]), f["group"], z=f["z"], dose=split(f["dose"]),
                 exposure=split(f["exposure"]), on=split(f["on"]), domains=doms,
                 year=split(f["year"]) if "year" in f.files else None)

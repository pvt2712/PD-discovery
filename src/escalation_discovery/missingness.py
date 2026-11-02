"""missingness — PPMI-like missing values for simulated panels (Arm A, PROTOCOL.md §6.2).

Follow-up length (dropout) comes from the generator; this module masks values inside the observed years.

- 'complete': nothing masked.
- 'mar' (realistic): masks are drawn per participant-year from a pattern bank stratified by cohort, years since
  enrolment (0, 1-2, 3-5, 6+) and levodopa status, so the main driver of PPMI's missingness (the practically-defined
  OFF exam is skipped on treatment) and its joint structure (one exam, several nodes) are kept. `PatternBank.fit`
  builds the bank from PPMI participant-years (participant-level by provenance: it stays under data/derived/);
  `PatternBank.synthetic` is the parametric bank used without PPMI.
- 'mnar': 'mar' plus missingness that depends on the value itself for the assessments a worse state plausibly
  prevents (OFF motor exam, axial score, MoCA, GDS-15, Epworth: a value in the worse quartile is masked with extra
  probability 0.25) and informative dropout (after a year with Schwab & England below 80%, the disability threshold
  of Paper 2's illness-death model, each later year is lost with probability 0.25).
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import expert as E

# Values that are worse when lower (all other scales: worse when higher)
LOWER_IS_WORSE = {"cog", "se", "dat0"}
# Assessments whose missingness depends on the value itself in the 'mnar' setting
MNAR_NODES = ("motor_na", "axial", "cog", "dep", "ess")
# Nodes measured in one exam and therefore missing together
BLOCKS = {"motor_na": "off_exam", "axial": "off_exam", "fluct": "part4", "dysk": "part4"}


def _year_bin(y):
    y = np.asarray(y)
    return np.select([y == 0, y <= 2, y <= 5], [0, 1, 2], 3)


@dataclass
class PatternBank:
    tv: list
    baseline: list
    masks: dict = field(default_factory=dict)       # stratum -> boolean array (participant-years x tv), True = missing
    base_masks: dict = field(default_factory=dict)  # cohort -> boolean array (participants x baseline)
    rates: dict | None = None                       # synthetic: per-node rate (off-levodopa, on-levodopa)
    base_rates: dict | None = None
    source: str = "synthetic"

    # ------------------------------------------------------------------ construction
    @classmethod
    def synthetic(cls, spec: E.ExpertSpec):
        rates = {v: (0.03, 0.04) for v in spec.tv}
        rates.update({"motor_na": (0.06, 0.32), "axial": (0.06, 0.32), "adl": (0.008, 0.008), "se": (0.01, 0.01),
                      "cog": (0.05, 0.07), "fluct": (0.04, 0.06), "dysk": (0.04, 0.06), "hall": (0.01, 0.02)})
        for v in E.DOSE + ("t_ldopa", "cum_ldopa"):
            if v in rates:
                rates[v] = (0.0, 0.0)
        base_rates = {b: 0.0 for b in spec.baseline}
        base_rates.update({"dat0": 0.05, "apoe4": 0.08, "educ": 0.01})
        return cls(tv=list(spec.tv), baseline=list(spec.baseline), rates=rates, base_rates=base_rates)

    @classmethod
    def fit(cls, long: pd.DataFrame, base: pd.DataFrame, spec: E.ExpertSpec):
        """Bank of real masks (PPMI long panel with PATNO, year, era2, genetic and the nodes)."""
        tv = [v for v in spec.tv if v in long]
        on = (long["ldopa"].fillna(0) > 0).to_numpy().astype(int)
        coh = (long["era2"].astype(int).astype(str) + long["genetic"].astype(int).astype(str)).to_numpy()
        yb = _year_bin(long["year"].to_numpy())
        M = long[tv].isna().to_numpy()
        masks = {}
        for key in set(zip(coh, yb, on)):
            sel = (coh == key[0]) & (yb == key[1]) & (on == key[2])
            masks[key] = M[sel]
        bcoh = (base["era2"].astype(int).astype(str) + base["genetic"].astype(int).astype(str)).to_numpy()
        bl = [b for b in spec.baseline if b in base]
        base_masks = {c: base.loc[bcoh == c, bl].isna().to_numpy() for c in set(bcoh)}
        return cls(tv=tv, baseline=bl, masks=masks, base_masks=base_masks, source="PPMI pattern bank")

    # ------------------------------------------------------------------ application
    def apply(self, long: pd.DataFrame, base: pd.DataFrame, mechanism: str, seed: int):
        """Masked copies of (long, base)."""
        if mechanism == "complete":
            return long.copy(), base.copy()
        rng = np.random.default_rng(seed)
        long, base = long.copy(), base.copy()
        if mechanism == "mnar":
            long = _informative_dropout(long, rng)
        b = base.set_index("PATNO")
        coh = (b.era2.astype(int).astype(str) + b.genetic.astype(int).astype(str)).reindex(long.PATNO).to_numpy()
        on = (long["ldopa"].fillna(0) > 0).to_numpy().astype(int)
        yb = _year_bin(long["year"].to_numpy())
        tv = [v for v in self.tv if v in long]
        if self.rates is not None:                                   # parametric bank
            M = np.zeros((len(long), len(tv)), dtype=bool)
            shared = {}
            year = long["year"].to_numpy()
            for j, v in enumerate(tv):
                r0, r1 = self.rates[v]
                p = np.where(on == 1, r1, r0) * (1 + 0.03 * year)
                blk = BLOCKS.get(v)
                if blk:
                    if blk not in shared:
                        shared[blk] = rng.random(len(long))
                    M[:, j] = shared[blk] < p
                else:
                    M[:, j] = rng.random(len(long)) < p
            bmask = {k: rng.random(len(base)) < r for k, r in self.base_rates.items() if k in base}
        else:                                                        # resampled real patterns
            M = np.zeros((len(long), len(tv)), dtype=bool)
            keys = list(self.masks)
            for key in set(zip(coh, yb, on)):
                sel = np.nonzero((coh == key[0]) & (yb == key[1]) & (on == key[2]))[0]
                pool = self.masks.get(key)
                if pool is None or len(pool) == 0:      # nearest stratum: same cohort and treatment, any year
                    cand = [k for k in keys if k[0] == key[0] and k[2] == key[2] and len(self.masks[k])]
                    pool = np.vstack([self.masks[k] for k in cand]) if cand else np.zeros((1, len(self.tv)), bool)
                rows = pool[rng.integers(len(pool), size=len(sel))]
                M[sel] = rows[:, [self.tv.index(v) for v in tv]]
            bcoh = (base.era2.astype(int).astype(str) + base.genetic.astype(int).astype(str)).to_numpy()
            bmask = {k: np.zeros(len(base), bool) for k in self.baseline if k in base}
            for c in set(bcoh):
                sel = np.nonzero(bcoh == c)[0]
                pool = self.base_masks.get(c)
                if pool is None or len(pool) == 0:
                    continue
                rows = pool[rng.integers(len(pool), size=len(sel))]
                for j, k in enumerate(self.baseline):
                    if k in bmask:
                        bmask[k][sel] = rows[:, j]
        if mechanism == "mnar":
            for j, v in enumerate(tv):
                if v not in MNAR_NODES:
                    continue
                x = long[v].to_numpy(dtype=float)
                cut = np.nanquantile(x, 0.25 if v in LOWER_IS_WORSE else 0.75)
                worse = x <= cut if v in LOWER_IS_WORSE else x >= cut
                M[:, j] |= worse & (rng.random(len(long)) < 0.25)
        for j, v in enumerate(tv):
            col = long[v].to_numpy(dtype=float).copy()
            col[M[:, j]] = np.nan
            long[v] = col
        for k, m in bmask.items():
            col = base[k].to_numpy(dtype=float).copy()
            col[m] = np.nan
            base[k] = col
        return long.reset_index(drop=True), base


def _informative_dropout(long, rng, threshold=80.0, p=0.25):
    """After a year with Schwab & England below `threshold` %, each later year is lost with probability p."""
    if "se" not in long:
        return long
    long = long.sort_values(["PATNO", "year"]).reset_index(drop=True)
    bad = (long.se.to_numpy(dtype=float) < threshold).astype(int)
    seen = pd.Series(bad).groupby(long.PATNO).cumsum().shift(1, fill_value=0).to_numpy().copy()   # bad year before this one
    first = long.groupby("PATNO").cumcount().to_numpy() == 0
    seen[first] = 0
    lost = (seen > 0) & (rng.random(len(long)) < p)
    gone = pd.Series(lost.astype(int)).groupby(long.PATNO).cummax().to_numpy().astype(bool)
    return long[~gone].reset_index(drop=True)

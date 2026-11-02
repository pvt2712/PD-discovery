"""Paths and shared constants, read from pyproject.toml [tool.escalation_discovery].

Ported from escalation-study-v2/src/escalation_v2/config.py (29 October 2026); added: the Tetrad and Java settings,
the cohort context nodes of the pooled analysis, and the PPMI approval gate (PROTOCOL.md §11).

Participant-level tables live only under `derived_dir()` (gitignored); everything written under `results_dir()` or
`docs/` must be an aggregate (PPMI data-use agreement). Fitted generators, missingness pattern banks and simulated
panels derived from PPMI are participant-level by provenance and also stay under `derived_dir()`.
"""
import glob
import os
import tomllib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def _settings():
    with open(ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)["tool"]["escalation_discovery"]


def raw_dir() -> Path:
    return (ROOT / _settings()["raw"]).resolve()


def derived_dir(sub: str | None = None) -> Path:
    p = ROOT / _settings()["derived"]
    if sub:
        p = p / sub
    p.mkdir(parents=True, exist_ok=True)
    return p


def results_dir(sub: str | None = None) -> Path:
    p = ROOT / _settings()["results"]
    if sub:
        p = p / sub
    p.mkdir(parents=True, exist_ok=True)
    return p


def docs_dir() -> Path:
    return ROOT / "docs"


def tetrad_jar() -> Path:
    return (ROOT / _settings()["tetrad_jar"]).resolve()


def java_home() -> str:
    return os.environ.get("JAVA_HOME") or _settings()["java_home"]


def ppmi_runs_approved() -> bool:
    return bool(_settings().get("ppmi_runs_approved", False))


def require_ppmi_approval(what: str):
    """Stop before any step that reads PPMI participant data until the protocol is approved (PROTOCOL.md §11)."""
    if not ppmi_runs_approved():
        raise SystemExit(f"{what} reads PPMI participant data. It is blocked until the protocol is approved: set "
                         "ppmi_runs_approved = true in pyproject.toml [tool.escalation_discovery] and date the change "
                         "in PROTOCOL.md §12.")


def read_raw(folder: str, name: str) -> pd.DataFrame:
    """Read `<raw>/<folder>/<name>_<DDMonYYYY>.csv` (the PPMI download suffix) or `<name>.csv`."""
    pats = [os.path.join(raw_dir(), folder, name + "_[0-9][0-9][A-Z][a-z][a-z][0-9][0-9][0-9][0-9].csv"),
            os.path.join(raw_dir(), folder, name + ".csv")]
    for pat in pats:
        hit = sorted(glob.glob(pat))
        if hit:
            return pd.read_csv(hit[-1], low_memory=False)
    raise FileNotFoundError(f"{folder}/{name} not found under {raw_dir()}")


def month_date(s: pd.Series) -> pd.Series:
    """PPMI dates are MM/YYYY; every date is placed on the 15th of its month."""
    return pd.to_datetime("15/" + s.astype("string"), format="%d/%m/%Y", errors="coerce")


# Cohorts: key -> label used in tables (pooled in the discovery analyses, PROTOCOL.md §3)
COHORTS = {"ppmi1": "PPMI 1.0 de novo", "ppmi2": "PPMI 2.0 sporadic", "genetic": "genetic PD"}
# Context nodes of the pooled analysis (joint causal inference: exogenous, may point into any node)
CONTEXT = {"era2": "enrolled in the PPMI 2.0 era (2020 onwards)", "genetic": "enrolled in a genetic PD arm"}

# Time-varying nodes of the annual grid: short name -> column of visits.csv
NODES = {
    "motor": "updrs3_off", "adl": "updrs2", "se": "schwab_england", "cog": "moca", "dep": "gds",
    "aut": "scopa_aut", "ess": "ess", "rbd": "rbdsq", "icd": "quip", "cx": "updrs4", "oh": "bp_drop",
    "ldopa": "ledd_levodopa", "da": "ledd_agonist", "maob": "ledd_maob", "ama": "ledd_amantadine",
    "comt": "ledd_comt", "fluct": "fluct", "dysk": "dysk", "hall": "np1_hall", "motor_na": "motor_nonaxial_off",
    "axial": "axial_off", "icd_core": "quip_core", "t_ldopa": "months_on_ldopa", "cum_ldopa": "cum_ldopa",
    "hy": "hy_off", "weight": "weight", "response": "ldopa_response", "motor_on": "updrs3_on",
    "ledd_total": "ledd_total",
}
# The nodes whose availability decides which visit represents a grid year (the core clinical battery)
CORE_NODES = ["motor", "adl", "se", "cog", "dep", "aut", "ess", "rbd", "icd", "cx", "oh", "ldopa", "da"]
# Baseline covariates: short name -> column of subjects.csv
BASELINE = {"age": "age_enroll", "male": "male", "educ": "educ_years", "apoe4": "apoe4", "dat0": "bl_dat_putamen",
            "dur0": "disease_dur_months"}
DOSE_NODES = ["ldopa", "da", "maob", "ama"]
MONTH = 30.4375

import json

import numpy as np
import pandas as pd

from pdx import wp0


def _table(n=203, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        ph = "indeterminate" if i < 3 else ("PIGD" if i % 4 == 0 else "TD")
        cohort = "PPMI 2.0" if i == 5 else "PPMI 1.0"
        for y in range(4 if i % 7 else 1):
            rows.append({"participant": f"P{i:03d}", "cohort": cohort, "year": y,
                         "date_years": y + rng.uniform(-0.1, 0.1), "phenotype": ph, "td_pigd_ratio": rng.lognormal(),
                         "site": i % 5, "mode": "remote" if (y == 1 and i < 2) else "in person", "treated": 0,
                         "dose_change": 0, "motor_state": "OFF", "tremor": rng.normal(),
                         "axial": np.nan if y == 2 else rng.normal()})
    return pd.DataFrame(rows)


def _suppressed(v):
    return isinstance(v, str)


def _check_cross_table(t):
    """No suppressed cell is recoverable: every row and column with a suppressed cell has its
    margin suppressed, and at least two cells are suppressed if any are."""
    cells = t["cells"]
    rows = list(cells)
    cols = list(cells[rows[0]])
    n_sup = sum(_suppressed(cells[r][c]) for r in rows for c in cols)
    assert n_sup != 1
    for r in rows:
        if any(_suppressed(cells[r][c]) for c in cols):
            assert _suppressed(t["row_totals"][r])
    for c in cols:
        if any(_suppressed(cells[r][c]) for r in rows):
            assert _suppressed(t["column_totals"][c])


def test_audit_is_aggregate_and_suppressed_cells_cannot_be_recovered():
    rep = wp0.audit(_table(), wp0.Columns(domains=["tremor", "axial"]), min_cell=5)
    text = json.dumps(rep)
    assert "P0" not in text                                     # no identifiers
    assert rep["participant_flow"]["total"] == 203
    _check_cross_table(rep["participant_flow"])
    _check_cross_table(rep["landmark"])
    assert sum(_suppressed(v) for v in rep["visits_per_participant"].values()) != 1
    assert isinstance(rep["missing_share"]["tremor"]["by_mode"]["remote"], str)   # denominator 2
    assert "sites_by_participants" in rep["sites"]
    assert rep["intervals"]["step1_annual_tables_ok"] is True


def test_large_cells_are_shown_exactly():
    rep = wp0.audit(_table(n=400, seed=1).query("phenotype != 'indeterminate' and cohort == 'PPMI 1.0'"),
                    wp0.Columns(domains=["tremor"]), min_cell=5)
    flow = rep["participant_flow"]
    assert all(isinstance(v, int) for v in flow["column_totals"].values())

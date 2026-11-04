"""Guide §7.17: smoke_test.synthetic_visit_table on a small simulated panel (synthetic only).

    OMP_NUM_THREADS=1 ../ppmi-causal/.venv/bin/python docs/guide_examples/s07/ex_smoke_visit_table.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
import smoke_test as S  # noqa: E402
from pdx import wp0  # noqa: E402
from pdx.sim import scenarios  # noqa: E402

panel, info = scenarios.generate("S2", 40, np.random.default_rng(7), d=4)
df = S.synthetic_visit_table(panel, np.random.default_rng(0))
print("panel:", panel.n, "participants; table:", df.shape[0], "rows x", df.shape[1], "columns")
print("columns:", df.columns.tolist())
cols = ["participant", "cohort", "year", "date_years", "phenotype", "treated", "motor_state", "d0", "d3"]
print(df[df.participant == "S0001"][cols].round(3).to_string(index=False))
print("rows per year:", df.groupby("year").size().to_dict())
rep = wp0.audit(df, wp0.Columns(domains=[f"d{j}" for j in range(panel.d)]))
print("audit sections:", list(rep))
print("participant flow:", rep["participant_flow"])

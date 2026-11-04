"""Sections 1.36-1.37: domain_summaries and audit on a simulated visit table (made-up participants)."""
import json

import numpy as np
import pandas as pd
from _setup import TMP  # noqa: F401  (import paths)

from pdx import wp0

rng = np.random.default_rng(3)
rows = []
for i in range(160):
    ph = "indeterminate" if i % 10 == 0 else ("PIGD" if i % 3 == 0 else "TD")
    cohort = "PPMI 2.0" if i % 4 == 0 else "PPMI 1.0"
    years = [0] if i % 9 == 0 else ([0, 1, 3] if i % 7 == 0 else [0, 1, 2, 3])      # some never reach year 1
    ratio = {"TD": rng.uniform(1.0, 4.0), "PIGD": rng.uniform(0.0, 1.05), "indeterminate": rng.uniform(0.9, 1.15)}[ph]
    for y in years:
        rows.append({"participant": f"P{i:03d}", "cohort": cohort, "year": y,
                     "date_years": y + (rng.uniform(-0.3, 0.3) if y else 0.0), "phenotype": ph,
                     "td_pigd_ratio": ratio if i % 13 else np.inf, "site": i % 23 if i < 150 else 99,
                     "mode": "remote" if (cohort == "PPMI 2.0" and y == 2) else "in person",
                     "treated": int(y >= 2), "dose_change": float(y >= 2 and rng.uniform() < 0.4) if y else np.nan,
                     "motor_state": "OFF" if y < 2 else rng.choice(["OFF", "ON only", "missing"]),
                     "tremor": rng.normal() if rng.uniform() > 0.05 else np.nan,
                     "hallucinations": float(rng.uniform() < 0.1),
                     "phenotype_y1": ph if rng.uniform() < 0.8 else "indeterminate", "phenotype_y2": None,
                     "event_24m": float(rng.uniform() < 0.03) if y == 0 else np.nan})
df = pd.DataFrame(rows)
cols = wp0.Columns(domains=["tremor", "hallucinations"])
rep = wp0.audit(df, cols, min_cell=5)

print("== 1.36 domain_summaries ==")
print(json.dumps(rep["domain_summaries"]["year_0"], indent=None))

print("\n== 1.37 audit ==")
for key in ("participant_flow", "landmark", "transitions_after_landmark", "intervals", "visits_per_participant",
            "mode", "motor_state", "sites", "overlap", "phenotype_stability", "part2_events"):
    print(f"{key}: {json.dumps(rep[key])}")
print("missing_share tremor:", json.dumps(rep["missing_share"]["tremor"]))
print("missing_share_by_motor_state tremor:", json.dumps(rep["missing_share_by_motor_state"]["tremor"]))
print("an identifier anywhere in the output:", "P0" in json.dumps(rep))

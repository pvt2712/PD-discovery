"""Build the study tables from the local PPMI download (protocol v0.5, §4, §5, §6.1; WP0).

Guarded: it reads participant-level data, so it refuses to run without the investigator's
written approval. Participant-level tables go to data/derived/ (git-ignored); the participant
flow (counts, small cells suppressed) and a provenance manifest (download date, file hashes,
counts; §14) go to results/wp0/. It prints counts only.

    python scripts/build_ppmi_tables.py --approval "<who, date, document>" [--raw ../PPMI/raw]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re

import _common
import pandas as pd

from pdx import config, ppmi, wp0

USED = ["Participant_Status", "Demographics", "PD_Diagnosis_History", "Conclusion_of_Study_Participation",
        "MDS-UPDRS_Part_III", "MDS_UPDRS_Part_II__Patient_Questionnaire", "MDS-UPDRS_Part_I",
        "MDS-UPDRS_Part_IV__Motor_Complications", "Modified_Schwab___England_Activities_of_Daily_Living",
        "Montreal_Cognitive_Assessment__MoCA_", "Geriatric_Depression_Scale__Short_Version_", "SCOPA-AUT",
        "Epworth_Sleepiness_Scale", "REM_Sleep_Behavior_Disorder_Screening_Questionnaire",
        "LEDD_Concomitant_Medication_Log"]


def manifest(raw, part, wp0_table) -> dict:
    files = {}
    for name in USED:
        for f in glob.glob(os.path.join(raw, "*", name + "_*.csv")):
            if re.search(r"_\d\d[A-Z][a-z]{2}\d{4}\.csv$", f):
                h = hashlib.sha256(open(f, "rb").read()).hexdigest()
                files[os.path.basename(f)] = h
    stamps = sorted({re.search(r"_(\d\d[A-Z][a-z]{2}\d{4})\.csv$", f).group(1) for f in files})
    return {"download_stamps": stamps, "files_sha256": files, "pd_cohort": int(len(part)),
            "primary_population": int(part.primary.sum()), "grid_rows_primary": int(len(wp0_table)),
            "rules": "see src/pdx/ppmi.py docstring (provisional choices fixed in WP1)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ppmi.RAW))
    ap.add_argument("--approval", default=None)
    ap.add_argument("--min-cell", type=int, default=5)
    a = ap.parse_args()
    _common.require_approval(a.approval, "the PPMI table build")
    derived = _common.DATA / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    out = _common.RESULTS / "wp0"
    out.mkdir(parents=True, exist_ok=True)
    treatment = config.load()["treatment"]
    tables = ppmi.build(a.raw, treatment["ldr_half_life_days"])
    part, grid, w = tables["participants"], tables["grid"], tables["wp0"]
    part.to_parquet(derived / "participants.parquet")
    grid.to_parquet(derived / "grid.parquet")
    w.to_parquet(derived / "wp0_visits.parquet")
    flow = wp0.protect_2way(ppmi.flow_table(part), a.min_cell)
    (out / "participant_flow.json").write_text(json.dumps(flow, indent=1))
    prim = part[part.primary]
    axes = {"motor_phenotype_by_clinical_subtype": wp0.protect_2way(
        pd.crosstab(prim.phenotype, prim.clinical_subtype.fillna("not classifiable")), a.min_cell),
        "clinical_subtype_by_cohort": wp0.protect_2way(
        pd.crosstab(prim.clinical_subtype.fillna("not classifiable"), prim.era), a.min_cell)}
    (out / "phenotype_axes.json").write_text(json.dumps(axes, indent=1))
    retest = ppmi.screening_retest(tables["visits"], part, grid)
    (out / "retest_reliability.json").write_text(json.dumps(retest, indent=1))
    man = manifest(a.raw, part, w)
    man["approval"] = a.approval
    man["treatment"] = treatment
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    print(f"PD cohort {len(part)}; primary population {int(part.primary.sum())}; "
          f"grid rows {len(w)}; wrote data/derived/*.parquet and results/wp0/participant_flow.json, manifest.json")


if __name__ == "__main__":
    main()

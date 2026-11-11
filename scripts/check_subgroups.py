"""Can PPMI support H-P with other groupings than TD/PIGD? Aggregate counts on Paper 2's development
participants (D33).

Guarded: it reads participant-level tables (data/derived/participants.parquet and grid.parquet, and three
files of the raw download: the UPSIT smell test, the genetic consensus and the DaTscan binding ratios), so it
refuses to run without the investigator's written approval. It keeps only the development participants
before anything else, so the held-out participants stay unread, and prints and writes aggregates only, with
cells under 5 suppressed (WP0's rule). Output: results/wp0/subgroup_check.json.

Population: the primary population (sporadic PD, untreated at baseline, with a usable baseline examination)
seen at the year-1 landmark with at least one transition after it, as H-P's panel, but whatever the motor
phenotype (the panel itself keeps TD and PIGD only). Groupings come from baseline (year-0) values:

  age at onset (median split; also onset before 50), sex, era (PPMI 1.0 / 2.0),
  TD / PIGD / indeterminate (Stebbins 2013, the protocol's phenotype),
  akinetic-rigid / mixed / TD (Schiess 2000 ratio on MDS-UPDRS III: mean tremor item over mean
      bradykinesia-rigidity item; TD >= 1.0, AR <= 0.80),
  probable RBD (RBDSQ >= 5; >= 6 as the PD-specific cut), MoCA < 26, depressive symptoms (GDS-15 >= 5),
  autonomic burden (SCOPA-AUT above its median), the Fereshtehnejad clinical subtype,
  smell (UPSIT 40 items: anosmia at 18 or fewer correct; median split), GBA and LRRK2 carriers (genetic
  consensus file), putamen DaTscan binding (median split).

For labels measured again at later visits (TD/PIGD, AR/TD, RBDSQ, MoCA, GDS) it also gives how often the
year-0 label is still the label at years 1 and 2.

    python scripts/check_subgroups.py --approval "<who, date, instruction>"
"""
from __future__ import annotations

import argparse
import json

import _common
import numpy as np
import pandas as pd

from pdx import ppmi, split

MIN_CELL = 5
RAW = ppmi.RAW


def sup(n: int):
    n = int(n)
    return n if n >= MIN_CELL else "<5"


def raw_csv(sub: str, stem: str) -> pd.DataFrame:
    paths = sorted((RAW / sub).glob(stem + "*.csv"))
    if not paths:
        raise FileNotFoundError(f"{sub}/{stem}*.csv")
    return pd.read_csv(paths[-1], low_memory=False)


def baseline_rows(df: pd.DataFrame, events=("BL", "SC")) -> pd.DataFrame:
    """One row per participant: the baseline record, else the screening one."""
    df = df[df.EVENT_ID.isin(events)].copy()
    df["order"] = df.EVENT_ID.map({e: i for i, e in enumerate(events)})
    return df.sort_values(["PATNO", "order"]).drop_duplicates("PATNO").set_index("PATNO")


def ar_label(v: pd.DataFrame) -> pd.Series:
    """Schiess 2000 on MDS-UPDRS III: mean tremor item over mean bradykinesia-rigidity item."""
    trem = v[ppmi.TREMOR_III].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    ar = pd.to_numeric(v["brady_rigidity"], errors="coerce") / len(ppmi.BRADY_RIGID)
    ratio = trem / ar.where(ar > 0)
    lab = np.select([ratio >= 1.0, ratio <= 0.80, ratio.notna()], ["TD", "akinetic-rigid", "mixed"], None)
    lab = np.where((ar == 0) & (trem > 0), "TD", lab)
    return pd.Series(lab, index=v.index, dtype=object).where(trem.notna() & ar.notna())


def label_at(grid: pd.DataFrame, year: int, fn) -> pd.Series:
    v = grid[grid.year == year].drop_duplicates("PATNO").set_index("PATNO")
    return fn(v)


def stability(grid: pd.DataFrame, ids, fn) -> dict:
    y0 = label_at(grid, 0, fn).reindex(ids)
    out = {}
    for yr in (1, 2):
        yk = label_at(grid, yr, fn).reindex(ids)
        both = y0.notna() & yk.notna()
        same = (y0[both] == yk[both])
        out[f"same_label_year_{yr}"] = {str(k): {"n": sup((y0[both] == k).sum()),
                                                  "share_unchanged": round(float(same[y0[both] == k].mean()), 2)
                                                  if (y0[both] == k).sum() >= MIN_CELL else None}
                                        for k in sorted(y0[both].dropna().unique())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--approval", default=None)
    a = ap.parse_args()
    _common.require_approval(a.approval, "the subgroup check")
    d = _common.DATA / "derived"
    part = pd.read_parquet(d / "participants.parquet")
    part = part[split.in_sets(part.PATNO, "development")].set_index("PATNO")       # held-out rows dropped here
    grid = pd.read_parquet(d / "grid.parquet")
    grid = grid[grid.PATNO.isin(part.index)]

    # the analysis population, whatever the phenotype: primary, from the year-1 landmark, >= 1 transition
    g1 = grid[grid.year >= 1].sort_values(["PATNO", "year"])
    first = g1.groupby("PATNO").year.min()
    nvis = g1.groupby("PATNO").size()
    pop = part.index[part.primary & part.reached_y1]
    pop = pd.Index([p for p in pop if first.get(p) == 1 and nvis.get(p, 0) >= 2])
    trans = (nvis.reindex(pop) - 1).clip(lower=0)
    b0 = grid[grid.year == 0].drop_duplicates("PATNO").set_index("PATNO").reindex(pop)
    P = part.reindex(pop)

    groups = {}
    med_onset = float(P.age_onset.median())
    groups["age at onset (median split)"] = pd.Series(np.where(P.age_onset < med_onset, "younger onset", "older onset"),
                                                     index=pop).where(P.age_onset.notna())
    groups["onset before 50"] = pd.Series(np.where(P.age_onset < 50, "before 50", "50 or later"), index=pop).where(P.age_onset.notna())
    groups["sex"] = P.male.map({1: "men", 1.0: "men", True: "men", 0: "women", 0.0: "women", False: "women"})
    groups["era"] = P.era.astype(str)
    groups["TD / PIGD (Stebbins)"] = P.phenotype
    groups["akinetic-rigid / TD (Schiess)"] = ar_label(b0)
    rbd = pd.to_numeric(b0.rbd, errors="coerce")
    groups["probable RBD (RBDSQ >= 5)"] = pd.Series(np.where(rbd >= 5, "RBD", "no RBD"), index=pop).where(rbd.notna())
    groups["probable RBD (RBDSQ >= 6)"] = pd.Series(np.where(rbd >= 6, "RBD", "no RBD"), index=pop).where(rbd.notna())
    moca = pd.to_numeric(b0.MCATOT, errors="coerce")
    groups["MoCA < 26"] = pd.Series(np.where(moca < 26, "MoCA < 26", "MoCA >= 26"), index=pop).where(moca.notna())
    gds = pd.to_numeric(b0.mood, errors="coerce")
    groups["depressive symptoms (GDS >= 5)"] = pd.Series(np.where(gds >= 5, "GDS >= 5", "GDS < 5"), index=pop).where(gds.notna())
    aut = pd.to_numeric(b0.autonomic, errors="coerce")
    groups["autonomic burden (SCOPA-AUT median split)"] = pd.Series(
        np.where(aut > aut.median(), "higher", "lower"), index=pop).where(aut.notna())
    groups["clinical subtype (Fereshtehnejad)"] = P.clinical_subtype

    raw_notes = {}
    try:
        up = baseline_rows(raw_csv("Non-motor_Assessments", "University_of_Pennsylvania_Smell_Identification_Test"))
        cols = [c for c in up.columns if c.startswith("SCENT_") and c.endswith("_CORRECT")]
        items = up[cols].apply(pd.to_numeric, errors="coerce")
        tot = items.sum(axis=1, min_count=int(0.9 * len(cols))).reindex(pop)
        groups["smell: anosmia (UPSIT <= 18)"] = pd.Series(np.where(tot <= 18, "anosmia", "not anosmic"), index=pop).where(tot.notna())
        groups["smell: UPSIT median split"] = pd.Series(np.where(tot < tot.median(), "worse smell", "better smell"), index=pop).where(tot.notna())
        raw_notes["upsit_items"] = len(cols)
    except Exception as exc:
        raw_notes["upsit"] = f"not available: {exc!r}"
    try:
        gen = raw_csv("_Subject_Characteristics", "iu_genetic_consensus").drop_duplicates("PATNO").set_index("PATNO").reindex(pop)
        for gene in ("GBA", "LRRK2"):
            v = gen[gene]
            raw_notes[f"{gene}_values"] = {str(k): sup(n) for k, n in v.astype(str).value_counts().items()}
            carrier = v.astype(str).str.strip().str.lower()
            known = v.notna() & ~carrier.isin(["", "nan", "none"])
            is_carrier = known & ~carrier.isin(["0", "0.0", "no", "negative", "wt", "wildtype", "wild type"])
            groups[f"{gene} carrier"] = pd.Series(np.where(is_carrier, "carrier", "non-carrier"), index=pop).where(known)
    except Exception as exc:
        raw_notes["genetics"] = f"not available: {exc!r}"
    try:
        sbr = baseline_rows(raw_csv("Imaging", "Xing_Core_Lab_-_Quant_SBR"), events=("SC", "BL", "V04"))
        put = pd.to_numeric(sbr.PUTAMEN_REF_CWM, errors="coerce").reindex(pop)
        groups["DaTscan putamen binding (median split)"] = pd.Series(
            np.where(put < put.median(), "lower binding", "higher binding"), index=pop).where(put.notna())
    except Exception as exc:
        raw_notes["datscan"] = f"not available: {exc!r}"

    rows = {}
    for name, lab in groups.items():
        lab = lab.reindex(pop)
        counts = lab.value_counts()
        rows[name] = {"groups": {str(k): {"participants": sup(n), "transitions": sup(trans[lab == k].sum())}
                                 for k, n in counts.items()},
                      "unclassified": sup(lab.isna().sum()),
                      "smallest_group": sup(counts.min()) if len(counts) else 0}
    stab = {"TD / PIGD (Stebbins)": stability(grid, pop, lambda v: v.phenotype),
            "akinetic-rigid / TD (Schiess)": stability(grid, pop, ar_label),
            "probable RBD (RBDSQ >= 5)": stability(grid, pop, lambda v: pd.Series(
                np.where(pd.to_numeric(v.rbd, errors="coerce") >= 5, "RBD", "no RBD"), index=v.index).where(v.rbd.notna())),
            "MoCA < 26": stability(grid, pop, lambda v: pd.Series(
                np.where(pd.to_numeric(v.MCATOT, errors="coerce") < 26, "MoCA < 26", "MoCA >= 26"), index=v.index).where(v.MCATOT.notna())),
            "depressive symptoms (GDS >= 5)": stability(grid, pop, lambda v: pd.Series(
                np.where(pd.to_numeric(v.mood, errors="coerce") >= 5, "GDS >= 5", "GDS < 5"), index=v.index).where(v.mood.notna()))}
    out = {"approval": a.approval, "participant_set": "development (D33)", "min_cell": MIN_CELL,
           "population": {"participants": sup(len(pop)), "transitions": sup(trans.sum()),
                          "rule": "primary, seen at the year-1 landmark, at least one transition after it, any phenotype"},
           "age_onset_median_years": round(med_onset), "groupings": rows, "stability": stab, "raw_files": raw_notes}
    path = _common.RESULTS / "wp0" / "subgroup_check.json"
    path.write_text(json.dumps(_common.to_json(out), indent=1))
    print(json.dumps(_common.to_json(out), indent=1))


if __name__ == "__main__":
    main()

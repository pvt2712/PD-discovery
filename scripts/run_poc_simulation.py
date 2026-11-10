"""The proof-of-concept simulation of D35 (protocol §19): H-P on synthetic panels shaped like the
4-domain PPMI analysis. Synthetic data only; it reads WP0's aggregate simulation settings
(configs/wp0_sim_settings.json), never participant-level data.

The simulator's 4-domain preset has tremor and axial feeding daily function, which feeds
independence: 4 domains and 3 links, like the PPMI analysis's map (three motor domains feeding
function). The panels have the development panel's size (686 participants, TD and PIGD in WP0's
proportions), WP0's visits per participant and missing shares. The analysis is the PPMI one under
D35: the map's links fixed (no Step 1 selection), 20 quadrature nodes, the load not moving with
speed (D31) and the [fit] rules.

  S1  speed only: PIGD runs the same mechanism faster, and speeds vary between participants. H-P
      should reject about 5%; the protocol's bound is 7.5% (Bradley, §8.7). Each dataset also
      draws one bootstrap replicate from its own fitted null, and the pooled replicates give the
      bootstrap's critical value (the warp-speed method).
  S2  one link stronger in PIGD beyond speed (function -> independence by 2 epsilon): power.

    python scripts/run_poc_simulation.py --null 200 --power 100 --jobs 4
    python scripts/run_poc_simulation.py --summarise

Each dataset is saved in results/poc/datasets/ as it finishes, so a stopped run resumes where it
stopped; results/poc/summary.json holds the summary. A failed dataset counts against H-P: as a
rejection under S1 and as a miss under S2.
"""
from __future__ import annotations

import argparse
import json

import _common
import numpy as np
from run_power_check import wp0_sim_settings
from scipy import stats

from pdx import config
from pdx import qualification as Q

OUT = _common.RESULTS / "poc"
N_DEVELOPMENT = 686                                  # the development panel's participants (D33)
WP0_ORDER = ["tremor", "axial", "brady_rigidity", "function", "independence", "cognition", "mood", "autonomic",
             "hallucinations", "sleepiness"]          # the order of WP0's settings file
PRESET = ["tremor", "axial", "function", "independence"]   # the simulator's 4-domain preset
ALPHA, BRADLEY = 0.05, 0.075


def sim_settings() -> dict:
    s = wp0_sim_settings()
    td, pigd = s["target_counts"]
    n_td = int(round(N_DEVELOPMENT * td / (td + pigd)))
    return {"target_counts": (n_td, N_DEVELOPMENT - n_td),
            "visit_probs": {int(k): v for k, v in s["visit_probs"].items()},
            "p_missing": [s["p_missing"][WP0_ORDER.index(n)] for n in PRESET]}


def tasks(n_null: int, n_power: int) -> list:
    cfg = config.load()
    sim = sim_settings()
    common = {"d": 4, "step1": False, "n_quad": cfg["engine"]["n_quad"], "treatment": cfg["engine"]["treatment"],
              "load_on_speed": cfg["engine"]["load_on_speed"], "fit_kw": dict(cfg["fit"]), "sim_kw": sim}
    n_gen = 5 * sum(sim["target_counts"])
    return (Q.make_tasks("S1", n_null, n_gen, "A", replicate=True, **common)
            + Q.make_tasks("S2", n_power, n_gen, "A", replicate=False, **common))


def _rate(hits) -> dict:
    hits = np.asarray(hits, bool)
    p = float(hits.mean()) if len(hits) else float("nan")
    return {"rate": p, "mcse": float(np.sqrt(p * (1 - p) / len(hits))) if len(hits) else float("nan")}


def summarise(rows: list) -> dict:
    out = {"settings": {**sim_settings(), "visit_probs": "WP0's (configs/wp0_sim_settings.json)"},
           "alpha": ALPHA, "bradley_bound": BRADLEY}
    s1 = [r for r in rows if r["scenario"] == "S1"]
    s2 = [r for r in rows if r["scenario"] == "S2"]
    crit = None
    if s1:
        ok = [r for r in s1 if "error" not in r]
        stars = [np.inf if r["lr_star"] is None else r["lr_star"] for r in ok]
        crit = Q._rank_quantile(stars, 1 - ALPHA) if ok else float("inf")
        chi = [("error" in r) or r["lr"] > stats.chi2.ppf(1 - ALPHA, r["df"]) for r in s1]
        boot = [("error" in r) or r["lr"] > crit for r in s1]
        out["S1_speed_only"] = {
            "datasets": len(s1), "failed": len(s1) - len(ok), "df": sorted({r["df"] for r in ok}),
            "false_positives_bootstrap": _rate(boot), "false_positives_chi2": _rate(chi),
            "bootstrap_critical_value": crit,
            "chi2_critical_value": float(stats.chi2.ppf(1 - ALPHA, ok[0]["df"])) if ok else None,
            "within_bradley_bound": bool(np.mean(boot) <= BRADLEY),
            "speed_ratio_true": ok[0]["true_speed_ratio"] if ok else None,
            "speed_ratio_estimates": {"median": float(np.median([r["sp1"] for r in ok])),
                                      "q10_q90": np.quantile([r["sp1"] for r in ok], [0.1, 0.9]).tolist()} if ok else None,
            "not_converged": int(sum(not r["converged"] for r in ok)),
            "median_minutes": float(np.median([r["seconds"] for r in ok]) / 60) if ok else None}
    if s2:
        ok = [r for r in s2 if "error" not in r]
        chi = [("error" not in r) and r["lr"] > stats.chi2.ppf(1 - ALPHA, r["df"]) for r in s2]
        out["S2_one_link_differs"] = {
            "datasets": len(s2), "failed": len(s2) - len(ok), "power_chi2": _rate(chi),
            "power_bootstrap": _rate([("error" not in r) and r["lr"] > crit for r in s2]) if crit is not None else None,
            "speed_ratio_true": ok[0]["true_speed_ratio"] if ok else None,
            "speed_ratio_median_estimate": float(np.median([r["sp1"] for r in ok])) if ok else None,
            "median_minutes": float(np.median([r["seconds"] for r in ok]) / 60) if ok else None}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--null", type=int, default=200, help="S1 datasets, each with one bootstrap replicate")
    ap.add_argument("--power", type=int, default=100, help="S2 datasets")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--summarise", action="store_true", help="only summarise the datasets already saved")
    a = ap.parse_args()
    ds = OUT / "datasets"
    if not a.summarise:
        Q.run_tasks(tasks(a.null, a.power), ds, a.jobs)
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(ds.glob("*.json"))]
    summary = summarise(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(_common.to_json(summary), indent=1))
    print(json.dumps(_common.to_json(summary), indent=1))


if __name__ == "__main__":
    main()

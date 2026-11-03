"""WP0's gate (protocol v0.5, §3): power of H-P at alpha = 5% against one deviation of 2 epsilon,
with the synthetic engine re-run at WP0's counts, follow-up and missingness. Synthetic data only;
it reads WP0's aggregate tables (results/wp0/wp0_audit.json), never participant-level data.

    python scripts/run_power_check.py --datasets 8 --jobs 4        # pilot (non-centrality estimate)
    python scripts/run_power_check.py --datasets 20 --jobs 4       # optional direct Monte Carlo check
    python scripts/run_power_check.py --summarise

Power is reported two ways: the share of datasets whose statistic exceeds its chi-square critical
value, and the noncentral chi-square power at the estimated non-centrality (the mean of LR - df,
each dataset with its own df), which is far more precise from few datasets. With S1 datasets run with --null (one bootstrap replicate
each), the bootstrap critical value is used as well.
"""
from __future__ import annotations

import argparse
import json

import _common  # noqa: F401
import numpy as np
from scipy import stats

from pdx import qualification as Q

OUT = _common.RESULTS / "wp0" / "power"


def wp0_settings(path) -> dict:
    """Simulation settings from WP0's aggregate tables."""
    r = json.loads(path.read_text())
    reached = {k: v for k, v in r["landmark"]["cells"].items()}
    land = {ph: reached[ph]["reached year 1"] for ph in ("TD", "PIGD")}
    vpp = {}
    for k, v in r["visits_per_participant"].items():
        vpp[int(k)] = v if isinstance(v, int) else 2          # suppressed small cells: 2
    known = {k: v for k, v in vpp.items() if k >= 2}
    one = vpp.get(1, 0) / max(sum(vpp.values()), 1)
    counts = tuple(int(round(land[ph] * (1 - one))) for ph in ("TD", "PIGD"))   # with at least one transition
    miss = {d: v["overall"] for d, v in r["missing_share"].items()}
    names = ["tremor", "axial", "brady_rigidity", "function", "independence", "cognition", "mood", "autonomic",
             "hallucinations", "sleepiness"]                     # the 10-domain simulator's order
    return {"target_counts": counts, "visit_probs": {str(k): v for k, v in known.items()},
            "p_missing": [round(float(miss[d]), 4) for d in names]}


def summarise(rows: list, alpha: float = 0.05) -> dict:
    """H-P re-selects its support in every dataset, so the degrees of freedom vary: each dataset is
    judged against its own critical value, the non-centrality is the mean of LR - df, and the
    noncentral power is averaged over the datasets' degrees of freedom."""
    ok = [r for r in rows if "error" not in r and r["scenario"] == "S2"]
    if not ok:
        return {"datasets": 0}
    df = np.array([r["df"] for r in ok])
    lr = np.array([r["lr"] for r in ok])
    crit = stats.chi2.ppf(1 - alpha, df)
    rej = lr > crit
    lam = max(float((lr - df).mean()), 0.0)
    se_lam = float((lr - df).std(ddof=1) / np.sqrt(len(lr))) if len(lr) > 1 else float("nan")
    power_nc = lambda l: float(np.mean(stats.ncx2.sf(crit, df, max(l, 1e-9))))
    values, counts = np.unique(df, return_counts=True)
    out = {"datasets": len(ok), "failed": len([r for r in rows if "error" in r]),
           "df_counts": {str(v): int(c) for v, c in zip(values, counts)},
           "chi2_critical": {str(v): float(stats.chi2.ppf(1 - alpha, v)) for v in values},
           "power_mc_chi2": float(rej.mean()), "mcse_mc": float(np.sqrt(rej.mean() * (1 - rej.mean()) / len(lr))),
           "noncentrality": lam, "noncentrality_se": se_lam, "power_noncentral": power_nc(lam),
           "power_noncentral_range": [power_nc(lam - 1.96 * se_lam), power_nc(lam + 1.96 * se_lam)],
           "median_seconds": float(np.median([r["seconds"] for r in ok]))}
    null = [r for r in rows if "error" not in r and r["scenario"] == "S1" and r.get("lr_star") is not None]
    if null:
        c = Q._rank_quantile([r["lr_star"] for r in null], 1 - alpha)
        out["bootstrap_critical"] = c
        out["power_mc_bootstrap"] = float((lr > c).mean())
    out["gate_80_percent"] = "pass" if out["power_noncentral_range"][0] >= 0.8 else (
        "fail" if out["power_noncentral_range"][1] < 0.8 else "undecided: run more datasets")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", type=int, default=8)
    ap.add_argument("--null", type=int, default=0, help="S1 datasets with one bootstrap replicate each")
    ap.add_argument("--d", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--summarise", action="store_true")
    a = ap.parse_args()
    ds = OUT / "datasets"
    settings = wp0_settings(_common.RESULTS / "wp0" / "wp0_audit.json")
    n_gen = 5 * sum(settings["target_counts"])
    sim_kw = {"target_counts": settings["target_counts"], "p_missing": settings["p_missing"][:a.d],
              "visit_probs": {int(k): v for k, v in settings["visit_probs"].items()}}
    if not a.summarise:
        tasks = Q.make_tasks("S2", a.datasets, n_gen, "A", d=a.d, step1=True, sim_kw=sim_kw, replicate=False)
        if a.null:
            tasks += Q.make_tasks("S1", a.null, n_gen, "A", d=a.d, step1=True, sim_kw=sim_kw, replicate=True)
        Q.run_tasks(tasks, ds, a.jobs)
    rows = [json.loads(p.read_text()) for p in sorted(ds.glob("*.json"))]
    summary = {"settings_from_wp0": settings, **summarise(rows)}
    (OUT / "summary.json").write_text(json.dumps(_common.to_json(summary), indent=1))
    print(json.dumps(_common.to_json({k: v for k, v in summary.items() if k != "settings_from_wp0"}), indent=1))


if __name__ == "__main__":
    main()

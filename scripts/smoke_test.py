"""End-to-end smoke test of the whole analysis chain on one small synthetic dataset (no PPMI data).

    python scripts/smoke_test.py [--n 300] [--out results/smoke]

Runs: support selection (FCI), H-P with a short sequential bootstrap, the speed estimates with a
short participant bootstrap, anchor-free per-link and per-domain tests with the link classes,
the anchored tests, joint discovery, the shared-eigenvector check, the comparators implemented
here, a small separate-graph reference and the WP0 audit on a synthetic visit table. Sizes are
far below the protocol's; the point is that every piece runs and returns aggregates.
"""
from __future__ import annotations

import argparse
import json
import time

import _common  # noqa: F401  (import path)
import numpy as np
import pandas as pd

from pdx import hp, localise, speed, wp0
from pdx.comparators import run as comp
from pdx.engine.likelihood import prepare
from pdx.qualification import spec_for
from pdx.separate import separate_graph_reference
from pdx.sim import scenarios
from pdx.step1 import Step1Config, make_support_fn, reported_map


def synthetic_visit_table(panel, rng) -> pd.DataFrame:
    rows = []
    for i in range(panel.n):
        ph = ["TD", "PIGD"][panel.group[i]]
        for k, t in enumerate(panel.t[i]):
            rows.append({"participant": f"S{i:04d}", "cohort": "PPMI 1.0" if i % 3 else "PPMI 2.0",
                         "year": k + 1, "date_years": t, "phenotype": ph,
                         "td_pigd_ratio": rng.lognormal(0, 0.5), "site": int(i % 12),
                         "mode": "remote" if rng.random() < 0.1 else "in person",
                         "treated": int(panel.dose[i][k] > 0), "dose_change": int(rng.random() < 0.3),
                         "motor_state": "ON" if panel.on[i][k] else "OFF",
                         **{f"d{j}": panel.y[i][k, j] for j in range(panel.d)}})
        rows.append({"participant": f"S{i:04d}", "cohort": "PPMI 1.0" if i % 3 else "PPMI 2.0", "year": 0,
                     "date_years": 0.0, "phenotype": ph, "td_pigd_ratio": rng.lognormal(0, 0.5),
                     "site": int(i % 12), "mode": "in person", "treated": 0, "dose_change": 0,
                     "motor_state": "OFF", **{f"d{j}": np.nan for j in range(panel.d)}})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(_common.RESULTS / "smoke"))
    a = ap.parse_args()
    _common.RESULTS.mkdir(exist_ok=True)
    path = _common.require_inside(_common.Path(a.out), _common.RESULTS, "the output") / "summary.json"
    rng = np.random.default_rng(a.seed)
    timings, out = {}, {}

    def tick(name, t0):
        timings[name] = round(time.time() - t0, 1)
        print(f"{name}: {timings[name]} s", flush=True)

    panel, info = scenarios.generate("S2", a.n, rng)
    spec = spec_for(panel, info)
    out["counts"] = panel.aggregate_summary()
    fit_kw = {"maxiter": 500}

    t0 = time.time()
    out["step1_map"] = reported_map(panel, Step1Config(depth=2), names=panel.domains)
    tick("step1_map", t0)

    t0 = time.time()
    support_fn = make_support_fn(info["support"], Step1Config(depth=2))
    res = hp.run_hp(panel, spec, support_fn, n_max=9, h=3, rng=rng, fit_kw=fit_kw)
    out["hp"] = res.summary()
    tick("hp", t0)

    spec_s = spec.with_support(res.support)
    prep = prepare(panel, spec_s.meas_var, spec_s.n_groups)
    t0 = time.time()
    out["speed"] = speed.bootstrap_speed(panel, spec_s, res.null_fit, res.free_fit, B=4, rng=rng, fit_kw=fit_kw)
    tick("speed", t0)

    t0 = time.time()
    links = localise.per_link_tests(res.free_fit, panel, M=500, rng=rng, prep=prep)
    out["links"] = localise.classify_links(links, eps=0.10)
    out["speed_and_load_only"] = localise.speed_and_load_only(links, 0.10)
    out["speed_diagnostics"] = links.diagnostics
    out["speed_set"] = links.speed_set
    out["domains"] = localise.per_domain_tests(links, spec_s)
    tick("per_link_and_domain", t0)

    t0 = time.time()
    out["anchored"] = localise.anchored_tests(panel, spec_s, res.free_fit, prep=prep, fit_kw=fit_kw)
    tick("anchored", t0)

    t0 = time.time()
    jd = localise.joint_discovery(panel, spec_s, res.free_fit, res.null_fit, prep=prep, n_lambda=4,
                                  fit_kw=fit_kw)
    out["joint_discovery"] = {k: jd[k] for k in ("selected", "best", "identified", "speed_weights")}
    out["joint_discovery"]["refit"] = jd.get("refit")
    tick("joint_discovery", t0)

    t0 = time.time()
    out["commutator"] = localise.commutator_test(panel, B=50, rng=rng)
    tick("commutator", t0)

    t0 = time.time()
    out["C3"] = {k: v for k, v in comp.c3_nct(panel, n_perm=50, rng=rng).items() if k != "p_edges"}
    out["C5"] = comp.c5_tsldiffpc(panel)
    out["C11"] = {k: v for k, v in comp.c11_naive(panel, Step1Config(depth=2)).items() if k != "p_interaction"}
    out["C12"] = comp.c12_dynotears(panel)
    tick("comparators", t0)

    t0 = time.time()
    out["separate_graphs"] = separate_graph_reference(panel, n_sub=3, n_perm=3, rng=rng, cfg=Step1Config(depth=2))
    tick("separate_graphs", t0)

    t0 = time.time()
    df = synthetic_visit_table(panel, rng)
    out["wp0"] = wp0.audit(df, wp0.Columns(domains=[f"d{j}" for j in range(panel.d)]))
    tick("wp0", t0)

    out["timings_s"] = timings
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_common.to_json(out), indent=1, default=str))
    print("wrote", path)


if __name__ == "__main__":
    main()

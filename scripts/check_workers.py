"""Sanity check of the deep-learning / pretrained workers on a known synthetic graph (no PPMI data).

    uv run python scripts/check_workers.py data/derived/tmp/test_small.npz data/derived/tmp/test_small_truth.npz [learners]

Runs every worker of methods/external.py on IN.npz (in parallel, each in its own side environment), then scores W
against the true coefficient matrix B: AUROC of the skeleton (max of W_ij, W_ji against B_ij or B_ji non-zero), AUROC
of the directed edges, and the share of the strong edges among the top-k skeleton scores (k = number of strong edges).
Seconds are the worker's own (it saves them), so learners run in parallel are timed separately.
Writes results/worker_check.csv (aggregates of synthetic data only).
"""
import json
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from escalation_discovery import config as C
from escalation_discovery.methods.external import SPECS, External, worker_env

p_in, p_truth = sys.argv[1], sys.argv[2]
only = sys.argv[3].split(",") if len(sys.argv) > 3 else list(SPECS)
truth = np.load(p_truth, allow_pickle=False)
B, strong = truth["B"] != 0, truth["strong"] != 0
d = B.shape[0]
off = ~np.eye(d, dtype=bool)
upper = np.triu(np.ones((d, d), dtype=bool), 1)
procs = {}
for name in only:
    m = External(name)
    ok, why = m.available()
    if not ok:
        procs[name] = (None, why, 0.0)
        continue
    out = str(C.derived_dir("tmp") / f"check_{name}.npz")
    cfg = {k: v for k, v in m.cfg.items() if k != "threshold"}
    procs[name] = (subprocess.Popen([str(m.python()), str(m.script()), p_in, out, json.dumps(cfg)], cwd=str(C.ROOT),
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=worker_env()), out, time.time())
rows = []
for name, (proc, out, t0) in procs.items():
    if proc is None:
        rows.append(dict(learner=name, status=f"unavailable: {out}"))
        continue
    try:
        so, se = proc.communicate(timeout=3600)
    except subprocess.TimeoutExpired:
        proc.kill()
        rows.append(dict(learner=name, status="timeout (1 h)"))
        continue
    secs = time.time() - t0
    if proc.returncode != 0:
        rows.append(dict(learner=name, status="failed", seconds=round(secs, 1), error=se.strip()[-400:]))
        continue
    res = np.load(out, allow_pickle=False)
    W = np.abs(res["W"])
    secs = float(res["seconds"]) if "seconds" in res.files else secs
    skel_score = np.maximum(W, W.T)[upper]
    skel_true = (B | B.T)[upper]
    k = int(strong.sum())
    top = np.argsort(-np.maximum(W, W.T)[upper])[:k]
    strong_skel = (strong | strong.T)[upper]
    rows.append(dict(learner=name, status="ok", seconds=round(secs, 1), kind=str(res["kind"]),
                     auroc_skeleton=round(roc_auc_score(skel_true, skel_score), 3),
                     auroc_directed=round(roc_auc_score(B[off], W[off]), 3),
                     strong_in_top_k=round(float(strong_skel[top].mean()), 3), note=so.strip()[-120:]))
    print(rows[-1], flush=True)
df = pd.DataFrame(rows)
path = C.results_dir() / "worker_check.csv"
if path.exists() and len(only) < len(SPECS):                  # a partial rerun replaces only its learners' rows
    old = pd.read_csv(path)
    df = pd.concat([old[~old.learner.isin(only)], df], ignore_index=True)
df.to_csv(path, index=False)
print(df.to_string(index=False))

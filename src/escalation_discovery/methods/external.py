"""external — learners that need their own Python version run as workers in side environments (PROTOCOL.md §5.2).

causica (DECI) needs Python 3.10, AVICI and SDCD at most 3.11; the harness is 3.12. Each learner has a worker script
in workers/ and a uv environment in envs/<env>/. The contract:

    envs/<env>/.venv/bin/python workers/<worker>.py IN.npz OUT.npz '<json config>'

IN.npz holds X (n x d float64, complete rows: the shared imputed table, z-scored unless the units stress setting asks
for raw units), names, seed. OUT.npz holds W (d x d, W[i, j] for i -> j: a weight, or an edge probability when
kind == 'probability') and seconds. The adapter thresholds W into a DAG. The temporary files are written under
data/derived/tmp/ and deleted afterwards: on PPMI they are participant-level (PROTOCOL.md §11).
"""
import json
import os
import subprocess
import tempfile
import time

import numpy as np

from .. import config as C
from ..graphs import from_weighted
from .base import Method

# learner -> (environment folder, worker script, family, default config, tuning grid)
SPECS = {
    "deci": ("causica", "deci", "end-to-end", dict(threshold=0.5, max_steps=5000),
             [dict(threshold=t, max_steps=s) for t in (0.3, 0.5, 0.7) for s in (2500, 5000)]),
    "avici": ("avici", "avici", "pretrained", dict(threshold=0.5, model="scm-v0"),
              [dict(threshold=t, model="scm-v0") for t in (0.1, 0.2, 0.3, 0.5, 0.7, 0.9)]),
    "sdcd": ("sdcd", "sdcd", "neural", dict(threshold=0.3), [dict(threshold=t) for t in (0.1, 0.2, 0.3, 0.4, 0.5, 0.7)]),
    # DiBS: the noise variance sets how readily edges enter on z-scored data (1.0 = a node's total variance, an upper
    # bound; 0.3 = a node with a well-explained mean); at the default, base-size fits kept 4 edges (29 October 2026)
    "dibs": ("jax312", "dibs", "neural", dict(threshold=0.3, particles=20, steps=2000, obs_noise=1.0),
             [dict(threshold=t, particles=20, steps=2000, obs_noise=s) for s in (1.0, 0.3) for t in (0.1, 0.3, 0.5)]),
    "cdfm": ("jax312", "cdfm", "pretrained", dict(threshold=0.5), [dict(threshold=t) for t in (0.1, 0.2, 0.3, 0.5, 0.7, 0.9)]),
    "arrow": ("jax312", "arrow", "pretrained", dict(threshold=0.5), [dict(threshold=t) for t in (0.1, 0.2, 0.3, 0.5, 0.7, 0.9)]),
}


def worker_env():
    """Thread limits for a worker (ESCD_WORKER_THREADS, default 1): Arm A and B run several learners at once, and
    torch, JAX and BLAS would otherwise each take every core."""
    n = os.environ.get("ESCD_WORKER_THREADS", "1")
    env = dict(os.environ)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[k] = n
    env["XLA_FLAGS"] = f"--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads={n}"
    return env


class External(Method):
    output = "DAG"
    accepts_knowledge = False
    missing = "impute"
    scale_sensitive = True

    def __init__(self, learner, **cfg):
        self.learner = learner
        self.env, self.worker, self.family, self._defaults, self._grid = SPECS[learner]
        super().__init__(**cfg)
        self.name = learner
        self.version = f"envs/{self.env} (see envs/{self.env}/pyproject.toml)"

    def defaults(self):
        return dict(self._defaults)

    def grid(self):
        return list(self._grid)

    def python(self):
        return C.ROOT / "envs" / self.env / ".venv" / "bin" / "python"

    def script(self):
        return C.ROOT / "workers" / f"{self.worker}.py"

    def available(self):
        if not self.python().exists():
            return False, f"side environment envs/{self.env} not installed"
        if not self.script().exists():
            return False, f"worker workers/{self.worker}.py missing"
        return True, ""

    def _fit(self, data):
        X = data.complete_input(self.scale_sensitive).astype(np.float64)
        tmp = C.derived_dir("tmp")
        fd_in, p_in = tempfile.mkstemp(suffix=".npz", dir=tmp)
        fd_out, p_out = tempfile.mkstemp(suffix=".npz", dir=tmp)
        os.close(fd_in), os.close(fd_out)
        try:
            np.savez(p_in, X=X, names=np.array(data.names), seed=int(data.seed))
            cfg = {k: v for k, v in self.cfg.items() if k != "threshold"}
            t0 = time.time()
            r = subprocess.run([str(self.python()), str(self.script()), p_in, p_out, json.dumps(cfg)],
                               capture_output=True, text=True, timeout=int(self.cfg.get("timeout", 3600)), cwd=str(C.ROOT),
                               env=worker_env())
            if r.returncode != 0:
                raise RuntimeError(f"{self.learner} worker failed: {r.stderr[-800:]}")
            out = np.load(p_out, allow_pickle=False)
            W = np.asarray(out["W"], dtype=float)
            self.last_seconds = time.time() - t0
        finally:
            for p in (p_in, p_out):
                if os.path.exists(p):
                    os.remove(p)
        W = W.copy()
        np.fill_diagonal(W, 0.0)
        return from_weighted(W, data.names, float(self.cfg["threshold"]))

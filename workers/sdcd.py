"""sdcd worker: SDCD, Stable Differentiable Causal Discovery (Nazaret, Hong, Azizi & Blei, ICML 2024, arXiv 2311.10263).
Package sdcd 0.1.4 (MIT) on PyTorch, CPU only; environment envs/sdcd (Python 3.11).

    envs/sdcd/.venv/bin/python workers/sdcd.py IN.npz OUT.npz '{}'

Fits SDCD with the package defaults: stage 1 pre-selection and stage 2 DAG learning (power-iteration DAG penalty), up to
2,000 epochs each with early stopping on a random 20 % validation split, batch 256, one-hidden-layer (10 units) MLP per
node with a learned variance; observational data only (every row gets the "no intervention" mask). Optional config keys
stage1_kwargs / stage2_kwargs / model_kwargs are passed to SDCD.train unchanged.
W[i, j] = SDCD's unthresholded adjacency for i -> j: the L2 norm of the first-layer weights from input i into node j's
MLP (kind "weight"; edges removed by the stage-1 mask are exactly 0). SDCD's own default threshold is 0.1.
Preprocessing: none (standard_scale=False, the package default): X is used as given.
Seeds: SDCD.train reseeds random/numpy/torch with 0; the worker redirects that call to the input seed.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.curdir) != HERE]  # this file must not shadow `sdcd`
os.environ.update(CUDA_VISIBLE_DEVICES="", WANDB_MODE="disabled", WANDB_SILENT="true")

import numpy as np  # noqa: E402


def main(p_in, p_out, cfg):
    t0 = time.time()
    data = np.load(p_in, allow_pickle=False)
    X, seed = np.asarray(data["X"], dtype=np.float64), int(data["seed"])
    if not np.isfinite(X).all():
        raise ValueError("X must be complete (no NaN/inf)")
    import warnings
    warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
    import torch
    import sdcd.models._sdcd as sdcd_module
    from sdcd.utils import set_random_seed_all
    sdcd_module.set_random_seed_all = lambda _seed=0: set_random_seed_all(seed)
    set_random_seed_all(seed)
    n, d = X.shape
    ds = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32),
                                        torch.ones((n, d), dtype=torch.long),   # 1 = not intervened, per node
                                        torch.zeros(n, dtype=torch.long))       # regime 0 = observational
    model = sdcd_module.SDCD()
    model.train(ds, stage1_kwargs=cfg.get("stage1_kwargs"), stage2_kwargs=cfg.get("stage2_kwargs"),
                model_kwargs=cfg.get("model_kwargs"))
    W = np.abs(np.asarray(model.get_adjacency_matrix(threshold=False), dtype=np.float64))
    np.fill_diagonal(W, 0.0)
    np.savez(p_out, W=W, kind="weight", seconds=time.time() - t0)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], json.loads(sys.argv[3]) if len(sys.argv) > 3 else {})

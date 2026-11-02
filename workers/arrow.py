"""arrow worker: Arrow, a zero-shot foundation model for causal discovery (Thompson, Zhao, Steinberg & Bonilla 2026,
arXiv 2605.07204). Package arrowfm 0.1.0 (MIT; github.com/ryan-thompson/arrowfm at commit 8487278) on PyTorch, CPU
only; environment envs/jax312 (Python 3.12).

    envs/jax312/.venv/bin/python workers/arrow.py IN.npz OUT.npz '{}'

Weights: arrowfm-base.pt (147 MB) from Hugging Face ryan-thompson/arrowfm-base (revision 49a256f), downloaded once into
envs/jax312/weights/arrow/ and loaded from there (no network).

W[i, j] = Arrow's DAG edge probability p_hat[i, j] for i -> j = sigmoid(edge logit) x sigmoid(order score i - order
score j) (kind "probability"); one forward pass, no fitting. Arrow's own DAG decoder (MAP edges + MAP node order) is not
used: thresholding happens in the harness.
Preprocessing: Arrow standardises each column internally, so raw vs z-scored input gives the same W.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.curdir) != HERE]  # this file must not shadow any `arrow`
os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")

import numpy as np  # noqa: E402

WEIGHTS = os.path.join(os.path.dirname(HERE), "envs", "jax312", "weights", "arrow", "arrowfm-base.pt")


def main(p_in, p_out, cfg):
    t0 = time.time()
    data = np.load(p_in, allow_pickle=False)
    X, seed = np.asarray(data["X"], dtype=np.float64), int(data["seed"])
    if not np.isfinite(X).all():
        raise ValueError("X must be complete (no NaN/inf)")
    import torch
    from arrowfm import ArrowPredictor
    torch.manual_seed(seed), np.random.seed(seed)
    predictor = ArrowPredictor.from_checkpoint(WEIGHTS, device="cpu", use_cache=False)
    p_hat, _ = predictor.predict_parameters(torch.from_numpy(X.astype(np.float32)))
    W = p_hat.double().numpy()
    np.fill_diagonal(W, 0.0)
    np.savez(p_out, W=W, kind="probability", seconds=time.time() - t0)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], json.loads(sys.argv[3]) if len(sys.argv) > 3 else {})

"""cdfm worker: CDFM, a causal discovery foundation model (Qiao, Cai, Li, Chen, Hua, Xu, Chen, Hao & Cui 2026,
arXiv 2607.11508). Package cdfm-base 0.1.0 (Apache-2.0) on PyTorch, CPU only; environment envs/jax312 (Python 3.12).

    envs/jax312/.venv/bin/python workers/cdfm.py IN.npz OUT.npz '{}'

Weights: DMIRLAB/CDFM from Hugging Face (revision fd3ac4f; config.json + model.safetensors, 38.7 MB), downloaded once
into envs/jax312/weights/cdfm/ and loaded from that folder (no network).

W[i, j] = sigmoid(edge logit) = CDFM's probability of the edge i -> j (kind "probability"); one forward pass, no fitting.
CDFM's calibrated auto-threshold and binary adjacency are not used: thresholding happens in the harness.
Preprocessing: CDFM z-scores each column internally (predict(standardize=True), its default), so raw vs z-scored input
gives the same W; no missing-value mask is passed (complete rows).
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.curdir) != HERE]  # this file must not shadow `cdfm`
os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")

import numpy as np  # noqa: E402

WEIGHTS = os.path.join(os.path.dirname(HERE), "envs", "jax312", "weights", "cdfm")


def main(p_in, p_out, cfg):
    t0 = time.time()
    data = np.load(p_in, allow_pickle=False)
    X, seed = np.asarray(data["X"], dtype=np.float64), int(data["seed"])
    if not np.isfinite(X).all():
        raise ValueError("X must be complete (no NaN/inf)")
    import torch
    from cdfm import CDFM
    torch.manual_seed(seed), np.random.seed(seed)
    model = CDFM.from_pretrained(WEIGHTS, device="cpu", threshold=0.5)  # fixed threshold: skips the auto-calibration
    W = np.asarray(model.predict(X, threshold=0.5).probabilities, dtype=np.float64)
    np.fill_diagonal(W, 0.0)
    np.savez(p_out, W=W, kind="probability", seconds=time.time() - t0)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], json.loads(sys.argv[3]) if len(sys.argv) > 3 else {})

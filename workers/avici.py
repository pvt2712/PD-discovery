"""avici worker: AVICI, amortized causal structure inference (Lorch, Sussex, Rothfuss, Krause & Schölkopf, NeurIPS 2022,
arXiv 2205.12934). Package avici 1.0.7 (MIT) on JAX 0.4.30, CPU only; environment envs/avici (Python 3.11).

    envs/avici/.venv/bin/python workers/avici.py IN.npz OUT.npz '{"model": "scm-v0"}'

Weights: the pretrained "scm-v0" checkpoint (kwargs.json + checkpoint_0300000.pkl, 68 MB) from the Hugging Face repo
larslorch/avici, downloaded once into envs/avici/weights/ (Hugging Face cache layout) and loaded offline.

W[i, j] = AVICI's predicted marginal probability of the edge i -> j (kind "probability"); one forward pass, no fitting.
Preprocessing: AVICI z-standardises each column internally (so raw vs z-scored input gives the same W). scm-v0 was
trained on n = 200 rows; when n > chunk (config "chunk", default 1000) the rows are shuffled with the seed, split into
k = ceil(n / chunk) equal chunks (the n mod k leftover rows are dropped) and attention over rows runs within each chunk
before AVICI's max-pool over rows (avici's experimental_chunk_size), which bounds memory at n = 7,000.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.curdir) != HERE]  # this file must not shadow `avici`
os.environ.update(JAX_PLATFORMS="cpu", HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")

import numpy as np  # noqa: E402

WEIGHTS = os.path.join(os.path.dirname(HERE), "envs", "avici", "weights")


def main(p_in, p_out, cfg):
    t0 = time.time()
    data = np.load(p_in, allow_pickle=False)
    X, seed = np.asarray(data["X"], dtype=np.float64), int(data["seed"])
    if not np.isfinite(X).all():
        raise ValueError("X must be complete (no NaN/inf)")
    np.random.seed(seed)
    import avici
    model = avici.load_pretrained(download=cfg.get("model", "scm-v0"), cache_path=WEIGHTS, verbose=False)
    n, chunk, size = X.shape[0], int(cfg.get("chunk", 1000)), None
    if n > chunk:
        k = -(-n // chunk)
        size = n // k
        X = X[np.random.default_rng(seed).permutation(n)[: k * size]]
    W = np.asarray(model(X.astype(np.float32), shard_if_possible=False, experimental_chunk_size=size), dtype=np.float64)
    np.fill_diagonal(W, 0.0)
    np.savez(p_out, W=W, kind="probability", seconds=time.time() - t0)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], json.loads(sys.argv[3]) if len(sys.argv) > 3 else {})

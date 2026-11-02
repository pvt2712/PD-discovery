"""dibs worker: DiBS, Differentiable Bayesian Structure Learning (Lorch, Rothfuss, Schölkopf & Krause, NeurIPS 2021,
arXiv 2105.11839). Package dibs-lib 1.3.3 (MIT) on JAX, CPU only; environment envs/jax312 (Python 3.12).

    envs/jax312/.venv/bin/python workers/dibs.py IN.npz OUT.npz '{"particles": 20, "steps": 2000}'

Model: JointDiBS (SVGD over latent graph particles and their edge weights, package defaults otherwise) with dibs'
linear-Gaussian likelihood (dibs.models.LinearGaussian: x_j = sum_i theta_ij x_i + noise, theta ~ N(0, 1), one shared
noise variance obs_noise) and an Erdos-Renyi graph prior with 2 expected edges per node (dibs' default prior density;
config "graph_prior": "sf" for scale-free). `particles` and `steps` come from the config (defaults 20 and 2000).
obs_noise defaults to 1.0, the total variance of a z-scored column and so an upper bound on any node's noise variance
(dibs' own default, 0.1, is tied to its simulator's scale); config "obs_noise" overrides it.
Speed: the observational log-likelihood is evaluated from the Gram matrix X'X instead of row by row. It is the same
function (sum of Gaussian log-densities, checked equal to the package's to ~1e-7 relative) but costs O(d^3) per graph
rather than O(n d^2), which keeps n = 7,000 tractable.
W[i, j] = marginal posterior probability of i -> j: the share of the final particle graphs containing the edge (dibs'
empirical particle distribution; config "weighting": "mixture" weights particles by their unnormalised posterior,
the paper's DiBS+) (kind "probability").
Preprocessing: none; X is used as given (z-scored by the harness unless the units setting asks for raw units).
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.curdir) != HERE]  # this file must not shadow `dibs`
os.environ.update(JAX_PLATFORMS="cpu")

import numpy as np  # noqa: E402


def main(p_in, p_out, cfg):
    t0 = time.time()
    data = np.load(p_in, allow_pickle=False)
    X, seed = np.asarray(data["X"], dtype=np.float64), int(data["seed"])
    if not np.isfinite(X).all():
        raise ValueError("X must be complete (no NaN/inf)")
    import warnings
    warnings.filterwarnings("ignore", category=SyntaxWarning)
    import jax.numpy as jnp
    from jax import random
    from dibs.inference import JointDiBS
    from dibs.models import ErdosReniDAGDistribution, LinearGaussian, ScaleFreeDAGDistribution

    class GramLinearGaussian(LinearGaussian):
        """dibs LinearGaussian whose observational log-likelihood is computed from X'X (same value, O(d^3))."""

        def __init__(self, *, x, **kw):
            super().__init__(**kw)
            self.n_obs, self.xtx = x.shape[0], x.T @ x

        def log_likelihood(self, *, x, theta, g, interv_targets):  # interv_targets is all zero (observational)
            a = jnp.eye(self.n_vars) - g * theta                   # residuals R = X (I - G*Theta)
            rss = jnp.sum(a * (self.xtx @ a))                       # ||R||_F^2 = tr(A' X'X A)
            return -0.5 * rss / self.obs_noise - 0.5 * self.n_obs * self.n_vars * jnp.log(2 * jnp.pi * self.obs_noise)

    np.random.seed(seed)
    n, d = X.shape
    x = jnp.asarray(X, dtype=jnp.float32)
    prior = ScaleFreeDAGDistribution if cfg.get("graph_prior") == "sf" else ErdosReniDAGDistribution
    likelihood = GramLinearGaussian(x=x, n_vars=d, obs_noise=float(cfg.get("obs_noise", 1.0)))
    model = JointDiBS(x=x, interv_mask=None, graph_model=prior(n_vars=d, n_edges_per_node=2),
                      likelihood_model=likelihood)
    gs, thetas = model.sample(key=random.PRNGKey(seed), n_particles=int(cfg.get("particles", 20)),
                              steps=int(cfg.get("steps", 2000)))
    dist = model.get_mixture(gs, thetas) if cfg.get("weighting") == "mixture" else model.get_empirical(gs, thetas)
    w = np.exp(np.asarray(dist.logp, dtype=np.float64))
    W = np.einsum("k,kij->ij", w / w.sum(), np.asarray(dist.g, dtype=np.float64))
    np.fill_diagonal(W, 0.0)
    np.savez(p_out, W=W, kind="probability", seconds=time.time() - t0)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], json.loads(sys.argv[3]) if len(sys.argv) > 3 else {})

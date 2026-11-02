"""deci worker: DECI, Deep End-to-end Causal Inference (Geffner et al. 2022, arXiv 2202.02195). Package causica 0.4.6
(MIT, Microsoft) on PyTorch, CPU only; environment envs/causica (Python 3.10).

    envs/causica/.venv/bin/python workers/deci.py IN.npz OUT.npz '{"epochs": 2000}'

Training follows causica's examples/csuite_example.ipynb recipe: every column is one continuous node; ENCO variational
distribution over graphs; DECIEmbed functional relationships (embedding 32, two-layer g and zeta); spline noise (config
"noise": "gaussian" switches); Gibbs DAG prior with sparsity lambda 5; Gumbel-softmax temperature 0.25; Adam with
causica's augmented-Lagrangian scheduler AugLagLR at its default settings (learning rates: graph 0.1, functions 3e-4,
noise 3e-3); batch `batch` rows (config, default 512); training stops at AugLagLR convergence (as causica's
Lightning callback does), after `epochs` passes over the data, or after `max_steps` optimiser steps (config, default
5000), whichever comes first. The step budget is escalation-discovery's (PROTOCOL.md §5.4): with causica's batch of 128
and 2,000 epochs, one fit took 14.5 min on 1,000 rows x 10 variables and would take hours at the pooled cohort's
7,300 rows x 43 variables. Whether the augmented Lagrangian converged is printed and returned (`converged`).
W[i, j] = posterior probability of the edge i -> j under the fitted ENCO graph distribution (kind "probability").
Preprocessing: none; X is used as given (complete rows: causica's likelihood has no missing-data mask).
"""
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.curdir) != HERE]  # keep workers/ off the import path
os.environ.update(CUDA_VISIBLE_DEVICES="")

import numpy as np  # noqa: E402


def main(p_in, p_out, cfg):
    t0 = time.time()
    data = np.load(p_in, allow_pickle=False)
    X, seed = np.asarray(data["X"], dtype=np.float64), int(data["seed"])
    if not np.isfinite(X).all():
        raise ValueError("X must be complete (no NaN/inf)")
    import torch
    from tensordict import TensorDict
    from torch.utils.data import DataLoader
    from causica.datasets.tensordict_utils import tensordict_shapes
    from causica.datasets.variable_types import VariableTypeEnum
    from causica.distributions import (ContinuousNoiseDist, ENCOAdjacencyDistributionModule, GibbsDAGPrior,
                                       JointNoiseModule, create_noise_modules)
    from causica.functional_relationships import DECIEmbedFunctionalRelationships
    from causica.graph.dag_constraint import calculate_dagness
    from causica.sem.sem_distribution import SEMDistributionModule
    from causica.training.auglag import AugLagLossCalculator, AugLagLR, AugLagLRConfig

    random.seed(seed), np.random.seed(seed), torch.manual_seed(seed)
    n, d = X.shape
    keys = [f"x{j}" for j in range(d)]
    td = TensorDict({k: torch.tensor(X[:, [j]], dtype=torch.float32) for j, k in enumerate(keys)}, batch_size=[n])
    shapes = tensordict_shapes(td)
    noise = ContinuousNoiseDist.GAUSSIAN if cfg.get("noise") == "gaussian" else ContinuousNoiseDist.SPLINE
    noise_module = JointNoiseModule(create_noise_modules(shapes, {k: VariableTypeEnum.CONTINUOUS for k in keys}, noise))
    adjacency = ENCOAdjacencyDistributionModule(d)
    functions = DECIEmbedFunctionalRelationships(shapes=shapes, embedding_size=32, out_dim_g=32, num_layers_g=2,
                                                 num_layers_zeta=2)
    sem_module = SEMDistributionModule(adjacency, functions, noise_module)
    prior = GibbsDAGPrior(num_nodes=d, sparsity_lambda=5.0)
    auglag_config = AugLagLRConfig()
    modules = {"functional_relationships": functions, "vardist": adjacency, "noise_dist": noise_module}
    optimizer = torch.optim.Adam([{"params": m.parameters(), "lr": auglag_config.lr_init_dict[k], "name": k}
                                  for k, m in modules.items()])
    scheduler = AugLagLR(config=auglag_config)
    auglag_loss = AugLagLossCalculator(init_alpha=0.0, init_rho=1.0)
    loader = DataLoader(td, collate_fn=lambda x: x, batch_size=int(cfg.get("batch", 512)), shuffle=True, drop_last=False,
                        generator=torch.Generator().manual_seed(seed))
    converged, epoch, steps, max_steps = False, 0, 0, int(cfg.get("max_steps", 5000))
    for epoch in range(int(cfg.get("epochs", 2000))):
        for batch in loader:
            steps += 1
            optimizer.zero_grad()
            sem_distribution = sem_module()
            sem, *_ = sem_distribution.relaxed_sample(torch.Size([]), temperature=0.25)
            objective = (-sem_distribution.entropy() - prior.log_prob(sem.graph)) / n - sem.log_prob(batch).mean()
            constraint = calculate_dagness(sem.graph)
            loss = auglag_loss(objective, constraint / n)
            loss.backward()
            optimizer.step()
            converged, _ = scheduler.step(optimizer=optimizer, loss=auglag_loss, loss_value=loss,
                                          lagrangian_penalty=constraint)
            if converged or steps >= max_steps:
                break
        if converged or steps >= max_steps:
            break
    with torch.no_grad():
        W = adjacency().mean.double().numpy()
    np.fill_diagonal(W, 0.0)
    print(f"deci: {epoch + 1} epochs, {steps} steps, auglag converged={converged}, dagness={float(constraint):.3g}", flush=True)
    np.savez(p_out, W=W, kind="probability", seconds=time.time() - t0, converged=bool(converged), steps=steps)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], json.loads(sys.argv[3]) if len(sys.argv) > 3 else {})

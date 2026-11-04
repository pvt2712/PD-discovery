"""Sections 2.28-2.29: the batch size and batches of participants with similar numbers of visits."""
import time

import numpy as np
import torch

from _setup import s1_panel
from helpers import layout, random_point, spec_for
from pdx.engine.likelihood import chunks, default_chunk, loglik, prepare

print("default_chunk with 1 thread:", default_chunk())
torch.set_num_threads(4)
print("default_chunk with 4 threads:", default_chunk())
torch.set_num_threads(1)

panel, info = s1_panel(n=60, seed=1, d=4)
spec = spec_for(panel, info)
prep = prepare(panel, spec.meas_var, spec.n_groups)
print("\nvisits per participant (sorted):", sorted(prep.Ti.tolist()))
for idx in chunks(prep, 16):
    Ti = prep.Ti[idx]
    print(f"  chunk of {len(idx):2d}: visits from {int(Ti.min())} to {int(Ti.max())}, "
          f"loop runs over {int(Ti.max()) - 1} intervals")


def work(groups):
    """Participant-intervals the loop computes (padding included)."""
    return sum(len(ix) * (int(prep.Ti[ix].max()) - 1) for ix in groups)


rng = np.random.default_rng(0)
random_order = torch.as_tensor(rng.permutation(prep.N))
unsorted = [random_order[a:a + 16] for a in range(0, prep.N, 16)]
print(f"participant-intervals computed: sorted {work(chunks(prep, 16))}, unsorted {work(unsorted)}, "
      f"actual transitions {int((prep.Ti - 1).sum())}")

lay = layout(spec, "null")
x = torch.as_tensor(random_point(panel, spec, "null", seed=7))
ref = float(loglik(x, lay, prep, chunk=prep.N))
for size in (5, 16, 32, 128):
    t0 = time.perf_counter()
    val = float(loglik(x, lay, prep, chunk=size))
    print(f"chunk {size:3d}: loglik {val:.12f}  (difference from one batch {val - ref:+.1e}, {time.perf_counter() - t0:.3f} s)")

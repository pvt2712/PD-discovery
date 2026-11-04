"""Section 2.12: blocked batched products (linalg.bmm) give the same result and avoid PyTorch's slow path.

Run single-threaded (OMP_NUM_THREADS=1), as the engine's parallel workers are. Timings depend on
the machine and on what else runs (macOS may move a single-threaded process between performance
and efficiency cores), so each timing is the best of 7 rounds. The ratios are the point.
"""
import time

import torch

from _setup import s1_panel
from helpers import layout, random_point, spec_for
from pdx.engine import likelihood, transitions
from pdx.engine.linalg import _LIMIT, bmm

torch.manual_seed(0)
print("threads:", torch.get_num_threads(), "| limit on rows x inner x columns:", _LIMIT)


def best(f, reps=10, rounds=7):
    f()
    times = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        for _ in range(reps):
            f()
        times.append((time.perf_counter() - t0) / reps * 1e3)
    return min(times)


shape = (32, 12, 6)                              # participants x nodes x intervals in one chunk
print(f"batch {shape} of d x d products; best time per forward + backward pass (ms)")
print(" d  r*k*c  block   a @ b     bmm   ratio  max|difference|")
for d in (4, 7, 8, 10):
    a = torch.randn(*shape, d, d, requires_grad=True)
    b = torch.randn(*shape, d, d)
    blk = max(1, int(((_LIMIT - 1) // d) ** 0.5)) if d ** 3 >= _LIMIT else d
    t_plain = best(lambda: (a @ b).sum().backward())
    t_block = best(lambda: bmm(a, b).sum().backward())
    diff = float((a @ b - bmm(a, b)).detach().abs().max())
    print(f"{d:2d} {d ** 3:6d} {blk:5d} {t_plain:8.2f} {t_block:7.2f} {t_plain / t_block:6.1f}   {diff:.1e}")

# the whole likelihood and gradient at ten domains, with the blocking switched off for comparison
panel, info = s1_panel(n=200, seed=4, d=10)
spec = spec_for(panel, info, treatment="emax")
prep = likelihood.prepare(panel, spec.meas_var, spec.n_groups)
lay = layout(spec, "null")
x = random_point(panel, spec, "null", seed=3)
v1, _ = likelihood.value_and_grad(x, lay, prep, chunk=32)
t_block = best(lambda: likelihood.value_and_grad(x, lay, prep, chunk=32), reps=1, rounds=5)
plain = lambda u, w: u @ w                       # noqa: E731
likelihood.bmm = transitions.bmm = plain
v2, _ = likelihood.value_and_grad(x, lay, prep, chunk=32)
t_plain = best(lambda: likelihood.value_and_grad(x, lay, prep, chunk=32), reps=1, rounds=5)
print(f"\nvalue_and_grad, {panel.n} synthetic participants, d = 10, chunks of 32:")
print(f"  with bmm {t_block:.0f} ms, with plain @ {t_plain:.0f} ms, ratio {t_plain / t_block:.1f}; "
      f"|value difference| {abs(v1 - v2):.1e}")

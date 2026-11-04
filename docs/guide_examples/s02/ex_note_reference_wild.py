"""Notes for the author: the engine bounds log-speed at the nodes (|log s| <= 15); the reference does
not. The two agree as sigma grows, until the reference stops returning (scipy's expm on blocks of
norm near 1e39), which is run here in a child process with a 20-second limit."""
import subprocess
import sys
import time

import numpy as np
import torch

from _setup import ROOT, s1_panel
from helpers import layout, random_point, spec_for
from pdx.engine.likelihood import loglik, prepare
from pdx.engine.reference import reference_loglik

panel, info = s1_panel(n=30, seed=1, d=4)
spec = spec_for(panel, info)
prep = prepare(panel, spec.meas_var, spec.n_groups)
lay = layout(spec, "null")
x = random_point(panel, spec, "null", seed=1)
for log_sigma in (np.log(0.3), 1.0, 1.5, 2.0):
    x[lay.slices["log_sigma"][0]] = log_sigma
    le = float(loglik(torch.as_tensor(x), lay, prep))
    lr = reference_loglik(x, lay, panel, prep.mu1.numpy(), prep.S1.numpy(), prep.y1c.numpy())
    print(f"sigma = {np.exp(log_sigma):5.2f}: engine {le:.8f}  reference {lr:.8f}  difference {le - lr:+.1e}")

child = f"""
import sys; sys.path[:0] = [{str(ROOT / 'docs/guide_examples/s02')!r}]
import numpy as np, torch
from _setup import s1_panel
from helpers import layout, random_point, spec_for
from pdx.engine.likelihood import loglik, prepare
from pdx.engine.reference import reference_loglik
panel, info = s1_panel(n=30, seed=1, d=4)
spec = spec_for(panel, info); prep = prepare(panel, spec.meas_var, spec.n_groups); lay = layout(spec, "null")
x = random_point(panel, spec, "null", seed=1); x[lay.slices["log_sigma"][0]] = 3.0
import time; t0 = time.perf_counter(); v = float(loglik(torch.as_tensor(x), lay, prep))
print(f"sigma = 20.09: engine {{v:.6f}} in {{time.perf_counter() - t0:.2f}} s", flush=True)
reference_loglik(x, lay, panel, prep.mu1.numpy(), prep.S1.numpy(), prep.y1c.numpy())
print("reference finished")
"""
t0 = time.time()
try:
    out = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True, timeout=20)
    print(out.stdout.strip())
except subprocess.TimeoutExpired as e:
    print((e.stdout or b"").decode().strip() if isinstance(e.stdout, bytes) else (e.stdout or "").strip())
    print(f"reference at sigma = 20.09: no result after {time.time() - t0:.0f} s (stopped)")

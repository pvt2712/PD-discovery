"""Sections 5.20-5.24: the minimal-L1 speed as a weighted median, by hand and with the code;
its breakdown; the batch version; the mode, least-squares and diagnostic speeds. Hand-made drift
entries, no fitting. Synthetic numbers only."""
import numpy as np

import _fixture  # noqa: F401  (import path)
from pdx.engine.model import ModelSpec
from pdx.localise import (speed_diagnostics, speed_ls, speed_mode, speed_rule, speed_rule_batch,
                          weighted_median)

# support entries in ModelSpec order (row-major over (to, from)): four own rates and four cross-links
sup = np.eye(4, dtype=bool)
sup[2, 0] = sup[2, 1] = sup[3, 1] = sup[3, 2] = True
spec = ModelSpec(d=4, support=sup)
links = [tuple(map(int, r)) for r in spec.sup_idx]
a0 = np.array([-0.25, -0.25, 0.10, 0.20, -0.25, 0.02, 0.25, -0.25])   # TD (reference) entries
a1 = 2.0 * a0                                                           # PIGD = TD at speed 2 ...
a1[3] = 0.90           # ... except (2,1): stronger beyond speed (2 x 0.20 = 0.40 expected)
a1[5] = -0.10          # (3,1): reference entry near zero, and a sign change
print("link (to,from):", links)
print("a0:", a0)
print("a1:", a1)

usable = (np.abs(a0) > 0.05) & (a1 / a0 > 0)
r, w = a1 / a0, np.abs(a0)
print("ratio a1/a0  :", np.round(r, 3))
print("usable       :", usable.astype(int), "(|a0| > 0.05 and same sign)")
o = np.argsort(r[usable])
print("sorted ratios:", np.round(r[usable][o], 3))
print("their weights:", w[usable][o], " cumulative:", np.cumsum(w[usable][o]), " half:", w[usable].sum() / 2)
print("weighted_median by hand -> first sorted ratio whose cumulative weight reaches half")
print("speed_rule:", speed_rule(a0, a1), "| weighted_median:", weighted_median(r[usable], w[usable]))

grid = np.linspace(0.5, 4.0, 3501)
obj_usable = [np.abs(a1[usable] - s * a0[usable]).sum() for s in grid]
obj_all = [np.abs(a1 - s * a0).sum() for s in grid]
print("grid argmin of sum|a1 - s a0| over usable entries:", round(grid[int(np.argmin(obj_usable))], 3))
print("grid argmin over all entries (sign change and near-zero entry included):", round(grid[int(np.argmin(obj_all))], 3))

print("\nbreakdown: more and more entries deviate the same way (their ratio becomes 3 instead of 2)")
order = [2, 6, 3, 0, 1]                              # (2,0), (3,2), (2,1), then two own rates
for k in range(len(order) + 1):
    b1 = 2.0 * a0
    b1[order[:k]] = 3.0 * a0[order[:k]]
    dev_w = np.abs(a0[order[:k]]).sum()
    print(f"  {k} deviating entries carrying {dev_w:.2f} of {np.abs(a0[usable]).sum():.2f}: "
          f"speed_rule = {speed_rule(a0, b1):.3f}")

print("\nbatch version equals the scalar one row by row")
rows0 = np.vstack([a0, a0 * 1.1, a0 * 0.9])
rows1 = np.vstack([a1, a1 * 1.2, a1 * 0.8])
print("  batch :", np.round(speed_rule_batch(rows0, rows1), 6))
print("  scalar:", np.round([speed_rule(x, y) for x, y in zip(rows0, rows1)], 6))

print("\ndiagnostics:", {k: round(v, 3) for k, v in speed_diagnostics(a0, a1, spec).items()})
lr_u, w_u = np.log(r[usable]), w[usable] / w[usable].sum()
sd_w = np.sqrt(np.sum(w_u * (lr_u - np.sum(w_u * lr_u)) ** 2))
mad = 1.4826 * np.median(np.abs(lr_u - np.median(lr_u)))
print(f"mode bandwidth: weighted sd {sd_w:.4f}, MAD {mad:.4f} -> 0.9 x {max(min(sd_w, mad) if min(sd_w, mad) > 0 else max(sd_w, mad), 1e-3):.4f} x {len(lr_u)}^(-1/5)")
print("mode:", round(speed_mode(a0, a1), 3), "| least squares a1.a0/a0.a0:", round(float(a1 @ a0 / (a0 @ a0)), 3),
      "=", round(speed_ls(a0, a1), 3))

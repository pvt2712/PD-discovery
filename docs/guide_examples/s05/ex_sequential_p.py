"""Section 5.7: Besag-Clifford sequential Monte Carlo p-values (hp.sequential_p) on made-up streams
of statistics. No model is fitted. Synthetic numbers only."""
import itertools

import numpy as np

import _fixture  # noqa: F401  (import path)
from pdx.hp import sequential_p


def stream(values, oks=None):
    it = iter(zip(values, oks or [True] * len(values)))
    return lambda: next(it)


print("1. early stop: observed 4.0, h = 3, n_max = 99")
r = sequential_p(stream([5.0, 1.0, 6.0, 7.0] + [0.1] * 95), observed=4.0, n_max=99, h=3)
print("  ", {k: r[k] for k in ("p", "n_drawn", "n_exceed", "stopped_early")}, "-> p = h / drawn = 3/4")

print("2. no stop: 99 replicates all below the observed")
r = sequential_p(stream([0.1] * 99), observed=4.0, n_max=99, h=20)
print("  ", {k: r[k] for k in ("p", "n_drawn", "n_exceed", "stopped_early")}, "-> p = (0 + 1)/(99 + 1)")

print("3. a tie and a failed replicate both count as at least as large")
r = sequential_p(stream([4.0, np.inf, 0.0], [True, False, True]), observed=4.0, n_max=3, h=20)
print("  ", {k: r[k] for k in ("p", "n_drawn", "n_exceed", "failures")}, "-> p = (2 + 1)/(3 + 1)")

print("4. resume from a checkpoint gives the same answer")
vals = [0.5, 5.0, 0.2, 6.0, 0.1, 7.0] + [0.0] * 50
full = sequential_p(stream(vals), observed=4.0, n_max=50, h=3)
resumed = sequential_p(stream(vals[2:]), observed=4.0, n_max=50, h=3, resume=[(0.5, True), (5.0, True)])
print("   full   :", {k: full[k] for k in ("p", "n_drawn", "n_exceed")})
print("   resumed:", {k: resumed[k] for k in ("p", "n_drawn", "n_exceed")})

print("5. monotone: turning any replicate into an exceedance never lowers p (all 2^7 sequences, h = 3)")
n_max, h, worst = 7, 3, 0.0


def p_of(ind):
    return sequential_p(stream([1.0 if e else -1.0 for e in ind]), observed=0.0, n_max=n_max, h=h)["p"]


for ind in itertools.product([0, 1], repeat=n_max):
    for j in range(n_max):
        if ind[j] == 0:
            flipped = ind[:j] + (1,) + ind[j + 1:]
            worst = min(worst, p_of(flipped) - p_of(ind))
print("   smallest change in p after a flip:", worst)

print("6. exact validity under the null and the expected number of replicates")
rng = np.random.default_rng(1)
res = [sequential_p(lambda: (rng.chisquare(7), True), rng.chisquare(7), n_max=999, h=20) for _ in range(20000)]
ps = np.array([r["p"] for r in res])
drawn = np.array([r["n_drawn"] for r in res])
for a in (0.01, 0.05, 0.10, 0.50):
    print(f"   P(p <= {a:.2f}) = {np.mean(ps <= a):.4f}  (MC SE {np.sqrt(a * (1 - a) / len(ps)):.4f})")
h, n_max = 20, 999
expected = h + h * sum(1.0 / m for m in range(h + 1, n_max + 1))
print(f"   mean replicates drawn: {drawn.mean():.1f}; formula h(1 + sum_(m=h+1..n_max) 1/m) = {expected:.1f}")
print("   attainable p-values near 0.05 when stopping early:", [round(20 / l, 4) for l in (380, 400, 420)])

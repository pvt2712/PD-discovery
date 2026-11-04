"""Sections 4.23-4.29: the comparator wrappers of comparators/run.py on synthetic family-A panels
(four domains for the helpers; six domains, 800 TD and 250 PIGD for the wrappers).
S0: equal speeds, one mechanism; S1: PIGD twice as fast, one mechanism. Neither has any mechanism
difference, so every flagged link is a false mechanism claim. Synthetic data only."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import numpy as np  # noqa: E402

from pdx.comparators import run as comp  # noqa: E402
from pdx.sim import scenarios  # noqa: E402
from pdx.step1 import Step1Config, lagged_adjacency, run_fci  # noqa: E402

panel, _ = scenarios.generate("S1", 600, np.random.default_rng(3), d=4)
X, X1, who = comp._pairs(panel)
std = comp._standardiser(panel)
both = std(np.vstack([X, X1]))
print(f"_pairs: {len(X)} complete consecutive pairs from {len(set(who))} participants")
print("pooled standardised x_t and x_t+1: means", both.mean(0).round(6), "SDs", both.std(0).round(6))
for g in (0, 1):
    print(f"   phenotype {g}: {int((panel.group[who] == g).sum())} pairs")
print("_split example:", comp._split(np.array([[True, True], [False, True]])))


def flags(split, names):
    """Flagged links of a _split dict as 'from -> to' (cross-links) and 'own lag of x' (self-lags)."""
    cl = np.array(split["cross_links"])
    out = [f"{names[k]} -> {names[j]}" for j, k in zip(*np.nonzero(cl))]
    out += [f"own lag of {names[j]}" for j in np.nonzero(split["self_lags"])[0]]
    return out or "none"


for scen in ("S0", "S1"):           # six domains, 800 TD and 250 PIGD, as in the protocol's synthetic checks
    panel, _ = scenarios.generate(scen, 4000, np.random.default_rng(3), d=6, target_counts=(800, 250))
    names = panel.domains
    print(f"\n=== {scen}: {panel.n} participants; domains {names}")
    for name, call in (("C12 DYNOTEARS", lambda: comp.c12_dynotears(panel)),
                       ("C5  tsDCI", lambda: comp.c5_tsldiffpc(panel)),
                       ("C5  tsLDiffPC", lambda: comp.c5_tsldiffpc(panel, variant="tsldiffpc")),
                       ("C3  NCT-style", lambda: comp.c3_nct(panel, n_perm=500, rng=np.random.default_rng(0))),
                       ("C11 naive", lambda: comp.c11_naive(panel, Step1Config()))):
        t0 = time.time()
        out = call()
        dt = time.time() - t0
        if name.startswith("C11"):
            print(f"{name:14s} ({dt:3.1f} s) separate graphs: {flags(out['separate_graphs'], names)}")
            print(f"{'':22s} interaction tests: {flags(out['interaction_tests'], names)}")
            for g, label in ((0, "TD"), (1, "PIGD")):            # which group found which cross-link
                sub = panel.subset(np.where(panel.group == g)[0])
                L = lagged_adjacency(*run_fci(sub, Step1Config())[:2], panel.d) & ~np.eye(panel.d, dtype=bool)
                print(f"{'':22s} {label} graph ({sub.n}): {flags({'cross_links': L, 'self_lags': []}, names)}")
            continue
        extra = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items()
                 if k not in ("cross_links", "self_lags", "p_edges", "p_edges_familywise")}
        print(f"{name:14s} ({dt:3.1f} s) flagged: {flags(out, names)}; {extra}")

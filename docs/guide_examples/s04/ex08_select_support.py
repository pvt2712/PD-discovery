"""Sections 4.12-4.14: run_fci, select_support and make_support_fn on synthetic family-A panels
(scenario S1: PIGD twice as fast, one shared mechanism). Synthetic data only."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import numpy as np  # noqa: E402

from pdx.sim import family_a, scenarios  # noqa: E402
from pdx.step1 import Step1Config, lagged_adjacency, make_support_fn, run_fci, select_support  # noqa: E402

panel, info = scenarios.generate("S1", 150, np.random.default_rng(1), d=4)
truth = family_a.true_support(info["truth"])
expert = info["support"]
print("domains:", panel.domains, "| participants:", panel.n)
print("true drift support [to, from]:\n", truth.astype(int))
print("expert map handed to the analysis (truth + candidate independence <- axial):\n", expert.astype(int))

t0 = time.time()
G, cols, n_rows, n_part, n_complete = run_fci(panel, Step1Config())
print(f"\nrun_fci: {n_rows} rows from {n_part} participants, {n_complete} complete rows, {time.time() - t0:.2f} s")
print("lagged adjacencies found [to, from]:\n", lagged_adjacency(G, cols, panel.d).astype(int))
print("select_support without the map:\n", select_support(panel).astype(int))
sup = select_support(panel, expert)
print("select_support with the map (the support of Steps 2-5):\n", sup.astype(int))
fn = make_support_fn(expert, Step1Config())
print("make_support_fn gives the same support:", np.array_equal(fn(panel), sup))

big, info_b = scenarios.generate("S1", 600, np.random.default_rng(1), d=4)
t0 = time.time()
found = lagged_adjacency(*run_fci(big, Step1Config())[:2], big.d)
print(f"\n{big.n} participants ({time.time() - t0:.2f} s): lagged adjacencies found [to, from]:\n", found.astype(int))

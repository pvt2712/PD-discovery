import numpy as np

from pdx.data import Panel
from pdx.step1 import Step1Config, fci_graph, select_support, step1_table


def _panel(n=400, T=5, seed=0):
    """Linear yearly dynamics: domain 0 -> domain 1 -> domain 2, plus persistence."""
    rng = np.random.default_rng(seed)
    B = np.diag([0.6, 0.6, 0.6])
    B[1, 0], B[2, 1] = 0.5, 0.5
    ys, ts = [], []
    for _ in range(n):
        x = rng.normal(size=3)
        rows = [x]
        for _ in range(T - 1):
            x = B @ x + rng.normal(scale=0.6, size=3)
            rows.append(x)
        ys.append(np.array(rows))
        ts.append(np.arange(1.0, T + 1))
    z = rng.normal(size=(n, 3))
    return Panel(ys, ts, rng.integers(2, size=n), z=z - z.mean(0))


def test_tiers_and_lagged_support():
    panel = _panel()
    sup = select_support(panel, cfg=Step1Config(depth=2))
    assert sup[1, 0] and sup[2, 1] and sup.diagonal().all()
    assert not sup[0, 2] and not sup[0, 1]


def test_no_edge_runs_backwards_in_time():
    panel = _panel(seed=1)
    X, tiers, who, cols, exo = step1_table(panel, Step1Config())
    G = fci_graph(X, tiers, 0.05, 2)
    t1 = [cols.index(f"y{j}_t1") for j in range(3)]
    t0 = [cols.index(f"y{j}_t") for j in range(3)]
    for a in t1:
        for b in t0:
            assert not (G[a, b] == -1 and G[b, a] == 1)       # no year t+1 -> year t arrow


def test_cluster_test_runs():
    panel = _panel(n=200, seed=2)
    sup = select_support(panel, cfg=Step1Config(depth=2, test="cluster_fisherz"))
    assert sup[1, 0] and sup[2, 1]


def test_hidden_confounding_keeps_a_circle_at_the_earlier_node():
    """A pair linked only by a hidden cause, a in year t and b in year t+1: tiers allow a o-> b
    (or a <-> b), never a --> b."""
    rng = np.random.default_rng(3)
    L = rng.normal(size=5000)
    a = L + rng.normal(size=5000)
    b = L + rng.normal(size=5000)
    G = fci_graph(np.column_stack([a, b]), [0, 1], 0.05, -1)
    assert G[1, 0] == 1 and G[0, 1] in (2, 1)


def test_testwise_deletion_keeps_incomplete_rows():
    panel = _panel(n=400, seed=4)
    rng = np.random.default_rng(5)
    y = []
    for yi in panel.y:
        yi = yi.copy()
        yi[rng.random(yi.shape) < 0.15] = np.nan
        y.append(yi)
    panel = panel.with_y(y)
    X, tiers, who, cols, exo = step1_table(panel, Step1Config())
    assert np.isnan(X).any()                                   # rows with gaps are kept
    for test in ("fisherz", "cluster_fisherz"):
        sup = select_support(panel, cfg=Step1Config(depth=2, test=test))
        assert sup[1, 0] and sup[2, 1]

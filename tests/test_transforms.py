import numpy as np
from scipy import stats

from pdx.transforms import NormalScores, orient


def test_normal_scores_midranks_and_frozen():
    x = np.array([[1.0], [2.0], [2.0], [3.0], [np.nan]])
    ns = NormalScores().fit(x)
    z = ns.transform(x)
    n = 4
    expect = stats.norm.ppf((np.array([1.0, 2.5, 2.5, 4.0]) - 0.5) / n)
    assert np.allclose(z[:4, 0], expect) and np.isnan(z[4, 0])
    assert np.allclose(ns.transform(np.array([[2.0]])), expect[1])        # frozen: same value, same score
    assert np.allclose(ns.inverse(ns.transform(x[:4])), x[:4])


def test_orientation():
    v = orient(np.array([[26.0, 90.0, 3.0]]), ["moca", "schwab_england", "tremor"])
    assert np.allclose(v, [[-26.0, -90.0, 3.0]])

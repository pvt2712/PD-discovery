import numpy as np
from statsmodels.stats.multitest import multipletests

from pdx.multiplicity import bh, by, holm, holm_adjusted, iut_similar, simultaneous_intervals


def test_against_statsmodels():
    rng = np.random.default_rng(0)
    for _ in range(50):
        p = np.concatenate([rng.uniform(size=8), rng.uniform(0, 0.01, size=3)])
        for ours, method in ((holm, "holm"), (bh, "fdr_bh"), (by, "fdr_by")):
            assert np.array_equal(ours(p, 0.05), multipletests(p, 0.05, method)[0]), method
        assert np.allclose(holm_adjusted(p), multipletests(p, 0.05, "holm")[1])


def test_simultaneous_intervals_cover_jointly():
    rng = np.random.default_rng(1)
    draws = rng.normal(size=(20000, 6))
    lo, hi = simultaneous_intervals(np.zeros(6), draws, 0.90)
    inside = np.all((draws > lo) & (draws < hi), axis=1).mean()
    assert abs(inside - 0.90) < 0.01
    assert iut_similar(lo / 10, hi / 10, eps=1.0) and not iut_similar(lo, hi, eps=0.5)

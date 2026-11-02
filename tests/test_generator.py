"""The Arm A simulator: marginals, determinism, calibration, missingness, two-slice layout."""
import numpy as np
import pytest

from escalation_discovery import expert as E
from escalation_discovery import generator as G
from escalation_discovery import missingness as M


@pytest.fixture(scope="module")
def gen():
    return G.synthetic("G1")


def test_marginal_types():
    m = G._synthetic_marginals()
    x = m["male"].ppf(np.random.default_rng(0).random(5000))
    assert set(np.unique(x)) <= {0.0, 1.0} and abs(x.mean() - 0.63) < 0.03
    d = m["ldopa"].ppf(np.random.default_rng(1).random(20000))
    assert abs((d == 0).mean() - 0.45) < 0.02 and d.min() >= 0
    s = m["se"]
    assert set(np.unique(s.ppf(np.random.default_rng(2).random(5000)))) <= set(s.levels)
    sc = m["ldopa"].score(np.array([0.0, 0.0, 400.0]))
    assert sc[0] == sc[1] < sc[2]


def test_simulation_is_deterministic_and_typed(gen):
    a, ba = G.simulate(gen, 300, seed=5)
    b, bb = G.simulate(gen, 300, seed=5)
    assert a.equals(b) and ba.equals(bb)
    assert a.groupby("PATNO").year.apply(lambda y: (np.diff(y) == 1).all()).all()
    for v, info in E.NODE_INFO.items():
        if v in a and info["type"] in ("ordinal", "count", "binary"):
            assert np.allclose(a[v], np.round(a[v]))


def test_calibration_reproduces_marginals(gen):
    long, _ = G.simulate(gen, 4000, seed=9)
    for v in ("adl", "cog", "ess"):
        mg = gen.marginals[v]
        target = float(np.sum(mg.levels * mg.probs))
        assert abs(long[v].mean() - target) < 0.15 * max(1.0, mg.levels.std())
    assert abs((long.ldopa == 0).mean() - 0.45) < 0.12
    early, late = long[long.year == 0].ldopa, long[long.year >= 6].ldopa
    assert (early > 0).mean() < 0.15 < (late > 0).mean()       # levodopa starts after enrolment


def test_missingness_mechanisms(gen):
    spec = E.load()
    long, base = G.simulate(gen, 1500, seed=3)
    bank = M.PatternBank.synthetic(spec)
    c, _ = bank.apply(long, base, "complete", 1)
    assert not c[list(spec.tv)].isna().any().any()
    mar, _ = bank.apply(long, base, "mar", 1)
    on = long.ldopa > 0
    assert mar.motor_na[on.to_numpy()].isna().mean() > 3 * mar.motor_na[~on.to_numpy()].isna().mean()
    assert (mar.motor_na.isna() == mar.axial.isna()).all()               # one OFF exam
    mnar, _ = bank.apply(long, base, "mnar", 1)
    assert len(mnar) < len(long)                                          # informative dropout
    hi = long.cog <= long.cog.quantile(0.25)
    keep = long.set_index(["PATNO", "year"]).index.isin(mnar.set_index(["PATNO", "year"]).index)
    m2 = mnar.set_index(["PATNO", "year"]).cog
    worse = m2[hi[keep].to_numpy()].isna().mean()
    better = m2[~hi[keep].to_numpy()].isna().mean()
    assert worse > better + 0.1


def test_two_slice_layout(gen):
    spec = E.load()
    long, base = G.simulate(gen, 200, seed=4)
    ts = G.two_slice(long, base, spec)
    assert list(ts.columns) == ["PATNO"] + E.two_slice_names(spec)
    first = ts.iloc[0]
    row0 = long[(long.PATNO == first.PATNO) & (long.year == first.t)].iloc[0]
    row1 = long[(long.PATNO == first.PATNO) & (long.year == first.t + 1)].iloc[0]
    assert first.adl_0 == row0.adl and first.adl_1 == row1.adl


def test_strengths_flag_weak_edges(gen):
    st = G.strengths(gen)
    assert st and all(v >= 0 for v in st.values())
    weak = G.weak_pairs(gen, threshold=1.0)                  # everything is weak at threshold 1
    assert len(weak) > 50

"""fit — the Arm A generators fitted to the pooled PPMI cohorts (PROTOCOL.md §6.1). Gated (PROTOCOL.md §11).

    uv run escd fit-generator

1. Marginals: level frequencies (at most 60 levels) or 201 quantiles per variable, a zero mass for doses.
2. Equations, per node and regime (enrolment year; later years): the node's normal score regressed on its
   expert-DAG inputs (generator._inputs) with the cohort context saturated: G1 by least squares, G2 by gradient
   boosting with out-of-fold residuals (folds partition participants) resampled as noise.
3. The trait: a one-factor model on the participant means of the residuals of the trait's children (and of baseline
   DaT binding); the loadings become the trait coefficients and leave the residual variance. The remaining
   participant-level variance is the person-effect scale of the 'person' stress setting.
4. Cohort mix, follow-up (years from enrolment to the last grid year, gaps kept as missing years) and the
   missingness pattern bank.
Aggregates are written to results/arm_a/fitted/ (fidelity: rank correlations of real and simulated two-slice tables
on the scored pairs); the generators and the pattern bank are participant-level by provenance and stay in
data/derived/arm_a/.
"""
import pickle

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import FactorAnalysis
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from . import config as C
from . import expert as E
from . import falsify as F
from . import generator as G
from . import missingness as M


def _scores(long, base, marg, spec):
    S = long[["PATNO", "year"]].copy()
    for v in spec.tv:
        S[v] = marg[v].score(long[v].to_numpy(dtype=float))
    B = pd.DataFrame({"PATNO": base.PATNO, "era2": base.era2 * 2 - 1, "genetic": base.genetic * 2 - 1})
    for b in spec.baseline:
        B[b] = marg[b].score(base[b].to_numpy(dtype=float))
    return S, B


def _design(S, B, inputs, v, regime):
    cur = S.merge(B, on="PATNO", how="left")
    if regime == "trans":
        prev = S.copy()
        prev["year"] += 1
        cur = cur.merge(prev, on=["PATNO", "year"], how="inner", suffixes=("", "@1"))
    else:
        cur = cur[cur.year == 0]
    cols = []
    for key in inputs:
        if key == "t":
            cols.append(cur.year.to_numpy(dtype=float))
        elif key.endswith("@0"):
            cols.append(cur[key[:-2]].to_numpy(dtype=float))
        elif key.endswith("@1"):
            cols.append(cur[f"{key[:-2]}@1"].to_numpy(dtype=float))
        else:
            cols.append(cur[key].to_numpy(dtype=float))
    X = np.column_stack(cols) if cols else np.zeros((len(cur), 0))
    return X, cur[v].to_numpy(dtype=float), cur.PATNO.to_numpy()


def fit_generator(long, base, kind="G1", spec=None, seed=0):
    spec = spec or E.load()
    ctx = F.saturated_context(spec)
    marg = {v: G.Marginal.from_data(long[v], E.NODE_INFO[v]["type"]) for v in spec.tv}
    marg.update({b: G.Marginal.from_data(base[b], E.NODE_INFO[b]["type"]) for b in spec.baseline})
    S, B = _scores(long, base, marg, spec)
    models, resid_person = {}, {}
    for b in spec.baseline:
        keys = G._inputs(spec, b, "base", ctx)
        X = B[keys].to_numpy(dtype=float)
        y = B[b].to_numpy(dtype=float)
        ok = ~np.isnan(y)
        A = np.column_stack([np.ones(ok.sum()), X[ok]])
        coef, *_ = np.linalg.lstsq(A, y[ok], rcond=None)
        r = y[ok] - A @ coef
        models[("base", b)] = G.NodeModel(node=b, inputs=keys, coef=coef[1:], intercept=float(coef[0]), sigma=float(r.std()))
        resid_person[b] = pd.Series(r, index=B.PATNO.to_numpy()[ok])
    for v in spec.tv:
        for regime in ("year0", "trans"):
            keys = G._inputs(spec, v, regime, ctx)
            X, y, pid = _design(S, B, keys, v, regime)
            ok = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
            if kind == "G1" or ok.sum() < 200:
                A = np.column_stack([np.ones(ok.sum()), X[ok]])
                coef, *_ = np.linalg.lstsq(A, y[ok], rcond=None)
                r = y[ok] - A @ coef
                m = G.NodeModel(node=v, inputs=keys, coef=coef[1:], intercept=float(coef[0]), sigma=float(r.std()))
            else:
                hgb = dict(max_depth=3, max_iter=200, learning_rate=0.05, min_samples_leaf=50, random_state=seed)
                r = np.empty(ok.sum())
                Xo, yo, go = X[ok], y[ok], pid[ok]
                for tr, te in GroupKFold(n_splits=5).split(Xo, yo, go):
                    r[te] = yo[te] - HistGradientBoostingRegressor(**hgb).fit(Xo[tr], yo[tr]).predict(Xo[te])
                booster = HistGradientBoostingRegressor(**hgb).fit(Xo, yo)
                m = G.NodeModel(node=v, inputs=keys, coef=np.zeros(len(keys)), booster=booster, noise="quantile",
                                resid_q=np.quantile(r - r.mean(), G.Q_GRID), sigma=float(r.std()))
            models[(regime, v)] = m
            if regime == "trans":
                resid_person[v] = pd.Series(r, index=pid[ok]).groupby(level=0).mean()
    # the trait from the participant means of its children's residuals
    kids = [k for k in G.trait_children(spec) if k in resid_person]
    P = pd.concat({k: resid_person[k] for k in kids}, axis=1).dropna()
    if len(P) > 50 and len(kids) >= 3:
        fa = FactorAnalysis(n_components=1, random_state=seed).fit(P.to_numpy())
        load = fa.components_[0]
        load = load * np.sign(load.sum()) if load.sum() else load
        for k, l in zip(kids, load):
            for regime in ("year0", "trans", "base"):
                if (regime, k) in models:
                    m = models[(regime, k)]
                    m.trait = float(l)
                    if m.noise != "quantile":
                        m.sigma = float(np.sqrt(max(m.sigma ** 2 - l ** 2, 0.05 * m.sigma ** 2)))
    for v in spec.tv:                                       # person-effect scale for the 'person' stress setting
        pm = resid_person.get(v)
        if pm is not None and len(pm) > 20:
            m = models[("trans", v)]
            m.person_sd = float(np.sqrt(max(pm.var() - m.trait ** 2 - m.sigma ** 2 / 6.0, 0.0)))
    combos = list(zip(base.era2.astype(int), base.genetic.astype(int)))
    cp = pd.Series(combos).value_counts(normalize=True)
    last = long.groupby("PATNO").year.max().rename("last")
    fu = base[["PATNO"]].assign(combo=combos).merge(last, left_on="PATNO", right_index=True, how="left").dropna()
    followup = {}
    for c, g in fu.groupby("combo"):
        yrs, n = np.unique(g["last"].astype(int) + 1, return_counts=True)
        followup[c] = (yrs, n / n.sum())
    order = G._within_year_order(spec)
    gen = G.Generator(kind=kind, spec=spec, marginals=marg, models=models, context_probs={k: float(p) for k, p in cp.items()},
                      followup=followup, context_edges=ctx, order0=order, order_t=order, source="PPMI pooled PD cohorts")
    G.calibrate(gen)
    G.strengths(gen)
    return gen


def complete_years(long):
    """Grid years from enrolment to each participant's last grid year; a missing year becomes an all-missing row."""
    rng = long.groupby("PATNO").year.max()
    idx = pd.MultiIndex.from_tuples([(p, y) for p, m in rng.items() for y in range(int(m) + 1)], names=["PATNO", "year"])
    return long.set_index(["PATNO", "year"]).reindex(idx).reset_index()


def fidelity(gen, long, base, spec, n_sim=None, seed=1):
    """Spearman correlations of every scored pair in the real and in one simulated two-slice table (aggregates)."""
    from . import truth as T
    real = G.two_slice(long, base, spec)
    sl, sb = G.simulate(gen, n_sim or base.PATNO.nunique(), seed)
    sim = G.two_slice(sl, sb, spec)
    obs = E.observed(spec)
    rows = []
    for p in T.evaluation_pairs(obs):
        a, b = sorted(p)
        x = real[[a, b]].dropna()
        y = sim[[a, b]].dropna()
        if len(x) > 30 and len(y) > 30:
            rows.append(dict(a=a, b=b, real=spearmanr(x[a], x[b])[0], simulated=spearmanr(y[a], y[b])[0]))
    df = pd.DataFrame(rows)
    df["abs_diff"] = (df.real - df.simulated).abs()
    return df


def main(argv=None):
    C.require_ppmi_approval("fitting the Arm A generators")
    from . import arm_b
    spec = E.load()
    long, base = arm_b.ppmi_panel()
    full = complete_years(long)
    out_d, out_r = C.derived_dir("arm_a"), C.results_dir("arm_a/fitted")
    for kind in ("G1", "G2"):
        gen = fit_generator(long, base, kind, spec)
        with open(out_d / f"generator_{kind}.pkl", "wb") as f:
            pickle.dump(gen, f)
        fid = fidelity(gen, long, base, spec)
        fid.to_csv(out_r / f"fidelity_{kind}.csv", index=False)
        print(kind, "median |rank-correlation difference| on scored pairs:", round(float(fid.abs_diff.median()), 3))
    bank = M.PatternBank.fit(full.merge(base[["PATNO", "era2", "genetic"]], on="PATNO"), base, spec)
    with open(out_d / "pattern_bank.pkl", "wb") as f:
        pickle.dump(bank, f)

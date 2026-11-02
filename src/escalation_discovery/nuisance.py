# Ported from escalation-study-v2/src/escalation_v2/nuisance.py (29 October 2026). Changes are marked 'escalation-discovery:'.
"""nuisance — the machine-learning layer (DESIGN.md §4): learners behind one interface, patient-grouped
cross-fitting and tuning, prediction clipping, calibration and positivity diagnostics, the learner grid.

    uv run escalation nuisance

Learners (`make_learner(name, task)`, task "reg" or "clf"; fit(X, y, groups) -> predict(X)):
  boosting        histogram gradient boosting, hyperparameters tuned by inner patient-grouped 3-fold CV (default)
  elastic_net     elastic net (regression) / elastic-net logistic (classification) on cubic-spline features
  random_forest   random forest, tuned by inner patient-grouped CV
  super_learner   stacked ensemble of the three with non-negative weights fitted on inner patient-grouped CV predictions
Rules: outer folds partition patients (K = 5); median imputation and missingness indicators are fitted on each fold's
training rows (FoldImputer); tuning sees training folds only; held-out regression predictions are
clipped to the training range of the target; probabilities to [1e-3, 1 - 1e-3] (callers trim propensities to
[0.02, 0.98] and cap density ratios at 10). Every cross-fit returns held-out R2 or AUC/Brier.
Outputs of `main`: results/learner_grid.csv (size, power, seconds per test on simulated clustered data),
results/size_sweep.csv, results/nuisance_performance.csv, results/propensity_calibration.csv, results/positivity.csv.
"""
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from scipy.optimize import nnls
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV
from sklearn.metrics import brier_score_loss, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import SplineTransformer, StandardScaler

from . import config as C
from .citest import ClusterGCM, cluster_folds, copula

LEARNERS = ["boosting", "elastic_net", "random_forest", "super_learner"]
PROPENSITY_TRIM = (0.02, 0.98)
RATIO_CAP = 10.0
INNER_FOLDS = 3


# ---------------------------------------------------------------------------- helpers
def impute(X: pd.DataFrame):
    """Median imputation on all rows with a missingness indicator for every column that has gaps; returns (array,
    names). Used only where a single complete matrix is needed (the plasmode's seed population); the estimators pass
    NaN through to `cross_fit`, which imputes inside each training fold."""
    X = X.loc[:, ~X.columns.duplicated()].copy()
    for c in list(X.columns):
        if X[c].isna().any():
            X[f"{c}_na"] = X[c].isna().astype(float)
            X[c] = X[c].fillna(X[c].median())
    X = X.loc[:, X.std(ddof=0) > 0] if X.shape[1] else X
    return X.to_numpy(dtype=float), list(X.columns)


def _inner_cv(make, X, y, groups, task):
    """Mean squared error (regression) or Brier score (classification) over inner patient-grouped folds."""
    n_groups = len(np.unique(groups))
    gkf = GroupKFold(n_splits=min(INNER_FOLDS, n_groups))
    err = []
    for tr, te in gkf.split(X, y, groups):
        if task == "clf" and len(np.unique(y[tr])) < 2:
            err.append(np.mean((y[te] - y[tr].mean()) ** 2))
            continue
        m = make().fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1] if task == "clf" else m.predict(X[te])
        err.append(np.mean((y[te] - p) ** 2))
    return float(np.mean(err))


class _Const:
    def __init__(self, v):
        self.v = v

    def predict(self, X):
        return np.full(len(X), self.v)

    def predict_proba(self, X):
        return np.column_stack([np.full(len(X), 1 - self.v), np.full(len(X), self.v)])


class _Learner:
    name = "base"

    def __init__(self, task="reg", tune=True, seed=0):
        self.task, self.tune, self.seed, self.chosen = task, tune, seed, None

    def _fit_model(self, make, X, y):
        if self.task == "clf" and len(np.unique(y)) < 2:
            return _Const(float(y.mean()))
        return make().fit(X, y)

    def predict(self, X):
        return self.model.predict_proba(X)[:, 1] if self.task == "clf" else self.model.predict(X)


class Boosting(_Learner):
    name = "boosting"
    GRID = [dict(learning_rate=0.03, max_depth=2), dict(learning_rate=0.03, max_depth=3),
            dict(learning_rate=0.1, max_depth=2), dict(learning_rate=0.1, max_depth=3)]
    DEFAULT = dict(learning_rate=0.05, max_depth=3)

    def _make(self, cfg):
        cls = HistGradientBoostingClassifier if self.task == "clf" else HistGradientBoostingRegressor
        return lambda: cls(max_iter=200, min_samples_leaf=20, l2_regularization=1.0, random_state=self.seed, **cfg)

    def fit(self, X, y, groups):
        cfg = self.DEFAULT
        if self.tune and len(np.unique(groups)) >= 3 * INNER_FOLDS:
            cfg = min(self.GRID, key=lambda c: _inner_cv(self._make(c), X, y, groups, self.task))
        self.chosen = cfg
        self.model = self._fit_model(self._make(cfg), X, y)
        return self


class SplineFeatures:
    """Cubic B-splines (5 knots, constant extrapolation) for continuous columns, passthrough for the rest, then scaling."""

    def fit(self, X):
        self.cont = [j for j in range(X.shape[1]) if np.unique(X[:, j]).size > 6]
        self.rest = [j for j in range(X.shape[1]) if j not in self.cont]
        self.st = SplineTransformer(n_knots=5, degree=3, include_bias=False, extrapolation="constant").fit(X[:, self.cont]) if self.cont else None
        self.sc = StandardScaler().fit(self._raw(X))
        return self

    def _raw(self, X):
        parts = [X[:, self.rest]] if self.rest else []
        if self.st is not None:
            parts.append(self.st.transform(X[:, self.cont]))
        return np.column_stack(parts) if parts else np.zeros((len(X), 0))

    def transform(self, X):
        return self.sc.transform(self._raw(X))


class ElasticNet(_Learner):
    name = "elastic_net"

    def fit(self, X, y, groups):
        self.feat = SplineFeatures().fit(X)
        Z = self.feat.transform(X)
        cv = GroupKFold(n_splits=min(INNER_FOLDS, len(np.unique(groups))))
        splits = list(cv.split(Z, y, groups))
        if self.task == "clf":
            if len(np.unique(y)) < 2:
                self.model = _Const(float(y.mean()))
                return self
            self.model = LogisticRegressionCV(Cs=6, l1_ratios=[0.5], solver="saga", cv=splits, max_iter=2000, scoring="neg_brier_score",
                                              random_state=self.seed).fit(Z, y)
            self.chosen = dict(C=float(self.model.C_[0]))
        else:
            self.model = ElasticNetCV(l1_ratio=[0.2, 0.5, 0.9], alphas=15, cv=splits, max_iter=5000, random_state=self.seed).fit(Z, y)
            self.chosen = dict(alpha=float(self.model.alpha_), l1_ratio=float(self.model.l1_ratio_))
        return self

    def predict(self, X):
        Z = self.feat.transform(X)
        return self.model.predict_proba(Z)[:, 1] if self.task == "clf" else self.model.predict(Z)


class RandomForest(_Learner):
    name = "random_forest"
    GRID = [dict(min_samples_leaf=5, max_features="sqrt"), dict(min_samples_leaf=5, max_features=0.5),
            dict(min_samples_leaf=20, max_features="sqrt"), dict(min_samples_leaf=20, max_features=0.5)]
    DEFAULT = dict(min_samples_leaf=10, max_features=0.5)

    def _make(self, cfg):
        cls = RandomForestClassifier if self.task == "clf" else RandomForestRegressor
        return lambda: cls(n_estimators=200, random_state=self.seed, n_jobs=4, **cfg)

    def fit(self, X, y, groups):
        cfg = self.DEFAULT
        if self.tune and len(np.unique(groups)) >= 3 * INNER_FOLDS:
            cfg = min(self.GRID, key=lambda c: _inner_cv(self._make(c), X, y, groups, self.task))
        self.chosen = cfg
        self.model = self._fit_model(self._make(cfg), X, y)
        return self


class SuperLearner(_Learner):
    """Stacked ensemble: non-negative least-squares weights on inner patient-grouped CV predictions of the base
    learners (fitted with their default hyperparameters), then the bases are refitted on all training rows."""
    name = "super_learner"

    def _bases(self):
        return [Boosting(self.task, tune=False, seed=self.seed), ElasticNet(self.task, tune=False, seed=self.seed),
                RandomForest(self.task, tune=False, seed=self.seed)]

    def fit(self, X, y, groups):
        gkf = GroupKFold(n_splits=min(INNER_FOLDS, len(np.unique(groups))))
        P = np.zeros((len(y), 3))
        for tr, te in gkf.split(X, y, groups):
            for j, b in enumerate(self._bases()):
                P[te, j] = b.fit(X[tr], y[tr], groups[tr]).predict(X[te])
        w, _ = nnls(P, y.astype(float))
        self.weights = w / w.sum() if w.sum() > 0 else np.ones(3) / 3
        self.chosen = dict(zip(["boosting", "elastic_net", "random_forest"], np.round(self.weights, 3).tolist()))
        self.models = [b.fit(X, y, groups) for b in self._bases()]
        return self

    def predict(self, X):
        p = sum(w * m.predict(X) for w, m in zip(self.weights, self.models))
        return np.clip(p, 1e-3, 1 - 1e-3) if self.task == "clf" else p


def make_learner(name, task="reg", tune=True, seed=0):
    return {"boosting": Boosting, "elastic_net": ElasticNet, "random_forest": RandomForest, "super_learner": SuperLearner}[name](task, tune, seed)


# ---------------------------------------------------------------------------- cross-fitting
class FoldImputer:
    """Median imputation with a missingness indicator for every column that has gaps, fitted on the training rows of
    one fold and applied to its held-out rows (no held-out value reaches the preprocessing); constant training
    columns are dropped. Columns are expected as floats with NaN for missing."""

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.med = np.nanmedian(X, axis=0)
        self.med[np.isnan(self.med)] = 0.0
        self.na_cols = np.where(np.isnan(X).any(axis=0))[0]
        Z = self._fill(X)
        self.keep = Z.std(axis=0) > 0
        if not self.keep.any():
            self.keep[:] = True
        return self

    def _fill(self, X):
        Z = np.where(np.isnan(X), self.med, X)
        return np.column_stack([Z, np.isnan(X[:, self.na_cols]).astype(float)]) if len(self.na_cols) else Z

    def transform(self, X):
        return self._fill(np.asarray(X, dtype=float))[:, self.keep]


def cross_fit(X, y, groups, task="reg", learner="boosting", K=5, seed=0, tune=True, fit_mask=None, extra=None):
    """Held-out predictions with folds that partition patients. X may contain NaN: median imputation and the
    missingness indicators are fitted on each fold's training rows (`FoldImputer`). `fit_mask` restricts the training
    rows (e.g. one treatment arm, uncensored rows) while predictions are made for every row; `extra` lists further
    design matrices (row-aligned with X, e.g. shifted doses) to predict for the held-out rows with the same fold
    models. Returns predictions, fold labels, held-out performance and the hyperparameters chosen in each fold."""
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    extra = [np.asarray(Xe, dtype=float) for Xe in (extra or [])]
    fold = cluster_folds(groups, K, seed)
    fit_mask = np.ones(len(y), bool) if fit_mask is None else np.asarray(fit_mask, bool)
    pred = np.zeros(len(y))
    pred_extra = [np.zeros(len(y)) for _ in extra]
    chosen, fallback, t0 = [], 0, time.time()
    for k in np.unique(fold):
        te, tr = fold == k, (fold != k) & fit_mask
        if tr.sum() < 20:      # too few training rows for a learner: predict their mean (still held out; never refit in-sample)
            const = float(y[tr].mean()) if tr.any() else float(y[fit_mask].mean())
            pred[te] = const
            for j in range(len(extra)):
                pred_extra[j][te] = const
            chosen.append(None); fallback += 1
            continue
        imp = FoldImputer().fit(X[tr])
        m = make_learner(learner, task, tune, seed).fit(imp.transform(X[tr]), y[tr], groups[tr])
        lo, hi = (1e-3, 1 - 1e-3) if task == "clf" else (y[tr].min(), y[tr].max())
        pred[te] = np.clip(m.predict(imp.transform(X[te])), lo, hi)
        for j, Xe in enumerate(extra):
            pred_extra[j][te] = np.clip(m.predict(imp.transform(Xe[te])), lo, hi)
        chosen.append(m.chosen)
    ev = fit_mask
    if task == "clf":
        score = dict(auc=float(roc_auc_score(y[ev], pred[ev])) if len(np.unique(y[ev])) > 1 else np.nan,
                     brier=float(brier_score_loss(y[ev], pred[ev])))
    else:
        score = dict(r2=float(r2_score(y[ev], pred[ev])))
    return dict(pred=pred, pred_extra=pred_extra, fold=fold, score=score, chosen=chosen, seconds=round(time.time() - t0, 1), learner=learner, fallback_folds=fallback)


def calibration_table(p, y, bins=10):
    """Mean predicted vs observed rate by decile of the prediction."""
    q = pd.qcut(pd.Series(p), bins, duplicates="drop", labels=False)
    tab = pd.DataFrame({"bin": q, "p": p, "y": y}).groupby("bin").agg(n=("y", "size"), predicted=("p", "mean"), observed=("y", "mean"))
    return tab.reset_index()


def positivity_table(e, E):
    """Propensity overlap by arm and the share of rows trimmed."""
    rows = []
    for arm, lab in ((1, "escalation"), (0, "maintenance")):
        v = e[E == arm]
        rows.append(dict(arm=lab, n=int(len(v)), p05=round(float(np.percentile(v, 5)), 3), p50=round(float(np.median(v)), 3),
                         p95=round(float(np.percentile(v, 95)), 3), min=round(float(v.min()), 3), max=round(float(v.max()), 3),
                         share_below_0_05=round(float((v < 0.05).mean()), 3), share_above_0_95=round(float((v > 0.95).mean()), 3)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------- simulated learner grid
def simulate_clustered(kind, rng, G=400, m=8):
    """Clustered data with a person-level trait and a clustered confounder Z (the design of the first implementation)."""
    g = np.repeat(np.arange(G), m)
    z = rng.normal(size=G)[g] * 0.6 + rng.normal(size=G * m) * 0.8
    u, v = rng.normal(size=G)[g], rng.normal(size=G)[g]
    e1, e2 = rng.normal(size=G * m) * 0.6, rng.normal(size=G * m) * 0.6
    if kind == "null_linear":
        x, y = 0.8 * z + u + e1, 0.8 * z + v + e2
    elif kind == "null_nonlinear":
        x, y = 0.9 * z ** 2 + u + e1, 0.9 * z ** 2 + v + e2
    elif kind == "alt_quadratic":
        x = z + u + e1
        xs = (x - x.mean()) / x.std()
        y = 0.35 * (xs ** 2 - 1) + 0.8 * z + v + e2
    else:  # alt_linear: weak linear dependence
        x = z + u + e1
        y = 0.08 * x + 0.8 * z + v + e2
    return copula(np.column_stack([x, y, z])), g


def naive_fisherz(D):
    """Row-level partial correlation test (ignores clustering)."""
    n = len(D)
    Z = np.column_stack([np.ones(n), D[:, 2:]])
    r = D[:, :2] - Z @ np.linalg.lstsq(Z, D[:, :2], rcond=None)[0]
    rho = np.corrcoef(r[:, 0], r[:, 1])[0, 1]
    zst = 0.5 * np.log((1 + rho) / (1 - rho)) * np.sqrt(n - Z.shape[1] - 2)
    return float(2 * stats.norm.sf(abs(zst)))


def _one_rep(kind, seed, tests, G, m):
    rng = np.random.default_rng(seed)
    D, g = simulate_clustered(kind, rng, G, m)
    out = {}
    for lab, spec in tests.items():
        t0 = time.time()
        if spec == "naive":
            p = naive_fisherz(D)
        else:
            learner, K, quad = spec
            p = ClusterGCM(D, g, learner=learner, cross_fit=K, quadratic=quad, seed=seed).test(0, 1, [2])
        out[lab] = (p, time.time() - t0)
    return out


def learner_grid(reps_fast=200, reps_ml=60, G=400, m=8, n_jobs=8):
    tests = {"naive Fisher-z": "naive", "cluster linear (in-sample)": ("linear", 1, False),
             "splines (in-sample)": ("splines", 1, False), "splines (cross-fitted)": ("splines", 5, False),
             "splines quadratic (cross-fitted)": ("splines", 5, True),
             "boosting": ("boosting", 5, False), "elastic net on splines": ("elastic_net", 5, False),
             "random forest": ("random_forest", 5, False), "super learner": ("super_learner", 5, False)}
    fast = {k: v for k, v in tests.items() if v == "naive" or v[0] in ("linear", "splines")}
    ml = {k: v for k, v in tests.items() if k not in fast}
    rows = []
    for kind in ("null_linear", "null_nonlinear", "alt_linear", "alt_quadratic"):
        for group, reps in ((fast, reps_fast), (ml, reps_ml)):
            res = Parallel(n_jobs=n_jobs)(delayed(_one_rep)(kind, 1000 + r, group, G, m) for r in range(reps))
            for lab in group:
                ps = np.array([r[lab][0] for r in res]); secs = np.array([r[lab][1] for r in res])
                rows.append(dict(scenario=kind, test=lab, replicates=reps, reject_rate=round(float((ps < 0.05).mean()), 3),
                                 seconds_per_test=round(float(secs.mean()), 3), patients=G, rows_per_patient=m))
        print(kind, "done", flush=True)
    return pd.DataFrame(rows)


def size_sweep(reps=200, n_jobs=8):
    tests = {"naive Fisher-z": "naive", "cluster linear (in-sample)": ("linear", 1, False), "splines (cross-fitted)": ("splines", 5, False)}
    rows = []
    for G in (50, 100, 200, 400):
        for m in (4, 8):
            for kind in ("null_linear", "null_nonlinear", "alt_linear"):
                res = Parallel(n_jobs=n_jobs)(delayed(_one_rep)(kind, 2000 + r, tests, G, m) for r in range(reps))
                for lab in tests:
                    ps = np.array([r[lab][0] for r in res])
                    rows.append(dict(patients=G, rows_per_patient=m, scenario=kind, test=lab, reject_rate=round(float((ps < 0.05).mean()), 3)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------- nuisance performance on PPMI
def ppmi_nuisance_performance(cohort="ppmi1", learners=LEARNERS):
    from .trials import HIST
    d = pd.read_csv(C.derived_dir() / f"trials_{cohort}.csv")
    base = list(C.BASELINE)
    covs = [f"h_{c}" for c in HIST if f"h_{c}" in d] + base + ["t", "trait0"]
    arm = np.where(d.delta >= 100, 1, np.where(d.delta.abs() <= 50, 0, -1))
    m = arm >= 0
    W = d.loc[m, covs].to_numpy(dtype=float); E = arm[m].astype(float); pid = d.loc[m, "PATNO"].to_numpy()   # NaN imputed inside each fold
    Y1 = d.loc[m, "y1_adl"].to_numpy(dtype=float); A = d.loc[m, "A"].to_numpy(dtype=float)
    C2 = d.loc[m, "y2_adl"].notna().to_numpy().astype(float)
    rows, calib, posit = [], None, None
    for lr in learners:
        t0 = time.time()
        e = cross_fit(W, E, pid, "clf", lr)
        rows.append(dict(cohort=cohort, nuisance="propensity P(escalate | H)", learner=lr, **e["score"], seconds=e["seconds"]))
        o = cross_fit(np.column_stack([E, W]), Y1, pid, "reg", lr)
        rows.append(dict(cohort=cohort, nuisance="outcome E[UPDRS II t+1 | E, H]", learner=lr, **o["score"], seconds=o["seconds"]))
        a = cross_fit(W, A, pid, "reg", lr)
        rows.append(dict(cohort=cohort, nuisance="dose E[A | H]", learner=lr, **a["score"], seconds=a["seconds"]))
        c = cross_fit(np.column_stack([E, W]), C2, pid, "clf", lr)
        rows.append(dict(cohort=cohort, nuisance="censoring P(UPDRS II at t+2 observed | E, H)", learner=lr, **c["score"], seconds=c["seconds"]))
        Xn = np.column_stack([A, W]); Xs = np.column_stack([A + 100, W])
        lab = np.r_[np.zeros(len(A)), np.ones(len(A))]
        r = cross_fit(np.vstack([Xn, Xs]), lab, np.r_[pid, pid], "clf", lr)
        rows.append(dict(cohort=cohort, nuisance="density-ratio classifier (natural vs +100 mg)", learner=lr, **r["score"], seconds=r["seconds"]))
        if lr == "boosting":
            et = np.clip(e["pred"], *PROPENSITY_TRIM)
            calib = calibration_table(et, E); calib.insert(0, "cohort", cohort)
            posit = positivity_table(et, E); posit.insert(0, "cohort", cohort)
            posit["share_trimmed"] = round(float(((e["pred"] < PROPENSITY_TRIM[0]) | (e["pred"] > PROPENSITY_TRIM[1])).mean()), 4)
        print(f"  {lr}: {time.time() - t0:.0f} s", flush=True)
    return pd.DataFrame(rows), calib, posit


def main(argv=None):
    C.require_ppmi_approval("the learner diagnostics")  # escalation-discovery: PROTOCOL.md §11 gate
    args = argv or []
    out = C.results_dir()
    pd.set_option("display.width", 250)
    if "--grid-only" not in args:
        print("nuisance performance on the PPMI 1.0 trial rows", flush=True)
        perf, calib, posit = ppmi_nuisance_performance()
        perf.to_csv(out / "nuisance_performance.csv", index=False)
        calib.to_csv(out / "propensity_calibration.csv", index=False)
        posit.to_csv(out / "positivity.csv", index=False)
        print(perf.to_string(index=False)); print(calib.round(3).to_string(index=False)); print(posit.to_string(index=False))
    print("\nlearner grid on simulated clustered data (400 patients x 8 rows)", flush=True)
    grid = learner_grid()
    grid.to_csv(out / "learner_grid.csv", index=False)
    print(grid.pivot(index="test", columns="scenario", values="reject_rate").to_string())
    print(grid.groupby("test").seconds_per_test.mean().round(3).to_string())
    print("\nsize sweep", flush=True)
    sw = size_sweep()
    sw.to_csv(out / "size_sweep.csv", index=False)
    print(sw.pivot_table(index=["patients", "rows_per_patient"], columns=["scenario", "test"], values="reject_rate").round(3).to_string())


if __name__ == "__main__":
    main()

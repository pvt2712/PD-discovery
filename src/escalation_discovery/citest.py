# Ported from escalation-study-v2/src/escalation_v2/citest.py (29 October 2026). Changes are marked 'escalation-discovery:'.
"""citest — the cluster-robust generalised covariance measure (GCM) for repeated patient observations.

Test of X _||_ Y | Z on rows clustered by patient (Shah & Peters 2020, clustered; DESIGN.md §3):
  1. the features of X and Y (centred value; optionally also the centred square) are residualised on Z with a
     nuisance learner, cross-fitted over folds that partition PATIENTS (K = 5; K = 1 is the in-sample test);
     held-out predictions are clipped to the training range of the target;
  2. the residual products are summed within patient;
  3. a Wald statistic with the CR1 cluster covariance over patients is referred to chi-square(k).
At zero effect the statistic is the double-machine-learning score, so the same object returns the partially
linear effect theta = sum(rX rY) / sum(rX^2) with its patient-cluster variance (`estimate`).
Learners: "splines" (additive cubic B-splines + OLS; the default inside the DAG tests), "linear" (OLS on the raw
conditioning set: the parametric clustered test used as a contrast) or any learner name of nuisance.py
("boosting", "elastic_net", "random_forest", "super_learner"); `products=True` adds the pairwise products of the
standardised conditioning variables as linear columns to the spline design (a sensitivity analysis for confounding
that runs through an interaction, training guide §4.4).
Corrections of 28 October 2026 (DESIGN.md §11.11; training guide §4.2-§4.5): the spline knots sit at de-duplicated
quantiles of each variable (eight, as before, so the column count is unchanged); a test needs at least
MIN_ROWS_PER_COLUMN training rows per design column in every fold (the guard counted variables, not columns), and a
test that fails the guard, or whose covariance is singular, returns NaN ("untestable") instead of p = 1; the effect
estimate carries the interval obtained by inverting the cluster score test, the number of design columns per training
row and the largest patient share of the information sum.
"""
import numpy as np
from scipy import stats
from scipy.stats import norm, rankdata
from sklearn.preprocessing import SplineTransformer


def copula(X):
    """Gaussian-copula (normal-score) transform, column-wise, keeping NaN."""
    X = np.array(X, dtype=float, copy=True)
    if X.ndim == 1:
        X = X[:, None]
    for j in range(X.shape[1]):
        m = ~np.isnan(X[:, j])
        n = m.sum()
        if n > 1:
            X[m, j] = norm.ppf(rankdata(X[m, j]) / (n + 1))
    return X


MIN_ROWS_PER_COLUMN = 3          # training rows per design column below which a test is untestable (E4.3)
APPROX_ROWS_PER_COLUMN = 10      # ... and below which it is flagged as approximate
CALIBRATION_COLS_PER_ROW = 0.2   # columns per training row above which the score test over-rejects (E4.1 design C, E4.3b)


def spline_basis(col, n_knots=8, degree=3):
    """Cubic B-spline basis of one column (NaN rows stay NaN), knots at de-duplicated quantiles (corrected 28 October
    2026; uniform knots left most of a skewed variable's rows to one or two basis functions); columns with <= 6
    distinct values, or fewer than 4 distinct quantile knots, enter linearly."""
    col = np.asarray(col, dtype=float)
    ok = ~np.isnan(col)
    kn = np.unique(np.quantile(col[ok], np.linspace(0, 1, n_knots))) if ok.any() else np.array([])
    if np.unique(col[ok]).size <= 6 or kn.size < 4:
        out = np.full((len(col), 1), np.nan)
        out[ok, 0] = col[ok]
        return out
    st = SplineTransformer(knots=kn[:, None], degree=degree, include_bias=False).fit(col[ok, None])
    b = st.transform(col[ok, None])
    out = np.full((len(col), b.shape[1]), np.nan)
    out[ok] = b
    return out


def cluster_folds(groups, K, seed=0):
    """Fold label per row such that every cluster's rows share one fold."""
    _, inv = np.unique(groups, return_inverse=True)
    G = inv.max() + 1
    if K <= 1:
        return np.zeros(len(groups), dtype=int)
    return (np.random.default_rng(seed).permutation(G) % K)[inv]


def cluster_sums(values, groups):
    """Sum of row contributions within cluster -> (G x k) array, and the cluster sizes."""
    _, inv = np.unique(groups, return_inverse=True)
    G = inv.max() + 1
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    S = np.zeros((G, values.shape[1]))
    np.add.at(S, inv, values)
    return S, np.bincount(inv)


def cluster_wald(prod, groups):
    """Wald test that E[prod] = 0 with the CR1 sandwich over clusters; prod is n x k."""
    S, _ = cluster_sums(prod, groups)
    G = S.shape[0]
    tot = S.sum(0)
    V = G / (G - 1) * (S.T @ S)
    try:
        stat = float(tot @ np.linalg.solve(V, tot))
    except np.linalg.LinAlgError:
        return np.nan, S.shape[1], np.nan          # untestable, not "independent" (corrected 28 October 2026)
    if not np.isfinite(stat):
        return np.nan, S.shape[1], np.nan
    return stat, S.shape[1], float(stats.chi2.sf(stat, S.shape[1]))


def effective_clusters(groups):
    """Kish-type effective number of clusters, (sum n_g)^2 / sum n_g^2."""
    n = np.bincount(np.unique(groups, return_inverse=True)[1])
    return float(n.sum() ** 2 / (n ** 2).sum())


class ClusterGCM:
    """Cached cluster-robust GCM tests on one data matrix (columns = variables, NaN allowed)."""

    def __init__(self, data, groups, learner="splines", cross_fit=5, quadratic=False, n_knots=8, seed=0,
                 min_rows=30, min_groups=20, tune=False, products=False):
        """Rows with a missing value in x, y or any conditioning variable are left out of that test (complete-case
        conditioning: a missing-indicator fill of the conditioning set biases the estimate when a confounder is
        missing, because x then carries the confounder's information)."""
        self.data = np.asarray(data, dtype=float)
        _, self.groups = np.unique(np.asarray(groups), return_inverse=True)
        self.learner, self.K, self.quadratic, self.tune, self.products = learner, int(cross_fit), quadratic, tune, products
        self.min_rows, self.min_groups = min_rows, min_groups
        self.fold = cluster_folds(self.groups, self.K, seed)
        self.bases = [spline_basis(self.data[:, j], n_knots) for j in range(self.data.shape[1])] if learner == "splines" else None
        self.cache = {}
        self.untestable = 0             # tests that failed the guard or had a singular covariance

    # -- nuisance residualisation -------------------------------------------------------------
    def _features(self, v):
        v = v - v.mean()
        return np.column_stack([v, v ** 2 - (v ** 2).mean()]) if self.quadratic else v[:, None]

    def _products(self, ok, cond):
        """Pairwise products of the standardised conditioning variables (the `products` sensitivity)."""
        if not (self.products and len(cond) >= 2):
            return []
        Z = self.data[ok][:, cond]
        Z = (Z - Z.mean(0)) / np.where(Z.std(0) > 0, Z.std(0), 1.0)
        return [Z[:, a] * Z[:, b] for a in range(len(cond)) for b in range(a + 1, len(cond))]

    def n_columns(self, cond):
        """Design columns of the residualisation (intercept, splines or raw columns, products); NaN for a learner."""
        if self.learner == "splines":
            return 1 + sum(self.bases[j].shape[1] for j in cond) + (len(cond) * (len(cond) - 1) // 2 if self.products and len(cond) >= 2 else 0)
        if self.learner == "linear":
            return 1 + len(cond)
        return np.nan

    def guard(self, ok, cond):
        """(passes, smallest number of training rows over the folds, design columns): a test needs min_rows rows,
        min_groups patients and, for a least-squares learner, MIN_ROWS_PER_COLUMN training rows per design column
        in every fold."""
        g = self.groups[ok]
        n_tr = int(min(((self.fold != k) & ok).sum() for k in np.unique(self.fold))) if self.K > 1 else int(ok.sum())
        cols = self.n_columns(cond)
        passes = ok.sum() >= self.min_rows and len(np.unique(g)) >= self.min_groups and (np.isnan(cols) or n_tr >= MIN_ROWS_PER_COLUMN * cols)
        return passes, n_tr, cols

    def _residualise(self, ok, cond, F):
        fold = self.fold[ok]
        if self.learner in ("splines", "linear"):
            B = np.column_stack([np.ones(ok.sum())] + ([self.bases[j][ok] for j in cond] if self.learner == "splines" else [self.data[ok][:, cond]])
                                + self._products(ok, cond))
            if self.K <= 1:
                coef, *_ = np.linalg.lstsq(B, F, rcond=None)
                return F - B @ coef
            R = np.empty_like(F)
            for k in np.unique(fold):
                te, tr = fold == k, fold != k
                if tr.sum() < B.shape[1] + 5:
                    coef, *_ = np.linalg.lstsq(B, F, rcond=None)
                else:
                    coef, *_ = np.linalg.lstsq(B[tr], F[tr], rcond=None)
                lo, hi = F[tr].min(0), F[tr].max(0)
                R[te] = F[te] - np.clip(B[te] @ coef, lo, hi)
            return R
        from .nuisance import make_learner
        Z = self.data[ok][:, cond] if cond else np.zeros((ok.sum(), 0))
        if Z.shape[1] == 0:
            return F - F.mean(0)
        R = np.empty_like(F)
        folds = np.unique(fold) if self.K > 1 else [None]
        for k in folds:
            te, tr = (fold == k, fold != k) if k is not None else (np.ones(len(F), bool), np.ones(len(F), bool))
            for j in range(F.shape[1]):
                m = make_learner(self.learner, "reg", tune=self.tune)   # fixed defaults inside a test unless asked
                m.fit(Z[tr], F[tr, j], self.groups[ok][tr])
                R[te, j] = F[te, j] - np.clip(m.predict(Z[te]), F[tr, j].min(), F[tr, j].max())
        return R

    def _rows(self, targets, cond):
        return ~np.isnan(self.data[:, list(targets) + list(cond)]).any(axis=1)

    def _prepare(self, x, y, cond):
        cond = sorted(set(cond) - {x, y})
        ok = self._rows([x, y], cond)
        g = self.groups[ok]
        passes, self.last_train_rows, self.last_columns = self.guard(ok, cond)
        if not passes:
            return None
        Fx, Fy = self._features(self.data[ok, x]), self._features(self.data[ok, y])
        R = self._residualise(ok, cond, np.column_stack([Fx, Fy]))
        return R[:, :Fx.shape[1]], R[:, Fx.shape[1]:], g, ok

    # -- public ----------------------------------------------------------------------------------
    def test(self, x, y, cond=()):
        """p-value of X _||_ Y | cond (column indices)."""
        key = (min(x, y), max(x, y), tuple(sorted(set(cond) - {x, y})))
        if key in self.cache:
            return self.cache[key]
        prep = self._prepare(x, y, cond)
        if prep is None:
            p = np.nan                      # untestable (was p = 1, i.e. "independent")
        else:
            rx, ry, g, _ = prep
            prod = (rx[:, :, None] * ry[:, None, :]).reshape(len(rx), -1)
            p = cluster_wald(prod, g)[2]
        if np.isnan(p):
            self.untestable += 1
        self.cache[key] = p
        return p

    __call__ = test

    def estimate(self, x, y, cond=(), weights=None):
        """Partially linear effect of x on y given cond (linear features): theta, cluster SE, Wald and score p.
        `weights` (one per row of the data matrix, e.g. inverse probabilities of observing a post-decision node) weight
        the residual products, as in the censoring-weighted residual-on-residual estimator; the nuisance regressions
        are fitted on the complete rows unweighted."""
        prep = self._prepare(x, y, cond)
        if prep is None:
            return None
        rx, ry, g, ok = prep
        ra, rb = rx[:, 0], ry[:, 0]
        w = np.ones(len(ra)) if weights is None else np.asarray(weights, dtype=float)[ok]
        J = (w * ra ** 2).sum()
        theta = float((w * ra * rb).sum() / J)
        S, n_g = cluster_sums(w * ra * (rb - theta * ra), g)
        G = len(n_g)
        se = float(np.sqrt(G / (G - 1) * ((S[:, 0] - S[:, 0].mean()) ** 2).sum()) / J)
        p_score = cluster_wald(w * ra * rb, g)[2]
        a_i, b_i = cluster_sums(w * ra * rb, g)[0][:, 0], cluster_sums(w * ra ** 2, g)[0][:, 0]
        lo_s, hi_s = score_interval(a_i, b_i)
        return dict(theta=theta, se=se, p_wald=float(2 * norm.sf(abs(theta / se))) if se > 0 else 1.0, p_score=p_score,
                    lo_score=lo_s, hi_score=hi_s, rows=int(len(ra)), patients=int(G), g_star=round(float(n_g.sum() ** 2 / (n_g ** 2).sum()), 1),
                    columns=self.last_columns, train_rows=self.last_train_rows,
                    cols_per_train_row=round(self.last_columns / self.last_train_rows, 3) if not np.isnan(self.last_columns) else np.nan,
                    max_patient_share=round(float(b_i.max() / b_i.sum()), 4) if b_i.sum() > 0 else np.nan)

    def block_test(self, xs, y, cond=()):
        """Grouped test that a block of variables xs (e.g. the pre-visit trajectory of the motor and functional scales)
        is independent of y given cond: each x and y are residualised on cond with the nuisance learner (cross-fitted by
        patient), the n x m matrix of residual products r_x * r_y is summed within patient and the CR1 Wald statistic
        is referred to chi-square(m). Also returns the joint coefficients of r_y on the m residuals (each variable
        adjusted for the rest of the block and for cond) with their patient-cluster SEs. Used for collinear clinical
        domains, where single-variable tests leave too little unique variance (DESIGN.md Pillar 1)."""
        xs = list(xs)
        cond = sorted(set(cond) - set(xs) - {y})
        ok = self._rows(xs + [y], cond)
        g = self.groups[ok]
        passes, n_tr, cols = self.guard(ok, cond)
        if not passes:
            return None
        F = np.column_stack([self.data[ok, c] - self.data[ok, c].mean() for c in xs + [y]])
        R = self._residualise(ok, cond, F)
        rx, ry = R[:, :len(xs)], R[:, len(xs)]
        stat, k, p = cluster_wald(rx * ry[:, None], g)
        J = rx.T @ rx
        try:
            coef, Jinv = np.linalg.solve(J, rx.T @ ry), np.linalg.inv(J)
        except np.linalg.LinAlgError:          # a block column without residual variation (cluster_wald then gives
            coef, Jinv = np.full(len(xs), np.nan), np.full_like(J, np.nan)   # p = NaN, untestable): no coefficients
        e = ry - rx @ coef
        S, n_g = cluster_sums(rx * e[:, None], g)
        G = len(n_g)
        cov = Jinv @ (G / (G - 1) * S.T @ S) @ Jinv
        return dict(stat=stat, df=k, p=p, coef=coef, se=np.sqrt(np.diag(cov)), rows=int(ok.sum()), patients=int(G),
                    g_star=round(float(n_g.sum() ** 2 / (n_g ** 2).sum()), 1), columns=cols, train_rows=n_tr,
                    cols_per_train_row=round(cols / n_tr, 3) if not np.isnan(cols) else np.nan)


def score_interval(a, b, level=0.95):
    """The interval of theta values the cluster score test does not reject: with per-patient sums a_i = sum w rU rV
    and b_i = sum w rU^2, the score at theta0 is S_i = a_i - theta0 b_i and the statistic (sum S_i)^2 /
    (G/(G-1) sum S_i^2) is referred to chi-square(1) (the p_score of `estimate` at theta0 = 0). Solving the quadratic
    inequality gives (lo, hi); an unbounded or empty set gives (-inf, inf) or (nan, nan)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    G = len(a)
    c = stats.chi2.ppf(level, 1) * G / (G - 1)
    A, B = a.sum(), b.sum()
    qa = B ** 2 - c * (b ** 2).sum()
    qb = -2 * (A * B - c * (a * b).sum())
    qc = A ** 2 - c * (a ** 2).sum()
    disc = qb ** 2 - 4 * qa * qc
    if qa <= 0 or disc < 0:
        return (-np.inf, np.inf) if qa <= 0 else (np.nan, np.nan)
    r = np.sqrt(disc)
    return float((-qb - r) / (2 * qa)), float((-qb + r) / (2 * qa))


def main(argv=None):
    print(__doc__)

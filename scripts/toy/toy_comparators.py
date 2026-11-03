"""Toy simulation 4 (synthetic data only, no PPMI): do named difference-graph and
shared-plus-deviation methods read a speed difference as a mechanism difference, and do
speed-adjusted versions of the same machinery avoid it?

Truth: toy 1's six domains and seven links, dX = s_i (A_g X + b_g) dt + sqrt(s_i) G dW,
sampled yearly; log s_i = log s_g + 0.4 u_i; PIGD has extra load on Axial.
  S0  equal speed (s_PIGD = 1), load difference only
  S1  PIGD twice as fast (s_PIGD = 2), load difference, shared A
  S2  S1 plus a PIGD-only link Axial -> Independence (0.12)
  S3  S1 plus Function -> Independence stronger in PIGD (0.25 -> 0.40)

Naive comparators run on the yearly scale. Each is an analogue of a named class, written
here from the published definitions, not the authors' code:
  N1  link-wise group-interaction tests, BH 5% (v0.3; multigroup RI-CLPM / panel GVAR /
      moderated VAR tested link by link); N1o the omnibus invariance test of all lagged effects
  N2  node-level shift detection: phenotype -> node given the lagged parents (JCI / S-FCI
      context edges, CD-NOD changing modules, MSS, iSCAN, FANS); N2b slopes only
  N3  difference-graph search with temporal tiers (LDiffPC-style coefficient-equality search,
      the DCI / DDAG target restricted to lagged edges)
  N4  shared effects plus group deviations, lasso (data-shared lasso, multi-VAR, joint Granger);
      cross-validated (N4) or BIC (N4b) tuning
  N5  naive continuous-time comparison (group-specific drift, entry-wise, BH 5%)
  N6  global network-invariance permutation test (NCT / NetANOVA-type omnibus)
  N7  separate graphs: link significant in one phenotype only
Speed-adjusted versions (Step 2): person-level random speed, drift support = the map:
  O1  speed-only likelihood-ratio test (13 df)
  O2  speed-adjusted joint discovery: A_PIGD = s (A + D), adaptive lasso on D, BIC
  O3  per-link score tests of D = 0 against the speed-scaled reference, BH 5%
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import itertools  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from numpy.polynomial.hermite import hermgauss  # noqa: E402
from scipy import linalg, optimize, stats  # noqa: E402
from sklearn.linear_model import LassoCV, LassoLarsIC  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import toy_speed_vs_pathway as T  # noqa: E402  (reuses ols, bh)

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

NAMES = ["Brady", "Axial", "Cognition", "Mood", "Function", "Independence"]
P = len(NAMES)
LINKS = [(1, 0), (4, 0), (4, 1), (5, 4), (5, 2), (2, 3), (3, 1)]  # (to, from), as in toy 1
EXTRA = (5, 1)  # Axial -> Independence
STRONG = (5, 4)  # Function -> Independence
MAP = LINKS + [EXTRA]  # Step 1's map: every link any group might use
SUP = np.eye(P, dtype=bool)
for _i, _j in MAP:
    SUP[_i, _j] = True
SUP_IDX = np.argwhere(SUP)  # row-major, matches A[SUP]
NS = len(SUP_IDX)
SUP_OFF = [k for k, (i, j) in enumerate(SUP_IDX) if i != j]
OFF = [(i, j) for i in range(P) for j in range(P) if i != j]
NT = P * (P + 1) // 2
SD_LOGS = 0.4
LOAD = 0.05  # extra PIGD push on Axial
Q_TRUE = 0.10 * np.eye(P)
SCEN = {"S0": (1.0, None), "S1": (2.0, None), "S2": (2.0, "extra"), "S3": (2.0, "strong")}
TRUE_DIFF = {"S0": set(), "S1": set(), "S2": {EXTRA}, "S3": {STRONG}}
GH_X, GH_W = hermgauss(12)
U_NODES = torch.tensor(np.sqrt(2) * GH_X)
LOG_W = torch.tensor(np.log(GH_W / np.sqrt(np.pi)))
I_SUP = torch.tensor(SUP_IDX[:, 0])
J_SUP = torch.tensor(SUP_IDX[:, 1])
TRI = torch.tril_indices(P, P)
LOG2PI = float(np.log(2 * np.pi))


def drift(kind=None):
    A = -0.20 * np.eye(P)
    for (i, j), v in zip(LINKS, [0.15, 0.10, 0.20, 0.25, 0.15, 0.10, 0.08]):
        A[i, j] = v
    if kind == "extra":
        A[EXTRA] = 0.12
    if kind == "strong":
        A[STRONG] = 0.40
    return A


def transition_np(A, b, Qm, s):
    """Exact yearly transition for speed s: X' = E X + a + w, w ~ N(0, S)."""
    M1 = np.zeros((P + 1, P + 1))
    M1[:P, :P] = A
    M1[:P, P] = b
    F1 = linalg.expm(M1 * s)
    M2 = np.zeros((2 * P, 2 * P))
    M2[:P, :P] = -A
    M2[:P, P:] = Qm
    M2[P:, P:] = A.T
    F2 = linalg.expm(M2 * s)
    S = F2[P:, P:].T @ F2[:P, P:]
    return F1[:P, :P], F1[:P, P], (S + S.T) / 2


def simulate(rng, scenario, n1=800, n2=250, Tn=5):
    s_pigd, kind = SCEN[scenario]
    groups = []
    for g, (n, s_med) in enumerate([(n1, 1.0), (n2, s_pigd)]):
        A = drift(kind if g == 1 else None)
        b = -A @ np.ones(P)
        if g == 1:
            b[1] += LOAD
        s = s_med * np.exp(SD_LOGS * rng.normal(size=n))
        X = np.zeros((n, Tn + 1, P))
        X[:, 0] = rng.normal(0, 0.3, size=(n, P))
        for i in range(n):
            E, a, S = transition_np(A, b, Q_TRUE, s[i])
            W = rng.normal(size=(Tn, P)) @ np.linalg.cholesky(S).T
            for t in range(Tn):
                X[i, t + 1] = E @ X[i, t] + a + W[t]
        groups.append(X)
    return groups


def stack(groups):
    Xt = np.concatenate([X[:, :-1].reshape(-1, P) for X in groups])
    Yt = np.concatenate([X[:, 1:].reshape(-1, P) for X in groups])
    (n1, T1, _), (n2, _, _) = groups[0].shape, groups[1].shape
    g = np.repeat([0.0, 1.0], [n1 * (T1 - 1), n2 * (T1 - 1)])
    pid = np.repeat(np.arange(n1 + n2), T1 - 1)
    return Xt, Yt, g, pid


def person_grams(groups):
    """Per-person cross-products of [1, X(t)] with itself and with X(t+1)."""
    out = []
    for X in groups:
        n, T1, _ = X.shape
        Z = np.concatenate([np.ones((n, T1 - 1, 1)), X[:, :-1]], axis=2)
        out.append((np.einsum("ntk,ntl->nkl", Z, Z), np.einsum("ntk,ntl->nkl", Z, X[:, 1:])))
    return out


def B_from(G, H):
    return np.linalg.solve(G.sum(0), H.sum(0))[1:].T


# ---------------------------------------------------------------- naive comparators

def interaction_tests(Xt, Yt, g, pid):
    """N1, N1o, N2, N2b: pooled OLS of X(t+1) on [1, G, X(t), G*X(t)], person-clustered."""
    n = len(Xt)
    Z = np.column_stack([np.ones(n), g, Xt, g[:, None] * Xt])
    k = Z.shape[1]
    ZtZi = np.linalg.inv(Z.T @ Z)
    C = ZtZi @ Z.T @ Yt
    E = Yt - Z @ C
    npers = int(pid.max()) + 1
    Tn = n // npers
    S = np.concatenate([(Z * E[:, [j]]).reshape(npers, Tn, k).sum(1) for j in range(P)], axis=1)
    meat = S.T @ S * (npers / (npers - 1)) * ((n - 1) / (n - k))
    Binv = linalg.block_diag(*[ZtZi] * P)
    V = Binv @ meat @ Binv
    theta = C.T.ravel()  # outcome-major: theta[j * k + m] = C[m, j]
    inter = np.array([[j * k + 2 + P + m for m in range(P)] for j in range(P)])
    z = theta[inter] / np.sqrt(np.diag(V)[inter])
    p_link = 2 * stats.norm.sf(np.abs(z))  # p_link[j, m]: link m -> j
    I = inter.ravel()
    w = theta[I] @ np.linalg.solve(V[np.ix_(I, I)], theta[I])
    p_omni = stats.chi2.sf(w, len(I))

    def wald(idx):
        idx = np.asarray(idx)
        return stats.chi2.sf(theta[idx] @ np.linalg.solve(V[np.ix_(idx, idx)], theta[idx]), len(idx))

    p_node = np.array([wald([j * k + 1] + list(inter[j])) for j in range(P)])
    p_node_slope = np.array([wald(inter[j]) for j in range(P)])
    return p_link, p_omni, p_node, p_node_slope


def diff_graph_search(Xt, Yt, g, alpha=0.05):
    """N3: keep the lagged difference edge k -> j unless some set S of other lagged domains
    makes the coefficient of X_k(t) in the regression of X_j(t+1) on {X_k(t)} + S equal
    across phenotypes (two-sample z-test at alpha, no multiplicity correction, as in PC)."""
    gs = []
    for gg in (0.0, 1.0):
        Xg, Yg = Xt[g == gg], Yt[g == gg]
        Z = np.column_stack([np.ones(len(Xg)), Xg])
        gs.append((Z.T @ Z, Z.T @ Yg, (Yg ** 2).sum(0), len(Xg)))
    subsets = [c for r in range(P) for c in itertools.combinations(range(P), r)]
    keep = np.zeros((P, P), bool)
    for j in range(P):
        for k in range(P):
            kept = True
            for S in subsets:
                if k in S:
                    continue
                cols = [0, 1 + k] + [1 + m for m in S]
                est = []
                for G, ZY, yy, n in gs:
                    Gs = G[np.ix_(cols, cols)]
                    Gi = np.linalg.inv(Gs)
                    bs = Gi @ ZY[cols, j]
                    s2 = (yy[j] - bs @ ZY[cols, j]) / (n - len(cols))
                    est.append((bs[1], np.sqrt(s2 * Gi[1, 1])))
                z = (est[1][0] - est[0][0]) / np.hypot(est[0][1], est[1][1])
                if 2 * stats.norm.sf(abs(z)) > alpha:
                    kept = False
                    break
            keep[j, k] = kept
    return keep


def shared_lasso(Xt, Yt, g, pid, tuning):
    """N4 / N4b: beta_g = beta + D_g with penalty lam (|beta|_1 + sum_g r_g |D_g|_1),
    r_g = sqrt(n_g / N); group intercepts unpenalised (within-group centring)."""
    n1, n2 = int((g == 0).sum()), int((g == 1).sum())
    r = (np.sqrt(n1 / (n1 + n2)), np.sqrt(n2 / (n1 + n2)))
    Xc, Yc = Xt.copy(), Yt.copy()
    for gg in (0.0, 1.0):
        m = g == gg
        Xc[m] -= Xc[m].mean(0)
        Yc[m] -= Yc[m].mean(0)
    D = np.column_stack([Xc, Xc * (g == 0)[:, None] / r[0], Xc * (g == 1)[:, None] / r[1]])
    folds = list(GroupKFold(5).split(D, groups=pid))
    flags = np.zeros((P, P), bool)
    for j in range(P):
        if tuning == "cv":
            m = LassoCV(cv=folds, fit_intercept=False, alphas=60, max_iter=20000).fit(D, Yc[:, j])
        else:
            m = LassoLarsIC(criterion="bic", fit_intercept=False).fit(D, Yc[:, j])
        c = m.coef_
        flags[j] = np.abs(c[2 * P:] / r[1] - c[P:2 * P] / r[0]) > 1e-10
    return flags


def naive_ct(grams, rng, nboot=100):
    """N5: matrix logarithm of each phenotype's yearly OLS matrix, person bootstrap SEs."""
    est, se = [], []
    for G, H in grams:
        n = len(G)
        A0 = np.real(linalg.logm(B_from(G, H)))
        boots = []
        for _ in range(nboot):
            idx = rng.integers(0, n, n)
            boots.append(np.real(linalg.logm(B_from(G[idx], H[idx]))))
        est.append(A0)
        se.append(np.std(boots, axis=0, ddof=1))
    z = (est[1] - est[0]) / np.hypot(se[0], se[1])
    return 2 * stats.norm.sf(np.abs(z))


def perm_test(grams, rng, nperm=200):
    """N6: permutation test of phenotype labels for the yearly network difference
    (Frobenius norm, and the largest single difference as in NCT's structure test)."""
    G = np.concatenate([grams[0][0], grams[1][0]])
    H = np.concatenate([grams[0][1], grams[1][1]])
    n1 = len(grams[0][0])
    lab = np.repeat([0, 1], [n1, len(G) - n1])

    def stat(lb):
        d = B_from(G[lb == 1], H[lb == 1]) - B_from(G[lb == 0], H[lb == 0])
        return np.sqrt((d ** 2).sum()), np.abs(d).max()

    obs = stat(lab)
    perm = np.array([stat(rng.permutation(lab)) for _ in range(nperm)])
    return [(1 + (perm[:, c] >= obs[c]).sum()) / (nperm + 1) for c in (0, 1)]


def separate_graphs(Xt, Yt, g):
    """N7: per-phenotype OLS; a link significant (5%) in one phenotype only."""
    sig = []
    for gg in (0.0, 1.0):
        B, se = T.ols(Xt[g == gg], Yt[g == gg])
        sig.append(2 * stats.norm.sf(np.abs(B / se)) < 0.05)
    return sig[0] != sig[1]


# ---------------------------------------------------------------- speed-adjusted models

def chol(v):
    L = torch.zeros(P, P).index_put((TRI[0], TRI[1]), v)
    d = torch.diagonal(L)
    return L - torch.diag(d) + torch.diag(torch.exp(d))


def unpack(x, model):
    """'H0': shared A, PIGD median log-speed gamma; 'HA': free PIGD drift (gamma absorbed);
    'J': PIGD drift A + D run at speed exp(gamma) (speed-adjusted joint model)."""
    k = 0

    def take(m):
        nonlocal k
        v = x[k:k + m]
        k += m
        return v

    A = torch.zeros(P, P).index_put((I_SUP, J_SUP), take(NS))
    A2 = torch.zeros(P, P).index_put((I_SUP, J_SUP), take(NS)) if model == "HA" else A
    bT, bP = take(P), take(P)
    LT, LP = chol(take(NT)), chol(take(NT))
    gamma = take(1)[0] if model in ("H0", "J") else torch.zeros(())
    sigma = torch.exp(take(1)[0])
    if model == "J":
        A2 = A + torch.zeros(P, P).index_put((I_SUP, J_SUP), take(NS))
    return A, A2, bT, bP, LT, LP, gamma, sigma


def group_nll(X, Y, A, b, L, mlog, sigma):
    """Minus the marginal log-likelihood of one phenotype, integrating log-speed (12-node
    Gauss-Hermite); drift, load and noise are all run at the person's speed."""
    s = torch.exp(mlog + sigma * U_NODES)
    K = s.shape[0]
    n, Tn, _ = X.shape
    M1 = torch.cat([torch.cat([A, b[:, None]], 1), torch.zeros(1, P + 1)], 0)
    F1 = torch.linalg.matrix_exp(s[:, None, None] * M1)
    E, a = F1[:, :P, :P], F1[:, :P, P]
    M2 = torch.cat([torch.cat([-A, L @ L.T], 1), torch.cat([torch.zeros(P, P), A.T], 1)], 0)
    F2 = torch.linalg.matrix_exp(s[:, None, None] * M2)
    S = F2[:, P:, P:].transpose(1, 2) @ F2[:, :P, P:]
    Ls = torch.linalg.cholesky(0.5 * (S + S.transpose(1, 2)))
    logdet = 2 * torch.log(torch.diagonal(Ls, dim1=1, dim2=2)).sum(-1)
    R = Y[None] - torch.einsum("ntp,kqp->kntq", X, E) - a[:, None, None, :]
    Zs = torch.linalg.solve_triangular(Ls, R.reshape(K, -1, P).transpose(1, 2), upper=False)
    q = (Zs ** 2).sum(1).reshape(K, n, Tn).sum(-1)
    ll = -0.5 * (Tn * (P * LOG2PI + logdet)[:, None] + q)
    return -torch.logsumexp(ll + LOG_W[:, None], 0).sum()


def nll(x, model, data):
    A, A2, bT, bP, LT, LP, gamma, sigma = unpack(x, model)
    (XT, YT), (XP, YP) = data
    return group_nll(XT, YT, A, bT, LT, torch.zeros(()), sigma) + group_nll(XP, YP, A2, bP, LP, gamma, sigma)


def make_obj(model, data, lam=0.0, w=None, n0=None, fixed_zero=None):
    """Value and gradient for L-BFGS-B. With lam > 0 (model 'J'), D = u - v with u, v >= 0
    and penalty lam * sum w (u + v). fixed_zero masks D entries held at 0 (refits)."""
    wt = None if w is None else torch.tensor(w)

    def f(x):
        xt = torch.tensor(x, requires_grad=True)
        try:
            if lam > 0:
                u, v = xt[n0:n0 + NS], xt[n0 + NS:]
                th = torch.cat([xt[:n0], u - v])
                val = nll(th, model, data) + lam * (wt * (u + v)).sum()
            elif fixed_zero is not None:
                th = torch.cat([xt[:-NS], xt[-NS:] * torch.tensor(~fixed_zero, dtype=xt.dtype)])
                val = nll(th, model, data)
            else:
                val = nll(xt, model, data)
            if not torch.isfinite(val):
                return 1e10, np.zeros_like(x)
            val.backward()
            return float(val.detach()), xt.grad.numpy().copy()
        except RuntimeError:
            return 1e10, np.zeros_like(x)

    return f


def minimize(f, x0, bounds=None):
    return optimize.minimize(f, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                             options={"maxiter": 5000, "maxfun": 20000})


def lq_of(Qm):
    L = np.linalg.cholesky(Qm)
    L[np.diag_indices(P)] = np.log(np.diag(L))
    return L[np.tril_indices(P)]


def Q_of(lq):
    L = np.zeros((P, P))
    L[np.tril_indices(P)] = lq
    L[np.diag_indices(P)] = np.exp(np.diag(L))
    return L @ L.T


def ols_group(X):
    Xt, Yt = X[:, :-1].reshape(-1, P), X[:, 1:].reshape(-1, P)
    Z = np.column_stack([np.ones(len(Xt)), Xt])
    C, *_ = np.linalg.lstsq(Z, Yt, rcond=None)
    return C[1:].T, C[0], np.cov((Yt - Z @ C).T)


def load_of(A, a, s):
    """Continuous-time load b from the yearly intercept a at speed s: a = A^-1 (e^{sA} - I) b."""
    return A @ np.linalg.solve(linalg.expm(s * A) - np.eye(P), a)


def start_values(groups):
    (BT, aT, ST), (BP, aP, SP) = ols_group(groups[0]), ols_group(groups[1])
    AT, AP = np.real(linalg.logm(BT)), np.real(linalg.logm(BP))
    A0 = np.where(SUP, AT, 0.0)
    s0 = float(np.clip(np.sum(AP[SUP] * AT[SUP]) / np.sum(AT[SUP] ** 2), 0.3, 5.0))
    h0 = np.concatenate([A0[SUP], load_of(A0, aT, 1.0), load_of(A0, aP, s0), lq_of(ST),
                         lq_of(SP / s0), [np.log(s0)], [np.log(0.3)]])
    AP0 = np.where(SUP, AP, 0.0)
    ha = np.concatenate([A0[SUP], AP0[SUP], load_of(A0, aT, 1.0), load_of(AP0, aP, 1.0),
                         lq_of(ST), lq_of(SP), [np.log(0.3)]])
    return h0, ha


def h0_to_ha(x):
    """The H0 solution written in HA's parameters (A2 = e^g A, b_P and Q_P times e^g)."""
    A, bT, bP = x[:NS], x[NS:NS + P], x[NS + P:NS + 2 * P]
    lT, lP = x[NS + 2 * P:NS + 2 * P + NT], x[NS + 2 * P + NT:NS + 2 * P + 2 * NT]
    g, ls = x[-2], x[-1]
    eg = np.exp(g)
    return np.concatenate([A, eg * A, bT, eg * bP, lT, lq_of(eg * Q_of(lP)), [ls]])


def weighted_median(v, w):
    o = np.argsort(v)
    c = np.cumsum(w[o])
    return v[o][np.searchsorted(c, 0.5 * c[-1])]


def speed_adjusted(groups):
    """O1, O2, O3 and the share of the phenotype difference in drift explained by speed."""
    data = [(torch.tensor(X[:, :-1]), torch.tensor(X[:, 1:])) for X in groups]
    n_trans = sum(X.shape[0] * (X.shape[1] - 1) for X in groups)
    h0s, has = start_values(groups)
    r0 = minimize(make_obj("H0", data), h0s)
    fa = make_obj("HA", data)
    ra = min([minimize(fa, h0_to_ha(r0.x)), minimize(fa, has)], key=lambda r: r.fun)
    lrt = 2 * (r0.fun - ra.fun)
    out = {"lrt": float(lrt), "p_lrt": float(stats.chi2.sf(max(lrt, 0.0), NS - 1)),
           "s_hat": float(np.exp(r0.x[-2])), "sd_logs_hat": float(np.exp(r0.x[-1])),
           "conv": bool(r0.success and ra.success)}

    # share of the drift difference explained by speed (least-squares speed, HA estimates)
    AT, AP = ra.x[:NS], ra.x[NS:2 * NS]
    s_ls = AP @ AT / (AT @ AT)
    out["r2_speed"] = float(1 - np.sum((AP - s_ls * AT) ** 2) / np.sum((AP - AT) ** 2))

    # O3: score test of each D entry at the H0 fit, efficient information from the Hessian
    xj = np.concatenate([r0.x, np.zeros(NS)])
    xt = torch.tensor(xj)
    H = torch.autograd.functional.hessian(lambda z: nll(z, "J", data), xt).numpy()
    gr = torch.autograd.functional.jacobian(lambda z: nll(z, "J", data), xt).numpy()
    n0 = len(r0.x)
    Hth = H[:n0, :n0]
    p_score = np.zeros(NS)
    for l in range(NS):
        hl = H[n0 + l, :n0]
        sol = np.linalg.solve(Hth, hl)
        u = gr[n0 + l] - sol @ gr[:n0]
        info = H[n0 + l, n0 + l] - hl @ sol
        p_score[l] = stats.chi2.sf(u * u / info, 1) if info > 0 else 1.0
    out["p_score"] = p_score.tolist()

    # O2: adaptive lasso path for D, weights from a robust (L1) speed fit to the HA drifts
    s_rob = weighted_median(AP / np.where(np.abs(AT) > 1e-8, AT, 1e-8), np.abs(AT))
    w = 1.0 / np.maximum(np.abs(AP / s_rob - AT), 1e-4)
    lam_max = float(np.max(np.abs(gr[n0:]) / w))
    bounds = [(None, None)] * n0 + [(0, None)] * (2 * NS)
    x = np.concatenate([r0.x, np.zeros(2 * NS)])
    supports = {(): (r0.fun, r0.x)}
    for lam in lam_max * 2.0 ** (-np.arange(1, 13) / 2):
        r = minimize(make_obj("J", data, lam=lam, w=w, n0=n0), x, bounds)
        x = r.x
        D = x[n0:n0 + NS] - x[n0 + NS:]
        supp = tuple(int(l) for l in np.where(np.abs(D) > 1e-6)[0])
        if supp not in supports:
            mask = np.ones(NS, bool)
            mask[list(supp)] = False
            rr = minimize(make_obj("J", data, fixed_zero=mask), np.concatenate([x[:n0], D]))
            supports[supp] = (rr.fun, rr.x)
    bic = {s: 2 * v[0] + np.log(n_trans) * len(s) for s, v in supports.items()}
    best = min(bic, key=bic.get)
    out["o2_support"] = list(best)
    out["o2_n_paths"] = len(supports)
    return out


# ---------------------------------------------------------------- one cohort

def flags_off(F):
    return {f"{NAMES[j]}->{NAMES[i]}": bool(F[i, j]) for i, j in OFF}


def one_rep(args):
    scenario, seed = args
    rng = np.random.default_rng(seed)
    t0 = time.time()
    groups = simulate(rng, scenario)
    Xt, Yt, g, pid = stack(groups)
    grams = person_grams(groups)
    rec = {"scenario": scenario, "seed": seed}

    p_link, p_omni, p_node, p_node_slope = interaction_tests(Xt, Yt, g, pid)
    rec["N1"] = flags_off(T.bh(p_link.ravel()).reshape(P, P))
    rec["N1o"] = bool(p_omni < 0.05)
    rec["N2_nodes"] = T.bh(p_node).tolist()
    rec["N2b_nodes"] = T.bh(p_node_slope).tolist()
    rec["N3"] = flags_off(diff_graph_search(Xt, Yt, g))
    rec["N4"] = flags_off(shared_lasso(Xt, Yt, g, pid, "cv"))
    rec["N4b"] = flags_off(shared_lasso(Xt, Yt, g, pid, "bic"))
    rec["N5"] = flags_off(T.bh(naive_ct(grams, rng).ravel()).reshape(P, P))
    p_f, p_m = perm_test(grams, rng)
    rec["N6_frob"], rec["N6_max"] = bool(p_f < 0.05), bool(p_m < 0.05)
    rec["N7"] = flags_off(separate_graphs(Xt, Yt, g))

    sa = speed_adjusted(groups)
    F2 = np.zeros((P, P), bool)
    for l in sa["o2_support"]:
        F2[tuple(SUP_IDX[l])] = True
    F3 = np.zeros((P, P), bool)
    F3[SUP] = T.bh(np.array(sa["p_score"]))
    rec["O1"] = bool(sa["p_lrt"] < 0.05)
    rec["O2"] = flags_off(F2)
    rec["O3"] = flags_off(F3)
    rec["O2_diag"] = int(sum(F2[i, i] for i in range(P)))
    rec["O3_diag"] = int(sum(F3[i, i] for i in range(P)))
    for k in ("lrt", "p_lrt", "s_hat", "sd_logs_hat", "r2_speed", "conv", "o2_n_paths"):
        rec[k] = sa[k]
    rec["secs"] = time.time() - t0
    return rec


# ---------------------------------------------------------------- summary

LINK_METHODS = ["N1", "N3", "N4", "N4b", "N5", "N7", "O2", "O3"]
MAP_KEYS = [f"{NAMES[j]}->{NAMES[i]}" for i, j in MAP]
INDEPENDENCE = NAMES.index("Independence")


def summarise(reps, scenario):
    true = {f"{NAMES[j]}->{NAMES[i]}" for i, j in TRUE_DIFF[scenario]}
    R = len(reps)
    s = {"R": R}
    for m in LINK_METHODS:
        shared_map = [k for k in MAP_KEYS if k not in true]
        shared_all = [k for k in reps[0][m] if k not in true]
        fl_map = np.array([[r[m][k] for k in shared_map] for r in reps])
        fl_all = np.array([[r[m][k] for k in shared_all] for r in reps])
        d = {"any_off_flag": float(np.mean([any(r[m].values()) for r in reps])),
             "false_rate_map_links": float(fl_map.mean()),
             "false_rate_all_off": float(fl_all.mean()),
             "false_map_links_mean": float(fl_map.sum(1).mean())}
        if true:
            (tk,) = true
            hit = np.array([r[m][tk] for r in reps])
            exact = np.array([r[m][tk] and not any(r[m][k] for k in shared_all) for r in reps])
            d.update({"power_true_link": float(hit.mean()), "exactly_true_link": float(exact.mean())})
        s[m] = d
    for m in ("N1o", "N6_frob", "N6_max", "O1"):
        v = np.array([r[m] for r in reps])
        s[m] = {"reject": float(v.mean()), "mcse": float(np.sqrt(v.mean() * (1 - v.mean()) / R))}
    for m in ("N2_nodes", "N2b_nodes"):
        nodes = np.array([r[m] for r in reps])
        others = [i for i in range(P) if not (true and i == INDEPENDENCE)]
        d = {"any_node": float(nodes.any(1).mean()), "false_node_rate": float(nodes[:, others].mean()),
             "per_node": dict(zip(NAMES, nodes.mean(0).round(3).tolist()))}
        if true:
            d["independence_flagged"] = float(nodes[:, INDEPENDENCE].mean())
        s[m] = d
    s["O2_diag_mean"] = float(np.mean([r["O2_diag"] for r in reps]))
    s["O3_diag_mean"] = float(np.mean([r["O3_diag"] for r in reps]))
    s["lrt_mean"] = float(np.mean([r["lrt"] for r in reps]))
    s["s_hat_median"] = float(np.median([r["s_hat"] for r in reps]))
    s["sd_logs_hat_median"] = float(np.median([r["sd_logs_hat"] for r in reps]))
    s["r2_speed_median"] = float(np.median([r["r2_speed"] for r in reps]))
    s["converged"] = float(np.mean([r["conv"] for r in reps]))
    s["secs_median"] = float(np.median([r["secs"] for r in reps]))
    return s


def main(R, seed, workers):
    plan = [(sc, seed + 1000 * k + i) for k, sc in enumerate(SCEN)
            for i in range(R if sc in ("S0", "S1") else max(R // 2, 1))]
    with Pool(workers) as pool:
        reps = pool.map(one_rep, plan, chunksize=1)
    out = {sc: summarise([r for r in reps if r["scenario"] == sc], sc) for sc in SCEN}
    with open("toy_comparators_results.json", "w") as f:
        json.dump(out, f, indent=1)
    with open("toy_comparators_reps.json", "w") as f:
        json.dump(reps, f)
    for sc in SCEN:
        print(sc, json.dumps(out[sc]), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        for sc in SCEN:
            r = one_rep((sc, 7))
            print(sc, round(r["secs"], 1), "s", {k: r[k] for k in ("O1", "lrt", "s_hat", "r2_speed", "conv")},
                  "O2", [k for k, v in r["O2"].items() if v], "N1", sum(r["N1"].values()),
                  "N3", sum(r["N3"].values()), "N4", sum(r["N4"].values()), flush=True)
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else 100, 20261103,
             int(sys.argv[2]) if len(sys.argv) > 2 else 8)

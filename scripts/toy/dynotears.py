"""DYNOTEARS (Pamfil, Sriwattanaworachai, Desai et al. 2020, AISTATS), re-implemented from the
authors' causalnex code (causalnex/structure/dynotears.py, Apache 2.0; read, not run).

Model: X = X W + Y A + Z. X is the current slice (n x d), Y the lagged slices (n x pd), W the
same-time (intra-slice) weights and A the lagged (inter-slice) weights; W[i, j] and A[i, j] are
edges from i to j. The objective is
    (1/2n) ||X (I - W) - Y A||_F^2 + lambda_w |W|_1 + lambda_a |A|_1,
subject to h(W) = tr(exp(W o W)) - d = 0, which holds exactly when W is acyclic (Zheng 2018).
As in the authors' code:
- W and A are split into non-negative parts (W = W+ - W-), so the L1 terms are linear and
  L-BFGS-B handles the bounds; self-loops in W are fixed at zero by the bounds;
- augmented Lagrangian: rho starts at 1 and is multiplied by 10 until h falls below a quarter of
  its previous value (while rho < 1e20); then alpha += rho * h; stop when h <= h_tol (1e-8) or
  after max_iter (100) dual steps; every inner problem starts from the last accepted estimate;
- the search starts at zero, and edges below the thresholds are set to zero at the end.
The authors' code applies one threshold to W and A; the paper's simulations use separate ones
(tau_w, tau_a), so both are arguments. The loss is computed from cross-product matrices, which is
algebraically the authors' residual form.
"""
import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt


def _layout(d, p):
    nw, na = d * d, p * d * d
    bw = [(0, 0) if i == j else (0, None) for i in range(d) for j in range(d)]
    return nw, na, 2 * bw + [(0, None)] * (2 * na)


def _unpack(v, d, p):
    """Vector [W+, W-, then A+_k, A-_k for each lag k] to (W, A), as the authors' _reshape_wa."""
    nw = d * d
    W = (v[:nw] - v[nw:2 * nw]).reshape(d, d)
    a = v[2 * nw:].reshape(2 * p, nw)
    A = (a[::2] - a[1::2]).reshape(p * d, d)
    return W, A


def _objective(v, S, d, p, lambda_w, lambda_a, rho, alpha):
    Sxx, Sxy, Syy = S
    nw = d * d
    W, A = _unpack(v, d, p)
    M = np.eye(d) - W
    E = slin.expm(W * W)
    h = np.trace(E) - d
    loss = 0.5 * (np.sum(M * (Sxx @ M)) - 2 * np.sum(M * (Sxy @ A)) + np.sum(A * (Syy @ A)))
    f = loss + 0.5 * rho * h * h + alpha * h + lambda_w * v[:2 * nw].sum() + lambda_a * v[2 * nw:].sum()
    gW = -(Sxx @ M - Sxy @ A) + (rho * h + alpha) * E.T * W * 2
    gA = (-(Sxy.T @ M - Syy @ A)).reshape(p, nw)
    g = np.concatenate([gW.ravel(), -gW.ravel(), np.hstack([gA, -gA]).ravel()])
    g[:2 * nw] += lambda_w
    g[2 * nw:] += lambda_a
    return f, g


def _h(v, d, p):
    W, _ = _unpack(v, d, p)
    return np.trace(slin.expm(W * W)) - d


def dynotears(X, Y, lambda_w=0.1, lambda_a=0.1, tau_w=0.0, tau_a=0.0, max_iter=100, h_tol=1e-8):
    """Return (W, A, info). X: n x d current slice; Y: n x pd lagged slices (lag 1 first)."""
    n, d = X.shape
    p = Y.shape[1] // d
    S = (X.T @ X / n, X.T @ Y / n, Y.T @ Y / n)
    nw, na, bnds = _layout(d, p)
    v_est = np.zeros(2 * (nw + na))
    v_new = v_est.copy()
    rho, alpha, h_val, h_new = 1.0, 0.0, np.inf, np.inf
    converged = False
    for _ in range(max_iter):
        while rho < 1e20 and (h_new > 0.25 * h_val or h_new == np.inf):
            v_new = sopt.minimize(_objective, v_est, args=(S, d, p, lambda_w, lambda_a, rho, alpha),
                                  method="L-BFGS-B", jac=True, bounds=bnds).x
            h_new = _h(v_new, d, p)
            if h_new > 0.25 * h_val:
                rho *= 10
        v_est = v_new
        h_val = h_new
        alpha += rho * h_val
        if h_val <= h_tol:
            converged = True
            break
    W, A = _unpack(v_est, d, p)
    W[np.abs(W) < tau_w] = 0.0
    A[np.abs(A) < tau_a] = 0.0
    return W, A, {"h": float(h_val), "rho": rho, "converged": converged}


def gradient_check(seed=0, d=4, p=1, n=60):
    """Largest relative difference between the analytic gradient and central differences."""
    rng = np.random.default_rng(seed)
    X, Y = rng.normal(size=(n, d)), rng.normal(size=(n, p * d))
    S = (X.T @ X / n, X.T @ Y / n, Y.T @ Y / n)
    v = np.abs(rng.normal(scale=0.3, size=2 * (d * d + p * d * d)))
    _, g = _objective(v, S, d, p, 0.1, 0.1, 10.0, 0.5)
    num = np.zeros_like(v)
    for k in range(len(v)):
        e = np.zeros_like(v)
        e[k] = 1e-6
        num[k] = (_objective(v + e, S, d, p, 0.1, 0.1, 10.0, 0.5)[0]
                  - _objective(v - e, S, d, p, 0.1, 0.1, 10.0, 0.5)[0]) / 2e-6
    return float(np.max(np.abs(num - g)) / np.max(np.abs(g)))

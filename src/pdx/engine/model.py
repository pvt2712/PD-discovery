"""Model specification and parameter layout for the continuous-time engine (protocol v0.5, §2.3, §7.1).

A participant i in phenotype g has true domain levels X that follow

    dX = s_i [A_g X + b_g + l_i] dt + sqrt(s_i) G_g dW,
    log s_i = gamma_g + beta' z_i + theta' y1c_i + sigma_g u_i,  u_i ~ N(0, 1),
    l_i | u_i ~ N(gam_y,g * y1c_i + Gam_z,g z_i + lam_g * u_i, diag(tau_g^2)),

and is observed at the visits as Y = X + h + e, e ~ N(0, diag(meas_var)), where h holds the
treatment terms. The noise covariance is exp(omega * era_i) G_g G_g' when an era column is named
(protocol §2.3: era also shifts the noise). Everything about the load is phenotype-specific, so
the speed-only model is exactly nested in the free model (a phenotype's speed rescales its load
along with its drift, load mean and noise). Three kinds of model share this layout:

- "null": one mechanism A for all phenotypes, run at the phenotype's speed exp(gamma_g) (H-P's null);
- "free": a free drift for every phenotype (H-P's alternative; the phenotype speed is absorbed);
- "joint": phenotype g's drift is A + D_g, run at speed exp(gamma_g) (localisation, §7.3.2).

Drift entries live on the support (diagonal always included). Own rates are kept negative
(-exp(raw)) unless negative_own_rates is False.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

KINDS = ("null", "free", "joint")


@dataclass
class ModelSpec:
    d: int
    support: np.ndarray
    p: int = 0
    treat_mask: np.ndarray | None = None
    treatment: str = "linear"          # "linear", "emax" or "none"
    ed50: float = 3.0                  # Emax half-maximal dose, in units of 100 mg LEDD
    ec50_ldr: float = 3.0              # the same for the long-duration exposure
    meas_var: np.ndarray | None = None  # fixed measurement-error variances (D28); None means none
    n_quad: int = 12
    n_groups: int = 2
    load_on_z: bool = True
    load_on_y1: bool = True
    speed_on_y1: bool = True
    negative_own_rates: bool = True
    noise_era_col: int | None = None   # column of z holding era, if era shifts the noise

    def __post_init__(self):
        self.support = np.asarray(self.support, bool).copy()
        np.fill_diagonal(self.support, True)
        self.treat_mask = np.zeros(self.d, bool) if self.treat_mask is None else np.asarray(self.treat_mask, bool)
        self.meas_var = np.zeros(self.d) if self.meas_var is None else np.asarray(self.meas_var, float)
        if self.treatment != "none" and not self.treat_mask.any():
            self.treatment = "none"

    @property
    def sup_idx(self) -> np.ndarray:
        """Support entries as (to, from) pairs, row-major."""
        return np.argwhere(self.support)

    @property
    def n_sup(self) -> int:
        return int(self.support.sum())

    @property
    def diag_pos(self) -> np.ndarray:
        si = self.sup_idx
        return np.where(si[:, 0] == si[:, 1])[0]

    @property
    def n_treat(self) -> int:
        return int(self.treat_mask.sum()) if self.treatment != "none" else 0

    def with_support(self, support: np.ndarray) -> "ModelSpec":
        return replace(self, support=np.asarray(support, bool))


class Layout:
    """Maps names to slices of the flat parameter vector for one kind of model."""

    def __init__(self, spec: ModelSpec, kind: str):
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        self.spec, self.kind = spec, kind
        d, p, G, ns, nt = spec.d, spec.p, spec.n_groups, spec.n_sup, spec.n_treat
        tri = d * (d + 1) // 2
        blocks = [("A", (ns,))]
        if kind == "free":
            blocks += [(f"A{g}", (ns,)) for g in range(1, G)]
        if kind == "joint":
            blocks += [(f"D{g}", (ns,)) for g in range(1, G)]
        if kind in ("null", "joint"):
            blocks += [("gamma", (G - 1,))]
        blocks += [("beta", (p,)), ("theta", (d if spec.speed_on_y1 else 0,)), ("log_sigma", (G,)),
                   ("b", (G, d)), ("L", (G, tri)),
                   ("gam_y", (G, d) if spec.load_on_y1 else (0,)),
                   ("gam_z", (G, d, p) if (spec.load_on_z and p > 0) else (0,)),
                   ("lam", (G, d)), ("log_tau", (G, d)),
                   ("omega", (1,) if spec.noise_era_col is not None else (0,))]
        if nt:
            blocks += [("kappa", (G, nt)), ("kappa_ldr", (G, nt))]
        self.blocks = blocks
        self.slices: dict[str, tuple[slice, tuple]] = {}
        k = 0
        for name, shape in blocks:
            n = int(np.prod(shape))
            self.slices[name] = (slice(k, k + n), shape)
            k += n
        self.size = k

    def has(self, name: str) -> bool:
        return name in self.slices and self.slices[name][0].stop > self.slices[name][0].start

    def get(self, x, name: str):
        sl, shape = self.slices[name]
        return x[sl].reshape(shape)

    def put(self, x: np.ndarray, name: str, value) -> None:
        sl, shape = self.slices[name]
        x[sl] = np.asarray(value, float).reshape(-1)


# ---------------------------------------------------------------- unpacking (torch)

def _fill(vals: torch.Tensor, spec: ModelSpec, constrain_diag: bool) -> torch.Tensor:
    si = spec.sup_idx
    if constrain_diag and spec.negative_own_rates:
        dmask = torch.zeros(len(si), dtype=torch.bool)
        dmask[torch.as_tensor(spec.diag_pos)] = True
        vals = torch.where(dmask, -torch.exp(vals), vals)
    A = torch.zeros(spec.d, spec.d, dtype=vals.dtype)
    return A.index_put((torch.as_tensor(si[:, 0]), torch.as_tensor(si[:, 1])), vals)


def drifts(x: torch.Tensor, lay: Layout) -> torch.Tensor:
    """(G, d, d): each phenotype's drift at the reference speed (before exp(gamma_g))."""
    spec, G = lay.spec, lay.spec.n_groups
    A = _fill(lay.get(x, "A"), spec, True)
    out = [A]
    for g in range(1, G):
        if lay.kind == "free":
            out.append(_fill(lay.get(x, f"A{g}"), spec, True))
        elif lay.kind == "joint":
            out.append(A + _fill(lay.get(x, f"D{g}"), spec, False))
        else:
            out.append(A)
    return torch.stack(out)


def log_speeds(x: torch.Tensor, lay: Layout) -> torch.Tensor:
    """(G,): each phenotype's log-speed offset (0 for the reference; 0 for all in the free model)."""
    G = lay.spec.n_groups
    if lay.kind == "free":
        return torch.zeros(G, dtype=x.dtype)
    return torch.cat([torch.zeros(1, dtype=x.dtype), lay.get(x, "gamma")])


def noise_cov(x: torch.Tensor, lay: Layout) -> torch.Tensor:
    """(G, d, d): diffusion covariance Q_g = L_g L_g' per unit time at the reference speed."""
    d, G = lay.spec.d, lay.spec.n_groups
    raw = lay.get(x, "L")
    tri = torch.tril_indices(d, d)
    Ls = []
    for g in range(G):
        L = torch.zeros(d, d, dtype=x.dtype).index_put((tri[0], tri[1]), raw[g])
        dg = torch.diagonal(L)
        Ls.append(L - torch.diag(dg) + torch.diag(torch.exp(dg)))
    Ls = torch.stack(Ls)
    return Ls @ Ls.transpose(-1, -2)


def treat_effect(x: torch.Tensor, lay: Layout, dose: torch.Tensor, expo: torch.Tensor,
                 on: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """(n, T, d) treatment terms h added to the observed scores (negative means improvement)."""
    spec = lay.spec
    n, T = dose.shape
    h = torch.zeros(n, T, spec.d, dtype=x.dtype)
    if spec.n_treat == 0:
        return h
    kap = lay.get(x, "kappa")[g]        # (n, nt)
    kapl = lay.get(x, "kappa_ldr")[g]   # (n, nt)
    if spec.treatment == "emax":
        fd = dose / (spec.ed50 + dose)
        fe = expo / (spec.ec50_ldr + expo)
    else:
        fd, fe = dose, expo
    eff = -(kap[:, None, :] * (fd * on)[:, :, None] + kapl[:, None, :] * fe[:, :, None])
    tm = torch.as_tensor(np.where(spec.treat_mask)[0])
    return h.index_copy(2, tm, eff)


def unpack_numpy(x: np.ndarray, lay: Layout) -> dict:
    """Named model quantities as numpy arrays (for reporting and simulation)."""
    xt = torch.as_tensor(np.asarray(x, float))
    out = {"A": drifts(xt, lay).numpy(), "log_speed": log_speeds(xt, lay).numpy(),
           "Q": noise_cov(xt, lay).numpy(), "b": lay.get(xt, "b").numpy(),
           "sigma": np.exp(lay.get(xt, "log_sigma").numpy()),
           "beta": lay.get(xt, "beta").numpy(), "lam": lay.get(xt, "lam").numpy(),
           "tau": np.exp(lay.get(xt, "log_tau").numpy())}
    G, d, p = lay.spec.n_groups, lay.spec.d, lay.spec.p
    out["theta"] = lay.get(xt, "theta").numpy() if lay.has("theta") else np.zeros(d)
    out["gam_y"] = lay.get(xt, "gam_y").numpy() if lay.has("gam_y") else np.zeros((G, d))
    out["gam_z"] = lay.get(xt, "gam_z").numpy() if lay.has("gam_z") else np.zeros((G, d, p))
    out["omega"] = float(lay.get(xt, "omega")[0]) if lay.has("omega") else 0.0
    if lay.spec.n_treat:
        out["kappa"] = lay.get(xt, "kappa").numpy()
        out["kappa_ldr"] = lay.get(xt, "kappa_ldr").numpy()
    if lay.kind == "joint":
        out["D"] = out["A"][1:] - out["A"][:1]
    return out


def pack_drift(A: np.ndarray, spec: ModelSpec, constrain_diag: bool = True) -> np.ndarray:
    """Raw support values for a (d, d) drift (inverse of _fill)."""
    si = spec.sup_idx
    v = A[si[:, 0], si[:, 1]].astype(float).copy()
    if constrain_diag and spec.negative_own_rates:
        dp = spec.diag_pos
        v[dp] = np.log(np.maximum(-v[dp], 1e-4))
    return v


def pack_chol(Q: np.ndarray) -> np.ndarray:
    d = Q.shape[0]
    w, V = np.linalg.eigh(0.5 * (Q + Q.T))
    Q = (V * np.maximum(w, 1e-6)) @ V.T
    L = np.linalg.cholesky(Q)
    L[np.diag_indices(d)] = np.log(np.diag(L))
    return L[np.tril_indices(d)]

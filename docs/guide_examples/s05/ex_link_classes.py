"""Sections 5.26-5.34: anchor-free per-link tests, link classes, the intersection-union verdict and
the per-domain test, on hand-made Monte Carlo draws (no model is fitted). Synthetic numbers only."""
from types import SimpleNamespace

import numpy as np

import _fixture  # noqa: F401  (import path)
from pdx.engine.model import ModelSpec
from pdx.localise import (FreeDraws, SpeedView, _basic_ci, _classes_at, _domain_p, _mc_p, classify_links,
                          per_domain_tests, per_link_tests, speed_and_load_only)
from pdx.multiplicity import simultaneous_intervals


def setup(entries, d, noise=0.01, M=4000, seed=0, extra_noise=None):
    """entries {(to, from): (a0, a1)}; draws = estimate + noise (both phenotypes)."""
    sup = np.zeros((d, d), bool)
    for (i, j) in entries:
        sup[i, j] = True
    spec = ModelSpec(d=d, support=sup)
    order = [tuple(map(int, r)) for r in spec.sup_idx]
    a0 = np.array([entries[k][0] for k in order])
    a1 = np.array([entries[k][1] for k in order])
    sd = np.full(len(order), noise)
    for k, v in (extra_noise or {}).items():
        sd[order.index(k)] = v
    rng = np.random.default_rng(seed)
    dr = FreeDraws(a0, a1, a0 + sd * rng.normal(size=(M, len(order))), a1 + sd * rng.normal(size=(M, len(order))))
    return SimpleNamespace(spec=spec), spec, dr


BASE = {(0, 0): (-0.25, -0.5), (1, 1): (-0.25, -0.5), (2, 2): (-0.25, -0.5), (3, 3): (-0.25, -0.5),
        (1, 0): (0.20, 0.64),        # stronger beyond speed: Delta = 0.64/2 - 0.20 = 0.12
        (2, 0): (0.00, 0.50),        # present in PIGD only
        (2, 1): (0.15, 0.30), (3, 1): (0.15, 0.30), (3, 2): (0.15, 0.30)}   # pure speed
fit, spec, dr = setup(BASE, 4)
res = per_link_tests(fit, None, draws=dr)
print(f"s_hat = {res.s_hat:.3f}; speed set = ({res.speed_set[0]:.3f}, {res.speed_set[1]:.3f}); "
      f"probe speeds = {np.round(res.probe_speeds, 3)}")
print("diagnostics:", {k: round(v, 3) for k, v in res.diagnostics.items()})
print(f"{'link':8}{'a0':>7}{'a1':>7}{'Delta':>8}{'p':>8}   {'ci90':16} verdict")
for c, a0, a1 in zip(classify_links(res, eps=0.10, method="by"), res.a0, dr.a1):
    print(f"{str(c['link']):8}{a0:7.2f}{a1:7.2f}{c['delta']:8.3f}{c['p']:8.4f}   "
          f"[{c['ci90'][0]:6.3f},{c['ci90'][1]:6.3f}] {c['verdict']}")

k = res.links.index((1, 0))
D = res.main.d_draws[:, k]
print(f"\nMonte Carlo p of (1,0) by hand: (1 + #|D* - Delta| >= |Delta|)/(M + 1) = "
      f"{(1 + np.sum(np.abs(D - res.delta[k]) >= abs(res.delta[k]))) / (len(D) + 1):.4f}; _mc_p = "
      f"{_mc_p(res.delta, res.main.d_draws)[k]:.4f}")
k = res.links.index((2, 1))
lo90, hi90 = _basic_ci(res.delta, res.main.d_draws, 0.90)
c = res.main.d_draws[:, k] - res.delta[k]
lower95 = res.delta[k] - np.quantile(c, 0.95)             # one-sided lower 95% bound
upper95 = res.delta[k] - np.quantile(c, 0.05)             # one-sided upper 95% bound
print(f"TOST for (2,1): lower 95% bound {lower95:.4f} > -0.10 and upper 95% bound {upper95:.4f} < 0.10; "
      f"basic 90% interval [{lo90[k]:.4f}, {hi90[k]:.4f}]")
print("speed and load only (BASE):", speed_and_load_only(res, 0.10))
pure = {key: (v[0], 2.0 * v[0]) for key, v in BASE.items() if key != (2, 0)}
fitp, specp, drp = setup(pure, 4)
print("speed and load only (pure speed, noise 0.01):", speed_and_load_only(per_link_tests(fitp, None, draws=drp), 0.10))

dom = per_domain_tests(res, spec)
print("\nper-domain p:", np.round(dom["p"], 4), "verdicts:", dom["verdict"])
j = 2
L = np.where(spec.sup_idx[:, 0] == j)[0]
Dj = res.main.d_draws[:, L] - res.main.delta[L]
Si = np.linalg.pinv(np.cov(Dj.T))
t_obs = res.main.delta[L] @ Si @ res.main.delta[L]
t_mc = np.einsum("mi,ij,mj->m", Dj, Si, Dj)
print(f"domain {j} by hand: entries {[res.links[i] for i in L]}, T = {t_obs:.1f}, "
      f"p = {(1 + np.sum(t_mc >= t_obs)) / (len(t_mc) + 1):.5f} (code {_domain_p(res.main, spec)[j]:.5f})")

print("\nclass rules on a hand-made speed view (rejections set by hand, method 'none')")
rng = np.random.default_rng(3)
M = 4000
a0 = np.array([0.20, 0.20, 0.20, 0.20, 0.15, 0.15])
a1s = np.array([0.40, 0.02, -0.20, 0.10, 0.16, 0.15])           # PIGD entries at the common speed
sd1 = np.array([0.02, 0.02, 0.02, 0.15, 0.01, 0.10])
a0_draws = a0 + 0.02 * rng.normal(size=(M, 6))
a1_draws = a1s + sd1 * rng.normal(size=(M, 6))
delta, d_draws = a1s - a0, a1_draws - a0_draws
p = np.array([0.001, 0.001, 0.001, 0.001, 0.80, 0.80])
view = SpeedView(1.0, delta, d_draws, a1s, a1_draws, p)
fam = np.ones(6, bool)
names = ["both non-zero, same sign", "PIGD within +-eps", "opposite signs", "PIGD interval wide",
         "not rejected, narrow", "not rejected, wide"]
for nm, cl in zip(names, _classes_at(view, a0, a0_draws, fam, 0.10, "none", 0.05)):
    print(f"  {nm:26} -> {cl}")

print("\na marginal 90% interval inside +-eps is not enough for 'similar' (simultaneous over 6 links)")
six = {(0, 0): (-0.25, -0.5), (1, 1): (-0.25, -0.5), (2, 2): (-0.25, -0.5), (3, 3): (-0.25, -0.5),
       (1, 0): (0.15, 0.30), (2, 0): (0.15, 0.30), (2, 1): (0.15, 0.30), (3, 0): (0.15, 0.30),
       (3, 1): (0.15, 0.30), (3, 2): (0.15, 0.30)}
noisy = {key: 0.045 for key in six if key[0] != key[1]}
f6, s6, d6 = setup(six, 4, extra_noise=noisy, seed=5)
r6 = per_link_tests(f6, None, draws=d6)
fam6 = r6.family
slo, shi = simultaneous_intervals(r6.delta[fam6], r6.main.d_draws[:, fam6], 0.90)
for c, lo_s, hi_s in zip([c for c in classify_links(r6, 0.10) if c["class"] != "outside the family"], slo, shi):
    print(f"  {str(c['link']):7} marginal 90% [{c['ci90'][0]:6.3f},{c['ci90'][1]:6.3f}]  "
          f"simultaneous [{lo_s:6.3f},{hi_s:6.3f}]  -> {c['class']} / {c['verdict']}")

print("\ndiagnostics that disagree make the verdict 'not separable'")
ent = {(0, 0): (-0.25, -0.5), (1, 1): (-0.25, -0.5), (2, 2): (-0.25, -0.5),
       (1, 0): (0.20, 0.64), (2, 1): (0.15, 0.30)}
f3, s3, d3 = setup(ent, 3)
r3 = per_link_tests(f3, None, draws=d3)
print("  diagnostics:", {k: round(v, 3) for k, v in r3.diagnostics.items()}, "speed set:", np.round(r3.speed_set, 3))
for c in classify_links(r3, 0.10, "none"):
    if c["class"] != "outside the family":
        print(f"  {c['link']}: class at s_hat '{c['class']}', stable {c['stable_across_speeds']} -> {c['verdict']}")

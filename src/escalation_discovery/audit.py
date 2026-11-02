# Ported from escalation-study-v2/src/escalation_v2/audit.py (29 October 2026). Changes are marked 'escalation-discovery:'.
"""audit — Q1: do the guideline's reasons drive real escalation decisions?

    uv run esc2 audit [ppmi1 ppmi2 genetic] [--quick]

For each guideline-graded reason into the decision (core map: three triggers, four brakes, the dose in force) and for
each of the six analysis blocks, on the on-strategy assessments (escalate or hold) of a cohort:

- beyond everything else recorded: a generalised covariance measure test (Shah & Peters 2020) with a patient-clustered
  Wald statistic. The decision E and each column of the reason are residualised on all other pre-decision covariates
  with cross-fitted gradient boosting (folds partition patients); the products of the residuals are summed within
  patient and referred to chi-square with one degree of freedom per column (a pseudo-inverse drops collinear columns).
  Test-wise deletion (primary; Witte, Foraita & Didelez 2022): the rows with every variable of the test recorded. The
  available-case version, with missing covariates median-imputed inside folds and no row dropped, is reported beside
  it as a sensitivity analysis: there, a reason correlated with a missing confounder absorbs its information and
  looks associated (tests/test_audit.py).
- effect size: the projection of the decision residual on the reason's residuals, per SD of each column, in percentage
  points of escalation probability (a linear probability scale: an approximation for a clinician to read, not a
  causal effect of the symptom).
- the cross-validated AUC of the reason alone.

Verdicts across cohorts (fixed before the estimates, DESIGN.md §4.1): replicated = p < 0.05 with the guideline's sign in
at least two cohorts; one cohort = p < 0.05 with the guideline's sign in one; opposite = p < 0.05 against the
guideline's sign in at least one cohort; not detected otherwise. The sign is read from the level column (the first).
Outputs: results/audit_reasons.csv, results/audit_blocks.csv, results/audit_verdicts.csv.
"""
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import config as C
from . import coremap
from scipy import stats

from .citest import cluster_sums
from .nuisance import cross_fit

N_JOBS = 6


def load(cohort):
    d = pd.read_csv(C.derived_dir() / f"trials_dt_{cohort}.csv", low_memory=False)
    return d[d.E.isin([0, 1])].reset_index(drop=True)


def wald(prod, pid):
    """CR1 Wald test that E[prod] = 0 (prod is n x k), with a pseudo-inverse so that collinear residual columns (a
    block whose members carry the same information once the rest is known) reduce the degrees of freedom instead of
    failing."""
    S, _ = cluster_sums(prod, pid)
    G = S.shape[0]
    tot = S.sum(0)
    V = G / (G - 1) * (S.T @ S)
    df = int(np.linalg.matrix_rank(V))
    if df == 0:
        return np.nan, 0, np.nan
    stat = float(tot @ np.linalg.pinv(V) @ tot)
    return stat, df, float(stats.chi2.sf(stat, df))


def projection(rE, R, pid):
    """OLS of the decision residual on the reason's residuals (no intercept) with the CR1 covariance over patients."""
    inv = np.linalg.pinv(R.T @ R)
    beta = inv @ (R.T @ rE)
    S, _ = cluster_sums(R * (rE - R @ beta)[:, None], pid)
    G = S.shape[0]
    V = G / (G - 1) * inv @ (S.T @ S) @ inv
    return beta, np.sqrt(np.clip(np.diag(V), 0, None))


def gcm_test(d, xs, conditioning, seed=0, tune=False, min_rows=100, mode="complete"):
    """The reason (columns xs) against the decision, given the conditioning columns; None when too few rows.
    mode="complete" (primary): the rows with every variable of this test recorded. mode="available" (sensitivity): a
    row enters when one of the reason's columns is recorded, missing conditioning variables are median-imputed with
    indicators inside each training fold, and each column contributes its residual product where it is recorded.
    Effect sizes are per column (its residual alone)."""
    if mode == "complete":
        d = d[d[list(xs) + list(conditioning)].notna().all(axis=1)]
    obs = d[xs].notna().to_numpy()
    rows = obs.any(axis=1)
    if rows.sum() < min_rows or d.loc[rows, "PATNO"].nunique() < 30 or d.loc[rows, "E"].nunique() < 2:
        return None
    sub = d.loc[rows]
    obs = obs[rows]
    pid = sub.PATNO.to_numpy()
    W = sub[conditioning].to_numpy(dtype=float)
    E = sub.E.to_numpy(dtype=float)
    rE = E - cross_fit(W, E, pid, "clf", "boosting", seed=seed, tune=tune)["pred"]
    R, kept, eff = [], [], []
    for j, x in enumerate(xs):
        if obs[:, j].sum() < min_rows:
            continue
        xv = sub[x].to_numpy(dtype=float)
        r = np.where(obs[:, j], xv - cross_fit(W, np.nan_to_num(xv), pid, "reg", "boosting", seed=seed, tune=tune, fit_mask=obs[:, j])["pred"], 0.0)
        if r[obs[:, j]].std() <= 1e-8:
            continue
        R.append(r); kept.append(x)
        o = obs[:, j]
        beta, se = projection(rE[o], r[o][:, None], pid[o])
        sd = float(np.nanstd(xv))
        eff.append((x, 100 * beta[0] * sd, 100 * se[0] * sd, int(o.sum())))
    if not R:
        return None
    R = np.column_stack(R)
    stat, df, p = wald(R * rE[:, None], pid)
    return dict(chi2=round(stat, 2) if np.isfinite(stat) else np.nan, df=df, p=float(f"{p:.3g}") if np.isfinite(p) else np.nan, rows=int(rows.sum()),
                patients=int(len(np.unique(pid))), escalation_rate=round(float(E.mean()), 3),
                pp_per_sd="; ".join(f"{x} {b:+.2f} ({s:.2f}; n {n})" for x, b, s, n in eff),
                lead_pp_per_sd=round(eff[0][1], 2), lead_se=round(eff[0][2], 2), lead_column=eff[0][0])


def auc_alone(d, xs, seed=0):
    pid = d.PATNO.to_numpy()
    fit = cross_fit(d[xs].to_numpy(dtype=float), d.E.to_numpy(dtype=float), pid, "clf", "boosting", seed=seed, tune=False)
    return round(fit["score"]["auc"], 3)


def _sensitivity(d, xs, cond):
    """The available-case p-value and effect, reported beside the primary complete-case test."""
    s = gcm_test(d, xs, cond, mode="available")
    return dict(p_available=s["p"], lead_pp_per_sd_available=s["lead_pp_per_sd"], rows_available=s["rows"]) if s is not None else {}


def _reason(cohort, d, node, arrow, allcols):
    t0 = time.time()
    xs = node["columns"]
    cond = [c for c in allcols if c not in xs]
    r = gcm_test(d, xs, cond)
    row = dict(cohort=cohort, reason=node["id"], label=node["label"], role=node["role"], columns=", ".join(xs), expected_sign=arrow["sign"],
               grade=arrow["grade"], locator=arrow.get("locator", ""), auc_alone=auc_alone(d, xs), **_sensitivity(d, xs, cond),
               seconds=round(time.time() - t0, 1))
    if r is None:
        return dict(row, note="untestable: too few complete rows")
    sign = np.sign(r["lead_pp_per_sd"])
    row.update(r, sign_agrees=(bool(sign == (1 if arrow["sign"] == "+" else -1)) if arrow["sign"] in ("+", "-") else ""))
    return row


def _block(cohort, d, bid, b, allcols):
    t0 = time.time()
    cond = [c for c in allcols if c not in b["columns"]]
    r = gcm_test(d, b["columns"], cond)
    row = dict(cohort=cohort, block=bid, label=b["label"], guideline=b["guideline"], columns=len(b["columns"]), auc_alone=auc_alone(d, b["columns"]),
               **_sensitivity(d, b["columns"], cond), seconds=round(time.time() - t0, 1))
    return dict(row, **r) if r is not None else dict(row, note="untestable")


def verdicts(reasons):
    """One verdict per reason across cohorts (the rule of the module docstring)."""
    rows = []
    for rid, g in reasons.groupby("reason", sort=False):
        sig = g[(g.p < 0.05)] if "p" in g else g.iloc[:0]
        agree = sig[sig.sign_agrees == True]        # noqa: E712 (object column)
        against = sig[sig.sign_agrees == False]     # noqa: E712
        context = g.expected_sign.iloc[0] == "context"
        if context:
            v = "replicated" if len(sig) >= 2 else "one cohort" if len(sig) == 1 else "not detected"
        else:
            v = ("opposite sign" if len(against) else "replicated" if len(agree) >= 2 else "one cohort" if len(agree) == 1 else "not detected")
        rows.append(dict(reason=rid, label=g.label.iloc[0], role=g.role.iloc[0], expected_sign=g.expected_sign.iloc[0], grade=g.grade.iloc[0],
                         locator=g.locator.iloc[0], verdict=v, significant_in=", ".join(C.COHORTS[c] for c in sig.cohort),
                         pp_per_sd=" / ".join(f"{x:+.2f}" if pd.notna(x) else "–" for x in g.get("lead_pp_per_sd", pd.Series([np.nan] * len(g))))))
    return pd.DataFrame(rows)


def main(argv=None):
    C.require_ppmi_approval("the Q1 decision-node audit")  # escalation-discovery: PROTOCOL.md §11 gate
    args = argv if argv is not None else sys.argv[1:]
    cohorts = [a for a in args if a in C.COHORTS] or list(C.COHORTS)
    quick = "--quick" in args
    allcols = coremap.adjustment_columns()
    data = {c: load(c) for c in cohorts}
    if quick:
        data = {c: d[d.PATNO.isin(d.PATNO.drop_duplicates().sample(frac=0.4, random_state=1))].reset_index(drop=True) for c, d in data.items()}
    t0 = time.time()
    jobs = [delayed(_reason)(c, d, node, arrow, allcols) for c, d in data.items() for node, arrow in coremap.decision_arrows()]
    jobs += [delayed(_block)(c, d, bid, b, allcols) for c, d in data.items() for bid, b in coremap.blocks().items()]
    res = Parallel(n_jobs=N_JOBS)(jobs)
    reasons = pd.DataFrame([r for r in res if "reason" in r])
    blocks = pd.DataFrame([r for r in res if "block" in r])
    allauc = pd.DataFrame([dict(cohort=c, block="everything recorded", label="all pre-decision covariates", guideline=False, columns=len(allcols),
                                auc_alone=auc_alone(d, allcols)) for c, d in data.items()])
    blocks = pd.concat([blocks, allauc], ignore_index=True)
    sfx = "_quick" if quick else ""
    out = C.results_dir("q1_decision_audit")  # escalation-discovery: own folder
    reasons.to_csv(out / f"audit_reasons{sfx}.csv", index=False)
    blocks.to_csv(out / f"audit_blocks{sfx}.csv", index=False)
    v = verdicts(reasons)
    v.to_csv(out / f"audit_verdicts{sfx}.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
    print(f"audit: {len(reasons)} reason tests and {len(blocks)} block rows in {time.time() - t0:.0f} s")
    print(reasons[[c for c in ["cohort", "reason", "p", "lead_pp_per_sd", "lead_se", "sign_agrees", "auc_alone", "rows"] if c in reasons]].to_string(index=False))
    print(blocks[[c for c in ["cohort", "block", "p", "auc_alone", "rows"] if c in blocks]].to_string(index=False))
    print(v[["reason", "verdict", "significant_in", "pp_per_sd"]].to_string(index=False))

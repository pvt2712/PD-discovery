"""arm_b — real PPMI: agreement and plausibility, not accuracy (PROTOCOL.md §7).

    uv run escd arm-b --data ppmi          # gated until the protocol is approved (PROTOCOL.md §11)
    uv run escd arm-b --data synthetic     # the same pipeline on one simulated cohort (a test of the code)

1. The expert DAG is tested first (falsify.py): implied independences against relabelled DAGs, and edge support.
2. Stability: every learner on B participant-level subsamples (90% of the participants of each cohort, without
   replacement; Banerjee, Andrews & Kummerfeld 2025 found subsampling frequencies far better calibrated than the
   bootstrap's). A pair is stable at frequency >= tau, tau calibrated in Arm A (`calibrate_tau`).
3. Agreement with the expert DAG, edge by edge (the expert DAG is not ground truth).
4. Negative controls: every learner without knowledge; arrowheads into the cohort context or into an exogenous node
   (sex, APOE e4, age, education, years since enrolment) and arrowheads pointing back in time are errors.
5. A blinded sheet of the disagreements for clinicians, mixed with control items.
Everything written is aggregate: pair-level frequencies, counts, rates.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import config as C
from . import expert as E
from . import falsify as F
from . import generator as G
from .graphs import MarkGraph
from .methods import registry
from .methods.base import DiscoveryData

B_SUBSAMPLES = 50             # Plan B (PROTOCOL.md §7.2)
FRACTION = 0.9
DEFAULT_TAU = 0.6
TAU_GRID = (0.5, 0.6, 0.7, 0.8, 0.9)
TAU_PRECISION = 0.8
TAU_REPS, TAU_SUBSAMPLES = 2, 25   # tau calibration: replicates of each base setting x subsamples (PROTOCOL.md §6.6)


# ---------------------------------------------------------------------------- data
def ppmi_panel():
    """Pooled PD cohorts from this repository's build (data/derived/grid.csv, subjects.csv): long panel and baseline
    table with the cohort context nodes. Gated (PROTOCOL.md §11)."""
    C.require_ppmi_approval("Arm B on PPMI")
    d = C.derived_dir()
    subj = pd.read_csv(d / "subjects.csv", low_memory=False)
    subj = subj[subj.cohort.isin(list(C.COHORTS))]
    grid = pd.read_csv(d / "grid.csv", low_memory=False)
    spec = E.load()
    base = pd.DataFrame({"PATNO": subj.PATNO, "era2": (subj.era == "PPMI2").astype(float),
                         "genetic": subj.genetic_enrolled.astype(float)})
    for k, col in C.BASELINE.items():
        base[k] = pd.to_numeric(subj[col], errors="coerce").to_numpy()
    long = grid[grid.PATNO.isin(base.PATNO)][["PATNO", "year"] + [v for v in spec.tv if v in grid]].copy()
    return long.reset_index(drop=True), base.reset_index(drop=True)


def synthetic_panel(n=1741, seed=777, kind="G1"):
    from . import missingness as M
    spec = E.load()
    gen = G.synthetic(kind)
    long, base = G.simulate(gen, n, seed)
    return M.PatternBank.synthetic(spec).apply(long, base, "mar", seed + 1)


def subset(long, base, name):
    """Sensitivity subsets (PROTOCOL.md §3): one cohort (PPMI 1.0 de novo) or the pooled cohorts minus one."""
    if name in (None, "all"):
        return long, base
    c = base.era2.astype(int).astype(str) + base.genetic.astype(int).astype(str)
    if name == "ppmi1":
        keep = c == "00"
    elif name == "no_ppmi1":
        keep = c != "00"
    elif name == "no_ppmi2":
        keep = c != "10"
    elif name == "no_genetic":
        keep = base.genetic.astype(int) == 0
    else:
        raise ValueError(name)
    ids = set(base.PATNO[keep])
    return long[long.PATNO.isin(ids)].reset_index(drop=True), base[keep].reset_index(drop=True)


def fixed_context(base):
    """Context nodes with one value in a subset (`genetic` in no_genetic; `era2` and `genetic` in ppmi1). Fixing a
    root node removes its edges, so the run leaves them out of the data and of the expert DAG (PROTOCOL.md §3)."""
    return tuple(c for c in E.CONTEXT if base[c].nunique() < 2)


def make_data(long, base, ids=None, seed=0, one_per_person=False, drop=()):
    """`drop`: the context nodes a subset fixes (`fixed_context`). Any other variable without variation stops here: a
    constant gives undefined tests, and PC, which cannot then separate it, would search every conditioning set."""
    spec = E.load()
    if ids is not None:
        keep = set(ids)
        long, base = long[long.PATNO.isin(keep)], base[base.PATNO.isin(keep)]
    ts = G.two_slice(long, base, spec)
    if one_per_person:                     # i.i.d. check: one random transition per participant
        ts = ts.groupby("PATNO", group_keys=False).sample(n=1, random_state=seed).reset_index(drop=True)
    obs = [n for n in E.observed(spec) if n not in drop]
    flat = [n for n in obs if ts[n].nunique() < 2]
    if flat:
        raise ValueError(f"no variation in {', '.join(flat)}")
    tiers = [t for t in ([n for n in tier if n not in drop] for tier in E.tiers(spec)) if t]
    return DiscoveryData(X=ts[obs].to_numpy(dtype=float), names=obs, groups=ts.PATNO.to_numpy(), tiers=tiers,
                         context=tuple(c for c in E.CONTEXT if c not in drop), long=long, base=base,
                         seed=int(seed)), ts


def subsample_ids(base, b, fraction=FRACTION, seed=0):
    """Participants of subsample b: `fraction` of each cohort (era2 x genetic stratum), without replacement."""
    rng = np.random.default_rng(seed * 100_003 + b)
    strata = base.era2.astype(int).astype(str) + base.genetic.astype(int).astype(str)
    out = []
    for _, g in base.groupby(strata.to_numpy()):
        ids = g.PATNO.to_numpy()
        out.extend(rng.choice(ids, size=int(round(fraction * len(ids))), replace=False))
    return out


# ---------------------------------------------------------------------------- one fit
def _fit_one(entry_name, b, long, base, configs, knowledge, seed, one_per_person=False, drop=()):
    entry = next(e for e in registry.panel() if e.name == entry_name)
    entry.method.cfg.update(configs.get(entry_name, configs.get(entry_name.replace("_noknow", ""), {})))
    ids = None if b < 0 else subsample_ids(base, b, seed=seed)
    try:                                   # a subsample in which a variable does not vary is a failed fit
        data, _ = make_data(long, base, ids, seed=seed + max(b, 0), one_per_person=one_per_person, drop=drop)
    except ValueError as ex:
        return dict(method=entry_name, b=b, graph=None, seconds=0.0, error=repr(ex)[:300])
    data.knowledge = knowledge
    t0 = time.time()
    try:
        g = entry.method.fit(data)
        return dict(method=entry_name, b=b, graph=g.to_json(), seconds=time.time() - t0, error="")
    except Exception as ex:
        return dict(method=entry_name, b=b, graph=None, seconds=time.time() - t0, error=repr(ex)[:300])


def frequencies(fits, names):
    """Pair-level adjacency and mark frequencies over subsample fits (b >= 0)."""
    subs = [MarkGraph.from_json(f["graph"]) for f in fits if f["b"] >= 0 and f["graph"] is not None]
    n = len(subs)
    rows = {}
    for g in subs:
        for (a, b), m in g.edges.items():
            r = rows.setdefault((a, b), {"adj": 0})
            r["adj"] += 1
            key = "".join(m)
            r[key] = r.get(key, 0) + 1
    out = []
    for (a, b), r in rows.items():
        marks = {k: v for k, v in r.items() if k != "adj"}
        modal = max(marks, key=marks.get)
        out.append(dict(a=a, b=b, freq=r["adj"] / n, modal_marks=modal, modal_share=marks[modal] / r["adj"]))
    df = pd.DataFrame(out, columns=["a", "b", "freq", "modal_marks", "modal_share"])
    df.attrs["n"] = n
    return df.sort_values("freq", ascending=False)


def stable_graph(freq, full_json, names, tau):
    full = MarkGraph.from_json(full_json) if full_json else None
    g = MarkGraph(names, "PAG")
    for r in freq.itertuples():
        if r.freq >= tau:
            m = full.marks(r.a, r.b) if full is not None and full.adjacent(r.a, r.b) else tuple(r.modal_marks)
            g.add(r.a, r.b, *m)
    return g


# ---------------------------------------------------------------------------- negative controls
def violations(g: MarkGraph, tiers, context=E.CONTEXT, exogenous=E.EXOGENOUS):
    """Counts on a graph learned without knowledge.
    - into_root: arrowheads into a context node, or into an exogenous node from a non-context node (any arrowhead
      there says the node has a cause, observed or latent);
    - backwards: cross-tier edges oriented later -> earlier (tail at the later node, arrowhead at the earlier one);
      possible_backwards: later o-> earlier. A bidirected cross-tier edge is not a time violation (it encodes
      confounding), so it is not counted;
    - backwards_rate: backwards / cross-tier edges oriented in either direction (for a DAG, the share of directed
      cross-tier edges pointing back in time, as in ppmi-causal/gradient_baselines.violation_stats)."""
    tof = {n: k for k, t in enumerate(tiers) for n in t}
    ctx, exo = set(context), set(exogenous)
    into_root = back = possible = directed = 0
    for (a, b), (ma, mb) in g.edges.items():
        for x, y, mark_at_y in ((a, b, mb), (b, a, ma)):          # arrowhead at y on the edge x *-> y
            if mark_at_y == "A" and (y in ctx or (y in exo and x not in ctx)):
                into_root += 1
        if a in tof and b in tof and tof[a] != tof[b]:
            m_early, m_late = (ma, mb) if tof[a] < tof[b] else (mb, ma)
            if (m_late, m_early) == ("T", "A"):
                back += 1
            elif (m_late, m_early) == ("C", "A"):
                possible += 1
            if {m_early, m_late} == {"T", "A"}:
                directed += 1
    return dict(into_root=into_root, backwards=back, possible_backwards=possible, cross_tier_directed=directed,
                backwards_rate=back / directed if directed else np.nan)


# ---------------------------------------------------------------------------- agreement and the clinician sheet
def agreement(g: MarkGraph, spec=None):
    """Each expert edge (year-t+1 pair) found or not, with the learner's marks; each learned pair the expert lacks."""
    spec = spec or E.load()
    exp = {frozenset(p): (p, k) for p, k in E.summary_pairs(spec) if p[0] in g.index and p[1] in g.index}
    rows = []
    for key, ((a, b), kind) in exp.items():
        m = g.marks(a, b)
        status = "not found" if m is None else ("found, compatible" if m[0] != "A" and m[1] != "T" else "found, reversed or bidirected")
        rows.append(dict(a=a, b=b, source="expert", kind=kind, status=status, marks="".join(m) if m else ""))
    for (a, b), m in g.edges.items():
        if frozenset((a, b)) in exp or E.slice_of(a) != 1 and E.slice_of(b) != 1 or a in E.CONTEXT or b in E.CONTEXT:
            continue
        rows.append(dict(a=a, b=b, source="data only", kind="", status="data only", marks="".join(m)))
    return pd.DataFrame(rows)


def _phrase(n):
    base = n[:-2] if E.slice_of(n) is not None else n
    lab = E.NODE_INFO.get(base, {}).get("label", base)
    when = {0: " (this year)", 1: " (next year)"}.get(E.slice_of(n), " (at enrolment)")
    return lab + when


def clinician_sheet(agree: pd.DataFrame, spec=None, seed=11, controls_per_side=None):
    """Blinded items: every disagreement (data-only pairs; expert edges not found) plus as many control items
    (confirmed expert edges and tier-respecting pairs neither source proposes), shuffled, without source or
    statistics. Returns (sheet for clinicians, key kept by the analyst)."""
    spec = spec or E.load()
    rng = np.random.default_rng(seed)
    dis = agree[agree.status.isin(["data only", "not found"])]
    conf = agree[agree.status == "found, compatible"]
    k = controls_per_side if controls_per_side is not None else max(1, len(dis) // 2)
    ctrl = conf.sample(min(k, len(conf)), random_state=seed) if len(conf) else conf
    names = E.observed(spec)
    proposed = {frozenset((r.a, r.b)) for r in agree.itertuples()}
    cands = [(a, b) for a in names for b in names if a != b and E.slice_of(b) == 1 and a not in E.CONTEXT
             and frozenset((a, b)) not in proposed and E.tier_of(spec).get(a, 9) <= E.tier_of(spec).get(b, 0)]
    pick = rng.choice(len(cands), size=min(k, len(cands)), replace=False) if cands else []
    items = [("disagreement: " + r.status, r.a, r.b) for r in dis.itertuples()]
    items += [("control: confirmed expert edge", r.a, r.b) for r in ctrl.itertuples()]
    items += [("control: proposed by neither", *cands[i]) for i in pick]
    order = rng.permutation(len(items))
    sheet, key = [], []
    for j, i in enumerate(order, 1):
        cat, a, b = items[i]
        sheet.append(dict(item=j, statement=f"Could {_phrase(a)} plausibly influence {_phrase(b)}?",
                          choice="", choices="plausible new hypothesis / likely artifact / cannot judge", comment=""))
        key.append(dict(item=j, category=cat, a=a, b=b))
    return pd.DataFrame(sheet), pd.DataFrame(key)


def ensemble_fits(fits, names):
    """The family-majority ensemble of every resample (PROTOCOL.md §5.1): 'ensemble' from the primary variants (the
    knowledge-using ones where they exist) for stability, 'ensemble_noknow' from the knowledge-free variants and the
    learners that take no knowledge, for the negative controls."""
    from .arm_a import ensemble_of
    know = {e.name: e.knowledge for e in registry.panel()}
    by_b = {}
    for f in fits:
        if f["graph"] is not None:
            by_b.setdefault(f["b"], {})[f["method"]] = MarkGraph.from_json(f["graph"])
    out = []
    for b, graphs in sorted(by_b.items()):
        g = ensemble_of(graphs, names)
        free = {m[: -len("_noknow")]: gr for m, gr in graphs.items() if m.endswith("_noknow")}
        free.update({m: gr for m, gr in graphs.items() if not m.endswith("_noknow") and not know.get(m, True)})
        g2 = ensemble_of(free, names)
        out.append(dict(method="ensemble", b=b, graph=g.to_json() if g else None, seconds=0.0, error=""))
        out.append(dict(method="ensemble_noknow", b=b, graph=g2.to_json() if g2 else None, seconds=0.0, error=""))
    return out


# ---------------------------------------------------------------------------- tau calibration (run in Arm A)
def calibrate_tau(source="synthetic", methods=("fci_cgcm",), reps=TAU_REPS, B=TAU_SUBSAMPLES, n_jobs=8):
    """PROTOCOL.md §7.2: on Arm A base-setting replicates (both generators), the precision of pairs selected at each
    tau; the smallest tau whose mean precision reaches 0.8 for the pre-specified anchor is used in Arm B."""
    from . import arm_a
    spec = E.load()
    gens = arm_a.load_generators(source)
    bank = arm_a.load_bank(source, spec)
    rows = []
    for s in [s for s in arm_a.settings(spec) if s.name in ("G1-base", "G2-base")]:
        truth = arm_a.get_truth(source, gens[s.generator], s.hidden)
        true_adj = {p for p in truth.mag.adjacencies() if p in truth.scored()}
        for r in range(reps):
            seed = arm_a.seed_of(s.name, r, 20_000_000)
            data, ts = arm_a.make_data(gens[s.generator], bank, s, seed)
            long, base = data.long, data.base
            for m in methods:
                fits = Parallel(n_jobs=n_jobs)(delayed(_fit_one)(m, b, long, base, {}, True, seed) for b in range(B))
                fq = frequencies(fits, data.names)
                for tau in TAU_GRID:
                    sel = {frozenset((x.a, x.b)) for x in fq.itertuples() if x.freq >= tau} & truth.scored()
                    prec = len(sel & true_adj) / len(sel) if sel else np.nan
                    rows.append(dict(setting=s.name, rep=r, method=m, tau=tau, selected=len(sel), precision=prec,
                                     recall=len(sel & true_adj) / max(1, len(true_adj))))
    df = pd.DataFrame(rows)
    out = arm_a.source_dir(source)
    df.to_csv(out / "stability_calibration.csv", index=False)
    agg = df.groupby(["method", "tau"]).precision.mean().reset_index()
    first = agg[(agg.method == methods[0]) & (agg.precision >= TAU_PRECISION)].tau
    tau = float(first.min()) if len(first) else max(TAU_GRID)
    (out / "stability_tau.json").write_text(json.dumps(dict(tau=tau, method=methods[0], target_precision=TAU_PRECISION)))
    return tau, agg


# ---------------------------------------------------------------------------- the decision rule (Arm B part)
def select(decision_arm_a: pd.DataFrame, stability: pd.DataFrame, controls: pd.DataFrame):
    """PROTOCOL.md §8: eligible = median subsample frequency of the full-data pairs >= 0.6 and the knowledge-free
    backwards rate's upper 95% limit below 0.5; the selected learner is the eligible one with the best Arm A score
    among the anchors and those contenders that beat the best anchor; otherwise the best eligible anchor."""
    d = decision_arm_a[decision_arm_a.primary.astype(bool) & decision_arm_a.role.isin(["anchor", "contender"])].copy()
    st = stability.set_index("method")
    nc = controls.set_index("method") if len(controls) else pd.DataFrame()
    def eligible(m):
        s_ok = m in st.index and st.loc[m, "median_freq"] >= 0.6
        key = f"{m}_noknow" if f"{m}_noknow" in nc.index else m
        c_ok = key in nc.index and nc.loc[key, "backwards_rate_hi"] < 0.5
        return bool(s_ok and c_ok)
    d["eligible"] = d.method.map(eligible)
    anchors = d[(d.role == "anchor") & d.eligible].sort_values("adj_f1_realistic", ascending=False)
    winners = d[(d.role == "contender") & d.eligible & d.get("beats_best_anchor", pd.Series(False, index=d.index)).fillna(False).astype(bool)]
    pool = pd.concat([anchors.head(1), winners]).sort_values("adj_f1_realistic", ascending=False)
    chosen = pool.method.iloc[0] if len(pool) else None
    return chosen, d


# ---------------------------------------------------------------------------- the runner
def run(data_label="synthetic", B=B_SUBSAMPLES, n_perm=200, method_names=None, n_jobs=8, source="synthetic", seed=0,
        panel=None, subset_name="all", one_per_person=False):
    """`panel` = (long, base) replaces the data source (tests); `subset_name` and `one_per_person` give the
    sensitivity analyses of PROTOCOL.md §3, written to their own folder, without the context nodes the subset fixes."""
    label = data_label + ("" if subset_name == "all" else f"_{subset_name}") + ("_one_per_person" if one_per_person else "")
    out = C.results_dir(f"arm_b/{label}")
    if panel is not None:
        long, base = panel
    else:
        long, base = ppmi_panel() if data_label == "ppmi" else synthetic_panel()
    long, base = subset(long, base, subset_name)
    drop = fixed_context(base)
    spec = E.load()
    data, ts = make_data(long, base, seed=seed, one_per_person=one_per_person, drop=drop)
    t0 = time.time()
    if drop:
        print(f"fixed by the subset, left out: {', '.join(drop)}", flush=True)
    # 1. the expert DAG first
    summ, stm = F.test_expert(ts, spec, n_perm=n_perm, seed=seed, drop=drop)
    sup = F.edge_support(ts, spec, seed=seed, drop=drop)
    (out / "expert_falsification.json").write_text(json.dumps(summ, indent=1))
    stm.to_csv(out / "expert_statements.csv", index=False)
    sup.to_csv(out / "expert_edge_support.csv", index=False)
    print(f"expert DAG: {summ['violation_rate']} of {summ['statements']} statements violated vs {summ['null_mean']} "
          f"under relabelling (p {summ['p_value']}); {time.time() - t0:.0f} s", flush=True)
    # 2-4. learners with knowledge (stability) and without (negative controls)
    cfg_path = C.results_dir(f"arm_a/{source}") / "tuned_configs.json"
    configs = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    tau_path = C.results_dir(f"arm_a/{source}") / "stability_tau.json"
    tau = json.loads(tau_path.read_text())["tau"] if tau_path.exists() else DEFAULT_TAU
    entries, missing = registry.available(registry.panel())
    entries = [e for e in entries if not method_names or e.name in method_names or e.name.replace("_noknow", "") in method_names]
    jobs = [(e.name, b, e.knowledge) for e in entries for b in range(-1, B)]
    heavy = [j for j in jobs if j[0].startswith("tetrad_")]
    light = [j for j in jobs if not j[0].startswith("tetrad_")]
    fits = Parallel(n_jobs=n_jobs)(delayed(_fit_one)(m, b, long, base, configs, k, seed, one_per_person, drop)
                                   for m, b, k in light)
    fits += Parallel(n_jobs=2)(delayed(_fit_one)(m, b, long, base, configs, k, seed, one_per_person, drop)
                               for m, b, k in heavy)
    stab, ctrl, agree_all = [], [], []
    names = data.names
    # stability for every primary learner and every knowledge-using variant; negative controls for every learner run
    # without knowledge (a primary that takes no knowledge, e.g. NOTEARS, gets both)
    groups = [(e.name, e.knowledge or e.primary, not e.knowledge) for e in entries]
    ens = ensemble_fits(fits, names)
    for name, is_stab, is_ctrl in (("ensemble", True, False), ("ensemble_noknow", False, True)):
        mine = [f for f in ens if f["method"] == name]
        if any(f["graph"] for f in mine):          # a run of one learner (the sensitivity runs, §3) has no ensemble
            fits += mine
            groups.append((name, is_stab, is_ctrl))
    for name, is_stab, is_ctrl in groups:
        mine = [f for f in fits if f["method"] == name]
        if not mine:
            continue
        full = next((f for f in mine if f["b"] < 0), None)
        fq = frequencies(mine, names)
        fq.to_csv(out / f"freq_{name}.csv", index=False)
        errors = sum(f["graph"] is None for f in mine)
        if full and full["graph"]:
            fg = MarkGraph.from_json(full["graph"])
            fr = fq.set_index(["a", "b"]).freq
            med = float(np.median([fr.get(k, 0.0) for k in fg.edges])) if fg.edges else np.nan
        else:
            fg, med = None, np.nan
        if is_stab:
            sg = stable_graph(fq, full["graph"] if full else None, names, tau)
            (out / f"stable_{name}.json").write_text(sg.dumps())
            ag = agreement(sg, spec).assign(method=name)
            agree_all.append(ag)
            stab.append(dict(method=name, full_edges=fg.n_edges() if fg else np.nan, stable_edges=sg.n_edges(), tau=tau,
                             median_freq=med, failed_fits=errors,
                             expert_found=float((ag[ag.source == "expert"].status != "not found").mean()) if len(ag) else np.nan))
        if is_ctrl:
            vs = [violations(MarkGraph.from_json(f["graph"]), data.tiers) for f in mine if f["graph"]]
            rates = np.array([v["backwards_rate"] for v in vs], dtype=float)
            fv = violations(fg, data.tiers) if fg else {}
            ctrl.append(dict(method=name, into_root_full=fv.get("into_root"), backwards_full=fv.get("backwards"),
                             possible_backwards_full=fv.get("possible_backwards"), backwards_rate_full=fv.get("backwards_rate"),
                             backwards_rate_hi=float(np.nanpercentile(rates, 97.5)) if np.isfinite(rates).any() else np.nan,
                             backwards_rate_lo=float(np.nanpercentile(rates, 2.5)) if np.isfinite(rates).any() else np.nan,
                             into_root_mean=float(np.mean([v["into_root"] for v in vs])) if vs else np.nan, fits=len(vs)))
    stab, ctrl = pd.DataFrame(stab), pd.DataFrame(ctrl)
    stab.to_csv(out / "stability.csv", index=False)
    ctrl.to_csv(out / "negative_controls.csv", index=False)
    agree = pd.concat(agree_all, ignore_index=True) if agree_all else pd.DataFrame()
    agree.to_csv(out / "agreement.csv", index=False)
    # 5. selection and the clinician sheet
    dec_path = C.results_dir(f"arm_a/{source}") / "decision_arm_a.csv"
    chosen = None
    if dec_path.exists() and len(stab):
        chosen, table = select(pd.read_csv(dec_path), stab, ctrl)
        table.to_csv(out / "selection.csv", index=False)
    chosen = chosen or ("fci_cgcm" if "fci_cgcm" in set(stab.method) else (stab.method.iloc[0] if len(stab) else None))
    if chosen is not None and len(agree):
        sheet, key = clinician_sheet(agree[agree.method == chosen], spec, seed=seed + 11)
        sheet.to_csv(out / "clinician_sheet.csv", index=False)
        key.to_csv(out / "clinician_sheet_key.csv", index=False)
    meta = dict(data=data_label, subset=subset_name, dropped=list(drop), one_per_person=one_per_person, rows=len(ts),
                participants=int(ts.PATNO.nunique()), B=B, fraction=FRACTION, tau=tau,
                selected=chosen, unavailable=missing, seconds=round(time.time() - t0))
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(prog="escd arm-b")
    ap.add_argument("--data", default="synthetic", choices=["synthetic", "ppmi"])
    ap.add_argument("--source", default="synthetic", help="Arm A output folder (results/arm_a/<name>) whose tuned "
                    "configurations, tau and decision table are used")
    ap.add_argument("--B", type=int, default=B_SUBSAMPLES)
    ap.add_argument("--perm", type=int, default=200)
    ap.add_argument("--subset", default="all", choices=["all", "ppmi1", "no_ppmi1", "no_ppmi2", "no_genetic"])
    ap.add_argument("--one-per-person", action="store_true")
    ap.add_argument("--methods", default="", help="comma-separated learners (default: the whole panel), or 'selected': "
                    "the learner the main run selected, as the sensitivity runs of PROTOCOL.md §3 use")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args(argv)
    methods = selected_methods(a.data) if a.methods == "selected" else [m for m in a.methods.split(",") if m]
    run(a.data, a.B, a.perm, methods, a.jobs, a.source, subset_name=a.subset, one_per_person=a.one_per_person)


def selected_methods(data_label):
    """The learner the main Arm B run selected; its knowledge-free variant comes along. A selected ensemble needs its
    members, so it returns [] (the whole panel)."""
    meta = C.results_dir("arm_b") / data_label / "meta.json"
    if not meta.exists():
        raise SystemExit(f"no main run to take the selection from: run `escd arm-b --data {data_label}` first")
    chosen = json.loads(meta.read_text())["selected"]
    if chosen is None:
        raise SystemExit("the main run selected no learner")
    return [] if chosen == "ensemble" else [chosen]

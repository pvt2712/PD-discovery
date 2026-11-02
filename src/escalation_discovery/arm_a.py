"""arm_a — recoverability: simulate from the expert DAG, learn, score (PROTOCOL.md §6).

    uv run escd arm-a [--source synthetic|fitted] [--reps 20] [--settings G1-base,...] [--methods fci_cgcm,...]
    uv run escd tune  [--source synthetic|fitted] [--reps 5]

Settings: around a realistic base (pooled PD cohorts: 1,741 participants, their follow-up and cohort mix, realistic
missingness, the trait latent, z-scores for scale-sensitive learners), one factor at a time: sample size, hidden
measured nodes, missingness mechanism, units, per-node person effects; every setting under both generators.
Each (setting, replicate) is one job: simulate, mask, build the two-slice table, run every method with its frozen
configuration, score against the scenario's truth, and score the family-majority ensemble. The deep-learning
learners, and so the ensemble they belong to, run in the realistic settings only (Plan B, PROTOCOL.md §6.3). Rows go to
results/arm_a/<source>/parts/ as they finish (a rerun skips finished jobs); graphs to graphs/ for the stability
calibration. A fitted source is derived from PPMI and stays under data/derived/arm_a/ (PROTOCOL.md §11).
"""
import argparse
import json
import pickle
import time
import zlib
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import config as C
from . import expert as E
from . import generator as G
from . import metrics as MT
from . import missingness as M
from . import truth as T
from .methods import ensemble, registry
from .graphs import MarkGraph
from .methods.base import DiscoveryData

BASE_N = 1741                 # pooled PD cohorts: PPMI 1.0 de novo 431 + PPMI 2.0 sporadic 957 + genetic PD 353
TUNE_REPS = 5                 # tuning replicates per generator, seeds disjoint from evaluation
BEAT_MARGIN = 0.02            # decision rule: a contender beats the best anchor by more than this in adjacency F1
HEAVY_JOBS = 2                # parallel jobs for the Tetrad pass (each JVM takes up to ESCD_JAVA_HEAP, default 6 GB)


@dataclass(frozen=True)
class Setting:
    name: str
    generator: str = "G1"
    n: int = BASE_N
    hidden: tuple = ()
    missing: str = "mar"
    standardize: bool = True
    person_effects: bool = False
    listwise: bool = False        # complete-row learners use listwise deletion (sensitivity, PROTOCOL.md §5.3)
    realistic: bool = False       # in the decision set R (PROTOCOL.md §8)
    reps: int = 10
    noknow: bool = False          # also run the knowledge-free variants


def settings(spec: E.ExpertSpec | None = None):
    """PROTOCOL.md §6.3: the realistic settings R (20 replicates) and the diagnostic ones (10)."""
    spec = spec or E.load()
    h1, h2 = E.hide_rule(spec, 1), E.hide_rule(spec, 2)
    out = []
    for g in ("G1", "G2"):
        out += [Setting(f"{g}-base", g, realistic=True, reps=20, noknow=True),
                Setting(f"{g}-hide1", g, hidden=h1, realistic=True, reps=20),
                Setting(f"{g}-mnar", g, missing="mnar", realistic=True, reps=20),
                Setting(f"{g}-n300", g, n=300), Setting(f"{g}-n431", g, n=431), Setting(f"{g}-n1000", g, n=1000),
                Setting(f"{g}-n4500", g, n=4500), Setting(f"{g}-hide2", g, hidden=h2),
                Setting(f"{g}-complete", g, missing="complete"), Setting(f"{g}-raw", g, standardize=False),
                Setting(f"{g}-person", g, person_effects=True), Setting(f"{g}-listwise", g, listwise=True)]
    return out


def is_heavy(name):
    return name.startswith("tetrad_")


def seed_of(name, rep, offset=0):
    return (zlib.crc32(f"{name}|{rep}".encode()) + offset) % (2 ** 31 - 1)


# ---------------------------------------------------------------------------- sources: generators, bank, truths
def source_dir(source):
    """Aggregate outputs of a source (truths, the default output folder); a fitted (PPMI-derived) source keeps its
    generators under data/derived."""
    return C.results_dir(f"arm_a/{source}")


def out_dir_of(source, label=None):
    """Output folder of a run: results/arm_a/<label>, by default the source's folder. A partial or demonstration run
    takes its own label, because a run skips every (setting, replicate) part already present in its folder."""
    return C.results_dir(f"arm_a/{label or source}")


def load_generators(source):
    if source == "synthetic":
        return {"G1": G.synthetic("G1"), "G2": G.synthetic("G2")}
    C.require_ppmi_approval("loading the PPMI-fitted generators")
    gens = {}
    for k in ("G1", "G2"):
        with open(C.derived_dir("arm_a") / f"generator_{k}.pkl", "rb") as f:
            gens[k] = pickle.load(f)
    return gens


def load_bank(source, spec):
    if source == "synthetic":
        return M.PatternBank.synthetic(spec)
    C.require_ppmi_approval("loading the PPMI missingness pattern bank")
    with open(C.derived_dir("arm_a") / "pattern_bank.pkl", "rb") as f:
        return pickle.load(f)


def get_truth(source, gen, hidden):
    """The scenario's truth, built once and cached (aggregate: structure only), with the generator's near-zero pairs."""
    key = "-".join(hidden) or "none"
    path = source_dir(source) / f"truth_{key}.json"
    if path.exists():
        tr = T.load(path)
    else:
        tr = T.build(gen.spec, f"{source}:{key}", context_edges=gen.context_edges, hidden=hidden)
        T.save(tr, path)
    if not gen.strengths:
        G.strengths(gen)
    tr.excluded_pairs = {p for p in G.weak_pairs(gen) if all(n in tr.observed for n in p)}
    return tr


# ---------------------------------------------------------------------------- one job
def make_data(gen, bank, setting: Setting, seed):
    spec = gen.spec
    long, base = G.simulate(gen, setting.n, seed, person_effects=setting.person_effects)
    long, base = bank.apply(long, base, setting.missing, seed + 1)
    ts = G.two_slice(long, base, spec)
    obs = E.observed(spec, hidden=setting.hidden)
    hid = set(setting.hidden)
    data = DiscoveryData(X=ts[obs].to_numpy(dtype=float), names=obs, groups=ts.PATNO.to_numpy(),
                         tiers=E.tiers(spec, hidden=setting.hidden), standardize=setting.standardize, listwise=setting.listwise,
                         long=long.drop(columns=[c for c in long.columns if c in hid]),
                         base=base.drop(columns=[c for c in base.columns if c in hid]), seed=int(seed))
    return data, ts


def run_methods(data, entries, configs, ensemble_too=True):
    """Fit every entry; returns {name: (MarkGraph or None, seconds, error)}, plus the ensemble of the primary entries."""
    out = {}
    for e in entries:
        e.method.cfg.update(configs.get(e.name, {}))
        data.knowledge = e.knowledge
        t0 = time.time()
        try:
            g, err = e.method.fit(data), ""
        except Exception as ex:                      # one learner failing must not lose the others
            g, err = None, repr(ex)[:300]
        out[e.name] = (g, time.time() - t0, err)
    if ensemble_too:
        g = ensemble_of({k: v[0] for k, v in out.items()}, data.names)
        if g is not None:
            out["ensemble"] = (g, 0.0, "")
    return out


def ensemble_of(graphs, names):
    """Family-majority ensemble of the primary anchors, controls and contenders (PROTOCOL.md §5.1)."""
    entries = {e.name: e for e in registry.panel()}
    prim = {m: g for m, g in graphs.items() if m in entries and entries[m].primary
            and entries[m].role in ("anchor", "contender", "control") and g is not None}
    if len(prim) < 2:
        return None
    return ensemble.majority(prim, {m: entries[m].method.family for m in prim}, names)


def _ensemble_job(setting: Setting, rep: int, source: str, out_dir):
    """Score the ensemble of a (setting, replicate) from the graphs both passes saved."""
    part = out_dir / "parts" / f"{setting.name}__{rep}__ensemble.csv"
    if part.exists():
        return str(part)
    graphs, meta = {}, None
    for grp in ("main", "heavy"):
        gp = out_dir / "graphs" / f"{setting.name}__{rep}__{grp}.json"
        if gp.exists():
            graphs.update({k: MarkGraph.from_json(v) for k, v in json.loads(gp.read_text()).items()})
        pp = out_dir / "parts" / f"{setting.name}__{rep}__{grp}.csv"
        if pp.exists() and meta is None:
            meta = pd.read_csv(pp).iloc[0][["setting", "generator", "rep", "seed", "n", "rows", "participants",
                                             "incomplete_rows", "realistic"]].to_dict()
    if not graphs or meta is None:
        return None
    gens = load_generators(source)
    truth = get_truth(source, gens[setting.generator], setting.hidden)
    g = ensemble_of(graphs, truth.observed)
    if g is None:
        return None
    row = dict(meta, method="ensemble", seconds=0.0, error="")
    row.update(MT.score_all(g, truth, MT.AdjustmentOracle(truth.dag, truth.observed)))
    pd.DataFrame([row]).to_csv(part, index=False)
    (out_dir / "graphs" / f"{part.stem}.json").write_text(json.dumps({"ensemble": g.to_json()}))
    return str(part)


def _job(setting: Setting, rep: int, source: str, method_names, configs, out_dir, tune_cfg=None, seed_offset=0,
         group="main"):
    part = out_dir / "parts" / f"{setting.name}__{rep}__{group}{'__tune' if tune_cfg is not None else ''}.csv"
    if part.exists() or not method_names:
        return str(part)
    spec = E.load()
    gens = load_generators(source)
    gen = gens[setting.generator]
    bank = load_bank(source, spec)
    truth = get_truth(source, gen, setting.hidden)
    oracle = MT.AdjustmentOracle(truth.dag, truth.observed)
    seed = seed_of(setting.name, rep, seed_offset)
    data, ts = make_data(gen, bank, setting, seed)
    entries = [e for e in registry.panel() if e.name in method_names and (setting.noknow or tune_cfg is not None or e.knowledge
                                                                            or e.primary)
               and (setting.realistic or not e.realistic_only)]
    rows, graphs = [], {}
    meta = dict(setting=setting.name, generator=setting.generator, rep=rep, seed=seed, n=setting.n, rows=len(ts),
                participants=int(ts.PATNO.nunique()), incomplete_rows=float(np.isnan(data.X).any(axis=1).mean()),
                realistic=setting.realistic)
    if tune_cfg is None:
        results = run_methods(data, entries, configs, ensemble_too=False)
    else:                                            # tuning: every grid configuration of every method
        results = {}
        for e in entries:
            for k, cfg in enumerate(e.method.grid()):
                e.method.cfg = dict(e.method.defaults(), **cfg)
                data.knowledge = e.knowledge
                t0 = time.time()
                try:
                    g, err = e.method.fit(data), ""
                except Exception as ex:
                    g, err = None, repr(ex)[:300]
                results[f"{e.name}#{k}"] = (g, time.time() - t0, err)
    for name, (g, secs, err) in results.items():
        row = dict(meta, method=name, seconds=round(secs, 2), error=err)
        if g is not None:
            row.update(MT.score_all(g, truth, oracle))
            graphs[name] = g.to_json()
        rows.append(row)
    part.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(part, index=False)
    gdir = out_dir / "graphs"
    gdir.mkdir(exist_ok=True)
    (gdir / f"{part.stem}.json").write_text(json.dumps(graphs))
    return str(part)


def run(source="synthetic", reps=None, setting_names=None, method_names=None, n_jobs=8, configs=None, label=None):
    """Two passes (the light learners with n_jobs workers, the Tetrad learners with HEAVY_JOBS), then the ensemble.
    `reps` overrides each setting's replicate count (quick runs)."""
    out_dir = out_dir_of(source, label)
    spec = E.load()
    sets = [s for s in settings(spec) if not setting_names or s.name in setting_names]
    entries, missing = registry.available(registry.panel())
    names = [e.name for e in entries if not method_names or e.name in method_names]
    if missing:
        print("not available (skipped):", missing)
    if configs is None:
        p = out_dir / "tuned_configs.json"
        configs = json.loads(p.read_text()) if p.exists() else {}
        if not configs:
            print("no tuned configurations found: running the adapters' defaults (PROTOCOL.md §5.4 requires `escd tune` first)")
    gens = load_generators(source)                       # build the truths once before the workers start
    for s in sets:
        get_truth(source, gens[s.generator], s.hidden)
    t0 = time.time()
    light, heavy = [m for m in names if not is_heavy(m)], [m for m in names if is_heavy(m)]
    jobs = [(s, r) for s in sets for r in range(reps if reps is not None else s.reps)]
    Parallel(n_jobs=n_jobs)(delayed(_job)(s, r, source, light, configs, out_dir, group="main") for s, r in jobs)
    if heavy:
        Parallel(n_jobs=HEAVY_JOBS)(delayed(_job)(s, r, source, heavy, configs, out_dir, group="heavy") for s, r in jobs)
    Parallel(n_jobs=n_jobs)(delayed(_ensemble_job)(s, r, source, out_dir) for s, r in jobs if s.realistic)
    print(f"arm A ({source} -> {out_dir.name}): {len(jobs)} datasets in {time.time() - t0:.0f} s")
    return summarize(source, label=label)


def tune(source="synthetic", reps=TUNE_REPS, method_names=None, n_jobs=8, label=None):
    """Tuning on synthetic data only (PROTOCOL.md §5.4): every grid configuration on TUNE_REPS replicates of each
    generator's base setting, seeds disjoint from the evaluation seeds; the configuration with the highest mean
    adjacency F1 (ties: the first in the grid) is frozen for Arm A and Arm B."""
    out_dir = out_dir_of(source, label)
    spec = E.load()
    entries, missing = registry.available(registry.panel())
    names = [e.name for e in entries if e.primary and (not method_names or e.name in method_names)]
    base = [s for s in settings(spec) if s.name in ("G1-base", "G2-base")]
    gens = load_generators(source)
    for s in base:
        get_truth(source, gens[s.generator], s.hidden)
    light, heavy = [m for m in names if not is_heavy(m)], [m for m in names if is_heavy(m)]
    Parallel(n_jobs=n_jobs)(delayed(_job)(s, r, source, light, {}, out_dir, tune_cfg=True, seed_offset=10_000_000)
                            for s in base for r in range(reps))
    if heavy:
        Parallel(n_jobs=HEAVY_JOBS)(delayed(_job)(s, r, source, heavy, {}, out_dir, tune_cfg=True, seed_offset=10_000_000,
                                                  group="heavy") for s in base for r in range(reps))
    df = pd.concat([pd.read_csv(p) for p in sorted((out_dir / "parts").glob("*__tune.csv"))], ignore_index=True)
    df["entry"], df["config"] = df.method.str.split("#").str[0], df.method.str.split("#").str[1].astype(int)
    df = df[df.entry.isin(names)]
    agg = df.groupby(["entry", "config"]).agg(adj_f1=("adj_f1", "mean"), ah_f1=("ah_f1", "mean"), seconds=("seconds", "mean"),
                                              n=("adj_f1", "size")).reset_index()
    configs = {}
    lookup = {e.name: e for e in entries}
    for name, g in agg.groupby("entry"):
        best = g.sort_values(["adj_f1", "config"], ascending=[False, True]).iloc[0]
        configs[name] = lookup[name].method.grid()[int(best.config)]
        if f"{name}_noknow" in lookup:                   # the knowledge-free variant keeps its method's configuration
            configs[f"{name}_noknow"] = configs[name]
    agg.to_csv(out_dir / "tuning.csv", index=False)
    (out_dir / "tuned_configs.json").write_text(json.dumps(configs, indent=1))
    print(json.dumps(configs, indent=1))
    return configs


# ---------------------------------------------------------------------------- summaries and the Arm A decision table
KEY = ["adj_f1", "adj_precision", "adj_recall", "ah_f1", "ah_precision", "ah_recall", "ahmag_f1", "shd", "sid_lower",
       "sid_upper", "pa_mean_f1", "pa_decision_recall", "expert_recall", "adj_f1_allpairs", "seconds"]


def collect(source, label=None):
    parts = sorted((out_dir_of(source, label) / "parts").glob("*.csv"))
    parts = [p for p in parts if not p.stem.endswith("__tune")]
    return pd.concat([pd.read_csv(p) for p in parts], ignore_index=True) if parts else pd.DataFrame()


def summarize(source="synthetic", boot=2000, seed=0, label=None):
    df = collect(source, label)
    if df.empty:
        return df
    out_dir = out_dir_of(source, label)
    cols = [c for c in KEY if c in df]
    summ = df.groupby(["setting", "method"])[cols].agg(["mean", "std", "count"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ.reset_index().to_csv(out_dir / "summary.csv", index=False)
    decision = decision_table(df, boot, seed)
    decision.to_csv(out_dir / "decision_arm_a.csv", index=False)
    generator_agreement(df).to_csv(out_dir / "generator_agreement.csv", index=False)
    return decision


def generator_agreement(df):
    """PROTOCOL.md §6.5: Spearman correlation between the methods' mean adjacency F1 under G1 and under G2, setting by
    setting; below 0.5 the ranking depends on the generator."""
    from scipy.stats import spearmanr
    d = df[df.adj_f1.notna()].copy()
    d["stem"] = d.setting.str.split("-", n=1).str[1]
    rows = []
    for stem, g in d.groupby("stem"):
        m = g.groupby(["generator", "method"]).adj_f1.mean().unstack(0)
        if {"G1", "G2"} <= set(m.columns):
            m = m.dropna()
            rho = spearmanr(m.G1, m.G2)[0] if len(m) >= 3 else float("nan")
            rows.append(dict(setting=stem, methods=len(m), spearman=round(float(rho), 3), generator_dependent=bool(rho < 0.5)))
    return pd.DataFrame(rows)


def decision_table(df, boot=2000, seed=0):
    """PROTOCOL.md §8: mean adjacency F1 over the realistic settings per method; each contender's paired difference
    from the best anchor with a replicate-bootstrap 95% CI; 'beats' = difference > 0.02 and CI lower bound > 0."""
    roles = {e.name: (e.role, e.primary, e.method.family) for e in registry.panel()}
    roles["ensemble"] = ("contender", True, "ensemble")
    real = df[df.realistic.astype(bool) & df.adj_f1.notna()]
    if real.empty:
        return pd.DataFrame()
    wide = real.pivot_table(index=["setting", "rep"], columns="method", values="adj_f1")
    means = wide.mean()
    anchors = [m for m in wide.columns if roles.get(m, ("",))[0] == "anchor" and roles[m][1]]
    best = max(anchors, key=lambda m: means[m]) if anchors else None
    rng = np.random.default_rng(seed)
    rows = []
    for m in wide.columns:
        role, primary, family = roles.get(m, ("", False, ""))
        r = dict(method=m, role=role, primary=primary, family=family, adj_f1_realistic=round(float(means[m]), 4),
                 datasets=int(wide[m].notna().sum()), best_anchor=best)
        if best is not None and m != best:
            d = (wide[m] - wide[best]).dropna().to_numpy()
            if len(d):
                bs = rng.choice(d, size=(boot, len(d)), replace=True).mean(1)
                lo, hi = np.percentile(bs, [2.5, 97.5])
                r.update(diff_vs_best_anchor=round(float(d.mean()), 4), diff_lo=round(float(lo), 4), diff_hi=round(float(hi), 4),
                         beats_best_anchor=bool(d.mean() > BEAT_MARGIN and lo > 0))
        rows.append(r)
    return pd.DataFrame(rows).sort_values("adj_f1_realistic", ascending=False)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="escd arm-a")
    ap.add_argument("--source", default="synthetic", choices=["synthetic", "fitted"])
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--settings", default="")
    ap.add_argument("--methods", default="")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--label", default=None, help="output folder results/arm_a/<label> (default: the source)")
    a = ap.parse_args(argv)
    sets = [s for s in a.settings.split(",") if s]
    meths = [m for m in a.methods.split(",") if m]
    if a.tune:
        return tune(a.source, a.reps or TUNE_REPS, meths, a.jobs, a.label)
    res = run(a.source, a.reps, sets, meths, a.jobs, label=a.label)
    if res is not None and len(res):
        pd.set_option("display.width", 200)
        print(res.to_string(index=False))

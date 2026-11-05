## 1. The data layer: from raw PPMI tables to the analysis panel

> **Changed on 5 November 2026** (audit A4, A5, A6, A2; [§8](08_audit_findings.md), "Status after the revision"). This chapter describes commit `a9b577f`. Since then:
> - `_ledd` (§1.19) counts a dose change within the visit's month once, at the new dose (`_in_force`), and adds the long-duration exposure through a slow effect compartment;
> - `LEVODOPA` (§1.13) has the missing names and no longer matches carbidopa alone;
> - `visits` (§1.22) no longer uses an untreated-looking examination when the log shows a dose in force;
> - `analysis_panel` (§1.30) takes the new exposure and reports the year-0 and landmark variances;
> - `screening_retest` (§1.31) scales the error variance by the pairs' variance;
> - `build_analysis_panel.py` (§1.51) converts reliabilities into error variances with that scale and reports the margins.

Every number the study reports is computed from a handful of tables that this layer derives from the raw download of the Parkinson's Progression Markers Initiative (PPMI). This chapter reads that derivation block by block. When a later result surprises you (a participant classed as PIGD whom you expected to be tremor-dominant, a domain that is missing at a visit where the form exists, a dose of 700 mg/day where you expected 400), the explanation is often here, so it is worth reading slowly.

**Contents.** [1.0 Overview](#10-overview) · [1.1 DEFAULT and load](#11-default-and-load--the-pre-specified-constants) · [1.2 Panel](#12-panel--the-fields-and-the-checks-of-__post_init__) · [1.3 n, d, p, n_groups, subset, with_y, n_transitions](#13-n-d-p-n_groups-subset-with_y-n_transitions--sizes-and-copies) · [1.4 aggregate_summary](#14-aggregate_summary--counts-by-phenotype) · [1.5 two_slice_table](#15-two_slice_table--step-1s-pooled-year-to-year-transitions) · [1.6 save_panel and load_panel](#16-save_panel-and-load_panel--the-panel-on-disk-without-pickling) · [1.7 REVERSED and orient](#17-reversed-and-orient--higher-means-worse) · [1.8 NormalScores.fit](#18-normalscoresfit--mid-ranks-and-the-normal-scores-table) · [1.9 NormalScores.transform and NormalScores.inverse](#19-normalscorestransform-and-normalscoresinverse--the-frozen-map-and-its-inverse) · [1.10 RAW, NOT_GRID, ENROLLED](#110-raw-not_grid-enrolled--where-the-download-is-and-which-events-and-statuses-count) · [1.11 TREMOR_III, BRADY_RIGID, AXIAL, FUNCTION_II](#111-tremor_iii-brady_rigid-axial-function_ii--the-items-of-the-motor-and-function-domains) · [1.12 STEBBINS_TREMOR, STEBBINS_PIGD, TD_CUT, PIGD_CUT](#112-stebbins_tremor-stebbins_pigd-td_cut-pigd_cut--the-phenotype-rules-items-and-cut-offs) · [1.13 GENETIC_FLAGS, GDS_YES, GDS_NO, RBDSQ_ITEMS, RBDSQ_NEURO, LEVODOPA, DOMAINS](#113-genetic_flags-gds_yes-gds_no-rbdsq_items-rbdsq_neuro-levodopa-domains--the-remaining-vocabulary) · [1.14 read](#114-read--locating-one-ppmi-table) · [1.15 mdate](#115-mdate--dates-at-month-resolution) · [1.16 items_valid, item_sum, in_range](#116-items_valid-item_sum-in_range--valid-item-codes-and-the-80-rule) · [1.17 participants](#117-participants--one-row-per-participant-of-ppmis-pd-cohort) · [1.18 _motor](#118-_motor--the-part-iii-examination-of-each-visit-and-its-motor-state) · [1.19 _ledd](#119-_ledd--the-ledd-in-force-on-each-visit-date) · [1.20 visits (lines 191–209)](#120-visits-lines-191209--the-visit-table-and-its-add-helper) · [1.21 visits (lines 211–247)](#121-visits-lines-211247--the-five-questionnaire-scorers) · [1.22 visits (lines 248–260)](#122-visits-lines-248260--orientation-remote-visits-treatment-and-complications) · [1.23 stebbins](#123-stebbins--tremor-dominant-or-pigd) · [1.24 clinical_subtype](#124-clinical_subtype--the-provisional-fereshtehnejad-subtype) · [1.25 annual_grid](#125-annual_grid--one-visit-per-participant-and-year) · [1.26 build (lines 326–356)](#126-build-lines-326356--the-study-population-and-the-exclusion-cascade) · [1.27 build (lines 357–383)](#127-build-lines-357383--the-wp0-visit-table-dose-changes-and-the-24-month-endpoint) · [1.28 flow_table](#128-flow_table--participant-flow-by-era) · [1.29 CORE_DOMAINS and TREATED_DOMAINS](#129-core_domains-and-treated_domains--what-enters-h-p) · [1.30 analysis_panel](#130-analysis_panel--the-h-p-panel) · [1.31 screening_retest](#131-screening_retest--testretest-reliability-icc21) · [1.32 TD_CUT, PIGD_CUT, QUANTILE_MIN_N, SITE_BANDS, Columns](#132-td_cut-pigd_cut-quantile_min_n-site_bands-columns--the-audits-constants-and-input-schema) · [1.33 _small and protect_1way](#133-_small-and-protect_1way--suppression-in-a-one-way-table) · [1.34 protect_2way](#134-protect_2way--complementary-suppression-in-a-cross-table) · [1.35 _share, _quantiles, _count_series](#135-_share-_quantiles-_count_series--shares-quantiles-and-counts) · [1.36 domain_summaries](#136-domain_summaries--means-standard-deviations-and-floorceiling-shares) · [1.37 audit](#137-audit--the-wp0-feasibility-tables) · [1.38 DISPLAY, LABELS, ARROW, MINUS, MIN_RATERS](#138-display-labels-arrow-minus-min_raters--the-consensus-modules-vocabulary) · [1.39 domain and parse_link](#139-domain-and-parse_link--reading-a-link-label) · [1.40 answer](#140-answer--one-raters-answer-to-one-link) · [1.41 candidates](#141-candidates--the-questionnaires-candidate-links) · [1.42 read_table, read_csv, read_docx, read_rater](#142-read_table-read_csv-read_docx-read_rater--reading-a-returned-questionnaire) · [1.43 tally](#143-tally--anonymised-counts-for-one-link) · [1.44 round1](#144-round1--unanimity) · [1.45 round2 and key](#145-round2-and-key--majority-or-the-investigator) · [1.46 build_map](#146-build_map--the-draft-expert-map) · [1.47 round2_markdown](#147-round2_markdown--the-round-2-table) · [1.48 scripts/_common.py](#148-scripts_commonpy--root-data-results-require_inside-require_approval-to_json) · [1.49 scripts/build_ppmi_tables.py](#149-scriptsbuild_ppmi_tablespy--used-manifest-main) · [1.50 scripts/run_wp0_audit.py](#150-scriptsrun_wp0_auditpy--main) · [1.51 scripts/build_analysis_panel.py](#151-scriptsbuild_analysis_panelpy--main) · [1.52 scripts/wp1_consensus.py](#152-scriptswp1_consensuspy--save-load-read_raters-main) · [1.53 Notes for the author](#153-notes-for-the-author)

### 1.0 Overview

**What goes in.**
- The raw PPMI download, outside the repository at `../PPMI/raw` (15 tables in 5 folders, listed in `scripts/build_ppmi_tables.py`, `USED`): participant status, demographics, diagnosis history, conclusion of participation; MDS-UPDRS (the Movement Disorder Society revision of the Unified Parkinson's Disease Rating Scale) Parts I, II, III and IV; the Modified Schwab & England activities of daily living scale (S&E); the Montreal Cognitive Assessment (MoCA); the 15-item Geriatric Depression Scale (GDS-15); the autonomic scale SCOPA-AUT; the Epworth Sleepiness Scale (ESS); the REM sleep behaviour disorder screening questionnaire (RBDSQ); and the log of levodopa-equivalent daily doses (LEDD).
- The pre-specified constants, `configs/protocol_v0.5.toml`.
- For work package 1 (WP1), the expert questionnaire `docs/clinical_expert_questionnaire.md` and the raters' returned questionnaires.

**What comes out.**

| file | one row per | made by | level |
|---|---|---|---|
| `data/derived/participants.parquet` | participant of PPMI's PD cohort | `ppmi.build` | participant |
| `data/derived/grid.parquet` | participant × study year (the annual grid) | `ppmi.build` | participant |
| `data/derived/wp0_visits.parquet` | grid row of the primary population, in the WP0 audit's schema | `ppmi.build` | participant |
| `results/wp0/participant_flow.json`, `phenotype_axes.json`, `retest_reliability.json`, `manifest.json` | — | `scripts/build_ppmi_tables.py` | aggregate |
| `results/wp0/wp0_audit.json` | — | `wp0.audit` | aggregate |
| `data/derived/panel_tdpigd.npz`, `panel_spec.json`, `normal_scores.npz` | the H-P panel, its specification, the frozen transform | `ppmi.analysis_panel` | participant |
| `results/wp1/round1.json`, `round2_section_E.md`, `round2.json`; `configs/expert_map_draft.json` | — | `wp1` | aggregate (no PPMI data) |

**How the files fit together.**

```text
PPMI download (15 tables)                           configs/protocol_v0.5.toml ── config.load
 │  ppmi.read, ppmi.mdate                                       (read by the WP4 run script)
 ├─ participants() ......... PD cohort, era, genetic enrolment, dates, death
 └─ visits(ids) ............ one row per participant × event
       ├─ _motor ........... Part III: OFF/untreated and ON examinations, motor state, 3 motor domains
       ├─ add(...) ......... Part II, I, IV, S&E, MoCA, GDS, SCOPA-AUT, ESS, RBDSQ → 9 more domains
       └─ _ledd ............ LEDD in force on the visit date (total, levodopa, drug classes)
 build():  annual_grid ── stebbins ── exclusion cascade ── clinical_subtype ── WP0 visit table
 │                                                              │
 │   data/derived/participants, grid, wp0_visits                └─ wp0.audit ─▶ results/wp0/wp0_audit.json
 │                                                                  (protect_1way / protect_2way)
 └─ analysis_panel(): TD and PIGD from year 1, transforms.NormalScores (frozen), z, dose, exposure
        └─ data.Panel ── save_panel ─▶ data/derived/panel_tdpigd.npz ── load_panel ─▶ the engine (§2–§5)
                       └─ two_slice_table ─▶ Step 1 (§4)
questionnaires ── wp1.read_rater ── round1 ── round2 ── build_map ─▶ configs/expert_map_draft.json
```

**Which protocol sections the code implements.**

| protocol | what | code |
|---|---|---|
| §4 Population and data | primary population, annual grid, landmark at year 1 | `participants`, `annual_grid`, `build`, `analysis_panel` |
| §5 Phenotypes | TD/PIGD by Stebbins 2013; the Fereshtehnejad clinical subtype; overlap and stability | `stebbins`, `clinical_subtype`, `wp0.audit` |
| §6.1 Domains | items, missing-item rule, orientation (higher = worse), treatment, motor state | item lists, `items_valid`, `item_sum`, `visits`, `_motor`, `_ledd` |
| §6.2–§6.3 Time | tiers of the two-slice table; the interval rule | `two_slice_table`, `wp0.audit` |
| §6.4 Expert map | two-round consensus | `wp1`, `scripts/wp1_consensus.py` |
| §7.1 Measurement | one frozen normal-scores transform; fixed measurement-error variances (D28) | `transforms`, `analysis_panel`, `screening_retest`, `scripts/build_analysis_panel.py` |
| §3, WP0 | the feasibility audit | `wp0`, `scripts/run_wp0_audit.py` |
| §14 Governance | approval guard, folders, disclosure control, provenance manifest | `scripts/_common.py`, `wp0.protect_*`, `manifest` |

In the text, a bare section number (§1.19, §2) is a section of this guide; a protocol section is written "protocol §6.1", and the decisions D1–D30 are those of protocol §17.

**New notation for this section.**
- $x_{j}$ is the score of item $j$ of a scale with $n$ items; $\mathcal O$ is the set of items with a valid score (present and within the item's range).
- $\mathrm{NS}_j(\cdot)$ is the frozen normal-scores map of domain $j$ (§1.8).
- $L_{tot}(t)$ and $L_{lev}(t)$ are the total and the levodopa LEDD in force at date $t$, in mg/day; $\mu_e$ is the multiplier of a catechol-O-methyltransferase (COMT) inhibitor episode $e$.
- $m$ is the number of months from the participant's anchor date (enrolment) to a visit, and $a_{ik}$ the grid year of visit $k$.
- $R_i$ is the Stebbins tremor/PIGD ratio of participant $i$.
- $c_0$ is the disclosure threshold (`min_cell`, 5): a count from 1 to $c_0-1$ is "small".

In the code, `t` in a `Panel` is **years since the enrolment date**, so the year-1 landmark visit sits near $t=1$. The shared notation of §0.4 says "years from the landmark"; only differences of $t$ enter the likelihood, so this does not change any result (§1.53, note 15).

**How the examples were run.** No example touches PPMI data. Every example runs the repository's own functions on synthetic data, from scripts in [`docs/guide_examples/s01/`](../guide_examples/s01/). Run one from the repository folder:

```text
OMP_NUM_THREADS=1 ../ppmi-causal/.venv/bin/python docs/guide_examples/s01/ex_ledd.py
```

Each script imports [`_setup.py`](../guide_examples/s01/_setup.py), which does three things:
- it makes a fresh temporary folder for the run and deletes it at exit;
- it wraps `pdx.ppmi.read` so that it refuses any table outside that folder. `participants(raw=RAW)` and `build(raw=RAW)` default to the real download, and the guard makes even such a call fail;
- it prints the temporary folder as `<tmp>`.

The functions that read raw tables run on the repository's mock download, `tests/ppmi_mock.py`, which writes made-up participants in PPMI's folder layout, with PPMI's file names and columns. Some examples edit a cell or two of the mock, or write a small hand-made table, to reach an edge case.

| mock PATNO | what it exercises |
|---|---|
| 1–12 | the common pattern: PD cohort, enrolled; PPMI 1.0 (enrolled 01/2015) except 12; visits BL, V04 and V06 a year apart; odd numbers have the TD pattern (tremor items 2, PIGD items 0), even numbers the PIGD pattern (tremor items 0, PIGD items 2); all other Part III items scored 1 |
| 3 | a remote visit `R04` on the anniversary and an in-person `V04` two months later |
| 5 | no V06; died 06/2016 |
| 9 | LRRK2 enrolment (genetic PD) |
| 10 | treated at baseline: OFF examination with `PDTRTMNT` = 1, Sinemet 300 mg/day since 06/2014 |
| 11 | no baseline visit |
| 12 | PPMI 2.0 (enrolled 01/2021), PIGD; Sinemet 300 mg/day from 06/2021, OFF examinations at years 1 and 2 |
| 13 | screen failed (never enrolled) |

Every mock visit has S&E 90, MoCA 27, GDS 0, ESS 8, RBDSQ 0, hallucinations 0 and motor complications 0 (2 once treated); SCOPA-AUT items are random 0/1. The environment was Python 3.13.12, numpy 2.5.4, pandas 3.0.6 and scipy 1.18.1.

---

### 1.1 `DEFAULT` and `load` — the pre-specified constants

[`config.py:7`](../../src/pdx/config.py#L7), [`config.py:10`](../../src/pdx/config.py#L10)

**What it does.** `DEFAULT` is the path of `configs/protocol_v0.5.toml`, found from where `config.py` sits: `parents[2]` climbs from `src/pdx/config.py` to `src/pdx`, `src`, and then the repository folder. `load(path=None)` opens that file (or another one passed in) and parses it with the standard library's `tomllib` into a dictionary of dictionaries. The file holds the constants the protocol fixes in advance, so that a run script reads them instead of hard-coding them.

**Inputs → outputs.** An optional path → `dict[str, dict]`, one inner dictionary per TOML table (`hp`, `engine`, `speed`, `localisation`, `step1`, `separate_graphs`, `qualification`). TOML types are kept: `12` is an `int`, `0.10` a `float`, `"emax"` a `str`.

**The mathematics.** There is nothing to compute. What matters is which constant governs which symbol elsewhere in the guide.

| key | value | symbol / role | protocol |
|---|---|---|---|
| `hp.alpha`, `hp.n_max`, `hp.h` | 0.05, 999, 20 | level of H-P; the cap $n_{max}$ and the Besag–Clifford count $h$ | §2.4, §7.3.1 |
| `engine.n_quad` | 12 | $K$, the Gauss–Hermite nodes $u_m, w_m$ | §7.1 |
| `engine.treatment`, `ed50`, `ec50_ldr` | "emax", 3.0, 3.0 | the form of $h_{ik}$; its half-maximal dose and exposure, in units of 100 mg LEDD | §6.1, D24 |
| `engine.noise_era_col` | 0 | the column of $z_i$ that multiplies $Q_g$ by $e^{\omega\,\mathrm{era}_i}$ | §2.3 |
| `localisation.eps` | 0.10 | $\varepsilon$ | §2.3, D12 |
| `step1.max_dt`, `interval_rule_share` | 1.5, 0.90 | "consecutive" without grid years (§1.5); the 90% rule of §6.3 | §6.3, D13 |

The data layer itself reads none of these: `ppmi.py` and `wp0.py` carry their own constants (§1.10–§1.13, §1.32). The TOML is read by `scripts/run_hp_real.py`, which builds the `ModelSpec` from `[engine]`.

**Worked example** ([`ex_config.py`](../guide_examples/s01/ex_config.py)):

```python
cfg = config.load()
print("tables:", list(cfg))
print("engine:", cfg["engine"])
alt = TMP / "amended.toml"
alt.write_text('[engine]\nn_quad = 20\ntreatment = "linear"\n')
print("amended copy:", config.load(alt))
```

```text
DEFAULT = configs/protocol_v0.5.toml
tables: ['hp', 'engine', 'speed', 'localisation', 'step1', 'separate_graphs', 'qualification']
engine: {'n_quad': 12, 'treatment': 'emax', 'ed50': 3.0, 'ec50_ldr': 3.0, 'noise_era_col': 0}
hp alpha, n_max, h: 0.05 999 20
types: eps float | n_quad int
amended copy: {'engine': {'n_quad': 20, 'treatment': 'linear'}}
missing file -> FileNotFoundError
```

**Design choices and pitfalls.**
- No schema is checked. A misspelt key in the TOML surfaces only as a `KeyError` where it is read, and an amended copy that leaves out a table loads without complaint (the example's copy has no `[hp]`).
- `parents[2]` assumes the source layout (`src/pdx/`). It holds because the scripts put `src/` on the import path; an installed wheel would point elsewhere.
- `path or DEFAULT` treats an empty string as "no path".

### 1.2 `Panel` — the fields and the checks of `__post_init__`

[`data.py:15`](../../src/pdx/data.py#L15), [`data.py:28`](../../src/pdx/data.py#L28)

**What it does.** `Panel` is the container the engine, Step 1 and the simulators all read: one ragged sequence of visits per participant, from the year-1 landmark on. It is a dataclass, so `Panel(y, t, group, ...)` just stores the arguments; `__post_init__` then normalises their types, fills in defaults and checks that the pieces fit together.

**Inputs → outputs.**

| field | per participant $i$ | meaning |
|---|---|---|
| `y` | array $(T_i, d)$ | $Y_{i1},\dots,Y_{iT_i}$; row 0 is the landmark visit; NaN = missing |
| `t` | array $(T_i,)$ | visit times in years (in the PPMI panel: since the enrolment date) |
| `group` | int | $g_i$: 0 = TD (the reference), 1 = PIGD |
| `z` | row of an $(N,p)$ array | $z_i$, centred baseline covariates; $(N,0)$ if none |
| `dose` | array $(T_i,)$ | current total LEDD at each visit, in units of 100 mg; zeros if not given |
| `exposure` | array $(T_i,)$ | the long-duration exposure (in the PPMI panel: current levodopa LEDD, 100 mg); zeros if not given |
| `on` | array $(T_i,)$ | 1 if the examination was in the ON state; zeros if not given |
| `year` | array $(T_i,)$ of int | grid year of each visit (1 = landmark); optional |
| `ids`, `domains`, `meta` | — | identifiers, domain names, a free dictionary |

After `__post_init__`: `y[i]` is a float array of shape $(T_i, d)$, `t[i]`, `dose[i]`, `exposure[i]`, `on[i]` are float vectors, `year[i]` an int vector, `group` an int array, `z` a float $(N,p)$ array.

**The mathematics.** The panel is the data of the measurement model of §0.4:

$$
\mathcal D=\big\{(g_i,\ z_i,\ (t_{ik},\ Y_{ik},\ \mathrm{dose}_{ik},\ \mathrm{expo}_{ik},\ \mathrm{on}_{ik})_{k=1}^{T_i})\big\}_{i=1}^{N}.
$$

The engine needs the intervals $\Delta t_{ik}=t_{i,k+1}-t_{ik}$ and computes the transition $E=e^{s_iA_g\Delta t_{ik}}$ over each. Three conditions make that well defined, and the code checks each one:

| condition | line | error |
|---|---|---|
| $\lvert\{y_i\}\rvert=\lvert\{t_i\}\rvert=\lvert\{g_i\}\rvert=N$ | 31–32 | "y, t and group must have one entry per participant" |
| $T_i$ rows of $Y_i$ = $T_i$ times | 35–37 | "each participant needs one time per visit" |
| $\Delta t_{ik}>0$ for every $k$ | 38–39, `np.diff(ti) <= 0` | "visit times must increase" |

A repeated time ($\Delta t=0$) would give $E=I$ and two observations of the same state; a decreasing time has no meaning. The reshape `reshape(len(v), -1)` (line 33) turns each `y[i]` into a matrix with one row per visit, so a participant with one visit can be given as `[[...]]`.

**Worked example** ([`ex_panel.py`](../guide_examples/s01/ex_panel.py)), three made-up participants with $d=2$:

```python
y = [np.array([[0.1, -0.2], [0.4, nan], [0.9, 0.3]]),     # participant 0 (TD): 3 visits, one domain missing
     np.array([[1.0, 0.5], [1.6, 1.1]]),                   # participant 1 (PIGD): 2 visits
     np.array([[-0.3, nan]])]                              # participant 2 (PIGD): landmark only
t = [[1.0, 2.1, 3.0], [0.9, 2.0], [1.2]]
p = Panel(y, t, group=[0, 1, 1], z=[[-0.5, 0.2], [0.5, -0.1], [0.0, -0.1]], year=[[1, 2, 3], [1, 2], [1]])
```

```text
== 1.2 defaults filled in by __post_init__ ==
y[2] shape: (1, 2) | t[0]: [1.  2.1 3. ] | group dtype: int64
dose[0]: [0. 0. 0.] | exposure[1]: [0. 0.] | on[2]: [0.]
z shape without covariates: (3, 0)

== 1.2 the checks ==
times not increasing               -> ValueError: visit times must increase
one time missing                   -> ValueError: each participant needs one time per visit
group too short                    -> ValueError: y, t and group must have one entry per participant
one visit given as a flat vector   -> ValueError: each participant needs one time per visit
domains differ between participants  -> accepted; d = 2 but y[1] has 3 columns
```

A single visit given as a flat vector `[0.1, -0.2]` is reshaped to two visits of one domain, $(2,1)$; the time check then catches it, because there is only one time. The last line shows what is not checked: participants with different numbers of domains are accepted.

**Design choices and pitfalls.**
- Ragged lists rather than one padded array: participants have 1 to 15 visits, and padding is done later, once, by the engine's `prepare` (`likelihood.py:63`).
- Defaults of zero for `dose`, `exposure` and `on` mean "no treatment term": the engine multiplies them by $\kappa$, so a synthetic panel without treatment needs no special case.
- The number of domains is taken from `y[0]` only (§1.3). A participant with a different $d$ passes `__post_init__` and fails later, in `prepare`, with a broadcasting error.
- `group` is cast to `int` (line 29); a float label such as `0.7` would silently become 0.

### 1.3 `n`, `d`, `p`, `n_groups`, `subset`, `with_y`, `n_transitions` — sizes and copies

[`data.py:55`](../../src/pdx/data.py#L55), [`data.py:70`](../../src/pdx/data.py#L70), [`data.py:78`](../../src/pdx/data.py#L78), [`data.py:81`](../../src/pdx/data.py#L81)

**What it does.** Four read-only properties give the sizes; `subset` keeps some participants (used by every participant bootstrap, the separate-graph subsamples and the per-group comparators); `with_y` swaps in new scores (used by the simulator to put simulated data on the observed visit pattern); `n_transitions` counts visit-to-visit transitions.

**Inputs → outputs.**
- `n` = $N$; `d` = columns of `y[0]`; `p` = columns of `z`; `n_groups` = $\max_i g_i+1$.
- `subset(idx)` → a new `Panel` with the participants `idx`, in that order (repeats allowed, as a bootstrap needs).
- `with_y(y)` → a new `Panel` with the same times, groups, covariates and treatment, and the new `y`.
- `n_transitions()` → $\sum_i (T_i-1)$.

**The mathematics.** Only counting:

$$
N=\texttt{len(y)},\qquad G=\max_i g_i+1,\qquad \#\text{transitions}=\sum_{i=1}^{N}(T_i-1).
$$

`subset` builds the new lists with `pick = lambda L: [L[i] for i in idx]` and calls `dataclasses.replace`, which runs `__post_init__` again, so a subset is checked like any panel.

**Worked example** (same three participants):

```python
print("n, d, p, n_groups, n_transitions:", p.n, p.d, p.p, p.n_groups, p.n_transitions())
s = p.subset([2, 0])
w = p.with_y([v + 1.0 for v in p.y])
print("only group 1 kept -> n_groups =", p.subset([1, 2]).n_groups, "| only group 0 kept -> n_groups =",
      p.subset([0]).n_groups)
```

```text
== 1.3 properties and helpers ==
n, d, p, n_groups, n_transitions: 3 2 2 2 3
subset([2, 0]): groups [1 0] | years [[1], [1, 2, 3]] | z rows [[0.0, -0.1], [-0.5, 0.2]]
with_y: new y[0][0] = [1.1 0.8] | old y[0][0] = [ 0.1 -0.2] | t shares memory: True
only group 1 kept -> n_groups = 2 | only group 0 kept -> n_groups = 1
```

The transitions are $2+1+0=3$.

**Design choices and pitfalls.**
- `n_groups` is $\max g+1$, not the number of distinct groups. A subset holding only PIGD participants still reports two groups, the first of them empty; a subset holding only TD reports one. Code that loops over `range(panel.n_groups)` must cope with an empty group (`aggregate_summary` does, §1.4).
- `with_y` and `subset` share the underlying arrays of `t`, `dose`, `exposure` and `on` with the original (`np.asarray` makes no copy of a float array). Nothing in the repository modifies them in place, but an in-place edit of one panel's times would change the other's.
- `subset` keeps `domains` and copies `meta`; `ids` is subset if present.

### 1.4 `aggregate_summary` — counts by phenotype

[`data.py:84`](../../src/pdx/data.py#L84)

**What it does.** Returns, for each group, the counts the run scripts print and write to `results/`: participants, visits, transitions, the median number of visits and a share of missing scores. It is the only `Panel` method meant for output, and it returns aggregates only.

**Inputs → outputs.** A `Panel` → `{"0": {...}, "1": {...}}` with keys `participants`, `visits`, `transitions`, `median_visits`, `share_missing`.

**The mathematics.** With $\mathcal I_g=\{i:g_i=g\}$ and $N_g=\lvert\mathcal I_g\rvert$:

$$
\text{visits}_g=\sum_{i\in\mathcal I_g}T_i,\qquad
\text{transitions}_g=\sum_{i\in\mathcal I_g}(T_i-1),\qquad
\text{share\_missing}_g=\frac{1}{N_g}\sum_{i\in\mathcal I_g}\frac{1}{T_i d}\sum_{k=1}^{T_i}\sum_{j=1}^{d}\mathbf 1\{Y_{ikj}\ \text{missing}\}.
$$

The last one is a mean of per-participant shares (`np.mean([np.isnan(self.y[i]).mean() for i in idx])`), not the pooled share of missing cells $\sum\sum\sum\mathbf 1\{\cdot\}/\sum_i T_i d$. The two differ when participants with few visits have more gaps.

**Worked example** (same three participants):

```python
print(p.aggregate_summary())
miss = [np.isnan(v).mean() for v in p.y[1:]]
pooled = np.isnan(np.vstack(p.y[1:])).mean()
```

```text
== 1.4 aggregate_summary ==
{'0': {'participants': 1, 'visits': 3, 'transitions': 2, 'median_visits': 3.0, 'share_missing': 0.16666666666666666}, '1': {'participants': 2, 'visits': 3, 'transitions': 1, 'median_visits': 1.5, 'share_missing': 0.25}}
PIGD: mean of per-participant shares = 0.25 | pooled share of cells = 0.1667
```

By hand, for PIGD: participant 1 has 0 of 4 cells missing and participant 2 has 1 of 2, so the mean of the shares is $(0+0.5)/2=0.25$, while the pooled share is $1/6=0.167$.

**Design choices and pitfalls.**
- The counts are not passed through disclosure control. They are meant for the console and for result files whose groups are large; a small group (fewer than $c_0$ participants) would be reported exactly.
- An empty group (possible after `subset`, §1.3) gives zeros and `share_missing` NaN rather than an error.

### 1.5 `two_slice_table` — Step 1's pooled year-to-year transitions

[`data.py:97`](../../src/pdx/data.py#L97)

**What it does.** Builds the table that Step 1's causal-discovery algorithm reads (protocol §6.2–§6.3): one row per pair of consecutive grid years of one participant, with the baseline covariates, optionally the interval, the scores at year $t$, optionally the dose at $t$ and $t+1$, and the scores at year $t+1$. It also returns each column's temporal tier, the participant behind each row (the resampling unit) and the column names.

**Inputs → outputs.** A `Panel`, an optional baseline matrix $(N,b)$ (Step 1 passes `panel.z`), `max_dt`, and three switches → `(X, tiers, who, cols)`:
- `X`: $(R, b+[1]+d+[2]+d)$ float array, NaN kept unless `drop_incomplete`;
- `tiers`: one integer per column;
- `who`: $(R,)$ participant indices;
- `cols`: names `b0…`, `interval`, `y<j>_t`, `dose_t`, `dose_t1`, `y<j>_t1`.

**The mathematics.** A row is formed for each $(i,k)$ with

$$
\text{consecutive}(i,k)=\begin{cases}a_{i,k+1}-a_{ik}=1 & \text{if grid years are present,}\\ \Delta t_{ik}\le \texttt{max\_dt}\ (1.5\text{ years}) & \text{otherwise,}\end{cases}
$$

and the row is

$$
X_r=\big(b_i,\ [\Delta t_{ik}],\ Y_{ik},\ [\mathrm{dose}_{ik},\ \mathrm{dose}_{i,k+1}],\ Y_{i,k+1}\big).
$$

The tiers encode the order of protocol §6.2, "baseline | year $t$ | treatment at $t+1$ | the rest of year $t+1$":

| columns | tier |
|---|---|
| baseline and interval | 0 |
| $Y_{ik}$ and $\mathrm{dose}_{ik}$ | 1 |
| $\mathrm{dose}_{i,k+1}$ (if included) | 2 |
| $Y_{i,k+1}$ | last = max + 1 |

How the block is built from the rule: the loop over `k in range(len(yi) - 1)` visits each adjacent pair once; `consecutive` (line 119) applies the rule above; `r` collects the pieces in the column order; `tiers` is built alongside `cols` so the two can never disagree; `last = max(tiers) + 1` puts year $t+1$ after the dose tier when it exists and right after year $t$ when it does not. `step1.step1_table` then adds 1 to every tier and moves era to tier 0, so that era sits before the other baseline variables.

**Worked example** ([`ex_two_slice.py`](../guide_examples/s01/ex_two_slice.py)). Participant 0 is seen in grid years 1, 2, 3 and 5 (year 4 missed) and has one missing score; participant 1 in years 1 and 2:

```python
p = Panel(y=[np.array([[0.1, 0.2], [0.3, 0.4], [0.5, nan], [0.9, 1.0]]),   # years 1, 2, 3, 5 (year 4 missed)
             np.array([[1.0, 1.1], [1.2, 1.3]])],                         # years 1, 2
          t=[[1.0, 2.1, 2.9, 5.0], [1.1, 2.0]], group=[0, 1],
          dose=[[0, 0, 3, 4], [1, 1]], year=[[1, 2, 3, 5], [1, 2]])
base = np.array([[-0.5], [0.5]])
X, tiers, who, cols = two_slice_table(p, base)
X, tiers, who, cols = two_slice_table(p, base, add_interval=True, include_dose=True, drop_incomplete=True)
```

```text
columns: ['b0', 'y0_t', 'y1_t', 'y0_t1', 'y1_t1']
tiers:   [0, 1, 1, 2, 2]
rows from participant: [0, 0, 1]
[[-0.5  0.1  0.2  0.3  0.4]
 [-0.5  0.3  0.4  0.5  nan]
 [ 0.5  1.   1.1  1.2  1.3]]

with interval and dose, complete rows only:
columns: ['b0', 'interval', 'y0_t', 'y1_t', 'dose_t', 'dose_t1', 'y0_t1', 'y1_t1']
tiers:   [0, 0, 1, 1, 1, 2, 3, 3]
[[-0.5  1.1  0.1  0.2  0.   0.   0.3  0.4]
 [ 0.5  0.9  1.   1.1  1.   1.   1.2  1.3]]

without grid years: rows from participant [0, 0, 1] (the 2.1-year step 2.9 -> 5.0 is dropped)
```

The step from year 3 to year 5 is not a row (a missed year). The row 2 → 3 keeps its NaN unless `drop_incomplete`, so that each conditional-independence test can delete only what it needs (test-wise deletion, protocol §7.11).

**Design choices and pitfalls.**
- With grid years, "consecutive" ignores the actual interval: two grid visits one grid year apart can be 4 to 20 months apart (each sits within ±4 months of its anniversary, §1.25). The interval rule of protocol §6.3 is then a separate check (`wp0.audit`, §1.37) and the `interval` column a separate option.
- The interval is placed in tier 0 with the baseline, as protocol §6.3 asks ("added … as a baseline-tier variable").
- `dose_t` shares tier 1 with the year-$t$ scores, and `dose_t1` gets its own tier 2, so the dose recorded at $t+1$ may affect the other year-$t+1$ measurements but not the reverse (protocol §6.1, last row of the domain table).

### 1.6 `save_panel` and `load_panel` — the panel on disk without pickling

[`data.py:149`](../../src/pdx/data.py#L149), [`data.py:160`](../../src/pdx/data.py#L160)

**What it does.** `save_panel` writes a panel as a compressed `.npz` of flat arrays: every participant's rows stacked, plus the vector of visit counts. `load_panel` reads it back and cuts the stacks at the right places. No Python objects are pickled, so loading cannot execute code (`allow_pickle=False`).

**Inputs → outputs.** `save_panel(panel, path)` → a file with arrays `lens`, `y`, `t`, `group`, `z`, `dose`, `exposure`, `on`, `domains` and, if present, `year`. `load_panel(path)` → a `Panel`.

**The mathematics.** With visit counts $T_1,\dots,T_N$, participant $i$ occupies rows $o_{i-1},\dots,o_i-1$ of each stacked array, where $o_0=0$ and $o_i=\sum_{q\le i}T_q$. `np.cumsum(f["lens"])[:-1]` gives the cut points $o_1,\dots,o_{N-1}$, and `np.split(a, cuts)` returns the $N$ blocks.

**Worked example** ([`ex_save_load.py`](../guide_examples/s01/ex_save_load.py)), three made-up participants with 2, 1 and 3 visits:

```python
save_panel(p, path)
with np.load(path) as f:
    print("stored arrays:", {k: f[k].shape for k in f.files})
    print("lens:", f["lens"], "-> cut points:", np.cumsum(f["lens"])[:-1])
q = load_panel(path)
```

```text
stored arrays: {'lens': (3,), 'y': (6, 2), 't': (6,), 'group': (3,), 'z': (3, 1), 'dose': (6,), 'exposure': (6,), 'on': (6,), 'domains': (2,), 'year': (6,)}
lens: [2 1 3] -> cut points: [2 3]
y identical: True | t identical: True
dose[2]: [0. 0. 3.] | year[2]: [1 2 3] | domains: ['tremor', 'axial']
ids after loading: None | meta after loading: {}
```

**Design choices and pitfalls.**
- `ids` and `meta` are not saved. The PPMI panel has no `ids` anyway (`analysis_panel` does not set them, which keeps identifiers out of the file), and its `meta` (`source`, `motor_on`) is also written to `panel_spec.json` by `scripts/build_analysis_panel.py`.
- `.npz` files are git-ignored (`*.npz` in `.gitignore`); the docstring adds that the file must stay inside `data/`.

### 1.7 `REVERSED` and `orient` — higher means worse

[`transforms.py:18`](../../src/pdx/transforms.py#L18), [`transforms.py:21`](../../src/pdx/transforms.py#L21)

**What it does.** The protocol orients every domain so that a higher score means worse (protocol §6.1). `orient` multiplies by −1 every column whose name is in `REVERSED` (compared in lower case): the MoCA and S&E, under their instrument names (`moca`, `schwab_england`, `se_adl`) and under the engine's domain names (`cognition`, `independence`).

**Inputs → outputs.** An array whose last axis holds the domains, the list of domain names, and optionally another reversed list → a float copy with those columns negated.

**The mathematics.**

$$
\tilde x_j=\begin{cases}-x_j & \text{if domain } j \text{ is higher-is-better,}\\ x_j & \text{otherwise.}\end{cases}
$$

Negation and the reflections the PPMI pipeline uses ($100-\text{S\&E}$, $30-\text{MoCA}$, §1.22) give the same analysis scale. Both are strictly decreasing maps, and the normal-scores transform depends on the data only through ranks, so $\mathrm{NS}(c-x)=\mathrm{NS}(-x)=-\mathrm{NS}(x)$ (derived in §1.8 and checked numerically there).

**Worked example** ([`ex_normal_scores.py`](../guide_examples/s01/ex_normal_scores.py)):

```python
raw = np.array([[26.0, 90.0, 3.0], [21.0, 70.0, 5.0]])
print(orient(raw, ["MoCA", "schwab_england", "tremor"]))
already = np.array([[4.0, 10.0]])          # ppmi.py output: cognition = 30 - MoCA, independence = 100 - S&E
print("orient applied to ppmi.py's already-oriented columns:", orient(already, ["cognition", "independence"]))
```

```text
== 1.7 orient ==
[[-26. -90.   3.]
 [-21. -70.   5.]]
REVERSED: ('moca', 'schwab_england', 'se_adl', 'cognition', 'independence')
orient applied to ppmi.py's already-oriented columns: [[ -4. -10.]]
```

**Design choices and pitfalls.**
- `orient` is not called by the PPMI pipeline: `ppmi.visits` orients by reflection (§1.22). Because `REVERSED` also lists the engine names `cognition` and `independence`, applying `orient` to `ppmi.py`'s output would flip those two domains back to higher-is-better, as the last line shows (§1.53, note 20).
- The docstring asks that the list be passed explicitly when the WP1 domain definitions change; nothing enforces it.

### 1.8 `NormalScores.fit` — mid-ranks and the normal-scores table

[`transforms.py:32`](../../src/pdx/transforms.py#L32), [`transforms.py:36`](../../src/pdx/transforms.py#L36)

**What it does.** For each column (domain), `fit` builds a lookup table from every distinct observed value to its normal score: the standard-normal quantile of its mid-rank. Protocol §7.1 asks for one such transform, "computed once on the pooled primary population over all visits and groups, with mid-ranks for ties, and then frozen". The fitted object holds only the tables; `transform` (§1.9) applies them.

**Inputs → outputs.** An $(n_{rows}, d)$ array with NaN for missing → `self` with `tables[j] = (u, s)`: the sorted distinct values of column $j$ and their scores.

**The mathematics.** Take column $j$ and its $n$ non-missing values. Let $u_1<\dots<u_m$ be the distinct values and $c_1,\dots,c_m$ their counts, and let $U_l=\sum_{q\le l}c_q$ (so $U_0=0$, $U_m=n$). The $c_l$ copies of $u_l$ would occupy ranks $U_{l-1}+1,\dots,U_l$. Their mid-rank is the average of those ranks:

$$
\bar r_l=\frac{(U_{l-1}+1)+U_l}{2}=U_l-\frac{c_l-1}{2}.
$$

The score is the normal quantile at the plotting position $(\bar r_l-\tfrac12)/n$:

$$
s_l=\Phi^{-1}\!\left(\frac{\bar r_l-\tfrac12}{n}\right).
$$

The position lies in $[\tfrac{1}{2n},\,1-\tfrac{1}{2n}]$, so every score is finite.

*Code ↔ symbols:* `v = v[~np.isnan(v)]` keeps the $n$ observed values; `u, counts = np.unique(v, return_counts=True)` gives $u_l$ and $c_l$; `upper = np.cumsum(counts)` is $U_l$; `mid = upper - (counts - 1) / 2.0` is $\bar r_l$; `stats.norm.ppf((mid - 0.5) / len(v))` is $s_l$.

*Three properties worth knowing.*
1. **Only ranks matter.** If $f$ is strictly increasing, the values $f(u_l)$ have the same counts and order, so $\mathrm{NS}(f(x))=\mathrm{NS}(x)$. If $f$ is strictly decreasing (for example $-x$ or $30-x$), the order reverses, so $c_l$ moves to position $m+1-l$ and the mid-rank becomes $n+1-\bar r_l$. Then $\big(n+1-\bar r_l-\tfrac12\big)/n=1-(\bar r_l-\tfrac12)/n$, and since $\Phi^{-1}(1-p)=-\Phi^{-1}(p)$, $\mathrm{NS}(f(x))=-\mathrm{NS}(x)$. So it does not matter whether a domain is reversed by negation or by reflection (§1.7).
2. **Without ties** the scores are symmetric: rank $r$ and rank $n+1-r$ get opposite scores, so their mean is exactly 0. Their variance is $\frac1n\sum_r\Phi^{-1}\big(\frac{r-1/2}{n}\big)^2$, slightly below 1 for finite $n$, tending to 1.
3. **With ties** a block of tied values gets one score, $\Phi^{-1}$ of its average position. On the lower half, $\Phi^{-1}$ is concave, so by Jensen's inequality $\Phi^{-1}(\text{mean position})\ge$ the mean of $\Phi^{-1}$ over the block. A large block at the floor (many zeros) therefore pulls the mean of the scores above 0 and shrinks their variance well below 1. This is why the protocol defines $\varepsilon$ through the standard deviations of the pooled baseline scores on this scale (protocol §2.3), rather than assuming they are 1.

**Worked example** ([`ex_normal_scores.py`](../guide_examples/s01/ex_normal_scores.py)). Seven observed values with ties, one missing:

```python
x = np.array([[0.0], [0.0], [0.0], [1.0], [2.0], [2.0], [5.0], [np.nan]])
ns = NormalScores().fit(x)
u, sc = ns.tables[0]
```

```text
== 1.8 fit: mid-ranks and Phi^-1((r - 1/2) / n) ==
unique values u: [0. 1. 2. 5.] | counts: [3 1 2 1] | upper ranks: [3 4 6 7] | mid-ranks: [2.  4.  5.5 7. ]
p = (mid - 0.5)/n, n = 7: [0.2143 0.5    0.7143 0.9286]
scores: [-0.7916  0.      0.5659  1.4652] | by hand: [-0.7916  0.      0.5659  1.4652]
transformed sample: [-0.7916 -0.7916 -0.7916  0.      0.5659  0.5659  1.4652     nan]
mean, variance of the transformed sample: 0.0317 0.6658
```

By hand: the three zeros occupy ranks 1–3, mid-rank 2, position $(2-0.5)/7=0.2143$, score $\Phi^{-1}(0.2143)=-0.792$. The value 1 has rank 4, position $3.5/7=0.5$, score 0. The two 2s occupy ranks 5–6, mid-rank 5.5, position $5/7$, score 0.566. The 5 has rank 7, position $6.5/7$, score 1.465. The tied block at the bottom makes the mean positive (0.032) and the variance 0.67.

The same script checks property 1 on 200 made-up MoCA scores, and property 3 on a domain with 70% zeros:

```text
== reversing by -x or by 30 - x gives minus the same scores ==
max |NS(-x) + NS(x)| = 8.881784197001252e-16 | max |NS(30 - x) - NS(-x)| = 0.0

== a floor effect: 70% zeros ==
distinct scores: [-0.3853  0.628   0.8687  1.1952  1.7684] | mean 0.0688 | variance 0.5356
no ties, n = 1000, variance of the fitted sample's scores: 0.9987
```

**Design choices and pitfalls.**
- The plotting position $(r-\tfrac12)/n$ (Hazen's) is one of several. Blom's $(r-\tfrac38)/(n+\tfrac14)$ and van der Waerden's $r/(n+1)$ give slightly smaller extreme scores. With hundreds of values the choice hardly matters; what matters is that it is fixed and frozen.
- A domain with a floor (hallucinations, motor complications in early PD) collapses to one score for every floor value. That is the honest rank information, but such a domain is far from normal, which is why D26 sends a domain with more than half its scores at a bound to a latent-threshold model or out of H-P's support.
- A column with no observed value fits without error and fails only at `transform` (§1.9).

### 1.9 `NormalScores.transform` and `NormalScores.inverse` — the frozen map and its inverse

[`transforms.py:48`](../../src/pdx/transforms.py#L48), [`transforms.py:57`](../../src/pdx/transforms.py#L57)

**What it does.** `transform` maps raw values to normal scores with the fitted tables. A value seen in the fit gets exactly its table score; a value between two fitted values is interpolated linearly; a value outside the fitted range gets the score of the nearest end. `inverse` maps scores back to raw values (for reporting in clinical units) by interpolating the other way. Missing stays missing.

**Inputs → outputs.** An array whose last axis is the domain (2-D in every call in the repository) → an array of the same shape.

**The mathematics.** For column $j$ with table $(u_l,s_l)_{l=1}^m$:

$$
\mathrm{NS}(x)=\begin{cases}
s_1 & x\le u_1,\\[2pt]
s_l+\dfrac{x-u_l}{u_{l+1}-u_l}\,(s_{l+1}-s_l) & u_l\le x\le u_{l+1},\\[6pt]
s_m & x\ge u_m,
\end{cases}
$$

which is exactly `np.interp(x, u, sc)`: linear interpolation with constant extrapolation. Because $\bar r_l$ strictly increases with $l$ and $\Phi^{-1}$ is strictly increasing, $s_1<\dots<s_m$, so the map is monotone and `np.interp(z, sc, u)` inverts it on $[s_1,s_m]$, clamping outside.

"Frozen" means that the tables are computed once, on the pooled primary population, and never recomputed: `analysis_panel` fits them on the primary population's grid (all years, all phenotypes including indeterminate; §1.30) and applies them to every participant; simulated and bootstrap datasets are generated on that scale and never re-transformed (protocol §7.1).

**Worked example** ([`ex_normal_scores.py`](../guide_examples/s01/ex_normal_scores.py)), with the table of §1.8:

```python
new = np.array([[-3.0], [0.5], [1.5], [3.5], [9.0], [np.nan]])
print("normal scores:  ", np.round(ns.transform(new)[:, 0], 4))
print("back (inverse): ", np.round(ns.inverse(ns.transform(new))[:, 0], 4))
print("inverse of z = -2, 0, 2:", np.round(ns.inverse(np.array([[-2.0], [0.0], [2.0]]))[:, 0], 4))
```

```text
== 1.9 transform (frozen) and inverse ==
new values:      [-3.   0.5  1.5  3.5  9.   nan]
normal scores:   [-0.7916 -0.3958  0.283   1.0156  1.4652     nan]
back (inverse):  [0.  0.5 1.5 3.5 5.  nan]
inverse of z = -2, 0, 2: [0. 1. 5.]
```

By hand: 0.5 lies halfway between $u=0$ (score −0.792) and $u=1$ (score 0), so it gets −0.396; 1.5 lies halfway between 1 (0) and 2 (0.566), so 0.283; 3.5 lies halfway between 2 (0.566) and 5 (1.465), so 1.016. The values −3 and 9 are clamped to the end scores, and their inverses come back as 0 and 5, not −3 and 9. The end of the same script fits a table to a column with no observed value and then transforms a value:

```text
== a column with no observed value ==
transform -> ValueError: array of sample points is empty
```

**Design choices and pitfalls.**
- Clamping means a new value beyond the pooled range (possible in Part II's external data, or for a simulated score) cannot be told apart from the extreme observed value.
- Interpolating between fitted values is a convention. A value never observed between two observed ones gets a score between theirs, which keeps the map monotone.
- `out[..., j][ok] = ...` works because `out[..., j]` (basic indexing) is a view of `out`, so the masked assignment lands in `out`. It gives the same result for a 1-D vector of one visit's $d$ scores, a 2-D array and a 3-D array (checked while writing this guide).
- The tables contain every distinct value in the pooled data, so `normal_scores.npz` is participant-derived and stays in `data/` (the module docstring says so).

---

### 1.10 `RAW`, `NOT_GRID`, `ENROLLED` — where the download is, and which events and statuses count

[`ppmi.py:36`](../../src/pdx/ppmi.py#L36), [`ppmi.py:37`](../../src/pdx/ppmi.py#L37), [`ppmi.py:38`](../../src/pdx/ppmi.py#L38)

**What it does.**
- `RAW` is the default location of the download: `Path(__file__).resolve().parents[3] / "PPMI" / "raw"`. From `src/pdx/ppmi.py`, `parents[3]` is the folder that holds the repository (`paper 2`), so the raw files stay outside version control.
- `NOT_GRID` lists the event codes that may not represent a study year on the annual grid, which the module docstring calls unscheduled events: screening (`SC`), the symptomatic-therapy visit (`ST`), premature withdrawal (`PW`), `TRANS`, the unscheduled visit `U01`, and `RS1`.
- `ENROLLED` lists the enrolment statuses that count as enrolled: "Enrolled", "Withdrew", "Complete", "Withdraw Deceased", "Baseline Withdraw" and "Baseline". PPMI's PD cohort table also lists people who were screened but never enrolled; their statuses ("Screen failed", "Excluded", "Declined", "Pending" and so on) are not in the list.

**Inputs → outputs.** Constants.

**The mathematics.** Two set-membership rules: a visit is a grid candidate if $\texttt{EVENT\_ID}\notin$ `NOT_GRID` (§1.25), and a participant is enrolled if $\texttt{ENROLL\_STATUS}\in$ `ENROLLED` (§1.17). Both are exact string matches.

**Worked example** ([`ex_item_lists.py`](../guide_examples/s01/ex_item_lists.py)):

```python
print("NOT_GRID:", ppmi.NOT_GRID)
for code in ("SC", "U01", "U02", "R08", "V04", "BL"):
    print(f"  {code}: excluded from the grid = {code in ppmi.NOT_GRID}")
```

```text
== 1.10 ==
RAW = <folder holding the repository>/PPMI/raw
NOT_GRID: ('SC', 'ST', 'PW', 'TRANS', 'U01', 'RS1')
  SC: excluded from the grid = True
  U01: excluded from the grid = True
  U02: excluded from the grid = False
  R08: excluded from the grid = False
  V04: excluded from the grid = False
  BL: excluded from the grid = False
```

**Design choices and pitfalls.**
- The exclusion is by exact code. A second unscheduled visit (`U02`), if the download has one, is a grid candidate, and §1.25's example shows it winning a grid year (§1.53, note 18). A prefix rule (`startswith("U")`) would be the robust form.
- Remote visits (codes starting with `R`, §1.22) are grid candidates; they lose only to an in-person visit in the same window.

### 1.11 `TREMOR_III`, `BRADY_RIGID`, `AXIAL`, `FUNCTION_II` — the items of the motor and function domains

[`ppmi.py:40`](../../src/pdx/ppmi.py#L40), [`ppmi.py:42`](../../src/pdx/ppmi.py#L42), [`ppmi.py:44`](../../src/pdx/ppmi.py#L44), [`ppmi.py:45`](../../src/pdx/ppmi.py#L45)

**What it does.** These lists define the four domains built from MDS-UPDRS items, as in the provisional table of protocol §6.1.

| list | PPMI columns | MDS-UPDRS items | $n$ | needs | range |
|---|---|---|---|---|---|
| `TREMOR_III` | `NP3PTRMR/L`, `NP3KTRMR/L`, `NP3RTARU`, `NP3RTALU`, `NP3RTARL`, `NP3RTALL`, `NP3RTALJ`, `NP3RTCON` | 3.15 postural tremor (R, L), 3.16 kinetic tremor (R, L), 3.17 rest-tremor amplitude (4 limbs, lip/jaw), 3.18 constancy of rest tremor | 10 | 8 | 0–40 |
| `BRADY_RIGID` | `NP3RIGN`, `NP3RIGRU/LU/RL/LL`, `NP3FTAPR/L`, `NP3HMOVR/L`, `NP3PRSPR/L`, `NP3TTAPR/L`, `NP3LGAGR/L`, `NP3BRADY` | 3.3 rigidity (neck, 4 limbs), 3.4 finger tapping, 3.5 hand movements, 3.6 pronation–supination, 3.7 toe tapping, 3.8 leg agility (each R, L), 3.14 global spontaneity of movement | 16 | 13 | 0–64 |
| `AXIAL` | `NP3RISNG`, `NP3GAIT`, `NP3FRZGT`, `NP3PSTBL`, `NP3POSTR` | 3.9 arising from chair, 3.10 gait, 3.11 freezing of gait, 3.12 postural stability, 3.13 posture | 5 | 4 | 0–20 |
| `FUNCTION_II` | `NP2SPCH`, `NP2SALV`, `NP2SWAL`, `NP2EAT`, `NP2DRES`, `NP2HYGN`, `NP2HWRT`, `NP2HOBB`, `NP2TURN`, `NP2RISE` | Part II without 2.10 tremor, 2.12 walking and balance, 2.13 freezing | 10 | 8 | 0–40 |

Items 3.1 (speech) and 3.2 (facial expression) belong to no domain yet; protocol §6.1 leaves them to WP1.

**Inputs → outputs.** Constants.

**The mathematics.** Each domain score is a sum of valid items with the 80% rule of §1.16,

$$
S=\sum_{j\in\mathcal O}x_j\quad\text{if }\lvert\mathcal O\rvert\ge\lceil 0.8\,n\rceil,\ \text{else missing},
$$

so the "needs" column is $\lceil 0.8n\rceil$. The three Part III lists are disjoint (`allitems` in `_motor` has 31 entries), which is protocol §5.4's "nodes share no items". A Part III record also counts as an examination only if at least $0.8\times 31=24.8$, so 25, of these items are valid (§1.18).

**Worked example** ([`ex_item_lists.py`](../guide_examples/s01/ex_item_lists.py)):

```python
for name, items in [("TREMOR_III", ppmi.TREMOR_III), ("BRADY_RIGID", ppmi.BRADY_RIGID), ("AXIAL", ppmi.AXIAL),
                    ("FUNCTION_II", ppmi.FUNCTION_II)]:
    print(f"  {name:12s} {len(items):2d} items, needs {int(np.ceil(0.8 * len(items))):2d}, range 0-{4 * len(items)}")
```

```text
== 1.11 domain item lists and the 80% rule ==
  TREMOR_III   10 items, needs  8, range 0-40
  BRADY_RIGID  16 items, needs 13, range 0-64
  AXIAL         5 items, needs  4, range 0-20
  FUNCTION_II  10 items, needs  8, range 0-40
  Part III items in the three motor domains: 31 | shared items: 0 | an examination needs >= 24.8
  ceil(0.8 n) equals the exact integer ceiling for n = 1..199: True
```

The last line checks a floating-point worry: $0.8$ has no exact binary representation, so `np.ceil(0.8 * n)` could in principle round up one too far when $0.8n$ is an integer. It does not, for any scale length up to 199.

**Design choices and pitfalls.**
- The axial domain shares three items (3.10, 3.11, 3.12) with the Stebbins PIGD score (§1.12). That is allowed: the phenotype uses year-0 values and the domains enter the model from year 1 (protocol §4, landmark).
- Sums are not prorated: a domain scored from 8 of 10 items is the sum of 8 items (§1.16, §1.53 note 7).

### 1.12 `STEBBINS_TREMOR`, `STEBBINS_PIGD`, `TD_CUT`, `PIGD_CUT` — the phenotype rule's items and cut-offs

[`ppmi.py:47`](../../src/pdx/ppmi.py#L47)

**What it does.** These encode the published TD/PIGD rule (Stebbins et al. 2013, protocol §5.2):
- the tremor score uses 11 items, MDS-UPDRS 2.10 (`NP2TRMR`) and the ten Part III tremor items 3.15–3.18;
- the PIGD score uses 5 items, 2.12 (`NP2WALK`), 2.13 (`NP2FREZ`), 3.10 (`NP3GAIT`), 3.11 (`NP3FRZGT`) and 3.12 (`NP3PSTBL`);
- the cut-offs are 1.15 and 0.90.

**Inputs → outputs.** Constants; `stebbins` (§1.23) applies them.

**The mathematics.**

$$
R=\frac{\bar T}{\bar P},\qquad \bar T=\frac{1}{11}\sum_{j\in\text{tremor}}x_j,\qquad \bar P=\frac{1}{5}\sum_{j\in\text{PIGD}}x_j,
\qquad
\text{phenotype}=\begin{cases}\text{TD}&R\ge 1.15\\ \text{PIGD}&R\le 0.90\\ \text{indeterminate}&\text{otherwise.}\end{cases}
$$

The ratio compares means, not sums, so the two item counts do not bias it.

**Worked example** ([`ex_item_lists.py`](../guide_examples/s01/ex_item_lists.py)):

```text
== 1.12 Stebbins items ==
  tremor: 11 items; PIGD: 5 items; cut-offs 1.15 0.9
  PIGD items that are also axial-domain items: ['NP3FRZGT', 'NP3GAIT', 'NP3PSTBL']
```

**Design choices and pitfalls.** The cut-offs are used as published (protocol §5.4). `wp0.py` repeats them (`TD_CUT, PIGD_CUT = 1.15, 0.90`, line 37) for the overlap diagnostic; the two copies must be changed together.

### 1.13 `GENETIC_FLAGS`, `GDS_YES`, `GDS_NO`, `RBDSQ_ITEMS`, `RBDSQ_NEURO`, `LEVODOPA`, `DOMAINS` — the remaining vocabulary

[`ppmi.py:51`](../../src/pdx/ppmi.py#L51), [`ppmi.py:52`](../../src/pdx/ppmi.py#L52), [`ppmi.py:55`](../../src/pdx/ppmi.py#L55), [`ppmi.py:58`](../../src/pdx/ppmi.py#L58), [`ppmi.py:60`](../../src/pdx/ppmi.py#L60)

**What it does.**
- `GENETIC_FLAGS`: the six enrolment flags for a genetic cohort (LRRK2, GBA, SNCA, PRKN, PINK1, other variant). Any flag equal to 1 makes the participant "genetic PD" (§1.17).
- `GDS_YES`, `GDS_NO`: the GDS-15 items scored when answered "yes" (10 items, the depressive answer is yes) and when answered "no" (5 reverse-worded items: satisfied with life, in good spirits, happy, wonderful to be alive, full of energy).
- `RBDSQ_ITEMS`: the 12 one-point RBDSQ items, questions 1–5, the four parts of question 6, and questions 7–9. `RBDSQ_NEURO`: the nine conditions of question 10 (stroke, head trauma, parkinsonism, restless legs, narcolepsy, depression, epilepsy, inflammatory brain disease, other).
- `LEVODOPA`: a regular expression that marks a drug name as a levodopa preparation.
- `DOMAINS`: the 12 provisional domains of protocol §6.1, in the order used everywhere downstream.

**Inputs → outputs.** Constants.

**The mathematics.**
- GDS-15 (scored in §1.21): $p_j=\mathbf 1\{x_j=1\}$ for $j\in$ `GDS_YES`, $p_j=\mathbf 1\{x_j=0\}$ for $j\in$ `GDS_NO`; total $0$–$15$.
- RBDSQ: $\sum_{j=1}^{12}x_j+\mathbf 1\{\text{any condition of question 10}\}$; total $0$–$13$.
- `LEVODOPA` is a set of names: a drug is levodopa if its lower-cased name contains any of the alternatives `lev[a-z]*d[oa]p` (levodopa and its misspellings), `levocomp`, `levopar`, `nacom`, `isicom`, `sinemet`, `madopar`, `rytary`, `crexont`, `stalevo`, `duodopa`, `inbrija`, `dopicar`, `syndopa`, `prolopa`, `parcopa`, `benserazid`, `carbidopa`.

**Worked example** ([`ex_item_lists.py`](../guide_examples/s01/ex_item_lists.py)), the levodopa pattern on made-up drug names:

```python
for nm in names:
    print(f"  {nm:22s} levodopa = {bool(re.search(ppmi.LEVODOPA, nm.lower()))}")
```

```text
== 1.13 questionnaires, flags, the levodopa pattern ==
  GDS: yes-items 10 no-items 5 | RBDSQ items 12 + question 10 from 9 flags | genetic flags 6
  DOMAINS: 12 ['tremor', 'brady_rigidity', 'axial', 'function', 'independence', 'cognition', 'mood', 'autonomic', 'hallucinations', 'sleepiness', 'rbd', 'complications']
  Sinemet CR             levodopa = True
  Carbidopa/Levodopa     levodopa = True
  Levadopa               levodopa = True
  levodopa/benserazide   levodopa = True
  Madopar HBS            levodopa = True
  Stalevo                levodopa = True
  Rytary                 levodopa = True
  Duodopa                levodopa = True
  Inbrija                levodopa = True
  Carbidopa              levodopa = True
  L-Dopa                 levodopa = False
  Duopa                  levodopa = False
  Co-careldopa           levodopa = False
  Co-beneldopa           levodopa = False
  Dhivy                  levodopa = False
  Entacapone             levodopa = False
  Pramipexole            levodopa = False
  Rasagiline             levodopa = False
  Amantadine             levodopa = False
```

**Design choices and pitfalls.**
- The pattern misses some names a medication log can hold: "L-dopa", "Duopa" (the US name of Duodopa), the British approved names co-careldopa and co-beneldopa, and Dhivy. It also marks carbidopa alone (given without levodopa) as levodopa. A missed levodopa entry still counts towards the total LEDD, but not towards `ledd_levodopa`, the panel's exposure (§1.19, §1.53 note 8). Counting the distinct `LEDTRT` names that the pattern does not match (counts only) would show whether this matters.
- `DOMAINS` includes hallucinations and motor complications, which `CORE_DOMAINS` leaves out until D26 is decided (§1.29).

### 1.14 `read` — locating one PPMI table

[`ppmi.py:66`](../../src/pdx/ppmi.py#L66)

**What it does.** PPMI names each download file after the table and the download date, for example `Participant_Status_13Oct2026.csv`. `read(raw, folder, name)`:
1. globs `<raw>/<folder>/<name>_DDMonYYYY.csv` (two digits, a capitalised three-letter month, four digits);
2. falls back to a plain `<name>.csv` if nothing matches;
3. raises `FileNotFoundError` if neither exists, and otherwise reads the last file in sorted order with `low_memory=False` (types inferred from the whole file, not chunk by chunk);
4. with `usecols`, checks that every requested column exists (`KeyError` otherwise) and keeps only those.

**Inputs → outputs.** The raw folder, a sub-folder and a table name → a DataFrame of PPMI's raw columns.

**The mathematics.** Only an ordering: `sorted(paths)[-1]` is the maximum in lexicographic (character-by-character) order of the full path.

**Worked example** ([`ex_read_and_items.py`](../guide_examples/s01/ex_read_and_items.py)): three tiny synthetic "downloads" of the same table, each a one-row CSV naming itself:

```python
for tag in ["05Sep2025", "13Oct2026", "02Nov2026"]:
    pd.DataFrame({"file": [f"Participant_Status_{tag}.csv"], "PATNO": [1]}).to_csv(
        folder / f"Participant_Status_{tag}.csv", index=False)
print("read picked:  ", ppmi.read(raw, "_Subject_Characteristics", "Participant_Status").file.iloc[0])
```

```text
== 1.14 read ==
files present: ['Participant_Status_02Nov2026.csv', 'Participant_Status_05Sep2025.csv', 'Participant_Status_13Oct2026.csv']
read picked:   Participant_Status_13Oct2026.csv
usecols check -> "Participant_Status: columns not found: ['COHORT']"
plain-name fallback: {'PATNO': [7]}
```

After the common prefix the comparison reaches the day digits, "0" < "1", so the October file is the maximum although the November one is newer.

**Design choices and pitfalls.**
- With one download per folder, the intended state, the rule is harmless. With two, the "latest" file is chosen by its day of the month, not its date (§1.53, note 19).
- The plain-name fallback is used only when no dated file exists.

### 1.15 `mdate` — dates at month resolution

[`ppmi.py:80`](../../src/pdx/ppmi.py#L80)

**What it does.** PPMI releases dates as `MM/YYYY`, a de-identification measure. `mdate` puts every event on the 15th of its month: it prefixes `"15/"` and parses with `%d/%m/%Y`; anything unparseable becomes `NaT`.

**Inputs → outputs.** A Series of `MM/YYYY` strings (or missing) → a `datetime64` Series.

**The mathematics.**

$$
\mathrm{date}(\text{“MM/YYYY”})=(\text{YYYY},\ \text{MM},\ 15).
$$

If the true day is uniform within the month, the error of one date is roughly uniform on ±15 days, with standard deviation $30.44/\sqrt{12}\approx 8.8$ days; the difference of two independently placed dates has a triangular error on ±1 month, with standard deviation $30.44/\sqrt6\approx 12.4$ days. Events in the same month are tied, and the code's conventions decide their order (§1.19, §1.26).

**Worked example** ([`ex_read_and_items.py`](../guide_examples/s01/ex_read_and_items.py)):

```python
s = pd.Series(["03/2015", "3/2015", "12/1999", None, "2015-03", "13/2015"])
print(pd.DataFrame({"raw": s, "mdate": ppmi.mdate(s)}).to_string(index=False))
```

```text
== 1.15 mdate ==
    raw      mdate
03/2015 2015-03-15
 3/2015 2015-03-15
12/1999 1999-12-15
    NaN        NaT
2015-03        NaT
13/2015        NaT
```

**Design choices and pitfalls.** `errors="coerce"` is forgiving but silent: an ISO date or an impossible month becomes `NaT` without a warning. If a future download changed the date format, every date would become missing; the share of missing dates after a rebuild is the check.

### 1.16 `items_valid`, `item_sum`, `in_range` — valid item codes and the 80% rule

[`ppmi.py:85`](../../src/pdx/ppmi.py#L85), [`ppmi.py:91`](../../src/pdx/ppmi.py#L91), [`ppmi.py:95`](../../src/pdx/ppmi.py#L95)

**What it does.**
- `items_valid(df, items, lo=0, hi=4)` converts the item columns to numbers and sets every value outside $[lo,hi]$ to missing. PPMI codes such as 101 ("unable to rate") or a negative code then count as unanswered.
- `item_sum(df, items, min_frac=0.8, lo, hi)` sums the valid items of each row and returns missing when fewer than $\lceil 0.8n\rceil$ are valid.
- `in_range(s, lo, hi)` does the same check for a single total score (S&E, MoCA, item 1.2, Part IV).

**Inputs → outputs.** A DataFrame (or Series) → a DataFrame of valid items, a Series of sums, or a Series of valid totals.

**The mathematics.** For item $j$ with code $x_j$,

$$
\tilde x_j=\begin{cases}x_j & lo\le x_j\le hi\\ \text{missing}&\text{otherwise,}\end{cases}\qquad
S=\begin{cases}\sum_{j\in\mathcal O}\tilde x_j & \lvert\mathcal O\rvert\ge\lceil 0.8n\rceil\\ \text{missing}&\text{otherwise,}\end{cases}
$$

with $\mathcal O$ the items whose $\tilde x_j$ is present. In the code, `x.where((x >= lo) & (x <= hi))` is $\tilde x$ (a comparison with a missing value is false, so missing stays missing), and `.sum(axis=1, min_count=int(np.ceil(min_frac * len(items))))` is the case distinction.

The sum is not prorated. If the items $j\notin\mathcal O$ would have scored $x_j$, the code's score is $S_{full}-\sum_{j\notin\mathcal O}x_j$: biased down by the points of the missing items. The prorated alternative $\frac{n}{\lvert\mathcal O\rvert}\sum_{j\in\mathcal O}\tilde x_j$ would be unbiased if the missing items were like the others.

**Worked example** ([`ex_read_and_items.py`](../guide_examples/s01/ex_read_and_items.py)), five items scored 0–4, with codes 101, "9", −1 and blanks:

```python
df = pd.DataFrame({"a": [1, 2, 101, 1, np.nan], "b": [0, 3, 2, -1, np.nan], "c": [4, "9", 1, 2, 1],
                   "d": [2, 2, 2, np.nan, 1], "e": [1, 1, 1, 1, 1]})
print(ppmi.items_valid(df, list("abcde")).to_string())
print("item_sum, 5 items, needs ceil(0.8 * 5) = 4:", ppmi.item_sum(df, list("abcde")).tolist())
print("in_range(0, 30):", ppmi.in_range(pd.Series([26, 31, -1, "x", 30]), 0, 30).tolist())
```

```text
== 1.16 items_valid, item_sum, in_range ==
       a    b  c    d  e
0    1.0  0.0  4  2.0  1
1    2.0  3.0  9  2.0  1
2  101.0  2.0  1  2.0  1
3    1.0 -1.0  2  NaN  1
4    NaN  NaN  1  1.0  1
items_valid (0-4):
     a    b    c    d  e
0  1.0  0.0  4.0  2.0  1
1  2.0  3.0  NaN  2.0  1
2  NaN  2.0  1.0  2.0  1
3  1.0  NaN  2.0  NaN  1
4  NaN  NaN  1.0  1.0  1
item_sum, 5 items, needs ceil(0.8 * 5) = 4: [8.0, 8.0, 6.0, nan, nan]
in_range(0, 30): [26.0, nan, nan, nan, 30.0]
```

By hand: row 0 has all five items, $1+0+4+2+1=8$. Row 1 loses the "9" and sums four items, $2+3+2+1=8$. Row 2 loses the 101 and sums four, $2+1+2+1=6$. Rows 3 and 4 keep only three valid items, so they are missing.

**Design choices and pitfalls.**
- Treating out-of-range codes as missing, rather than as scores, avoids the trap of adding 101 points to a sum.
- No prorating (above). A record whose missing items are systematically the higher-scoring ones scores lower. This matters most for examinations that cannot include some items: if a remote video examination cannot rate rigidity (five items of `BRADY_RIGID`) or postural stability (one item of `AXIAL`), its bradykinesia–rigidity domain is missing (11 of 16 < 13) and its axial domain is the sum of four items without the pull test (§1.53, note 7). Protocol §6.1 leaves item scoring and missing-item rules to WP1.

### 1.17 `participants` — one row per participant of PPMI's PD cohort

[`ppmi.py:102`](../../src/pdx/ppmi.py#L102)

**What it does.** Reads four participant tables and returns one row per participant of PPMI's PD cohort (`COHORT == 1`), with:
- the enrolment date, the era ("PPMI 1.0" if enrolled before 2020, otherwise "PPMI 2.0") and genetic enrolment;
- age at enrolment and sex (`SEX`, taken as `male`), from the demographics table;
- the dates of symptom onset and of diagnosis;
- the date of death, from the conclusion-of-participation table (the earliest record with `WDDEATH == 1`), and `died`;
- `enrolled`, from the status list of §1.10;
- disease duration at enrolment and age at onset, in years.

**Inputs → outputs.** The raw folder → a DataFrame with columns `PATNO`, `enroll_date`, `era`, `genetic`, `age_enroll`, `male`, `onset_date`, `dx_date`, `death_date`, `died`, `enrolled`, `duration_years`, `age_onset`.

**The mathematics.** With dates at month resolution (§1.15),

$$
\text{duration}_i=\frac{t^{enrol}_i-t^{dx}_i}{365.25},\qquad
\text{age\_onset}_i=\text{age}^{enrol}_i-\frac{t^{enrol}_i-t^{onset}_i}{365.25},
$$

where $t^{onset}$ falls back on $t^{dx}$ when the onset date is missing (`onset_date.fillna(dx_date)`), and

$$
\text{era}_i=\begin{cases}\text{PPMI 1.0}&\text{year}(t^{enrol}_i)<2020\\ \text{PPMI 2.0}&\text{otherwise (including a missing date).}\end{cases}
$$

`genetic` is `(pdc[flags].fillna(0).astype(float) == 1).any(axis=1)`: any of the flags present in the file equals 1. `died` is a dated death record or the status "Withdraw Deceased".

**Worked example** ([`ex_participants.py`](../guide_examples/s01/ex_participants.py)), on the mock download:

```python
part = ppmi.participants(raw)
print(part[cols].loc[part.PATNO.isin([1, 2, 5, 9, 12, 13])].to_string(index=False))
```

```text
 PATNO enroll_date      era  genetic  age_enroll  male onset_date    dx_date death_date  died  enrolled  duration_years  age_onset
     1  2015-01-15 PPMI 1.0    False        62.0   1.0 2014-06-15 2014-09-15        NaT False      True        0.334018    61.4141
     2  2015-01-15 PPMI 1.0    False        62.0   0.0 2014-06-15 2014-09-15        NaT False      True        0.334018    61.4141
     5  2015-01-15 PPMI 1.0    False        62.0   1.0 2014-06-15 2014-09-15 2016-06-15  True      True        0.334018    61.4141
     9  2015-01-15 PPMI 1.0     True        62.0   1.0 2014-06-15 2014-09-15        NaT False      True        0.334018    61.4141
    12  2021-01-15 PPMI 2.0    False        62.0   0.0 2020-06-15 2020-09-15        NaT False      True        0.334018    61.4141
    13  2021-01-15 PPMI 2.0    False        60.0   NaN        NaT        NaT        NaT False     False             NaN        NaN
by hand, participant 1: duration = 122 / 365.25 = 0.334 | age at onset = 62 - 0.5859
```

By hand: 15 September 2014 to 15 January 2015 is 122 days, $122/365.25=0.334$ years; onset in June 2014 is 214 days before enrolment, $0.586$ years, so the age at onset is $62-0.586=61.414$.

The script then edits three cells: participant 4 loses its enrolment date, participant 6 gets a death record without a date, and participant 1 gets a second demographics row (`SEX` 0) at the end of the file.

```text
after the edits:
      enroll_date       era  male death_date   died  duration_years
PATNO                                                              
1      2015-01-15  PPMI 1.0   0.0        NaT  False        0.334018
4             NaT  PPMI 2.0   0.0        NaT  False             NaN
6      2015-01-15  PPMI 1.0   0.0        NaT  False        0.334018
```

**Design choices and pitfalls.**
- A missing enrolment date gives era "PPMI 2.0" (participant 4), because `NaN < 2020` is false. `build` later falls back on the baseline date for the grid's anchor, but not for the era (§1.53, note 13).
- The era is derived from the enrolment year, not from a PPMI phase field. Protocol §4 asks WP0 to reconcile the counts with PPMI's curated cohort definitions, which is the check.
- A death record without a date does not set `died` (participant 6), unless the status says "Withdraw Deceased".
- Duplicate demographics or diagnosis rows are resolved by `drop_duplicates("PATNO", keep="last")`, the last row in file order, not the latest record (participant 1's sex flips).
- `SEX` is used as `male` directly, which assumes PPMI's coding 1 = male, 0 = female.

### 1.18 `_motor` — the Part III examination of each visit and its motor state

[`ppmi.py:130`](../../src/pdx/ppmi.py#L130)

**What it does.** Reduces the Part III table (several records per visit are common: OFF and ON examinations, dosing records, repeats) to one row per participant and event, with:
- the visit's date (the earliest record's date);
- the three motor domains from the **primary examination**, the most complete OFF or untreated examination (D5);
- the ten tremor items and the three Part III PIGD items of that examination, for the phenotype;
- the three motor domains of the most complete ON examination (`tremor_on`, `brady_rigidity_on`, `axial_on`);
- `pdtrtmnt_any` (the largest `PDTRTMNT` of the visit's records) and a `motor_state` label.

**Inputs → outputs.** The raw folder and a set of `PATNO`s → a DataFrame keyed by (`PATNO`, `EVENT_ID`).

**The mathematics.** Each record gets a state by an ordered decision list (lines 136–139):

$$
\text{state}=\begin{cases}
\text{OFF} & \texttt{PDSTATE}=\text{OFF}\\
\text{ON} & \texttt{PDSTATE}=\text{ON},\ \text{or page }\texttt{NUPDRS3A}\text{ or }\texttt{NUPDR3ON}\\
\text{OFF} & \text{page }\texttt{NUPDR3OF}\\
\text{unrecorded} & \texttt{PDTRTMNT}=1\\
\text{untreated} & \text{otherwise.}
\end{cases}
$$

(A recorded `PDSTATE` wins; without one, the code reads the form pages `NUPDRS3A` and `NUPDR3ON` as ON examinations and `NUPDR3OF` as OFF.) A record is an **examination** if $n_{items}\ge 0.8\times 31=24.8$ (line 148). Among a visit's OFF or untreated examinations, the primary one maximises $(n_{items},\ \text{date})$ in lexicographic order (`sort_values(["n_items", "date"])` then `keep="last"`); the ON examination is chosen the same way. The visit's label is then a second decision list (line 162):

$$
\texttt{motor\_state}=\begin{cases}
\text{the primary examination's state (untreated or OFF)} & \text{if one exists}\\
\text{ON only} & \text{else if } \texttt{tremor\_on}\text{ is present}\\
\text{OFF recorded, no examination} & \text{else if an OFF record lacks items}\\
\text{treated, state unrecorded} & \text{else if a record is "unrecorded"}\\
\text{none (NaN; the WP0 table writes “missing”)} & \text{otherwise.}
\end{cases}
$$

How the block is built: lines 142–145 validate the 31 items once and compute the three domain sums for every record; lines 147–148 count items and set the examination flag; `first` (line 150) fixes the date; `prim` and `on` (lines 151–156) select the best examination of each kind; `flags` (lines 158–160) summarises what else was recorded; the merges (line 161) put everything on one row per visit, and `np.select` (lines 162–164) labels it.

**Worked example** ([`ex_motor.py`](../guide_examples/s01/ex_motor.py)), a hand-made Part III table (items are 1 unless stated):

```python
rows = [rec(1, "BL", "NUPDRS3", "01/2015", None, 0),                     # untreated
        rec(1, "V04", "NUPDRS3", "01/2016", "OFF", 1, value=2),          # OFF, recorded
        rec(1, "V04", "NUPDRS3A", "01/2016", None, 1),                   # page NUPDRS3A: read as ON
        rec(2, "BL", "NUPDRS3", "01/2015", None, None),                  # PDTRTMNT blank: counted as untreated
        rec(2, "V04", "NUPDRS3", "01/2016", None, 1),                    # treated, no state: unrecorded
        rec(3, "V04", "NUPDR3ON", "01/2016", None, 1, missing=ppmi.TREMOR_III[:3]),   # ON exam, 3 tremor items blank
        rec(4, "V04", "NUPDR3OF", "01/2016", None, 1, missing=allitems[:7]),           # OFF page with 24 of 31 items
        rec(5, "BL", "NUPDRS3", "01/2015", None, 0, missing=allitems[:6]),             # 25 of 31 items: an exam
        rec(5, "BL", "NUPDRS3", "02/2015", None, 0, value=0),                          # a second, complete record
        rec(6, "BL", "NUPDRS3", "01/2015", None, 0, value=101)]                        # every item 101 (unable to rate)
m = ppmi._motor(raw, {1, 2, 3, 4, 5, 6}).sort_values(["PATNO", "EVENT_ID"])
```

```text
 PATNO EVENT_ID       date  tremor  brady_rigidity  axial  tremor_on  axial_on  pdtrtmnt_any                  motor_state
     1       BL 2015-01-15    10.0            16.0    5.0        NaN       NaN           0.0                    untreated
     1      V04 2016-01-15    20.0            32.0   10.0       10.0       5.0           1.0                          OFF
     2       BL 2015-01-15    10.0            16.0    5.0        NaN       NaN           NaN                    untreated
     2      V04 2016-01-15     NaN             NaN    NaN        NaN       NaN           1.0    treated, state unrecorded
     3      V04 2016-01-15     NaN             NaN    NaN        NaN       5.0           1.0                          NaN
     4      V04 2016-01-15     NaN             NaN    NaN        NaN       NaN           1.0 OFF recorded, no examination
     5       BL 2015-01-15     0.0             0.0    0.0        NaN       NaN           0.0                    untreated
     6       BL 2015-01-15     NaN             NaN    NaN        NaN       NaN           0.0                          NaN

by hand, participant 1 V04 OFF: tremor 10 x 2 = 20 | brady 16 x 2 = 32 | axial 5 x 2 = 10
participant 5 BL keeps the complete record (31 items, all 0), not the earlier one with 25 items
```

Reading the output:
- Participant 1's V04 has an OFF and an ON examination; the OFF one is primary and the ON values are kept alongside.
- Participant 2's baseline record has `PDTRTMNT` blank and no state, so it is labelled untreated.
- Participant 3's ON examination exists (`axial_on` = 5), but three tremor items are blank, so `tremor_on` is missing and the visit gets no label at all, not "ON only".
- Participant 5's baseline takes the scores of the complete February record but the date of the January one.
- Participant 6's record is all "unable to rate": no examination, no label.

**Design choices and pitfalls.**
- "Untreated" is decided from `PDTRTMNT` alone. A record with no state, not on an ON/OFF page, and `PDTRTMNT` 0 or blank is untreated even if the participant is on medication by the LEDD log; it then enters the primary (OFF or untreated) stream. The docstring's "a treated participant's examination without a recorded state is not used" holds only when `PDTRTMNT` = 1. §1.26's example shows a participant on 300 mg/day whose later examinations are labelled untreated (§1.53, note 2).
- "ON only" is keyed on `tremor_on` being present, not on the existence of an ON examination (participant 3; §1.53, note 9).
- Choosing the most complete examination first, and the latest only as a tie-breaker, means that a repeated examination is chosen by completeness, which does not look at the scores.

### 1.19 `_ledd` — the LEDD in force on each visit date

[`ppmi.py:168`](../../src/pdx/ppmi.py#L168)

**What it does.** For each visit, sums the LEDD of every medication-log entry active on the visit date. It returns the total LEDD, the levodopa part and the drug classes in use (1 levodopa only, 2 other only, 3 both), which `build` uses to detect medication changes. The log gives a COMT inhibitor's LEDD as a multiplier of the levodopa dose (a string such as `"LD x 0.33"`) instead of a number, and the code adds that fraction of the levodopa LEDD.

**Inputs → outputs.** The raw folder and the visit table (`PATNO`, `EVENT_ID`, `date`) → one row per visit with at least one active entry: `ledd`, `ledd_levodopa`, `classes`. Visits without an active entry are absent (and filled with 0 by `visits`, §1.22).

**The mathematics.** Each log entry $e$ has a start month $s_e$, a stop month $q_e$ (missing if ongoing), an LEDD $\ell_e$ (mg/day; missing for a COMT entry), a levodopa flag and possibly a multiplier $\mu_e$. The set of entries active at visit date $t$ is

$$
\mathcal A(t)=\{e:\ s_e\le t\ \text{and}\ (q_e\ \text{missing or}\ q_e\ge t)\},
$$

with both ends inclusive at month resolution. Then

$$
L_{lev}(t)=\sum_{e\in\mathcal A(t),\ \text{levodopa}}\ell_e,\qquad
L_{tot}(t)=\sum_{e\in\mathcal A(t)}\ell_e+\Big(\max_{e\in\mathcal A(t)}\mu_e\Big)\,L_{lev}(t),\qquad
\text{classes}=\mathbf 1\{\text{any levodopa}\}+2\cdot\mathbf 1\{\text{any other}\},
$$

with an empty maximum taken as 0. *Code ↔ symbols:* line 180 is $\mathcal A(t)$ (the inner merge on `PATNO` pairs every visit with every entry of the participant); `lev_ledd` (line 181) is $\ell_e$ for levodopa entries and 0 otherwise; the `groupby` sums give $\sum\ell_e$ (the COMT rows have $\ell_e$ missing and are skipped) and $L_{lev}$; `comt_mult = ("ld_mult", "max")` is $\max\mu_e$; line 186 adds $\mu L_{lev}$. Exact duplicate entries are dropped first (line 177), on drug, dose, frequency, strength and dates.

**Worked example** ([`ex_ledd.py`](../guide_examples/s01/ex_ledd.py)), one made-up participant:

```python
log = pd.DataFrame(
    [[1, "LOG", "Sinemet", 100, 3, 100, "01/2016", "03/2017", "300"],        # 300 mg/day until 03/2017
     [1, "LOG", "Sinemet", 100, 4, 100, "03/2017", None, "400"],             # switched to 400 mg/day in 03/2017
     [1, "LOG", "Entacapone", 200, 3, 200, "06/2017", None, "LD x 0.33"],    # COMT inhibitor: a multiplier
     [1, "LOG", "Pramipexole", 0.5, 3, 0.5, "06/2017", None, "150"],
     [1, "LOG", "Pramipexole", 0.5, 3, 0.5, "06/2017", None, "150"]],        # the same entry twice
    columns=["PATNO", "EVENT_ID", "LEDTRT", "LEDDOSE", "LEDDOSFRQ", "LEDDSTRMG", "STARTDT", "STOPDT", "LEDD"])
visits = pd.DataFrame({"PATNO": 1, "EVENT_ID": ["BL", "V04", "V05", "V06", "V07"],
                       "date": ppmi.mdate(pd.Series(["12/2015", "01/2017", "03/2017", "04/2017", "07/2017"]))})
```

```text
 PATNO EVENT_ID       date  ledd  ledd_levodopa  classes
     1       BL 2015-12-15   NaN            NaN      NaN
     1      V04 2017-01-15 300.0          300.0      1.0
     1      V05 2017-03-15 700.0          700.0      1.0
     1      V06 2017-04-15 400.0          400.0      1.0
     1      V07 2017-07-15 682.0          400.0      3.0

by hand, V07: 400 (levodopa) + 150 (pramipexole, once) + 0.33 x 400 (entacapone) = 682.0
V05 (03/2017) counts both Sinemet entries, the one stopped and the one started that month: 300 + 400 = 700
```

**Design choices and pitfalls.**
- **A switch in the visit's month is counted twice.** PPMI logs a dose change by stopping one entry and starting another, often in the same month. Because both ends are inclusive, a visit in that month sees both entries: 300 + 400 = 700 mg/day at V05, where the participant took 400. Dose changes are often made at a visit, so this month is a common one. It inflates `ledd`, `ledd_levodopa` (and so the panel's `dose` and `exposure`) and `treated` at those visits, and flags a dose change (§1.53, note 1). Making the stop exclusive, or netting a stopped and a started entry of the same drug in the same month, would remove it.
- The levodopa part excludes the COMT boost (V07: 400, not 532); the total includes it.
- The multiplier uses the largest $\mu_e$ among active COMT entries, so two COMT inhibitors at once are not added.

### 1.20 `visits` (lines 191–209) — the visit table and its `add` helper

[`ppmi.py:191`](../../src/pdx/ppmi.py#L191), [`ppmi.py:196`](../../src/pdx/ppmi.py#L196)

**What it does.** Starts from the Part III table of `_motor` and joins every other form onto it, one form at a time, with the inner helper `add(folder, name, cols, fn=None, sc_as_bl=False)`. For each form `add`:
1. reads it and keeps the study's participants;
2. with `sc_as_bl`, relabels the screening record (`SC`) as baseline (`BL`) for participants who have no baseline record of that form (PPMI 1.0 gave the MoCA, GDS, SCOPA-AUT, ESS and RBDSQ at screening);
3. applies the scorer `fn` (§1.21), if any;
4. keeps the latest record per (`PATNO`, `EVENT_ID`);
5. merges it onto the visit table with an **outer** join on (`PATNO`, `EVENT_ID`), and fills the visit's date from this form if it had none.

**Inputs → outputs.** The raw folder and the set of `PATNO`s → the visit table, one row per participant and event (built up across lines 194–247 and finished in lines 248–260, §1.22).

**The mathematics.** The visit table is the outer join of the forms on the key (participant, event). The date of a visit is the first non-missing date in the order the forms are merged: Part III (from `_motor`), then Part II, Part I, Part IV, S&E, MoCA, GDS, SCOPA-AUT, ESS, RBDSQ. Within a form, "latest" means the maximum `INFODT` (`sort_values("date_").drop_duplicates(..., keep="last")`).

**Worked example.** See §1.21: the visit table of the edited mock has 35 rows over the events BL, R04, V04 and V06, and participant 7, whose MoCA was given only at screening, gets it at baseline.

**Design choices and pitfalls.**
- The outer join keeps an event that appears in any form, even without a Part III record; its motor columns are then missing, and its date comes from the first form that has it.
- `sc_as_bl` moves a screening value only when the participant has no baseline record **in that form**. It is not applied to Part II, so the phenotype, which needs Part II items at the baseline event, cannot borrow a screening Part II (§1.53, note 28).
- The relabelled screening record keeps its own date in `date_`, but `date.fillna(date_)` fills only a missing date, so the visit keeps the baseline Part III date when there is one and the screening date is not used.

### 1.21 `visits` (lines 211–247) — the five questionnaire scorers

[`ppmi.py:211`](../../src/pdx/ppmi.py#L211), [`ppmi.py:215`](../../src/pdx/ppmi.py#L215), [`ppmi.py:222`](../../src/pdx/ppmi.py#L222), [`ppmi.py:228`](../../src/pdx/ppmi.py#L228), [`ppmi.py:232`](../../src/pdx/ppmi.py#L232)

**What it does.** Five small functions, passed to `add`, compute a domain score from a form's items, each with the 80% rule:

| scorer | domain | items, valid range | needs | total |
|---|---|---|---|---|
| `part2` | `function` | `FUNCTION_II`, 0–4 | 8 of 10 | 0–40 |
| `gds` | `mood` | GDS-15, 0/1 (point for the depressive answer) | 12 of 15 | 0–15 |
| `scopa` | `autonomic` | SCOPA-AUT items 1–21, 0–3 (code 9 → 3) | 17 of 21 | 0–63 |
| `ess` | `sleepiness` | ESS items 1–8, 0–3 | 7 of 8 | 0–24 |
| `rbdsq` | `rbd` | RBDSQ items, 0/1, plus question 10 | 10 of 12 | 0–13 |

After Part II is merged, line 239 also validates its three Stebbins items (`NP2TRMR`, `NP2WALK`, `NP2FREZ`) to 0–4.

**Inputs → outputs.** Each scorer takes the form's DataFrame and returns it with one new column.

**The mathematics.**
- **GDS-15.** With $p_j=\mathbf 1\{\tilde x_j=1\}$ for the yes-items and $p_j=\mathbf 1\{\tilde x_j=0\}$ for the no-items, $\text{mood}=\sum_{j\in\mathcal O}p_j$ if $\lvert\mathcal O\rvert\ge 12$. The code first validates (codes outside 0/1 → missing), then computes $p_j$ with `raw_items[GDS_YES] == 1` and `raw_items[GDS_NO] == 0`; because a comparison with a missing value is false, it puts the missing ones back (`items[raw_items.isna().to_numpy()] = np.nan`) before `sum(min_count=12)`. The two frames have the same column order (`GDS_YES + GDS_NO`), so the positional mask lines up.
- **SCOPA-AUT.** $\tilde x_j=3$ if $x_j=9$ ("uses a catheter", scored as the most severe answer), else $x_j$ if $0\le x_j\le3$, else missing; $\text{autonomic}=\sum_{\mathcal O}\tilde x_j$ if $\lvert\mathcal O\rvert\ge17$. The sexual items 22–25 are not used.
- **RBDSQ.** $\text{rbd}=\sum_{j\in\mathcal O}x_j+\mathbf 1\{\exists\,c\in\text{question 10}:x_c=1\}$ if $\lvert\mathcal O\rvert\ge 10$ of the 12 items. A missing condition counts as "no" (`neuro.fillna(0)`); a missing item sum makes the whole score missing (NaN + 1 = NaN).

**Worked example** ([`ex_visits.py`](../guide_examples/s01/ex_visits.py)). The mock download with item values edited for participants 1, 2, 4, 7 and 12, then `visits(raw, ids)`:

```python
def gds(df):
    df.loc[at(df, 1), ["GDSDROPD", "GDSBORED", "GDSMEMRY"]] = 1         # 3 depressive "yes"
    df.loc[at(df, 1), ["GDSGSPIR", "GDSENRGY"]] = 0                     # 2 depressive "no"
    df.loc[at(df, 1), ["GDSHOME", "GDSHAPPY", "GDSALIVE"]] = np.nan     # 12 of 15 answered
    df.loc[at(df, 2), ["GDSHOME", "GDSHAPPY", "GDSALIVE", "GDSWRTLS"]] = np.nan   # 11 answered
    df.loc[at(df, 4), "GDSDROPD"] = 9                                   # a code outside 0/1
    return df


def scopa(df):
    cols = [f"SCAU{i}" for i in range(1, 22)]
    df.loc[at(df, 1), cols] = [1, 0, 2, 1, 0, 0, 1, 9, 2, 1, 0, 0, 1, 0, 0, 1, np.nan, np.nan, np.nan, np.nan, 1]
    df.loc[at(df, 2), cols[:5]] = np.nan                                # 16 of 21 answered
    return df
...
edit(N, "Geriatric_Depression_Scale__Short_Version_", gds)
edit(N, "SCOPA-AUT", scopa)
...
ids = set(range(1, 14))
v = ppmi.visits(raw, ids)
```

```text
 PATNO EVENT_ID       date  function  mood  autonomic  sleepiness  rbd  MCATOT  cognition  independence  hallucinations  complications  treated  ledd  remote motor_state
     1       BL 2015-01-15       8.0   5.0       14.0        12.0  6.0      27          3          15.0             2.0            0.0        0   0.0   False   untreated
     2       BL 2015-01-15       NaN   NaN        NaN         7.0  NaN      27          3           NaN             NaN            0.0        0   0.0   False   untreated
     4       BL 2015-01-15      10.0   0.0       11.0         8.0  0.0      27          3          10.0             0.0            0.0        0   0.0   False   untreated
     7       BL 2015-01-15      10.0   0.0       15.0         8.0  0.0      25          5          10.0             0.0            0.0        0   0.0   False   untreated
    12      V04 2022-01-15      10.0   0.0       11.0         8.0  0.0      27          3          10.0             0.0            NaN        1 300.0   False         OFF

rows: 35 | events: ['BL', 'R04', 'V04', 'V06'] | remote rows: 1
by hand, participant 1: function = 8 items x 1 = 8 | mood = 3 + 2 = 5 | autonomic = 1+0+2+1+0+0+1+3+2+1+0+0+1+0+0+1+1 = 14 | sleepiness = 12 | rbd = 5 + 1 = 6 | independence = 100 - 85 = 15
```

Reading the rows:
- Participant 1: two Part II items coded 101 leave 8 valid items, sum 8; the GDS has 12 answered items, five of them depressive; SCOPA-AUT has 17 items with the catheter code 9 scored 3; the RBDSQ has 5 positive items plus question 10.
- Participant 2: 3 blank Part II items (7 of 10), 11 GDS answers, 16 SCOPA-AUT items and 9 RBDSQ items are each below the threshold, so those domains are missing; an ESS item coded 5 is out of range and the other seven sum to 7.
- Participant 4: a GDS answer coded 9 is treated as unanswered; 14 answers remain, none depressive.
- Participant 7: no baseline MoCA, but a screening MoCA of 25, which `sc_as_bl` moves to baseline (cognition $30-25=5$).

**Design choices and pitfalls.**
- `.replace(9, 3)` acts on all 21 SCOPA-AUT items, not only the urinary items 8–13 where 9 means a catheter. This is harmless if the code appears only there.
- No sum is prorated (§1.16). The GDS of participant 1 is 5 from 12 items; prorating would give $5\times15/12=6.25$.
- Each score's threshold is the 80% rule; protocol §6.1 leaves missing-item rules to WP1.

### 1.22 `visits` (lines 248–260) — orientation, remote visits, treatment and complications

[`ppmi.py:248`](../../src/pdx/ppmi.py#L248)

**What it does.** Finishes the visit table:
- drops rows without any date;
- orients the two higher-is-better instruments and validates the single-item scores: `independence` = 100 − S&E, `cognition` = 30 − MoCA, `hallucinations` = item 1.2 (0–4), `complications` = Part IV total (0–24);
- marks a visit remote when its event code starts with "R" (PPMI 2.0's remote schedule);
- adds the LEDD (§1.19), with 0 where no entry is active;
- sets `treated`;
- sets missing motor complications to 0 before treatment.

**Inputs → outputs.** The merged visit table → the final visit table, sorted by participant and date.

**The mathematics.**

$$
\text{independence}=100-\widetilde{\text{S\&E}},\qquad \text{cognition}=30-\widetilde{\text{MoCA}},\qquad
\text{treated}=\mathbf 1\{L_{tot}>0\ \text{or}\ \texttt{pdtrtmnt\_any}=1\},
$$

$$
\text{complications}=\begin{cases}\widetilde{\text{Part IV}} & \text{treated}\\ \widetilde{\text{Part IV}}\ \text{if present, else }0 & \text{untreated,}\end{cases}
$$

where the tilde means "valid, else missing" (`in_range`). The reflection $c-x$ reverses the order, so after the normal-scores transform it is the same as negation (§1.8, property 1).

**Worked example.** In §1.21's output: participant 1's S&E of 85 gives independence 15; participant 2's S&E of 105 is out of range, so independence is missing; participant 1's blank Part IV becomes 0 because the participant is untreated, while participant 12's blank Part IV at year 1 (treated, 300 mg/day) stays missing.

**Design choices and pitfalls.**
- "Remote" is a prefix rule on the event code: any code starting with "R", including `RS1` (which `NOT_GRID` excludes anyway).
- `treated` counts any active LEDD entry, including a monoamine oxidase B (MAO-B) inhibitor or amantadine alone.
- At month resolution a drug started in the visit's month is active at that visit (§1.19), so `treated` can be 1 at a visit that came before the first dose. The LEDD of a visit in the month of a dose switch is double-counted (§1.19).

### 1.23 `stebbins` — tremor-dominant or PIGD

[`ppmi.py:265`](../../src/pdx/ppmi.py#L265)

**What it does.** Applies the Stebbins 2013 rule (§1.12) to every row of a visit table: the ratio of the mean tremor score to the mean PIGD score, from the OFF or untreated examination and the Part II questionnaire. It returns the ratio and the label TD, PIGD or indeterminate, or no label when the items are too incomplete or the examination is not OFF or untreated.

**Inputs → outputs.** A visit table with the 16 Stebbins items and `motor_state` → a DataFrame with `td_pigd_ratio` and `phenotype`, on the same index.

**The mathematics.** With $\mathcal O_T$ and $\mathcal O_P$ the valid tremor and PIGD items,

$$
\bar T=\frac{1}{\lvert\mathcal O_T\rvert}\sum_{\mathcal O_T}x_j,\qquad \bar P=\frac{1}{\lvert\mathcal O_P\rvert}\sum_{\mathcal O_P}x_j,\qquad
R=\begin{cases}\bar T/\bar P & \bar P>0\\ +\infty & \bar P=0,\ \bar T>0\\ \text{undefined} & \bar P=\bar T=0,\end{cases}
$$

usable only if $\lvert\mathcal O_T\rvert\ge 9$ of 11 and $\lvert\mathcal O_P\rvert\ge 4$ of 5 (the 80% rule, rounded up). The label is TD if $R\ge1.15$ (so $+\infty$ is TD), PIGD if $R\le0.90$, and indeterminate otherwise, including the undefined case where both means are 0. That matches Stebbins' rule for a zero PIGD score. Averaging the available items is equivalent to prorating both scores.

*Code ↔ symbols:* `tm, pm = tr.mean(axis=1), pg.mean(axis=1)` are $\bar T,\bar P$ (pandas skips missing values); line 274 computes $R$, dividing by `pm.where(pm > 0, 1.0)` to avoid a division by zero; `np.select` (line 275) applies the cut-offs (an undefined $R$ fails both comparisons and falls to "indeterminate"); line 276 removes the label where the items are incomplete or `motor_state` is not untreated or OFF; line 277 keeps the ratio where the items are complete.

**Worked example** ([`ex_stebbins.py`](../guide_examples/s01/ex_stebbins.py)), hand-made item scores:

```python
rows = {"A clear TD": row([2, 1] + [1] * 9, [0, 1, 0, 0, 0]),
        "B clear PIGD": row([0] * 11, [2, 1, 2, 1, 1]),
        "C between": row([1] * 11, [1, 1, 1, 1, 1]),
        "D on 1.15": row([3, 3, 3, 2, 2, 2, 2, 2, 2, 2, np.nan], [2, 2, 2, 2, 2]),
        "E on 0.90": row([1] * 9 + [0, np.nan], [1, 1, 1, 1, 1]),
        "F PIGD mean 0": row([1] + [0] * 10, [0] * 5),
        "G all zero": row([0] * 11, [0] * 5),
        "H 3 tremor blank": row([2] * 8 + [np.nan] * 3, [1] * 5),
        "I ON exam": row([2] * 11, [0] * 5, state="ON only")}
```

```text
                  tremor_mean  pigd_mean  td_pigd_ratio      phenotype
A clear TD             1.0909        0.2       5.454545             TD
B clear PIGD           0.0000        1.4       0.000000           PIGD
C between              1.0000        1.0       1.000000  indeterminate
D on 1.15              2.3000        2.0       1.150000             TD
E on 0.90              0.9000        1.0       0.900000           PIGD
F PIGD mean 0          0.0909        0.0            inf             TD
G all zero             0.0000        0.0            NaN  indeterminate
H 3 tremor blank       2.0000        1.0            NaN            NaN
I ON exam              2.0000        0.0            inf            NaN

exact vs floating point: 4428 item-sum combinations, 10 exactly on a cut-off, 0 disagreements
```

By hand: row A has tremor mean $12/11=1.091$ and PIGD mean $1/5=0.2$, ratio 5.45. Row D has ten valid tremor items summing to 23, mean 2.3, and PIGD mean 2.0, ratio exactly 1.15, which is TD because the rule is "≥". Row E has ten tremor items summing to 9, ratio exactly 0.90, PIGD. Row H has 8 of 11 tremor items, below 9, so no label.

The last line answers a floating-point worry: could a ratio that is exactly on a cut-off in exact arithmetic land on the wrong side in floating point? The script enumerated every possible pair of item sums for 9–11 tremor items and 4–5 PIGD items (4,428 combinations), computed the ratio exactly with fractions and with the code's floating point, and found no disagreement. Ten combinations sit exactly on a cut-off, and all ten are classed correctly.

**Design choices and pitfalls.**
- The ratio is kept for a non-OFF examination (row I) although its label is removed. At baseline the primary population is untreated, so this matters only for `phenotype_y1` and `phenotype_y2`, which use the label.
- The ratio of means is undefined for the "all zero" participant, who is indeterminate; WP0's overlap diagnostic drops both $+\infty$ and undefined ratios (§1.37).

### 1.24 `clinical_subtype` — the provisional Fereshtehnejad subtype

[`ppmi.py:280`](../../src/pdx/ppmi.py#L280)

**What it does.** Classifies baseline rows into the three subtypes of Fereshtehnejad et al. (Brain 2017; protocol §5.2, secondary axis D2): diffuse malignant, intermediate, or mild motor-predominant. It compares a motor summary and three non-motor scores with their 75th percentiles, computed in the population passed in.

**Inputs → outputs.** Baseline rows (in `build`: the primary population's year-0 grid rows) → a Series of labels, missing where any of the four scores is missing.

**The mathematics.** The scores are
- motor: $M=\text{function}+\texttt{NP2TRMR}+\texttt{NP2WALK}+\texttt{NP2FREZ}+\text{tremor}+\text{brady\_rigidity}+\text{axial}$, all seven present (`min_count=7`): all 13 Part II items and 31 of the 33 Part III items (not 3.1, 3.2);
- cognition ($30-$MoCA), RBDSQ, SCOPA-AUT.

For each score $x$, $Q_x$ is its 75th percentile over the rows where it is present, by linear interpolation: with the sorted values $x_{(0)}\le\dots\le x_{(n-1)}$ and $h=0.75(n-1)$, $Q_x=x_{(\lfloor h\rfloor)}+(h-\lfloor h\rfloor)\big(x_{(\lfloor h\rfloor+1)}-x_{(\lfloor h\rfloor)}\big)$. With $H_x=\mathbf 1\{x>Q_x\}$:

$$
\text{diffuse malignant}=\big(H_M\wedge(H_C\vee H_R\vee H_A)\big)\vee\big(H_C\wedge H_R\wedge H_A\big),\qquad
\text{mild motor-predominant}=\neg H_M\wedge\neg H_C\wedge\neg H_R\wedge\neg H_A,
$$

and intermediate otherwise. *Code ↔ symbols:* `hi = lambda x: x > x.quantile(0.75)` is $H_x$; `dm` and `mild` are the two formulas; `ok` requires all four scores.

**Worked example** ([`ex_clinical_subtype.py`](../guide_examples/s01/ex_clinical_subtype.py)), twelve made-up baseline rows (the motor summary is put in `function`; participant 112 lacks one motor component):

```text
75th percentiles: motor 25.0 | non-motor {'cognition': np.float64(4.25), 'rbd': np.float64(4.0), 'autonomic': np.float64(12.5)}
       motor  cognition  rbd  autonomic                 subtype above
PATNO                                                                
101     10.0          1    2          5  mild motor-predominant  ----
102     12.0          2    2          6  mild motor-predominant  ----
103     14.0          3    3          7  mild motor-predominant  ----
104     16.0          4    3          8  mild motor-predominant  ----
105     18.0          5    3         20            intermediate  -C-A
106     20.0          6    3         21            intermediate  -C-A
107     22.0          2    4          9  mild motor-predominant  ----
108     24.0          3    4         10  mild motor-predominant  ----
109     26.0          9    2         22       diffuse malignant  MC-A
110     30.0          1    4          4            intermediate  M---
111     34.0          2    6          5       diffuse malignant  M-R-
112      NaN          3    8          6                     NaN  --R-
rbd values at the 75th percentile ( 4.0 ): count = 3 -> they are 'not above', so they can be mild motor-predominant
```

The column `above` marks which of motor (M), cognition (C), RBD (R) and autonomic (A) exceed their percentile. Participant 105 has two non-motor scores high but not the motor score and not all three non-motor scores, so intermediate. Participant 109 has motor and cognition high: diffuse malignant. The motor percentile is computed over the 11 classifiable rows ($h=7.5$, between 24 and 26), the non-motor ones over all 12, including participant 112, who cannot be classified.

**Design choices and pitfalls.**
- "Mild" uses "at or below the 75th percentile" (`~(x > Q)`). Fereshtehnejad's mild motor-predominant group was defined by scores below it. With integer scores, ties at the percentile are common (here three RBDSQ scores equal $Q=4$), and they are classed as not high, so they can be mild (participants 107 and 108).
- Each percentile is computed on the rows where that score is present, not on the rows where all four are present.
- The docstring says the cognitive composite and cut-offs are fixed in WP1 (D2); this function is provisional (§1.53, note 22).

### 1.25 `annual_grid` — one visit per participant and year

[`ppmi.py:301`](../../src/pdx/ppmi.py#L301)

**What it does.** Chooses, for each participant and study year, the one visit that represents that year (protocol §4). The candidate visits are those within ±4 months of the enrolment anniversary that are not in `NOT_GRID`; year 0 must be the baseline visit. Among the candidates the choice follows a fixed order that does not look at the scores:
1. in person before remote;
2. closest to the anniversary;
3. motor state: untreated or OFF, then "ON only", then anything else;
4. the number of the 12 domains present (more first), only as the last tie-breaker.

**Inputs → outputs.** The visit table and each participant's anchor date (enrolment date, or the baseline date if missing) → one row per (participant, year) with the visit's columns, `anchor`, `months`, `year` and `date_years`.

**The mathematics.** For a visit at date $t$ of a participant with anchor $t_0$,

$$
m=\frac{t-t_0}{30.4375\ \text{days}},\qquad y=\operatorname{round}(m/12),\qquad \delta=\lvert m-12y\rvert,
$$

and the visit is a candidate for year $y$ if $\delta\le4$, $0\le y\le 15$, its event is not in `NOT_GRID`, and ($y>0$ or the event is `BL`). Since $\delta\le 4<6$, the nearest year is never ambiguous, so the rounding rule at a half (`np.round` rounds half to even) never decides anything. The chosen visit minimises, in lexicographic order,

$$
\big(\text{remote},\ \delta,\ \text{state rank},\ -n_{core}\big),
$$

with state rank 0 for untreated or OFF, 1 for ON only, 2 otherwise. The time kept for the engine is $\texttt{date\_years}=(t-t_0)/365.25$ days.

*Code ↔ symbols:* lines 309–311 compute $m$, $y$, $\delta$; line 312 applies the window; line 313 removes year-0 candidates other than `BL`; lines 315–316 compute the state rank and $n_{core}$; `sort_values([..., "remote", "off_by", "state_rank", "n_core"], ascending=[..., True, True, True, False])` and `drop_duplicates(keep="first")` take the lexicographic minimum.

**Worked example** ([`ex_annual_grid.py`](../guide_examples/s01/ex_annual_grid.py)), a hand-made visit table:

```text
candidates (before the filters):
 PATNO EVENT_ID       date  remote motor_state  months  year  off_by
     1       BL 2015-01-15   False   untreated   0.000   0.0   0.000
     1      V03 2015-10-15   False   untreated   8.969   1.0   3.031
     1      R04 2016-01-15    True   untreated  11.992   1.0   0.008
     2       BL 2015-01-15   False   untreated   0.000   0.0   0.000
     2      V04 2016-02-15   False     ON only  13.010   1.0   1.010
     2      U02 2016-01-15   False         OFF  11.992   1.0   0.008
     2      V06 2017-05-15   False   untreated  27.959   2.0   3.959
     3       BL 2015-06-15   False   untreated   0.000   0.0   0.000
     3      V06 2017-10-15   False   untreated  28.025   2.0   4.025
     4       SC 2014-12-15   False   untreated  -1.018  -0.0   1.018
     4      V01 2015-04-15   False   untreated   2.957   0.0   2.957
     4      V04 2016-01-15   False   untreated  11.992   1.0   0.008
     5       BL 2015-01-15   False   untreated   0.000   0.0   0.000
     5      V04 2016-01-15   False     ON only  11.992   1.0   0.008
     5      V05 2016-01-15   False   untreated  11.992   1.0   0.008
     5      V06 2016-01-15   False   untreated  11.992   1.0   0.008

the grid:
 PATNO  year EVENT_ID       date  date_years  remote motor_state  n_domains
     1     0       BL 2015-01-15    0.000000   False   untreated         12
     1     1      V03 2015-10-15    0.747433   False   untreated         12
     2     0       BL 2015-01-15    0.000000   False   untreated         12
     2     1      U02 2016-01-15    0.999316   False         OFF         12
     2     2      V06 2017-05-15    2.329911   False   untreated         12
     3     0       BL 2015-06-15    0.000000   False   untreated         12
     4     1      V04 2016-01-15    0.999316   False   untreated         12
     5     0       BL 2015-01-15    0.000000   False   untreated         12
     5     1      V06 2016-01-15    0.999316   False   untreated         10
```

Reading the grid:
- Participant 1's year 1 is the in-person visit at 9 months, not the remote visit on the anniversary: mode comes before closeness.
- Participant 2's year 1 is an unscheduled `U02` (not in `NOT_GRID`), closer than V04.
- Participants 2 and 3 each have a visit four calendar months after their second anniversary. Participant 2's (January → May, $\delta=3.96$) is kept; participant 3's (June → October, $\delta=4.03$) is not. Four calendar months are 120 to 123 days, and the limit is $4\times30.4375=121.75$ days (§1.53, note 17).
- Participant 4 has no year 0: the screening visit is excluded by `NOT_GRID`, and the 3-month V01 is not `BL`.
- Participant 5's three visits on the same date are separated by motor state (ON only loses) and then completeness (10 domains beat 6).

**Design choices and pitfalls.**
- The window is in units of 30.4375 days on dates placed on the 15th, so "±4 months" is "±121.75 days". Calendar-month arithmetic would make a visit exactly four calendar months away always eligible.
- A visit at 8 to 16 months can represent year 1 (participant 1's 9-month visit). That is the protocol's ±4 months; WP0's interval distribution shows how often it happens.
- `date_years` is measured from the anchor, so the year-1 visit sits near 1.0, and `analysis_panel` uses it as the panel's time (§1.30).

### 1.26 `build` (lines 326–356) — the study population and the exclusion cascade

[`ppmi.py:326`](../../src/pdx/ppmi.py#L326)

**What it does.** Runs the pipeline and decides who is in the primary population (protocol §4):
1. `participants()`, then `visits()` for every PD-cohort participant;
2. the grid's anchor: the enrolment date, or the earliest baseline date if it is missing;
3. `annual_grid`, then `stebbins` on every grid row;
4. per participant, from the year-0 grid row: whether there is a baseline, whether treated at baseline, the baseline phenotype and ratio; whether a year-1 grid row exists;
5. the exclusion cascade, which records the first reason that applies;
6. the clinical subtype of the primary population.

**Inputs → outputs.** The raw folder → (inside `build`) the participants table with `has_baseline`, `treated_bl`, `phenotype`, `td_pigd_ratio`, `reached_y1`, `exclusion`, `primary`, `clinical_subtype`.

**The mathematics.** The cascade is an ordered decision list; a participant gets the first reason that holds:

| order | reason | condition |
|---|---|---|
| 1 | not enrolled (screen failed, excluded, declined or pending) | status not in `ENROLLED` |
| 2 | not sporadic (genetic enrolment) | any genetic flag |
| 3 | no baseline visit | no year-0 grid row |
| 4 | treated at baseline | `treated` at the year-0 row |
| 5 | no usable baseline phenotype | no label at year 0 |
| — | primary | none of the above |

The code writes each reason with `part.loc[mask & part.exclusion.isna(), "exclusion"] = label`, so earlier reasons win; the `~part.genetic &` terms in the masks are therefore redundant but harmless. "Treated at baseline" is `treated` of §1.22: an active LEDD entry on the baseline date or `PDTRTMNT` = 1. An indeterminate phenotype is a usable phenotype, so indeterminate participants are primary (Step 1 uses them; H-P does not).

**Worked example** ([`ex_build.py`](../guide_examples/s01/ex_build.py)), `build` on the mock download:

```text
== 1.26 population and exclusions ==
 PATNO      era  enrolled  genetic  has_baseline  treated_bl phenotype  td_pigd_ratio  reached_y1                                                   exclusion  primary       clinical_subtype
     1 PPMI 1.0      True    False          True       False        TD            inf        True                                                        None     True           intermediate
     2 PPMI 1.0      True    False          True       False      PIGD            0.0        True                                                        None     True mild motor-predominant
     3 PPMI 1.0      True    False          True       False        TD            inf        True                                                        None     True mild motor-predominant
     4 PPMI 1.0      True    False          True       False      PIGD            0.0        True                                                        None     True mild motor-predominant
     5 PPMI 1.0      True    False          True       False        TD            inf        True                                                        None     True mild motor-predominant
     6 PPMI 1.0      True    False          True       False      PIGD            0.0        True                                                        None     True mild motor-predominant
     7 PPMI 1.0      True    False          True       False        TD            inf        True                                                        None     True           intermediate
     8 PPMI 1.0      True    False          True       False      PIGD            0.0        True                                                        None     True mild motor-predominant
     9 PPMI 1.0      True     True          True       False        TD            inf        True                            not sporadic (genetic enrolment)    False                    NaN
    10 PPMI 1.0      True    False          True        True      PIGD            0.0        True                                         treated at baseline    False                    NaN
    11 PPMI 1.0      True    False         False       False       NaN            NaN        True                                           no baseline visit    False                    NaN
    12 PPMI 2.0      True    False          True       False      PIGD            0.0        True                                                        None     True mild motor-predominant
    13 PPMI 2.0     False    False         False       False       NaN            NaN       False not enrolled (screen failed, excluded, declined or pending)    False                    NaN
```

The mock's TD participants have PIGD items all 0, so their ratio is $+\infty$; its PIGD participants have tremor items all 0, ratio 0. Participant 9 (genetic) has a baseline phenotype but is excluded at step 2; participant 11 reached year 1 but has no baseline. The same script prints participant 10's grid rows:

```text
participant 10 (treated at baseline) on the grid:
 year EVENT_ID motor_state  treated  ledd  classes
    0       BL         OFF        1 300.0      1.0
    1      V04   untreated        1 300.0      1.0
    2      V06   untreated        1 300.0      1.0
```

The participant is on 300 mg/day throughout, but the year-1 and year-2 examinations carry no state and `PDTRTMNT` = 0, so `_motor` labels them untreated (§1.18).

**Design choices and pitfalls.**
- At month resolution, a drug started in the baseline month is active on the baseline date, so the participant is "treated at baseline" and excluded, even if the drug was started after the examination.
- `reached_y1` is computed for everyone, including excluded participants; `analysis_panel` uses it only within the primary population.

### 1.27 `build` (lines 357–383) — the WP0 visit table: dose changes and the 24-month endpoint

[`ppmi.py:357`](../../src/pdx/ppmi.py#L357)

**What it does.** Builds the table that the WP0 audit reads (§1.37): the primary population's grid rows, with the baseline phenotype and ratio repeated on every row, the clinical subtype, the visit mode, treatment, a dose-change flag, the motor state, the 12 domains, the labels at years 1 and 2 (for the stability table) and, on the year-0 row, an indicator of Part II's provisional 24-month endpoint. `build` returns the participants, the grid, this table and the visit table.

**Inputs → outputs.** (Inside `build`.) → `{"participants", "grid", "wp0", "visits"}`; the `wp0` frame has the columns of `wp0.Columns` (§1.32) plus the domains.

**The mathematics.** On consecutive grid rows $k-1,k$ of a participant (sorted by year, whether or not a year was missed),

$$
\text{dose\_change}_k=\mathbf 1\big\{\lvert L_{tot,k}-L_{tot,k-1}\rvert>0\ \text{or}\ \text{classes}_k\ne\text{classes}_{k-1}\big\},
$$

missing on each participant's first row. The endpoint indicator, on the year-0 row, is

$$
\text{event}_i=\begin{cases}1 & \text{death within 730.5 days of enrolment}\\
\mathbf 1\{\text{independence at year 2}>20\}=\mathbf 1\{\text{S\&E}<80\} & \text{else, if S\&E is recorded at year 2}\\
\text{missing} & \text{otherwise.}\end{cases}
$$

*Code ↔ symbols:* `groupby("PATNO").ledd.shift()` and `.classes.shift()` give the previous row's values; `dead24` is the first case; `ev[...] = (se2 > 20)` the second (assigned through a boolean mask from a Series aligned on `PATNO`); `ev[dead24] = 1.0` makes death override.

**Worked example** ([`ex_build.py`](../guide_examples/s01/ex_build.py)):

```text
== 1.27 the WP0 visit table ==
columns: ['participant', 'cohort', 'year', 'date_years', 'phenotype', 'td_pigd_ratio', 'clinical_subtype', 'mode', 'treated', 'dose_change', 'motor_state', 'tremor', 'brady_rigidity', 'axial', 'function', 'independence', 'cognition', 'mood', 'autonomic', 'hallucinations', 'sleepiness', 'rbd', 'complications', 'phenotype_y1', 'phenotype_y2', 'event_24m']
participant   cohort  year  date_years phenotype      mode  treated  dose_change motor_state  event_24m phenotype_y1 phenotype_y2
          1 PPMI 1.0     0    0.000000        TD in person        0          NaN   untreated        0.0           TD           TD
          1 PPMI 1.0     1    0.999316        TD in person        0          0.0   untreated        NaN           TD           TD
          1 PPMI 1.0     2    2.001369        TD in person        0          0.0   untreated        NaN           TD           TD
          5 PPMI 1.0     0    0.000000        TD in person        0          NaN   untreated        1.0           TD          NaN
          5 PPMI 1.0     1    0.999316        TD in person        0          0.0   untreated        NaN           TD          NaN
         12 PPMI 2.0     0    0.000000      PIGD in person        0          NaN   untreated        0.0         PIGD         PIGD
         12 PPMI 2.0     1    0.999316      PIGD in person        1          1.0         OFF        NaN         PIGD         PIGD
         12 PPMI 2.0     2    1.998631      PIGD in person        1          0.0         OFF        NaN         PIGD         PIGD
```

Participant 12 starts levodopa between years 0 and 1, so year 1 has `dose_change` 1; participant 5 died 17 months after enrolment, so `event_24m` is 1 although there is no year-2 visit; participant 1 has S&E 90 at year 2, so 0.

**Design choices and pitfalls.**
- The endpoint is measured from enrolment to year 2. Protocol §12.4's provisional endpoint is "alive and independent 24 months after the landmark", with the landmark at year 2 (so the window is years 2 to 4), "among participants alive and independent at it". The code counts a different window and does not restrict the denominator (§1.53, note 10).
- `dose_change` compares the dose in force on two visit dates, so a change and a change back between visits is missed, and the double count of §1.19 makes a switch month look like a change (which it is, although its size is wrong).
- The participant identifier is the PPMI `PATNO` as a string: the table is participant-level and stays in `data/derived/`.

### 1.28 `flow_table` — participant flow by era

[`ppmi.py:386`](../../src/pdx/ppmi.py#L386)

**What it does.** Cross-tabulates every PD-cohort participant by exclusion reason (or "primary: TD", "primary: PIGD", "primary: indeterminate") and era. `scripts/build_ppmi_tables.py` passes it through `wp0.protect_2way` before writing `results/wp0/participant_flow.json`.

**Inputs → outputs.** The participants table → a count DataFrame (rows: reasons, columns: eras).

**The mathematics.** $n_{r,e}=\#\{i:\text{row label}_i=r,\ \text{era}_i=e\}$, with the row label `exclusion.fillna("primary: " + phenotype)`.

**Worked example** ([`ex_build.py`](../guide_examples/s01/ex_build.py)):

```text
== 1.28 flow_table ==
era                                                          PPMI 1.0  PPMI 2.0
exclusion                                                                      
no baseline visit                                                   1         0
not enrolled (screen failed, excluded, declined or pending)         0         1
not sporadic (genetic enrolment)                                    1         0
primary: PIGD                                                       4         1
primary: TD                                                         4         0
treated at baseline                                                 1         0
```

**Design choices and pitfalls.** The era of a participant without an enrolment date is "PPMI 2.0" (§1.17), which mostly concerns the "not enrolled" row. Protocol §4 asks for the flow "by phase, recruitment group, site, visit mode and treatment status"; this table gives reason × era only (§1.53, note 14).

### 1.29 `CORE_DOMAINS` and `TREATED_DOMAINS` — what enters H-P

[`ppmi.py:395`](../../src/pdx/ppmi.py#L395), [`ppmi.py:397`](../../src/pdx/ppmi.py#L397)

**What it does.**
- `CORE_DOMAINS`: the ten domains of the H-P panel, `DOMAINS` without hallucinations and motor complications. Both are floor-bound in early PD (motor complications are zero before treatment), and D26 decides whether they enter through a latent-threshold model or leave H-P's support.
- `TREATED_DOMAINS`: the domains that receive treatment terms in the measurement equation: the three motor domains and daily function (protocol §6.1: "for each motor domain and for daily function").

**Inputs → outputs.** Constants. `analysis_panel` writes the treated domains of its panel to `info["treated_domains"]`, and `run_hp_real.py` turns them into `spec.treat_mask`.

**The mathematics.** $\texttt{treat\_mask}_j=\mathbf 1\{\text{domain } j\in\texttt{TREATED\_DOMAINS}\}$; the treatment term $h_{ikj}$ is zero for the other domains (§2).

**Worked example** ([`ex_analysis_panel.py`](../guide_examples/s01/ex_analysis_panel.py)):

```text
== 1.29 ==
CORE_DOMAINS: ['tremor', 'brady_rigidity', 'axial', 'function', 'independence', 'cognition', 'mood', 'autonomic', 'sleepiness', 'rbd'] 
TREATED_DOMAINS: ('tremor', 'brady_rigidity', 'axial', 'function')
```

**Design choices and pitfalls.** The order of `CORE_DOMAINS` fixes the column order of the panel, the normal-scores tables, `meas_var` and the support matrix; `configs/measurement_error.json` is read by name, so it does not depend on the order.

### 1.30 `analysis_panel` — the H-P panel

[`ppmi.py:400`](../../src/pdx/ppmi.py#L400)

**What it does.** Builds the `Panel` on which H-P, the speed estimates and the localisation run (protocol §2.3, §4, §7.1):
1. fits the normal-scores transform on the **primary population's grid, all years and all phenotypes** (indeterminate included), and freezes it;
2. keeps the TD and PIGD participants of the primary population who reached the year-1 landmark, and their grid visits from year 1 on;
3. builds the centred covariates $z_i$ = (era, age at onset, sex, disease duration), with missing values set to the mean;
4. for each participant: the transformed scores, the times (`date_years`), the dose (total LEDD/100), the exposure (levodopa LEDD/100), the examination state and the grid years;
5. codes the group: PIGD = 1, TD = 0.

With `motor_on=True` (a D5 option), the motor domains of "ON only" visits are filled from their ON examination, before the transform is fitted, and those visits get `on` = 1.

**Inputs → outputs.** The participants table, the grid, the domain list and `motor_on` → `(Panel, NormalScores, info)`. `info` holds the domains, the covariate names, the treated domains, `motor_on` and the panel's `aggregate_summary` counts; it is the only aggregate.

**The mathematics.**

$$
Y_{ikj}=\mathrm{NS}_j\big(x_{ikj}\big),\qquad
t_{ik}=\frac{t^{visit}_{ik}-t^{enrol}_i}{365.25},\qquad
\mathrm{dose}_{ik}=\frac{L_{tot}(t_{ik})}{100},\qquad
\mathrm{expo}_{ik}=\frac{L_{lev}(t_{ik})}{100},
$$

$$
\mathrm{on}_{ik}=\mathbf 1\{\text{motor\_on}\ \wedge\ \text{visit } k \text{ is ON only}\},\qquad
g_i=\mathbf 1\{\text{PIGD}\},\qquad
z_{ic}=\hat z_{ic}-\bar z_c,\quad \hat z_{ic}=\begin{cases}z^{raw}_{ic}&\text{observed}\\ \bar z_c&\text{missing,}\end{cases}
$$

where $\bar z_c$ is the mean over the panel's participants. Mean imputation followed by centring sets every imputed value to exactly 0, because filling with the mean does not change the mean. The covariate order is era (1 = PPMI 2.0), age at onset, male, duration, so era is column 0, as `noise_era_col = 0` in the TOML expects.

*Code ↔ symbols:* line 418 fits $\mathrm{NS}$; line 419 selects TD/PIGD participants who reached year 1; lines 420–421 keep their rows from year 1 on and check that each starts at year 1; lines 423–427 build $z$; the loop (lines 429–435) builds $Y_i$, $t_i$, dose, exposure and `on`; line 436 codes $g_i$.

**Worked example** ([`ex_analysis_panel.py`](../guide_examples/s01/ex_analysis_panel.py)), the mock download with three edits: participant 12's year-2 examination is ON only, participant 3 has no diagnosis dates (so no age at onset or duration), and ages at enrolment differ:

```python
panel, ns, info = ppmi.analysis_panel(out["participants"], out["grid"])
...
panel_on, ns_on, _ = ppmi.analysis_panel(out["participants"], out["grid"], motor_on=True)
```

```text
== 1.30 analysis_panel (default: OFF or untreated examinations only) ==
participants: 9 | d: 10 | p: 4 | group: [0, 1, 0, 1, 0, 1, 0, 1, 1]
visits per participant: [2, 2, 2, 2, 1, 2, 2, 2, 2] | first grid year of each: [1, 1, 1, 1, 1, 1, 1, 1, 1]
t of participant 0 (years since enrolment): [0.9993 2.0014]
z columns: ['era (PPMI 2.0)', 'age at onset', 'male', 'duration']
[[-0.111  7.875  0.556  0.   ]
 [-0.111 -7.125 -0.444  0.   ]
 [-0.111 -0.     0.556  0.   ]
 [-0.111  3.875 -0.444  0.   ]
 [-0.111 -0.125  0.556  0.   ]
 [-0.111 -4.125 -0.444  0.   ]
 [-0.111 -0.125  0.556  0.   ]
 [-0.111 -0.125 -0.444  0.   ]
 [ 0.889 -0.125 -0.444  0.   ]]
column means of z: [ 0. -0.  0.  0.]
participant 12: dose [3. 3.] | exposure [3. 3.] | on [0. 0.]
participant 12, scores at year 2 (tremor, brady_rigidity, axial): [nan nan nan]
info: {'treated_domains': ['tremor', 'brady_rigidity', 'axial', 'function'], 'motor_on': False} | counts: {'0': {'participants': 4, 'visits': 7, 'transitions': 3, 'median_visits': 2.0, 'share_missing': 0.0}, '1': {'participants': 5, 'visits': 10, 'transitions': 5, 'median_visits': 2.0, 'share_missing': 0.03}}

== with motor_on=True (the D5 option) ==
participant 12: on [0. 1.] | year-2 scores (tremor, brady_rigidity, axial): [-0.5579  0.      0.5579]
tremor table, default: [ 0. 20.] | with motor_on: [ 0. 20.]
```

Reading the output:
- The era column is $1-\tfrac19=0.889$ for the one PPMI 2.0 participant and $-0.111$ for the others: era enters centred.
- Participant 3 (third row) has no age at onset; it is imputed and becomes 0 after centring (printed `-0.`). Duration is the same for every mock participant, so its centred column is 0.
- The first time of participant 0 is 0.9993, not 0: times are years since enrolment.
- By default participant 12's ON-only year-2 visit keeps missing motor domains; with `motor_on` they are filled from the ON examination and `on` becomes 1. (Here the ON values add no new tremor value, so the tremor table is unchanged.)
- `share_missing` 0.03 for PIGD is the three missing motor domains at that one visit: $3/20$ for participant 12, averaged over five participants.

**Design choices and pitfalls.**
- **Exposure is the current levodopa LEDD.** Protocol §6.1 (D24) describes the long-duration effect as an Emax function of "the exposure over the preceding weeks (a slow effect compartment … built from the medication log)". The code uses the levodopa LEDD in force on the visit date (the docstring calls it provisional), without the COMT boost (§1.53, note 6).
- **`on` is 0 at every visit unless `motor_on`.** The engine multiplies the current-dose (short-duration) term by `on` (`model.py:191`), so in the default panel `dose` never enters the likelihood, for any treated domain, including daily function, which is a questionnaire about the past week rather than an examination in a state (§1.53, note 5).
- The transform is fitted on all primary participants and years, including indeterminate participants and year 0, as protocol §7.1 asks; with `motor_on` it is fitted after the ON values are filled in, so the scale depends on that option.
- Times start near 1, not 0 (§1.53, note 15), and era is centred, so $Q_g$ in the engine is the noise at the mean era rather than in PPMI 1.0 (note 16).
- `ids` are not set, so the saved panel carries no `PATNO`.

### 1.31 `screening_retest` — test–retest reliability, ICC(2,1)

[`ppmi.py:445`](../../src/pdx/ppmi.py#L445)

**What it does.** PPMI gave MDS-UPDRS Parts II and III at screening and again at baseline, before any treatment. `screening_retest` uses these repeats to estimate the test–retest reliability of the four motor and function domains, as one input to D28 (the fixed measurement-error variances). For each primary participant with an untreated screening and baseline visit at most 120 days apart, it transforms both scores with a normal-scores transform fitted on the grid (pooled, as in §1.30), and computes, per domain with at least 50 pairs, the intraclass correlation ICC(2,1), the Pearson correlation and an implied error variance.

**Inputs → outputs.** The visit table, the participants table and the grid → a dictionary: `pairs`, `median_gap_days`, and per domain either `{"n", "icc_2_1", "pearson_r", "error_variance"}` or "fewer than 50 pairs".

**The mathematics.** Arrange the $n$ pairs as an $n\times2$ table $Y_{ij}$ (participant $i$, occasion $j\in\{1,2\}$, $k=2$ occasions). The two-way random-effects model (Shrout & Fleiss 1979) is

$$
Y_{ij}=\mu+a_i+b_j+e_{ij},\qquad a_i\sim(0,\sigma_a^2),\ b_j\sim(0,\sigma_b^2),\ e_{ij}\sim(0,\sigma_e^2).
$$

With row means $\bar Y_{i\cdot}$, column means $\bar Y_{\cdot j}$ and grand mean $\bar Y$:

$$
\text{MSR}=\frac{k\sum_i(\bar Y_{i\cdot}-\bar Y)^2}{n-1},\qquad
\text{MSC}=\frac{n\sum_j(\bar Y_{\cdot j}-\bar Y)^2}{k-1},\qquad
\text{MSE}=\frac{\sum_{ij}(Y_{ij}-\bar Y)^2-(n-1)\text{MSR}-(k-1)\text{MSC}}{(n-1)(k-1)}.
$$

Their expectations are $k\sigma_a^2+\sigma_e^2$, $n\sigma_b^2+\sigma_e^2$ and $\sigma_e^2$. Solving for the variance components and plugging them into $\sigma_a^2/(\sigma_a^2+\sigma_b^2+\sigma_e^2)$ gives the absolute-agreement, single-measure coefficient:

$$
\text{ICC}(2,1)=\frac{\text{MSR}-\text{MSE}}{\text{MSR}+(k-1)\,\text{MSE}+k(\text{MSC}-\text{MSE})/n}.
$$

*Code ↔ symbols:* with $k=2$, `msr = 2 * np.sum((Y.mean(1) - grand) ** 2) / (n - 1)`, `msc = n * np.sum((Y.mean(0) - grand) ** 2) / 1`, `mse = (np.sum((Y - grand) ** 2) - msr * (n - 1) - msc) / (n - 1)` and `icc = (msr - mse) / (msr + mse + 2 * (msc - mse) / n)`.

The code then reports `error_variance` $=1-\text{ICC}$. Under the model, $1-\text{ICC}=(\sigma_b^2+\sigma_e^2)/(\sigma_a^2+\sigma_b^2+\sigma_e^2)$: the error share of the retest sample's total variance. It equals the error variance on the pooled normal-scores scale only if that total variance is 1 on that scale. The direct estimates on that scale are $\text{MSE}$ (consistency) or $\text{MSE}+(\text{MSC}-\text{MSE})/n$ (absolute agreement), which equal $(1-\text{ICC})\times\widehat{\operatorname{Var}}(Y)$ to first order.

**Worked example** ([`ex_retest.py`](../guide_examples/s01/ex_retest.py)). 300 simulated participants with a stable true tremor score, measured twice before treatment with error (SD 1.5 points); the grid adds five more years in which scores rise by 3 points a year. The axial domain has only 40 pairs.

```python
res = ppmi.screening_retest(v, part, grid, domains=("tremor", "axial"))
...
print(f"1 - ICC = {1 - icc:.4f}   (1 - ICC) x V = {(1 - icc) * V:.4f}   MSE = {mse:.4f}   "
      f"MSE + (MSC - MSE)/n = {mse + (msc - mse) / n:.4f}")
```

```text
{'pairs': 300, 'median_gap_days': 31.0, 'tremor': {'n': 300, 'icc_2_1': 0.777681823412186, 'pearson_r': 0.7804914999120984, 'error_variance': 0.22231817658781405}, 'axial': 'fewer than 50 pairs'}
MSR 0.7005  MSC 0.2765  MSE 0.0871  ICC 0.7777
variance of the retest scores on the pooled scale: 0.3936  (the pooled grid scores: 0.9913)
1 - ICC = 0.2223   (1 - ICC) x V = 0.0875   MSE = 0.0871   MSE + (MSC - MSE)/n = 0.0877
```

The hand computation reproduces the function's ICC. The point of the last two lines: on the pooled scale, which spans five years of progression, the early (screening and baseline) scores have variance 0.39, not 1. The error variance on that scale is about 0.087, while `error_variance` reports 0.222, two and a half times larger.

**Design choices and pitfalls.**
- Pooled over the primary population, with no phenotype split, as D28 needs.
- `error_variance` $=1-\text{ICC}$ overstates the error variance on the pooled scale whenever the retest sample is narrower than the pooled population, which is the case for screening–baseline pairs (§1.53, note 4). `configs/measurement_error.json` converts literature reliabilities the same way ("error variance on the pooled normal-scores scale = 1 − reliability").
- `pairs` and `median_gap_days` are not passed through disclosure control (§1.53, note 21).
- A domain with no observed value on the grid makes the transform fail, so the function raises instead of reporting "fewer than 50 pairs" (§1.53, note 26).

---

### 1.32 `TD_CUT`, `PIGD_CUT`, `QUANTILE_MIN_N`, `SITE_BANDS`, `Columns` — the audit's constants and input schema

[`wp0.py:37`](../../src/pdx/wp0.py#L37), [`wp0.py:43`](../../src/pdx/wp0.py#L43)

**What it does.** `wp0.py` is the WP0 feasibility audit (protocol §3): it reads one analysis-ready visit table and returns only aggregate tables for the clinical panel. Its constants are:
- `TD_CUT, PIGD_CUT = 1.15, 0.90`, the Stebbins cut-offs again, for the overlap diagnostic;
- `QUANTILE_MIN_N = 50`: quantiles (and domain summaries) are reported only from at least 50 values;
- `SITE_BANDS`: site sizes are reported as counts of sites in the bands 1–4, 5–19, 20–49 and 50 or more participants, never as exact sizes.

`Columns` is a dataclass of column names, so the audit can read a table whose columns are named differently: `participant`, `cohort`, `year`, `date_years`, `phenotype`, `ratio` (`td_pigd_ratio`), `site`, `mode`, `treated`, `dose_change`, `motor_state`, the list of `domains`, `phenotype_y1`, `phenotype_y2` and `event` (`event_24m`).

**Inputs → outputs.** Constants; `Columns(domains=[...])` → a name map.

**The mathematics.** The disclosure rules of the module docstring, with $c_0$ = `min_cell`:
1. a count $n$ with $0<n<c_0$ is suppressed (primary suppression);
2. within a table, a second cell is suppressed wherever one suppressed cell could be recovered from a published total or margin (complementary suppression, §1.33–§1.34);
3. a share is shown only if its denominator is at least $c_0$;
4. quantiles need at least 50 values; site sizes are banded.

**Worked example** ([`ex_disclosure.py`](../guide_examples/s01/ex_disclosure.py)):

```text
== 1.32 constants ==
TD_CUT, PIGD_CUT: 1.15 0.9 | QUANTILE_MIN_N: 50 | SITE_BANDS: ((1, 4), (5, 19), (20, 49), (50, None))
Columns() defaults: Columns(participant='participant', cohort='cohort', year='year', date_years='date_years', phenotype='phenotype', ratio='td_pigd_ratio', site='site', mode='mode', treated='treated', dose_change='dose_change', motor_state='motor_state', domains=[], phenotype_y1='phenotype_y1', phenotype_y2='phenotype_y2', event='event_24m')
```

**Design choices and pitfalls.**
- The default threshold is 5, with the docstring's note "confirm against PPMI's guidance".
- `build` writes no `site` column into the WP0 visit table, so the audit's site summary is never produced on PPMI (§1.53, note 14).

### 1.33 `_small` and `protect_1way` — suppression in a one-way table

[`wp0.py:63`](../../src/pdx/wp0.py#L63), [`wp0.py:67`](../../src/pdx/wp0.py#L67)

**What it does.** `_small(v, k)` is true for $0<v<k$; the code's `k` is the threshold $c_0$ (`min_cell`, 5 by default). `protect_1way(counts, k)` protects a one-way table of counts (a dictionary):
1. suppresses every small cell;
2. if exactly one cell is suppressed, also suppresses the smallest other non-zero cell, so that a published total cannot give the first one back;
3. labels a primarily suppressed cell `"<5"` (with the default threshold), a complementary cell `"suppressed"`, and shows every other count exactly (zeros included).

**Inputs → outputs.** A dictionary of counts and $c_0$ → a dictionary with string keys and values that are integers or labels.

**The mathematics.** With categories $a$, counts $n_a$ and $\mathcal S_0=\{a:0<n_a<c_0\}$,

$$
\mathcal S=\begin{cases}\mathcal S_0\cup\big\{\arg\min_{a\notin\mathcal S_0,\ n_a>0}(n_a,\ a)\big\} & \lvert\mathcal S_0\rvert=1\text{ and such a cell exists,}\\ \mathcal S_0 & \text{otherwise.}\end{cases}
$$

Why one complement suffices for a one-way table: the only published constraint is the total $N=\sum_a n_a$. If one cell is hidden, $n_a=N-\sum_{a'\ne a}n_{a'}$ gives it back; with two hidden cells, the total gives only their sum. With two or more primary cells, the total already binds at least two hidden cells, so nothing is added.

**Worked example** ([`ex_disclosure.py`](../guide_examples/s01/ex_disclosure.py)):

```text
== 1.33 protect_1way ==
_small(v, 5) for v = 0..6: [False, True, True, True, True, False, False]
  {'TD': 120, 'PIGD': 64, 'indeterminate': 3} -> {'TD': 120, 'PIGD': 'suppressed', 'indeterminate': '<5'}
  {'TD': 120, 'PIGD': 4, 'indeterminate': 3} -> {'TD': 120, 'PIGD': '<5', 'indeterminate': '<5'}
  {'TD': 120, 'PIGD': 64, 'indeterminate': 0} -> {'TD': 120, 'PIGD': 64, 'indeterminate': 0}
  {'1': 2, '2': 0, '3': 0} -> {'1': '<5', '2': 0, '3': 0}
```

In the first table the small indeterminate count (3) is hidden, and PIGD (64, the smaller of the other two) is hidden with it. In the last table the small cell is the only non-zero one, so there is nothing to pair it with.

**Design choices and pitfalls.**
- Zeros are shown. That is standard, but it means that in the last table a published total of 2 gives the hidden cell back exactly; the audit publishes such totals next to its one-way tables (§1.34's example, §1.53 note 3).
- The two labels differ: `"<5"` says the cell is small, `"suppressed"` says it is a complementary cell, which by construction is at least $c_0$. §1.34 shows how this narrows what the hidden cells can be.
- Ties in the smallest count are broken by the key, through tuple comparison; keys of mixed types (a string and a number with equal counts) would raise a `TypeError`. The audit's keys are all strings or all integers.

### 1.34 `protect_2way` — complementary suppression in a cross-table

[`wp0.py:80`](../../src/pdx/wp0.py#L80)

**What it does.** Protects a cross-table that is published with its margins:
1. primary suppression of the small cells;
2. complementary suppression, repeated until no row and no column has exactly one suppressed cell among its non-zero cells: in such a row (or column), the smallest unsuppressed non-zero cell is suppressed too;
3. if a single suppressed cell remains in the whole table (its row and column had nothing to pair it with), the smallest other non-zero cell anywhere is suppressed;
4. a row or column total is shown only if none of its cells is suppressed; the grand total is always shown.

**Inputs → outputs.** A count DataFrame and $c_0$ → `{"cells": {row: {col: value}}, "row_totals": {...}, "column_totals": {...}, "total": int}`.

**The mathematics.** Let $\mathcal S$ be the set of suppressed cells. The loop (lines 88–97) iterates

$$
\text{for every row } r \text{ (then every column)}:\quad \lvert\mathcal S\cap r\rvert=1\ \Rightarrow\ \mathcal S\leftarrow\mathcal S\cup\Big\{\arg\min_{(r,c)\notin\mathcal S,\ n_{rc}>0}(n_{rc},\ c)\Big\},
$$

until a full pass changes nothing. It terminates because $\mathcal S$ only grows and the table is finite. The code runs the same loop over $T$ and its transpose `T.T`; `S.T` is a view of `S`, so marking a cell in the transposed pass marks it in `S`.

*Why the docstring's claim holds.* After step 4, every published margin is the sum of shown cells only. So the only published equation involving hidden cells is the grand total:

$$
\sum_{(r,c)\in\mathcal S}n_{rc}=N-\sum_{(r,c)\notin\mathcal S}n_{rc}=:R .
$$

One linear equation in $\lvert\mathcal S\rvert\ge2$ unknowns determines none of them, which is the docstring's "the only constraint on the suppressed cells is the grand total, which binds at least two of them".

*What linear algebra misses: the labels give bounds.* A cell labelled `"<5"` lies in $[1,c_0-1]$. A cell labelled `"suppressed"` is complementary; it was chosen among cells that were not small, so it lies in $[c_0,R]$. With $p$ primary and $q$ complementary cells, a primary cell can be at most

$$
n_c\ \le\ \min\big(c_0-1,\ R-(p-1)\cdot1-q\cdot c_0\big),
$$

the value it takes when every other hidden cell sits at its lower bound. When $R=p+q\,c_0$, that bound is 1 and every hidden cell is determined. Because step 2 picks the smallest available complement, complements are often just above $c_0$, which is exactly when the bound bites.

**Worked example** ([`ex_disclosure.py`](../guide_examples/s01/ex_disclosure.py)). A 3 × 3 table with two small cells; then, for each hidden cell, every value consistent with its label and the published total (found by enumeration):

```python
t = pd.DataFrame([[40, 3, 12], [25, 0, 9], [30, 6, 1]], index=["A", "B", "C"], columns=["x", "y", "z"])
res = wp0.protect_2way(t, 5)
...
small = pd.DataFrame([[1, 5]], index=["PPMI 2.0"], columns=["PIGD", "TD"])
res_small = wp0.protect_2way(small, 5)
print("   intervals:", intervals(res_small))
print("  ", {"total": 3, "by_cohort": wp0.protect_1way({"PPMI 1.0": 3, "PPMI 2.0": 0}, 5)})
```

```text
== 1.34 protect_2way ==
    x  y   z
A  40  3  12
B  25  0   9
C  30  6   1
  cells: {'A': {'x': 40, 'y': '<5', 'z': 'suppressed'}, 'B': {'x': 25, 'y': 0, 'z': 9}, 'C': {'x': 30, 'y': 'suppressed', 'z': '<5'}}
  row_totals: {'A': 'suppressed', 'B': 34, 'C': 'suppressed'}
  column_totals: {'x': 95, 'y': 'suppressed', 'z': 'suppressed'}
  total: 126
  what an outsider can say: the suppressed cells sum to 126 - 104 = 22

  interval left for each suppressed cell, given its label and the published total:
   {"('A', 'y') '<5'": (1, 4), "('A', 'z') 'suppressed'": (5, 15), "('C', 'y') 'suppressed'": (5, 15), "('C', 'z') '<5'": (1, 4)}

  a one-row table:
   {'cells': {'PPMI 2.0': {'PIGD': '<5', 'TD': 'suppressed'}}, 'row_totals': {'PPMI 2.0': 'suppressed'}, 'column_totals': {'PIGD': 'suppressed', 'TD': 'suppressed'}, 'total': 6}
   intervals: {"('PPMI 2.0', 'PIGD') '<5'": (1, 1), "('PPMI 2.0', 'TD') 'suppressed'": (5, 5)}

  a one-way count published next to its total (as audit does for the transitions):
   {'total': 3, 'by_cohort': {'PPMI 1.0': '<5', 'PPMI 2.0': 0}}
```

The 3 × 3 table by hand:
- Primary: (A, y) = 3 and (C, z) = 1.
- Row A has one hidden cell, so its smallest other non-zero cell, (A, z) = 12, is hidden. Row C likewise gets (C, y) = 6. Row B has none.
- Now columns y and z each hold two hidden cells, and the loop stops.
- Row B and column x have no hidden cell, so their totals (34, 95) are shown; the grand total 126 is shown.
- The hidden cells sum to $126-104=22$. Each primary cell can still be anything from 1 to 4: protected.

The one-row table is not. Its hidden cells sum to 6, one is below 5 and the other, labelled "suppressed", is at least 5, so they must be 1 and 5 ($R=p+q\,c_0=1+5$). The last line is the one-way case of §1.33: a small count that is the only non-zero cell, next to a published total of 3.

**Design choices and pitfalls.**
- Choosing the smallest complement keeps the information loss low; it is the usual heuristic.
- The distinct labels and the smallest-complement choice together can let the hidden cells be recovered, as the one-row example shows. Labelling every hidden cell alike, or requiring the complement to leave each primary cell an interval of at least $c_0-1$ values, would close this (§1.53, note 3). The test `test_audit_is_aggregate_and_suppressed_cells_cannot_be_recovered` checks the linear condition (margins hidden, never a single hidden cell), not these bounds.
- Hiding every margin that touches a hidden cell can hide most margins of a sparse table; the docstring therefore asks that margins be published from here rather than as separate tables.

### 1.35 `_share`, `_quantiles`, `_count_series` — shares, quantiles and counts

[`wp0.py:111`](../../src/pdx/wp0.py#L111), [`wp0.py:115`](../../src/pdx/wp0.py#L115), [`wp0.py:122`](../../src/pdx/wp0.py#L122)

**What it does.**
- `_share(num, den, k)` returns `num/den` if `den` ≥ $c_0$, else the label "denominator <5".
- `_quantiles(x)` drops infinite and missing values and returns the 10th, 25th, 50th, 75th and 90th percentiles if at least 50 values remain, else "fewer than 50 values". The minimum and maximum, which are single participants' values, are never reported.
- `_count_series(s)` counts the values of a Series, missing values under the key "missing", sorted by value.

**Inputs → outputs.** Numbers or Series → a float or a label; a dictionary of quantiles or a label; a dictionary of counts.

**The mathematics.** `pandas.Series.quantile` with its default linear interpolation: for sorted values $x_{(0)}\le\dots\le x_{(n-1)}$ and $h=q(n-1)$, $Q(q)=x_{(\lfloor h\rfloor)}+(h-\lfloor h\rfloor)(x_{(\lfloor h\rfloor+1)}-x_{(\lfloor h\rfloor)})$.

**Worked example** ([`ex_disclosure.py`](../guide_examples/s01/ex_disclosure.py)):

```text
== 1.35 _share, _quantiles, _count_series ==
_share(3, 40, 5) = 0.075 | _share(1, 4, 5) = denominator <5
_quantiles(49 values): fewer than 50 values
_quantiles(60 values + inf + NaN): {'0.1': 0.278, '0.25': 0.487, '0.5': 1.052, '0.75': 1.987, '0.9': 3.853}
_count_series: {'OFF': 2, 'ON only': 1, 'untreated': 1, 'missing': 1}
```

**Design choices and pitfalls.**
- A share is protected through its denominator only. With the denominator known, a small numerator can be read back: 0.075 of 40 is 3. In the audit the shares are of visits (missing scores, dose changes), not of people, which limits the risk.
- Dropping $+\infty$ before the quantiles matters for the TD/PIGD ratio (§1.37).

### 1.36 `domain_summaries` — means, standard deviations and floor/ceiling shares

[`wp0.py:127`](../../src/pdx/wp0.py#L127)

**What it does.** For each domain, at the baseline (year 0) and landmark (year 1) visits: the number of values, their mean and standard deviation, and the shares at the lowest and highest observed value. The standard deviations put $\varepsilon$ into clinical units (D12); the shares feed the floor/ceiling rule of D26 ("more than half of its scores at a bound").

**Inputs → outputs.** The visit table, the `Columns` and $c_0$ → `{"year_0": {domain: {...} or label}, "year_1": {...}}`.

**The mathematics.** For the $n$ non-missing values $x_1,\dots,x_n$ of a domain in a year (reported only if $n\ge\max(c_0,50)$):

$$
\bar x=\frac1n\sum x_i,\qquad s=\sqrt{\frac{1}{n-1}\sum(x_i-\bar x)^2},\qquad
\text{share\_at\_min}=\frac1n\sum\mathbf 1\{x_i=\min_l x_l\},\qquad \text{share\_at\_max}=\frac1n\sum\mathbf 1\{x_i=\max_l x_l\}.
$$

**Worked example** ([`ex_audit.py`](../guide_examples/s01/ex_audit.py)), a simulated visit table of 160 made-up participants, at year 0:

```text
== 1.36 domain_summaries ==
{"tremor": {"n": 152, "mean": -0.13204711668173513, "sd": 0.9880452081383972, "share_at_min": 0.006578947368421052, "share_at_max": 0.006578947368421052}, "hallucinations": {"n": 160, "mean": 0.14375, "sd": 0.35193758380298307, "share_at_min": 0.85625, "share_at_max": 0.14375}}
```

The simulated hallucinations domain is 0 for 86% of the baseline visits, so it would meet D26's rule.

**Design choices and pitfalls.**
- The bounds are the observed minimum and maximum, not the instrument's floor and ceiling. A domain where nobody scores 0 has its "floor" at the lowest observed score. For D26 the instrument's bounds are the relevant ones; for the floor-bound domains of early PD (hallucinations, complications) the two coincide.
- The summaries are of the raw oriented scores in the WP0 table, not of the normal scores. The protocol's $\varepsilon$ uses the standard deviations "of the pooled baseline scores on the normal-scores scale" (protocol §2.3); these raw SDs translate it into clinical units.

### 1.37 `audit` — the WP0 feasibility tables

[`wp0.py:148`](../../src/pdx/wp0.py#L148)

**What it does.** Produces every aggregate table WP0 lists in protocol §3, from one visit table:

| key | content | protection |
|---|---|---|
| `participant_flow` | cohort × phenotype at baseline, with margins | `protect_2way` |
| `landmark` | phenotype × reached year 1 | `protect_2way` |
| `transitions_after_landmark` | consecutive-year transitions from year 1: total, by phenotype, by cohort; the count across a missed year | `protect_1way` (the last count is not protected) |
| `intervals` | share of intervals within 12 ± 3 months; quantiles; share with a dose change; the Step 1 interval rule | shares, quantiles |
| `visits_per_participant` | distribution of the number of visits from year 1 | `protect_1way` |
| `mode`, `motor_state` | counts over all rows | `protect_1way` |
| `missing_share`, `missing_share_by_motor_state` | per domain: overall, by phenotype, by mode; by motor state | shares |
| `domain_summaries` | §1.36 | $n\ge 50$ |
| `sites` | number of sites; sites per size band | bands |
| `overlap` | count of ratios near a cut-off; quantiles of the ratio | `"<5"`; quantiles |
| `phenotype_stability` | phenotype at year 0 × at year 1, and × at year 2 | `protect_2way` |
| `part2_events` | events and participants at risk for the 24-month endpoint | `"<5"` for events |

**Inputs → outputs.** The visit table (in practice `wp0_visits.parquet` from §1.27), the `Columns`, $c_0$ and the overlap band (0.10) → a nested dictionary of aggregates.

**The mathematics.**
- *Transitions.* Within a participant, sorted by grid year from year 1 on, $\text{step}_k=a_k-a_{k-1}$ and $\text{gap}_k=t_k-t_{k-1}$ (`date_years`). The transitions are the rows with $\text{step}_k=1$; "across a missed year" counts $\text{step}_k>1$.
- *The interval rule of protocol §6.3 (D13).*

$$
\text{share}=\frac{\#\{k:\ \text{step}_k=1,\ 0.75\le\text{gap}_k\le1.25\}}{\#\{k:\ \text{step}_k=1\}},\qquad \texttt{step1\_annual\_tables\_ok}=\mathbf 1\{\text{share}\ge0.90\}.
$$

- *Dose changes.* $\sum_{k:\text{step}_k=1}\text{dose\_change}_k\ /\ \#\{k:\text{step}_k=1\}$.
- *Overlap (protocol §5.4).* Over the finite baseline ratios, $\#\{i:\lvert R_i-1.15\rvert\le b\ \text{or}\ \lvert R_i-0.90\rvert\le b\}$ with $b=0.10$, that is $R_i\in[0.80,1.00]\cup[1.05,1.25]$.
- *Sites.* For site sizes $n_s$ (baseline participants), the number of sites with $lo\le n_s\le hi$ for each band.

**Worked example** ([`ex_audit.py`](../guide_examples/s01/ex_audit.py)). The simulated table has 160 made-up participants in two cohorts and 24 sites; every ninth participant never reaches year 1, every seventh misses year 2, intervals vary by ±0.3 years, PPMI 2.0's year-2 visits are remote, and every thirteenth participant has an infinite ratio:

```python
cols = wp0.Columns(domains=["tremor", "hallucinations"])
rep = wp0.audit(df, cols, min_cell=5)
for key in ("participant_flow", "landmark", "transitions_after_landmark", "intervals", "visits_per_participant",
            "mode", "motor_state", "sites", "overlap", "phenotype_stability", "part2_events"):
    print(f"{key}: {json.dumps(rep[key])}")
```

```text
== 1.37 audit ==
participant_flow: {"cells": {"PPMI 1.0": {"PIGD": 37, "TD": 75, "indeterminate": 8}, "PPMI 2.0": {"PIGD": 11, "TD": 21, "indeterminate": 8}}, "row_totals": {"PPMI 1.0": 120, "PPMI 2.0": 40}, "column_totals": {"PIGD": 48, "TD": 96, "indeterminate": 16}, "total": 160}
landmark: {"cells": {"PIGD": {"not reached": "suppressed", "reached year 1": "suppressed"}, "TD": {"not reached": 0, "reached year 1": 96}, "indeterminate": {"not reached": "<5", "reached year 1": "suppressed"}}, "row_totals": {"PIGD": "suppressed", "TD": 96, "indeterminate": "suppressed"}, "column_totals": {"not reached": "suppressed", "reached year 1": "suppressed"}, "total": 160}
transitions_after_landmark: {"total": 244, "by_phenotype": {"PIGD": 54, "TD": 166, "indeterminate": 24}, "by_cohort": {"PPMI 1.0": 184, "PPMI 2.0": 60}, "across_a_missed_year": 20}
intervals: {"share_within_12_pm_3_months": 0.6639344262295082, "quantiles_years": {"0.1": 0.6699181347552633, "0.25": 0.8420753546991833, "0.5": 0.9908290764150431, "0.75": 1.178706172122495, "0.9": 1.3375207292597853}, "share_with_dose_change": 0.4180327868852459, "step1_annual_tables_ok": false}
visits_per_participant: {"2": 20, "3": 122}
mode: {"in person": 536, "remote": 30}
motor_state: {"OFF": 379, "ON only": 97, "missing": 90}
sites: {"n_sites": 24, "sites_by_participants": {"1-4": 0, "5-19": 24, "20-49": 0, "50+": 0}}
overlap: {"near_cutoffs": 23, "band": 0.1, "ratio_quantiles": {"0.1": 0.30351642590228056, "0.25": 0.8478700843454962, "0.5": 1.6215735042430037, "0.75": 2.7629530812885683, "0.9": 3.456035944394445}}
phenotype_stability: {"phenotype_y1": {"cells": {"PIGD": {"PIGD": 37, "TD": 0, "indeterminate": 11}, "TD": {"PIGD": 0, "TD": 71, "indeterminate": 25}, "indeterminate": {"PIGD": 0, "TD": 0, "indeterminate": 16}}, "row_totals": {"PIGD": 48, "TD": 96, "indeterminate": 16}, "column_totals": {"PIGD": 37, "TD": 71, "indeterminate": 52}, "total": 160}, "phenotype_y2": {"cells": {}, "row_totals": {}, "column_totals": {}, "total": 0}}
part2_events: {"events": "<5", "at_risk": 160}
missing_share tremor: {"overall": 0.06007067137809187, "by_phenotype": {"PIGD": 0.04316546762589928, "TD": 0.07277628032345014, "indeterminate": 0.017857142857142856}, "by_mode": {"in person": 0.055970149253731345, "remote": 0.13333333333333333}}
missing_share_by_motor_state tremor: {"OFF": 0.05013192612137203, "ON only": 0.05154639175257732, "missing": 0.1111111111111111}
an identifier anywhere in the output: False
```

Reading it:
- In the landmark table, two indeterminate participants never reached year 1. That cell is hidden (`"<5"`), its row partner (14) and two cells of the PIGD row (16 and 32) are hidden to cover it, and the TD row, which has no hidden cell, keeps its margin. The four hidden cells sum to $160-96=64$, so the small cell can still be anything from 1 to 4.
- Only 66% of the simulated intervals lie within 12 ± 3 months, so `step1_annual_tables_ok` is false and Step 1 would add the interval to its tables (protocol §6.3).
- The year-2 stability table is empty because the simulated table has no year-2 labels; `pd.crosstab` drops missing labels, so the table covers only participants labelled at that year.
- No identifier reaches the output.

**Design choices and pitfalls.**
- The overlap quantiles are over finite ratios only; TD participants with a zero PIGD score ($R=+\infty$) and the all-zero participants (undefined $R$) are left out, so the upper quantiles describe the finite part of the distribution.
- `across_a_missed_year`, `total`, `n_sites` and `at_risk` are exact counts (§1.53, note 21).
- Missing data by site, which protocol §3 lists, is not computed (§1.53, note 14).
- The 12 ± 3 months window is in years of 365.25 days on dates placed on the 15th; an interval of exactly 9 or 15 calendar months can fall on either side of it (§1.53, note 17).
- The name `participant_flow` is used for two different tables: this cohort × phenotype table of the primary population, and the exclusion × era table that `scripts/build_ppmi_tables.py` writes to `results/wp0/participant_flow.json` (§1.49).

---

### 1.38 `DISPLAY`, `LABELS`, `ARROW`, `MINUS`, `MIN_RATERS` — the consensus module's vocabulary

[`wp1.py:22`](../../src/pdx/wp1.py#L22)

**What it does.** `wp1.py` turns the clinical panel's questionnaire answers into the expert map (protocol §6.4). It uses only the standard library, so it also runs under the system Python that has `python-docx` for reading the returned Word files. Its constants:
- `DISPLAY`: the code's domain names and the questionnaire's labels, for the 12 domains and "treatment" (13 entries); `LABELS` is the reverse map, from lower-cased label to code name;
- `ARROW`: the link separators `→`, `->`, `=>` and `⟶`, with optional spaces;
- `MINUS`: the characters read as a minus sign, hyphen `-`, minus `−` and en dash `–`;
- `MIN_RATERS = 5`: with fewer raters the map is described as investigator-led (D25).

**Inputs → outputs.** Constants.

**The mathematics.** None; `DISPLAY` and `ppmi.DOMAINS` must name the same domains (a test checks `set(wp1.DISPLAY) == set(ppmi.DOMAINS) | {"treatment"}`).

**Worked example** ([`ex_wp1.py`](../guide_examples/s01/ex_wp1.py)):

```text
== 1.38 constants ==
MIN_RATERS: 5 | domains: 13 | arrows split: [['A', 'B'], ['A', 'B'], ['A', 'B']]
```

**Design choices and pitfalls.** Reading the en dash as a minus sign makes `"2 –"` an improving link, as intended, but it also makes a range typed as `"1–2"` an improving link of strength 1 (§1.40).

### 1.39 `domain` and `parse_link` — reading a link label

[`wp1.py:34`](../../src/pdx/wp1.py#L34), [`wp1.py:45`](../../src/pdx/wp1.py#L45)

**What it does.** `domain(label)` turns a questionnaire label into the code's domain name: it removes asterisks and anything in parentheses, collapses spaces, lower-cases, and looks the result up among the labels, then among the code names; otherwise it raises `ValueError`. `parse_link(text)` splits a cell such as `"Tremor → Daily function"` at the arrow and returns `("tremor", "function")`, or `None` when the cell is not a link (a section heading, a blank, a dangling arrow, two arrows).

**Inputs → outputs.** A string → a domain name, or a `(from, to)` tuple or `None`.

**The mathematics.** A normalisation $\nu(\text{label})=\text{lower}(\text{collapse}(\text{strip}(\text{label}\setminus\{*,\ (\dots)\})))$ followed by a lookup. A link is accepted only if splitting at the arrow gives exactly two non-empty parts.

**Worked example** ([`ex_wp1.py`](../guide_examples/s01/ex_wp1.py)):

```text
== 1.39 domain and parse_link ==
  domain('Treatment (a higher dose)') = 'treatment'
  domain('Axial and gait*') = 'axial'
  domain('  daily   FUNCTION ') = 'function'
  domain('rbd') = 'rbd'
  parse_link('Tremor → Daily function') = ('tremor', 'function')
  parse_link('**Motor signs and daily life**') = None
  parse_link('Mood -> ') = None
  parse_link('Tremor → Daily → Mood') = None
  parse_link('Fatigue → Mood') -> ValueError: unknown domain 'Fatigue'
```

**Design choices and pitfalls.** A well-formed link with an unknown label raises rather than being skipped, so one mistyped label in a returned questionnaire stops the whole read. That is deliberate (it prevents a link from silently disappearing), but a rater's free-text addition such as "Fatigue → Mood" in Section E1b must be mapped by hand before the file can be read.

### 1.40 `answer` — one rater's answer to one link

[`wp1.py:53`](../../src/pdx/wp1.py#L53)

**What it does.** Reads one rater's Keep?, Strength and Confidence cells leniently ("Y", "yes", "2-", "2 −", "?", blanks) and classes the answer as "in" (kept with strength 1 or 2), "out" (not kept, or strength 0), "unsure" (a "?" anywhere, or kept without a strength) or "missing" (no answer). It also returns the strength, the sign (−1 for an effect that improves the target, only for "in") and the confidence (H, M or L).

**Inputs → outputs.** Three strings → `{"verdict", "strength", "sign", "confidence"}`.

**The mathematics.** With $\kappa$ the first character of Keep? (upper case), $\ell$ the first digit 0, 1 or 2 in Strength (none if absent) and $\sigma=-1$ if Strength contains a minus-like character:

$$
\text{verdict}=\begin{cases}
\text{out} & \kappa=\text{N},\ \text{or}\ (\kappa\in\{\text{blank},\text{Y}\}\ \text{and}\ \ell=0)\\
\text{unsure} & \text{else if } \kappa=\text{?}\ \text{or Strength contains “?”}\\
\text{in} & \text{else if } \ell\in\{1,2\}\\
\text{unsure} & \text{else if } \kappa=\text{Y}\\
\text{missing} & \text{otherwise.}
\end{cases}
$$

The order matters: it is a decision list, evaluated top to bottom. Protocol §6.4's codes are "2 strong, 1 weak, 0 none, ? unsure, with a minus sign for an effect that improves Y".

**Worked example** ([`ex_wp1.py`](../guide_examples/s01/ex_wp1.py)):

```text
== 1.40 answer ==
  keep='Y'    strength='2'     confidence='H'       -> {'verdict': 'in', 'strength': 2, 'sign': 1, 'confidence': 'H'}
  keep='yes'  strength='1 −'   confidence='medium'  -> {'verdict': 'in', 'strength': 1, 'sign': -1, 'confidence': 'M'}
  keep='N'    strength='2'     confidence=''        -> {'verdict': 'out', 'strength': 2, 'sign': None, 'confidence': None}
  keep='Y'    strength='0'     confidence=''        -> {'verdict': 'out', 'strength': 0, 'sign': None, 'confidence': None}
  keep=''     strength='0'     confidence=''        -> {'verdict': 'out', 'strength': 0, 'sign': None, 'confidence': None}
  keep='?'    strength='1'     confidence='L'       -> {'verdict': 'unsure', 'strength': 1, 'sign': None, 'confidence': 'L'}
  keep='Y'    strength='2?'    confidence=''        -> {'verdict': 'unsure', 'strength': 2, 'sign': None, 'confidence': None}
  keep='Y'    strength='0?'    confidence=''        -> {'verdict': 'out', 'strength': 0, 'sign': None, 'confidence': None}
  keep='Y'    strength=''      confidence=''        -> {'verdict': 'unsure', 'strength': None, 'sign': None, 'confidence': None}
  keep=''     strength='2'     confidence=''        -> {'verdict': 'in', 'strength': 2, 'sign': 1, 'confidence': None}
  keep=''     strength=''      confidence=''        -> {'verdict': 'missing', 'strength': None, 'sign': None, 'confidence': None}
  keep='Y'    strength='3'     confidence='H'       -> {'verdict': 'unsure', 'strength': None, 'sign': None, 'confidence': 'H'}
  keep='Y'    strength='1-2'   confidence='H'       -> {'verdict': 'in', 'strength': 1, 'sign': -1, 'confidence': 'H'}
  keep='Y'    strength='1–2'   confidence='H'       -> {'verdict': 'in', 'strength': 1, 'sign': -1, 'confidence': 'H'}
```

Reading the edge cases:
- "N" wins over a strength of 2 (not kept is not kept).
- "0?" is out, because the "out" rule comes before the "?" rule; "2?" is unsure.
- A blank Keep? with a strength counts as kept, which is how Section E1b (added links, no Keep? column) is read.
- A strength of "3", outside the codes, is ignored, and "Y" without a valid strength is unsure.
- A range typed as "1-2" or "1–2" is read as strength 1 with a minus sign: an improving link.

**Design choices and pitfalls.** Leniency avoids losing answers to formatting, but the last two lines show where it misreads (§1.53, note 12). Printing the answers that contain a digit outside 0–2 or two digits, for the investigator to check, would catch both.

### 1.41 `candidates` — the questionnaire's candidate links

[`wp1.py:75`](../../src/pdx/wp1.py#L75)

**What it does.** Reads the numbered candidate links of Section E1 from the questionnaire's markdown: every table row whose first cell is a plain number and whose second cell is a link.

**Inputs → outputs.** The path of `docs/clinical_expert_questionnaire.md` → a list of `(number, from, to)`.

**The mathematics.** A filter over table rows: $\{(n,\text{from},\text{to}):\ \text{cell}_1\ \text{is all digits},\ \text{parse\_link}(\text{cell}_2)\ne\text{None}\}$.

**Worked example** ([`ex_wp1.py`](../guide_examples/s01/ex_wp1.py)):

```text
== 1.41 candidates ==
   31 candidate links; first (1, 'tremor', 'function') ; last (31, 'treatment', 'sleepiness')
```

**Design choices and pitfalls.** Any numbered table row with a link in its second cell counts, wherever it is in the document. Section F's rows are numbered "F1.1" and so are not read; the test checks that exactly the numbers 1 to 31 are found.

### 1.42 `read_table`, `read_csv`, `read_docx`, `read_rater` — reading a returned questionnaire

[`wp1.py:87`](../../src/pdx/wp1.py#L87), [`wp1.py:114`](../../src/pdx/wp1.py#L114), [`wp1.py:121`](../../src/pdx/wp1.py#L121), [`wp1.py:130`](../../src/pdx/wp1.py#L130)

**What it does.** `read_table(rows)` reads one table (a list of rows of cell texts) whose first row names its columns. It finds the columns whose header starts with "link", "keep", "strength" and "confidence"; a table without link, strength and confidence columns gives nothing (Section F, Section E2). For each row with a link it calls `answer`. If the table has no Keep? column (Section E1b, links a rater adds), a non-blank strength counts as kept and the answer is marked `added`. `read_csv` reads a CSV answer sheet with the same columns; `read_docx` reads every table of a Word file (with `python-docx`) and merges them; `read_rater` picks one by the file's extension.

**Inputs → outputs.** Rows, or a file path → `{(from, to): answer}`.

**The mathematics.** None beyond the column lookup `next(i for i, c in enumerate(low) if c.startswith(name))`: the first header that starts with each name.

**Worked example** ([`ex_wp1.py`](../guide_examples/s01/ex_wp1.py)):

```python
e1 = [["#", "Link (this year → next year)", "Keep?", "Strength", "Confidence"],
      ["", "**Motor signs and daily life**", "", "", ""],
      ["1", "Tremor → Daily function", "Y", "2", "H"],
      ["2", "Mood → Cognition", "Y", "?", "L"]]
e1b = [["Link (this year → next year)", "Strength", "Confidence"], ["RBD symptoms → Hallucinations", "1", "M"]]
f = [["#", "Link (this year → next year)", "Expected (P/T/0/?)", "Plausible (1–5)"], ["F1.1", "Axial and gait → Independence", "P", "4"]]
```

```text
== 1.42 read_table ==
  E1:  {('tremor', 'function'): {'verdict': 'in', 'strength': 2, 'sign': 1, 'confidence': 'H', 'added': False}, ('mood', 'cognition'): {'verdict': 'unsure', 'strength': None, 'sign': None, 'confidence': 'L', 'added': False}}
  E1b: {('rbd', 'hallucinations'): {'verdict': 'in', 'strength': 1, 'sign': 1, 'confidence': 'M', 'added': True}}
  F:   {}
```

**Design choices and pitfalls.**
- `read_docx` merges the tables with `dict.update`, so a link that appears in two rating tables of one file (say, a rater "adds" in E1b a link already in E1) keeps the later answer.
- `read_docx` imports `python-docx` only when called, so the module itself has no third-party dependency.

### 1.43 `tally` — anonymised counts for one link

[`wp1.py:134`](../../src/pdx/wp1.py#L134)

**What it does.** Counts the raters' answers to one link: verdicts (in, out, unsure, missing), strengths among the "in" answers (strong = 2, weak = 1), improving signs among the "in" answers, and confidences.

**Inputs → outputs.** A list of answers → a dictionary of counts, with no rater identity.

**The mathematics.** $\#\text{in}=\sum_r\mathbf 1\{v_r=\text{in}\}$, and likewise for the others; $\#\text{improving}=\sum_r\mathbf 1\{v_r=\text{in},\ \sigma_r=-1\}$.

**Worked example.** See §1.44's output: the link tremor → function has 5 "in" answers, 4 strong and 1 weak, all with confidence H.

**Design choices and pitfalls.** Confidence is counted over every answer that has one, not only the "in" answers.

### 1.44 `round1` — unanimity

[`wp1.py:145`](../../src/pdx/wp1.py#L145)

**What it does.** Applies the round-1 rule of protocol §6.4 to every link: the listed candidates, plus every link any rater answered (including the links a rater added). A link enters the map when every rater's answer is "in", stays out when every rater's answer is "out", and returns in round 2 otherwise. A link a rater added has no answer from the other raters (they count as "missing"), so it always returns.

**Inputs → outputs.** `raters` (rater code → answers) and the listed links → `{"raters": n, "links": {(from, to): {"verdict", "tally", "added"}}}`.

**The mathematics.** With $n$ raters and answers $v_1,\dots,v_n$ (missing if a rater did not answer),

$$
\text{verdict}=\begin{cases}\text{in}&\#\text{in}=n\\ \text{out}&\#\text{out}=n\\ \text{round 2}&\text{otherwise.}\end{cases}
$$

**Worked example** ([`ex_wp1.py`](../guide_examples/s01/ex_wp1.py)), five made-up raters:

```python
raters = {f"r{r}": {("tremor", "function"): A("Y", "2" if r else "1", "H"),
                    ("axial", "independence"): A("N" if r < 4 else "Y", "" if r < 4 else "0"),
                    ("mood", "cognition"): A("Y", "1" if r < 3 else "?")} for r in range(5)}
raters["r0"][("rbd", "hallucinations")] = {**A("", "1", "M"), "added": True}
r1 = wp1.round1(raters, listed)
```

```text
== 1.43-1.44 tally and round1 (five raters) ==
  tremor->function       in       added=False tally={'in': 5, 'out': 0, 'unsure': 0, 'missing': 0, 'strong': 4, 'weak': 1, 'improving': 0, 'confidence': {'H': 5, 'M': 0, 'L': 0}}
  axial->independence    out      added=False tally={'in': 0, 'out': 5, 'unsure': 0, 'missing': 0, 'strong': 0, 'weak': 0, 'improving': 0, 'confidence': {'H': 0, 'M': 0, 'L': 0}}
  mood->cognition        round 2  added=False tally={'in': 3, 'out': 0, 'unsure': 2, 'missing': 0, 'strong': 0, 'weak': 3, 'improving': 0, 'confidence': {'H': 0, 'M': 0, 'L': 0}}
  rbd->hallucinations    round 2  added=True  tally={'in': 1, 'out': 0, 'unsure': 0, 'missing': 4, 'strong': 0, 'weak': 1, 'improving': 0, 'confidence': {'H': 0, 'M': 1, 'L': 0}}
```

Axial → independence is out with four "N" answers and one "Y" with strength 0; mood → cognition has three "in" and two "unsure" and returns; the added link returns with four missing answers.

**Design choices and pitfalls.** Unanimity is demanding: one missing answer sends a link to round 2. That follows protocol §6.4 ("when every rater gives it 1 or 2").

### 1.45 `round2` and `key` — majority, or the investigator

[`wp1.py:161`](../../src/pdx/wp1.py#L161), [`wp1.py:181`](../../src/pdx/wp1.py#L181)

**What it does.** `round2` decides each returning link by the majority of the round-2 raters. A rater who misses round 2 is not in `raters`, so that rater leaves the counts. Without a majority either way, the investigator's decision (from `decisions`, keyed `"from->to"`) settles the link, which is recorded as disputed; without a decision the verdict reads "investigator to decide". `key` writes a link as `"from->to"`.

**Inputs → outputs.** The round-2 raters' answers, the returning links and optional decisions → `{"raters": n2, "links": {link: {"verdict", "disputed", "tally"}}}`.

**The mathematics.** With $n_2$ round-2 raters,

$$
\text{verdict}=\begin{cases}\text{in}&2\cdot\#\text{in}>n_2\\ \text{out}&2\cdot\#\text{out}>n_2\\ \text{the investigator's decision, disputed}&\text{otherwise.}\end{cases}
$$

The majority is an absolute one, over all round-2 raters: an "unsure" or a blank answer counts in $n_2$ but for neither side.

**Worked example** ([`ex_wp1.py`](../guide_examples/s01/ex_wp1.py)), five round-2 raters on the two returning links:

```text
== 1.45 round2 (five raters) ==
  mood->cognition        in=4 out=1 unsure=0 missing=0 -> 'in', disputed=False
  rbd->hallucinations    in=2 out=1 unsure=1 missing=1 -> 'investigator to decide', disputed=True
```

The second link has more "in" than "out" answers (2 against 1) but no majority of the five raters, so it goes to the investigator.

**Design choices and pitfalls.** Protocol §6.4 says "after round 2 the majority decides; a tie is settled by the investigator". Read as "more in than out", the second link would be in; the code instead sends every link without an absolute majority to the investigator, so unsure and blank answers act as votes for "investigator" (§1.53, note 11). The docstring states the code's reading ("without a majority either way").

### 1.46 `build_map` — the draft expert map

[`wp1.py:185`](../../src/pdx/wp1.py#L185)

**What it does.** Combines rounds 1 and 2 into the draft map, in the format the H-P run reads (`configs/expert_map.json`, links as `[from, to]` pairs of domain names):
- `links`: kept links between domains; `treatment_links`: kept links that involve treatment;
- `disputed` and `disputed_left_out`: links settled by the investigator, in and out;
- `improving`: kept links whose modal sign is improving;
- `not_settled`: links still without a verdict;
- `raters`, and `investigator_led` if either round had fewer than five raters;
- a `status` saying the map is a draft until the investigator records it in the protocol.

**Inputs → outputs.** The round-1 result, the round-2 result (or `None`) and a source note → the map dictionary.

**The mathematics.** A link is in the map if its settled verdict is "in". It is improving if more than half of its "in" answers carry a minus sign:

$$
\text{improving}=\mathbf 1\Big\{\#\text{improving}>\tfrac12\,\#\text{in}\Big\},
$$

counted in the round that settled the link. The map is investigator-led if $\min(n_1,n_2)<5$.

**Worked example** ([`ex_wp1.py`](../guide_examples/s01/ex_wp1.py)):

```text
== 1.46 build_map ==
   {'status': 'draft: frozen only when the investigator records it in the protocol (§17)', 'source': 'made-up ratings', 'raters': {'round_1': 5, 'round_2': 5}, 'investigator_led': False, 'links': [['tremor', 'function'], ['mood', 'cognition']], 'treatment_links': [], 'disputed': [], 'disputed_left_out': [], 'improving': [], 'not_settled': ['rbd->hallucinations']}
   with the investigator's decision: {'links': [['tremor', 'function'], ['mood', 'cognition'], ['rbd', 'hallucinations']], 'disputed': [['rbd', 'hallucinations']], 'not_settled': [], 'investigator_led': False}
   four raters, treatment links: {'treatment_links': [['treatment', 'axial'], ['treatment', 'tremor']], 'improving': [['treatment', 'tremor']], 'investigator_led': True, 'raters': {'round_1': 4, 'round_2': None}}
```

With four raters, treatment → tremor has three improving answers out of four (improving), while treatment → axial has two out of four, a tie, and is recorded as worsening.

**Design choices and pitfalls.**
- A tie in the sign defaults to worsening (+1).
- D25 requires at least three raters; nothing checks it (with one or two raters the map is simply "investigator-led").
- D25 also asks that the map-dependent analyses be repeated without the disputed links; the map lists them for that purpose.

### 1.47 `round2_markdown` — the round-2 table

[`wp1.py:209`](../../src/pdx/wp1.py#L209)

**What it does.** Writes Section E of the round-2 questionnaire: each returning link with the group's anonymised round-1 answers (strong / weak / none / unsure) and empty columns for the new rating; a link a rater added is marked so.

**Inputs → outputs.** The round-1 result → a markdown string.

**The mathematics.** The four counts are $\#\text{strong}$, $\#\text{weak}$, $\#\text{out}$ and $\#\text{unsure}+\#\text{missing}$.

**Worked example** ([`ex_wp1.py`](../guide_examples/s01/ex_wp1.py)):

```text
== 1.47 round2_markdown ==
**E (round 2).** These links did not reach agreement in round 1 (5 raters). For each,
the group's round-1 answers are shown without names. Please rate it again, as in round 1.

| Link (this year → next year) | Round 1: strong / weak / none / unsure | Keep? | Strength | Confidence |
|---|---|---|---|---|
| Mood → Cognition | 0 / 3 / 0 / 2 | | | |
| RBD symptoms → Hallucinations | 0 / 1 / 0 / 4 (added by a rater) | | | |
```

**Design choices and pitfalls.** For an added link, the raters who were never asked about it appear under "unsure" (4 here); the note "(added by a rater)" explains it, but a separate "not asked" count would be clearer to the panel.

---

### 1.48 `scripts/_common.py` — `ROOT`, `DATA`, `RESULTS`, `require_inside`, `require_approval`, `to_json`

[`_common.py:7`](../../scripts/_common.py#L7), [`_common.py:14`](../../scripts/_common.py#L14), [`_common.py:21`](../../scripts/_common.py#L21), [`_common.py:28`](../../scripts/_common.py#L28)

**What it does.** The run scripts share these helpers, which carry the data-use rules of protocol §14:
- `ROOT` is the repository folder; importing `_common` puts `ROOT/src` on the import path, so `from pdx import ...` works without installing the package. `DATA` and `RESULTS` are `ROOT/data` (participant-level, git-ignored) and `ROOT/results` (aggregates).
- `require_inside(path, folder, what)` resolves the path (symbolic links included) and stops the script unless it lies strictly inside the folder.
- `require_approval(reference, what)` stops the script unless an approval reference of at least 8 characters is given: "every command that reads PPMI participant data waits for protocol approval".
- `to_json(obj)` makes results JSON-safe: numpy arrays become lists, numpy numbers become Python numbers, non-finite floats become `null`, dictionary keys become strings.

**Inputs → outputs.** Paths and strings → a resolved path, or `SystemExit` with a message; any nested object → a JSON-safe copy.

**The mathematics.** `require_inside` checks $\text{folder}\in\text{parents}(\text{path})$ after both are resolved, so the folder itself does not qualify and `..` cannot climb out. `to_json` maps $x\mapsto$ `None` when $x$ is a float with $\lvert x\rvert=\infty$ or $x$ = NaN.

**Worked example** ([`ex_scripts.py`](../guide_examples/s01/ex_scripts.py)), with `DATA` and `RESULTS` pointed at a temporary folder (`_setup.protect_script_paths`):

```python
print("require_inside, inside: ", _common.require_inside(TMP / "data" / "x.parquet", _common.DATA, "x").relative_to(TMP))
for ref in (None, "ok", "synthetic run, 4 Nov 2026"):
    try:
        _common.require_approval(ref, "the demo")
        print(f"require_approval({ref!r}): passes")
    except SystemExit as e:
        print(f"require_approval({ref!r}): {str(e)[:60]}...")
print("to_json:", _common.to_json({"a": np.float64(0.5), "b": np.array([1, 2]), "c": float("nan"), "d": np.int64(3),
                                    "e": np.bool_(True), ("x", 1): [np.inf, 2.0]}))
```

```text
== 1.48 _common ==
ROOT is the repository: True
require_inside, inside:  data/x.parquet
require_inside, outside: Refusing: the visit table must be inside <tmp>/data (got <tmp>/elsewhere.parquet).
require_inside, outside: Refusing: the folder itself must be inside <tmp>/data (got <tmp>/data).
require_approval(None): Refusing to run the demo: it reads PPMI participant data, wh...
require_approval('ok'): Refusing to run the demo: it reads PPMI participant data, wh...
require_approval('synthetic run, 4 Nov 2026'): passes
to_json: {'a': 0.5, 'b': [1, 2], 'c': None, 'd': 3, 'e': True, "('x', 1)": [None, 2.0]}
```

**Design choices and pitfalls.**
- The approval check is a deliberate speed bump, not a security control: any string of 8 characters passes. Its value is that the reference is written into every output it guards (`manifest.json`, `wp0_audit.json`, `panel_spec.json`), so each output records under which approval it was made.
- `to_json` handles what the audit returns. It does not handle `pd.NA` or timestamps, which `json.dumps` would refuse; and `build_ppmi_tables.py` writes some outputs without it (§1.49).

### 1.49 `scripts/build_ppmi_tables.py` — `USED`, `manifest`, `main`

[`build_ppmi_tables.py:24`](../../scripts/build_ppmi_tables.py#L24), [`build_ppmi_tables.py:32`](../../scripts/build_ppmi_tables.py#L32), [`build_ppmi_tables.py:45`](../../scripts/build_ppmi_tables.py#L45)

**What it does.** The WP0 build. After the approval check, `main`:
1. runs `ppmi.build` on the raw folder (`--raw`, default `ppmi.RAW`);
2. writes `participants.parquet`, `grid.parquet` and `wp0_visits.parquet` to `data/derived/`;
3. writes to `results/wp0/`: `participant_flow.json` (`flow_table` through `protect_2way`), `phenotype_axes.json` (motor phenotype × clinical subtype, and clinical subtype × era, both through `protect_2way`), `retest_reliability.json` (`screening_retest`) and `manifest.json`;
4. prints three counts.

`USED` lists the 15 tables `ppmi.py` reads. `manifest` records the provenance protocol §14 asks for: the download date stamps, a SHA-256 hash of every dated file of those tables, the size of the PD cohort and of the primary population, the number of WP0 rows, a pointer to the rules, and the approval.

**Inputs → outputs.** Command-line options → the files above; console output of counts only.

**The mathematics.** The hash is $\mathrm{SHA256}(\text{file bytes})$, so any change to a file changes its entry. The stamps are the set of `DDMonYYYY` strings found in the file names.

**Worked example** ([`ex_scripts.py`](../guide_examples/s01/ex_scripts.py)), on the mock download, first without an approval:

```python
run(build_ppmi_tables)                                                   # no approval: refused
run(build_ppmi_tables, "--approval", "synthetic run, 4 Nov 2026", "--raw", raw)
```

```text
== 1.49 build_ppmi_tables.py ==
   SystemExit: Refusing to run the PPMI table build: it reads PPMI participant data, which needs the investigator's written approval (protocol §14). Pass --approval '<who, date, document>' once it is given.
PD cohort 13; primary population 9; grid rows 26; wrote data/derived/*.parquet and results/wp0/participant_flow.json, manifest.json
written: ['data/derived/grid.parquet', 'data/derived/participants.parquet', 'data/derived/wp0_visits.parquet', 'results/wp0/manifest.json', 'results/wp0/participant_flow.json', 'results/wp0/phenotype_axes.json', 'results/wp0/retest_reliability.json']
participant_flow.json cells: {'no baseline visit': {'PPMI 1.0': '<5', 'PPMI 2.0': 0}, 'not enrolled (screen failed, excluded, declined or pending)': {'PPMI 1.0': 0, 'PPMI 2.0': '<5'}, 'not sporadic (genetic enrolment)': {'PPMI 1.0': '<5', 'PPMI 2.0': 0}, 'primary: PIGD': {'PPMI 1.0': '<5', 'PPMI 2.0': '<5'}, 'primary: TD': {'PPMI 1.0': '<5', 'PPMI 2.0': 0}, 'treated at baseline': {'PPMI 1.0': '<5', 'PPMI 2.0': 0}}
manifest: {'download_stamps': ['01Jan2026'], 'files_sha256': 15, 'pd_cohort': 13, 'primary_population': 9, 'grid_rows_primary': 26, 'rules': 'see src/pdx/ppmi.py docstring (provisional choices fixed in WP1)', 'approval': 'synthetic run, 4 Nov 2026'}
retest_reliability.json: {'pairs': 0, 'median_gap_days': None, 'tremor': 'fewer than 50 pairs', 'brady_rigidity': 'fewer than 50 pairs', 'axial': 'fewer than 50 pairs', 'function': 'fewer than 50 pairs'}
```

Every cell of the mock's flow is below 5, so all are suppressed. The script then copies the mock and adds two files: an older download of the participant-status table (with every age set to 99, to show which file is read) and a file whose name begins with another table's name:

```text
   manifest and read when a second download and a longer table name sit in the folders:
   stamps: ['01Jan2026', '05Sep2025'] | files hashed: 17 | includes the Part I patient file: True
   age at enrolment read by participants(): [99.0]
```

The manifest hashes both downloads and also `MDS-UPDRS_Part_I_Patient_Questionnaire_01Jan2026.csv`, which matches `MDS-UPDRS_Part_I_*.csv` but is never read. `read` takes the September 2025 file, the older one, because "05" sorts after "01" (§1.14).

**Design choices and pitfalls.**
- The manifest records every dated file matching `<name>_*.csv`, not the file `read` actually used. With one download per folder the two agree (§1.53, note 19).
- `retest_reliability.json` is written with `json.dumps` directly, so a non-finite value (an ICC of NaN when a domain has no variance) would be written as `NaN`, which is not valid JSON (§1.53, note 21).
- The participant-level visit table that `screening_retest` needs is used in memory and not saved.

### 1.50 `scripts/run_wp0_audit.py` — `main`

[`run_wp0_audit.py:22`](../../scripts/run_wp0_audit.py#L22)

**What it does.** Runs `wp0.audit` on the WP0 visit table. It refuses to run without an approval, refuses a visit table outside `data/` and an output folder outside `results/`, reads a Parquet (or CSV) file, uses the domains given with `--domains` or else `ppmi.DOMAINS`, adds the approval to the report and writes `results/wp0/wp0_audit.json` through `to_json`.

**Inputs → outputs.** `--visits`, `--approval`, optionally `--domains`, `--min-cell` (default 5) and `--out` → `wp0_audit.json`.

**The mathematics.** None beyond §1.32–§1.37.

**Worked example** ([`ex_scripts.py`](../guide_examples/s01/ex_scripts.py)), first with a visit table outside `data/`, then with the one `build_ppmi_tables.py` wrote:

```python
run(run_wp0_audit, "--visits", TMP / "wp0_visits.parquet", "--approval", "synthetic run, 4 Nov 2026")
run(run_wp0_audit, "--visits", visits, "--approval", "synthetic run, 4 Nov 2026")
```

```text
== 1.50 run_wp0_audit.py ==
   SystemExit: Refusing: the visit table must be inside <tmp>/data (got <tmp>/wp0_visits.parquet).
wrote <tmp>/results/wp0/wp0_audit.json
audit keys: ['min_cell', 'participant_flow', 'landmark', 'transitions_after_landmark', 'intervals', 'visits_per_participant', 'mode', 'motor_state', 'missing_share', 'missing_share_by_motor_state', 'domain_summaries', 'overlap', 'phenotype_stability', 'part2_events', 'approval']
participant_flow: {'cells': {'PPMI 1.0': {'PIGD': '<5', 'TD': '<5'}, 'PPMI 2.0': {'PIGD': '<5', 'TD': 0}}, 'row_totals': {'PPMI 1.0': 'suppressed', 'PPMI 2.0': 'suppressed'}, 'column_totals': {'PIGD': 'suppressed', 'TD': 'suppressed'}, 'total': 9}
```

There is no `sites` key: the WP0 visit table has no site column (§1.53, note 14).

**Design choices and pitfalls.** The `--out` default is computed when `main` runs, from `_common.RESULTS`, and checked with `require_inside`, so the audit cannot write outside `results/`.

### 1.51 `scripts/build_analysis_panel.py` — `main`

[`build_analysis_panel.py:27`](../../scripts/build_analysis_panel.py#L27)

**What it does.** Builds the H-P panel from the WP0 tables. After the approval check it reads `participants.parquet` and `grid.parquet` from `data/derived/`, calls `ppmi.analysis_panel` with the chosen domains (default `CORE_DOMAINS`) and `--motor-on`, reads the fixed measurement-error variances of those domains from a JSON file (default `configs/measurement_error.json`, D28), and writes to `data/derived/`:
- `panel_tdpigd.npz` (`save_panel`);
- `normal_scores.npz`, the frozen tables as arrays `values_j` and `scores_j`;
- `panel_spec.json`: the domains, covariates, treated domains, `motor_on`, the counts, `meas_var`, the measurement source and the approval.

It prints the panel's counts.

**Inputs → outputs.** `--approval`, optionally `--domains`, `--measurement`, `--motor-on` → the three files.

**The mathematics.** `meas_var` is the vector $\big(\sigma^2_{e,j}\big)_{j}$ in the panel's domain order, which becomes $\Sigma_e=\mathrm{diag}(\sigma^2_{e,j})$, `spec.meas_var`, in the engine. The JSON states its scale as "error variance on the pooled normal-scores scale = 1 − reliability"; §1.31 discusses that conversion.

**Worked example** ([`ex_scripts.py`](../guide_examples/s01/ex_scripts.py)), with a synthetic measurement file (0.25 for every domain):

```python
meas.write_text(json.dumps({"source": "synthetic", "error_variance": {d: 0.25 for d in ppmi.DOMAINS}}))
run(build_analysis_panel, "--approval", "synthetic run, 4 Nov 2026", "--measurement", meas)
```

```text
== 1.51 build_analysis_panel.py ==
panel: {"0": {"participants": 4, "visits": 7, "transitions": 3, "median_visits": 2.0, "share_missing": 0.0}, "1": {"participants": 5, "visits": 10, "transitions": 5, "median_visits": 2.0, "share_missing": 0.0}}
panel_spec.json: {'domains': ['tremor', 'brady_rigidity', 'axial', 'function', 'independence', 'cognition', 'mood', 'autonomic', 'sleepiness', 'rbd'], 'treated_domains': ['tremor', 'brady_rigidity', 'axial', 'function'], 'meas_var': [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25], 'measurement_source': 'synthetic', 'motor_on': False}
panel: 9 participants, 10 domains | normal_scores.npz arrays: 20
```

**Design choices and pitfalls.**
- A domain missing from the measurement file stops the script with a `KeyError`, which is the right failure.
- The printed counts are exact counts by phenotype, for the console; they are small only on synthetic data.

### 1.52 `scripts/wp1_consensus.py` — `save`, `load`, `read_raters`, `main`

[`wp1_consensus.py:33`](../../scripts/wp1_consensus.py#L33), [`wp1_consensus.py:38`](../../scripts/wp1_consensus.py#L38), [`wp1_consensus.py:43`](../../scripts/wp1_consensus.py#L43), [`wp1_consensus.py:56`](../../scripts/wp1_consensus.py#L56)

**What it does.** Runs the WP1 consensus from the command line, with no PPMI data. The returned questionnaires identify the raters, so they must sit in `data/wp1/` (git-ignored); only anonymised counts go to `results/wp1/`, with raters numbered in sorted file order.
- `sheet <file>` writes a blank CSV answer sheet with the 31 listed links.
- `round1 <files>` reads each rater's file, prints how many listed links each answered, applies `round1`, and writes `results/wp1/round1.json` and the round-2 table `round2_section_E.md`.
- `round2 <files> [--decisions file]` reloads `round1.json`, reads the round-2 answers to the returning links, applies `round2` with the investigator's decisions, writes `round2.json`, builds the map and writes `configs/expert_map_draft.json`; it lists the links still to settle.

`save` and `load` write and read a round's result as JSON, with each link keyed `"from->to"`; `read_raters` reads the files and stops if one yields no ratings.

**Inputs → outputs.** A step, file paths and optional decisions → the files above.

**The mathematics.** None beyond §1.44–§1.46.

**Worked example** ([`ex_scripts.py`](../guide_examples/s01/ex_scripts.py)). With `results/wp1` and the draft map's folder redirected to the temporary folder, three made-up raters fill the answer sheet: links 1 and 2 kept, link 5 not kept, everything else "?" (one rater is also unsure on link 2). In round 2, two of three raters keep every returning link (all three keep bradykinesia–rigidity → daily function), except axial → daily function, which gets one "Y", one "N" and one "?". Round 2 is run first without and then with a decision for that link:

```text
== 1.52 wp1_consensus.py ==
wrote <tmp>/data/wp1/answer_sheet.csv
sheet: 31 links; first rows: [['link', 'keep', 'strength', 'confidence'], ['Tremor → Daily function', '', '', '']]
rater 1: 31 links read, 31 of the 31 listed answered
rater 2: 31 links read, 31 of the 31 listed answered
rater 3: 31 links read, 31 of the 31 listed answered
3 raters: 1 links in, 1 out, 29 back in round 2. Wrote results/wp1/round1.json and round2_section_E.md.
round1.json: 3 raters; {'tremor->function': 'in', 'brady_rigidity->function': 'round 2', 'axial->function': 'round 2', 'function->independence': 'round 2', 'axial->independence': 'out'} ...
rater 1: 29 links read, 29 of the 29 listed answered
rater 2: 29 links read, 29 of the 29 listed answered
rater 3: 29 links read, 29 of the 29 listed answered
Map: 21 domain links and 8 treatment links; 0 disputed; investigator-led: True. Wrote expert_map_draft.json.
Still to settle (no round-2 majority): axial->function. Add them to --decisions and rerun.
rater 1: 29 links read, 29 of the 29 listed answered
rater 2: 29 links read, 29 of the 29 listed answered
rater 3: 29 links read, 29 of the 29 listed answered
Map: 22 domain links and 8 treatment links; 1 disputed; investigator-led: True. Wrote expert_map_draft.json.
expert_map_draft.json: {'status': 'draft: frozen only when the investigator records it in the protocol (§17)', 'source': 'clinical_expert_questionnaire.md, rounds 1 and 2', 'raters': {'round_1': 3, 'round_2': 3}, 'investigator_led': True, 'links': 22, 'treatment_links': 8, 'disputed': [['axial', 'function']], 'disputed_left_out': [], 'improving': [], 'not_settled': []}
```

Reading it: link 2 (bradykinesia–rigidity → daily function) is not unanimous in round 1 because one rater was unsure; in round 2 all three keep it. With three raters the map is investigator-led. Axial → daily function has no majority (1 "Y", 1 "N", 1 "?"), so the first round-2 run leaves it unsettled, and the decision file settles it as a disputed link.

**Design choices and pitfalls.**
- Every input file must be inside `data/` (`require_inside`), so a returned questionnaire cannot be read from, or left in, a shared folder by mistake.
- Raters are numbered by sorted file name in each round, so "rater 1" of round 1 need not be "rater 1" of round 2. Nothing downstream needs the link.
- `round2` writes the draft map into `configs/`, outside `data/` and `results/`; it contains only links and counts.

---

### 1.53 Notes for the author

Things noticed while reading the code and running the examples. Nothing here was changed in the code. They are ordered by how much they could change a reported number or a published table.

**Could change a reported number**

1. **A dose switch in the visit's month is counted twice.** [`ppmi.py:180`](../../src/pdx/ppmi.py#L180). The active-entry rule is inclusive at both ends at month resolution, `(vd.start <= vd.date) & (vd.stop.isna() | (vd.stop >= vd.date))`. When the log stops one entry and starts its replacement in the visit's month, both count: in [`ex_ledd.py`](../guide_examples/s01/ex_ledd.py) a switch from 300 to 400 mg/day gives `ledd` = `ledd_levodopa` = 700 at the visit of that month (§1.19). Dose changes are often decided at a visit, so this is likely to be a common case. Suggested check (counts only): the number of visits at which a stopped and a started entry of the same drug are both active. The double count feeds `dose`, `exposure` (the long-duration treatment term), `treated`, `dose_change` and the WP0 counts. Options: make the stop exclusive (`vd.stop > vd.date`), or net a stopped and a started entry of the same drug within a month. The mock has no switch, so no test covers it.

2. **"Untreated" is decided from `PDTRTMNT` alone.** [`ppmi.py:136–139`](../../src/pdx/ppmi.py#L136). A Part III record with no `PDSTATE`, not on an ON/OFF page, and `PDTRTMNT` 0 **or blank** is labelled untreated and enters the primary (OFF or untreated) stream, even when the participant has an active LEDD on that date. The mock's participant 10 (300 mg/day throughout) has its year-1 and year-2 examinations labelled untreated while `treated` = 1 (§1.26). The docstring says a treated participant's examination without a recorded state is not used; that holds only when `PDTRTMNT` = 1. After treatment starts, such an examination is of unknown state (it could be ON). Suggested check (counts only): the number of grid rows with `motor_state == "untreated"` and `treated == 1`, by year. A fix would take the LEDD into account (it is computed later, in `visits`) when deciding between "untreated" and "unrecorded".

3. **Hidden cells can be recovered from the labels and the grand total.** [`wp0.py:76–77`](../../src/pdx/wp0.py#L76), [`wp0.py:103`](../../src/pdx/wp0.py#L103); [`wp0.py:72–75`](../../src/pdx/wp0.py#L72) with [`wp0.py:163`](../../src/pdx/wp0.py#L163). Two separate leaks (§1.33–§1.34, [`ex_disclosure.py`](../guide_examples/s01/ex_disclosure.py)):
   - A primary cell is labelled `"<5"` and a complementary cell `"suppressed"`; complementary cells are by construction at least 5. With the grand total published, a one-row table [1, 5] gives back both cells exactly ($R=6=p+q\,c_0$). In general the hidden cells are pinned whenever they sum to $p+q\,c_0$ ($p$ primary and $q$ complementary cells), and choosing the smallest complement makes that more likely.
   - `protect_1way` cannot pair a small cell that is the only non-zero cell, and `audit` publishes `total` next to `by_phenotype` and `by_cohort`, so `{"total": 3, "by_cohort": {"PPMI 1.0": "<5", "PPMI 2.0": 0}}` gives the 3 back.

   The unit test checks only the linear condition. Options: label every hidden cell the same; require the complement to leave each primary cell an interval of at least $c_0-1$ values; suppress the total (or a zero) when a single non-zero cell is small. With PPMI's sizes most cells are large, but the guarantee should not depend on that before publication.

4. **`error_variance = 1 − ICC` is a share, not a variance on the pooled scale.** [`ppmi.py:480`](../../src/pdx/ppmi.py#L480). $1-\text{ICC}$ is the error share of the retest sample's variance. It equals the error variance on the pooled normal-scores scale only if the screening and baseline scores have variance 1 on that scale, but that scale is fitted on all years, so the early scores are narrower. In [`ex_retest.py`](../guide_examples/s01/ex_retest.py) the reported 0.222 compares with an error variance of 0.087 on the pooled scale (the MSE the code already computes), because the retest scores have variance 0.39 there (§1.31). `configs/measurement_error.json` converts literature reliabilities with the same rule ("= 1 − reliability"), so the fixed $\Sigma_e$ of D28 may be too large if the literature samples were narrower than the pooled PPMI population. Suggested: report MSE (or MSE + (MSC − MSE)/n) alongside, and convert a reliability $\rho$ from a sample with variance $V$ on the pooled scale as $(1-\rho)V$.

5. **In the default panel the current-dose term never enters the likelihood.** [`ppmi.py:434`](../../src/pdx/ppmi.py#L434) with [`model.py:191`](../../src/pdx/engine/model.py#L191). `on` is 1 only for ON-only visits and only with `motor_on`; the engine multiplies the short-duration Emax term by `on`. So with the default (OFF or untreated examinations, `motor_on=False`) every `on` is 0 and `panel.dose` has no effect ([`ex_notes.py`](../guide_examples/s01/ex_notes.py): $h=0$ at 0, 300 and 900 mg with `on` = 0). That is coherent for the motor examinations, but daily function (MDS-UPDRS Part II) is a patient questionnaire about the past week, not an examination in a state, so its short-duration response is also switched off. Worth a decision in WP1 (D24): a per-domain state indicator, with `function` always "on".

6. **Exposure is the current levodopa LEDD, not an effect-compartment exposure.** [`ppmi.py:433`](../../src/pdx/ppmi.py#L433). Protocol §6.1 (D24) describes the long-duration effect as an Emax function of "the exposure over the preceding weeks (a slow effect compartment …) built from the medication log". The code uses the levodopa LEDD in force on the visit date (the docstring calls it provisional), without the COMT boost, and with the double count of note 1.

7. **Domain sums are not prorated.** [`ppmi.py:91–92`](../../src/pdx/ppmi.py#L91). A domain scored from 80–99% of its items is the sum of the items present, so it is lower than the full score (§1.16). If remote examinations systematically lack items (rigidity and the pull test cannot be rated by video), their bradykinesia–rigidity domain is missing (11 of 16 < 13) and their axial domain is the sum of four items without the pull test, so remote visits would score lower on the axial domain for a reason unrelated to the disease. Protocol §6.1 leaves missing-item rules to WP1; prorating ($S\cdot n/\lvert\mathcal O\rvert$), or requiring all items for the motor domains, are the alternatives. Suggested check (counts only): how often each motor item is missing, by visit mode.

8. **The levodopa name pattern misses some names.** [`ppmi.py:58`](../../src/pdx/ppmi.py#L58). Not matched: "L-dopa", "Duopa", "co-careldopa", "co-beneldopa", "Dhivy"; matched although not levodopa: carbidopa alone ([`ex_item_lists.py`](../guide_examples/s01/ex_item_lists.py), §1.13). A missed levodopa entry counts in `ledd` but not in `ledd_levodopa` (the exposure) and makes `classes` "other". Suggested check: the distinct `LEDTRT` values that the pattern does not match, with their counts.

9. **"ON only" requires the ON examination's tremor sum.** [`ppmi.py:163`](../../src/pdx/ppmi.py#L163). `m.tremor_on.notna()` is the test, so an ON examination (25 or more of 31 items) with three or more of its ten tremor items blank gets no `motor_state` at all ([`ex_motor.py`](../guide_examples/s01/ex_motor.py), participant 3). On the grid it then ranks with "anything else" (state rank 2), and with `motor_on` its motor domains are not used. Testing whether an ON examination exists (a flag set when the `on` frame has a row for the visit) would fix it.

10. **The 24-month endpoint differs from protocol §12.4.** [`ppmi.py:370–375`](../../src/pdx/ppmi.py#L370). The code counts death within 730.5 days of enrolment, or S&E below 80 at the year-2 visit, among everyone with a year-2 S&E or a death within 24 months. Protocol §12.4's provisional endpoint is alive and independent 24 months **after** a year-2 landmark (years 2 to 4), among participants alive and independent at the landmark. The WP0 event count therefore answers a different question from the one protocol §12.4's sample-size calculation needs. Also, a death record without a date never counts (note 24).

11. **Round 2's majority is an absolute majority of all round-2 raters.** [`wp1.py:171–176`](../../src/pdx/wp1.py#L171). Unsure and blank answers count in the denominator, so 2 "in", 1 "out", 1 "unsure" and 1 blank go to the investigator (§1.45). Protocol §6.4 says "the majority decides; a tie is settled by the investigator", which reads more naturally as a comparison of in with out. Either reading is defensible; the protocol and the code should say the same.

12. **A range typed as a strength is read as an improving link.** [`wp1.py:57–59`](../../src/pdx/wp1.py#L57). "1-2" and "1–2" give strength 1 with sign −1 (§1.40); "0?" is "out" rather than "unsure"; a "3" is ignored. Printing answers with two digits or a digit outside 0–2 for the investigator to confirm would catch these.

**Differences from the protocol or from the shared notation**

13. **Era comes from the enrolment year, and a missing date makes it PPMI 2.0.** [`ppmi.py:106`](../../src/pdx/ppmi.py#L106). `np.where(year < 2020, "PPMI 1.0", "PPMI 2.0")` sends a missing enrolment date to PPMI 2.0 ([`ex_participants.py`](../guide_examples/s01/ex_participants.py)), and `build`'s fallback to the baseline date (line 333) is applied to the anchor but not to the era. Protocol §4 asks WP0 to reconcile the eras with PPMI's curated cohort definitions.

14. **No site column reaches the audit.** [`ppmi.py:376–382`](../../src/pdx/ppmi.py#L376); [`wp0.py:182–196`](../../src/pdx/wp0.py#L182). The WP0 visit table has no `site`, so `audit` skips its site summary, and `audit` computes missing shares by phenotype, mode and motor state but never by site. Protocol §3 lists "missing data by phenotype, site, mode and motor state", and protocol §4 a flow table "by phase, recruitment group, site, visit mode and treatment status"; `flow_table` gives reason × era only.

15. **Panel times are years since enrolment, not since the landmark.** [`ppmi.py:431`](../../src/pdx/ppmi.py#L431). The shared notation (§0.4) defines $t_{ik}$ as years from the year-1 landmark; the code uses `date_years`, years from the enrolment date, so the landmark visit is near $t=1$ (0.9993 in [`ex_analysis_panel.py`](../guide_examples/s01/ex_analysis_panel.py)). The engine uses only differences (`likelihood.py:76`), so no result changes; the notation table should say so, or the panel should subtract the landmark time.

16. **Era enters $z_i$ centred, so $Q_g$ is the noise at the mean era.** [`ppmi.py:427`](../../src/pdx/ppmi.py#L427) with [`likelihood.py:165`](../../src/pdx/engine/likelihood.py#L165). The noise is multiplied by $e^{\omega z_{i,\text{era}}}$ with the centred era ($-\bar e$ or $1-\bar e$), not the 0/1 indicator. The notation "$Q_g$ × $e^{\omega\,\mathrm{era}_i}$" is right if $\mathrm{era}_i$ is read as centred; $Q_g$ itself is then the noise at the average era, not PPMI 1.0's. Worth one sentence in §0.4.

    Checking the rest of §0.4 against `model.py` and `likelihood.py` found one more point worth a word: $\Gamma^y_g$ is diagonal. `gam_y` has shape $(G,d)$ and is applied elementwise (`y1c * gam_y[g]`, `likelihood.py:143`), so each domain's load mean depends on its own year-1 score only, while $\Gamma^z_g$ (`gam_z`, $(G,d,p)$) is a full matrix. Everything else in the table matches the code.

17. **Calendar-month windows are applied in day units.** [`ppmi.py:309–312`](../../src/pdx/ppmi.py#L309); [`wp0.py:170`](../../src/pdx/wp0.py#L170). With dates on the 15th, "±4 months" is ±121.75 days, so a visit exactly four calendar months from the anniversary is in when the months span 120–121 days and out when they span 122–123 ([`ex_annual_grid.py`](../guide_examples/s01/ex_annual_grid.py), participants 2 and 3). Likewise the 12 ± 3 months rule of protocol §6.3: exactly 9 calendar months is out for February → November (0.7474 years) and in for March → December (0.7529), and exactly 15 months can fall either side ([`ex_notes.py`](../guide_examples/s01/ex_notes.py)). Counting calendar months (`12·Δyear + Δmonth`) would make both rules exact.

**Smaller items**

18. **`NOT_GRID` matches exact codes.** [`ppmi.py:37`](../../src/pdx/ppmi.py#L37). `U02` and later unscheduled codes, if present, are grid candidates, and §1.25's example shows a `U02` winning a grid year. A prefix rule would match the docstring's "unscheduled events".

19. **`read` picks the latest file by its day digits, and the manifest does not record which file was read.** [`ppmi.py:71`](../../src/pdx/ppmi.py#L71); [`build_ppmi_tables.py:35–38`](../../scripts/build_ppmi_tables.py#L35). With two downloads in a folder, `05Sep2025` beats `01Jan2026` ([`ex_scripts.py`](../guide_examples/s01/ex_scripts.py)). The manifest hashes every file matching `<name>_*.csv`, which includes older downloads and longer table names (`MDS-UPDRS_Part_I_Patient_Questionnaire_…` under `MDS-UPDRS_Part_I`). Sorting by the parsed date, and recording the path `read` used, would make both exact.

20. **`transforms.orient` would undo `ppmi.py`'s orientation.** [`transforms.py:18`](../../src/pdx/transforms.py#L18). `REVERSED` includes the engine names `cognition` and `independence`, which `ppmi.visits` has already oriented as $30-\text{MoCA}$ and $100-\text{S\&E}$ (§1.7). `orient` is called only by a test, so nothing is wrong today; a future caller should pass `reversed_domains=()` for `ppmi.py`'s output.

21. **Some counts are published without disclosure control, and one file can hold invalid JSON.** [`wp0.py:166`](../../src/pdx/wp0.py#L166) (`across_a_missed_year`), [`wp0.py:215`](../../src/pdx/wp0.py#L215) (`at_risk`), [`ppmi.py:465`](../../src/pdx/ppmi.py#L465) (`pairs`, `median_gap_days`, the latter from fewer than 50 values), [`build_ppmi_tables.py:70`](../../scripts/build_ppmi_tables.py#L70) (`json.dumps` without `to_json`). [`ex_notes.py`](../guide_examples/s01/ex_notes.py) shows `across_a_missed_year: 1`, `pairs: 3` and `{"icc_2_1": NaN}`.

22. **The clinical subtype's percentile rule.** [`ppmi.py:287–295`](../../src/pdx/ppmi.py#L287). Mild motor-predominant uses "at or below the 75th percentile"; Fereshtehnejad's definition used "below", which matters with tied integer scores ([`ex_clinical_subtype.py`](../guide_examples/s01/ex_clinical_subtype.py): three RBDSQ scores equal the percentile). Each percentile is computed on the rows where that score is present, not on the classifiable rows. The motor summary leaves out MDS-UPDRS 3.1 and 3.2. All three are for WP1 (D2).

23. **The map's details.** [`wp1.py:199`](../../src/pdx/wp1.py#L199), [`wp1.py:204`](../../src/pdx/wp1.py#L204), [`wp1.py:222`](../../src/pdx/wp1.py#L222). A tie in the sign defaults to worsening; D25's minimum of three raters is not checked; in the round-2 table, raters never asked about an added link appear as "unsure".

24. **Participants table details.** [`ppmi.py:111–120`](../../src/pdx/ppmi.py#L111). A death record without a date does not set `died` or `death_date`; duplicate demographics or diagnosis rows are resolved by file order (`keep="last"`), not by date ([`ex_participants.py`](../guide_examples/s01/ex_participants.py)).

25. **Month-resolution ties.** [`ppmi.py:150–152`](../../src/pdx/ppmi.py#L150), [`ppmi.py:340`](../../src/pdx/ppmi.py#L340). A drug started in the baseline month makes the participant "treated at baseline" (excluded), even if it was started after the examination. A visit's date is the earliest Part III record's, while its scores come from the most complete record, which can be a later one ([`ex_motor.py`](../guide_examples/s01/ex_motor.py), participant 5).

26. **A domain with no observed value breaks the transform.** [`transforms.py:54`](../../src/pdx/transforms.py#L54). `fit` accepts an all-missing column and `transform` then raises (`np.interp` on an empty table). `screening_retest` therefore raises, rather than reporting "fewer than 50 pairs", when a requested domain is missing on the whole grid (§1.31).

27. **`Panel` does not check that every participant has the same number of domains.** [`data.py:33`](../../src/pdx/data.py#L33). `d` is read from `y[0]`; a participant with another width passes `__post_init__` and fails later in the engine's `prepare` ([`ex_panel.py`](../guide_examples/s01/ex_panel.py)).

28. **The baseline phenotype needs Part II at the baseline event.** [`ppmi.py:237–247`](../../src/pdx/ppmi.py#L237), [`ppmi.py:271–272`](../../src/pdx/ppmi.py#L271). Two of the five PIGD items are Part II items, and the rule needs four of five, so a baseline without Part II has no phenotype ("no usable baseline phenotype"). `sc_as_bl` is applied to the non-motor questionnaires but not to Part II. [`ex_notes.py`](../guide_examples/s01/ex_notes.py) removes one mock participant's baseline Part II and gets exactly that exclusion. Worth a count (by era) of baselines without Part II.

29. **Branches no test exercises.** `tests/test_ppmi.py` runs `build` and `analysis_panel` on the mock, which has no screening visits, no ON-only or "unrecorded" examinations, no out-of-range item codes, no COMT entry, no dose switch, no S&E below 80 and no indeterminate participant. So the following are untested: `sc_as_bl`; `motor_on=True`; the motor-state labels other than untreated and OFF; the 80% thresholds; the COMT multiplier; the "no usable baseline phenotype" exclusion; `event_24m` through S&E; the subtype classes beyond the trivial; `screening_retest`'s ICC (the mock has no pairs); `save_panel`/`load_panel`; `protect_1way` on its own. The examples of this chapter exercise each of these once on synthetic data and could be turned into tests; [`ex_notes.py`](../guide_examples/s01/ex_notes.py) covers the two that no other example reaches:

    ```text
    == note 29: two branches the unit tests never reach ==
      participant 2: no usable baseline phenotype
      participant 4, event_24m on the year-0 row: 1.0
    ```

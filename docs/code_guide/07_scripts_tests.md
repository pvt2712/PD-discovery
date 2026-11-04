## 7. Run scripts, tests and toy simulations

The package (`src/pdx/`, chapters 1–6) is driven by a handful of command-line scripts, checked by a plain test runner, and was designed on the evidence of a set of toy simulations. This chapter covers all three:
- **the run scripts** that are not data builders: the engine gate, the WP0 power gate, the H-P analysis of the PPMI panel, the WP3 qualification, the smoke test, the thread benchmark and the Markdown-to-Word converter (§7.1–§7.25);
- **the tests** in `tests/`: for each test, what it proves, how, and what it does not cover (§7.26–§7.42);
- **the toy simulations** in `scripts/toy/`, the synthetic evidence of protocol Appendix E (§7.43–§7.57).

The data-build scripts (`_common.py`, `build_ppmi_tables.py`, `run_wp0_audit.py`, `build_analysis_panel.py`, `wp1_consensus.py`) are covered in [§1](01_data.md). This chapter uses only the guards of `_common.py`, summarised in §7.0.

### 7.0 Overview

**Which script runs which work package.**

| Script | Work package (protocol) | Data | Inputs | Outputs | Guards | Restart |
|---|---|---|---|---|---|---|
| `run_engine_gate.py` | WP2, the engine gate (§16) | synthetic | options; with `--wp0`, `configs/wp0_sim_settings.json` | `results/engine_gate/<name>.json` and `<name>.log` | output inside `results/` | `<name>.partial.json`, keyed by the settings, deleted when the report is written |
| `run_power_check.py` | WP0's power gate (§3) | synthetic | `configs/wp0_sim_settings.json` (or the WP0 audit's aggregates) | `results/wp0/power/datasets/*.json`, `results/wp0/power/summary.json` | none (no participant data) | one JSON per dataset, skipped if present |
| `run_hp_real.py` | WP4, H-P on PPMI (§7.3) | **PPMI panel** | `data/derived/panel_tdpigd.npz`, `panel_spec.json`, an expert map | `results/wp4/hp_first_look.json` or `hp_tdpigd.json`, with a `.log` | `--approval`; inputs inside `data/`; output inside `results/` | `results/wp4/hp_tdpigd_checkpoint.json` (full run only), reused if the observed statistic is unchanged |
| `run_qualification.py` | WP3, method qualification (§8) | synthetic | options | `results/qualification/datasets/*.json`, `results/qualification/summary.json` | none | one JSON per dataset, skipped if present |
| `smoke_test.py` | installation check | synthetic | options | `results/smoke/summary.json` | output inside `results/` | none |
| `bench_threads.py` | compute plan (D14) | synthetic | `configs/wp0_sim_settings.json`; for `--fit`, the power pilot's dataset files | `results/bench/threads.json`, added to on every run | none | none |
| `md_to_docx.py` | the Word versions of the documents | none | a Markdown file | a `.docx` file | none | none |

**The guards** (`scripts/_common.py`, [§1](01_data.md)):
- `require_approval(reference, what)` stops the script unless `--approval` holds at least 8 characters after stripping spaces. It does not check the text, which is copied into the report as an audit trail.
- `require_inside(path, folder, what)` resolves the path and stops unless `folder` is one of its parents, so the path must lie strictly inside the folder (`--out results` itself is refused).
- `to_json(obj)` makes a report JSON-safe: numpy arrays become lists, numpy scalars numbers, and non-finite floats `null`.
- Importing `_common` also puts `src/` first on `sys.path`, which is why every script imports it before `pdx`.

Of this chapter's scripts, only `run_hp_real.py` reads participant data, and it has all three guards.

**Two ways of running pieces in parallel.**
- `parallel.ordered` (the gate and H-P, [§5](05_hp_speed_localisation.md)): `--jobs N` single-threaded worker processes started with "spawn", results returned in item order. There is no cap on N. With N = 1, everything runs in the main process with its threads.
- `qualification.run_tasks` (the power check and the qualification, [§6](06_simulation_gate.md)): a spawn pool of `min(jobs, 4)` processes, one simulated dataset per task, with the BLAS thread variables set to 1.

**Three ways of resuming.**
- The gate's checkpoint is keyed by the run's settings. A restart with other settings is refused.
- The H-P checkpoint is keyed by the observed statistic. It is reused if the statistic matches to 1e-6, and discarded otherwise.
- The power check and the qualification write one file per dataset, named by family, scenario, a hash of the configuration and the seed. A rerun skips files that exist.

**How the examples were run.** Every example is in [`docs/guide_examples/s07/`](../guide_examples/s07/) and uses synthetic or invented inputs only. The examples import a script as a module (after putting `scripts/` on `sys.path`) and call its functions; no `main()` that reads `data/` or writes `results/` was called. Files go to temporary folders. Run them from the repository folder:

```
OMP_NUM_THREADS=1 ../ppmi-causal/.venv/bin/python docs/guide_examples/s07/ex_power_summarise.py
/usr/bin/python3 docs/guide_examples/s07/ex_md_to_docx.py      # python-docx is not in the ppmi-causal environment
```

The environment was Python 3.13 with numpy 2.5.4, scipy 1.18.1, torch 2.14.1 and pandas 3.0.6; the Word example ran with the system Python 3.9 and python-docx. The unit tests were run once (§7.26). No toy was run end to end: each takes hours. Their building blocks were checked in seconds (`ex_toys.py`).

---

### 7.1 `run_engine_gate.py: code_version` — which code a gate report ran

[`run_engine_gate.py:27`](../../scripts/run_engine_gate.py#L27)

**What it does.** It runs three read-only git commands in the repository folder and returns:
- `commit`: the output of `git rev-parse HEAD`;
- `uncommitted_diff_sha256`: the SHA-256 hash of `git diff HEAD` (every change to tracked files, staged or not), or `None` if there is none;
- `untracked`: `git ls-files --others --exclude-standard`, the new files that `.gitignore` does not hide.

If git is not installed, `subprocess.run` raises `OSError` and all three are `None`.

**Inputs → outputs.** None → a dict, stored as the report's `"code"` entry.

**The mathematics.** None. The hash identifies the diff: two reports with the same commit and the same hash ran the same tracked files.

**Worked example** ([`ex_gate_settings.py`](../guide_examples/s07/ex_gate_settings.py), part 1):

```text
1. code_version(): commit a9b577f | a hash of uncommitted changes: True | untracked files listed: 16
```

The working tree had an edited `PROTOCOL.md`, a tracked file, so the diff is hashed. The untracked files are this guide's own new files, so their number grows as the guide is written (a later rerun listed 128).

**Design choices and pitfalls.**
- The hash covers every tracked file, documents included. An edited protocol changes the hash without changing any computation.
- Untracked files are listed by name only, so the content of a new, uncommitted module is not identified.
- In a copy that is not a git checkout but where git is installed, the commands fail without raising: `commit` is the empty string and the diff hash is `None`, which looks like a clean checkout.
- It runs once, when `main` builds the report. The dictionary display `{"code": code_version(), **run_gate(...)}` evaluates its entries left to right, so the code is recorded at the start of the session that finishes the run. Sections finished in earlier sessions, possibly under other code, are not distinguished (§7.N).

### 7.2 `run_engine_gate.py: main` (lines 41–67) — options, sizes and names

[`run_engine_gate.py:41`](../../scripts/run_engine_gate.py#L41)

**What it does.** It reads the options, sets the threads, checks the output folder, builds the simulation settings and names the run.

| Option | Default | Effect |
|---|---|---|
| `--n` | 1400 | size of the simulated pool; ignored with `--wp0` |
| `--d` | 4 | domains: 4, 6 or 10, family A's presets |
| `--datasets` | 20 | recovery datasets in each of S1 and S2 |
| `--seed` | 2026 | master seed; each section has its own random stream (`[seed, k]`, or `seed` plus an offset for the recovery datasets and the timing) |
| `--wp0` | off | WP0's group sizes, visits and missingness |
| `--quick` | off | small sizes, to test the harness |
| `--threads` | PyTorch's | `torch.set_num_threads` in the main process |
| `--jobs` | 1 | dispersed starts, profiles and recovery datasets run this many at once |
| `--out` | `results/engine_gate` | must lie inside `results/` |

With `--wp0`, the settings come from `run_power_check.wp0_sim_settings()` (§7.5):
- `target_counts` = (TD, PIGD) participants with at least one transition;
- `p_missing[:d]`, the first `d` of the ten missing shares;
- `visit_probs`, with its keys turned into integers;
- the pool `n` = 5 × (TD + PIGD).

The run is named `gate_wp0_d{d}` with `--wp0`, `gate_n{n}_d{d}` otherwise, and `gate_quick_wp0` or `gate_quick` with `--quick`.

`--quick` acts inside `gate.run_gate` (`gate.py` lines 329–336, 358, 380 and 385; [§6](06_simulation_gate.md)):
- at most 2 recovery datasets per scenario;
- with `--wp0`, the target counts rescaled to 200 participants in the same proportion, and the pool 5 times their sum; without it, a pool of at most 200;
- 2 dispersed starts instead of 5, and profiles on 5 grid points instead of 9;
- no interval coverage in the null recovery.

**Inputs → outputs.** The command line → the pool size `n`, the simulation keywords `sim_kw`, and the file names.

**The mathematics.** The quick rescaling, with Python's `round`:

$$n_g^{quick}=\max\Big\{2,\ \mathrm{round}\Big(\frac{200\,n_g}{n_{TD}+n_{PIGD}}\Big)\Big\},\qquad n^{quick}=5\,(n^{quick}_{TD}+n^{quick}_{PIGD}).$$

The pool is 5 times the target because `family_a.simulate` simulates `n` participants from year 0, drops those the phenotype rule cannot classify and those with fewer than two visits, and only then draws the target counts without replacement. If a phenotype falls short of its target, it raises.

**Worked example** ([`ex_gate_settings.py`](../guide_examples/s07/ex_gate_settings.py), parts 2–3):

```python
pairs = list(zip(default_truth(d).domains, NAMES10[:d]))         # simulated domain <- WP0 share it receives
quick = tuple(max(2, round(200 * c / sum(tc))) for c in tc)       # gate.run_gate, lines 331-334
```

```text
2. which WP0 missing share each simulated domain receives with --wp0 (p_missing[:d])
   d =  4: mismatches ['function <- brady_rigidity', 'independence <- function']
   d =  6: mismatches none
   d = 10: mismatches none

3. --quick --wp0 with invented counts (600, 150): target_counts (160, 40), pool n = 1000
```

By hand: 200 × 600/750 = 160 and 200 × 150/750 = 40, so the pool is 5 × 200 = 1,000.

The ten missing shares are stored in the order tremor, axial, brady_rigidity, function, independence, cognition, mood, autonomic, hallucinations, sleepiness (§7.4). The 6- and 10-domain presets list their domains in the same order. The 4-domain preset is tremor, axial, function, independence, so at `--d 4` daily function receives bradykinesia's share and independence receives function's.

**Design choices and pitfalls.**
- `--d 4` is the default, so `run_engine_gate.py --wp0` without `--d` hits the mismatch above. The runbook's command uses `--d 10`, which is unaffected (§7.N).
- `--n` is silently ignored with `--wp0`, and neither run name records `n`. The report's `settings` block does.
- The quick names do not contain `d`. A quick run with another `--d` is refused by the checkpoint until the partial file is moved, and a finished quick report is overwritten by the next one.

### 7.3 `run_engine_gate.py: main` (lines 69–90) — log, run, resume and report

[`run_engine_gate.py:69`](../../scripts/run_engine_gate.py#L69)

**What it does.**
1. `log(message)` prefixes a time stamp (`YYYY-MM-DD HH:MM`), prints the line at once, and appends it to `<out>/<name>.log` in UTF-8. Every session appends; the log is never cleared.
2. It calls `run_gate(n, d, datasets, seed, quick, sim_kw=…, checkpoint=<name>.partial.json, log=log, jobs=…)` ([§6](06_simulation_gate.md)). The checkpoint (`gate.Checkpoint`) saves every finished section, and inside the long sections every dispersed-start fit, profile and recovery dataset. A restart with identical settings skips them; other settings raise `RuntimeError`.
3. Ctrl+C (`KeyboardInterrupt`) and any other exception are logged with "run the same command to resume" (and the traceback for an error), then raised again.
4. On success it writes `<name>.json` (`to_json(report)`, indented), deletes the partial file, logs whether all automatic checks pass, and prints the `all_automatic_checks_pass` flag and the `timing` block.

**Inputs → outputs.** The settings of §7.2 → the report, with the keys `code`, `settings`, `reference`, `gradient`, `participants`, `quadrature`, `dispersed_starts`, `identification`, `profiles`, `recovery_null`, `recovery_changed_link`, `timing`, `all_automatic_checks_pass` and `note`.

**The mathematics.** None of the script's own. The pass rules are `gate.py`'s ([§6](06_simulation_gate.md)): reference agreement below 1e-8, gradients below 1e-4, a quadrature spread below 0.01, a dispersed-start spread below 0.1, a condition number below 1e10 with spreads at zero set aside, profiles that fall by at least 1.92 on both sides (half the χ²₁ 95% point, 3.84), and no likelihood ratio below −1e-6.

**Worked example.** The gate itself was not run: the quick gate takes 15–25 minutes (runbook §1). The checkpoint's refusal was run with a stub section ([`ex_gate_settings.py`](../guide_examples/s07/ex_gate_settings.py), part 4):

```text
4. a quick run with --d 4 saved: ['reference: done (0 min)']
   the same command with --d 10 is refused: gate_quick.partial.json holds a gate run with other settings; move it away to start afresh
```

**Design choices and pitfalls.**
- The checkpoint's settings are `n`, `d`, `n_datasets`, `seed`, `quick`, `sim_kw` and `fit_kw`. `jobs` and `threads` are left out on purpose, so a restart may change them (runbook §4). One job runs in the main process with its threads, while several jobs run single-threaded, and the two agree only to the fits' numerical tolerance. A report that mixed both across sessions is therefore not reproducible bit for bit by either.
- The partial file is the only state. If it is deleted, the gate starts again from the beginning.
- `code_version` describes only the last session (§7.1).

---

### 7.4 `run_power_check.py: wp0_settings` — simulation settings from the WP0 audit

[`run_power_check.py:30`](../../scripts/run_power_check.py#L30)

**What it does.** It reads the WP0 audit's JSON, made of aggregate tables ([§1](01_data.md)), and returns three settings for family A's simulator:
- `target_counts` = (TD, PIGD): for each phenotype, the participants reaching the year-1 landmark (`landmark.cells[ph]["reached year 1"]`), times the share with more than one visit from the landmark on, rounded;
- `visit_probs`: the audit's `visits_per_participant` table (participants by number of visits from the landmark on), keeping 2 visits or more, with suppressed cells (strings) entered as 2;
- `p_missing`: the overall missing share of ten domains, rounded to 4 decimals, in the simulator's order: tremor, axial, brady_rigidity, function, independence, cognition, mood, autonomic, hallucinations, sleepiness.

**Inputs → outputs.** The audit's path → a dict. The keys of `visit_probs` stay strings.

**The mathematics.** Let $v_k$ be the number of participants with exactly $k$ visits from the landmark on (a suppressed cell counts as 2), and $L_g$ the number of phenotype $g$ reaching the landmark. The code computes

$$\pi_1=\frac{v_1}{\sum_k v_k},\qquad n_g=\mathrm{round}\big(L_g\,(1-\pi_1)\big),$$

the participants expected to have at least one transition, assuming the share with a single visit is the same in both phenotypes. `family_a.simulate` later turns the counts into probabilities: a simulated participant has $T=k$ visits with probability $v_k/\sum_{j\ge2}v_j$.

**Worked example** ([`ex_power_summarise.py`](../guide_examples/s07/ex_power_summarise.py), part 1), on an invented audit: 500 TD and 120 PIGD participants reach the landmark, and the visits table is {1: 30, 2: 50, 3: "<5", 4: 200, 5: 300}.

```text
1. wp0_settings on an invented audit
   target_counts: (474, 114)
   visit_probs  : {'2': 50, '3': 2, '4': 200, '5': 300}
   p_missing    : [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
   by hand: share with one visit = 30/582 = 0.05155; TD 500 x 0.94845 = 474.23; PIGD 120 x 0.94845 = 113.81
```

The suppressed cell "<5" became 2, so the denominator is 30 + 50 + 2 + 200 + 300 = 582. The audit's `rbd` share is ignored, because only the ten simulator domains are read.

**Design choices and pitfalls.**
- The share with a single visit is pooled over phenotypes, because the audit's visits table is not split by phenotype.
- A suppressed cell (1 to 4 participants) is entered as 2, a guess inside the suppression range.
- A key that is not an integer (such as "15+") would make `int(k)` fail. A domain missing from `missing_share` raises `KeyError`, and a suppressed overall share (a string) would make `float()` fail.
- The audit's overall share is computed over every row of its visit table ([§1](01_data.md) says which years that table holds). The simulator applies it to every simulated visit from the landmark on.
- The list is positional and in the 10-domain simulator's order, not the panel's (`ppmi.CORE_DOMAINS` puts brady_rigidity second). At `d` < 10 the first `d` entries are used (§7.2).

### 7.5 `SETTINGS`, `wp0_sim_settings`, `write_sim_settings` — where the settings live

[`run_power_check.py:48`](../../scripts/run_power_check.py#L48), [`run_power_check.py:51`](../../scripts/run_power_check.py#L51), [`run_power_check.py:60`](../../scripts/run_power_check.py#L60)

**What it does.**
- `wp0_sim_settings()` reads `configs/wp0_sim_settings.json` if it exists, so that any computer with the repository can run the synthetic gate, power check and benchmark. Otherwise it derives the settings from `results/wp0/wp0_audit.json` with `wp0_settings`. It returns `target_counts` as a tuple.
- `write_sim_settings()`, run by `--write-settings`, writes the file from the audit, with an explanatory note.

**The file's structure** (its numbers are WP0 aggregates and are not quoted here):
- `note`: a sentence saying what the numbers are and how to refresh them;
- `target_counts`: two integers, TD then PIGD;
- `visit_probs`: a dictionary from a number of visits to a count of participants, one entry per observed number of visits, small cells entered as 2;
- `p_missing`: ten shares, in the simulator's order of §7.4.

**Inputs → outputs.** None → the settings dict (`wp0_sim_settings`); the audit → the file (`write_sim_settings`).

**The mathematics.** None.

**Worked example.** None: the only real input is the committed file, whose numbers are not reproduced in this guide.

**Design choices and pitfalls.**
- The file wins over the audit even when the audit is newer. After a new WP0 run, `--write-settings` must be run, or every synthetic run keeps the old settings.
- The note is fixed text that names the audit of 3 November 2026, so a refresh from a later audit keeps the old date (§7.N).
- The file is in git. It holds aggregates only, but they derive from PPMI, so the data-governance checklist (runbook §8) applies to it.

### 7.6 `run_power_check.py: summarise` — power from a few datasets

[`run_power_check.py:69`](../../scripts/run_power_check.py#L69)

**What it does.** It keeps the S2 rows without an error, one row per simulated dataset with its statistic `lr` and degrees of freedom `df`, and computes:
1. each dataset's χ² critical value at its own df, and whether its statistic exceeds it; the share is the Monte Carlo power, with its Monte Carlo standard error (MCSE);
2. the non-centrality, estimated as the mean of LR − df (floored at 0), with its standard error;
3. the noncentral χ² power at that non-centrality, averaged over the datasets' df, and at the non-centrality ± 1.96 standard errors;
4. if S1 rows carry a bootstrap statistic (`--null`), the warp-speed bootstrap critical value and the share of S2 statistics above it;
5. the verdict: "pass" if the lower end of the noncentral range is at least 0.8, "fail" if its upper end is below 0.8, and "undecided: run more datasets" otherwise.

It also returns the number of datasets, `failed` (rows with an error, of any scenario), the count of datasets at each df, the χ² critical value at each df, and the median seconds per dataset.

**Inputs → outputs.** A list of row dicts (`scenario`, `lr`, `df`, `seconds`, and `lr_star` or `error`) → a summary dict, or `{"datasets": 0}` if no S2 row succeeded.

**The mathematics.**

*Why the df vary.* Each dataset re-selects its support ([§4](04_step1_comparators.md)): the expert map, plus the lagged links Step 1 finds, plus every own rate. So $df_i=(G-1)(n_{sup,i}-1)=n_{sup,i}-1$ differs between datasets. At $d=10$ the simulator's map has 12 links and 10 own rates, so $df_i\ge21$.

*(a) Monte Carlo power.* With $R$ datasets and $\alpha=0.05$:

$$c_i=F^{-1}_{\chi^2_{df_i}}(1-\alpha),\qquad \hat\pi_{MC}=\frac1R\sum_{i=1}^R\mathbf 1[LR_i>c_i],\qquad \mathrm{MCSE}=\sqrt{\hat\pi_{MC}(1-\hat\pi_{MC})/R}.$$

In the code: `crit`, `rej`, `power_mc_chi2` and `mcse_mc`.

*(b) The non-centrality.* Under a local alternative the likelihood ratio is approximately noncentral χ², with $df_i$ degrees of freedom and non-centrality $\lambda$, so $E[LR_i]=df_i+\lambda$ and $\mathrm{Var}(LR_i)=2(df_i+2\lambda)$. The deviation of 2ε is the same in every dataset; the extra links Step 1 adds are null, so they add degrees of freedom but no non-centrality. If $\lambda$ is common, the method-of-moments estimate is

$$\hat\lambda=\max\Big(0,\ \frac1R\sum_i(LR_i-df_i)\Big),\qquad \widehat{SE}(\hat\lambda)=\frac{s_{LR-df}}{\sqrt R},$$

with $s$ the standard deviation (`ddof=1`). In the code: `lam` and `se_lam`.

*(c) Noncentral power*, averaged over the datasets' df:

$$\hat\pi_{NC}(\lambda)=\frac1R\sum_i\Big[1-F_{\chi^2_{df_i}(\lambda)}(c_i)\Big].$$

In the code: `power_nc = lambda l: mean(ncx2.sf(crit, df, max(l, 1e-9)))`. The floor of 1e-9 keeps scipy's noncentral χ² away from λ = 0, where the power is α.

*(d) The range.* $\hat\pi_{NC}$ increases with $\lambda$, so $[\hat\pi_{NC}(\hat\lambda-1.96\,\widehat{SE}),\ \hat\pi_{NC}(\hat\lambda+1.96\,\widehat{SE})]$ is the image of a 95% Wald interval for $\lambda$.

*(e) The verdict.* "pass" ⇔ $\hat\pi_{NC}(\hat\lambda-1.96\,\widehat{SE})\ge0.8$, a one-sided 97.5% statement; "fail" ⇔ $\hat\pi_{NC}(\hat\lambda+1.96\,\widehat{SE})<0.8$.

*(f) The bootstrap critical value* (warp-speed, protocol §8.6). Each S1 dataset run with `--null` carries one bootstrap statistic $T^*$. With $R_0$ of them, failed replicates entered as $+\infty$, the critical value is the $\lceil0.95R_0\rceil$-th smallest (`qualification._rank_quantile`), $c^*=T^*_{(\lceil0.95R_0\rceil)}$, and $\hat\pi_{boot}=\frac1R\sum_i\mathbf 1[LR_i>c^*]$.

*(g) What the optional 20-dataset check can show.* The runbook (§2) says that 20 datasets that all reject would confirm 80% power from the rejection rate alone. The code does not compute a bound: with every dataset rejecting, its MCSE is 0. The exact one-sided lower confidence bound at level $1-a$, when all $R$ datasets reject, solves $\pi^R=a$:

$$\pi_L=a^{1/R}=0.05^{1/20}=0.861\ \ge0.8 .$$

With 19 rejections out of 20, the Clopper–Pearson bound is $\mathrm{Beta}^{-1}(0.05;\,19,\,2)=0.784<0.8$. So only 20 rejections out of 20 confirm 80% power.

**Worked example** ([`ex_power_summarise.py`](../guide_examples/s07/ex_power_summarise.py), parts 2–4). Eight invented S2 rows with df 13, 13, 14, 13, 15, 14, 13, 13 and statistics 38.2, 25.0, 31.5, 19.8, 44.1, 28.7, 33.0, 22.4; one failed S2 row; three invented S1 rows with bootstrap statistics 12.1, 18.5 and 9.7.

```text
2. summarise on 8 invented S2 rows, 1 failed row and 3 invented S1 (null) rows
   datasets: 8
   failed: 1
   df_counts: {'13': 5, '14': 2, '15': 1}
   chi2_critical: {'13': 22.362032494826938, '14': 23.68479130484058, '15': 24.99579013972862}
   power_mc_chi2: 0.875
   mcse_mc: 0.11692679333668567
   noncentrality: 16.8375
   noncentrality_se: 2.7215369541597525
   power_noncentral: 0.7628258507368402
   power_noncentral_range: [0.5529250656302324, 0.8907136963937718]
   median_seconds: 3810.0
   bootstrap_critical: 18.5
   power_mc_bootstrap: 1.0
   gate_80_percent: undecided: run more datasets
   by hand: critical values {13: 22.362, 14: 23.685, 15: 24.996}
   by hand: rejections [1, 1, 1, 0, 1, 1, 1, 1]
   by hand: LR - df = [25.2, 12.0, 17.5, 6.8, 29.1, 14.7, 20.0, 9.4], mean 16.8375, se 2.7215
   by hand: noncentral power for one dataset with df 13 at lambda-hat: 0.7701
```

By hand:
- Only the fourth statistic, 19.8, stays below its critical value of 22.36, so $\hat\pi_{MC}=7/8=0.875$, with MCSE $\sqrt{0.875\times0.125/8}=0.117$. The last statistic, 22.4, rejects by 0.04.
- The deviations LR − df sum to 134.7, so $\hat\lambda=134.7/8=16.84$, with standard error 2.72.
- At $\hat\lambda$, one dataset with 13 df has noncentral power 0.770; averaged over the eight datasets' df it is 0.763.
- The range [0.553, 0.891] straddles 0.8, so the verdict is "undecided".
- With three null rows, the bootstrap critical value is the $\lceil0.95\times3\rceil=3$rd smallest, the largest (18.5). Every S2 statistic exceeds it.

The two power estimates disagree here (0.875 and 0.763). With eight datasets, the rejection count uses only which side of the critical value each statistic falls, while the noncentral estimate uses their sizes but assumes the noncentral χ² shape.

```text
3. the optional 20-dataset check: exact (Clopper-Pearson) one-sided 95% lower bounds
   20 of 20 reject: lower bound 0.8609
   19 of 20 reject: lower bound 0.7839
   18 of 20 reject: lower bound 0.7174
   summarise on 20 rows that all reject: power_mc_chi2 = 1.0 mcse_mc = 0.0
   chance that all 20 reject when the true power is 0.80: 0.0115
   chance that all 20 reject when the true power is 0.90: 0.1216
   chance that all 20 reject when the true power is 0.95: 0.3585
   chance that all 20 reject when the true power is 0.99: 0.8179

4. edge cases
   one S2 row: {'noncentrality_se': nan, 'power_noncentral_range': [nan, nan], 'gate_80_percent': 'undecided: run more datasets'}
   statistics below their df (lambda-hat floored at 0): {'noncentrality': 0.0, 'power_noncentral': 0.05}
   no S2 row: {'datasets': 0}
```

Even with a true power of 95%, all 20 datasets reject only 36% of the time. The 20-dataset check can confirm 80% power only when the power is very high.

**Design choices and pitfalls.**
- The non-centrality method assumes one noncentral χ² shape with a common λ. The S1 rows (with `--null`) hold `lr` and `df`, and their mean of LR − df should be near 0; the code does not report it, and that would check the assumption.
- Power is computed against χ² critical values, not against H-P's bootstrap. If H-P's null distribution is wider than χ² (for example, because of the support selection), the power of the protocol's test is lower. `--null` adds the bootstrap critical value, but it pools statistics with different df into one critical value.
- Failed S2 datasets are dropped, so the power is conditional on a successful fit, and `failed` counts failures of every scenario.
- `main` pools every JSON in the datasets folder, whatever its configuration (§7.7, §7.N).

### 7.7 `run_power_check.py: main` — the pilot and the optional check

[`run_power_check.py:101`](../../scripts/run_power_check.py#L101)

**What it does.**

| Option | Default | Effect |
|---|---|---|
| `--datasets` | 8 | S2 datasets (one deviation of 2ε) |
| `--null` | 0 | S1 datasets, each with one bootstrap replicate |
| `--d` | 10 | domains |
| `--jobs` | 4 | datasets at once (at most 4, §7.0) |
| `--summarise` | off | skip the runs; summarise the files present |
| `--write-settings` | off | write `configs/wp0_sim_settings.json` from the audit and stop |

Unless `--summarise` is given, it builds the tasks and runs them with `qualification.run_tasks`, one JSON file per dataset in `results/wp0/power/datasets/`:
- S2 tasks: `make_tasks("S2", datasets, 5 × (TD + PIGD), "A", d=d, step1=True, sim_kw=…, replicate=False)`. Each dataset is simulated at WP0's counts, visits and missingness, re-selects its support, and gets both fits and its statistic, with no bootstrap replicate.
- S1 tasks, with `--null`: the same with `replicate=True`, so each also gets one bootstrap replicate.

It then reads every JSON in the folder, writes `results/wp0/power/summary.json` (the WP0 settings plus `summarise`), and prints the summary without the settings.

**Inputs → outputs.** Options → dataset files, the summary file and the printed summary.

**The mathematics.** Each task is `qualification.dataset_task` ([§6](06_simulation_gate.md)): simulate, select the support with Step 1's default configuration (α = 0.05, separating sets of at most 3, Fisher z), fit the speed-only and free models, and record LR and df. `summarise` (§7.6) does the rest.

**Worked example.** No run (a pilot takes about 9 hours, runbook §2). The file names follow from `make_tasks` ([§6](06_simulation_gate.md)): family A's seeds are 1,000,000 + 10,000 × (the scenario's place in the list S0, S0L, S1, S1S, S1T, S1V, S4, S7, S2, S2b, S3) + r. So the S2 datasets are `A_S2_<configuration hash>_seed1080000.json`, `…seed1080001.json`, and so on, and the S1 null datasets start at seed 1,020,000. A rerun with `--datasets 20` after the pilot of 8 finds seeds 1,080,000–1,080,007 done and runs the other 12.

**Design choices and pitfalls.**
- Every JSON in the folder is pooled, whatever its `d`, sizes or WP0 settings. Unlike `run_qualification.py`, which groups by configuration key, a run after `--write-settings` or with another `--d` is silently pooled with the old files (§7.N).
- `--jobs` above 4 is silently cut to 4 by `run_tasks`.
- `--summarise` on a computer with no runs fails when it writes `summary.json`: the folder `results/wp0/power/` is created only by a run.
- The S1 and S2 tasks differ in `replicate`, so their configuration keys differ. `summarise` ignores the keys, so this has no effect here.

---

### 7.8 `run_hp_real.py`: the docstring and `FIRST_LOOK` — what the PPMI run promises

[`run_hp_real.py:1`](../../scripts/run_hp_real.py#L1), [`run_hp_real.py:40`](../../scripts/run_hp_real.py#L40)

**What it does.** The docstring is the contract of the only script in this chapter that reads participant data:
- it refuses to run without the investigator's written approval, reads only from `data/`, and writes aggregates only to `results/`;
- `--first-look` stops after the observed fits: H-P's statistic with a χ² p-value instead of the bootstrap's, the speed estimates with a model-based interval for Sp-1, and the per-link localisation. It is exploratory, not the registered analysis;
- the panel and its specification come from `build_analysis_panel.py` ([§1](01_data.md)), and the expert map from a JSON file of links by domain name;
- the support is selected on this panel (TD and PIGD participants), so every bootstrap replicate repeats exactly the same selection. Step 1's reported map, from all participants, is a separate run;
- the bootstrap is checkpointed after every replicate and resumes if the observed statistic is unchanged.

`FIRST_LOOK` is the sentence written as the first-look report's `status`. It says that the run came before WP1–WP3 and registration, that the p-value is the χ² approximation, and that the Sp-1 interval is model-based.

**Inputs → outputs.** None (constants).

**The mathematics.** None.

**Worked example.** None.

**Design choices and pitfalls.**
- Selecting the support on the H-P panel, not on all participants, is what makes the bootstrap valid: each replicate is simulated for the same participants, so it can repeat the same selection rule on the same population (protocol §7.3.1, step 3).
- The labels are part of the output, so a reader of `hp_first_look.json` sees the caveats in its first line.

### 7.9 `run_hp_real.py: expert_support` — the frozen map as a support matrix

[`run_hp_real.py:45`](../../scripts/run_hp_real.py#L45)

**What it does.** It reads `{"links": [[from, to], ...]}` and sorts every link into one of three cases:
- a name not in `wp1.DISPLAY` (the twelve known domains and "treatment") stops the run, naming it;
- both names among the panel's domains: the link enters the support, `sup[to, from] = True`;
- otherwise (a treatment link, or a known domain the panel leaves out, D26): the link is set aside, returned and reported.

**Inputs → outputs.** The map's path and the panel's domain names (`panel_spec.json`) → a $d\times d$ boolean matrix without the diagonal, and the list of links set aside.

**The mathematics.** $\mathcal S_{jk}=1$ if and only if the map has the link $k\to j$. The matrix is stored [to, from], like the drift $A_{jk}$, the effect of domain $k$ on domain $j$.

**Worked example** ([`ex_hp_real_helpers.py`](../guide_examples/s07/ex_hp_real_helpers.py), part 1). An invented map for a four-domain panel, with a treatment link, two domains the panel lacks, and then a typo:

```text
1. expert_support
   support (rows = to, columns = from):
         tremor [0, 0, 0, 0]
          axial [0, 0, 0, 0]
       function [1, 1, 0, 0]
   independence [0, 0, 1, 0]
   set aside: [['treatment', 'axial'], ['mood', 'cognition']]
   a typo stops the run: Unknown domain in the expert map: fuction
```

The three panel links appear in the rows of their targets. "treatment" and "mood → cognition" are known names, so they are set aside, not refused.

**Design choices and pitfalls.**
- The diagonal stays empty; `ModelSpec` adds the own rates.
- Names must be the code's names (`axial`), not the questionnaire's labels ("Axial and gait"), which would stop the run.
- A duplicate link changes nothing, and a self-link only sets an own rate that is always in the support anyway.
- The map's `status` is read separately, by `main`.

### 7.10 `run_hp_real.py: write_atomic` — a checkpoint that is never half-written

[`run_hp_real.py:61`](../../scripts/run_hp_real.py#L61)

**What it does.** It writes the text to `<path>.tmp` and renames that file over the target with `Path.replace`, which is atomic on one file system: a reader sees the old checkpoint or the new one, never a mixture. Windows refuses the rename while another program (an editor, an antivirus scanner) holds the file open, so a `PermissionError` is retried up to 20 times, half a second apart, before it is raised.

**Inputs → outputs.** A path and a string → the file written.

**The mathematics.** None.

**Worked example** ([`ex_hp_real_helpers.py`](../guide_examples/s07/ex_hp_real_helpers.py), part 4). A checkpoint with one finished replicate and one failed one:

```text
4. write_atomic: files in the folder: ['bad.json', 'hp_tdpigd_checkpoint.json', 'map.json']
   read back: {'lr': 31.25, 'replicates': [[12.5, True], [None, False]]}
   as run_hp_real rebuilds it for resume: [(12.5, True), (inf, False)]
```

No `.tmp` file is left after a successful rename. A failed replicate is stored as `null` and read back as $+\infty$, which counts as at least as large as the observed statistic (§7.15).

**Design choices and pitfalls.**
- `write_text` uses the platform's default encoding. That is harmless here, because `json.dumps` writes ASCII by default.
- After 20 refusals (10 seconds) the error is raised and the run stops, but the previous checkpoint is intact.

### 7.11 `run_hp_real.py: name_links` and `named_support` — indices to names

[`run_hp_real.py:76`](../../scripts/run_hp_real.py#L76), [`run_hp_real.py:84`](../../scripts/run_hp_real.py#L84)

**What it does.**
- `name_links(classes, names)` replaces each verdict's `link`, a (to, from) pair of indices, by `[from name, to name]`. It changes the dicts in place and returns the same list.
- `named_support(support, expert, names)` lists the support's links between different domains as `{"from", "to", "source"}`. The source is "map" if the expert map has the link and "Step 1" otherwise.

**Inputs → outputs.** Verdicts or a support matrix, and the domain names → lists of dicts.

**The mathematics.** None.

**Worked example** ([`ex_hp_real_helpers.py`](../guide_examples/s07/ex_hp_real_helpers.py), parts 2–3). Step 1 is pretended to have added axial → independence:

```text
2. named_support
   {'from': 'tremor', 'to': 'function', 'source': 'map'}
   {'from': 'axial', 'to': 'function', 'source': 'map'}
   {'from': 'axial', 'to': 'independence', 'source': 'Step 1'}
   {'from': 'function', 'to': 'independence', 'source': 'map'}

3. name_links: [{'link': ['tremor', 'function'], 'verdict': 'similar'}, {'link': ['axial', 'independence'], 'verdict': 'structurally different'}]
   the input list was changed in place: {'link': ['tremor', 'function'], 'verdict': 'similar'}
```

**Design choices and pitfalls.**
- The links come out in row-major order of the [to, from] matrix, grouped by target.
- Because `name_links` changes its input, the index form of the verdicts is gone after the call.

### 7.12 `run_hp_real.py: sp1_interval` — a delta-method interval for the speed ratio

[`run_hp_real.py:90`](../../scripts/run_hp_real.py#L90)

**What it does.** It computes a model-based interval for Sp-1 from the speed-only fit, for the first look only. The protocol's interval comes from the participant bootstrap (protocol §2.5; `speed.bootstrap_speed`, [§5](05_hp_speed_localisation.md)).

**Inputs → outputs.** The speed-only fit, the panel and the level (0.95, `[speed] level`) → `[lower, upper]`.

**The mathematics.** Sp-1, the speed ratio ($\hat s_0$ in the notation of [§0.4](00_map.md#04-shared-notation)), is computed by `speed.sp1` ([§5](05_hp_speed_localisation.md)) as

$$\log\hat s_0=\hat\gamma+\hat\theta^\top\delta,\qquad \delta=\bar{\tilde y}_{1,\mathrm{PIGD}}-\bar{\tilde y}_{1,\mathrm{TD}},$$

where $\gamma$ is PIGD's log-speed offset, $\theta$ the effect of the centred year-1 state $\tilde y_{i1}$ on log-speed, and $\delta$ the difference between the phenotypes' mean year-1 states. The term $\beta^\top z$ drops out because $z$ is centred on the pooled population: both phenotypes are compared at the same covariates, each at its own mean year-1 state.

With $\delta$ treated as fixed, $\log\hat s_0$ is linear in the raw parameter vector $x$, with gradient $a=e_\gamma+(0,\dots,0,\delta,0,\dots,0)$: a 1 at $\gamma$'s position and $\delta$ in $\theta$'s positions. With $V$ the inverse of the observed information,

$$\widehat{\mathrm{Var}}(\log\hat s_0)=a^\top Va=V_{\gamma\gamma}+2\,\delta^\top V_{\theta\gamma}+\delta^\top V_{\theta\theta}\,\delta,$$

$$\text{interval}=\Big[\exp\big(\log\hat s_0-z\,\widehat{se}\big),\ \exp\big(\log\hat s_0+z\,\widehat{se}\big)\Big],\qquad z=\Phi^{-1}\Big(\tfrac12+\tfrac{\text{level}}2\Big).$$

The code line by line:
- `V = covariance_checked(hessian(f0, panel))[0]` is the inverse of the observed information, from central differences of the exact gradient ([§3](03_engine_fit.md));
- `a[lay.slices["gamma"][0].start] = 1.0` is $e_\gamma$;
- `a[lay.slices["theta"][0]] = y1c[g == 1].mean(0) - y1c[g == 0].mean(0)` puts $\delta$ in $\theta$'s positions;
- `max(a @ V @ a, 0.0)` guards against a negative quadratic form.

The interval is symmetric on the log scale, so on the ratio scale it is asymmetric and always positive.

**Worked example** ([`ex_first_look_synthetic.py`](../guide_examples/s07/ex_first_look_synthetic.py), part 6). Family A's S1 scenario, 150 simulated participants (135 kept by the phenotype rule), four domains, true speed ratio 2:

```text
6. by hand: gamma = 0.7348  theta = [0.0365, 0.1906, -0.1251, -0.1466]  mean year-1 difference = [-0.5717, 0.7518, 0.1683, -0.0731]
   log Sp-1 = 0.8468; Var = V_gg + 2 d'V_tg + d'V_tt d = 0.06251 + -0.02139 + 0.01124 = 0.05236
   interval exp(log Sp-1 -+ 1.96 se) = [1.4894, 3.6522]
   conditioning report that sp1_interval does not pass on: {'min_eigenvalue': 4.6805353822552303e-08, 'condition_number': 187615946077.25204, 'positive_definite': True, 'well_conditioned': False}
```

By hand:
- $\theta^\top\delta=0.0365(-0.5717)+0.1906(0.7518)-0.1251(0.1683)-0.1466(-0.0731)=0.1120$, so $\log\hat s_0=0.7348+0.1120=0.8468$ and $\hat s_0=2.33$.
- $\widehat{se}=\sqrt{0.05236}=0.2288$, so the interval is $\exp(0.8468\mp1.96\times0.2288)=[1.49,\ 3.65]$. It agrees with `sp1_interval`'s own output (§7.13) and covers the true ratio of 2.
- The correction for the year-1 state is not small: without it, $e^{0.7348}=2.08$.

The observed information here has a condition number of 1.9e11, above `covariance_checked`'s threshold of 1e10, so the covariance is flagged as not well conditioned. `sp1_interval` keeps only the matrix and drops that flag.

**Design choices and pitfalls.**
- The conditioning report is dropped, although `covariance_checked`'s docstring says it must travel with any interval computed from it. The example shows a flagged covariance producing an unflagged interval (§7.N).
- $\delta$ is treated as fixed, so the interval is conditional on the landmark states. It ignores the sampling variation of the year-1 means, which the participant bootstrap includes.
- If the information is not positive definite, `covariance_checked` uses a pseudo-inverse that drops the non-positive directions, which understates the standard error.

### 7.13 `run_hp_real.py: first_look` — H-P's observed fits, without the bootstrap

[`run_hp_real.py:105`](../../scripts/run_hp_real.py#L105)

**What it does.**
1. `hp.observed_fits(panel, spec, support_fn)`: support selection (Step 1's fast configuration around the map) and the speed-only and free fits ([§5](05_hp_speed_localisation.md)).
2. H-P's statistic and its χ² p-value. The `hp` block also holds the raw statistic, the χ² critical value at α, both log-likelihoods, both convergence flags, and the free fit's log-likelihood from each of its two starts.
3. The support by name, with each link's source (§7.11).
4. The speed estimates Sp-1, Sp-2 and Sp-3 from the same fits, plus the model-based Sp-1 interval (§7.12).
5. The exploratory localisation: `localise.per_link_tests` on the free fit (`[localisation] mc_draws` Monte Carlo draws, `near_zero`), `classify_links` with ε and Benjamini–Yekutieli (with the links named), `speed_and_load_only`, the speed set, the speed diagnostics and the per-domain tests ([§5](05_hp_speed_localisation.md)).

It logs each stage with its time.

**Inputs → outputs.** The panel, the model specification, the support rule, the map matrix, the domain names, the configuration, a random generator and the log function → a dict with the keys `hp`, `support`, `speed`, `links_exploratory`, `speed_and_load_only`, `speed_set`, `speed_diagnostics` and `domains`.

**The mathematics.**

$$\mathrm{LR}=\max\{0,\ 2(\hat\ell_{free}-\hat\ell_{null})\},\qquad df=(G-1)(n_{sup}-1),\qquad p_{\chi^2}=1-F_{\chi^2_{df}}(\mathrm{LR}).$$

The χ² p-value is only a first look. The protocol's reference distribution is the conditional bootstrap with the support re-selected in every replicate (protocol §7.3.1), because the support was chosen on the same data and the real scores add bounded scales, dropout and measurement error. In the synthetic checks the χ² reference held at 6.0–6.5% false rejections (protocol §7.3.1).

**Worked example** ([`ex_first_look_synthetic.py`](../guide_examples/s07/ex_first_look_synthetic.py), parts 1–5), on the synthetic S1 panel of §7.12. Two shortcuts keep it to under two minutes without changing any number: the observed fits and the speed-only Hessian are computed once and handed back when `first_look` asks for them, and the localisation uses 500 Monte Carlo draws instead of the configured 2,000. That is why the log reports 0 minutes for the fits.

```text
synthetic panel: 135 participants, groups [81, 54], true speed ratio 2.0
   log: support selection and the two observed fits: started (about an hour)
   log: fits done (0 min): LR 7.13 on 7 df, chi-square p = 0.416
   log: speed estimates done: Sp-1 = 2.33 (model-based 95% interval 1.49-3.65)
   log: localisation done (0 min in all)

1. report keys: ['hp', 'support', 'speed', 'links_exploratory', 'speed_and_load_only', 'speed_set', 'speed_diagnostics', 'domains']
2. hp: {'lr': 7.1253, 'lr_raw': 7.1253, 'df': 7, 'p_chi2': 0.4159, 'chi2_critical': 14.0671, 'null_loglik': -934.6948, 'free_loglik': -931.1322, 'null_converged': True, 'free_converged': True, 'free_start_logliks': [-931.6056511568056, -931.1321856958295]}
3. support: [{'from': 'tremor', 'to': 'function', 'source': 'map'}, {'from': 'axial', 'to': 'function', 'source': 'map'}, {'from': 'axial', 'to': 'independence', 'source': 'map'}, {'from': 'function', 'to': 'independence', 'source': 'map'}]
4. Sp-1 = 2.3323, model-based 95% interval [1.4894, 3.6522]
   Sp-2 sigma: [0.2858, 0.3996]  Sp-3 share: 0.4798
5. link verdicts: [(['tremor', 'tremor'], 'outside the family'), (['axial', 'axial'], 'outside the family'), (['tremor', 'function'], 'unresolved'), (['axial', 'function'], 'unresolved'), (['function', 'function'], 'outside the family'), (['axial', 'independence'], 'unresolved'), (['function', 'independence'], 'unresolved'), (['independence', 'independence'], 'outside the family')]
   speed and load only: False  domain verdicts: ['no difference detected', 'no difference detected', 'no difference detected', 'no difference detected']
```

Reading the output:
- The support has 4 own rates and 4 links, so $df=8-1=7$, and $2(-931.1322+934.6948)=7.125$. Its χ² p-value is 0.42, consistent with the shared mechanism the data were simulated from.
- Step 1 added no link, so every link's source is "map".
- The free fit's second start (its own default) found a higher likelihood than the start at the speed-only solution: −931.13 against −931.61. Keeping the better of the two raised the statistic by 0.95. That is the gain the engine gate measures as `max_gain_from_free_default_start` before deciding whether the second start is kept (runbook §4).
- With 135 participants the four cross-link verdicts are "unresolved" (wide intervals), the own rates are "outside the family" (not tested), and no domain differs.

**Design choices and pitfalls.**
- The random generator for the Monte Carlo draws is `default_rng(seed)` here and `default_rng([seed, 2])` in the full run (§7.15). The first look's localisation p-values therefore differ slightly from the full run's, even with identical fits.
- The log's "(about an hour)" is fixed text.
- Every entry is an aggregate: estimates, statistics and verdicts.

### 7.14 `run_hp_real.py: main` (lines 136–192) — options, guards, the model and the first look

[`run_hp_real.py:136`](../../scripts/run_hp_real.py#L136)

**What it does.**

| Option | Default | Effect |
|---|---|---|
| `--panel`, `--spec` | `data/derived/panel_tdpigd.npz`, `panel_spec.json` | inputs; must lie inside `data/` |
| `--expert` | required | the map (JSON); any folder |
| `--interval-in-step1` | off | Step 1 conditions every test on the interval (if protocol §6.3's rule failed in WP0) |
| `--dose-in-step1` | off | dose nodes in Step 1's table |
| `--approval` | none | required, at least 8 characters |
| `--seed` | 20261103 | seed of the bootstrap, speed samples and Monte Carlo draws |
| `--first-look` | off | §7.13 only |
| `--n-max` | 999 (`[hp] n_max`) | cap on H-P's bootstrap replicates |
| `--speed-bootstrap` | 200 (`[speed] bootstrap`) | speed bootstrap samples; 0 skips them |
| `--threads` | PyTorch's | threads of the main process |
| `--jobs` | 1 | replicates, then speed samples, run at once |
| `--out` | `results/wp4` | must lie inside `results/` |

The order of the checks:
1. threads; `require_approval`; inputs inside `data/`; output inside `results/` (created);
2. the run name `hp_first_look` or `hp_tdpigd`, and its log function (as in §7.3);
3. `config.load()` reads `configs/protocol_v0.5.toml`; the panel and its specification (domains, treated domains, measurement-error variances); the map (§7.9);
4. the model: `ModelSpec(d, support=map, p=panel.p, treat_mask, meas_var, treatment="emax", ed50=3, ec50_ldr=3, n_quad=12, noise_era_col=0)`, every constant from the configuration file;
5. Step 1's configuration: `Step1Config(alpha=0.05, depth=3, test="fisherz", add_interval=…, include_dose=…)`, and `support_fn = make_support_fn(map, that configuration)`;
6. a log line with the panel's size and the number of map links; the map's `status`;
7. with `--first-look`: the report `{status: FIRST_LOOK, approval, map, map_status, counts, map_links_set_aside, **first_look(...)}` is written to `hp_first_look.json`, and the script stops.

**Inputs → outputs.** The command line, the two panel files and the map → the first-look report, or on to §7.15.

**The mathematics.** None; this block is configuration. `counts` is `panel.aggregate_summary()`: participants, visits, transitions, median visits and the share missing, by phenotype.

**Worked example.** None: the block needs the PPMI panel. The synthetic run of §7.13 used the same configuration file and the same Step 1 configuration.

**Design choices and pitfalls.**
- The report records the approval text, the map's path and status, and the counts. It does not record the seed, the Step 1 options (`--interval-in-step1`, `--dose-in-step1`), the engine settings such as `n_quad`, or the code version that `run_engine_gate.py` records (§7.N).
- The first-look output has a fixed name, so a second first look overwrites the first.
- `counts` are exact, with no small-cell suppression, which is fine for two large groups.
- The map file is not checked against a folder, since it holds no participant data.

### 7.15 `run_hp_real.py: main` (lines 194–257) — the full run: bootstrap, checkpoint, speed and localisation

[`run_hp_real.py:194`](../../scripts/run_hp_real.py#L194)

**What it does.**
1. `n_max` is `--n-max` or 999, and `n_speed` is `--speed-bootstrap` or 200.
2. It loads `results/wp4/hp_tdpigd_checkpoint.json` if present.
3. Two callbacks save and log progress. `save` appends `[statistic or null, ok]` after every bootstrap replicate; `save_speed` stores the speed bootstrap's state after every sample. Both rewrite the checkpoint with `write_atomic`.
4. The observed fits are run again, as on every start (they are deterministic), and give the observed statistic.
5. **The resume rule.** If the checkpoint's `lr` matches the observed statistic to within 1e-6, its replicates are reused, failures as $+\infty$. Otherwise the checkpoint is reset to `{"lr": observed, "replicates": []}` and written at once.
6. **Parallel replicates.** `parallel.ordered(hp.boot_replicate, range(k, n_max), jobs, hp.boot_setup, (null fit, panel, spec, map, Step 1 configuration, None, seed))`, where $k$ is the number of replicates already in the checkpoint:
   - replicate $i$ draws from its own stream, `default_rng([seed, 3, i])`;
   - it simulates from the fitted speed-only model for the same participants, re-selects the support with the same Step 1 configuration around the same map, and refits both models ([§5](05_hp_speed_localisation.md)).
7. `hp.run_hp(..., callback=save, resume=…, fits=…, draw=lambda: next(draws))` applies the Besag–Clifford rule in replicate order. The `finally` clause closes the generator, which terminates the workers still computing replicates that are no longer needed.
8. The report: `status` is "EXPLORATORY: " plus the map's status when it starts with "PROVISIONAL" or "TEMPLATE", and "map as given" otherwise. It also holds the approval, the map and its status, the counts, the H-P summary with `n_max` and $h$, the support as a 0/1 matrix and as named links, the links set aside, and every bootstrap statistic.
9. **The speed bootstrap.** `speed.bootstrap_speed(..., n_speed, seed=seed, support=(map, Step 1 configuration), done=checkpoint["speed"], on_sample=save_speed, jobs=jobs)`: participants are resampled within phenotype with stream `[seed, 1, b]`, the support is re-selected in every sample, and the checkpoint is updated after each. With `--speed-bootstrap 0`, only the point estimates are reported.
10. The localisation of §7.13 on the free fit, with the stream `[seed, 2]`.
11. It writes `hp_tdpigd.json`.

**Inputs → outputs.** The model and data of §7.14 → `hp_tdpigd.json` (aggregates), the checkpoint and the log.

**The mathematics.** The Besag–Clifford sequential p-value ([§5](05_hp_speed_localisation.md)). Replicate $b$ is an exceedance if $T^*_b\ge\mathrm{LR}_{obs}$ or if it failed. With $h=20$ (`[hp] h`):

$$p=\begin{cases}h/L & \text{if the } h\text{-th exceedance comes at draw } L\le n_{max},\\[2pt] (e+1)/(n_{max}+1) & \text{otherwise, with } e \text{ exceedances in } n_{max} \text{ draws.}\end{cases}$$

- The smallest attainable p-value is $1/(n_{max}+1)$: 0.001 at 999 and 0.005 at `--n-max 199`.
- Under the null the replicates are exchangeable with the observed statistic, and $P(p\le p_j)=p_j$ at every attainable value (Besag & Clifford 1991; checked by `test_null_p_values_are_uniform`, §7.30).
- When the true p-value is $\pi$, the draw count needed to reach $h$ exceedances has mean about $h/\pi$ (a negative binomial). Far from rejection, $\pi\approx0.5$ needs about 40 replicates, the Windows guide's estimate.
- Each replicate's stream depends only on (seed, $i$). So the sequence $T^*_0,T^*_1,\dots$ is the same in one session or several, with one job or many, up to the numerical tolerance between a multithreaded and a single-threaded fit.

**Worked example.** The checkpoint's format and its rebuilding for `resume` are shown in §7.10. The case of a resume with a lower `--n-max` than the number of saved replicates runs on `hp.sequential_p` directly ([`ex_sequential_resume.py`](../guide_examples/s07/ex_sequential_resume.py)):

```python
saved = [(0.1, True)] * 10                     # ten replicates saved by an earlier run with --n-max 10
sequential_p(no_draw, observed=4.0, n_max=5, h=20, resume=saved)
```

```text
resumed with --n-max 5: {'p': 0.16666666666666666, 'n_drawn': 10, 'n_exceed': 0, 'stopped_early': False}
resumed with --n-max 10: {'p': 0.09090909090909091, 'n_drawn': 10, 'n_exceed': 0, 'stopped_early': False}
```

All ten saved replicates are counted, but the p-value uses the new cap: 1/6 instead of 1/11. The error is conservative, but the reported `n_max` and `n_drawn` disagree.

**Design choices and pitfalls.**
- The resume rule checks only the observed statistic. The seed, `n_max`, $h$, the Step 1 options and the speed bootstrap's size are not stored in the checkpoint. A resume with another `--seed` mixes two replicate streams; another `--interval-in-step1` that happens to leave the observed support unchanged mixes replicates selected under two rules. `gate.Checkpoint` refuses both cases (§7.N).
- The tolerance of 1e-6 assumes bit-for-bit repeatable observed fits. The batch size follows the thread count (32 participants per batch single-threaded, 128 otherwise), which changes the order of summation, and `test_parallel_gate_pieces_match_one_at_a_time` allows 1e-3 between a threaded and a single-threaded statistic. A restart with other `--threads`, or on another machine, will probably discard the checkpoint and start the bootstrap again: safe, but days of work.
- The checkpoint is kept after the run. Rerunning the same command rebuilds the report from it, after the hour of observed fits and the localisation.
- A map whose status starts with neither "PROVISIONAL" nor "TEMPLATE" is labelled "map as given". The WP1 draft map's status begins "draft:" (`wp1.build_map`), so a run on an unedited draft would not be marked exploratory.
- A larger `--speed-bootstrap` extends a finished speed bootstrap. A smaller one keeps every saved sample, so the report's `"B"` (the new value) is smaller than its `"B_used"`.

---

### 7.16 `run_qualification.py: main` — WP3's datasets and their summary

[`run_qualification.py:20`](../../scripts/run_qualification.py#L20)

**What it does.**

| Option | Default | Effect |
|---|---|---|
| `--scenarios` | S0 S1 | scenarios to simulate |
| `--family` | A | simulator family, A or B |
| `--datasets` | 10 | datasets per scenario |
| `--n` | 1400 | participants simulated per dataset |
| `--d` | 4 | domains |
| `--jobs` | 4 | datasets at once (at most 4) |
| `--no-step1` | off | keep the given support (faster; not the protocol) |
| `--maxiter` | 3000 | optimiser iterations per fit |
| `--sim-kw` | `{}` | stress options as JSON, e.g. `{"spurious": true}` |
| `--config-a`, `--config-b` | none | the configuration key of the protocol run, per family |
| `--summarise` | off | summarise the files present, without running |

Unless `--summarise` is given, it builds one task per dataset with `qualification.make_tasks(scenario, datasets, n, family, d=…, step1=…, fit_kw={"maxiter": …}, sim_kw=…)` and runs them with `run_tasks`. Each task simulates a dataset, selects its support, fits both models, draws one bootstrap replicate (warp-speed) and records the speed estimates ([§6](06_simulation_gate.md)). Each result is one JSON file in `results/qualification/datasets/`.

It then reads every file and splits the rows:
- **null rows** (S0, S0L, S1, S1S, S1T, S1V, S4, S7) → `calibration_summary`, then `h_m1` with the configuration keys of `--config-a/-b`;
- **the other rows** (S2, S2b, S3) → `power_summary`, against the S1 critical value of the same family and configuration;
- **all rows** → `speed_summary`, the bias and root-mean-square error of log Sp-1.

The summary goes to `results/qualification/summary.json` and is printed.

**Inputs → outputs.** Options → dataset files, the summary file and the printed summary.

**The mathematics** ([§6](06_simulation_gate.md)). For one scenario and configuration, with $R$ datasets, each with its statistic $\mathrm{LR}_r$ and one bootstrap statistic $T^*_r$ (failed replicates as $+\infty$):

$$c^*=T^*_{(\lceil0.95R\rceil)},\qquad \hat\alpha=\frac1R\sum_r\mathbf 1[\mathrm{LR}_r>c^*\ \text{or dataset } r\text{ failed}],\qquad \mathrm{MCSE}=\sqrt{\hat\alpha(1-\hat\alpha)/R}.$$

A cell passes Bradley's criterion if $\hat\alpha\le0.075$. H-M1 passes if every null cell of both families passes: "partial pass" if only A:S4 fails, and "incomplete" while a cell is missing.

**Worked example** ([`ex_qualification_summary.py`](../guide_examples/s07/ex_qualification_summary.py)). Part 1 is the task list the default command would build with `--datasets 2`; part 2 is the summary of invented rows: 20 S0 and 20 S1 rows with χ²₆ statistics, one failed bootstrap replicate, and 10 S2 rows with noncentral χ²₆(15) statistics.

```text
1. tasks of `--datasets 2` with the defaults:
   S0 1000000 30da841d98 A_S0_30da841d98_seed1000000.json
   S0 1000001 30da841d98 A_S0_30da841d98_seed1000001.json
   S1 1020000 30da841d98 A_S1_30da841d98_seed1020000.json
   S1 1020001 30da841d98 A_S1_30da841d98_seed1020001.json

2. calibration (one entry per family:scenario:config):
   A:S0:30da841d98: critical 13.270, rejection 0.000 (MCSE 0.000), chi-square rejection 0.000, replicates failed 0, passes Bradley True
   A:S1:30da841d98: critical 12.725, rejection 0.000 (MCSE 0.000), chi-square rejection 0.000, replicates failed 1, passes Bradley True
   H-M1: {'verdict': 'incomplete', 'failed': [], 'missing': ['A:S0L', 'A:S1S', 'A:S1T', 'A:S1V', 'A:S4', 'A:S7', 'B:S0', 'B:S1'], 'ambiguous': []}
   power: {'A:S2:30da841d98': {'datasets': 10, 'failed': 0, 'critical_value': 12.725, 'power': 0.9, 'mcse': 0.095}}
   speed: {'A:S0:30da841d98': 0.0083, 'A:S1:30da841d98': -0.0026, 'A:S2:30da841d98': 0.0}
   by hand, S1: the ceil(0.95 x 20) = 19th smallest bootstrap statistic is 12.725
```

Reading the output:
- S0 and S1 share one configuration key, because the key hashes everything except the seed and the scenario. That is how S2 finds S1's critical value.
- In S1 the failed replicate enters as $+\infty$, the largest value, and the 19th smallest of the 20 is 12.725, which `power_summary` uses for S2.
- The default command (S0 and S1 of family A) can never produce an H-M1 verdict other than "incomplete": eight more cells are needed.

**Design choices and pitfalls.**
- `alpha` = 0.05 and Bradley's 0.075 are defaults in `qualification.py`; the script does not read `[qualification]` in `configs/protocol_v0.5.toml`, which repeats them.
- Seeds depend only on the family, the scenario and the dataset's index. Datasets of another `--n`, `--d` or stress reuse the same seeds, in files with another configuration key.
- Family B's generator reads only the `noninvariant` stress from `--sim-kw`, and has only S0, S1 and S3. Any other family-B scenario fails dataset by dataset, and `calibration_summary` counts a failed dataset as a rejection.
- `--jobs` above 4 is cut to 4.

---

### 7.17 `smoke_test.py: synthetic_visit_table` — a visit table with the WP0 audit's schema

[`smoke_test.py:30`](../../scripts/smoke_test.py#L30)

**What it does.** It turns a simulated panel into the visit table that `wp0.audit` reads ([§1](01_data.md)):
- one row per participant and panel visit, with `year` = k + 1 for the k-th visit from the landmark, and `date_years` the visit time;
- one extra row at year 0 per participant, with every domain missing;
- `cohort` "PPMI 2.0" for every third participant, a log-normal `td_pigd_ratio`, `site` = i mod 12, `mode` "remote" with probability 0.1 (always "in person" at year 0);
- `treated` = 1 when the simulated dose is positive, `dose_change` at random with probability 0.3, `motor_state` from the panel's ON indicator, and domains `d0`, `d1`, ….

**Inputs → outputs.** A panel and a random generator → a pandas DataFrame.

**The mathematics.** None.

**Worked example** ([`ex_smoke_visit_table.py`](../guide_examples/s07/ex_smoke_visit_table.py)), on an S2 panel of 40 simulated participants (36 kept):

```text
panel: 36 participants; table: 238 rows x 15 columns
columns: ['participant', 'cohort', 'year', 'date_years', 'phenotype', 'td_pigd_ratio', 'site', 'mode', 'treated', 'dose_change', 'motor_state', 'd0', 'd1', 'd2', 'd3']
participant   cohort  year  date_years phenotype  treated motor_state     d0    d3
      S0001 PPMI 1.0     1       1.001      PIGD        0         OFF  0.295 0.953
      S0001 PPMI 1.0     2       1.940      PIGD        0         OFF  0.577 1.050
      S0001 PPMI 1.0     3       2.987      PIGD        0         OFF  0.097   NaN
      S0001 PPMI 1.0     4       3.889      PIGD        0         OFF -0.154 1.797
      S0001 PPMI 1.0     5       5.011      PIGD        0         OFF  0.442 1.843
      S0001 PPMI 1.0     6       6.149      PIGD        1         OFF  0.681 1.085
      S0001 PPMI 1.0     0       0.000      PIGD        0         OFF    NaN   NaN
rows per year: {0: 36, 1: 36, 2: 36, 3: 36, 4: 36, 5: 29, 6: 22, 7: 7}
audit sections: ['min_cell', 'participant_flow', 'landmark', 'transitions_after_landmark', 'intervals', 'visits_per_participant', 'mode', 'motor_state', 'missing_share', 'missing_share_by_motor_state', 'domain_summaries', 'sites', 'overlap']
participant flow: {'cells': {'PPMI 1.0': {'PIGD': 'suppressed', 'TD': 'suppressed'}, 'PPMI 2.0': {'PIGD': 'suppressed', 'TD': '<5'}}, 'row_totals': {'PPMI 1.0': 'suppressed', 'PPMI 2.0': 'suppressed'}, 'column_totals': {'PIGD': 'suppressed', 'TD': 'suppressed'}, 'total': 36}
```

The year-0 row comes last, which does not matter to the audit. With 36 participants, one cell of the cohort × phenotype table is below 5, and complementary suppression hides every cell and margin that could reveal it, leaving only the total.

**Design choices and pitfalls.**
- The ratio is unrelated to the phenotype, and the year-0 domains are missing, so the audit's overlap and baseline summaries are exercised only for running, not for meaning.
- The table is synthetic, so the smoke test can run the audit without approval.

### 7.18 `smoke_test.py: main` — every piece once, at a tiny size

[`smoke_test.py:49`](../../scripts/smoke_test.py#L49)

**What it does.** It simulates one S2 dataset (`--n` 300 participants, `--seed` 7, four domains), runs each piece of the analysis chain once at a small size, prints each piece's time, and writes the aggregates to `results/smoke/summary.json` (`--out` must lie inside `results/`). Every fit uses `maxiter=500` instead of 3,000.

| Piece | Call | Smoke size | Protocol size |
|---|---|---|---|
| Step 1's reported map | `reported_map(panel, Step1Config(depth=2))` | Fisher z, separating sets of at most 2 | clustered test, at most 4 (`reported_map`'s default) |
| H-P | `run_hp(n_max=9, h=3)`, support re-selected | p-values from 0.1 | $n_{max}$ = 999, $h$ = 20 |
| speed bootstrap | `bootstrap_speed(B=4)` | 4 samples | 200 |
| per-link and per-domain tests | `per_link_tests(M=500)`, `classify_links(eps=0.10)` | 500 draws | 2,000 |
| anchored tests | `anchored_tests` | default | default |
| joint discovery | `joint_discovery(n_lambda=4)` | 4 penalties | 12 |
| shared-eigenvector check | `commutator_test(B=50)` | 50 | 500 (default) |
| comparators | C3 (50 permutations), C5, C11 (separating sets ≤ 2), C12 | — | protocol §8.4 |
| separate graphs | `separate_graph_reference(n_sub=3, n_perm=3)` | 3 and 3 | 25 and 50 |
| WP0 audit | `wp0.audit` on §7.17's table | — | — |

**Inputs → outputs.** Options → the printed timings and `summary.json` (counts, every piece's output, timings).

**The mathematics.** None; the pieces are those of [§4](04_step1_comparators.md)–[§6](06_simulation_gate.md).

**Worked example.** Not run (15–25 minutes, runbook §1). §7.17 runs its table builder.

**Design choices and pitfalls.**
- There is no assertion: the smoke test proves that every piece runs and returns JSON-safe aggregates, not that any value is right.
- The speed bootstrap is called without `support=`, so the support is not re-selected in its samples. `run_hp_real.py` passes `support=(map, Step 1 configuration)`, and that branch of `bootstrap_speed` runs on no synthetic data anywhere: not here, not in the tests (§7.42).
- H-P's bootstrap runs serially through `run_hp`'s own draw function, not through `boot_replicate` and `parallel.ordered` as in `run_hp_real.py`.

---

### 7.19 `bench_threads.py: eval_child` and `fit_child` — one timed job

[`bench_threads.py:27`](../../scripts/bench_threads.py#L27), [`bench_threads.py:57`](../../scripts/bench_threads.py#L57)

**What it does.** Each benchmark job is a child process of the script itself.
- `eval_child(threads, evals, seed, chunk)` sets PyTorch's threads, simulates one S2 dataset at WP0's numbers with ten domains (`configs/wp0_sim_settings.json`, all ten missing shares, a pool of 5 × the target counts), builds the free model's layout at its start values, and runs two warm-up evaluations. It prints "ready", waits for a line on standard input, times `evals` likelihood-and-gradient evaluations, and prints `{"seconds_per_eval", "n"}`.
- `fit_child(threads, seed)` loads the power pilot's task for that seed from `results/wp0/power/datasets/A_S2_*_seed<seed>.json`, prints "ready", waits, runs the whole task (Step 1 and the fit pair, `qualification.dataset_task`), and prints its seconds, statistic and df.

**Inputs → outputs.** Threads, evaluations, seed and batch size → one JSON line on standard output.

**The mathematics.** None; it measures wall time. A fit needs the same number of evaluations however it is threaded, so evaluations per hour set the time of a run (the script's docstring).

**Worked example.** Not run: one evaluation at ten domains takes seconds, and a fit pair hours.

**Design choices and pitfalls.**
- The "ready"/"go" handshake keeps the set-up (simulation, preparation, warm-up) out of the timing.
- All jobs of one configuration evaluate the same dataset at the same point, the start values. The cost at the optimum can differ, for example if the drift becomes less stable and the transitions take the composition route.
- `fit_child` needs the pilot's files, which exist only where the power pilot ran. With several configurations in that folder, the glob takes the first match. The default seed, 1,080,003, is the fourth S2 dataset of the pilot (§7.7).

### 7.20 `bench_threads.py: _read`, `run_jobs` and `parse` — starting the jobs together

[`bench_threads.py:70`](../../scripts/bench_threads.py#L70), [`bench_threads.py:78`](../../scripts/bench_threads.py#L78), [`bench_threads.py:96`](../../scripts/bench_threads.py#L96)

**What it does.**
- `_read(p, start)` reads the child's output line by line until a line starts with `start`, skipping anything else (a library's progress messages). If the output ends first, it raises "a job ended early".
- `run_jobs(jobs, threads, args)` starts `jobs` children with the four thread variables (`OMP_NUM_THREADS` and the others) set to `threads`. It waits until every child has printed "ready", starts the clock, sends "go" to all of them, reads each child's JSON line, stops the clock at the last one, and waits for the children to exit.
- `parse("JxT")` returns (J, T).

**Inputs → outputs.** The layout and the children's arguments → (their results, the wall time).

**The mathematics.** The figure of merit:

$$\text{evaluations per hour}=\frac{3600\times J\times\text{evals}}{\text{wall seconds}},\qquad \text{fit pairs per hour}=\frac{3600\times J}{\text{wall seconds}}.$$

The wall time runs to the last job's result, so every job is charged the slowest job's time.

**Worked example** ([`ex_bench.py`](../guide_examples/s07/ex_bench.py)), with invented timings:

```text
1. parse: (4, 1) (1, 10)
   parse('4') fails: not enough values to unpack (expected 2, got 1)
2. evals_per_hour = 3600 x jobs x evals / wall = 3600 x 4 x 30 / 150.0 = 2880
   if every job had run alone at its own pace: 2987 an hour
3. _read skips other lines: ready | {"seconds_per_eval": 4.6}
   a job that ends early: a job ended early
```

The per-job pace (4.6 to 5.0 seconds per evaluation) would give 2,987 evaluations an hour. The script reports 2,880, because the fastest jobs wait for the slowest. That is the conservative choice for planning.

**Design choices and pitfalls.**
- `--fit JxT` uses only the first J seeds given; with fewer seeds, fewer jobs start, and the report's `jobs` says how many.
- A child that crashes before "ready" (for example, a missing pilot file) surfaces only as "a job ended early"; its own traceback is printed to the console.

### 7.21 `bench_threads.py: main` — the benchmark report

[`bench_threads.py:101`](../../scripts/bench_threads.py#L101)

**What it does.**

| Option | Default | Effect |
|---|---|---|
| `--configs` | none | layouts `JxT` to time: J jobs of T threads, all evaluating at once |
| `--evals` | 30 | evaluations per job |
| `--seed` | 1080003 | dataset seed |
| `--chunk` | the engine's default | participants per likelihood batch |
| `--fit` | none | `JxT`: J pilot datasets fitted at once with T threads each |
| `--seeds` | 1080003 | the pilot seeds for `--fit` |

The hidden options `--child` and `--fit-seed` turn the process into a child (§7.19). Otherwise it loads `results/bench/threads.json` if present, adds one entry per layout (`<JxT>_chunk<c>` or `<JxT>_default_chunk`: the batch size, jobs, threads, evaluations each, seconds per evaluation of each job, evaluations per hour, participants) and one `fit_<JxT>` entry for `--fit` (the wall hours, fit pairs per hour, and each dataset's time and statistic next to the pilot's), and writes the file back.

**Inputs → outputs.** Options → `results/bench/threads.json`, and a printed line per entry.

**The mathematics.** §7.20's figures of merit.

**Worked example.** Not run (a configuration takes minutes at ten domains, and a fit pair hours).

**Design choices and pitfalls.**
- The engine's default batch size follows the threads: 32 participants single-threaded, 128 otherwise (`likelihood.default_chunk`). `--chunk` overrides it, which is how the batch sizes were compared.
- Rerunning a layout overwrites its entry, and other programs running at the same time change the figures.

---

### 7.22 `md_to_docx.py`: `INLINE`, `BULLET`, `NUMBER`, `plain` and `runs` — inline formatting

[`md_to_docx.py:18`](../../scripts/md_to_docx.py#L18), [`md_to_docx.py:23`](../../scripts/md_to_docx.py#L23), [`md_to_docx.py:27`](../../scripts/md_to_docx.py#L27)

**What it does.** The converter handles the subset of Markdown the project's documents use (§7.25). Inline formatting goes through one regular expression, `INLINE`, with four alternatives in priority order:
1. `**…**` (bold, the shortest match);
2. `` `…` `` (code);
3. `[label](url)` (a link; the label is kept and the URL dropped);
4. `*…*` (italic), only if the opening star does not follow a letter, digit, underscore or star, the next character is neither a space nor a star, and the closing star is not followed by a letter, digit, underscore or star.

`runs(paragraph, text, size)` splits the text on `INLINE` (the capturing group keeps the matches), adds each piece as a run (bold; Consolas; plain; italic), turns a leading "[ ] " into "☐ ", and un-escapes `\|`. `plain(text)` removes stars, backticks and backslashes, for width estimates and the document title. `BULLET` matches "- item" and `NUMBER` "12. item", each with its indentation.

**Inputs → outputs.** A paragraph and a string → runs added to the paragraph.

**The mathematics.** None.

**Worked example** ([`ex_md_to_docx.py`](../guide_examples/s07/ex_md_to_docx.py), parts 1 and 5):

```text
   Normal         'Intro with ' + 'bold'[b] + ', ' + 'italic'[i] + ', ' + 'code'[mono] + ', a ' + 'link' + ' and 2 * 3 * 4 = 24. A second line of the same paragraph; snake_case and $a*b*c$ stay as typed.'
5. INLINE split of '2 * 3 * 4 and *x*': ['2 * 3 * 4 and ', '*x*']
```

The stars of `2 * 3 * 4` are followed by spaces, and those of `$a*b*c$` follow letters, so neither becomes italic. `*x*` does.

**Design choices and pitfalls.**
- Underscores never make italics, and bold inside italics is not supported.
- LaTeX passes through as typed: a document with formulas, like this guide, converts with its `$…$` visible.
- Link targets are dropped, so a Word reader cannot follow them.

### 7.23 `md_to_docx.py`: `shade`, `cell_margins`, `min_height`, `col_widths` and `table` — tables

[`md_to_docx.py:52`](../../scripts/md_to_docx.py#L52), [`md_to_docx.py:82`](../../scripts/md_to_docx.py#L82), [`md_to_docx.py:102`](../../scripts/md_to_docx.py#L102)

**What it does.** `table(doc, lines)`:
1. splits each row on unescaped `|`, drops the separator rows (cells such as `---` or `:-:`), and pads or cuts every row to the header's number of cells;
2. uses 9-point type and 90-twip side padding, or, for a grid of 10 or more columns (a domain-by-domain matrix), 8-point type, 40-twip padding, a first column clamped to 1.6–2.4 cm and equal widths for the rest;
3. sets fixed column widths from `col_widths`, adds the cell padding (`cell_margins`, placed before `tblLook` as the schema orders it), and fills the cells;
4. makes the header bold on a grey shade (`E7E6E6`) and repeats it on every page;
5. gives an empty row a minimum height (`min_height`): 900 twips in a one-column table (an answer box), 450 otherwise.

**Inputs → outputs.** The document and the table's lines → a Word table.

**The mathematics.** The column widths. With usable width $U=21-2\times2=17$ cm (A4 minus margins), $c=0.021\times\text{size}$ cm per character (Calibri, roughly) and padding $q$ = side/567 cm:

$$m_j=\ell_j\,c+2q,\qquad w_j=\min\Big(50,\ \max\big(1,\ \tfrac12\max_r|x_{rj}|+\tfrac12\,\overline{|x_{\cdot j}|}\big)\Big),$$

where $\ell_j$ is the length of column $j$'s longest word (header words count 1.1 times), $|x_{rj}|$ the length of cell $(r,j)$, and $w_j$ is raised to at least 20 if the header names a column to write in ("comment", "change", "your answer", "drives"). Then

$$\text{width}_j=\begin{cases}m_j+(U-\sum_k m_k)\,\dfrac{w_j}{\sum_k w_k} & \text{if } \sum_k m_k<U,\\[6pt] U\,\dfrac{m_j}{\sum_k m_k} & \text{otherwise.}\end{cases}$$

No word is broken, and the spare width goes by how much text a column holds.

**Worked example** ([`ex_md_to_docx.py`](../guide_examples/s07/ex_md_to_docx.py), part 2), a two-column table with header "Domain | Your answer", a row "Tremor \| rest | " and an empty row:

```text
2. table cells: [['Domain', 'Your answer'], ['Tremor | rest', ''], ['', '']]
   column widths (cm): [6.08, 10.91]
   rows given a minimum height (empty rows are space to write in): [False, False, True]
```

By hand, at 9 points: $c=0.189$ cm and $q=0.159$ cm.
- Both columns' longest word is a six-letter header word, so $m_1=m_2=6.6\times0.189+0.317=1.565$ cm.
- $w_1=\tfrac12\times13+\tfrac12\times\tfrac{6+13+0}{3}=9.67$, and $w_2=\tfrac12\times11+\tfrac12\times\tfrac{11}3=7.33$, raised to 20 for "your answer".
- The spare width $17-3.13=13.87$ cm is split 9.67 : 20, giving $1.565+4.52=6.08$ and $1.565+9.35=10.91$ cm.

The escaped pipe stayed inside its cell.

**Design choices and pitfalls.**
- Cells beyond the header's count are cut silently.
- An unescaped `|` inside inline code splits the cell.
- The widths are estimated from character counts, so a column of wide characters can overflow.

### 7.24 `md_to_docx.py`: `list_item`, `note`, `code` and `footer` — the other blocks

[`md_to_docx.py:138`](../../scripts/md_to_docx.py#L138), [`md_to_docx.py:150`](../../scripts/md_to_docx.py#L150), [`md_to_docx.py:159`](../../scripts/md_to_docx.py#L159), [`md_to_docx.py:174`](../../scripts/md_to_docx.py#L174)

**What it does.**
- `list_item`: a bullet becomes Word's "List Bullet" style ("List Bullet 2" one level in, "List Bullet 3" beyond). A numbered item becomes a Normal paragraph with a hanging indent, its number typed as text.
- `note`: a block quote becomes an indented paragraph in grey italics.
- `code`: each line of a code block becomes a paragraph in 8.5-point Consolas on a light shade, with no inline formatting.
- `footer`: "text · page " followed by a PAGE field.

**Inputs → outputs.** The document and the block's text → paragraphs.

**The mathematics.** None.

**Worked example** ([`ex_md_to_docx.py`](../guide_examples/s07/ex_md_to_docx.py), parts 1 and 3):

```text
   List Bullet    'a bullet that continues on the next line'
   List Bullet 2  'a nested bullet'
   Normal         '1. a numbered item'
   Normal         '2026. a sentence that starts with a year'
   Normal         'A note on two lines.'[i]
   Normal         'code line with **stars**'[mono]
3. footer: Guide example · footer  ·  page  + a PAGE field
```

**Design choices and pitfalls.**
- Numbers are typed, not Word numbering, so renumbering in Word does nothing.
- A paragraph line that starts with a number, a full stop and a space ("2026. a sentence…") is read as a numbered item.

### 7.25 `md_to_docx.py: convert` — the whole document

[`md_to_docx.py:190`](../../scripts/md_to_docx.py#L190)

**What it does.** It sets an A4 page with 2-cm margins, Calibri 10.5 points and a 20-point title, then reads the Markdown line by line. The tests are made in this order:
1. a blank line or a rule (`---`) ends the current paragraph;
2. `<!-- pagebreak -->` inserts a page break;
3. a code fence starts a code block that runs to the next fence;
4. a line starting with ☐ is an answer option;
5. `#` starts a heading: level 1 is the Title (and the document's title, if none is set), level 2 Heading 1, level 3 Heading 2, deeper levels Heading 3;
6. `>` starts a note, joined over consecutive lines;
7. `|` starts a table, over consecutive lines;
8. a bullet or numbered line starts a list; lines indented by two spaces continue its last item;
9. any other line joins the current paragraph, separated by a space.

At the end it adds the footer (the given text, or the title), sets the zoom percentage that python-docx's template omits but the schema requires, blanks the author, and saves.

**Inputs → outputs.** A Markdown path, a Word path and optional footer text → the Word file.

**The mathematics.** None.

**Worked example** ([`ex_md_to_docx.py`](../guide_examples/s07/ex_md_to_docx.py), run with `/usr/bin/python3`), on a 25-line invented document:

```text
1. paragraphs (style: runs)
   Title          'A tiny document'
   Normal         'Intro with ' + 'bold'[b] + ', ' + 'italic'[i] + ', ' + 'code'[mono] + ', a ' + 'link' + ' and 2 * 3 * 4 = 24. A second line of the same paragraph; snake_case and $a*b*c$ stay as typed.'
   Heading 1      'Section one'
   List Bullet    'a bullet that continues on the next line'
   List Bullet 2  'a nested bullet'
   Normal         '1. a numbered item'
   Normal         '2026. a sentence that starts with a year'
   Normal         'A note on two lines.'[i]
   Normal         'code line with **stars**'[mono]
   Normal         '☐ a task box'
   Normal         '☐ an answer option'
   Normal         ''  <page break>
   Heading 3      'Deep heading'
2. table cells: [['Domain', 'Your answer'], ['Tremor | rest', ''], ['', '']]
   column widths (cm): [6.08, 10.91]
   rows given a minimum height (empty rows are space to write in): [False, False, True]
3. footer: Guide example · footer  ·  page  + a PAGE field
4. core properties: title = 'A tiny document' | author = ''
```

The first heading became the Title and the document's title, a level-5 heading became Heading 3, and the author field is empty.

**Design choices and pitfalls.**
- python-docx is in neither `pyproject.toml` nor the ppmi-causal environment. The README's and runbook's commands use the system Python, which has it on this Mac.
- Rules are dropped, an unclosed code fence turns the rest of the document into code, and HTML other than the page-break comment is printed as text.
- The docstring's example still names the v0.2 protocol file.

---

### 7.26 `tests/run_tests.py` — the runner, and the run of record

[`run_tests.py:21`](../../tests/run_tests.py#L21)

**What it does.** pytest is not installed, so the tests have a plain runner:
1. it puts `src/` and `tests/` on `sys.path`;
2. it collects `tests/test_*.py`, keeping the files whose name contains the optional argument (`run_tests.py engine` runs `test_engine.py` only);
3. it imports each file and calls every module attribute whose name starts with `test_`, in alphabetical order;
4. it skips a function marked slow (`helpers.slow`) unless the environment variable `PDX_SLOW` is set;
5. it catches any exception, prints its traceback, and goes on; at the end it prints the counts and returns exit code 1 if any test failed.

**Inputs → outputs.** An optional pattern → printed results and an exit code.

**The mathematics.** None.

**Worked example: the run of record.** The whole suite was run once for this guide, without the slow tests (`OMP_NUM_THREADS=1 ../ppmi-causal/.venv/bin/python tests/run_tests.py`). Abridged:

```text
  ok    test_engine.test_likelihood_matches_reference  (0.7 s)
  ok    test_fit.test_recovery_and_nonnegative_lr  (68.5 s)
  ok    test_gate.test_dispersed_starts_resumed_midway_give_the_same_fits  (67.3 s)
  skip  test_gate.test_parallel_gate_pieces_match_one_at_a_time (slow; set PDX_SLOW=1)
  ok    test_gate.test_quadrature_check_reports_the_statistic_at_each_node_count  (31.3 s)
  skip  test_hp.test_bootstrap_replicates_agree_however_many_run_at_once (slow; set PDX_SLOW=1)
  skip  test_hp.test_speed_bootstrap_resumed_midway_gives_the_same_intervals (slow; set PDX_SLOW=1)
  ok    test_multiplicity.test_against_statsmodels  (6.1 s)
  ...
50 passed, 0 failed, 3 skipped
```

| File | Tests (slow) | Section | Time |
|---|---|---|---|
| `test_engine.py` | 8 | §7.28 | 1.5 s |
| `test_fit.py` | 1 | §7.29 | 68.5 s |
| `test_hp.py` | 7 (2) | §7.30 | 0.7 s |
| `test_localise.py` | 7 | §7.31 | < 0.1 s |
| `test_multiplicity.py` | 2 | §7.32 | 6.1 s |
| `test_step1.py` | 5 | §7.33 | 0.1 s |
| `test_comparators.py` | 1 | §7.34 | 0.8 s |
| `test_sim.py` | 4 | §7.35 | 1.4 s |
| `test_qualification.py` | 2 | §7.36 | < 0.1 s |
| `test_gate.py` | 4 (1) | §7.37 | 98.6 s |
| `test_transforms.py` | 2 | §7.38 | < 0.1 s |
| `test_ppmi.py` | 3 | §7.39 | 0.4 s |
| `test_wp0.py` | 2 | §7.40 | < 0.1 s |
| `test_wp1.py` | 5 | §7.41 | < 0.1 s |
| **all** | **53 (3)** | | **about 3 minutes** |

The three slow tests were not run for this guide.

**Design choices and pitfalls.**
- The tests use bare `assert` statements, which `python -O` removes, so under `-O` every test passes.
- Any non-empty value of `PDX_SLOW` turns the slow tests on, `PDX_SLOW=0` included.
- There is no time-out and no isolation: a hanging test stops the run, and module state is shared.
- A test file that fails to import stops the whole run, because the import is outside the `try`.
- Temporary folders made by `tempfile.mkdtemp` (in `test_gate`, `test_ppmi`, `test_wp1`) are not removed.
- The runbook (§1) and the README speak of 49 unit tests; the runner finds 53, 50 of them run by default (§7.N).

### 7.27 `tests/helpers.py` and `tests/ppmi_mock.py` — the shared fixtures

[`helpers.py`](../../tests/helpers.py), [`ppmi_mock.py`](../../tests/ppmi_mock.py)

**`helpers.py`.**

| Fixture | What it gives |
|---|---|
| `slow(fn)` | marks a test as slow (`fn.slow = True`), skipped unless `PDX_SLOW` is set |
| `small_panel(scenario="S1", n=120, seed=1, d=4)` | a family A panel and its `info` (truth, support), from `scenarios.generate` |
| `spec_for(panel, info, treatment="emax", era=True, n_quad=12)` | the analysis model on the true support, with the truth's treated domains and measurement-error variances, era shifting the noise; a near copy of `qualification.spec_for` |
| `random_point(panel, spec, kind, seed=0, scale=0.05)` | the start values plus N(0, 0.05²) noise, so that every parameter (load, era noise, treatment) is away from its start and active |
| `layout(spec, kind)` | the parameter layout of a model kind |

**`ppmi_mock.py`.** `make(root)` writes a mock PPMI download: the folders, file names (stamped `_01Jan2026.csv`) and columns that `ppmi.build` reads, with 13 made-up participants. Odd-numbered participants are TD and even-numbered PIGD: every Part III item is 1, except that the tremor items are 2 in TD and 0 in PIGD, and gait, freezing and postural stability the reverse. Each participant is designed to exercise one rule:

| Participant | Set-up | Rule exercised |
|---|---|---|
| 1–8 | enrolled 01/2015; BL, V04 and V06 visits a year apart | the TD/PIGD phenotype and the annual grid |
| 3 | a remote visit R04 on the anniversary and an in-person V04 two months later | the grid prefers the in-person visit |
| 5 | no V06; study ended 06/2016 with death recorded | the Part II event within 24 months |
| 9 | LRRK2 genetic enrolment | excluded: not sporadic |
| 10 | treated at baseline (OFF state), with a levodopa log entry | excluded: treated at baseline |
| 11 | no baseline visit | excluded |
| 12 | enrolled 01/2021 (PPMI 2.0); treated from year 1, with a log entry | the cohort and the dose change |
| 13 | "Screen failed" | not enrolled |

The other forms are filled with constant or random valid scores: Schwab & England 90, MoCA 27, a non-depressed GDS, random SCOPA-AUT items of 0 or 1, Epworth 1 per item, RBDSQ 0.

### 7.28 `test_engine.py` — the likelihood and the transitions

[`test_engine.py`](../../tests/test_engine.py). `_with_gaps` adds missing values: each score with probability 0.15 (0.075 at year 1, never the first domain of the landmark visit), plus the whole third visit of every participant with more than three.

| Test | What it proves | How | Not covered |
|---|---|---|---|
| `test_likelihood_matches_reference` | The fast likelihood (batched, composed transitions, masked Kalman filter) equals an independent implementation. | 40 participants with gaps; both treatment forms (linear, Emax) × the null, free and joint models; a random point each; agreement to a relative 1e-8 with `engine/reference.py` (scipy's matrix exponential with Van Loan's blocks, explicit selection of observed components). | The reference takes the year-1 prior (`mu1`, `S1`, `y1c`) from the fast code's `prepare()` and the parameter mapping from `unpack_numpy`, so an error in either would be shared and pass. |
| `test_gradient_matches_finite_differences` | Automatic differentiation gives the derivative of the likelihood. | 30 participants; null and free models; 12 random coordinates; central differences with step 1e-6; relative 1e-4. | 12 of the 112 (null) and 119 (free) coordinates, chosen once by seed 0, so a fault confined to an unsampled block would pass; the joint model is not checked. |
| `test_null_is_nested_in_free_and_joint` | The speed-only model is an exact special case of the free and joint models, so LR ≥ 0 when the free fit starts at the null solution. | A null point with speed ratio 1.8, mapped by `null_to`; equal log-likelihoods to a relative 1e-8. | That the optimiser reaches the maximum (§7.29 checks LR ≥ 0 on one fit). |
| `test_grouped_transitions_match_matrix_exponential` | The fast transitions E, M and S agree with Van Loan's block exponential, and S with the stationary identity for long intervals. | Random free-model drift and noise; 240 intervals uniform on (0, 10): agreement to 1e-10; intervals up to 60: $S=\Sigma-E\Sigma E^\top$ with scipy's Lyapunov solution, to 1e-12. | Drifts that are not stable (next tests). |
| `test_ten_domain_likelihood_matches_reference` | The 10-domain path (the blocked matrix products of `engine/linalg.py`) and batching are right. | 14 participants, 10 domains, batches of 5, null and free models, relative 1e-8. | The joint model at 10 domains. |
| `test_transitions_for_marginal_drift_and_long_intervals` | The composition route needs no stable drift, stays accurate over long intervals, and its cost grows only with the logarithm of the interval. | A drift with eigenvalues 0 and −0.2: agreement with Van Loan at intervals 0.3, 2 and 7.5 (1e-10); an interval of 5,000 computed in under 1 s and agreeing with `transition_np` (1e-8). | `transition_np` lives in the same module, so the long-interval check is a consistency check; the 1-second limit depends on the machine. |
| `test_wild_speed_parameters_stay_cheap` | An extreme trial point of a line search stays finite and cheap (log-speed bounded at ±15 at the quadrature nodes). | log σ = 3 (σ = 20); one evaluation finite and under 10 s. | The value's accuracy at such points; the time limit depends on the machine. |
| `test_both_noise_routes_agree` | The two routes to S, the stationary covariance and the composition, agree when the drift is stable. | 240 intervals up to 80; agreement to 1e-11. | — |

### 7.29 `test_fit.py` — recovery of a known speed

[`test_fit.py`](../../tests/test_fit.py)

| Test | What it proves | How | Not covered |
|---|---|---|---|
| `test_recovery_and_nonnegative_lr` | On one simulated dataset the fits recover the speed ratio roughly, and the likelihood ratio is not negative. | S1, 400 simulated participants, seed 11; `fit_pair` with `maxiter=1500`; asserts $\hat\ell_{free}\ge\hat\ell_{null}-10^{-6}$, Sp-1 in (1.4, 2.8) (truth 2), and Sp-3's least-squares speed above 1. | Bias (one dataset, a band of ±40%); load, noise and deviations; a changed link (S2); interval coverage. Protocol §16's "recovery of known speed, load and deviations, and calibration under the null" is left to the engine gate's recovery check and to WP3. |

### 7.30 `test_hp.py` — the sequential p-value, the speed bootstrap and parallel runs

[`test_hp.py`](../../tests/test_hp.py). `_draws(values, oks)` returns a draw function that yields the given (statistic, ok) pairs in turn.

| Test | What it proves | How | Not covered |
|---|---|---|---|
| `test_sequential_stops_at_h_exceedances` | Besag–Clifford stops at the $h$-th exceedance with $p=h/L$. | Draws 5, 1, 6, 7, … against 4 with $h=3$: stops at the fourth draw, $p=3/4$. | — |
| `test_sequential_runs_to_n_max` | Without $h$ exceedances it runs to the cap with $p=(e+1)/(n_{max}+1)$. | 99 draws of 0.1 against 4, $h=20$: $p=1/100$. | — |
| `test_ties_and_failures_count_as_exceedances` | A tie and a failed replicate both count as at least as large. | Draws (4, ok), (0, failed), (0, ok) against 4: 2 exceedances, 1 failure, $p=3/4$. | — |
| `test_null_p_values_are_uniform` | Under exchangeability the sequential p-value is valid. | 2,000 trials; observed and replicates χ²₃; $n_{max}=99$, $h=10$; share of $p\le0.05$ within 0.05 ± 0.015 and of $p\le0.5$ within 0.5 ± 0.04 (about 3 and 3.6 binomial SDs). | That H-P's bootstrap replicates are exchangeable with the observed statistic (WP3's calibration); the `resume` argument of `sequential_p`, which `run_hp_real.py` uses, is not exercised by any test. |
| `test_speed_bootstrap_resumed_midway_gives_the_same_intervals` *(slow, not run here)* | The speed bootstrap's checkpoint and parallel runs reproduce the uninterrupted result. | S1, 80 participants; B = 2; resumed after one sample: identical Sp-1 interval and sample count; 2 jobs: Sp-1 interval within a relative 1e-3. | Re-selection of the support in the samples (`support=`), the branch `run_hp_real.py` uses. |
| `test_parallel_runs_give_the_results_in_order_from_several_workers` | `parallel.ordered` returns results in item order, runs in other processes when jobs > 1 and in this one otherwise, and handles an empty list. | $k\,x^2$ for x = 0…5 with 1 and 3 jobs; process ids compared. | Closing the generator early (the pool is terminated when Besag–Clifford stops); an exception in a worker. |
| `test_bootstrap_replicates_agree_however_many_run_at_once` *(slow, not run here)* | Per-replicate seeds make H-P's bootstrap independent of the number of jobs. | S1, 80 participants; replicates 0–2 with 1 and 2 jobs: statistics within 1e-3, the same `ok` flags. | The combination with the sequential rule and its early stop. |

### 7.31 `test_localise.py` — the minimal-L1 speed and the link classes

[`test_localise.py`](../../tests/test_localise.py). `_setup(entries, d, noise, M)` builds a support from `{(to, from): (a0, a1)}` and hands `per_link_tests` Monte Carlo draws $a\pm$ noise directly, bypassing any fit. `BASE` has four own rates −0.25 → −0.5 (speed 2), one link stronger beyond speed (0.20 → 0.64, Δ = 0.64/2 − 0.20 = 0.12), one link present only in PIGD (0 → 0.50) and three similar links (0.15 → 0.30).

| Test | What it proves | How | Not covered |
|---|---|---|---|
| `test_weighted_median` | The weighted median takes the value where the cumulative weight first reaches half. | Three small cases, including unsorted values and a weight that favours an extreme. | A tie at exactly half the weight (the lower value is taken). |
| `test_speed_rule_recovers_pure_speed_and_resists_one_deviation` | The minimal-L1 speed recovers a pure speed difference, ignores one strong deviation, and does not let a sign change vote. | a1 = 2 a0 gives 2; one link raised to 0.9 still gives 2; a sign change still gives 2; the kernel mode gives 2 within 0.01. | Deviations carrying more than half of the weight (the identification limit); entries below `near_zero` (all here are at least 0.10). |
| `test_batch_matches_scalar` | The vectorised rule equals the scalar one. | 200 random rows, `near_zero` 0.05, NaN equal to NaN. | — |
| `test_link_classes` | The classes come out right in a clear case: "different strength beyond speed", "structurally different", "similar", "outside the family", and the per-domain verdicts. | `BASE` with noise 0.01 and 4,000 draws; $\hat s$ within 0.05 of 2; `classify_links(eps=0.10, method="none")`; "speed and load only" false; domains 1 and 2 "differs beyond speed". | Benjamini–Yekutieli (`method="none"` is used); draws from a fitted model's covariance (`free_draws`); verdicts near ε; domain 3's verdict. |
| `test_wide_interval_is_unresolved` | A link with a wide interval is "unresolved", not given a false class. | Noise 0.3 on one similar link. | — |
| `test_disagreeing_diagnostics_make_verdicts_not_separable` | When the diagnostic speeds disagree, a class that changes across the speed set becomes "speed and mechanism not separable". | Own rates give speed 2, the cross-links alone 3.2; the stronger link's class changes across [2, 3.2]. | — |
| `test_unidentified_speed` | With no entry able to vote, every link and domain verdict is "speed not identified". | Every reference entry at most 0.045, below `near_zero`. | — |

### 7.32 `test_multiplicity.py` — Holm, Benjamini–Hochberg, Benjamini–Yekutieli and simultaneous intervals

[`test_multiplicity.py`](../../tests/test_multiplicity.py)

| Test | What it proves | How | Not covered |
|---|---|---|---|
| `test_against_statsmodels` | The three procedures' decisions, and Holm's adjusted p-values, equal statsmodels'. | 50 random vectors of 11 p-values (8 uniform, 3 below 0.01) at 0.05. | Benjamini–Yekutieli's adjusted p-values; ties; missing values. |
| `test_simultaneous_intervals_cover_jointly` | The max-statistic band has the joint coverage asked for, and the intersection–union "similar" verdict needs every interval inside ±ε. | 20,000 draws of 6 independent N(0, 1); joint coverage 0.90 ± 0.01 of the same draws; `iut_similar` true for intervals a tenth as wide inside ±1, false inside ±0.5. | Coverage is measured on the draws that built the band, so it checks the quantile, not out-of-sample coverage; correlated draws. |

### 7.33 `test_step1.py` — Step 1's tiers and support

[`test_step1.py`](../../tests/test_step1.py). `_panel` simulates a linear yearly chain: persistence 0.6, domain 0 → 1 and 1 → 2 at 0.5, noise SD 0.6, 400 participants with five visits, random groups, three centred covariates.

| Test | What it proves | How | Not covered |
|---|---|---|---|
| `test_tiers_and_lagged_support` | The support finds the two lagged links and the own rates, and no reversed link. | `select_support` with separating sets of at most 2: 1 ← 0 and 2 ← 1 present, 0 ← 2 and 0 ← 1 absent. | A larger or confounded system. |
| `test_no_edge_runs_backwards_in_time` | Tier knowledge forbids an arrow from year t+1 to year t. | FCI (the Fast Causal Inference algorithm) on the two-slice table; no tail at a year-t+1 node with an arrowhead at a year-t node. | — |
| `test_cluster_test_runs` | The participant-clustered test recovers the same links. | 200 participants, `cluster_fisherz`. | Its size under the null. |
| `test_hidden_confounding_keeps_a_circle_at_the_earlier_node` | A link due only to a hidden cause stays a o→ b or a ↔ b, never a → b (the replacement of causal-learn's tier step). | 5,000 rows, a and b sharing a latent cause, tiers [0, 1]. | — |
| `test_testwise_deletion_keeps_incomplete_rows` | Rows with gaps are kept, and both tests still find the links. | 15% of scores missing at random. | Missingness that is not at random; the interval rule (`add_interval`), dose tiers (`include_dose`), the era and exogenous tiers, and `reported_map`. |

### 7.34 `test_comparators.py` — the comparators run

[`test_comparators.py`](../../tests/test_comparators.py)

| Test | What it proves | How | Not covered |
|---|---|---|---|
| `test_comparators_run_and_return_link_matrices` | C12 (DYNOTEARS), C5 (tsLDiffPC), C3 (a network comparison test, NCT, with 20 permutations) and C11's interaction tests run and return a 4 × 4 cross-link matrix and 4 self-lags; the registry lists C1–C12. | S1, 200 participants. | Every value: no check of what the comparators find. The re-implementations of C5 and C12 are validated by the toys (§7.52, §7.55), not by a test; C11's separate-graph part. |

### 7.35 `test_sim.py` — the simulators and the scenario table

[`test_sim.py`](../../tests/test_sim.py)

| Test | What it proves | How | Not covered |
|---|---|---|---|
| `test_phenotype_rule` | The Stebbins rule classifies clear cases: a tremor/PIGD ratio of at least 1.15 is TD, at most 0.90 PIGD, in between or both zero indeterminate. | Four participants with item means 2/1, 0/2, 1/1, 0/0. | The boundaries 1.15 and 0.90 themselves; PIGD items all zero with tremor present (TD). |
| `test_every_scenario_generates` | All 11 family-A scenarios and family B's S0, S1 and S3 produce data. | 120 simulated participants each; more than 50 kept, both groups present, a 4 × 4 support with its diagonal. | Any distributional property (realised speed ratio, missingness, dropout). |
| `test_mechanism_scenarios_follow_the_table` | The truths match protocol §8.2: S2 has one deviation of exactly 2ε on the standardised scale; S2b a link 4 times as strong in PIGD; S3 a PIGD-only and a TD-only link; every scenario the same map (so the same df); the null scenarios one mechanism. | The truth objects, to 1e-12. | — |
| `test_family_a_treatment_uses_the_engine_form` | The simulator's Emax constants equal the analysis model's fixed ones. | `ed50` and `ec50_ldr` compared with `ModelSpec`'s defaults. | — |

### 7.36 `test_qualification.py` — pooling keys and failed replicates

[`test_qualification.py`](../../tests/test_qualification.py)

| Test | What it proves | How | Not covered |
|---|---|---|---|
| `test_configuration_keys_and_seeds` | Different settings never pool; a power scenario finds its null reference; seeds are unique. | Keys differ when Step 1 or `maxiter` differ, and agree between S1 and S2 with equal settings; 15 tasks (S1 and S2 of family A, S1 of family B) have distinct seeds. | — |
| `test_failed_replicates_enter_the_pool_as_infinite` | A failed bootstrap replicate counts as +∞ in the pooled critical value, which is conservative. | `_rank_quantile([1, 2, None, 3], 0.5)` = 2; `([1, None], 0.95)` = ∞; 18 replicates of 1.0 and 2 failed: critical value ∞, rejection 0, 2 failures. | A failed dataset counted as a rejection; `h_m1`'s verdicts; `power_summary`; `speed_summary`. |

### 7.37 `test_gate.py` — the engine gate's checkpoint and pieces

[`test_gate.py`](../../tests/test_gate.py)

| Test | What it proves | How | Not covered |
|---|---|---|---|
| `test_checkpoint_keeps_finished_sections_and_refuses_other_settings` | A finished section is saved once (numpy values made plain), skipped after a restart with the same settings, and a restart with other settings is refused; without a path nothing is saved. | A section returning numpy values; a second `Checkpoint` on the same file; settings with a tuple and an integer key, which survive the JSON round trip; `n` changed to 11 raises. | The retry loop on Windows' `PermissionError`; a corrupt partial file. |
| `test_dispersed_starts_resumed_midway_give_the_same_fits` | Resuming the dispersed starts after the null fits gives exactly the uninterrupted result. | S1, 80 participants; one dispersed start per model (4 fits); resumed from the snapshot after the second fit; results equal. | That dispersed starts reach the same optimum (the pass rule is not asserted); resuming recovery datasets or profiles. |
| `test_quadrature_check_reports_the_statistic_at_each_node_count` | The quadrature check reports H-P's statistic at 12, 20 and 30 nodes, none negative, with its spread. | S1, 120 participants; a null fit; keys, LR > −1e-6, spread = max − min, a `pass` key. | Whether 12 nodes are enough (the value of `pass` is not asserted). |
| `test_parallel_gate_pieces_match_one_at_a_time` *(slow, not run here)* | Recovery, dispersed starts and profiles give the same results, in the same order, with 1 or 2 jobs. | Recovery of 2 S1 datasets: minimum LR within 1e-3 and mean drift error within 1e-4; log-likelihoods within 1e-3; profiles in order. | — |

### 7.38 `test_transforms.py` — normal scores and orientation

[`test_transforms.py`](../../tests/test_transforms.py)

| Test | What it proves | How | Not covered |
|---|---|---|---|
| `test_normal_scores_midranks_and_frozen` | Normal scores use mid-ranks, $\Phi^{-1}((r-0.5)/n)$; a missing value stays missing; the transform is frozen; the inverse recovers the values. | [1, 2, 2, 3, NaN]: ranks 1, 2.5, 2.5, 4 with n = 4; a new 2.0 gets the same score as the fitted 2s. | A new value outside the fitted range. |
| `test_orientation` | Scales where higher is better are turned round (MoCA, Schwab & England), the others kept. | [26, 90, 3] → [−26, −90, 3]. | — |

### 7.39 `test_ppmi.py` — the PPMI pipeline on the mock download

[`test_ppmi.py`](../../tests/test_ppmi.py). `_build()` writes the mock (§7.27) to a temporary folder and runs `ppmi.build` on it.

| Test | What it proves | How | Not covered |
|---|---|---|---|
| `test_population_phenotypes_and_grid` | The exclusions, phenotypes, grid and WP0 table follow the rules. | Participants 9–11 and 13 excluded for the right reasons; 9 in the primary population; 1 TD and 2 PIGD; participant 3's year 1 is the in-person V04; participant 12 treated with a dose change at year 1 and in PPMI 2.0; participant 5's 24-month event; every domain present; independence at year 0 = 100 − 90. | Real PPMI quirks: other date formats, duplicate or out-of-range item codes, several examinations at one visit, the ON/OFF choice when both exist, indeterminate phenotypes, boundaries of the cut-offs. |
| `test_flow_and_audit_run_on_the_tables` | The flow table counts everyone, and the audit on the mock's table holds no identifier. | Flow total 13; no `"participant"` in the audit's JSON; audit total 9. | — |
| `test_analysis_panel_from_the_mock` | The H-P panel has two groups, the core domains, visits from year 1 in increasing time, scores on the normal-scores scale, and counts that add up. | `ppmi.analysis_panel` on the mock. | The panel's covariates, dose and examination state. |

### 7.40 `test_wp0.py` — disclosure control of the audit

[`test_wp0.py`](../../tests/test_wp0.py). `_table` builds 203 synthetic participants: 3 indeterminate, every fourth PIGD, one in PPMI 2.0 (a small cell), one visit for every seventh participant, two remote year-1 visits, and axial missing at year 2. `_check_cross_table` requires that no table has exactly one suppressed cell and that every row and column with a suppressed cell has its margin suppressed.

| Test | What it proves | How | Not covered |
|---|---|---|---|
| `test_audit_is_aggregate_and_suppressed_cells_cannot_be_recovered` | The audit holds no identifier, and small cells cannot be recovered from the margins. | No "P0" in the JSON; total 203; the participant-flow and landmark tables pass `_check_cross_table`; not exactly one suppressed visits cell; the remote missing share (denominator 2) suppressed; the interval rule passes. | `_check_cross_table` is a necessary condition for protection, not a sufficient one; the overlap section; the quantile rule (at least 50 values); site bands at their boundaries; thresholds other than 5. |
| `test_large_cells_are_shown_exactly` | Large cells are not over-suppressed. | 400 participants, PPMI 1.0 only, no indeterminate: every column total is an integer. | — |

### 7.41 `test_wp1.py` — the consensus rules

[`test_wp1.py`](../../tests/test_wp1.py). `_rater` builds one rater's answers as read from a Section E1 table and an E1b table of added links.

| Test | What it proves | How | Not covered |
|---|---|---|---|
| `test_candidates_read_from_the_questionnaire_use_known_domains` | The 31 candidate links read from the questionnaire use known domains. | `docs/clinical_expert_questionnaire.md`: numbers 1–31; names within the domains and "treatment"; two named links present. | The test fails if the questionnaire changes, by design. |
| `test_answers_are_read_leniently` | Answers are read leniently: Y/yes, N, ?, blank, strengths with a minus sign. | Eight answers, including "2 −" (sign −1), "0" (out) and a blank (missing). | — |
| `test_round_one_needs_every_rater_and_sends_the_rest_to_round_two` | Round 1 needs unanimity; everything else, including links added by a rater, goes to round 2. | Five raters: all 1 or 2 → in; all 0 or N → out; mixed → round 2; an added link → round 2; the tally (4 strong, 1 weak, 5 in); round 2's table holds the right rows. | — |
| `test_round_two_majority_ties_and_the_map` | Round 2 goes by majority; a tie goes to the investigator and is recorded as disputed; fewer than five raters make the map investigator-led. | Four raters in round 2: 3–1 → in; 1–2 with one unsure → "investigator to decide"; with the decision "in" the map lists it as disputed, and the treatment link as improving. | A decision "out"; an even 2–2 split. |
| `test_other_tables_and_csv` | Non-rating tables are ignored, and a CSV answer sheet is read with any arrow. | A Section F table → nothing; "->" and "=>" in a CSV. | Reading Word files (`wp1_consensus.py`'s reader). |

### 7.42 What no test covers

The test tables' last columns, gathered. None of these is a known fault; they are places where a fault would not be caught by a test.
- **No test touches a script.** Nothing in `tests/` imports `scripts/`. `run_hp_real.py`'s checkpoint and resume logic, `run_power_check.summarise` and the gate verdict, `run_engine_gate.py`'s naming, and `md_to_docx.py` are checked only by the examples of this chapter.
- **Resuming H-P.** `sequential_p`'s `resume` argument, the heart of `run_hp_real.py`'s checkpoint, has no test (§7.15 reproduces one edge case).
- **The speed bootstrap with the support re-selected** (`bootstrap_speed(..., support=...)`), the branch `run_hp_real.py` uses, runs on no synthetic data: not in the tests, not in the smoke test.
- **Stopping early in parallel.** When Besag–Clifford stops, `run_hp_real.py` closes the generator and `parallel.ordered` terminates the workers; no test closes the generator early.
- **The gate's resume** is tested for the dispersed starts only, not for recovery datasets or profiles.
- **Qualification summaries:** `h_m1`'s verdicts, `power_summary` and a failed dataset counted as a rejection have no test.
- **The independent likelihood** shares `prepare()`'s year-1 prior and `unpack_numpy`'s parameter mapping with the fast code; the gradient test samples 12 of the 112 or 119 coordinates.
- **Three slow tests** (parallel gate pieces, the speed bootstrap's resume, replicates with 1 or 2 jobs) run only with `PDX_SLOW=1`, and were not run for this guide.

---

### 7.43 The toy simulations at a glance

The toys are small, self-contained simulations, written before the package, that answered the design questions behind protocol v0.4 and v0.5. They number themselves toys 1 to 6: speed against pathway (1), latent classes (2), speed varying within phenotypes (3, three files, called 3a–3c here), the method families (4, three files), tsLDiffPC (5) and DYNOTEARS (6). Their results files sit next to them and are the numbers Appendix E quotes. This guide reads those files; it does not rerun the toys.

| File | Question | Appendix E | Results file | Cohorts per cell |
|---|---|---|---|---|
| `toy_speed_vs_pathway.py` | Does a pure speed difference look like a pathway difference in yearly analyses? | check 1 | `toy_results.json` | 500 |
| `toy_latent_classes.py` | Can latent mechanism classes be told from latent speed classes? | check 2 | `toy_latent_results.json` | 200 |
| `toy_random_speed.py` | With speed varying within phenotypes, which speed-only test holds? | check 3 | `toy_random_speed_results.json` | 100; 50 with a mechanism difference |
| `toy_random_speed_scale.py` | Does the group-level test's excess grow with size? | check 3 | `toy_random_speed_scale_results.json` | 200, 200, 100 |
| `toy_random_speed_scale_person.py` | Does the person-level test stay calibrated as size grows? | check 3 | `toy_random_speed_scale_person_results.json` | 200, 200, 100 |
| `toy_comparators.py` | Do named method families read speed as mechanism; do speed-adjusted versions avoid it? | check 4 | `toy_comparators_results.json`, `toy_comparators_reps.json` | 200 (S0, S1); 100 (S2, S3) |
| `toy_comparators_extra.py` | One-at-a-time interaction tests, node-level score tests, the commutator check | check 4 | `toy_comparators_extra_results.json` | as above |
| `toy_comparators_joint.py` | Per-link and per-domain tests within one joint fit | check 4 | `toy_comparators_joint_results.json` | as above |
| `ts_ldiffpc.py` | tsLDiffPC and tsDCI, re-implemented | check 5 | — | — |
| `toy_tsldiffpc_validation.py` | Does the re-implementation reproduce the paper? | check 5 | `toy_tsldiffpc_validation_results.json` | 30 per setting |
| `toy_tsldiffpc.py` | tsLDiffPC and tsDCI on toy 4's cohorts | check 5 | `toy_tsldiffpc_results.json` | as toy 4 |
| `dynotears.py` | DYNOTEARS, re-implemented | check 6 | — | — |
| `toy_dynotears_validation.py` | Does the re-implementation reproduce the paper? | check 6 | `toy_dynotears_validation_results.json` | 20 per setting |
| `toy_dynotears.py` | DYNOTEARS on toy 4's cohorts | check 6 | `toy_dynotears_results.json` | as toy 4 |

**The model they share.** Toys 1–6 (not the two validation scripts, which use the papers' own generators) simulate the engine's linear stochastic differential equation without its participant load, measurement error or treatment:

$$dX_i=s_i\,(A_gX_i+b_g)\,dt+\sqrt{s_i}\,G\,dW_i,\qquad GG^\top=Q=0.1\,I,\qquad b_g=-A_g\mathbf 1\ (\text{a severe equilibrium at }1),$$

with one speed per group in toys 1 and 2, $\log s_i=\log s_g+0.4\,u_i$ from toy 3 on, and an extra PIGD load on Axial in toys 4–6. It is sampled once a year by its exact transition $X_{t+1}=EX_t+a+w$ with

$$E=e^{sA},\qquad a=A^{-1}(e^{sA}-I)\,b,\qquad w\sim N(0,S),\qquad S=\int_0^s e^{Au}Qe^{A^\top u}\,du,$$

computed from Van Loan's block exponentials or from the stationary identity $S=\Sigma_\infty-E\Sigma_\infty E^\top$, where $A\Sigma_\infty+\Sigma_\infty A^\top=-Q$. Because speed scales the noise variance as well as the drift, the stationary covariance does not depend on the speed: a faster patient moves through the same distribution of states in less time.

**Conventions.**
- They run from `scripts/toy/` (README). Each writes its results file into the current folder, and `toy_comparators_extra.py` reads `toy_comparators_reps.json` from it.
- The default numbers of cohorts in each script's `__main__` are not always those of the results files, and the files do not record the command line; each entry records its own `R`.
- Toys 4–6 share their seeds (20261103 + 1000 × scenario + cohort) and their draw order, so they analyse identical cohorts.
- From toy 3c on, cohorts run in a `multiprocessing` pool with the thread variables set to 1.
- Appendix E lists their limits: linear, Gaussian, with no hidden burden, dropout, ordinal scores or definition artefacts.

The building blocks were checked in seconds with [`ex_toys.py`](../guide_examples/s07/ex_toys.py); its output is quoted below where it belongs.

### 7.44 `toy_speed_vs_pathway.py` — toy 1: speed against pathway

[`toy_speed_vs_pathway.py`](../../scripts/toy/toy_speed_vs_pathway.py)

**Question.** Does a pure speed difference look like a pathway difference in yearly, discrete-time analyses?

**Design.**
- Six domains (Brady, Axial, Cognition, Mood, Function, Independence). $A=-0.2I$ plus seven links: Brady→Axial 0.15, Brady→Function 0.10, Axial→Function 0.20, Function→Independence 0.25, Cognition→Independence 0.15, Mood→Cognition 0.10, Axial→Mood 0.08.
- 800 TD patients at speed 1 and 250 PIGD at speed 2, one speed per group; five transitions; $X_0\sim N(0,0.3^2)$.
- Scenario "speed": the same $A$. Scenario "speed+mech": PIGD also has Axial→Independence 0.12.

Per cohort:
1. per-group ordinary least squares (OLS) of $X_{t+1}$ on $X_t$, with a z-test for each link;
2. the naive split: links significant in one group only;
3. link-wise difference tests: Welch-type z on $\hat B_{PIGD}-\hat B_{TD}$, Benjamini–Hochberg at 5%;
4. the group-level speed-only test: $B_{PIGD}=B_{TD}^{\,s}$ against free matrices, on centred data with the residual covariances profiled out, $-\ell=\tfrac12\sum_g n_g\log|\hat S_g(B_g)|$, against χ² with $P^2-1=35$ df. `mpow` computes the matrix power from an eigendecomposition;
5. the speed-adjusted deviations $(\hat B_{PIGD}-\hat B_{TD}^{\,\hat s})_{jk}/se_{jk}$, and which link has the largest.

**Found** (`toy_results.json`, 500 cohorts per scenario):

| | speed | speed + mech |
|---|---|---|
| link-wise tests flag at least one link | 100% | 100% |
| links flagged (mean) | 4.8 | 6.0 |
| naive split: discordant links (mean) | 3.2 | 3.4 |
| speed-only test rejects | 5.6% (MCSE 1.0%) | 90.8% |
| estimated speed, mean (SD) | 2.009 (0.070) | 2.146 (0.075) |
| largest deviation on Axial→Independence | 18 of 500 (no link favoured) | 485 of 500 (97%) |

**Why it happens.** Every own rate is −0.2 and the links form no cycle, so $A=-0.2I+N$ with $N$ nilpotent, and

$$e^{sA}=e^{-0.2s}\sum_{k\ge0}\frac{s^kN^k}{k!}.$$

A yearly entry carried by paths of length $k$ scales with $s^ke^{-0.2s}$. Doubling the speed multiplies a persistence by $e^{-0.2}=0.819$, a direct link by $2e^{-0.2}=1.637$, and a two-step link by about $4e^{-0.2}=3.27$ ([`ex_toys.py`](../guide_examples/s07/ex_toys.py), part 1):

```text
1. toy 1: yearly matrices of one drift at speeds 1 and 2
   max |B(2) - B(1)^2| = 5.55e-17
       Brady -> Axial        TD 0.1228  PIGD 0.2011  ratio 1.637
       Axial -> Function     TD 0.1637  PIGD 0.2681  ratio 1.637
    Function -> Independence TD 0.2047  PIGD 0.3352  ratio 1.637
       Axial -> Independence TD 0.0206  PIGD 0.0681  ratio 3.301
       Axial -> Axial        TD 0.8187  PIGD 0.6703  ratio 0.819
   mpow on the true B(1), whose six eigenvalues are all e^-0.2: max |mpow(B(1), 2) - B(2)| = 1.86e+42
   mpow on 20 OLS estimates (800 x 5 transitions), s = 2 and 1.7, against scipy: largest error 3.9e-15
```

Axial→Independence has no direct link, only paths through Function (length 2) and through Mood and Cognition (length 3), hence 3.30. One continuous-time mechanism gives three different yearly "effects", which is what the link-wise tests flag.

**How it informed the protocol.** It is the premise of protocol §1.3 and of H-P as a speed-only test in continuous time (Appendix E, check 1).

**Audit notes.**
- The true yearly matrix is defective: six equal eigenvalues with $N\ne0$. `mpow` returns nonsense on it instead of `None`. On the OLS estimates, whose eigenvalues differ, it agrees with scipy's `fractional_matrix_power` to 4e-15. The toy applies it only to estimates, so its results stand (§7.N).
- The deviation's z uses the PIGD estimate's standard error only. It serves only to find the largest deviation, not to test it.

### 7.45 `toy_latent_classes.py` — toy 2: latent speed classes against latent mechanism classes

[`toy_latent_classes.py`](../../scripts/toy/toy_latent_classes.py)

**Question.** Can latent *mechanism* classes be told apart from latent *speed* classes with PPMI-like follow-up?

**Design.**
- Toy 1's system. Each patient is in class 2 with probability 0.4. Class 2 runs at speed 2 ("speed") and also has Axial→Independence ("mech").
- Two designs: 430 patients with seven transitions (like PPMI 1.0) and 950 with two (like PPMI 2.0).
- Expectation–maximisation (EM) with patient-level responsibilities:
  - the free model (class-specific $B$, intercept and covariance, by weighted least squares) from four starts, one of them informed: patients with a larger mean absolute change start in class 2;
  - the speed model ($B_2=B_1^{\,s}$, intercepts and covariances by class) from the free model's best responsibilities and from the informed start. Its M-step for $(B_1,s)$ is at most 60 iterations of the L-BFGS optimiser, within at most 80 EM iterations: a generalised EM.
- Statistic: LR against χ² with 35 df; class recovery by the adjusted Rand index (ARI).

**Found** (`toy_latent_results.json`, 200 cohorts per cell):

| truth, design | rejects (MCSE) | mean LR (35 df) | ARI, free | ARI, speed | estimated speed (median) |
|---|---|---|---|---|---|
| speed, 430 × 7 | 6.0% (1.7%) | 36.0 | 0.76 | 0.77 | 2.02 |
| speed, 950 × 2 | 12.0% (2.3%) | 39.5 | 0.34 | 0.37 | 1.99 |
| mech, 430 × 7 | 69% | 56.5 | 0.82 | 0.83 | 2.17 |
| mech, 950 × 2 | 21% | 43.4 | 0.47 | 0.49 | 2.10 |

**How it informed the protocol.** Data-driven classes (Step 3, protocol §7.4) are exploratory and restricted to PPMI 1.0's long follow-up. Appendix E (check 2) states the simplifications: the χ² reference, $K$ fixed at 2, and no random speed within classes.

**Audit notes.**
- The speed model is maximised less thoroughly than the free model (capped M-steps, 80 EM iterations against 300). An under-maximised restricted fit can only raise the LR, so part of the 12% at two transitions may come from the optimiser rather than from the design.
- Label switching is handled by reporting $\max(s,1/s)$; the ARI ignores labels.

### 7.46 `toy_random_speed.py` — toy 3a: speed varying within phenotypes

[`toy_random_speed.py`](../../scripts/toy/toy_random_speed.py)

**Question.** If speed varies between patients within each phenotype, does the group-level speed-only test still hold, and does a person-level random-speed model?

**Design.**
- Three domains (Axial, Function, Independence): $A=-dI$ with Axial→Function $e_1$ and Function→Independence $e_2$. "Slow": $(d,e_1,e_2)=(0.20,0.20,0.25)$; "fast": $(0.30,0.40,0.40)$, which makes the second-order terms of the matrix exponential large. "speed+mech" adds Axial→Independence in PIGD (0.12 slow, 0.20 fast).
- $\log s_i=\log s_g+0.4\,u_i$ with $s_{TD}=1$, $s_{PIGD}=2$; 400 TD and 200 PIGD patients; five transitions; one exact transition per patient.

Two tests, each against χ² with $P^2-1=8$ df:
- **group-level:** toy 1's matrix-power test;
- **person-level:** H1 (one $A$; PIGD's median log-speed $\gamma$; loads and noise free by phenotype; one spread $\sigma$) against H3 (an $A$ per phenotype). The marginal likelihood integrates the random speed by 12-node Gauss–Hermite quadrature,

$$\ell_g=\sum_{i\in g}\log\sum_{m=1}^{12}w_m\prod_t\phi\big(X_{i,t+1};\ E(s_{gm})X_{it}+a(s_{gm}),\ S(s_{gm})\big),\qquad s_{gm}=e^{\mu_g+\sigma u_m}.$$

H3 is fitted from its own start and from the H1 solution mapped into its parameters (`h1_to_h3`: $A_2=e^\gamma A$, with the PIGD load and noise multiplied by $e^\gamma$), so its likelihood is at least H1's. The gradients are numerical.

**Found** (`toy_random_speed_results.json`; 100 speed-only cohorts, 50 with a mechanism difference):

| cell | person-level rejects | group-level rejects | mean LR, person / group |
|---|---|---|---|
| slow, speed | 7% | 12% | 8.2 / 9.0 |
| fast, speed | 3% | 9% | 8.4 / 9.7 |
| slow, speed + mech | 88% | 84% | 27.7 / 27.1 |
| fast, speed + mech | 100% | 98% | 45.8 / 41.6 |

Every fit converged, and no LR was negative.

**How it informed the protocol.** H-P gives every participant their own speed (protocol §7.3.1, "why speed is per participant"). The protocol's "10.5% at the base size" is the average of this file's slow and fast cells (12% and 9%).

### 7.47 `toy_random_speed_scale.py` — toy 3b: the group-level test at larger sizes

[`toy_random_speed_scale.py`](../../scripts/toy/toy_random_speed_scale.py)

**Question.** Does the group-level test's excess of false rejections grow with the sample size?

**Design.** Toy 3a's speed-only cells at 1, 4 and 16 times the size (400/200, 1,600/800 and 6,400/3,200 patients), with 200, 200 and 100 cohorts. Only the group-level test is run: the person-level model with numerical gradients is too slow at this size.

**Found** (`toy_random_speed_scale_results.json`):

| size | slow: rejects | fast: rejects | mean LR, slow / fast (8 expected) |
|---|---|---|---|
| × 1 | 6.5% | 12.5% | 8.6 / 9.4 |
| × 4 | 10.5% | 19% | 9.3 / 11.2 |
| × 16 | 27% | 65% | 12.0 / 20.8 |

**Why.** The group-level model gives every PIGD patient one speed. With speeds that vary, a group's yearly regression estimates a state-weighted average of its patients' $e^{s_iA}$, which is not a matrix power of TD's average. The misspecification has a fixed size, so the statistic grows in proportion to the number of patients and the rejection rate tends to 1.

**How it informed the protocol.** "Up to 65% in cohorts 16 times larger" (protocol §7.3.1; Appendix E, check 3).

### 7.48 `toy_random_speed_scale_person.py` — toy 3c: the person-level test at larger sizes

[`toy_random_speed_scale_person.py`](../../scripts/toy/toy_random_speed_scale_person.py)

**Question.** Does the person-level random-speed test stay calibrated as cohorts grow, where the group-level test did not?

**Design.** Toy 3b's cohorts, with identical seeds and draw order, analysed by toy 3a's H1-against-H3 test. The likelihood is rewritten in PyTorch for exact gradients. A `check` mode compares the PyTorch likelihood with toy 3a's numpy one.

**Found** (`toy_random_speed_scale_person_results.json`):

| size | slow: rejects | fast: rejects | mean LR, slow / fast | median speed ratio, slow / fast |
|---|---|---|---|---|
| × 1 | 4.5% | 5.0% | 8.19 / 8.29 | 2.014 / 2.011 |
| × 4 | 5.0% | 3.5% | 8.09 / 8.12 | 2.008 / 2.001 |
| × 16 | 3.0% | 7.0% | 8.24 / 7.93 | 2.003 / 2.001 |

That is 46 rejections in 1,000 cohorts, 4.6%. Every fit converged, and no LR was negative.

The check mode ([`ex_toys.py`](../guide_examples/s07/ex_toys.py), part 3):

```text
3. toy_random_speed_scale_person.py's check mode: torch and numpy negative log-likelihoods at the start values
   H1: torch 679.619265  numpy 679.619265
   H3: torch 688.201471  numpy 688.201471
```

**How it informed the protocol.** The engine's participant-level random speed (protocol §7.1, and §7.3.1: "the participant-level test stayed at 4.6%").

### 7.49 `toy_comparators.py` — toy 4: method families against speed-adjusted procedures

[`toy_comparators.py`](../../scripts/toy/toy_comparators.py)

**Question.** Do named difference-graph and shared-plus-deviation methods read a speed difference as a mechanism difference, and do speed-adjusted versions of the same machinery avoid it?

**Design.**
- Toy 1's six domains and seven links; $\log s_i=\log s_g+0.4u_i$; PIGD's load on Axial raised by 0.05; 800 TD and 250 PIGD patients with five transitions.
- The map: the seven links plus Axial→Independence, so 14 support entries with the own rates.
- Scenarios, labelled as in the script:
  - S0: equal speed with the load difference, like protocol §8.2's S0L;
  - S1: PIGD twice as fast;
  - S2: S1 plus a PIGD-only Axial→Independence of 0.12, like protocol §8.2's S3;
  - S3: S1 plus Function→Independence 0.25 → 0.40, like protocol §8.2's S2.

*Naive analogues on the yearly scale*, written from each family's published definition, not the authors' code:
- N1: link-wise interaction tests in a pooled, person-clustered regression of $X_{t+1}$ on $[1,G,X_t,G\cdot X_t]$, Benjamini–Hochberg. N1o: the omnibus Wald test of all interactions.
- N2: node-level shift (the group term and its slopes). N2b: slopes only.
- N3: a difference-graph search. A lagged edge is kept unless some set of other lagged domains makes its coefficient equal across groups (z-test at 5%, no correction, as in the PC algorithm).
- N4: shared effects plus group deviations, lasso with cross-validation (the data-shared lasso). N4b: the same tuned by the Bayesian information criterion (BIC).
- N5: naive continuous time, the matrix logarithm of each group's yearly matrix with person-bootstrap standard errors, Benjamini–Hochberg.
- N6: a permutation test of the phenotype labels for the yearly difference (Frobenius norm and largest entry).
- N7: separate graphs, a link significant in one group only.

*Speed-adjusted versions* (person-level random speed, drift on the map):
- O1: the speed-only LR, 13 df.
- O2: joint discovery, $A_{PIGD}=s(A+D)$ with an adaptive lasso on $D$. The weights are $1/\max(|a_P/s_{rob}-a_T|,10^{-4})$, with $s_{rob}$ the weighted median of $a_P/a_T$. A path of 12 penalties $\lambda_{max}2^{-k/2}$ is followed, each new support refitted without penalty, and the best chosen by $\mathrm{BIC}=2\,\mathrm{nll}+\log(n_{trans})\,|\text{support}|$.
- O3: per-link score tests of $D_{jk}=0$ at the speed-only fit, one link at a time, with Benjamini–Hochberg. The efficient score and information are $u_l=g_l-H_{l\theta}H_{\theta\theta}^{-1}g_\theta$ and $I_l=H_{ll}-H_{l\theta}H_{\theta\theta}^{-1}H_{\theta l}$, and $u_l^2/I_l\sim\chi^2_1$.
- The share of the drift difference that one speed explains: $R^2_{speed}=1-\|A_1-s_{LS}A_0\|^2/\|A_1-A_0\|^2$.

**Found** (`toy_comparators_results.json`):

| Method | S1: shared map links flagged | S2: true link found alone | S3: true link found alone |
|---|---|---|---|
| N1 interaction tests | 57% | 0% | 0% |
| N3 difference graph | 55% | 0% | 1% |
| N4 / N4b lasso | 98% / 92% | 0% | 0% |
| N5 naive continuous time | 63% | 0% | 0% |
| N7 separate graphs | 4.4% | 0% (power 36%) | 0% (power 0%) |
| O2 joint discovery | 0.4% | 96% | 97% |
| O3 score tests, one at a time | 0.6% | 35% | 4% |

- Every yearly method flagged something in 100% of S1 cohorts. N1o, N6 and N2 rejected in every S1 cohort; O1 rejected 6.0% (S0), 6.5% (S1), 98% (S2) and 100% (S3).
- With equal speed and a load difference (S0), the separate graphs disagreed in 99.5% of cohorts, the lasso flagged 28% (N4b) to 71% (N4) of the shared links, and N1o rejected 15%.
- In S1 the median estimated speed was 1.99 and one speed explained a median 95% of the drift difference (6% in S0).
- The speed-adjusted fits took about 30 seconds per cohort.

**How it informed the protocol.** Joint discovery (protocol §7.3.2), the decision to test links jointly rather than one at a time (protocol §7.3.3), the named comparators and their expected behaviour (protocol §8.4), and Appendix E, check 4.

**Audit notes.**
- O2's BIC penalises the logarithm of the number of transitions (`np.log(n_trans)`). The protocol (§7.3.2; `[localisation] bic_units = "participants"`) uses participants and leaves the transitions version as a WP3 comparison. The 0.4% and 96–97% quoted for joint discovery come from the transitions version.
- One-at-a-time score tests spill onto neighbouring links when one link truly differs: 12% (S2) and 17% (S3) of the shared map links were flagged.
- `toy_comparators_reps.json` holds the 600 per-cohort records that `toy_comparators_extra.py` reads.

### 7.50 `toy_comparators_extra.py` — toy 4: three more checks on the same cohorts

[`toy_comparators_extra.py`](../../scripts/toy/toy_comparators_extra.py)

**Question.** How do three more procedures behave on toy 4's cohorts?

**Design.**
- **N1s:** per-link tests that free one group interaction at a time (the per-edge step of score tests in multigroup random-intercept cross-lagged panel models, RI-CLPM, psychonetrics' equality score test and IVPP's partial pruning), Benjamini–Hochberg.
- **O4:** a node-level speed-adjusted score test of all deviations into one domain, own rate included, at the speed-only fit, with Benjamini–Hochberg over domains. It asks FANS's and CD-NOD's question against the speed-scaled reference.
- **O5:** the commutator check. If one mechanism generates both yearly matrices, $B_g=e^{s_gA}$, and they commute, so the statistic is $\|B_{TD}B_{PIGD}-B_{PIGD}B_{TD}\|_F$, with a centred person bootstrap of 200 samples.
- **The footprint:** the off-diagonal yearly entries that one continuous-time difference moves, $|(e^{2A_{PIGD}}-e^{2A})_{jk}|>10^{-3}$. Flags outside it are false even under a footprint-aware reading.

```text
2. yearly entries (off the diagonal) moved by the PIGD-only Axial -> Independence link at speed 2: ['Brady->Independence', 'Axial->Independence']
```

([`ex_toys.py`](../guide_examples/s07/ex_toys.py), part 2.) In S3 the footprint also includes Function→Independence.

**Found** (`toy_comparators_extra_results.json`):

| | S0 | S1 | S2 | S3 |
|---|---|---|---|---|
| N1s: shared map links flagged | 0.2% | 60% | 53% | 53% |
| N1s: true link found alone | — | — | 0% | 0% |
| O4: unchanged domains flagged | 1.7% | 0.9% | 6.0% | 13.8% |
| O4: changed domain named alone | — | — | 73% | 51% |
| O5 rejects | 4.0% | 3.5% | 28% | 78% |

Outside the footprint, the yearly methods still flagged 11–68% of the entries in S2 and S3, joint discovery 0.1%, and the one-at-a-time score tests 1.0–3.0%.

**How it informed the protocol.** The shared-eigenvector check (protocol §7.3.7: 3.5–4.0%, then 28% and 78%) and protocol §8.5's footprint-aware localisation measure. The per-domain tests were moved into the joint fit (§7.51), because O4 spills when a link differs (13.8% in S3).

**Audit notes.**
- It must run from `scripts/toy/`: it reads `toy_comparators_reps.json` from the current folder, and if the file is absent it skips the footprint-aware section without a warning (§7.N).
- Its "any node" for O4 leaves out the changed domain, while `toy_comparators.py`'s N2 counts every domain, so the two files' "any node" figures differ in definition in S2 and S3.

### 7.51 `toy_comparators_joint.py` — toy 4: joint per-link and per-domain tests

[`toy_comparators_joint.py`](../../scripts/toy/toy_comparators_joint.py)

**Question.** Do per-link tests within one joint fit avoid the spill-over of one-at-a-time tests, with the speed anchored or anchor-free?

**Design.**
- **O3j (anchored):** PIGD's drift is $A+D$, with $D$ on the 8 off-diagonal map links and each domain's own rate shared, which anchors the speed. Wald tests $\hat D_l/se_l$ from the inverse Hessian, Benjamini–Hochberg over the 8 links. Per domain: a Wald χ² of all deviations into it.
- **O3m (anchor-free):** free TD and PIGD drifts; the speed by the minimal-L1 rule, $\hat s=\mathrm{wmedian}(a_P/a_T;\ |a_T|)$, and $\hat D=a_P/\hat s-a_T$. The p-values come from 2,000 draws of the free fit's asymptotic normal distribution, $p=(1+\#\{|D^*-\hat D|\ge|\hat D|\})/(M+1)$. Per domain: a Hotelling-type statistic with the draws' covariance.

**Found** (`toy_comparators_joint_results.json`):

| | S0 | S1 | S2 | S3 |
|---|---|---|---|---|
| O3j: shared map links flagged | 0.44% | 1.0% | 1.0% | 1.4% |
| O3j: true link found alone | — | — | 92% | 91% |
| O3m: shared map links flagged | 0.25% | 0.38% | 0.43% | 1.1% |
| O3m: true link found alone | — | — | 94% | 89% |
| per domain, O3j / O3m: unchanged domains flagged | 1.3% / 0.6% | 1.0% / 0.7% | 1.0% / 0.6% | 1.2% / 0.8% |
| per domain, O3j / O3m: changed domain named alone | — | — | 95% / 93% | 94% / 96% |
| joint LR (8 df) rejects | 5.5% | 7.5% | 99% | 100% |

The median minimal-L1 speed was 1.00 in S0 and 1.99 in S1.

**How it informed the protocol.** Protocol §7.3.3, with the anchor-free tests primary (0.25–0.4%; 89–94%) and the anchored ones a sensitivity analysis (0.4–1.0%; 91–92%), and the per-domain tests of protocol §7.3.4 (0.6–1.3%; 93–96%).

**Audit notes.**
- The toy's speed (`to_delta`) uses every support entry, own rates included, with weights $|a_T|$, no near-zero exclusion and no sign filter. The package's `localise.speed_rule` drops reference entries below `near_zero` (0.05) and sign changes ([§5](05_hp_speed_localisation.md)). The synthetic evidence for the anchor-free tests therefore comes from a close but not identical rule.
- The joint LR rejected 7.5% of S1 cohorts (MCSE 1.9%); the protocol does not use or discuss it.

### 7.52 `ts_ldiffpc.py` — tsLDiffPC and tsDCI, re-implemented

[`ts_ldiffpc.py`](../../scripts/toy/ts_ldiffpc.py)

**What it is.** The skeleton, orientation and root causes of tsLDiffPC and tsDCI (Ruer, Loranchet, Bystrova & Assaad 2026), written from the authors' pyCIPHOD code, which was read, not run. The file is byte-identical to `src/pdx/comparators/ts_ldiffpc.py`, comparator C5 ([§4](04_step1_comparators.md)).

**The mathematics.** The data are two-slice tables $Z=[X_{t-1},X_t]$ in each regime. For a candidate edge $x$–$y$ and a set $S$ of other window nodes, the coefficient of $x$ in the OLS regression of $y$ on $[1,x,S]$ comes from the Gram matrix $G=W^\top W$ with $W=[1,Z]$:

$$\hat\beta=G_{II}^{-1}G_{Iy},\qquad s^2=\frac{G_{yy}-\hat\beta^\top G_{Iy}}{n-|I|},\qquad se=\sqrt{s^2\,[G_{II}^{-1}]_{xx}},\qquad z=\frac{\hat\beta^{(1)}-\hat\beta^{(2)}}{\sqrt{se_1^2+se_2^2}}.$$

- The candidates are $p^2$ lagged edges (self-lags included) and $p(p-1)/2$ same-time pairs.
- An edge is removed as soon as some $S$, tried by increasing size up to $2p-2$, gives $p>\alpha$ in either regression direction. There is no multiplicity correction.
- An edge can need up to $2\sum_s\binom{2p-2}{s}=2^{2p-1}$ tests.
- The admissible sets do not depend on the current graph, so both algorithms share the skeleton (`skeleton`); `skeleton_with_sepsets` follows each algorithm's loop order and records the separating sets.
- Lagged edges are oriented by time. Same-time edges are oriented by residual-variance F tests (tsDCI) or by colliders and time-restricted Meek rules 1–4 (tsLDiffPC).
- The root causes are the series heading a directed edge into the current block (strict), plus those in undirected same-time edges (relaxed).

**Worked example** ([`ex_toys.py`](../guide_examples/s07/ex_toys.py), part 5). Two series with yearly matrix [[0.6, 0], [0.3, 0.6]] against the same system run twice as fast (its square), 4,000 transitions each:

```text
5. ts_ldiffpc.skeleton, regime 2 = regime 1 run twice as fast (yearly matrix squared):
   retained lagged difference edges F[to, from]:
   [[1, 0], [0, 1]] | same-time pairs: [] | tests run: 25
   root causes (targets of retained lagged edges): [0, 1]
```

Both self-lags are retained, so both series are named root causes, although there is no mechanism difference: the persistence differences that speed creates.

**Audit notes.**
- Two copies of the same code must be kept in step.
- `wald_p` returns NaN when a fit fails, and `NaN > alpha` is false, so a failed test never removes an edge, as pyCIPHOD skips failed tests.

### 7.53 `toy_tsldiffpc_validation.py` — validating the tsLDiffPC/tsDCI re-implementation

[`toy_tsldiffpc_validation.py`](../../scripts/toy/toy_tsldiffpc_validation.py)

**Question.** Does the re-implementation reproduce the paper's published results (Ruer et al. 2026, Fig. 4)?

**Design.**
- The authors' generator (`setting2_lag1`): $d$ series; lag-1 edges with probability 0.3 per ordered pair plus every self-lag; same-time edges with probability 0.3 per forward pair of a random order, at least one.
- Coefficients ±U(0.2, 0.8), each target's same-time and lagged coefficients scaled to an absolute sum of at most 0.95; $X_t=A_0X_t+A_1X_{t-1}+e_t$ with N(0, 1) noise; two series of 1,000 after a burn-in of 200.
- The second regime changes only same-time coefficients: one edge ("single"), every same-time parent of one node ("all"), or every parent of a node with at least two ("min2"). Each is redrawn ±U(0.2, 1.0) with a change of at least 0.5.
- With one true root cause and a predicted set $P$, $F_1=2/(1+|P|)$ if the truth is in $P$, and 0 otherwise.
- 30 datasets per setting, $d$ = 3, 5, 7 and 9.

**Found** (`toy_tsldiffpc_validation_results.json`).
- Over the 24 settings, the mean absolute difference from the published strict F1 score was 0.10, and the largest 0.21.
- Our standard errors are at most 0.09 (0 where every dataset scored 0). The paper used 10 datasets, so its values carry standard errors of about 0.10–0.15, and every gap is within about 1.5 combined standard errors (`docs/tsldiffpc_tsdci_investigation.md`, §3).
- The pattern is reproduced: tsDCI is high throughout, and tsLDiffPC is near 0 when a single same-time edge changes.
- The median time was 0.2 seconds per fit.

**How it informed the protocol.** It justifies running C5 through the re-implementation (protocol §8.4, "faithful re-implementation"). A check against pyCIPHOD itself waits for its installation (D23).

### 7.54 `toy_tsldiffpc.py` — toy 5: tsLDiffPC and tsDCI on toy 4's cohorts

[`toy_tsldiffpc.py`](../../scripts/toy/toy_tsldiffpc.py)

**Question.** How do tsLDiffPC and tsDCI behave on our design, used naively and against a speed-matched reference?

**Design.**
- **T1:** TD's and PIGD's transitions as the two regimes, every conditioning set (up to 10 nodes at six domains). **T1c:** sets of at most 3 nodes, the cap needed at ten domains.
- **T2:** PIGD's real transitions against a reference sample of TD's mechanism run at each PIGD patient's own speed:
  - TD's drift, load and noise are fitted on TD patients alone, with random speed;
  - PIGD's load, noise, median speed and spread are fitted with TD's drift held fixed;
  - each PIGD patient's speed node is drawn from its posterior given the whole trajectory, $P(m\mid X_i)\propto w_m\prod_t\phi(X_{i,t+1}\mid X_{it},s_m)$;
  - four reference transitions are drawn per real one, from the patient's real states.
- The root causes are the targets of retained lagged edges.

**Found** (`toy_tsldiffpc_results.json`):

| | S0 | S1 | S2 | S3 |
|---|---|---|---|---|
| T1: self-lags flagged | 0.1% | 47% | 43% | 42% |
| T1: false root causes | 0.5% | 48% | 52% | 50% |
| T1: shared map links flagged | 0% | 0.7% | 0% | 0.1% |
| power for the true link, T1 / T1c / T2 | — | — | 49% / 79% / 35% | 10% / 11% / 4% |
| T2: false root causes | 0% | 0.2% | 0.2% | 0.2% |
| T1: median tests per fit | 108.5 | 6,745 | 7,236.5 | 6,755 |

Speed creates persistence differences, so edges survive longer and the number of tests rises from about a hundred to several thousand.

**How it informed the protocol.** C5's expected behaviour in protocol §8.4 ("persistence differences read as root causes in about half of the domains; low power"), the speed-matched arm of protocol §8.4, and Appendix E, check 5.

### 7.55 `dynotears.py` — DYNOTEARS, re-implemented

[`dynotears.py`](../../scripts/toy/dynotears.py)

**What it is.** DYNOTEARS (Pamfil et al. 2020), written from the authors' causalnex code, which was read, not run. The file is byte-identical to `src/pdx/comparators/dynotears.py`, comparator C12 ([§4](04_step1_comparators.md)).

**The mathematics.** With $X$ the current slice ($n\times d$), $Y$ the lagged slices, $W$ the same-time and $A$ the lagged weights:

$$\min_{W,A}\ \frac1{2n}\|X(I-W)-YA\|_F^2+\lambda_W\|W\|_1+\lambda_A\|A\|_1\quad\text{subject to}\quad h(W)=\mathrm{tr}\,e^{W\circ W}-d=0,$$

where $h(W)=0$ exactly when $W$ has no cycle (Zheng 2018).
- $W$ and $A$ are split into non-negative parts ($W=W^+-W^-$), so the L1 terms are linear and L-BFGS-B handles the bounds; same-time self-loops are fixed at 0 by the bounds.
- The augmented Lagrangian adds $\tfrac\rho2h^2+\alpha h$. Its gradient in $W$ is $-(S_{xx}M-S_{xy}A)+(\rho h+\alpha)\,2\,(e^{W\circ W})^\top\circ W$, with $M=I-W$ and $S_{xx}=X^\top X/n$, $S_{xy}=X^\top Y/n$, $S_{yy}=Y^\top Y/n$. The loss is computed from these cross-products, which is algebraically the authors' residual form.
- $\rho$ starts at 1 and is multiplied by 10 until $h$ falls below a quarter of its previous value (while $\rho<10^{20}$); then $\alpha\leftarrow\alpha+\rho h$. The loop stops when $h\le10^{-8}$ or after 100 dual steps.
- Edges below the thresholds $\tau_W$ and $\tau_A$ are set to 0 at the end.

**Worked example** ([`ex_toys.py`](../guide_examples/s07/ex_toys.py), part 4):

```text
4. dynotears.gradient_check(): largest relative error 4.51e-10 | with d = 5, two lags: 1.28e-09
```

**Audit notes.** The authors' code applies one threshold to both matrices; the paper's simulations use separate ones, so both are arguments here.

### 7.56 `toy_dynotears_validation.py` — validating the DYNOTEARS re-implementation

[`toy_dynotears_validation.py`](../../scripts/toy/toy_dynotears_validation.py)

**Question.** Does the re-implementation reproduce the paper's published results (Pamfil et al. 2020, Fig. 3, panels a and c)?

**Design.**
- The paper's Appendix B.4 and the causalnex generator. The same-time graph is an Erdős–Rényi directed acyclic graph (DAG) of mean degree 2 (lower-triangular Bernoulli($k/(d-1)$) entries, rows and columns permuted), with weights ±U(0.5, 2.0).
- The lagged graph has Bernoulli($k/d$) entries, self-lags included, with weights ±U(0.3, 0.5).
- $X_t=X_tW+X_{t-1}A+Z_t$ with N(0, 1) noise, one series of $n$ after a burn-in of $100+\max(n/10,50)$. A draw is rejected if a value is non-finite or at least 1,000, or if a graph is empty.
- Settings from the paper: $n=500$ with λ = 0.05, $\tau_W$ = 0.3, $\tau_A$ = 0.1; $n=50$ with λ = 0.2, $\tau_W$ = 0.3, $\tau_A$ = 0.2. Then $d$ = 5, 10, 20, 50 and 20 datasets per setting (the paper used 5).
- F1 on directed edges, separately for the same-time and the lagged matrices.

**Found** (`toy_dynotears_validation_results.json`):
- the mean absolute difference from the published F1 was 0.018, and the largest 0.067 (lagged edges, $n=50$, $d=20$: 0.70 against 0.77);
- every fit converged, and the gradient matched central differences to 1.3e-9;
- a fit took 44 seconds at $d=50$.

**How it informed the protocol.** It justifies running C12 through the re-implementation, with no installation (protocol §8.4; Appendix E, check 6).

### 7.57 `toy_dynotears.py` — toy 6: DYNOTEARS on toy 4's cohorts

[`toy_dynotears.py`](../../scripts/toy/toy_dynotears.py)

**Question.** What does an analyst running DYNOTEARS unmodified conclude about phenotype differences on our design?

**Design.**
- One graph per phenotype from the stacked two-slice tables, on scores standardised over the whole cohort.
- **C12:** the paper's large-sample settings (λ = 0.05, $\tau_W$ = 0.3, $\tau_A$ = 0.1), the pre-specified comparator. **C12d:** causalnex's defaults (λ = 0.1, no threshold).
- A link present in one phenotype's graph and absent from the other's is read as a mechanism difference. Same-time edges are compared as directed edges and reported separately.

**Found** (`toy_dynotears_results.json`, C12):

| | S0 | S1 | S2 | S3 |
|---|---|---|---|---|
| any discordant lagged link | 42.5% | 99.5% | 99% | 99% |
| shared map links flagged | 6.1% | 32% | 31% | 27% |
| any discordant same-time edge | 0% | 44.5% | 38% | 26% |
| median cross-links, TD / PIGD | 0 / 0 | 0 / 3 | 0 / 3 | 0 / 3 |
| true link flagged / found alone | — | — | 25% / 0% | 84% / 0% |

The slower phenotype's yearly effects fall below the threshold, so TD's graph usually has no cross-link. At the package's defaults (C12d), PIGD's same-time graph grows to a median of 18 edges in S1, against 4 for TD.

**How it informed the protocol.** C12's row in protocol §8.4 and Appendix E, check 6.

---

### 7.N Notes for the author

Things noticed while reading the scripts, the tests and the toys. Nothing was changed. The first group can affect a reported number or the reproducibility of a run; the rest are coverage gaps and documentation to bring in line.

**Reproducibility of the PPMI run (`run_hp_real.py`).**

1. **The Sp-1 interval drops its conditioning report** (`run_hp_real.py:94`). `sp1_interval` keeps `covariance_checked(...)[0]` and discards the report: the smallest eigenvalue, the condition number, and the flags for positive definiteness and good conditioning. `covariance_checked`'s own docstring says the report must travel with any interval computed from it. On the synthetic S1 panel of §7.12, the speed-only information had a condition number of 1.9e11 (`well_conditioned: False`), and the interval was printed without a flag (`ex_first_look_synthetic.py`). Storing the report next to `sp1_interval_model_based` would close this.

2. **The reports do not record the seed, the Step 1 options, the engine settings or the code version** (`run_hp_real.py:187`, `:236`). The first-look and full reports hold the approval text, the map, the counts and the results. They do not hold `--seed`, `--interval-in-step1`, `--dose-in-step1`, `n_quad`, `--jobs` or the commit. Protocol §16 asks for seeds and configurations to be logged for every fit, and `run_engine_gate.code_version()` already exists.

3. **The H-P checkpoint is keyed only by the observed statistic** (`run_hp_real.py:219`). The seed, `n_max`, $h$, the Step 1 options and the speed bootstrap's size are not stored.
   - A resume with another `--seed` mixes two replicate streams.
   - A resume with another `--interval-in-step1` that leaves the observed support unchanged mixes replicates selected under two rules.
   - A resume with a smaller `--n-max` than the number of saved replicates counts them all but uses the new cap: `sequential_p(..., n_max=5, resume=[(0.1, True)] * 10)` returns p = 1/6 with 10 replicates drawn, where 10 draws without stopping give 1/11 (§7.15).
   - A smaller `--speed-bootstrap` reports `"B"` below `"B_used"`.

   Storing a settings dict and refusing a mismatch, as `gate.Checkpoint` does, would remove all four.

4. **A change of threads or machine will probably discard the H-P checkpoint** (`run_hp_real.py:219`). The resume tolerance is 1e-6 on the observed statistic. The batch size follows the thread count, which changes the order of summation, and the slow test `test_parallel_gate_pieces_match_one_at_a_time` allows 1e-3 between a threaded and a single-threaded statistic. The safe outcome is a restarted bootstrap, which can cost days. The runbook (§6) and the Windows guide (§7c) say "run the same command again"; adding "with the same `--threads`, on the same machine" would warn of this.

5. **An unedited WP1 draft map would not be labelled exploratory** (`run_hp_real.py:235`). Only statuses starting with "PROVISIONAL" or "TEMPLATE" give "EXPLORATORY: …". `wp1.build_map` writes the status "draft: frozen only when the investigator records it in the protocol (§17)", so a run on the draft would be labelled "map as given".

**The WP0 power gate and the synthetic settings.**

6. **The power check pools every dataset file, whatever its configuration** (`run_power_check.py:124`). `summarise` reads every JSON in `results/wp0/power/datasets/` and filters only by scenario and error. A run with another `--d`, or after `--write-settings` with new WP0 counts, is pooled with the old files without a warning. `run_qualification.py` avoids this by grouping on the configuration key, which the file names already carry.

7. **`failed` mixes scenarios, and failed S2 datasets are dropped** (`run_power_check.py:84`). Power is therefore conditional on a successful fit; counting a failed S2 dataset as a non-rejection would be the conservative choice, mirroring the calibration's treatment of failures as rejections.

8. **The noncentrality assumption is not checked** (`run_power_check.py:80`). $\hat\lambda=\overline{LR-df}$ assumes $E[LR]=df$ under the null. With `--null`, the S1 rows hold `lr` and `df`, and reporting their mean of LR − df would check it.

9. **`--wp0 --d 4` gives two domains the wrong missing share** (`run_engine_gate.py:62`, `run_power_check.py:117`). `p_missing[:d]` takes the ten shares by position, in the 10-domain order. The 4-domain preset's domains are tremor, axial, function and independence, so function receives brady_rigidity's share and independence receives function's (`ex_gate_settings.py`, part 2). `--d 4` is `run_engine_gate.py`'s default. The planned runs use `--d 10`, which is unaffected, as is `--d 6`.

10. **The settings file's note has a fixed date** (`run_power_check.py:62–65`). `write_sim_settings` always writes "the WP0 audit of 3 November 2026", so a refresh from a later audit carries the wrong date.

**The engine gate.**

11. **`code_version` describes only the last session** (`run_engine_gate.py:77`). It runs once when the report is built, at the start of the session that finishes the gate. Sections finished in earlier sessions, possibly under other code, are not distinguished. Outside a git checkout (with git installed), `commit` is the empty string and the diff hash `None`, which reads as a clean checkout.

12. **The quick names lack `d`** (`run_engine_gate.py:66`). A quick run with another `--d` is refused until the partial file is moved, and finished quick reports overwrite each other.

13. **The Windows guide says that, after the first sections, "the rest run in parallel"** (`docs/run_on_windows.md`, §3). The last section, the timing of one whole bootstrap replicate (`gate.py:393`), runs on its own.

**Parallel jobs.**

14. **`run_tasks` caps the workers at 4** (`qualification.py:110`). `run_power_check.py --jobs 8` and `run_qualification.py --jobs 5` silently use 4, while `parallel.ordered` (the gate and H-P) has no cap. The Windows guide recommends `--jobs 5` only for the gate and H-P, which is consistent, but neither script's help text mentions the cap.

**Tests.**

15. **The test count** (runbook §1, README). Both say 49 unit tests; the runner finds 53, 50 of them run by default and 3 slow. The code map and the README give about 2 minutes; the run here took about 3.

16. **Coverage gaps** (§7.42). No test imports a script. `sequential_p`'s `resume` argument, `bootstrap_speed` with the support re-selected (the branch `run_hp_real.py` uses, run on no synthetic data at all, smoke test included), the early termination of `parallel.ordered`, the gate's resume of recovery datasets and profiles, and `h_m1`/`power_summary` have no test. The reference likelihood shares `prepare()`'s year-1 prior and `unpack_numpy` with the fast code, and the gradient test samples 12 of the 112 or 119 coordinates. Small synthetic tests of `sequential_p(resume=...)` and of `bootstrap_speed(support=...)` would close the two gaps nearest the PPMI run.

17. **The runner's switches** (`run_tests.py:31`). `PDX_SLOW=0` also turns the slow tests on (any non-empty value does), and `python -O` removes every `assert`, so every test would pass.

**Toys.**

18. **The toys depend on the current folder** (`toy_comparators_extra.py:151–155`). Every toy writes its results file into the current folder, and `toy_comparators_extra.py` reads `toy_comparators_reps.json` from it, skipping the footprint-aware section silently if the file is not there. The README says to run them from `scripts/toy/`, but nothing enforces it.

19. **`mpow` fails on defective matrices** (`toy_speed_vs_pathway.py:88`). The eigendecomposition gives nonsense (an error of 1.9e42) on the true yearly matrix, whose six eigenvalues are equal, instead of returning `None`. On estimates it is exact to 4e-15, and the toys use it only on estimates (toys 1–3), so no result changes. scipy's `fractional_matrix_power` would be safe if the function is reused.

20. **Two toy procedures differ from the protocol's.** Toy 4's joint discovery tunes BIC with the number of transitions (`toy_comparators.py:475`), while protocol §7.3.2 uses participants. The anchor-free tests' speed (`toy_comparators_joint.py:91–99`) has neither the package's near-zero exclusion nor its sign filter. The synthetic figures quoted in protocol §7.3.2–§7.3.3 come from these close but not identical procedures.

21. **"10.5% at the base size"** (protocol §7.3.1 and Appendix E, check 3) is the average of `toy_random_speed.py`'s slow and fast cells (12% and 9%). `toy_random_speed_scale.py`'s own base-size cells gave 6.5% and 12.5%. A sentence naming the source file would let a reader reproduce it.

**Documentation.**

22. **The map chapter's output folder for WP3** ([§0.2](00_map.md#02-the-pipeline-at-a-glance) lists `results/wp3/`). `run_qualification.py` writes `results/qualification/` (`run_qualification.py:17`).

23. **`md_to_docx.py`'s dependency and docstring.** python-docx is in neither `pyproject.toml` nor the ppmi-causal environment; the commands use the system Python, which has it on this Mac, while the Windows set-up (`pip install -e ".[test]"`) does not install it. The docstring's example still names the v0.2 protocol file.

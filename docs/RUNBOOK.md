# Runbook: running the phenotype-discovery experiment

> Protocol draft v0.5. Written 3 November 2026. Step-by-step instructions for every work package: what to run, in which order, what it needs, how long it takes, and what to check. `docs/code_map.md` maps protocol sections to code.

## 0. Before you start

**Where.** Run every command from the repository folder, in the Terminal app:

```
cd "/Users/thaipham/Documents/PhD/parkinson/paper 2/phenotype-discovery"
PY="../ppmi-causal/.venv/bin/python"
export PYTHONHASHSEED=0      # Step 1 gives the same answer on every run (code guide §8, A8)
```

The code uses the ppmi-causal Python environment (Python 3.13 with numpy, scipy, torch, pandas, causal-learn and statsmodels); nothing new needs installing. pytest is not installed, so the tests have their own runner.

**On a Windows PC,** follow `docs/run_on_windows.md`: a one-time setup, then the same scripts. It suits the synthetic runs (WP2, WP3); the PPMI runs stay on this Mac.

**Rules that every step follows** (protocol §14):
- **Approval.** A command that reads PPMI participant data refuses to start without `--approval "<who, date, document>"`. Give it only once the investigator's written approval exists. WP0 was approved in the chat on 3 November 2026.
- **Where data live.** Participant-level tables stay in `data/` (git-ignored). Results in `results/` are aggregates only and are git-ignored too, until you decide what to share. Never copy anything from `data/` elsewhere, and never paste it into an AI tool.
- **Long runs.** Start them from the Terminal app, with at most 4 parallel jobs. Keep the Mac awake with `caffeinate`. Every long run saves results as it goes and skips finished work when restarted.

**Measured cost** on the M4, for budgeting:

| Unit | 4 domains, about 900 participants | 10 domains, about 1,450 participants |
|---|---|---|
| One likelihood evaluation | 0.1–0.2 s | 2.5 s |
| Speed-only fit | about 30 s | about 9 minutes |
| Free fit (two starts) | about 1–2 minutes | about 30 minutes |
| One H-P bootstrap replicate (support selection and refits) | about 1.5 minutes | about 30–40 minutes |
| One fit pair (speed-only and free) in a 4-job pool, 816 participants | not timed | 2–5.4 hours each (median 3.4) at the old batch size, so about one an hour in all; about two an hour now |

**Batch size (measured on 4 November 2026, `scripts/bench_threads.py`).** The engine is limited by memory traffic, not by the number of cores, so the number of participants in each batch of the likelihood matters most. It changes only the order of summation: the values are identical.

| Layout | 512 per batch (before) | 128 | 32 |
|---|---|---|---|
| 4 single-threaded jobs (power check, WP3) | 1,861 evaluations an hour | 2,735–3,184 | **3,518–3,634** |
| 1 process with 4 threads (gate, H-P on PPMI) | 1,439 | **2,385** | 1,912 |

- **The default follows the threads** (`likelihood.default_chunk()`): 32 for a single-threaded job, 128 otherwise.
- **Other layouts were worse.** At the old batch size, 8 jobs managed 1,492 an hour, 2 jobs of 2 threads 1,552, and one process with 10 threads 1,420.

**The protocol's §8.8 assumed 1 minute per fit.** With the new batch size, WP3's replicate numbers for family A at 10 domains would take about 1,700–2,000 hours on this Mac (10–12 weeks), so the plan must be revised after the engine gate (D14).

## 1. Check the installation (minutes)

```
$PY tests/run_tests.py                  # 50 unit tests, about 3 minutes; all must pass (PDX_SLOW=1 adds 3 slow ones)
$PY scripts/smoke_test.py --n 300       # the whole chain on one small synthetic dataset, about 15-25 minutes
$PY scripts/run_engine_gate.py --quick  # the engine-gate harness at small size, about 15-25 minutes
```

**Outputs.** `results/smoke/summary.json` and `results/engine_gate/gate_quick.json`.

**Check.** In the quick gate, the reference agreement should be below 1e-8, the gradient check below 1e-4, and the quadrature spread below 0.01. At this small size a profile may show "not identified"; that is expected.

## 2. WP0: feasibility audit (done on 3 November 2026)

Rerun these after a new PPMI download, or if the rules in `src/pdx/ppmi.py` change.

```
AP="<investigator, date, approval document>"
$PY scripts/build_ppmi_tables.py --approval "$AP"                      # seconds; reads ../PPMI/raw
$PY scripts/run_wp0_audit.py --visits data/derived/wp0_visits.parquet --approval "$AP"
$PY scripts/run_power_check.py --datasets 8 --jobs 4                    # pilot, about 9 hours (synthetic)
$PY scripts/run_power_check.py --datasets 20 --jobs 4                   # optional direct Monte Carlo check, about half a day more
```

**What each step does.**
- The table build writes participant-level tables to `data/derived/` (`participants.parquet`, `grid.parquet`, `wp0_visits.parquet`), and to `results/wp0/` the flow (`participant_flow.json`), the phenotype axes (`phenotype_axes.json`) and the provisional rules' provenance (`manifest.json`, with file hashes).
- The audit writes `results/wp0/wp0_audit.json`, with small cells suppressed.
- The power check reads only the audit's aggregate counts and writes `results/wp0/power/summary.json`.

**Read.** `docs/wp0_feasibility_report.md` summarises the results and the decisions they inform (D5, D12, D13, D16, D26, power).

**The WP0 gate.** H-P needs at least 80% power at α = 5% against one deviation of 2ε. The power check prints its verdict as "pass", "fail" or "undecided: run more datasets". If it fails, the protocol says so before WP4, and a non-rejection is read with that power in mind (§8.7).
- **The verdict comes from the noncentral estimate.** It is precise from a few datasets, and the pilot passed with power above 99% (WP0 report, §7).
- **The 20-dataset run is optional.** Its datasets that all reject would confirm 80% power from the rejection rate alone. A rerun skips the datasets already finished.

## 3. WP1: clinical consensus (weeks; a script compiles the answers)

1. Fill in the return date and contact in `docs/clinical_expert_questionnaire.md` (v0.2). Then rebuild the Word version:

```
/usr/bin/python3 scripts/md_to_docx.py docs/clinical_expert_questionnaire.md docs/Clinical_Expert_Questionnaire_Round1_v0.2.docx "Phenotype causal discovery in PD · expert questionnaire, round 1 · v0.2"
```

2. Send round 1 to the neurologists, with the email drafted in `docs/wp1_invitation_email.md`. Ask about ethics review for an expert survey first (§14). Aim for at least five raters; with three or four, the map is investigator-led (D25).
3. Save each returned questionnaire in `data/wp1/round1/` (git-ignored: the files name the raters), then compile round 1 with the consensus rules of §6.4:

```
/usr/bin/python3 scripts/wp1_consensus.py round1 data/wp1/round1/*.docx
```

   - **What it does.** It reads Section E's ratings from each Word file and prints how many links each rater answered: check any rater with gaps.
   - **What it writes.** `results/wp1/round1.json` (anonymised counts) and `results/wp1/round2_section_E.md`, the table of links that return in round 2, with the group's round-1 answers.
   - **Without Word files.** A rater can also answer on a CSV sheet: `scripts/wp1_consensus.py sheet data/wp1/answer_sheet.csv` writes a blank one.
   - **Round 2's questionnaire** combines:
     - that table;
     - the other sections' disagreements;
     - ε in clinical units, from §5 of the WP0 report;
     - the request for forecasts (D21).
4. Save the round-2 answers in `data/wp1/round2/` and compile them:

```
/usr/bin/python3 scripts/wp1_consensus.py round2 data/wp1/round2/*.docx --decisions data/wp1/decisions.json
```

   - **The majority decides.** A link without a majority needs your decision in `decisions.json` (`{"from->to": "in"}` or `"out"`) and is recorded as disputed.
   - **The draft map** goes to `configs/expert_map_draft.json`: domain links, treatment links, disputed links, and whether the map is investigator-led.
5. Freeze the decisions:
   - the domains (D6);
   - the expert map: review the draft and save it as `configs/expert_map.json`. The H-P run uses its links between the panel's domains and lists the others it sets aside (treatment links, and domains left out under D26);
   - the confirmatory links (D7), ε (D12), the motor state (D5) and the floor-bound domains (D26);
   - the measurement-error variances (D28), in `configs/measurement_error.json`, from `docs/measurement_reliability.md`.
6. Record every decision in the protocol (§17) and commit.

## 4. WP2: engine gate and protocol freeze (days of compute)

The full gate runs on synthetic data at the real size, so no approval is needed. `--wp0` gives every dataset WP0's group sizes, visits and missingness (652 TD and 164 PIGD participants with at least one transition), as the power check does. The settings are read from `configs/wp0_sim_settings.json`, so any computer with the repository can run the gate. Set `--d` to the number of frozen domains; with 10 domains it takes about 1.5–2 days. On a Windows PC, follow `docs/run_on_windows.md`.

```
caffeinate -i nohup $PY scripts/run_engine_gate.py --wp0 --d 10 --datasets 20 > results/engine_gate/console.out 2>&1 &
```

**Progress.** A line for each finished piece is appended to `results/engine_gate/gate_wp0_d10.log` (follow it with `tail -f`). Some pieces take hours.

**Parallel jobs.** `--jobs N` runs N dispersed starts, profiles or recovery datasets at once, each in a single-threaded worker process. Every piece has its own seed. Any N of 2 or more gives identical results; one job, which runs in the main process with its threads, agrees with them to the fits' numerical tolerance (below 1e-4 in a statistic). A restart may use another N. On this Mac, 4 jobs did 1.5 times the work of one 4-thread process (§0). `scripts/bench_threads.py --configs 4x1 5x1 6x1 1x6` finds the best N on another machine.

**If it stops,** run the same command again. Finished sections, dispersed-start fits, recovery datasets and profiles are saved in `results/engine_gate/gate_wp0_d10.partial.json` and skipped. The file is removed when the report is written.

**Output.** `results/engine_gate/gate_wp0_d10.json`. It records the code it ran: the commit, plus a hash of any uncommitted changes.

**The first run (4 November 2026).** The Mac finished the reference, gradient and quadrature checks; the run was then moved to a Windows PC.
- The reference and gradient checks pass.
- **The quadrature check fails its rule.** At WP0's size (816 participants), the speed-only fit's log-likelihood at 12 nodes differs by 0.24 from the 20- and 30-node values, against a limit of 0.01. The 20- and 30-node values agree to within 0.0095.
- **H-P's statistic barely moves.** It is 29.33, 29.39 and 29.38 at 12, 20 and 30 nodes.
- **To decide before the freeze:** move to 20 nodes (roughly 1.7 times the compute), or keep 12 and judge agreement by the statistic.

**It passes (§16) if:**
- the automatic checks pass: reference agreement, gradients, quadrature, dispersed starts, conditioning with any spreads at zero set aside, profiles, and no negative likelihood ratio;
- the investigator and statistician accept the recovery bias and coverage.

**Then:**
- decide the number of quadrature nodes (the first run, above);
- revise the compute plan (D14) from the gate's `timing` block;
- decide whether the free model's second start is kept; the gate reports `max_gain_from_free_default_start`;
- register the protocol and analysis plan on OSF (stage 1, D11).

## 5. WP3: method qualification (weeks of compute; synthetic)

Each command runs one family and set of scenarios. Results go to one file per dataset in `results/qualification/datasets/`, and a restart skips finished datasets. Replicate numbers come from §8.6; cut them in the order of §8.8 if the gate's timing requires it.

```
caffeinate -i nohup $PY scripts/run_qualification.py --family A --scenarios S0 S1 --datasets 500 --n 900 --d 10 --jobs 4 > results/qualification/a_main.log 2>&1 &
$PY scripts/run_qualification.py --family A --scenarios S0L S1S S1T S1V S4 S7 --datasets 100 --n 900 --d 10 --jobs 4
$PY scripts/run_qualification.py --family B --scenarios S0 S1 --datasets 300 --n 900 --d 4 --jobs 4
$PY scripts/run_qualification.py --family A --scenarios S2 S2b S3 --datasets 200 --n 900 --d 10 --jobs 4
$PY scripts/run_qualification.py --summarise --config-a <key> --config-b <key>
```

- **Stress cells.** Use `--sim-kw`, for example `--sim-kw '{"spurious": true}'`, or `{"noninvariant": 0.08}` for family B.
- **Reading the summary.** `results/qualification/summary.json` has, by family, scenario and configuration key:
  - the rejection rates with their Monte Carlo standard errors;
  - the H-M1 verdict (Bradley's 7.5%, with the partial pass for S4);
  - power against the S1 critical value;
  - the bias of Sp-1.
- **Configuration keys.** Pass `--config-a`/`--config-b` once several configurations exist (for example trial runs).
- **Not yet in code.** The simulators use preset parameters; fitting them to PPMI's pooled population (§8.2) needs approval and is still to write. So are comparators C1, C2, C4, C6, C8 and C9 (software installs, D23) and the wrappers for C7 and C10.

Then log WP3's choices in the time-stamped stage-2 amendment before WP4 (§3).

## 6. WP4: PPMI analyses (needs approval, WP1's frozen map and a passed H-M1)

```
AP="<investigator, date, approval document>"
$PY scripts/build_analysis_panel.py --approval "$AP" --domains <WP1 domains> --measurement configs/measurement_error.json
caffeinate -i nohup $PY scripts/run_hp_real.py --expert configs/expert_map.json --approval "$AP" > results/wp4/hp.log 2>&1 &
```

**The panel build** writes `data/derived/panel_tdpigd.npz`, `panel_spec.json` and the frozen normal-scores tables. It covers TD and PIGD participants seen at year 1. Add `--motor-on` if D5 lets ON-only examinations enter with their state recorded.

**The H-P run:**
- **Steps.** It runs H-P with its sequential bootstrap (support re-selected in every replicate), then the speed estimates with 200 participant-bootstrap samples, then the anchor-free per-link and per-domain tests with the link classes.
- **Checkpointing.** H-P's bootstrap and the speed bootstrap save `results/wp4/hp_tdpigd_checkpoint.json` after every replicate or sample. If the run stops, run the same command again: it resumes when the observed statistic is unchanged.
- **Progress** is printed and appended to `results/wp4/hp_tdpigd.log`.
- **Shorter runs.** `--n-max` caps H-P's bootstrap, for example at 199, which gives p-values down to 0.005. `--speed-bootstrap` sets the number of speed samples. The report records both.
- **Restarting.** Restart with the same command on the same machine. On a restart the observed fits are recomputed, and the saved replicates are kept only if the statistic matches to 1e-6: a change of `--threads` or of machine can shift it by more and discard them. Do not lower `--n-max` during a run.
- **Parallel jobs.** `--jobs N` runs N bootstrap replicates, then N speed samples, at once. Each replicate has its own seed, and Besag–Clifford's rule is applied in replicate order, so the p-value does not depend on N (to the fits' numerical tolerance).
- **Output.** `results/wp4/hp_tdpigd.json`, aggregates only.
- **Duration.** At 10 domains, the replicates run one at a time, at roughly 30–40 minutes each.
  - **No clear rejection: a day or two.** H-P stops early once 20 replicates are at least as large as the observed statistic.
  - **A rejection: 3–4 weeks.** It runs all 999 replicates.

Step 1's reported map, the RBD axis and the robustness and sensitivity analyses (§7.10–§7.14) are run with the same modules, as set out in the analysis plan registered at WP2.

**The exploratory first look (4 November 2026).** On the investigator's instruction, H-P was run before WP1–WP3. It used the provisional map in `configs/expert_map_provisional.json`, which was built from the earlier v4.1 expert map and the clinical guidelines; the file gives the reasons for each link. The panel had the default domains and OFF-state motor examinations only.

```
$PY scripts/run_hp_real.py --expert configs/expert_map_provisional.json --first-look --approval "$AP"
```

`--first-look` stops after the observed fits, about an hour. It reports:
- H-P's statistic with a chi-square p-value instead of the bootstrap's;
- the speed estimates, with a model-based interval for Sp-1;
- the per-link and per-domain localisation.

The output is `results/wp4/hp_first_look.json`, labelled exploratory. Any result obtained this way is exploratory: the confirmatory analysis would need the frozen map, a passed H-M1 and registration first. To run the PPMI analysis on a Windows PC, see `docs/run_on_windows.md`, §7.

## 7. When something goes wrong

- **A run stopped (power cut, sleep, crash).** Start the same command again; finished datasets and replicates are skipped.
- **The Mac is slow or hot.** Use fewer jobs (`--jobs 2`). Each job runs single-threaded inside the pool.
- **A script refuses to start.**
  - It needs `--approval` for PPMI data;
  - inputs must sit inside `data/` and outputs inside `results/`;
  - the refusal message names the problem.
- **Tests fail after a change.** Fix them before running anything long. Commit only with the investigator's say-so (§14).
- **A new PPMI download.** Rebuild the tables, rerun the audit, and compare `results/wp0/manifest.json` with the previous one.

## 8. Data-governance checklist (before sharing anything)

- [ ] Only files from `results/` or `docs/` leave the machine, and they hold aggregates only.
- [ ] Small cells are suppressed (WP0's rule: counts below 5 and any cell that would reveal one). Confirm this threshold against PPMI's guidance.
- [ ] The GitHub repository is private. The documents quote PPMI aggregate counts.
- [ ] iCloud "Desktop & Documents" sync is off for this folder; otherwise `data/` would sync to Apple's servers.
- [ ] Manuscripts go to the PPMI Data and Publications Committee with the required acknowledgement (README).

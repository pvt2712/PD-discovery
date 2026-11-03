# Runbook: running the phenotype-discovery experiment

> Protocol draft v0.5. Written 3 November 2026. Step-by-step instructions for every work package: what to run, in which order, what it needs, how long it takes, and what to check. `docs/code_map.md` maps protocol sections to code.

## 0. Before you start

**Where.** Run every command from the repository folder, in the Terminal app:

```
cd "/Users/thaipham/Documents/PhD/parkinson/paper 2/phenotype-discovery"
PY="../ppmi-causal/.venv/bin/python"
```

The code uses the ppmi-causal Python environment (Python 3.13 with numpy, scipy, torch, pandas, causal-learn and statsmodels); nothing new needs installing. pytest is not installed, so the tests have their own runner.

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
| One fit pair (speed-only and free) in a 4-job pool, 816 participants | not timed | 2–5.4 hours each (median 3.4), so about one an hour in all |

The protocol's §8.8 assumed 1 minute per fit. At the pool's rate, WP3's replicate numbers for family A at 10 domains would take about 3,500–4,000 hours (about 5 months) on this Mac, so the plan must be revised after the engine gate (D14).

## 1. Check the installation (minutes)

```
$PY tests/run_tests.py                  # 41 unit tests, about 2-3 minutes; all must pass
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

## 3. WP1: clinical consensus (weeks; no code needed)

1. Fill in the return date and contact in `docs/clinical_expert_questionnaire.md` (v0.2). Then rebuild the Word version:

```
/usr/bin/python3 scripts/md_to_docx.py docs/clinical_expert_questionnaire.md docs/Clinical_Expert_Questionnaire_Round1_v0.2.docx "Phenotype causal discovery in PD · expert questionnaire, round 1 · v0.2"
```

2. Send round 1 to the neurologists: at least five, at least three (D25). Ask about ethics review for an expert survey first (§14).
3. Compile the answers with the consensus rules of §6.4. Round 2 shows:
   - the anonymised group answers;
   - ε in clinical units, from §5 of the WP0 report;
   - the request for forecasts (D21).
4. Freeze the decisions:
   - the domains (D6);
   - the expert map, as `configs/expert_map.json` in the format of `configs/expert_map_TEMPLATE.json`;
   - the confirmatory links (D7), ε (D12), the motor state (D5) and the floor-bound domains (D26);
   - the measurement-error variances (D28), in `configs/measurement_error.json`, from `docs/measurement_reliability.md`.
5. Record every decision in the protocol (§17) and commit.

## 4. WP2: engine gate and protocol freeze (days of compute)

The full gate runs on synthetic data at the real size, so no approval is needed. Set `--n` and `--d` to the frozen domains and WP0's counts.

```
caffeinate -i nohup $PY scripts/run_engine_gate.py --n 900 --d 10 --datasets 20 > results/engine_gate/gate.log 2>&1 &
```

**Output.** `results/engine_gate/gate_n900_d10.json`.

**It passes (§16) if:**
- the automatic checks pass: reference agreement, gradients, quadrature, dispersed starts, conditioning with any spreads at zero set aside, profiles, and no negative likelihood ratio;
- the investigator and statistician accept the recovery bias and coverage.

**Then:**
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
- **Checkpointing.** The bootstrap saves `results/wp4/hp_tdpigd_checkpoint.json` after every replicate. If it stops, run the same command again: it resumes when the observed statistic is unchanged.
- **Output.** `results/wp4/hp_tdpigd.json`, aggregates only.
- **Duration.** At 10 domains, the replicates run one at a time, at roughly half an hour each.
  - **No clear rejection: a day or two.** H-P stops early once 20 replicates are at least as large as the observed statistic.
  - **A rejection: 3–4 weeks.** It runs all 999 replicates.

Step 1's reported map, the RBD axis and the robustness and sensitivity analyses (§7.10–§7.14) are run with the same modules, as set out in the analysis plan registered at WP2.

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

# phenotype-discovery

**Do Parkinson's phenotypes progress through different mechanisms?** A causal-discovery study of year-to-year progression in PPMI. It compares clinical phenotypes: tremor-dominant versus postural instability and gait difficulty, and the clinical subtypes. This is Paper 2, experiment 1, redesigned from scratch on 2 November 2026.

## Status (3 November 2026)

Design stage.
- `PROTOCOL.md` is protocol draft v0.5 (3 November 2026). It narrows the study to one primary test and adds speed estimands:
  - H-P: do TD and PIGD share one progression mechanism up to a difference in speed? The only primary test, at 5%;
  - speed estimands as key secondary outcomes: the speed ratio, the spread of speed between patients and its predictors, and the share of the difference that speed explains;
  - one continuous-time engine with a speed and a load for each patient and measurement error in the scores;
  - confirmatory links gated on H-P under Holm; RBD as a secondary axis; data-driven classes and variation between patients as exploratory extensions;
  - an engine gate before registration, and a method-qualification gate (H-M1) in two families of simulators, with twelve named comparators as a descriptive benchmark.

  Part I is the phenotype study (Paper 2, experiment 1); Part II, clinical translation, is a conditional outline. Appendix D lists the changes from v0.3, v0.4 and v0.4.1, and Appendix E the synthetic evidence. Earlier drafts are in `docs/archive/`.
- `docs/protocol_review.md` reviews the Option B/C draft v0.2 and its critical review, and explains the v0.3 changes.
- `docs/pathway_heterogeneity_framework.md` is an investigation report (2 November 2026) on whether every patient progresses through the same pathway. It shows that comparing yearly effects confuses a faster progression with a different mechanism. It recommends speed-adjusted subgroup causal discovery for phenotypes, data-driven classes and individuals as the core of protocol v0.4, which adopts it. Its three synthetic simulations are in `scripts/toy/`; they use no PPMI data.
- `docs/methods_and_novelty_update.md` is an addendum to that report (2 November 2026). It reviews causal discovery methods newer than PCMCI, J-PCMCI+ and JCI for this study, and repeats the novelty check with verified references. The novelty claims hold only in a narrower form, which the report now uses.
- `docs/step2_comparison_methods.md` (3 November 2026) answers which Step 2 comparison methods to apply. It reviews four families in full text: difference graphs, shift detectors, shared-plus-deviation estimators, and the group-invariance tests of clinical panel research. A fourth synthetic check (toy 4) runs each family next to its speed-adjusted counterpart. The note recommends a speed-adjusted shared-plus-deviation model in continuous time, with jointly fitted per-link and per-domain tests, and names the comparators for the methods test (H-M1).
- `docs/tsldiffpc_tsdci_investigation.md` (3 November 2026) is a second, deeper round on tsLDiffPC and tsDCI (Ruer 2026). It covers:
  - the framework and the authors' code;
  - a re-implementation that reproduces the paper's published results;
  - the method's behaviour on our design: it reads speed as changed persistence, and has low power;
  - how it is used: as a named comparator, plus its root-cause vocabulary.
- `docs/clinical_expert_questionnaire.md` is round 1 (v0.2) of the two-round questionnaire for the clinical experts (WP1). The PPMI group sizes from WP0 are filled in; add the return date and contact before sending.
- `src/pdx/` is the analysis code for protocol v0.5 (3 November 2026). It contains:
  - the continuous-time engine, H-P with its conditional bootstrap, the speed estimates and localisation;
  - Step 1, the separate-graph reference and four comparators;
  - the simulators, the qualification harness, the engine gate and the WP0 audit.

  Two independent reviews were run, their findings fixed, and the unit tests pass (66 by default, plus 4 slow ones run with `PDX_SLOW=1`). `docs/code_map.md` maps each protocol section to its code and its checks, and lists what is not yet in code.
- **Compute is the main open issue.** At ten domains and about 1,450 synthetic participants, one likelihood evaluation takes about 2.5 s on the M4.
  - The speed-only fit needed 223 evaluations (about 9 minutes when run alone).
  - The free fit needed 375 evaluations from the null start (about 16 minutes), and the free model's second start comes on top.
  - So a bootstrap replicate costs roughly half an hour, against the 1 minute per fit assumed in §8.8.
  - In the WP0 power pilot (816 participants, 4 jobs at once), each fit pair took 2–5.4 hours (median 3.4), so this Mac completed about one fit pair an hour.
  - A timing test on 4 November found that the engine is limited by memory traffic. Smaller likelihood batches (32 participants in a single-threaded job, 128 otherwise; `scripts/bench_threads.py`) make it 1.7–2 times faster, with identical values. WP3's replicate numbers for family A would now take about 10–12 weeks here.
  - The compute plan (D14) has to be revised once the engine gate has timed the real size.
- **The engine gate (WP2) started on 4 November 2026.**
  - The Mac finished the reference, gradient and quadrature checks.
  - At WP0's size, 12 quadrature nodes miss the gate's rule (`docs/RUNBOOK.md`, §4). The engine now uses 20 (D32), and the load no longer moves with speed (D31), so the gate starts afresh.
  - The rest of the gate moves to a Windows PC; `docs/run_on_windows.md` covers setup and running there.
- **An exploratory first look at H-P on PPMI (4 November 2026)** ran on the investigator's instruction, before WP1–WP3. It used the provisional map in `configs/expert_map_provisional.json` and the observed fits only (`scripts/run_hp_real.py --first-look`, `docs/RUNBOOK.md` §6). Its result is exploratory.
- **The full exploratory run on the investigator's PC (4–5 November 2026) gave no usable result.**
  - **What went wrong.** Every bootstrap fit stopped at its start, so its p = 0.005 was empty, and neither observed fit converged.
  - **The revision of 5 November** fixed the engine, the fits and the data pipeline:
    - batching, failed fits, a guard on the bootstrap, alternating starts and warm starts;
    - the dose in force, the treatment state, the levodopa exposure and the error variances;
    - reproducible checkpoints;
    - D31 and D32.
  - **Where it is recorded.** Protocol Appendix D, and `docs/code_guide/08_audit_findings.md` under "What the PC run showed".
  - **Next.** Rebuild the panel, time one batch of replicates (`--time-replicate`), then run.
- **WP0, the feasibility audit, ran on 3 November 2026**, with the investigator's approval, on the PPMI download of 13 October 2026. `docs/wp0_feasibility_report.md` has the results, aggregate counts only:
  - 1,127 untreated sporadic participants: 804 TD, 209 PIGD, 114 indeterminate;
  - 986 seen at the year-1 landmark (177 PIGD), with 3,727 transitions;
  - 93% of intervals within 12 ± 3 months;
  - findings that inform D5, D12, D16 and D26;
  - the power check (the WP0 gate), which passed: H-P's power against one deviation of 2ε is above 99% at WP0's numbers (synthetic pilot).
- **WP1 materials.** The round-1 questionnaire is v0.2: background numbers filled in, and Section E shortened to 31 candidate links with confidence ratings. `docs/measurement_reliability.md` collects the published reliabilities for D28.
- `docs/RUNBOOK.md` gives step-by-step instructions for running every work package.

Word versions for sharing are in `docs/`, built from the Markdown:

```
/usr/bin/python3 scripts/md_to_docx.py PROTOCOL.md docs/Phenotype_Causal_Discovery_Protocol_v0.5.docx "Phenotype causal discovery in PD · protocol draft v0.5 · 3 November 2026"
/usr/bin/python3 scripts/md_to_docx.py docs/protocol_review.md docs/Protocol_Review_Option_BC_v0.2.docx "Review of the Option B/C protocol draft v0.2 · 2 November 2026"
/usr/bin/python3 scripts/md_to_docx.py docs/pathway_heterogeneity_framework.md docs/Pathway_Heterogeneity_Framework_Report.docx "Does every patient progress through the same pathway? · investigation report · 2 November 2026"
/usr/bin/python3 scripts/md_to_docx.py docs/methods_and_novelty_update.md docs/Methods_and_Novelty_Update.docx "Newer causal discovery methods and a second novelty check · addendum · 2 November 2026"
/usr/bin/python3 scripts/md_to_docx.py docs/tsldiffpc_tsdci_investigation.md docs/tsLDiffPC_tsDCI_Investigation.docx "tsLDiffPC and tsDCI: framework, behaviour on our design, and use · investigation note · 3 November 2026"
/usr/bin/python3 scripts/md_to_docx.py docs/step2_comparison_methods.md docs/Step2_Comparison_Methods.docx "Step 2 methods: what to apply when comparing phenotypes' causal mechanisms · investigation note · 3 November 2026"
/usr/bin/python3 scripts/md_to_docx.py docs/clinical_expert_questionnaire.md docs/Clinical_Expert_Questionnaire_Round1_v0.2.docx "Phenotype causal discovery in PD · expert questionnaire, round 1 · v0.2"
/usr/bin/python3 scripts/md_to_docx.py docs/wp0_feasibility_report.md docs/WP0_Feasibility_Report.docx "Phenotype causal discovery in PD · WP0 feasibility audit · 3 November 2026"
/usr/bin/python3 scripts/md_to_docx.py docs/RUNBOOK.md docs/Runbook.docx "Phenotype causal discovery in PD · runbook · protocol v0.5"
```

Earlier versions (protocol drafts v0.2, v0.3, v0.4 and v0.4.1, questionnaire v0.1) are in `docs/archive/`.

## Running the code

The code uses the ppmi-causal Python environment (Python 3.13 with numpy, scipy, torch, pandas, causal-learn and statsmodels); nothing new is installed.

```
PY="../ppmi-causal/.venv/bin/python"
$PY tests/run_tests.py                 # unit tests, about 3 minutes
$PY scripts/smoke_test.py --n 300      # the whole chain on one small synthetic dataset
$PY scripts/run_engine_gate.py --quick # the engine-gate harness at small size
```

Long runs (the real-size engine gate, the qualification) are started from the Terminal app, with at most 4 parallel jobs. On a Windows PC, set up and run as in `docs/run_on_windows.md`.

## Lineage

This project replaces two earlier designs:
- the escalation-discovery experiment (`../escalation-discovery`), approved on 1 November 2026 and stopped on 2 November 2026;
- the Option B/C proposal draft v0.1 (`Parkinsons_Option_B_C_Research_Protocol.docx`). Two v0.2 drafts were written from it: this project's phenotype-discovery protocol, and a review-integrated Option B/C protocol with its critical review (in the investigator's Downloads). Protocol v0.3 merges both.

Nothing is reused. The stopped run serves as pilot evidence (PROTOCOL.md §1.5 and Appendix B).

## Layout

| Path | Contents |
|---|---|
| `PROTOCOL.md` | The protocol (design draft) |
| `docs/` | The protocol review; the investigation report on pathway heterogeneity, with its methods and novelty addendum, the Step 2 methods note and the tsLDiffPC/tsDCI note; the expert questionnaire; Word versions of the documents |
| `docs/archive/` | Superseded versions (protocol drafts v0.2, v0.3, v0.4 and v0.4.1) |
| `src/pdx/` | Analysis code for protocol v0.5 (see `docs/code_map.md`) |
| `tests/` | Unit tests, run with `tests/run_tests.py` (pytest is not installed) |
| `configs/protocol_v0.5.toml` | The pre-specified constants |
| `configs/expert_map_provisional.json` | A provisional map for the exploratory first look (4 November 2026): the earlier v4.1 expert map translated to the ten domains and revised against the clinical guidelines; WP1's frozen map replaces it |
| `configs/wp0_sim_settings.json` | WP0's group sizes, visits and missingness for the synthetic runs (aggregates of the WP0 audit) |
| `docs/run_on_windows.md` | Setting up and running the engine gate, and the PPMI analysis, on a Windows PC |
| `scripts/smoke_test.py`, `run_engine_gate.py`, `run_qualification.py` | Synthetic runs: the whole chain, the engine gate, the WP3 qualification |
| `scripts/build_ppmi_tables.py`, `run_wp0_audit.py`, `build_analysis_panel.py`, `run_hp_real.py` | Real-data runs (WP0, WP4): they refuse to start without the investigator's written approval (`--approval`), and keep participant-level tables in `data/` |
| `scripts/run_power_check.py` | WP0's power gate, on synthetic data at WP0's counts |
| `scripts/wp1_consensus.py` | WP1: compiles the panel's returned questionnaires into the round-2 table and the draft expert map (§6.4). Run with the system Python, which reads Word files; returned files stay in `data/wp1/` |
| `docs/wp1_invitation_email.md` | Draft invitation to the expert panel, with the checks to make before sending |
| `docs/wp3_simulator_fitting_plan.md` | How the simulators are to be fitted to PPMI's pooled population (§8.2), keeping the confirmatory analyses blind |
| `scripts/bench_threads.py` | Timing test of batch size and job layout, for the compute plan (D14) |
| `docs/RUNBOOK.md` | How to run every work package |
| `scripts/md_to_docx.py` | Builds the Word versions |
| `scripts/toy/` | Synthetic simulations behind the investigation report and the Step 2 note: speed versus pathway, speed varying within phenotypes (with the person-level test at larger cohort sizes), latent classes, named comparator families against the speed-adjusted Step 2 procedures (`toy_comparators*.py`), and tsLDiffPC/tsDCI re-implemented (`ts_ldiffpc.py`), validated against the paper (`toy_tsldiffpc_validation.py`) and run on our design (`toy_tsldiffpc.py`), and DYNOTEARS re-implemented (`dynotears.py`), validated against its paper (`toy_dynotears_validation.py`) and run on our design (`toy_dynotears.py`). Run from that folder with the ppmi-causal Python environment; no PPMI data |
| `data/` | Later: participant-level tables (git-ignored; never shared) |

## Data governance

PPMI data are used under the PPMI Data Use Agreement. Participant-level data never leave `data/`; documents and results hold aggregates only; nothing participant-level goes to any external service.

Data used in the preparation of this work were obtained from the Parkinson's Progression Markers Initiative (PPMI) database (www.ppmi-info.org/access-data-specimens/download-data), RRID:SCR_006431. For up-to-date information on the study, visit www.ppmi-info.org. PPMI is funded by the Michael J. Fox Foundation for Parkinson's Research and funding partners.

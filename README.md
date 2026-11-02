# escalation-discovery

**Paper 2, experiment 1: causal discovery for the levodopa-escalation decision in Parkinson's disease (PPMI).**
Which causal-discovery learners, classical or deep, can be trusted to recover the structure behind escalation
decisions from PPMI-like data (Arm A, simulations with the expert DAG as truth), and does PPMI agree with the expert
DAG (Arm B: falsification, stability, negative controls, blinded clinician review)? Experiment 2, causal inference,
is `../escalation-study-v2`; this repository estimates no effect.

`PROTOCOL.md` is the pre-registration draft (cohort, variables, methods panel, Arm A and B, decision rule, compute).
`docs/inventory.md` maps the design onto earlier work; `docs/design_claims_check.md` checks the pasted design's
citations.

## Status (1 November 2026)

- **Protocol approved on 1 November 2026** (`PROTOCOL.md` §12; the approved text is kept unchanged in
  `docs/PROTOCOL_approved_2026-11-01.md`), so `ppmi_runs_approved = true`. The cohort is built (1,741 participants,
  8,829 grid rows, as in §3), and `scripts/run_protocol.sh` runs the rest of §10 in the background: one line per
  finished step in `results/run_status.tsv`, console output in `data/derived/logs/`.
- The harness runs end to end on synthetic data from the expert DAG (`generator.synthetic`): 44 tests pass
  (`uv run pytest -q`), including both arms end to end and the oracle PAG against brute-force FCI.
- The six deep-learning workers (`envs/`, 3.9 GB) pass a check on a known 10-node graph (PROTOCOL.md §5.2).
- Compute was measured at the pooled base size (PROTOCOL.md §10.1): uncapped FCI does not finish, Tetrad's tiered
  FCI-type learners exhaust memory, DECI exceeds an hour per fit. Scope chosen on 29 October 2026 (Plan B, §10.2,
  §12): FCI capped at 3; Tetrad's tiered BOSS only; DECI, PC with the clustered GCM and J-PCMCI+ dropped; the
  deep-learning learners in Arm A's realistic settings only; 50 Arm B subsamples; the Arm B sensitivity runs
  restricted to the expert-DAG test and the selected learner, without the context nodes a subset fixes (§3). About
  3.5 days on this Mac (`uv run escd methods` lists the panel).
- Decisions of 29 October 2026: expert DAG v5.1; pooled PD cohorts with cohort context nodes; this repository, with
  v2's Q1 audit and core-map falsification moved here (v2 untouched); OpenJDK 21 + Tetrad 7.6.10 and Python
  deep-learning side environments installed, no R.

## Run

```
uv sync
uv run escd methods                          # which learners can run here
uv run escd arm-a --tune                     # tuning on synthetic data
uv run escd arm-a                            # Arm A (synthetic source)
uv run escd tau                              # stability threshold for Arm B
uv run escd arm-b --data synthetic           # the Arm B pipeline on one simulated cohort
uv run escd handover --data synthetic
uv run pytest -q                             # add -m "not slow" for the fast subset
```
The PPMI run order is PROTOCOL.md §10; after `escd build && escd trials`, `scripts/run_protocol.sh` runs the rest.

## Layout

| Path | Contents |
|---|---|
| `src/escalation_discovery/` | `expert` (v5.1 layouts, tiers, generating DAG) · `truth` (oracle MAG/PAG) · `generator`, `missingness`, `fit` (Arm A data) · `metrics` · `methods/` (adapters: classical, Tetrad, continuous, temporal, external workers, ensemble, registry) · `knowledge`, `cgcm` (tiered FCI semantics, the clustered test for causal-learn) · `arm_a`, `arm_b`, `falsify`, `handover` · `cli` |
| ported from v2 (provenance line at the top) | `config`, `build`, `trials`, `citest`, `nuisance`, `coremap`, `audit` (Q1) |
| `envs/`, `workers/` | side environments and worker scripts of the learners that pin other Python versions |
| `docs/` | `expert_dags.yaml` (from ppmi-causal), `core_map.yaml` (from v2), inventory, claim check |
| `results/` | aggregates only |
| `data/derived/` | participant-level tables, fitted generators, pattern banks, worker files (gitignored; never shared) |
| `tools/` | `tetrad-gui-7.6.10-launch.jar` (Maven Central; SHA-1 d7d9dc08ad2209a8d5bca6e94ab58323b1f48007; gitignored) |

## Data governance

PPMI data are used under the PPMI data-use agreement. Participant-level data never leave `data/derived/`;
`results/` and `docs/` hold aggregates only; no PPMI content goes to an external service (the pretrained models run
locally). Data used in the preparation of this work were obtained from the Parkinson's Progression Markers Initiative
(PPMI) database (www.ppmi-info.org/access-data-specimens/download-data), RRID:SCR_006431. For up-to-date information
on the study, visit www.ppmi-info.org. PPMI is funded by the Michael J. Fox Foundation for Parkinson's Research and
funding partners.

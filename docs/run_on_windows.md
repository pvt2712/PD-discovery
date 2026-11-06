# Running the analyses on a Windows PC

> Protocol draft v0.5. Written 4 November 2026. Sections 1–6 cover the engine gate (WP2), which uses synthetic data only; the same setup runs the other synthetic scripts later (the power check, the WP3 qualification). Section 7 adds the PPMI analysis. Its data can be rebuilt on the PC from the raw PPMI download, or the two-file analysis panel can be copied from the Mac.

## 1. Set up once (about 15 minutes)

1. **Install Python 3.13 and Git.** In PowerShell:
   ```powershell
   winget install Python.Python.3.13
   winget install Git.Git
   ```
   Alternatively, use the installers from python.org (tick "Add python.exe to PATH") and git-scm.com. Close and reopen PowerShell afterwards.

2. **Get the code.** The branch `phenotype-discovery` of the GitHub repository holds this project. Put it in `C:\work`, outside any folder that OneDrive backs up, so that PPMI files placed in it later never sync to the cloud. On many PCs OneDrive moves Documents, and `$HOME\Documents` is then not where it seems:
   ```powershell
   New-Item -ItemType Directory -Force C:\work
   cd C:\work
   git clone -b phenotype-discovery https://github.com/pvt2712/PD-discovery.git phenotype-discovery
   cd phenotype-discovery
   ```

3. **Make the Python environment.** This installs the pinned versions from `pyproject.toml` (about 1 GB, mostly PyTorch):
   ```powershell
   py -3.13 -m venv .venv
   .\.venv\Scripts\python.exe -m pip install --upgrade pip
   .\.venv\Scripts\python.exe -m pip install -e ".[test]"
   ```
   Every command calls `.venv\Scripts\python.exe` directly, so the environment never needs activating, and PowerShell's script policy doesn't matter.

4. **Check the installation.** All tests must pass; this takes about 3–5 minutes:
   ```powershell
   $env:PYTHONUTF8 = "1"
$env:PYTHONHASHSEED = "0"
   .\.venv\Scripts\python.exe tests\run_tests.py
   ```
   `PYTHONUTF8=1` makes Python read and write text as UTF-8, as on the Mac. `PYTHONHASHSEED=0` fixes Python's hash seed, so Step 1's causal-discovery algorithm gives the same answer on every run (code guide §8, item A8). Both last until the window is closed, so set them in every new PowerShell window before running anything.

## 2. Before the long run

- **Sleep.** In Settings → System → Power, set "When plugged in, put my device to sleep after" to Never. The screen may still turn off.
- **Updates.** In Settings → Windows Update, choose "Pause updates" for a week. An automatic restart would interrupt the run. The run would resume, but would lose the piece it was working on.
- **Power.** Keep the PC plugged in, and avoid other heavy programs while the gate runs.

## 3. Run the gate (about a day with 5 jobs)

```powershell
cd C:\work\phenotype-discovery
$env:PYTHONUTF8 = "1"
$env:PYTHONHASHSEED = "0"
.\.venv\Scripts\python.exe scripts\run_engine_gate.py --wp0 --d 10 --datasets 20 --jobs 5
```

`--jobs 5` runs five independent pieces at once, each in its own single-threaded process: the dispersed starts, the profiles and the 40 recovery datasets. This uses the processor far better than one process with many threads, because the engine is limited by memory speed. Every piece has its own seed, so the results do not depend on the number of jobs. Five suits a 6-core processor such as the Ryzen 5 7600 and needs a few GB of the 32 GB of memory; section 4 shows how to check the best number.

- **Leave the window open.** It prints a line each time a piece finishes. Some pieces take hours, so long silences are normal. The same lines are appended to `results\engine_gate\gate_wp0_d10.log`. To follow that log from a second PowerShell window:
  ```powershell
  Get-Content C:\work\phenotype-discovery\results\engine_gate\gate_wp0_d10.log -Wait -Tail 20
  ```
- **If it stops,** run the same three lines again. That covers a closed window, a restart, Ctrl+C or an error. Finished sections, dispersed-start fits, recovery datasets and profiles are kept in `results\engine_gate\gate_wp0_d10.partial.json` and skipped.
- **After the revision of 5 November 2026 the gate starts afresh.** The model changed: the load no longer moves with speed (D31), there are 20 quadrature nodes (D32), and the fits follow new rules. The gate refuses a checkpoint made under other settings, so first move `results\engine_gate\gate_wp0_d10.partial.json` out of the folder.
- **The time** depends on the PC's processor and memory speed. The reference checks, the gate dataset's fit, quadrature, identification and the final timing run one at a time; the dispersed starts, profiles and recovery datasets run in parallel. With 5 jobs, expect roughly a day, against the Mac's 1.5–2 days in one process.
- **When it finishes,** the last log line says whether all automatic checks pass. The report is `results\engine_gate\gate_wp0_d10.json`. Copy it and the log to the Mac's `results/engine_gate/` folder, by USB stick or your own cloud drive; both are synthetic. Then ask Claude to read them, or follow `docs/RUNBOOK.md` §4.

## 4. Optional: choose the number of jobs first (10 minutes)

The best number of jobs depends on the processor and the memory. To compare, run the same likelihood work as 4, 5 and 6 single-threaded jobs, and as one process with 6 threads:

```powershell
.\.venv\Scripts\python.exe scripts\bench_threads.py --configs 4x1 5x1 6x1 1x6 --evals 20
```

Each line ends with `evals_per_hour`, the work done per hour by all jobs together. Use the best of `4x1`, `5x1` and `6x1` as `--jobs` for the gate and for H-P (section 7c).

## 5. Updating the code later

```powershell
cd C:\work\phenotype-discovery
git pull
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
```

Don't update the code while a gate run is going.

## 6. When something goes wrong

- **"'.\.venv\Scripts\python.exe' is not recognized".** PowerShell is not in the project folder, or the environment was never made. Run `cd C:\work\phenotype-discovery`, then step 3 of section 1.
- **"'git' or 'py' is not recognized".** Install it (section 1, step 1), then close and reopen PowerShell.
- **pip can't find torch.** Check that Python is 3.13, 64-bit: `py -3.13 -c "import sys; print(sys.version)"` should show "64 bit".
- **"holds a gate run with other settings".** The partial file belongs to a different command. Move it out of `results\engine_gate\` to start afresh.
- **Anything else.** The log records the error. Copy its last lines to Claude.

## 7. The PPMI analysis: H-P on the real data

> **Participant-level data.** PPMI's data use agreement is with you. Check that it allows a second computer of yours, and keep the data secure, as below. Nothing in this section sends data anywhere. Never paste rows of the data into an AI tool.

There are two ways to get the data onto the PC:
- **7a** rebuilds everything from the raw PPMI download, repeating the whole pipeline independently of the Mac.
- **7b** copies the finished analysis panel from the Mac, which is quicker.

Either way, H-P then runs as in **7c**.

**Keep the data safe, either way.**
- **Moving files.** Move PPMI files on an encrypted USB stick or another route you control. Don't use email, GitHub or a cloud drive.
- **Git.** `data\` is git-ignored, so git never uploads the tables the build writes there. Keep the raw download outside the repository folder, and never add data files with `git add -f`.
- **The PC.** Keep the Windows account password-protected and the disk encrypted (Settings → Privacy & security → Device encryption, or BitLocker).
- **Cleanup.** Delete the PPMI files from the PC when the analysis is finished.

### 7a. Build the data from the raw PPMI download

**What the build reads.** It reads 15 tables in 5 folders of the download, about 185 MB of its 902 MB:
- `_Subject_Characteristics`
- `Medical_History`
- `Study_Enrollment`
- `Motor___MDS-UPDRS`
- `Non-motor_Assessments`

The other folders are not used: imaging, `Data___Databases`, `PPMI_Online` and the other cohorts' follow-up. Copy the whole `raw` folder, or only these five, keeping their names. `--raw` points at the folder that holds them.

**Run these in order,** with your own path in place of `D:\PPMI\raw`:

```powershell
cd C:\work\phenotype-discovery
git pull
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
$env:PYTHONUTF8 = "1"
$env:PYTHONHASHSEED = "0"
$AP = "Investigator, 6 November 2026: Paper 2 protocol approved; H-P on the development participants (D33)"
.\.venv\Scripts\python.exe scripts\build_ppmi_tables.py --raw "D:\PPMI\raw" --approval $AP
.\.venv\Scripts\python.exe scripts\run_wp0_audit.py --visits data\derived\wp0_visits.parquet --approval $AP
.\.venv\Scripts\python.exe scripts\build_analysis_panel.py --set development --approval $AP
```

- **`pip install`** adds `pyarrow`, which writes the build's tables. It is needed once.
- **The table build** takes a few minutes. It writes the participant-level tables to `data\derived\`. In `results\wp0\` it writes the participant flow and a manifest of every file's date stamp and SHA-256.
- **The audit** repeats the WP0 feasibility tables in `results\wp0\wp0_audit.json`. H-P doesn't need it, but it allows a comparison with the WP0 report.
- **The panel build** keeps only Paper 2's development participants (training and validation; decision D33), so the held-out participants never enter the panel, its normal-scores tables or its error variances. It writes `data\derived\panel_tdpigd_development.npz` and `panel_spec_development.json`, the two files H-P reads. Its first line names the participant set and how many of the tables' participants it keeps.

**Check against the Mac.** The Mac used the download of 13 October 2026, whose file names end in `13Oct2026`. From that download, after the revision of 5 November:
- the table build prints `PD cohort 2262; primary population 1127; grid rows 6254`;
- the whole-cohort panel held 706 TD participants with 2,919 transitions, and 177 PIGD participants with 675, with error variances from 0.139 (daily function) to 0.208 (cognition). The development panel keeps about four fifths of them, so its counts are smaller and its error variances move a little. Send the printed lines, and Claude compares them with the Mac's development build.

A newer PPMI download holds more follow-up, so it gives different numbers. In that case, stop and ask Claude before going on: WP0's figures, and the settings of the synthetic runs, come from the October download.

### 7b. Or copy the analysis panel from the Mac

H-P needs only two files from the Mac's `phenotype-discovery/data/derived/` folder:
- `panel_tdpigd_development.npz` (about 60 KB) holds the development participants' domain scores by visit, with visit times, treatment, four baseline covariates and the participant numbers.
- `panel_spec_development.json` (about 3 KB) holds the domains, the treated domains, the measurement-error variances, how they were derived, and the participant set.

Copy the panel built after the revision of 5 November; an older one has the medication errors the revision fixed.

```powershell
cd C:\work\phenotype-discovery
git pull
New-Item -ItemType Directory -Force data\derived
```

Then put both files in `data\derived\`.

### 7c. Run H-P

**After the revision of 5 November 2026.** The run of 4–5 November gave no usable result: every bootstrap fit had stopped at its start, so its p = 0.005 meant nothing (`docs/code_guide/08_audit_findings.md`, "What the PC run showed"). The revision changed the engine, the fits and the data pipeline. So after `git pull`:
- rebuild the data (7a) or copy the rebuilt panel from the Mac (7b);
- then time one batch of replicates, and only then start the full run, both below.

**Development participants only (D33, 6 November 2026).** H-P is now supporting work for Paper 2, and it runs on Paper 2's development participants. The run names its outputs `hp_tdpigd_development.*`, and it refuses a panel built for another set. The old `hp_tdpigd_checkpoint.json` belongs to the whole-cohort run and is never reused; it holds only the zeros, so it can be deleted.

**1. Time one batch first.** This runs the two observed fits, then one batch of bootstrap replicates (six with `--jobs 6`), writes `results\wp4\hp_tdpigd_development_timing.json` and stops:

```powershell
cd C:\work\phenotype-discovery
$env:PYTHONUTF8 = "1"
$env:PYTHONHASHSEED = "0"
.\.venv\Scripts\python.exe scripts\run_hp_real.py --set development --expert configs\expert_map_provisional.json --n-max 199 --speed-bootstrap 100 --jobs 6 --time-replicate --approval "Investigator, 6 November 2026: Paper 2 protocol approved; H-P on the development participants (D33)"
```

- **What it does.**
  - The observed fits now try more starts. The speed-only fit is restarted from the free fit until neither improves.
  - Each start stops once it gains less than 0.01 in log-likelihood over 100 iterations.
  - Expect several hours before the first replicate line.
- **Checks on itself.** The script refuses to run unless `PYTHONHASHSEED` is 0. The bootstrap stops with the reason if its first five replicates all fail, all give 0, or all have fits that never left their start. This is the failure that went unseen on 4–5 November.
- **What to send.** Copy `hp_tdpigd_development_timing.json` and `hp_tdpigd_development.log` to the Mac and ask Claude to read them. The log's last line gives the minutes per replicate and the most the full run can take. Choose `--n-max` from it.

**2. The full run** is the same command without `--time-replicate`:

```powershell
.\.venv\Scripts\python.exe scripts\run_hp_real.py --set development --expert configs\expert_map_provisional.json --n-max 199 --speed-bootstrap 100 --jobs 6 --approval "Investigator, 6 November 2026: Paper 2 protocol approved; H-P on the development participants (D33)"
```

It continues from the checkpoint. The observed fits and the timed replicates are reused, not recomputed. `--jobs 6` runs six bootstrap replicates at once, then six speed samples, each in its own single-threaded process; section 4's measurement on this PC chose 6. Every replicate has its own seed and the stopping rule is applied in replicate order, so the result does not depend on the number of jobs.

- **Time.** The timing run tells us; nothing earlier is reliable.
  - The bootstrap stops once 20 replicates reach the observed statistic. If H-P is far from rejecting, that takes about 40 replicates. If it rejects, it runs to the cap of 199.
  - The speed bootstrap then adds 100 samples.
- **If it stops,** run the same command again.
  - **What is kept.** The checkpoint `results\wp4\hp_tdpigd_development_checkpoint.json` holds the run's settings, the observed fits with their parameters, and every replicate and speed sample. They are all reused while the inputs, map, seed and protocol constants are unchanged. `--jobs` and `--threads` may change.
  - **What is refused.** A lower `--n-max` than the checkpoint's would give a wrong p-value, so the script refuses it.
- **Progress** is printed and appended to `results\wp4\hp_tdpigd_development.log`. Each replicate line gives its statistic, its fits' evaluations and its minutes.
- **One run at a time.** Run H-P and the engine gate one after the other: each already uses the processor fully with `--jobs`.
- **Outputs** are aggregates only: `results\wp4\hp_tdpigd_development.json`, the timing file and the log. The report also records:
  - the settings;
  - the code versions used;
  - each fit's starts and how they stopped;
  - the fitted parameters.

  Copy them back to the Mac's `results/wp4/` folder.

**The first look** (observed fits only, with a χ² p-value) is optional:

```powershell
.\.venv\Scripts\python.exe scripts\run_hp_real.py --set development --expert configs\expert_map_provisional.json --first-look --approval "Investigator, 6 November 2026: Paper 2 protocol approved; H-P on the development participants (D33)"
```

- **Reading the result.** It is exploratory: the map is provisional, and WP1–WP3 come later. The report says so in its first line.

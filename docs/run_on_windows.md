# Running the analyses on a Windows PC

> Protocol draft v0.5. Written 4 November 2026. Sections 1–6 cover the engine gate (WP2), which uses synthetic data only; the same setup runs the other synthetic scripts later (the power check, the WP3 qualification). Section 7 adds the PPMI analysis. It needs two participant-level files copied from the Mac, and nothing else from `data/`.

## 1. Set up once (about 15 minutes)

1. **Install Python 3.13 and Git.** In PowerShell:
   ```powershell
   winget install Python.Python.3.13
   winget install Git.Git
   ```
   Alternatively, use the installers from python.org (tick "Add python.exe to PATH") and git-scm.com. Close and reopen PowerShell afterwards.

2. **Get the code.** The branch `phenotype-discovery` of the GitHub repository holds this project:
   ```powershell
   cd $HOME\Documents
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
   .\.venv\Scripts\python.exe tests\run_tests.py
   ```
   `PYTHONUTF8=1` makes Python read and write text as UTF-8, as on the Mac. It lasts until the window is closed, so set it in every new PowerShell window before running anything.

## 2. Before the long run

- **Sleep.** In Settings → System → Power, set "When plugged in, put my device to sleep after" to Never. The screen may still turn off.
- **Updates.** In Settings → Windows Update, choose "Pause updates" for a week. An automatic restart would interrupt the run. The run would resume, but would lose the piece it was working on.
- **Power.** Keep the PC plugged in, and avoid other heavy programs while the gate runs.

## 3. Run the gate (about 1.5–2 days)

```powershell
cd $HOME\Documents\phenotype-discovery
$env:PYTHONUTF8 = "1"
.\.venv\Scripts\python.exe scripts\run_engine_gate.py --wp0 --d 10 --datasets 20
```

- **Leave the window open.** It prints a line each time a piece finishes. Some pieces take hours, so long silences are normal. The same lines are appended to `results\engine_gate\gate_wp0_d10.log`. To follow that log from a second PowerShell window:
  ```powershell
  Get-Content $HOME\Documents\phenotype-discovery\results\engine_gate\gate_wp0_d10.log -Wait -Tail 20
  ```
- **If it stops,** run the same three lines again. That covers a closed window, a restart, Ctrl+C or an error. Finished sections, dispersed-start fits, recovery datasets and profiles are kept in `results\engine_gate\gate_wp0_d10.partial.json` and skipped.
- **The time** depends on the PC's processor and memory speed. The gate is one process, and the engine is limited by memory speed more than by the number of cores. Expect roughly the Mac's 1.5–2 days.
- **When it finishes,** the last log line says whether all automatic checks pass. The report is `results\engine_gate\gate_wp0_d10.json`. Copy it and the log to the Mac's `results/engine_gate/` folder, by USB stick or your own cloud drive; both are synthetic. Then ask Claude to read them, or follow `docs/RUNBOOK.md` §4.

## 4. Optional: choose the number of threads first (5 minutes)

By default PyTorch uses one thread per physical core. Because the engine is limited by memory speed, fewer threads can be as fast. To compare, put this PC's number of physical cores in place of the last setting (Task Manager → Performance → CPU shows "Cores"):

```powershell
.\.venv\Scripts\python.exe scripts\bench_threads.py --configs 1x4 1x8 1x16 --evals 20
```

Each line ends with `evals_per_hour`. If fewer threads come out as fast, add `--threads N` to the gate command, with N the best number.

## 5. Updating the code later

```powershell
cd $HOME\Documents\phenotype-discovery
git pull
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
```

Don't update the code while a gate run is going.

## 6. When something goes wrong

- **pip can't find torch.** Check that Python is 3.13, 64-bit: `py -3.13 -c "import sys; print(sys.version)"` should show "64 bit".
- **"holds a gate run with other settings".** The partial file belongs to a different command. Move it out of `results\engine_gate\` to start afresh.
- **Anything else.** The log records the error. Copy its last lines to Claude.

## 7. The PPMI analysis: H-P on the real data

> **Participant-level data.** PPMI's data use agreement is with you. Check that it allows a second computer of yours, and keep the copy small and secure, as below. Nothing in this section sends data anywhere. Never paste rows of the data into an AI tool.

**What to copy, once.** H-P needs only the analysis panel: two files from the Mac's `phenotype-discovery/data/derived/` folder.
- `panel_tdpigd.npz` (about 70 KB): the TD and PIGD participants' domain scores by visit.
- `panel_spec.json` (about 1 KB): the domains, the treated domains and the measurement-error variances.

In the PC's copy of the repository, create the folder and put both files there:

```powershell
cd $HOME\Documents\phenotype-discovery
New-Item -ItemType Directory -Force data\derived
```

- **How to move them.** Use an encrypted USB stick or another route you control. Don't use email, GitHub or a cloud drive.
- **Git.** `data\` is git-ignored, so git never uploads these files. Never add them with `git add -f`.
- **Security.** Keep the PC's Windows account password-protected and its disk encrypted (Settings → Privacy & security → Device encryption, or BitLocker).
- **Cleanup.** Delete the two files when the analysis is finished.

**First, update the code** (the provisional map and the PPMI options are newer than the first clone):

```powershell
git pull
```

**The first look** takes about an hour. The Mac ran it on 4 November, so this is only needed for a repeat:

```powershell
$env:PYTHONUTF8 = "1"
.\.venv\Scripts\python.exe scripts\run_hp_real.py --expert configs\expert_map_provisional.json --first-look --approval "Investigator, 4 November 2026: exploratory run before WP1-WP3"
```

**The full H-P run, with its bootstrap,** gives H-P's own p-value, the speed intervals and the localisation:

```powershell
$env:PYTHONUTF8 = "1"
.\.venv\Scripts\python.exe scripts\run_hp_real.py --expert configs\expert_map_provisional.json --n-max 199 --speed-bootstrap 100 --approval "Investigator, 4 November 2026: exploratory run before WP1-WP3"
```

- **Time.**
  - The observed fits take about an hour, then each bootstrap replicate about 30–40 minutes.
  - The bootstrap stops once 20 replicates reach the observed statistic. If H-P is far from rejecting, that takes about 40 replicates, roughly a day.
  - If H-P rejects, it runs to the cap: 199 replicates, about 5 days, which gives p-values down to 0.005. Without `--n-max` the cap is the protocol's 999, about 3 weeks.
  - The speed bootstrap then adds about 2–3 days for 100 samples.
- **If it stops,** run the same command again. Every replicate and speed sample is kept in `results\wp4\hp_tdpigd_checkpoint.json`.
- **Progress** is printed and appended to `results\wp4\hp_tdpigd.log`.
- **One job at a time.** Running it alongside the engine gate slows both, because the engine is limited by memory speed. If the PC has many cores, `--threads` can split them between the two runs.
- **Outputs** are aggregates only: `results\wp4\hp_first_look.json` or `hp_tdpigd.json`, with their logs. Copy them back to the Mac's `results/wp4/` folder.
- **Reading the result.** It is exploratory: the map is provisional, and WP1–WP3 come later. The report says so in its first line.

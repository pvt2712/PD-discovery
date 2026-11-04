# Running the engine gate on a Windows PC

> Protocol draft v0.5, WP2. Written 4 November 2026. The engine gate uses synthetic data only. The PC needs no PPMI files, and nothing from `data/` should be copied to it. The same setup runs the other synthetic scripts later (the power check, the WP3 qualification).

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

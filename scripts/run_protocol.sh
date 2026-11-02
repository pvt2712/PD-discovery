#!/usr/bin/env bash
# The PPMI analyses of PROTOCOL.md §10 that follow `escd build && escd trials` (protocol approved 1 November 2026).
# Start it from the macOS Terminal app and keep that window open until it finishes: a process started from an app
# loses its permission to write in ~/Documents when the app quits, which stopped the first run on 1 November 2026.
#
#     nohup caffeinate -i -s bash scripts/run_protocol.sh >> data/derived/logs/driver.log 2>&1 &
#
# FROM=<step number> resumes after a stop: FROM=05 skips steps 02-04, so the fitted generator that Arm A's finished
# jobs were drawn from is not refitted. ARM_A_JOBS (default 4) sets Arm A's parallel workers; at 8 its deep-learning
# learners ran this 16 GB Mac out of memory on 1 November 2026. Results do not depend on it: every job draws its own
# seed.
#
# One line per step (name, start, end, exit status) goes to results/run_status.tsv; each step's console output goes
# to data/derived/logs/<step>.log, kept with the participant-level files. The Q1 audit, the core-map test and the
# sensitivity runs feed nothing later, so a failure there is logged and the run goes on; any other failure stops it.
# Arm A skips the jobs it has finished, so starting the script again after a stop resumes the work.
set -uo pipefail
cd "$(dirname "$0")/.."
LOGS=data/derived/logs
STATUS=results/run_status.tsv
mkdir -p "$LOGS"
[ -f "$STATUS" ] || printf 'step\tstart\tend\texit\n' > "$STATUS"
FROM=${FROM:-00}               # skip the steps numbered below FROM
ARM_A_JOBS=${ARM_A_JOBS:-4}
echo "start $(date '+%Y-%m-%d %H:%M'), FROM=$FROM, ARM_A_JOBS=$ARM_A_JOBS"

step() {                       # step <name> <command ...>: run, log, record; returns the command's exit status
  local name=$1 start rc
  shift
  if [[ "${name:0:2}" < "$FROM" ]]; then echo "skipped $name"; return 0; fi
  [ -f "$LOGS/$name.log" ] && mv "$LOGS/$name.log" "$LOGS/$name.$(date '+%Y%m%d-%H%M').log"   # keep an earlier attempt's log
  start=$(date '+%Y-%m-%d %H:%M')
  "$@" > "$LOGS/$name.log" 2>&1
  rc=$?
  printf '%s\t%s\t%s\t%s\n' "$name" "$start" "$(date '+%Y-%m-%d %H:%M')" "$rc" >> "$STATUS"
  return $rc
}
stop() { echo "stopped at $1 (see $LOGS/$1.log)"; exit 1; }

step 02_fit_generator uv run escd fit-generator || stop 02_fit_generator
step 03_audit uv run escd audit
step 04_core_map uv run escd core-map
step 05_arm_a_tune uv run escd arm-a --source fitted --tune --jobs "$ARM_A_JOBS" || stop 05_arm_a_tune
step 06_arm_a uv run escd arm-a --source fitted --jobs "$ARM_A_JOBS" || stop 06_arm_a
step 07_tau uv run escd tau --fitted || stop 07_tau
step 08_arm_b uv run escd arm-b --data ppmi --source fitted || stop 08_arm_b
for s in ppmi1 no_ppmi1 no_ppmi2 no_genetic; do
  step "09_arm_b_$s" uv run escd arm-b --data ppmi --source fitted --methods selected --subset "$s"
done
step 09_arm_b_one_per_person uv run escd arm-b --data ppmi --source fitted --methods selected --one-per-person
step 10_handover uv run escd handover --data ppmi || stop 10_handover
step 11_tests uv run pytest -q
echo "all steps finished; see $STATUS"

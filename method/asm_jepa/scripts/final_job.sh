#!/usr/bin/env bash
# Control script for the final ASM-JEPA job (see scripts/final_job.py for the stages).
#   scripts/final_job.sh start [runner args...]   start in the background (setsid nohup); args are remembered
#   scripts/final_job.sh status | summary | log    progress, result tables, live log of the current stage
#   scripts/final_job.sh stop                      stop runner + child (resumable; also pauses the watchdog)
#   scripts/final_job.sh ensure                    (cron) restart the runner if it died and the job is unfinished
#   scripts/final_job.sh reset                     clear failure counters after fixing a failed stage
#   scripts/final_job.sh install-watchdog | remove-watchdog   crontab entry running `ensure` every 10 min
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
PY=/home/haipd/Sign_Representations/.venv-signrep/bin/python
JOB=${JOB_DIR:-runs/final_s0}
ARGS_FILE=$JOB/runner_args
TAG="# asm_jepa final job watchdog"
cmd=${1:-status}
shift || true
mkdir -p "$JOB"

job_status() { "$PY" -c "import json,sys; print(json.load(open(sys.argv[1])).get('status',''))" "$JOB/job_state.json" 2>/dev/null || true; }
runner_alive() { ! flock -n "$JOB/runner.lock" true 2>/dev/null; }

start() {
  if runner_alive; then echo "runner already running"; return 0; fi
  [ -f "$ARGS_FILE" ] || : > "$ARGS_FILE"
  mapfile -t A < "$ARGS_FILE"
  rm -f "$JOB/STOP"
  setsid nohup "$PY" scripts/final_job.py run --job-dir "$JOB" "${A[@]}" >> "$JOB/runner.out" 2>&1 < /dev/null &
  echo "runner started (pid $!); status: scripts/final_job.sh status"
}

case "$cmd" in
  start)
    if [ $# -gt 0 ]; then printf '%s\n' "$@" > "$ARGS_FILE"; fi
    start ;;
  stop)
    touch "$JOB/STOP"
    pid=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1])).get('runner_pid',''))" "$JOB/job_state.json" 2>/dev/null || true)
    if [ -n "$pid" ] && runner_alive; then kill -TERM "$pid" && echo "sent SIGTERM to runner $pid (child gets up to 90 s)"; else echo "runner not running"; fi ;;
  ensure)
    [ -f "$JOB/STOP" ] && exit 0
    case "$(job_status)" in done|failed) exit 0 ;; esac
    if ! runner_alive; then echo "[$(date -Is)] watchdog: runner not alive, restarting"; start; fi ;;
  status) "$PY" scripts/final_job.py status --job-dir "$JOB" ;;
  summary) "$PY" scripts/final_job.py summary --job-dir "$JOB" ;;
  reset) "$PY" scripts/final_job.py reset --job-dir "$JOB" ;;
  log)
    stage=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1])).get('current_stage') or '')" "$JOB/job_state.json" 2>/dev/null || true)
    tail -n 30 "$JOB/job.log"; echo "----- logs/${stage}.log"; tail -n 20 "$JOB/logs/${stage}.log" 2>/dev/null || true ;;
  install-watchdog)
    line="*/10 * * * * $(pwd)/scripts/final_job.sh ensure >> $(pwd)/$JOB/watchdog.log 2>&1  $TAG"
    (crontab -l 2>/dev/null | grep -vF "$TAG" || true; echo "$line") | crontab -
    echo "installed: $line" ;;
  remove-watchdog)
    (crontab -l 2>/dev/null | grep -vF "$TAG" || true) | crontab -
    echo "watchdog removed" ;;
  *) echo "unknown command $cmd"; exit 2 ;;
esac

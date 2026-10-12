#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="${UNISIGN_SESSION:-unisign_aslc_features}"
OUTPUT_DIR="${1:-$REPO_ROOT/out/asl_citizen_unisign_pose_features_t64}"
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session already exists: $SESSION" >&2
    exit 1
fi
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
printf -v JOB_COMMAND 'bash %q %q' "$REPO_ROOT/script/asl_citizen_features_job.sh" "$OUTPUT_DIR"
tmux new-session -d -s "$SESSION" -c "$REPO_ROOT" "$JOB_COMMAND"
tmux set-window-option -t "$SESSION:0" remain-on-exit on
printf 'Started tmux session: %s\nOutput directory: %s\n' "$SESSION" "$OUTPUT_DIR"

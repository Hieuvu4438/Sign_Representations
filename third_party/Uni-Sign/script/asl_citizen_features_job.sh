#!/usr/bin/env bash
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${1:-$REPO_ROOT/out/asl_citizen_unisign_pose_features_t64}"
mkdir -p "$OUTPUT_DIR"
cd "$REPO_ROOT"
for ATTEMPT in 1 2 3; do
    printf '{"stage":"running","attempt":%s,"pid":%s}\n' "$ATTEMPT" "$$" > "$OUTPUT_DIR/launcher_status.json"
    printf '\nAttempt %s started %s\n' "$ATTEMPT" "$(date -Is)" >> "$OUTPUT_DIR/run.log"
    bash script/run_asl_citizen_features.sh \
        --output-dir "$OUTPUT_DIR" --batch-size 64 --workers 8 --writers 4 \
        >> "$OUTPUT_DIR/run.log" 2>&1
    JOB_EXIT_CODE=$?
    if [[ "$JOB_EXIT_CODE" == 0 ]]; then
        printf '{"stage":"complete","attempt":%s,"exit_code":0}\n' "$ATTEMPT" > "$OUTPUT_DIR/launcher_status.json"
        exit 0
    fi
    printf 'Attempt %s exited with code %s at %s\n' "$ATTEMPT" "$JOB_EXIT_CODE" "$(date -Is)" >> "$OUTPUT_DIR/run.log"
    if [[ "$ATTEMPT" != 3 ]]; then
        sleep 10
    fi
done
printf '{"stage":"failed","attempt":3,"exit_code":%s}\n' "$JOB_EXIT_CODE" > "$OUTPUT_DIR/launcher_status.json"
exit "$JOB_EXIT_CODE"

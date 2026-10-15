#!/bin/bash
# Usage: ./scripts/run_background.sh [config_file] [--dry-run]
# Example: ./scripts/run_background.sh configs/fermi_pipeline_8perdecade.yaml
#
# Run it from a shell where the fermitools env is active:
#     micromamba activate fermi

# Always run from the repository root, no matter where the script is called from
cd "$(dirname "$0")/.." || exit 1

if ! command -v gtselect > /dev/null; then
    echo "ERROR: gtselect not found - activate the fermitools env first (micromamba activate fermi)" >&2
    exit 1
fi

CONFIG=${1:-configs/fermi_pipeline_8perdecade.yaml}
LOG_FILE="logs/pipeline_execution.log"
mkdir -p logs

echo "Starting pipeline in background..."
echo "Config: $CONFIG"
echo "Logging to: $LOG_FILE"

# Shift the first argument (config) so we can pass the rest (flags) to the script
if [ "$#" -ge 1 ]; then
    shift
fi

nohup python runners/parallel_run.py "$CONFIG" "$@" > "$LOG_FILE" 2>&1 &

PID=$!
echo "Process started with PID: $PID"
echo "To monitor progress, run: tail -f $LOG_FILE"

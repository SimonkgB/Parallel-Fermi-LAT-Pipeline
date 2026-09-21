#!/bin/bash
# Usage: ./scripts/run_background.sh [config_file]
# Example: ./scripts/run_background.sh configs/fermi_pipeline_8perdecade.yaml

# Always run from the repository root, no matter where the script is called from
cd "$(dirname "$0")/.." || exit 1

CONFIG=${1:-configs/fermi_pipeline_8perdecade.yaml}
LOG_FILE="logs/pipeline_execution.log"
mkdir -p logs

echo "Starting pipeline in background..."

# All Ftools environment setup (CALDB, FERMI_DIR, PFILES, ...) is handled inside
# runners/parallel_run.py, driven by the 'fermi_base' key in the config file.
# Here we only put the fermi bin dirs on PATH and init HEASoft so the tools
# can be found even when the env is not activated. NO hardcoded machine paths.
#
# The config is interpolated with OmegaConf, but we cannot use it here: the
# whole point of this block is to locate the fermitools env *before* that env's
# Python is on PATH, and the system python3 has neither omegaconf nor reliably
# yaml. So fermi_base is the one key that must stay a literal.
FERMI_BASE=$(grep -E '^fermi_base:' "$CONFIG" | sed 's/fermi_base:[[:space:]]*//; s/"//g; s/#.*//' | xargs)

case "$FERMI_BASE" in
    *'${'*)
        echo "ERROR: fermi_base in $CONFIG is interpolated: $FERMI_BASE" >&2
        echo "       It must be a literal path. This script has to read it with grep" >&2
        echo "       to find the fermitools env before that env's Python (and thus" >&2
        echo "       OmegaConf) is available, so it is the one key that cannot use" >&2
        echo "       \${...} interpolation." >&2
        exit 1
        ;;
esac

if [ -n "$FERMI_BASE" ] && [ -d "$FERMI_BASE" ]; then
    export PATH="$FERMI_BASE/bin:$FERMI_BASE/heasoft/bin:$PATH"
    # HEASoft init (needed for farith and general Ftools stability)
    if [ -f "$FERMI_BASE/heasoft/headas-init.sh" ]; then
        export HEADAS="$FERMI_BASE/heasoft"
        source "$HEADAS/headas-init.sh"
    fi
elif [ -n "$FERMI_BASE" ]; then
    echo "WARNING: fermi_base in $CONFIG points to a nonexistent dir: $FERMI_BASE"
fi

echo "Config: $CONFIG"
echo "Logging to: $LOG_FILE"

# Shift the first argument (config) so we can pass the rest (flags) to the script
if [ "$#" -ge 1 ]; then
    shift
fi

nohup python3 runners/parallel_run.py "$CONFIG" "$@" > "$LOG_FILE" 2>&1 &

PID=$!
echo "Process started with PID: $PID"
echo "To monitor progress, run: tail -f $LOG_FILE"

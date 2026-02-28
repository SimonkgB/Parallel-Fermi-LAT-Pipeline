#!/bin/bash
# Usage: ./run_background.sh [config_file]
# Example: ./run_background.sh fermi_pipeline.yaml

CONFIG=${1:-fermi_pipeline.yaml}
LOG_FILE="pipeline_execution.log"

echo "Starting pipeline in background..."

# Explicitly adding fermi environment to PATH to avoid shell/activation issues
# Based on standard micromamba layout and user's home path
FERMI_BASE="/net/hume.uio.no/uio/hume/student-u42/skgberg/micromamba/envs/fermi"
FERMI_BIN="$FERMI_BASE/bin"
export PATH="$FERMI_BIN:$PATH"

#set Ftools environment variables manually since activation is skipped
export CALDB="$FERMI_BASE/share/fermitools/data/caldb"
export CALDBCONFIG="$CALDB/software/tools/caldb.config"
export CALDBALIAS="$CALDB/software/tools/alias_config.fits"
export CALDBROOT="$CALDB"
export FERMI_DIR="$FERMI_BASE/share/fermitools"
export FERMI_INST_DIR="$FERMI_DIR"

# PFILES setup is critical for Ftools parameter handling
# Ensure a local pfiles directory exists to avoid locking issues
mkdir -p "$HOME/pfiles"
export PFILES="$HOME/pfiles:.;$FERMI_BASE/heasoft/syspfiles:$FERMI_DIR/syspfiles"

# Set up HEASoft (needed for farith and general Ftools stability)
# Using path found in environment
export HEADAS="/net/hume.uio.no/uio/hume/student-u42/skgberg/micromamba/envs/fermi/heasoft"
if [ -f "$HEADAS/headas-init.sh" ]; then
    source "$HEADAS/headas-init.sh"
else
    # Fallback to older structure or just adding bin
    export PATH="$HEADAS/bin:$PATH"
fi

echo "Config: $CONFIG"
echo "Logging to: $LOG_FILE"
echo "Runner command: python3 (with modified PATH)"

# Shift the first argument (config) so we can pass the rest (flags) to the script
if [ "$#" -ge 1 ]; then
    shift
fi

nohup python3 runners/parallel_run.py "$CONFIG" "$@" > "$LOG_FILE" 2>&1 &

PID=$!
echo "Process started with PID: $PID"
echo "To monitor progress, run: tail -f $LOG_FILE"

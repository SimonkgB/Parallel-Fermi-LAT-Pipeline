#!/bin/bash
# Kill all Fermi processing.
#
# IMPORTANT: the orchestrator (python3 runners/parallel_run.py) must die FIRST.
# It runs a multiprocessing.Pool that continuously spawns new gtselect/gtbin/...
# subprocesses, so killing only the Fermi tools is useless - the still-alive
# pool immediately launches the next batch. Kill the master, then the tools.

echo "Killing pipeline orchestrator (parallel_run.py / single_run.py)..."
# Matches the master process AND its forked pool workers (same cmdline on Linux)
pkill -9 -u "$USER" -f "runners/parallel_run.py"
pkill -9 -u "$USER" -f "runners/single_run.py"
pkill -9 -u "$USER" -f "run_background.sh"

# Give the pool a moment to actually be gone before mopping up its children
sleep 1

echo "Killing any in-flight Fermi tool processes..."
for tool in gtselect gtmktime gtbin gtltcube gtexpcube2 gtltsum farith ftmerge; do
    pkill -9 -u "$USER" -f "$tool"
done

sleep 1

# Verify (include the orchestrator, not the defunct run_background.sh)
PATTERN="runners/parallel_run\.py|runners/single_run\.py|gtselect|gtmktime|gtbin|gtltcube|gtexpcube2|gtltsum|farith|ftmerge"
remaining=$(pgrep -u "$USER" -f "$PATTERN" | wc -l)
if [ "$remaining" -eq 0 ]; then
    echo "All Fermi processes killed successfully"
else
    echo "Warning: $remaining processes may still be running:"
    pgrep -u "$USER" -af "$PATTERN" | head -10
fi

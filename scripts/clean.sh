#!/bin/bash
# Remove generated outputs and logs. Run from anywhere.
cd "$(dirname "$0")/.." || exit 1

rm -rf gt*
rm -f logs/*.log
rm -rf data/*
rm -rf benchmark_env/

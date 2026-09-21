# Parallel Fermi LAT Pipeline

A parallel processing pipeline for Fermi LAT data analysis using Ftools. Massively boosts processing speed, and is used to process large datasets.

## Prerequisites

- Python 3.6+
- Fermi Science Tools (`fermitools`) installed via conda/mamba
- PyYAML (`pip install pyyaml`)
- Astropy (optional, for `image_sum` merge strategy)

- Depending on usage, large amounts of RAM may be required

## Project Structure

```
├── runners/
│   ├── parallel_run.py        # Main parallel pipeline runner
│   └── single_run.py          # Single-core baseline runner (for benchmarking)
├── configs/
│   └── fermi_pipeline.yaml    # Pipeline configuration
├── scripts/
│   ├── run_background.sh      # Background execution wrapper
│   ├── kill.sh                # Kill all running instances of the pipeline
│   ├── clean.sh               # Remove generated outputs/logs
│   └── parallel_expcube.py    # Energy-parallel gtexpcube2 (one process per bin)
├── ltcache/                   # Cache to store the LTCube 
├── logs/                      # All runtime logs end up here
├── data/                      # Merged output FITS files
└── README.md
```

## Quick Start

> **Note:** Always run the python entry points from the repository root.

### Configure your environment

Set the `fermi_base` key in `configs/fermi_pipeline.yaml` to your fermitools conda/micromamba environment (no code editing needed):

```yaml
fermi_base: "/path/to/your/micromamba/envs/fermi"
```

If omitted, it is derived automatically from the location of `gtselect` when the fermi env is activated. Stale `CALDB`/`PFILES` environment variables pointing to nonexistent paths are detected and corrected at startup.

### Run the pipeline

```bash
# Dry run (preview commands without executing)
python3 runners/parallel_run.py configs/fermi_pipeline.yaml --dry-run

# Full run
python3 runners/parallel_run.py configs/fermi_pipeline.yaml

# Background execution
./scripts/run_background.sh configs/fermi_pipeline.yaml
```

Logs are written to `logs/` (`runner.log`, `pipeline_execution.log`, ...).

## Command line

The command line is intentionally minimal, everything about a run is described in the YAML config:

| Flag | Description |
|------|-------------|
| `--dry-run` | Preview commands without executing |

To run only specific steps, or to skip the merge/post-processing phases, use the optional `run:` section in the YAML config instead:

```yaml
run:
  steps: [gtselect, gtmktime]   # Only run these steps (omit to run all)
  skip_merge: false             # Skip the merging phase
  skip_post: false              # Skip post-processing
```

`scripts/run_background.sh` is the supported launcher (it puts the fermitools `bin`
dirs on `PATH` and initialises HEASoft). You can also call `runners/parallel_run.py`
directly with any Python: if that interpreter lacks `omegaconf`, it re-execs itself
under the one in `fermi_base`.


## Inspiration/credits
The example yaml and currently working pipeline take complete inspiration of the great work of the Fermi ScienceTools team: https://github.com/fermi-lat/ScienceTools

## License

MIT

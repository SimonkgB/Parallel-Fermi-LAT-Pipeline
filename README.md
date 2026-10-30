# Parallel Fermi LAT Pipeline

A parallel processing pipeline for Fermi LAT data analysis using Ftools. Massively boosts processing speed, and is used to process large datasets.

## Prerequisites

- Python 3.6+
- A Conda-compatible package manager 
- Depending on usage, large amounts of RAM may be required

## Project Structure

```
├── runners/
│   ├── parallel_run.py        # Main parallel pipeline runner
│   ├── time_exclusions.py     # Time windows cut from the data (solar flares, ...)
│   └── single_run.py          # Single-core baseline runner (for benchmarking)
├── configs/
│   └── fermi_pipeline_8perdecade.yaml  # Pipeline configuration
├── scripts/
│   ├── run_background.sh      # Background execution wrapper
│   ├── kill.sh                # Kill all running instances of the pipeline
│   ├── clean.sh               # Remove generated outputs/logs
│   ├── hpx_divide.py          # flux = counts / exposure for HEALPix maps (post-processing)
│   └── parallel_expcube.py    # Energy-parallel gtexpcube2 (one process per bin)
├── ltcache/                   # Cache to store the LTCube 
├── logs/                      # All runtime logs end up here
├── data/                      # Merged output FITS files
└── README.md
```

## Quick Start

> **Note:** Always run the python entry points from the repository root.

### Configure your environment

```bash
micromamba create -n fermi -c conda-forge -c fermi fermi-scitools
micromamba install -n fermi -c conda-forge omegaconf
```

Everything below runs with the env **activated**, which provides the Fermi
tools, `CALDB`/`PFILES`, and the Python packages:

```bash
micromamba activate fermi
```

In a non-interactive script (e.g. a job on a compute node) initialise the
shell hook first:

```bash
eval "$(micromamba shell hook -s bash)"
micromamba activate fermi
```

### Run the pipeline

```bash
# Dry run (preview commands without executing)
python runners/parallel_run.py configs/fermi_pipeline_8perdecade.yaml --dry-run

# Full run
python runners/parallel_run.py configs/fermi_pipeline_8perdecade.yaml

# Background execution
./scripts/run_background.sh configs/fermi_pipeline_8perdecade.yaml
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

`scripts/run_background.sh` starts the same command detached (`nohup`), from the
activated env.


## Time exclusions (solar flares)

Time windows can be cut from the data with the `time_exclusions:` section of
the config. The cut acts in `gtmktime`: the `{time_exclusion}` placeholder in
its filter is replaced, per week, by the windows that overlap that week, so
the photons and the livetime of those intervals are removed together (counts
map, livetime cube and exposure stay consistent).

```yaml
time_exclusions:
  flsf_dir: "/path/to/solar_flares_FLSF"   # First Fermi-LAT Solar Flare catalog CSVs
  pad_hours: 1.0                            # added on both sides of every window
  windows:                                  # extra UTC windows, e.g. flares after 2018
    - ["2024-05-14T16:00", "2024-05-15T06:00"]
```

- The FLSF catalogue (Ajello et al. 2021, ApJS 252, 13) holds 45 flares of
  2010-2018. A flare's window runs from its GOES start to the end of its last
  LAT detection window; with 1 h padding the 45 windows total ~251 h.
- Product names carry a tag (`zmax90-xflsf1h`), so cut and uncut products
  never overwrite each other.
- Configuring windows without `{time_exclusion}` in the gtmktime filter is an
  error, not a silent no-op.
- Weeks without a window keep hitting livetime cubes cached before the cut.

## Inspiration/credits
The example yaml and currently working pipeline take complete inspiration of the great work of the Fermi ScienceTools team: https://github.com/fermi-lat/ScienceTools


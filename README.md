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
│   ├── check_events.sh        # Sanity-check event counts in data/
│   ├── parallel_expcube.py    # Energy-parallel gtexpcube2 (one process per bin)
│   ├── validate.py            # Sanity-check counts/exposure/flux maps
│   └── test_diagnostic.sh     # Diagnose where events are being lost
├── logs/                      # All runtime logs end up here
├── data/                      # Merged output FITS files
├── analysis.py                # Standalone science analysis (synchrotron subtraction)
├── benchmark.py               # Performance comparison tool
└── README.md
```

## Quick Start

> **Note:** Always run the python entry points from the repository root (the folder containing this README).

### 1. Configure your environment

Set the `fermi_base` key in `configs/fermi_pipeline.yaml` to your fermitools conda/micromamba environment (no code editing needed):

```yaml
fermi_base: "/path/to/your/micromamba/envs/fermi"
```

If omitted, it is derived automatically from the location of `gtselect` when the fermi env is activated. Stale `CALDB`/`PFILES` environment variables pointing to nonexistent paths are detected and corrected at startup.

### 2. Edit the pipeline config

Modify `configs/fermi_pipeline.yaml` to match your data:

```yaml
input:
  directory: "../weekly/photon"          # Path to your input FITS files
  pattern: "lat_photon_weekly_w*.fits"   # Glob pattern
resources:
  cores: 128                             # Number of CPU cores
  slice_scfile: true                     # Per-week spacecraft slices (default true)
  parallel_accum: true                   # Parallel image_sum accumulation (default true)
  ram_disk:
    enabled: true
    path: "/dev/shm/fermi_processing"
    min_ram_gb: 500
```

### 3. Run the pipeline

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

The command line is intentionally minimal — everything about a run is described in the YAML config:

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

## Benchmarking

Compare parallel vs single-core performance (single-core runs without the RAM disk as a true disk-I/O baseline):

```bash
# Quick test (10 files)
python3 benchmark.py --limit 10
```

Benchmark logs are written to `logs/benchmark_single.log` and `logs/benchmark_parallel.log`.

## Configuration

Config files are loaded with [OmegaConf](https://omegaconf.readthedocs.io), so any
value used more than once is declared once in the `vars:` block at the top and
referenced as `${vars.x}`. Interpolation is resolved at load time, so by the time a
command runs the text is identical to a hand-written literal.

- **`${root:<path>}`** resolves against the repository root, so config paths never
  depend on the directory you launched from (the entry points also `chdir` there).
- `${...}` (config interpolation, resolved once at load) never collides with
  `{...}` (per-worker placeholders like `{input}`/`{temp_gti}`, resolved per task).
- **`fermi_base` must stay a literal path**: `scripts/run_background.sh` reads it with
  `grep` to locate the fermitools env *before* that env's Python is available.
- Declaring `emin`/`emax`/`enumbins` once in `vars:` is what actually guarantees the
  energy binning matches between `gtbin_healpix` and `gtexpcube2` - previously a
  comment asked a human to keep them in sync.

Unknown keys, and keys that are declared but not implemented, are reported as warnings
at startup instead of being silently ignored.

`scripts/run_background.sh` is the supported launcher (it puts the fermitools `bin`
dirs on `PATH` and initialises HEASoft). You can also call `runners/parallel_run.py`
directly with any Python: if that interpreter lacks `omegaconf`, it re-execs itself
under the one in `fermi_base`.

The YAML config supports:

- **`run`**: Optional run options (`steps`, `skip_merge`, `skip_post`)
- **`resources.ltcube_cache`**: `{enabled, path}` — reuse chunk livetime cubes
  across runs. gtltcube dominates pipeline CPU (~87% measured) and its output
  does not depend on energy binning or map geometry, so re-binning/re-gridding
  runs skip it entirely (cache key: week file + SC data + GTI-relevant params +
  full gtltcube command). Safe to delete the cache dir at any time.
- **`resources.stream_merge`**: merge finished livetime cubes in batches in the
  background while processing runs (default `true`); batches are fixed by task
  id so outputs stay bit-reproducible
- **`steps`**: Chain of Ftools commands with `{input}`, `{output}` placeholders
- **`merging`**: Strategies: `image_sum`, `hierarchical`, `ftmerge`. `image_sum` writes
  the **union of every chunk's GTI** (not the first chunk's), so the merged map's
  livetime describes all the data it contains - see `merge_gti_rows`
- **`post_processing`**: Single-threaded follow-up commands
- **`cleanup`**: Auto-delete intermediate files between steps

## How the speed comes about

- **Per-week spacecraft slices** (`resources.slice_scfile`, default on): gtmktime,
  gtbin and gtltcube parse the *entire* scfile they are given, so handing each week
  the full merged mission file costs ~N_weeks redundant multi-GB table scans. The
  runner slices the merged file into per-week files (±1h padding) on the RAM disk
  and points each task at its own slice. Outputs are identical - only the scanning
  disappears. Falls back to the merged file per week on any slicing problem.
- **Overlapped input copies**: each worker rsyncs its own input to the RAM disk as
  step 0 of its task, so the copy phase is hidden behind compute instead of being
  an upfront barrier.
- **Parallel `image_sum` accumulation** (`resources.parallel_accum`, default on):
  finished chunk maps are summed and deleted by dedicated accumulator processes
  instead of serially in the main process (which becomes the throughput ceiling
  with many cores producing ~700MB CCUBE chunks).
- **Energy-parallel exposure** (`scripts/parallel_expcube.py`): drop-in for the
  `gtexpcube2` command in `post_processing` - runs one gtexpcube2 per energy bin
  in parallel and stacks the planes (verified bit-identical to a monolithic run).
- Wall time per phase is logged at the end of each run (`logs/runner.log`).

## Data naming convention

Output files follow `{product}_{energy}_{selection}_{grid}_{week range}.fits`, e.g.
`counts_1-10GeV_zmax90_gal-0.1deg_w009-w921.fits`. Products: `counts` (photon map),
`livetime` (livetime cube, energy-independent), `exposure` (exposure map),
`flux` (counts/exposure, the final science map).

## WIP
- **Binning**: Currently working on a method for binning energy ranges, while maintaining a healthy usage of RAM vs speed

## Inspiration/credits
The example yaml and currently working pipeline take complete inspiration of the great work of the Fermi ScienceTools team: https://github.com/fermi-lat/ScienceTools

## License

MIT

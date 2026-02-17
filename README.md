# Parallel Fermi LAT Pipeline

A parallel processing pipeline for Fermi LAT data analysis using Ftools. Massivly boosts processing speed, and is used to process large datasets.


## Prerequisites

- Python 3.6+
- Fermi Science Tools (`fermitools`) installed via conda/micromamba
- PyYAML (`pip install pyyaml`)
- Astropy (optional, for `image_sum` merge)

- Depends on usage, large amounts of RAM may be required

## Quick Start

### 1. Configure your environment

Set the `fermi_base` path in `runners/parallel_run.py` and `run_background.sh` to point to your fermitools conda/micromamba environment:

```python
# In runners/parallel_run.py -> setup_fermi_environment()
fermi_base = "/path/to/your/conda/envs/fermi"
```

### 2. Edit the pipeline config

Modify `fermi_pipeline.yaml` to match your data:

```yaml
input:
  directory: "../weekly/photon"          # Path to your input FITS files
  pattern: "lat_photon_weekly_w*.fits"   # Glob pattern
resources:
  cores: 36                              # Number of CPU cores
  ram_disk:
    enabled: true
    path: "/dev/shm/fermi_processing"
    min_ram_gb: 500
```

### 3. Run the pipeline

```bash
# Dry run (preview commands)
python3 runners/parallel_run.py fermi_pipeline.yaml --dry-run

# Full run
python3 runners/parallel_run.py fermi_pipeline.yaml

# Background execution
./run_background.sh fermi_pipeline.yaml
```

### 4. Run specific steps

```bash
python3 runners/parallel_run.py fermi_pipeline.yaml --steps gtselect,gtmktime
```

## Benchmarking

Compare parallel vs single-core performance:

```bash
# Quick test (10 files)
python3 benchmark.py --limit 10
```

## Project Structure

```
├── runners/
│   ├── parallel_run.py        # Main parallel pipeline runner
│   └── single_run.py          # Single-core baseline runner
├── fermi_pipeline.yaml    # Pipeline configuration
├── benchmark.py               # Performance comparison tool
├── run_background.sh          # Background execution wrapper
├── kill.sh                    # Kill all running instances of the pipeline
└── README.md
```

## CLI Options

| Flag | Description |
|------|-------------|
| `--dry-run` | Preview commands without executing |
| `--steps STEP1,STEP2` | Run only specific pipeline steps |
| `--skip-merge` | Skip the merging phase |
| `--skip-post` | Skip post-processing |
| `--keep-intermediate` | Don't delete RAM disk after run |

## Configuration

The YAML config supports:

- **`steps`**: Chain of Ftools commands with `{input}`, `{output}` placeholders
- **`merging`**: Strategies: `image_sum`, `hierarchical`, `ftmerge`
- **`post_processing`**: Single-threaded follow-up commands
- **`cleanup`**: Auto-delete intermediate files between steps

## License

MIT

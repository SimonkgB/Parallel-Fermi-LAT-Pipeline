#!/usr/bin/env python3
import os
import re
import sys
import glob
import shutil
import time
import hashlib
import subprocess
import multiprocessing
import tempfile
import argparse
from pathlib import Path
from datetime import datetime
import contextlib

def _reexec_under_fermi_python():
    """Re-exec under the fermitools interpreter if this one lacks omegaconf.

    Bare `python3 runners/parallel_run.py` picks up the system interpreter,
    which has yaml but not omegaconf, so the config could not be loaded at all.
    Rather than fail, do the same two-stage bootstrap scripts/run_background.sh
    does: read 'fermi_base' out of the config with a parser the system python
    definitely has, then hand off to that env's python.

    Returns only if the hand-off was not possible; the sentinel env var makes
    looping impossible if the fermi env is itself missing omegaconf.
    """
    if os.environ.get("_FERMI_REEXEC"):
        return
    config_path = next((a for a in sys.argv[1:] if not a.startswith("-")), None)
    if not config_path or not os.path.exists(config_path):
        return

    fermi_base = None
    try:
        import yaml
        with open(config_path) as f:
            fermi_base = (yaml.safe_load(f) or {}).get("fermi_base")
    except Exception:
        # No yaml either - fall back to a regex. fermi_base is required to be a
        # literal (every config says so next to the key), so this is safe.
        import re
        try:
            with open(config_path) as f:
                for line in f:
                    m = re.match(r'^fermi_base:\s*["\']?([^"\'#\s]+)', line)
                    if m:
                        fermi_base = m.group(1)
                        break
        except Exception:
            return

    if not fermi_base or "${" in str(fermi_base):
        return

    candidate = os.path.join(str(fermi_base), "bin", "python3")
    if (os.path.exists(candidate)
            and os.path.realpath(candidate) != os.path.realpath(sys.executable)):
        env = os.environ.copy()
        env["_FERMI_REEXEC"] = "1"
        os.execve(candidate, [candidate] + sys.argv, env)


try:
    from omegaconf import OmegaConf
except ImportError:
    _reexec_under_fermi_python()
    sys.exit(
        "ERROR: omegaconf is not available to this Python interpreter\n"
        f"       ({sys.executable}).\n"
        "       The configs use ${...} interpolation and need it.\n"
        "       Either activate the fermitools env:\n"
        "           micromamba activate fermi\n"
        "       or launch via the wrapper, which sets PATH for you:\n"
        "           ./scripts/run_background.sh <config>\n"
        "       To install: micromamba install -n fermi -c conda-forge omegaconf"
    )

# Root of the repository (this file lives in <repo>/runners/). Used both by the
# '${root:...}' config resolver and by the os.chdir() in the entry points, so
# that nothing depends on the directory the pipeline was launched from.

REPO_ROOT = Path(__file__).resolve().parent.parent


@contextlib.contextmanager
def worker_pool(n_workers):
    """A multiprocessing.Pool that closes and joins instead of terminating.

    `with worker_pool(...)` calls terminate() on exit, never
    close()/join(), so workers are killed mid-task and exceptions in the parent
    leave the teardown order unspecified. close()+join() waits for workers to
    finish and reclaims them deterministically; terminate() is kept for the
    error path only.

    Note: this is a robustness change, NOT a fix for the EMFILE crash seen on
    owl46. Measured directly, 20 consecutive pools held steady at 42 fds under
    both patterns - the plain `with` form does not leak. raise_fd_limit() is
    what addresses EMFILE.
    """
    pool = multiprocessing.Pool(n_workers)
    try:
        yield pool
        pool.close()
    except BaseException:
        pool.terminate()
        raise
    finally:
        pool.join()


def raise_fd_limit(logger):
    """Raise the open-file soft limit to the hard limit.

    The pipeline legitimately wants many concurrent fds (128 workers, each
    running an Ftool that opens several large FITS files, plus a side pool for
    streamed merges). Distributions ship a conservative soft limit but a very
    generous hard one, and any process may raise its own soft limit up to the
    hard limit without privileges. A full-mission run died with
    "OSError: [Errno 24] Too many open files" building a merge pool.
    """
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft < hard:
            resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
            logger.log(f"Raised open-file limit: {soft} -> {hard}")
        else:
            logger.log(f"Open-file limit: {soft} (already at hard limit)")
    except Exception as e:
        logger.log(f"Could not raise open-file limit: {e}")


def open_fd_count():
    """Number of fds this process currently holds (for leak diagnostics)."""
    try:
        return len(os.listdir("/proc/self/fd"))
    except OSError:
        return -1


# Import astropy only if needed for merging
try:
    from astropy.io import fits
    import numpy as np
    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False

# Spacecraft data kept on each side of a week's TSTART/TSTOP when slicing the
# merged scfile per week (must bracket every GTI edge; 1h >> one 30s SC row).
SC_SLICE_PAD_S = 3600.0

class Logger:
    def __init__(self, log_file=None):
        """
        log_file: Path to log file [str]
        start_time: Start time [float]
        """
        self.log_file = log_file
        self.start_time = time.time()
        if self.log_file:
            # Make sure the directory for the log file exists (e.g. logs/)
            log_dir = os.path.dirname(self.log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
        
    def log(self, message, percent=None):
        elapsed = int(time.time() - self.start_time)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        msg_str = f"[{timestamp}] "
        if percent is not None:
            msg_str += f"[{percent}%] "
        msg_str += f"[{elapsed}s] {message}"
        
        print(msg_str)
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(msg_str + "\n")

    def error(self, message):
        self.log(f"ERROR: {message}")



# Top-level keys the runner understands. 'science_models' is never read - it is
# documentation for a likelihood stage that does not exist yet - but it is a
# legitimate part of the config, so it must not be flagged as a typo.
KNOWN_TOP_LEVEL_KEYS = {
    'vars', 'fermi_base', 'resources', 'input', 'common_files', 'steps',
    'merging', 'post_processing', 'science_models', 'run',
}

KNOWN_RESOURCE_KEYS = {
    'cores', 'ram_disk', 'ltcube_cache', 'stream_merge', 'slice_scfile',
    'parallel_accum',
}

# Keys a config may declare that this runner silently ignores. Keep this list
# honest: ltcube_cache and stream_merge ARE implemented here, so they must not
# appear (a sibling pipeline lacks them - do not copy its list over).
UNIMPLEMENTED_KEYS = []


def validate_config(config, logger=None):
    """Warn about config keys that are declared but will have no effect.

    These all used to fail silently: a key with a typo, or one this runner
    never reads, simply did nothing. Post-processing scripts are checked too,
    because a missing script otherwise only surfaces at the very end of a
    multi-hour run.
    """
    def warn(msg):
        if logger is not None:
            logger.log(f"WARNING: config: {msg}")
        else:
            print(f"WARNING: config: {msg}")

    for path, why in UNIMPLEMENTED_KEYS:
        node = config
        for part in path:
            if not isinstance(node, dict) or part not in node:
                node = None
                break
            node = node[part]
        if node is not None:
            warn(f"'{'.'.join(path)}' is declared but not implemented - {why}")

    unknown = set(config) - KNOWN_TOP_LEVEL_KEYS - {
        'dry_run', 'skip_merge', 'skip_post', 'selected_steps'}
    for key in sorted(unknown):
        warn(f"unknown top-level key '{key}' (ignored)")

    for key in sorted(set(config.get('resources', {}) or {}) - KNOWN_RESOURCE_KEYS):
        warn(f"unknown key 'resources.{key}' (ignored)")

    # A post_processing step that shells out to a missing script fails only
    # after everything else has run; catch it at startup instead.
    for step in config.get('post_processing', []) or []:
        parts = (step.get('command') or '').split()
        for i, tok in enumerate(parts):
            if tok in ('python3', 'python') and i + 1 < len(parts):
                script = parts[i + 1]
                if not os.path.exists(script):
                    warn(f"post_processing step '{step.get('name')}' runs "
                         f"'{script}', which does not exist")
                break


class ConfigLoader:
    @staticmethod
    def load(path, logger=None):
        """Load a YAML config, resolving OmegaConf '${...}' interpolations.

        Returns a PLAIN dict, never a DictConfig: the rest of the pipeline
        mutates the config (config['dry_run'] = ...) and pickles slices of it
        across multiprocessing.Pool, so a DictConfig must never escape here.

        OmegaConf's '${...}' syntax does not collide with the per-worker
        str.format placeholders ('{input}', '{temp_gti}'): interpolation is
        fully resolved here, long before any .format() call.
        """
        if not OmegaConf.has_resolver("root"):
            # ${root:data} -> <repo>/data, so config paths never depend on cwd.
            OmegaConf.register_new_resolver(
                "root", lambda *p: os.path.normpath(str(REPO_ROOT.joinpath(*p)))
            )
        cfg = OmegaConf.load(path)
        config = OmegaConf.to_container(cfg, resolve=True)  # -> plain dict, picklable
        validate_config(config, logger)
        return config

class RAMDiskManager:
    def __init__(self, config, logger):
        """
        config: Configuration dictionary [yaml file]
        enabled: Whether to use RAM disk [bool]
        path: Path to RAM disk [str]
        min_ram: Minimum RAM required [int]
        logger: Logger object [Logger]
        initialized: Whether RAM disk is initialized [bool]
        """
        self.dry_run = config.get('dry_run', False)
        self.config = config.get('resources', {}).get('ram_disk', {})
        self.enabled = self.config.get('enabled', False)
        self.path = self.config.get('path', '/dev/shm/fermi_processing')
        self.min_ram = self.config.get('min_ram_gb', 0)
        self.logger = logger
        self.initialized = False

    def setup(self):
        if not self.enabled:
        # Initial check if RAM disk is enabled, for later, no reason to go into /proc/meminfo if RAM disk is disabled
            return False

        try:
        # Check total RAM, if not enough, disable RAM disk
            with open('/proc/meminfo', 'r') as f:
                mem_total = int(f.readline().split()[1]) // 1024 // 1024 # GB
            
            if mem_total < self.min_ram:
                self.logger.log(f"RAM Disk disabled: System has {mem_total}GB RAM, required {self.min_ram}GB")
                self.enabled = False
                return False
        except Exception as e:
            # If we cant determine the RAM, disable RAM disk, then we will just run without RAM disk
            self.logger.log(f"Could not determine system RAM: {e}")
            return False

        self.logger.log(f"Setting up RAM disk at {self.path}")
        os.makedirs(self.path, exist_ok=True)
        self.initialized = True
        return True

    def cleanup(self):
        if self.dry_run:
            # A dry run staged nothing here, but a previous real run may have -
            # deleting it would be a destructive side effect of a "dry" run.
            self.logger.log(f"[DryRun] Leaving RAM disk at {self.path} untouched.")
            return
        if self.initialized and os.path.exists(self.path):
            # Will only run if RAM disk was initialized and exists thus rsynced files are there
            if self.config.get('keep_intermediate', False):
                # If we are asked to keep the intermediate files, we will just log it
                self.logger.log(f"Keeping RAM disk at {self.path} (as requested).")
            else:
                # If we are not asked to keep the intermediate files, we will clean it up
                self.logger.log(f"Cleaning up RAM disk at {self.path}")
                shutil.rmtree(self.path, ignore_errors=True)

class JobScheduler:
    # Job scheduler class is responsible for scheduling jobs and managing resources over multiple cores
    def __init__(self, config, logger):
        """
        config: Configuration dictionary [yaml file]
        logger: Logger object [Logger]
        ram_disk: RAMDiskManager object [RAMDiskManager]
        input_dir: Input directory [str]
        pattern: Input file pattern [str]
        output_dir: Output directory [str]
        cores: Number of cores to use [int]
        """
        self.config = config
        self.logger = logger
        self.ram_disk = RAMDiskManager(config, logger)
        
        # Determine paths
        self.input_dir = config['input']['directory']
        self.pattern = config['input']['pattern']
        self.output_dir = "./data" # Default, should extract from config if possible
        
        # Determine cores (from config, or auto-detect)
        self.cores = config.get('resources', {}).get('cores', 'auto')
        if self.cores == 'auto':
            self.cores = multiprocessing.cpu_count()
        self.cores = int(self.cores)

    def discover_files(self):
        # Finds all files matching the pattern in the input directory
        source_path = Path(self.input_dir)
        files = sorted(source_path.glob(self.pattern))
        if not files:
            raise FileNotFoundError(f"No files found matching {self.pattern} in {self.input_dir}")
        self.logger.log(f"Found {len(files)} input files.")
        return files

    def prepare_environment(self, files):
        """Prepare the execution environment for parallel processing.
        
        Sets up the RAM disk if configured, copies common files to the working
        directory, and creates a list of working files for the pipeline.
        """
        dry_run = self.config.get('dry_run', False)
        use_ram = self.ram_disk.setup()
        # When inputs are copied into the RAM disk they are throwaway copies and
        # can be deleted as soon as gtselect has read them (frees ~100s of GB
        # progressively). When NOT using the RAM disk, working files ARE the
        # user's originals - never delete those.
        self._inputs_are_copies = use_ram

        working_files = []
        common_file_context = {}
        
        if use_ram:
            ram_input_dir = os.path.join(self.ram_disk.path, "input")
            ram_common_dir = os.path.join(self.ram_disk.path, "common")
            os.makedirs(ram_input_dir, exist_ok=True)
            
            # Handle Common Files (Spacecraft file, etc.)
            common_files = self.config.get('common_files', {})
            if common_files:
                os.makedirs(ram_common_dir, exist_ok=True)
                self.logger.log(f"Caching {len(common_files)} common files to RAM disk...")
                for key, path in common_files.items():
                    filename = os.path.basename(path)
                    dest = os.path.join(ram_common_dir, filename)
                    # Copy if not exists
                    if not dry_run and not os.path.exists(dest):
                         shutil.copy(path, dest) # Standard copy for single large file
                    common_file_context[key] = dest
                self.logger.log("Common files cached." if not dry_run
                                else "[DryRun] Would cache common files "
                                     "(multi-GB scfile copy skipped).")

            # Inputs are NOT copied upfront: each worker rsyncs its own file as
            # step 0 of its task, so the network copy overlaps with compute
            # instead of being a startup barrier before any processing begins.
            self.copy_sources = {}
            for f in files:
                dest = os.path.join(ram_input_dir, f.name)
                working_files.append(dest)
                self.copy_sources[dest] = str(f)
            self.logger.log(f"{len(files)} input copies deferred to workers (overlapped with processing).")
            self.working_dir = self.ram_disk.path
        else:
            working_files = [str(f) for f in files]
            # No RAM disk: use a temporary scratch directory for intermediate files
            # (never the current directory, which would litter it with chunk files)
            self.working_dir = tempfile.mkdtemp(prefix="fermi_work_")
            self.logger.log(f"No RAM disk: intermediate files go to {self.working_dir}")

            # For non-RAM disk, common files just point to their original location
            common_files = self.config.get('common_files', {})
            common_file_context = common_files.copy()

        # Per-week spacecraft slices (see slice_spacecraft_file docstring)
        sc_overrides = self.slice_spacecraft_file(files, common_file_context)

        return working_files, common_file_context, sc_overrides

    def slice_spacecraft_file(self, files, common_file_context):
        """Create per-week slices of the merged spacecraft file.

        gtmktime/gtltcube/gtbin parse the ENTIRE scfile they are given, so
        handing each of N weeks the full multi-GB merged mission file costs
        ~N redundant full-table scans - the dominant per-week CPU cost on
        full-mission runs. A week-sized slice (padded by SC_SLICE_PAD_S on
        each side) is equivalent for all three tools because livetime and
        GTIs only accumulate inside the week's own GTIs.

        Returns {file_index: slice_path}; missing indices (or an empty dict
        on any failure) mean that week keeps using the merged file.
        """
        if not self.config.get('resources', {}).get('slice_scfile', True):
            return {}
        sc_src = common_file_context.get('scfile')
        if not sc_src or not ASTROPY_AVAILABLE:
            return {}
        if self.config.get('dry_run', False):
            self.logger.log(f"[DryRun] Would slice {sc_src} into {len(files)} per-week spacecraft files")
            return {}

        t0 = time.time()
        try:
            with fits.open(sc_src, memmap=True) as hdul:
                names = [h.name for h in hdul]
                sc_ext = hdul['SC_DATA'] if 'SC_DATA' in names else hdul[1]
                start_col = np.asarray(sc_ext.data['START'], dtype=np.float64)

            slice_dir = os.path.join(self.working_dir, "sc_slices")
            os.makedirs(slice_dir, exist_ok=True)

            # Header-only reads of each week's TSTART/TSTOP (cheap, but parallel
            # anyway - on network storage each open can cost ~100ms)
            n_workers = max(1, min(self.cores, len(files)))
            with worker_pool(n_workers) as pool:
                ranges = pool.map(read_time_range, [str(f) for f in files])

            slice_tasks = []
            for i, (f, (tmin, tmax)) in enumerate(zip(files, ranges)):
                i0 = int(np.searchsorted(start_col, tmin - SC_SLICE_PAD_S, side='left'))
                i1 = int(np.searchsorted(start_col, tmax + SC_SLICE_PAD_S, side='right'))
                # Rows are contiguous ~30s intervals: back up one row so a row
                # STARTing before the padded window but STOPping inside it is kept.
                i0 = max(i0 - 1, 0)
                if i1 <= i0:
                    self.logger.log(f"scfile slice for {os.path.basename(str(f))} is empty - keeping merged file")
                    continue
                out_path = os.path.join(slice_dir, f"sc_w{i:04d}.fits")
                slice_tasks.append((i, sc_src, out_path, i0, i1))

            if not slice_tasks:
                return {}

            n_workers = max(1, min(self.cores, len(slice_tasks)))
            with worker_pool(n_workers) as pool:
                outcomes = pool.map(slice_sc_worker, slice_tasks)

            overrides = {}
            for idx, out_path, err in outcomes:
                if err:
                    self.logger.log(f"scfile slice failed (week keeps merged file): {err}")
                else:
                    overrides[idx] = out_path
            self.logger.log(f"Spacecraft slicing: {len(overrides)}/{len(files)} weekly slices in {time.time()-t0:.1f}s")
            return overrides
        except Exception as e:
            self.logger.log(f"Spacecraft slicing failed ({e}) - all weeks use the merged file")
            return {}

    def run(self):
        """Run the parallel Fermi pipeline.
        
        Orchestrates the entire pipeline: discovers files, sets up the environment,
        distributes work across cores, executes the pipeline steps, merges results,
        and cleans up.
        """
        phase_times = {}
        t_phase = time.time()
        files = self.discover_files()
        working_files, common_context, sc_overrides = self.prepare_environment(files)
        phase_times['prepare (common files + sc slices)'] = time.time() - t_phase

        num_files = len(working_files)

        # One task per file: imap_unordered hands files to whichever core is free,
        # so fast (small/empty) weeks don't leave cores idle waiting on slow ones.
        tasks = []
        selected_steps = self.config.get('selected_steps', None)

        # Ltcube cache: chunk livetime cubes depend only on the week's GTIs,
        # the SC data and the gtltcube parameters - NOT on energy binning or
        # map geometry. gtltcube dominates pipeline CPU (~87% measured), so
        # reruns with a different binning/geometry reuse cached cubes.
        cache_cfg = self.config.get('resources', {}).get('ltcube_cache', {}) or {}
        cache_task_info = None
        if cache_cfg.get('enabled', False) and not self.config.get('dry_run', False):
            cache_dir = os.path.abspath(cache_cfg.get('path', './ltcube_cache'))
            os.makedirs(cache_dir, exist_ok=True)
            cache_task_info = {'dir': cache_dir, 'sig': ltcube_signature(self.config['steps'])}
            self.logger.log(f"Ltcube cache enabled: {cache_dir}")

        copy_sources = getattr(self, 'copy_sources', {})
        for i, f in enumerate(working_files):
            task_context = dict(common_context)
            if i in sc_overrides:
                task_context['scfile'] = sc_overrides[i]
            task = {
                'id': i,
                'files': [f],
                'steps': self.config['steps'],
                'working_dir': self.working_dir,
                'dry_run': self.config.get('dry_run', False),
                'selected_steps': selected_steps,
                'common_context': task_context,
                'inputs_are_copies': getattr(self, '_inputs_are_copies', False),
                'copy_sources': {f: copy_sources[f]} if f in copy_sources else {}
            }
            if cache_task_info:
                # Identity of the ORIGINAL week file (the RAM-disk copy may be
                # deleted before the gtltcube step runs).
                st = files[i].stat()
                task['ltcube_cache'] = dict(
                    cache_task_info,
                    input_identity=f"{files[i].name}:{st.st_size}:{int(st.st_mtime)}")
            tasks.append(task)

        # LPT (largest-first) submission order: per-week processing time grows
        # with photon count, i.e. input file size. imap_unordered balances
        # dynamically, but if a big week STARTED in the last wave, every other
        # core would idle while it finished alone - submitting the largest
        # files first lets the small weeks fill the tail instead. Chunk ids
        # keep the original file order, so results and merging are unaffected.
        try:
            input_sizes = {i: f.stat().st_size for i, f in enumerate(files)}
            tasks.sort(key=lambda t: input_sizes.get(t['id'], 0), reverse=True)
        except OSError:
            pass  # submission order is only an optimization

        self.logger.log(f"Scheduling {num_files} files dynamically across {self.cores} workers (largest first).")

        # Which chunk outputs get 'image_sum' merged? Those are accumulated into a
        # running total and DELETED as each chunk finishes, so the RAM disk never
        # holds hundreds of large chunk maps at once (they would otherwise pile up
        # until the merge phase and overflow /dev/shm for big grids like CCUBE 0.1deg).
        self._img_accum = {}
        image_sum_keys = set()
        if ASTROPY_AVAILABLE and not self.config.get('dry_run', False):
            for m in self.config.get('merging', []):
                if m.get('strategy') == 'image_sum':
                    image_sum_keys.add(m['input_pattern'].strip("{}"))

        # Accumulator processes: summing a chunk into the running image_sum total
        # costs seconds for big cubes, and doing it inline here serializes it in
        # the main process - with many cores that becomes the throughput ceiling.
        # A few dedicated processes drain a queue of finished chunk paths instead
        # (chunks arrive slower than accumulators consume, so files don't pile up
        # on the RAM disk). Set resources.parallel_accum: false for the old
        # inline path.
        accum_queue = None
        accum_procs = []
        parallel_accum = self.config.get('resources', {}).get('parallel_accum', True)
        if image_sum_keys and parallel_accum:
            n_acc = max(1, min(8, self.cores))
            accum_queue = multiprocessing.Queue()
            for wid in range(n_acc):
                p = multiprocessing.Process(target=accumulate_image_sum_worker,
                                            args=(accum_queue, self.working_dir, wid))
                p.start()
                accum_procs.append(p)
            self.logger.log(f"Started {n_acc} image_sum accumulator processes.")

        # Stream stage-1 hierarchical (gtltsum) merges while processing runs:
        # finished chunk ltcubes are merged in batches of 8 by a small side
        # pool, so most of the merge phase hides behind the processing phase
        # instead of running after it. resources.stream_merge: false disables.
        stream_state = {}
        merge_pool = None
        if (not self.config.get('dry_run', False)
                and not self.config.get('skip_merge', False)
                and self.config.get('resources', {}).get('stream_merge', True)):
            for m in self.config.get('merging', []):
                if m.get('strategy') == 'hierarchical':
                    stream_state[m['input_pattern'].strip("{}")] = {
                        'paths': {}, 'done': {}, 'jobs': [],
                        'tool': m.get('tool', 'gtltsum')}

        self.logger.log(f"Starting parallel execution on {self.cores} cores...")
        t_phase = time.time()
        results = []
        total_tasks = len(tasks)
        completed = 0

        try:
            with worker_pool(self.cores) as pool:
                # Use imap_unordered to track progress as tasks finish
                for result in pool.imap_unordered(execute_worker, tasks):
                    # Hand image_sum chunk outputs to the accumulators (or fold them
                    # in inline as fallback) so they are summed and freed immediately.
                    if result.get('success') and image_sum_keys:
                        for key in image_sum_keys:
                            for fp in result.get('outputs', {}).get(key, []):
                                if accum_queue is not None:
                                    accum_queue.put((key, fp))
                                else:
                                    try:
                                        self._accumulate_image_sum(key, fp)
                                    except Exception as e:
                                        self.logger.error(f"Incremental image_sum failed for {fp}: {e}")
                            # Mark consumed so the merge phase won't look for deleted files
                            if key in result.get('outputs', {}):
                                result['outputs'][key] = []

                    # Feed finished hierarchical-merge inputs to the side pool.
                    # Batches are FIXED by task id (batch b = ids [8b, 8b+8)) and
                    # submitted once all of a batch's tasks completed - completion
                    # order then affects only WHEN a batch merges, never its
                    # composition or summation order, keeping outputs reproducible
                    # (livetime cubes hold float32 - reordered adds would drift).
                    if stream_state:
                        rid = result['id']
                        b = rid // 8
                        n_in_batch = min(8, total_tasks - b * 8)
                        for key, st in stream_state.items():
                            if result.get('success'):
                                for p in result.get('outputs', {}).get(key, []):
                                    st['paths'].setdefault(b, []).append((rid, p))
                            st['done'][b] = st['done'].get(b, 0) + 1
                            if st['done'][b] == n_in_batch:
                                # Empty weeks contribute no file; submit what exists
                                batch = [p for _, p in sorted(st['paths'].pop(b, []))]
                                if not batch:
                                    continue
                                if merge_pool is None:
                                    n_merge = max(2, min(16, self.cores // 8))
                                    merge_pool = multiprocessing.Pool(n_merge)
                                    self.logger.log(f"Streaming hierarchical merges on {n_merge} side workers.")
                                out = os.path.join(self.working_dir, f"stream_{key}_batch{b}.fits")
                                st['jobs'].append((b, batch, merge_pool.apply_async(
                                    merge_batch_worker, ((batch, out, st['tool']),))))

                    results.append(result)
                    completed += 1
                    percent = (completed / total_tasks) * 100
                    self.logger.log(f"Progress: {completed}/{total_tasks} files ({percent:.1f}%)")

        finally:
            # The streamed-merge side pool is created lazily inside the loop
            # above; without this it is leaked if the main pool raises.
            if merge_pool is not None and not getattr(merge_pool, '_state_closed', False):
                try:
                    merge_pool.close()
                except Exception:
                    merge_pool.terminate()
        phase_times['parallel processing'] = time.time() - t_phase

        if accum_procs:
            t_phase = time.time()
            self._drain_accumulators(accum_queue, accum_procs, image_sum_keys)
            phase_times['accumulator drain'] = time.time() - t_phase

        # Collect streamed stage-1 merge results. Failed batches fold their
        # chunk files back in so the final merge phase retries them normally.
        self._streamed = {}
        if stream_state:
            t_phase = time.time()
            if merge_pool is not None:
                merge_pool.close()
                merge_pool.join()
            for key, st in stream_state.items():
                batch_outs, retry = [], []
                # Batch-index order keeps the stage-2 input order deterministic
                for b, batch, job in sorted(st['jobs'], key=lambda x: x[0]):
                    out_path, err = job.get()
                    if err:
                        self.logger.error(f"Streamed merge batch failed - retrying its chunks in the merge phase: {err}")
                        retry.extend(batch)
                    else:
                        batch_outs.append(out_path)
                        for f in batch:  # summed into out_path; free the RAM disk now
                            try:
                                os.remove(f)
                            except OSError:
                                pass
                if st['jobs']:
                    self._streamed[key] = batch_outs + retry
                    self.logger.log(f"Streamed {len(batch_outs)}/{len(st['jobs'])} stage-1 "
                                    f"batches for {key} during processing.")
            phase_times['streamed merge wait'] = time.time() - t_phase

        results.sort(key=lambda x: x['id'])

        hits = sum(r.get('ltcube_cache', [0, 0])[0] for r in results if r.get('success'))
        stored = sum(r.get('ltcube_cache', [0, 0])[1] for r in results if r.get('success'))
        if hits or stored:
            self.logger.log(f"Ltcube cache: {hits} hits, {stored} newly stored.")

        # Basic Error Checking
        failed = [r for r in results if not r['success']]
        if failed:
            self.logger.error(f"{len(failed)} chunks failed.")
            for res in failed:
                 self.logger.log(f"Chunk {res['id']} error: {res['error']}")
            
            # If significant failures, ABORT to prevent zombie merging
            if len(failed) > len(tasks) * 0.5:
                self.logger.error("Too many chunks failed (>50%). Aborting pipeline before merge phase.")
                # Ensure cleanup happens before we exit!
                self.ram_disk.cleanup()
                self.cleanup_workdir()
                sys.exit(1)

        # Merge
        t_phase = time.time()
        if not self.config.get('skip_merge', False):
            self.merge_results(results)
            phase_times['merge'] = time.time() - t_phase
        else:
            self.logger.log("Skipping merge phase.")

        # Post-Processing
        t_phase = time.time()
        if not self.config.get('skip_post', False):
            self.run_post_processing()
            phase_times['post-processing'] = time.time() - t_phase
        else:
            self.logger.log("Skipping post-processing phase.")

        # Cleanup
        self.ram_disk.cleanup()
        self.cleanup_workdir()
        self.logger.log("Pipeline complete!")

        # Performance Breakdown
        self.print_performance_stats(results, len(working_files), phase_times)

    def cleanup_workdir(self):
        """Remove the temporary scratch dir used when no RAM disk is active.

        When the RAM disk is used, working_dir IS the RAM disk and
        RAMDiskManager.cleanup() handles it - nothing to do here.
        """
        if self.ram_disk.initialized:
            return
        workdir = getattr(self, 'working_dir', None)
        if not workdir or not os.path.exists(workdir):
            return
        # Respect the same keep_intermediate flag as the RAM disk
        keep = self.config.get('resources', {}).get('ram_disk', {}).get('keep_intermediate', False)
        if keep:
            self.logger.log(f"Keeping intermediate files at {workdir} (as requested).")
        else:
            self.logger.log(f"Cleaning up scratch dir {workdir}")
            shutil.rmtree(workdir, ignore_errors=True)

    def print_performance_stats(self, results, total_files, phase_times=None):
        # Wall time per pipeline phase - this is what to look at to see where a
        # run actually spent its time (goes through the logger so it lands in
        # logs/runner.log, unlike stdout on nohup/background runs).
        if phase_times:
            total_wall = sum(phase_times.values())
            self.logger.log("PHASE TIMINGS (wall time)")
            for phase, dur in phase_times.items():
                self.logger.log(f"  {phase:<38} {dur:>9.1f}s  ({100*dur/total_wall:.0f}%)")
            self.logger.log(f"  {'TOTAL':<38} {total_wall:>9.1f}s")

        # Aggregate per-step CPU time across workers
        step_times = {}
        for res in results:
            if not res['success']: continue
            worker_timings = res.get('timings', {})
            for step, duration in worker_timings.items():
                step_times[step] = step_times.get(step, 0.0) + duration

        if not step_times:
            return

        self.logger.log("=" * 40)
        self.logger.log("PERFORMANCE BREAKDOWN (Cumulative CPU Time)")
        self.logger.log("=" * 40)
        self.logger.log(f"{'Step':<20} | {'Total Time (s)':<15} | {'Avg Time/File (s)':<15}")
        self.logger.log("-" * 56)

        for step, total_duration in step_times.items():
            avg = total_duration / total_files if total_files > 0 else 0
            self.logger.log(f"{step:<20} | {total_duration:<15.2f} | {avg:<15.4f}")
        self.logger.log("-" * 56)
        self.logger.log("(Note: Total Time is sum across all cores. Divide by core count for wall time estimate.)")

    def _accumulate_image_sum(self, key, filepath):
        """Add one chunk map into the running image_sum total, then delete the file.

        Keeps a copy of the first chunk's HDU structure as a template so the final
        merged file has the right header/format. Accumulates in float64. Handles
        both image maps (primary HDU) and HEALPix maps (binary table).
        """
        if not filepath or not os.path.exists(filepath):
            return
        with fits.open(filepath, memmap=False) as hdul:
            map_ext, data = find_map_data(hdul)
            gti = extract_gti_rows(hdul)
            acc = self._img_accum.get(key)
            if acc is None:
                template = fits.HDUList([h.copy() for h in hdul])
                self._img_accum[key] = {
                    'template': template,
                    'map_ext': map_ext,
                    'data': np.asarray(data, dtype=np.float64).copy(),
                    'count': 1,
                    # The template's GTI describes only THIS chunk; union every
                    # chunk's intervals or the merged map claims one week of
                    # observing time for many weeks of photons.
                    'gti': gti,
                }
            else:
                np.add(acc['data'], data, out=acc['data'])
                acc['count'] += 1
                acc['gti'] = merge_gti_rows(acc.get('gti', []), gti)
        os.remove(filepath)

    def _drain_accumulators(self, accum_queue, accum_procs, image_sum_keys):
        """Stop the accumulator processes and combine their partial sums.

        Each accumulator persists, per key, a float64 partial (.npz with its
        chunk count) and a template FITS of the first chunk it saw. Those few
        partials are summed here into self._img_accum so the merge phase's
        write_accumulated_image_sum works exactly as in the inline path.
        """
        t0 = time.time()
        for _ in accum_procs:
            accum_queue.put(None)
        for p in accum_procs:
            p.join()
        dead = [p for p in accum_procs if p.exitcode != 0]
        if dead:
            self.logger.error(f"{len(dead)} accumulator process(es) exited abnormally - "
                              f"image_sum totals may be incomplete.")

        for key in image_sum_keys:
            total, count, template = None, 0, None
            gti_rows = []
            for npz_path in sorted(glob.glob(os.path.join(self.working_dir, f"accum_{key}_*.npz"))):
                with np.load(npz_path) as z:
                    data, n = z['data'], int(z['count'])
                    # .get-style guard: partials written before this field existed
                    part_gti = z['gti'].tolist() if 'gti' in z.files else []
                if total is None:
                    total = data
                else:
                    np.add(total, data, out=total)
                count += n
                gti_rows = merge_gti_rows(gti_rows, part_gti)
                if template is None:
                    tpl_path = npz_path[:-4].replace(f"accum_{key}_", f"accum_template_{key}_") + ".fits"
                    with fits.open(tpl_path) as th:
                        template = fits.HDUList([h.copy() for h in th])
                os.remove(npz_path)
            if total is None:
                continue
            map_ext, _ = find_map_data(template)
            self._img_accum[key] = {'template': template, 'map_ext': map_ext,
                                    'data': total, 'count': count, 'gti': gti_rows}
            for tpl in glob.glob(os.path.join(self.working_dir, f"accum_template_{key}_*.fits")):
                os.remove(tpl)
        self.logger.log(f"Accumulator drain + partial combine took {time.time()-t0:.1f}s")

    def write_accumulated_image_sum(self, key, output_file):
        """Write the incrementally-accumulated image_sum total to output_file."""
        acc = self._img_accum[key]
        self.logger.log(f"Writing incrementally-merged {key}: {acc['count']} chunks -> {output_file}")
        write_map_data(acc['template'], acc['map_ext'], acc['data'])
        # Union of every chunk's GTI, not the first chunk's (see merge_gti_rows).
        live = write_merged_gti(acc['template'], acc.get('gti'))
        if live is not None:
            self.logger.log(f"  merged GTI: {len(acc['gti'])} intervals, "
                            f"{live / 86400.0:.2f} days livetime from {acc['count']} chunks")
        acc['template'].writeto(output_file, overwrite=True)
        self.logger.log(f"Created merged file: {output_file}")

    def merge_results(self, results):
        """Dispatch merge operations based on the 'merging' config.
        
        Collects all output files from worker results, groups them by type
        (e.g. chunk_ccube, chunk_ltcube), and merges each group using
        the strategy specified in the YAML config (ftmerge, image_sum, or hierarchical).
        """
        self.logger.log("Starting merging phase...")
        merging_config = self.config.get('merging', [])
        
        # Flatten results
        files_by_type = {}
        for res in results:
            if not res['success']: continue
            for key, filepath in res['outputs'].items():
                if key not in files_by_type:
                    files_by_type[key] = []
                files_by_type[key].extend(filepath)

        for merge_step in merging_config:
            strategy = merge_step['strategy']
            output_file = merge_step['output_file']
            pattern_key = merge_step['input_pattern'].strip("{}")

            # The output dir (e.g. ./data) need not exist on a fresh clone -
            # without this the whole run's work is lost at the final write.
            out_dir = os.path.dirname(output_file)
            if out_dir and not self.config.get('dry_run', False):
                os.makedirs(out_dir, exist_ok=True)

            # image_sum chunks were accumulated incrementally during processing
            # (files already summed and deleted) - just write the running total.
            if strategy == 'image_sum' and pattern_key in getattr(self, '_img_accum', {}):
                if self.config.get('dry_run', False):
                    self.logger.log(f"[DryRun] Would write accumulated {pattern_key} to {output_file}")
                else:
                    self.write_accumulated_image_sum(pattern_key, output_file)
                continue

            input_files = files_by_type.get(pattern_key, [])

            # Hierarchical inputs may already be partially merged by the
            # streaming side pool - continue from its stage-1 outputs (the
            # original chunk paths in files_by_type were consumed/deleted).
            streamed = getattr(self, '_streamed', {}).get(pattern_key)
            if strategy == 'hierarchical' and streamed is not None:
                input_files = streamed

            if not input_files:
                # If we are running partial steps, this is expected
                if self.config.get('selected_steps'):
                    self.logger.log(f"Skipping merge step {merge_step['name']}: No input files generated (expected during partial run).")
                else:
                    self.logger.error(f"No input files found for merge step {merge_step['name']} (looking for {pattern_key})")
                continue
                
            self.logger.log(f"Merging {len(input_files)} files for {merge_step['name']} using {strategy}...")
            
            if self.config.get('dry_run', False):
                 self.logger.log(f"[DryRun] Would merge {len(input_files)} files to {output_file} using {strategy}")
                 continue

            if strategy == 'image_sum':
                self.merge_image_sum(input_files, output_file)
            elif strategy == 'ftmerge':
                self.merge_ftmerge(input_files, output_file)
            elif strategy == 'hierarchical':
                self.merge_hierarchical(input_files, output_file, tool=merge_step.get('tool', 'ftmerge'))
            else:
                self.logger.error(f"Unknown merge strategy: {strategy}")

    def merge_ftmerge(self, files, output_file):
        """Merge FITS event tables using HEASoft's ftmerge.
        
        Writes input paths to a list file and calls ftmerge with @listfile syntax.
        Best for event-type data (photon lists, GTI tables) where rows are appended.
        """
        if not files: return
        
        list_file = output_file + ".list"
        with open(list_file, 'w') as f:
            for path in files:
                f.write(path + "\n")
        
        try:
            cmd = f"ftmerge @{list_file} {output_file} clobber=yes"
            subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.logger.log(f"Created merged file (ftmerge): {output_file}")
            os.remove(list_file)
        except Exception as e:
            self.logger.error(f"ftmerge failed: {e}")

    def merge_image_sum(self, files, output_file):
        """Merge FITS images by pixel-wise summation using astropy.
        
        Opens each file and adds pixel values to a running sum.
        Used for count maps (ccube) where the total counts = sum of individual maps.
        Requires astropy.
        """
        if not ASTROPY_AVAILABLE:
            self.logger.error("Astropy not installed, cannot perform image_sum")
            return

        if not files:
            return

        try:
            with fits.open(files[0]) as hdul:
                result_hdul = fits.HDUList([hdu.copy() for hdu in hdul])
                map_ext, _ = find_map_data(hdul)

            # Parallel partial sums: each worker reads+sums a slice of the files
            # (the file reads are the expensive part), then the few partial
            # arrays are added together here.
            n_workers = min(self.cores, len(files))
            if n_workers > 1:
                slices = [files[i::n_workers] for i in range(n_workers)]
                with worker_pool(n_workers) as pool:
                    partials = pool.map(partial_sum_worker, slices)
            else:
                partials = [partial_sum_worker(files)]

            base_data = partials[0]
            for p in partials[1:]:
                base_data += p

            write_map_data(result_hdul, map_ext, base_data)
            # Union the GTIs of every input, not just files[0] -- see merge_gti_rows.
            all_gti = []
            for f in files:
                with fits.open(f) as hdul:
                    all_gti = merge_gti_rows(all_gti, extract_gti_rows(hdul))
            live = write_merged_gti(result_hdul, all_gti)
            if live is not None:
                self.logger.log(f"  merged GTI: {len(all_gti)} intervals, "
                                f"{live / 86400.0:.2f} days livetime from {len(files)} files")
            result_hdul.writeto(output_file, overwrite=True)
            self.logger.log(f"Created merged file: {output_file}")

        except Exception as e:
            self.logger.error(f"Image sum failed: {e}")

    def merge_hierarchical(self, files, output_file, tool='gtltsum'):
        """Merge files in a tree-reduction pattern using a pairwise tool.

        Groups files into batches of 8, merges each batch pairwise, then
        repeats on the results until a single file remains. Used for livetime
        cubes (gtltsum) which require pairwise combination rather than
        simple concatenation or pixel addition.

        The batches within a stage are independent of each other, so they are
        merged in parallel across cores (the pairwise merging *within* a batch
        is inherently sequential).
        """
        current_files = files
        stage = 0
        # Write temp merge files to the fast local working dir (RAM disk), NOT the
        # final output dir - the latter is often a network filesystem (stornext),
        # and ~100 batch workers writing/locking there concurrently fails. Falls
        # back to the output dir if no working_dir is set.
        temp_dir = getattr(self, 'working_dir', None) or os.path.dirname(output_file)

        # Clear any stale temp merge files from a previous aborted run - both in
        # the current temp dir AND the output dir (older versions wrote temps
        # next to the output, e.g. into ./data, and could leave them behind).
        for d in {temp_dir, os.path.dirname(output_file)}:
            for stale in glob.glob(os.path.join(d, "temp_merge_stage*")):
                try:
                    os.remove(stale)
                except OSError:
                    pass

        while len(current_files) > 1:
            stage += 1
            batches = [current_files[i:i+8] for i in range(0, len(current_files), 8)]

            batch_tasks = []
            for i, batch in enumerate(batches):
                batch_out = os.path.join(temp_dir, f"temp_merge_stage{stage}_batch{i}.fits")
                batch_tasks.append((batch, batch_out, tool))

            n_workers = min(self.cores, len(batches))
            self.logger.log(f"Merge stage {stage}: {len(current_files)} files -> {len(batches)} batches ({n_workers} in parallel, {open_fd_count()} fds open)")

            if n_workers > 1:
                with worker_pool(n_workers) as pool:
                    outcomes = pool.map(merge_batch_worker, batch_tasks)
            else:
                outcomes = [merge_batch_worker(t) for t in batch_tasks]

            # Keep successful batch outputs; log (but don't abort on) failed ones
            next_files = []
            for out_path, err in outcomes:
                if err:
                    self.logger.error(f"Merge batch failed (skipping its files): {err}")
                elif out_path:
                    next_files.append(out_path)

            if not next_files:
                self.logger.error(f"All batches failed in merge stage {stage}; aborting hierarchical merge for {output_file}")
                return

            # Remove the previous stage's temp inputs (not the original chunks at stage 1)
            if stage > 1:
                for f in current_files:
                    try:
                        os.remove(f)
                    except OSError:
                        pass
            current_files = next_files

        if current_files:
            # shutil.move handles the RAM-disk -> network-disk case (os.rename can't
            # cross filesystems); overwrite any existing output.
            if os.path.exists(output_file):
                os.remove(output_file)
            shutil.move(current_files[0], output_file)
            self.logger.log(f"Hierarchical merge complete: {output_file}")

    def run_post_processing(self):
        post_steps = self.config.get('post_processing', [])
        dry_run = self.config.get('dry_run', False)
        
        for step in post_steps:
            self.logger.log(f"Running post-processing: {step['name']}")
            try:
                if dry_run:
                    self.logger.log(f"[DryRun] {step['command']}")
                else:
                    subprocess.run(step['command'], shell=True, check=True)
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Post-processing step {step['name']} failed: {e}")


LTCUBE_CACHE_VERSION = "v1"


def ltcube_signature(steps):
    """Build the parameter part of the ltcube cache key.

    Only what can change the livetime cube goes in: the FULL gtltcube command
    template (dcostheta/binsz/phibins/zmax/...) plus the pieces of earlier
    steps that shape the GTIs handed to it - gtselect time cuts and the
    gtmktime filter/roicut. Deliberately NOT included: energy range, event
    class, map geometry - so re-binning/re-gridding runs get cache hits.
    """
    parts = [LTCUBE_CACHE_VERSION]
    for st in steps:
        cmd = " ".join(st.get('command', '').split())
        tool = cmd.split()[0] if cmd else ""
        if tool == 'gtltcube':
            parts.append(cmd)
        elif tool == 'gtselect':
            parts.extend(t for t in cmd.split() if t.startswith(('tmin=', 'tmax=')))
        elif tool == 'gtmktime':
            m = re.search(r'filter="[^"]*"|filter=\S+', cmd)
            parts.append(m.group(0) if m else "")
            m = re.search(r'roicut=\S+', cmd)
            parts.append(m.group(0) if m else "")
    return "|".join(parts)


def sc_fingerprint(path):
    """Cheap content fingerprint of a spacecraft (slice) file.

    Row count, time span and total livetime pin down the SC data actually
    used, independent of the slice file's name/mtime (slice names shift when
    new weeks are added to the input set). For a big merged SC file (slicing
    disabled), fall back to name:size - content-reading hundreds of MB per
    task would cost a good chunk of what the cache saves.
    """
    try:
        size = os.path.getsize(path)
        if size > 200 * 1024 * 1024:
            return f"big:{os.path.basename(path)}:{size}"
        with fits.open(path, memmap=True) as hdul:
            d = hdul['SC_DATA'].data
            return (f"{len(d)}:{d['START'][0]:.3f}:{d['STOP'][-1]:.3f}:"
                    f"{float(np.sum(d['LIVETIME'])):.6f}")
    except Exception:
        try:
            st = os.stat(path)
            return f"stat:{os.path.basename(path)}:{st.st_size}:{int(st.st_mtime)}"
        except OSError:
            return "none"


def ltcube_cache_file(cache_info, scfile):
    """Cache path for one week's ltcube: params + input week + SC data."""
    key = "|".join([cache_info['sig'], cache_info['input_identity'],
                    sc_fingerprint(scfile or "")])
    digest = hashlib.sha256(key.encode()).hexdigest()[:32]
    return os.path.join(cache_info['dir'], f"ltcube_{digest}.fits")


def execute_worker(task):
    """Execute a chunk of pipeline steps on a list of files.
    
    This function is executed in parallel for each chunk of input files.
    It runs the specified pipeline steps sequentially for each file in the chunk,
    handling file naming, context substitution, and error checking.
    """
    chunk_id = task['id']
    files = task['files']
    steps = task['steps']
    working_dir = task['working_dir']
    dry_run = task.get('dry_run', False)
    selected_steps = task.get('selected_steps', None) # List of step names or None
    inputs_are_copies = task.get('inputs_are_copies', False)

    # Index of the last step that reads the raw {input}; once it has run, the
    # input RAM-disk copy can be deleted (frees the biggest RAM-disk consumer as
    # the run progresses instead of holding all inputs until the end).
    last_input_step_idx = -1
    for _si, _st in enumerate(steps):
        if '{input}' in _st.get('command', ''):
            last_input_step_idx = _si
    
    chunk_outputs = {}
    timings = {}
    cache_stats = [0, 0]  # [hits, stored]

    try:
        worker_id = chunk_id
        common_context = task.get('common_context', {})

        # Private PFILES dir for this pool worker (avoids .par file lock contention)
        worker_env = None if dry_run else make_worker_env()

        copy_sources = task.get('copy_sources', {})

        for file_idx, input_file in enumerate(files):
            # Deferred RAM-disk copy: fetch this task's input now (overlapped
            # with other workers' compute) instead of in an upfront barrier.
            copy_src = copy_sources.get(input_file)
            if copy_src and not dry_run:
                t_copy = time.time()
                subprocess.run(["rsync", "-a", "--update", copy_src, input_file],
                               check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                timings['input_copy'] = timings.get('input_copy', 0.0) + (time.time() - t_copy)

            file_context = {
                'input': input_file,
                'worker_id': worker_id,
                'file_idx': file_idx,
                'chunk_id': chunk_id,
                'basename': os.path.basename(input_file).replace('.fits', '')
            }
            # Merge common files (e.g. {scfile}) into context
            file_context.update(common_context)
            
            for step_idx, step in enumerate(steps):
                cmd_template = step['command']
                step_outputs = step.get('outputs', [])
                step_name = step.get('name', '')
                
                # Skip steps not in the selected list
                should_run = True
                if selected_steps is not None:
                     if step_name not in selected_steps:
                         should_run = False
                
                # Resolve output paths (even for skipped steps, to populate context)
                step_paths = []
                for pat in step_outputs:
                    key = pat.strip("{}") 
                    filename = f"core{chunk_id}_file{file_idx}_{key}.fits"
                    path = os.path.join(working_dir, filename)
                    file_context[key] = path
                    step_paths.append((key, path))

                if should_run:
                    # Format command
                    try:
                        cmd = cmd_template.format(**file_context)
                    except KeyError as e:
                        return {'id': chunk_id, 'success': False, 'error': f"Missing param {e} in command template: {cmd_template}"}
                    
                    # Execute
                    start_t = time.time()
                    if dry_run:
                        # Only print for the first task to avoid log spam
                        if chunk_id == 0 and file_idx == 0:
                            print(f"[DryRun] Worker {chunk_id}: {cmd}")
                        # Register "fake" outputs for dry run
                        for key, path in step_paths:
                            if key not in chunk_outputs:
                                chunk_outputs[key] = []
                            chunk_outputs[key].append(path)
                    else:
                        # Ltcube cache: a chunk livetime cube depends only on
                        # the week's GTIs + SC data + gtltcube params, so a
                        # matching cube from an earlier run replaces the single
                        # most expensive tool invocation with a file copy.
                        cache_file = None
                        cache_info = task.get('ltcube_cache')
                        if (cache_info and len(step_paths) == 1
                                and cmd_template.lstrip().startswith('gtltcube')):
                            cache_file = ltcube_cache_file(cache_info, file_context.get('scfile'))

                        if cache_file and os.path.exists(cache_file):
                            shutil.copy(cache_file, step_paths[0][1])
                            cache_stats[0] += 1
                            step_name = f"{step_name} (cached)"
                            for key, path in step_paths:
                                chunk_outputs.setdefault(key, []).append(path)
                        else:
                            try:
                                # Capture output to help debugging if it fails
                                proc = run_gtool_with_retry(cmd, worker_env)

                                # If success, register outputs
                                for key, path in step_paths:
                                    if key not in chunk_outputs:
                                        chunk_outputs[key] = []
                                    chunk_outputs[key].append(path)

                                if cache_file:
                                    # copy to temp + atomic rename so concurrent
                                    # workers never read a half-written cache entry
                                    try:
                                        tmp = f"{cache_file}.tmp{os.getpid()}"
                                        shutil.copy(step_paths[0][1], tmp)
                                        os.replace(tmp, cache_file)
                                        cache_stats[1] += 1
                                    except OSError as ce:
                                        print(f"Worker {chunk_id}: ltcube cache write failed (non-fatal): {ce}")

                            except subprocess.CalledProcessError as e:
                                # Benign empty-week errors - the week has no valid
                                # data (e.g. LAT safe-hold weeks): 'No GTIs found'
                                # comes from tools fed an empty GTI file, 'Zero rows
                                # returned' from gtmktime when the filter rejects
                                # every SC row of the week's slice. Skip this file's
                                # REMAINING steps too - they would only fail on the
                                # missing intermediate output.
                                stderr = e.stderr or ""
                                if "No GTIs found" in stderr or "Zero rows returned" in stderr:
                                    reason = stderr.strip().splitlines()[-1] if stderr.strip() else "empty"
                                    print(f"Worker {chunk_id}: Skipping empty week {file_context['basename']} ({reason})")
                                    break
                                else:
                                    # Include stderr in the error message
                                    error_msg = f"Command '{cmd}' failed.\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}"
                                    raise RuntimeError(error_msg) from e
                    
                    duration = time.time() - start_t
                    timings[step_name] = timings.get(step_name, 0.0) + duration

                    # Cleanup (only if not dry run)
                    if not dry_run:
                        cleanup_pats = step.get('cleanup', [])
                        for pat in cleanup_pats:
                            key = pat.strip("{}")
                            path = file_context.get(key)
                            if path and os.path.exists(path):
                                os.remove(path)

                    # Free the input RAM-disk copy once the last step that reads
                    # {input} has run - nothing after it needs the original.
                    if (inputs_are_copies and not dry_run
                            and step_idx == last_input_step_idx
                            and os.path.exists(input_file)):
                        try:
                            os.remove(input_file)
                        except OSError:
                            pass

        return {'id': chunk_id, 'success': True, 'outputs': chunk_outputs,
                'timings': timings, 'ltcube_cache': cache_stats}

    except Exception as e:
        return {'id': chunk_id, 'success': False, 'error': str(e)}


def accumulate_image_sum_worker(queue, out_dir, worker_id):
    """Drain (key, chunk_path) items from the queue: sum each map into an
    in-memory float64 partial and delete the chunk file. On the None sentinel,
    persist each key's partial as accum_{key}_{worker_id}.npz (data + count)
    and the first chunk's HDU structure as accum_template_{key}_{worker_id}.fits.

    Runs as a dedicated process so chunk summing never serializes in the main
    process. A failed chunk is reported and skipped (same policy as the inline
    path: the merge continues without it).
    """
    accum = {}
    while True:
        item = queue.get()
        if item is None:
            break
        key, fp = item
        try:
            if not fp or not os.path.exists(fp):
                continue
            with fits.open(fp, memmap=False) as hdul:
                _, data = find_map_data(hdul)
                gti = extract_gti_rows(hdul)
                a = accum.get(key)
                if a is None:
                    template = fits.HDUList([h.copy() for h in hdul])
                    accum[key] = {'template': template,
                                  'data': np.asarray(data, dtype=np.float64).copy(),
                                  'count': 1, 'gti': gti}
                else:
                    np.add(a['data'], data, out=a['data'])
                    a['count'] += 1
                    a['gti'] = merge_gti_rows(a.get('gti', []), gti)
            os.remove(fp)
        except Exception as e:
            print(f"Accumulator {worker_id}: image_sum failed for {fp}: {e}")

    for key, a in accum.items():
        # GTI travels with the partial: shape (N, 2), empty-safe.
        gti_arr = np.asarray(a.get('gti') or [], dtype=np.float64).reshape(-1, 2)
        np.savez(os.path.join(out_dir, f"accum_{key}_{worker_id}.npz"),
                 data=a['data'], count=a['count'], gti=gti_arr)
        a['template'].writeto(os.path.join(out_dir, f"accum_template_{key}_{worker_id}.fits"),
                              overwrite=True)


def read_time_range(fits_path):
    """TSTART/TSTOP of a FITS file from the first header that carries them."""
    with fits.open(fits_path) as hdul:
        for hdu in hdul:
            h = hdu.header
            if 'TSTART' in h and 'TSTOP' in h:
                return float(h['TSTART']), float(h['TSTOP'])
    raise ValueError(f"No TSTART/TSTOP header in {fits_path}")


def slice_sc_worker(args):
    """Write rows [i0:i1) of the spacecraft SC_DATA table as a standalone file.

    Returns (idx, out_path, None) on success or (idx, None, error) on failure
    so the caller can keep the merged file for just that week.
    """
    idx, sc_src, out_path, i0, i1 = args
    try:
        with fits.open(sc_src, memmap=True) as hdul:
            names = [h.name for h in hdul]
            sc_ext = hdul['SC_DATA'] if 'SC_DATA' in names else hdul[1]
            rows = sc_ext.data[i0:i1]
            new_sc = fits.BinTableHDU(data=rows, header=sc_ext.header.copy())
            primary = fits.PrimaryHDU(header=hdul[0].header.copy())
            # Keep the time-span keywords consistent with the sliced rows
            tstart, tstop = float(rows['START'][0]), float(rows['STOP'][-1])
            for h in (primary.header, new_sc.header):
                if 'TSTART' in h:
                    h['TSTART'] = tstart
                if 'TSTOP' in h:
                    h['TSTOP'] = tstop
            fits.HDUList([primary, new_sc]).writeto(out_path, overwrite=True)
        return (idx, out_path, None)
    except Exception as e:
        return (idx, None, f"{out_path}: {e}")


def extract_gti_rows(hdul):
    """GTI rows of an open HDUList as [[start, stop], ...] (empty if none)."""
    names = [h.name for h in hdul]
    if 'GTI' not in names or hdul['GTI'].data is None:
        return []
    g = hdul['GTI'].data
    return [[float(a), float(b)] for a, b in zip(g['START'], g['STOP'])]


def merge_gti_rows(existing, new_rows):
    """Union of two GTI row-sets, sorted by START and merged where they touch.

    Why this matters: image_sum only sums the MAP data, so without this the
    merged product inherits the first chunk's GTI verbatim -- i.e. one week's
    good-time intervals attached to a map containing many weeks of photons.
    That is not cosmetic. The GTI is the only record of how much observing time
    a counts map represents, so downstream anything that compares counts to an
    exposure/livetime cube (flux = counts/exposure) silently gets the ratio
    wrong, with no error and no warning.
    """
    rows = list(existing) + list(new_rows)
    rows.sort(key=lambda r: r[0])
    out = []
    for start, stop in rows:
        if out and start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], stop)   # overlapping/adjacent -> extend
        else:
            out.append([start, stop])
    return out


def write_merged_gti(hdul, gti_rows):
    """Replace hdul's GTI extension with `gti_rows` and sync the time keywords.

    Returns total livetime in seconds (or None if there was no GTI to write).
    TSTART/TSTOP are updated too: leaving them describing the first chunk while
    the GTI describes all of them is the same class of bug, one layer down.
    """
    if not gti_rows or 'GTI' not in [h.name for h in hdul]:
        return None
    starts = np.array([r[0] for r in gti_rows], dtype=np.float64)
    stops = np.array([r[1] for r in gti_rows], dtype=np.float64)
    gti_hdu = fits.BinTableHDU.from_columns(
        [fits.Column(name='START', format='D', unit='s', array=starts),
         fits.Column(name='STOP', format='D', unit='s', array=stops)],
        header=hdul['GTI'].header, name='GTI')
    hdul[[h.name for h in hdul].index('GTI')] = gti_hdu

    t0, t1 = float(starts.min()), float(stops.max())
    for hdu in hdul:
        if 'TSTART' in hdu.header:
            hdu.header['TSTART'] = t0
        if 'TSTOP' in hdu.header:
            hdu.header['TSTOP'] = t1
    return float((stops - starts).sum())


def find_map_data(hdul):
    """Locate the map data in a FITS file.

    Returns (ext_index, array):
    - Image maps (e.g. gtbin CCUBE): primary HDU image -> (0, ndarray)
    - HEALPix maps (e.g. gtbin HEALPIX): the map is a binary table extension
      (usually named SKYMAP), one column per energy bin
      -> (ext_index, 2D ndarray of shape (npix, nbins))
    """
    if hdul[0].data is not None:
        return 0, hdul[0].data
    for i, hdu in enumerate(hdul):
        if i == 0 or hdu.data is None or not hasattr(hdu, 'columns'):
            continue
        if hdu.name in ('EBOUNDS', 'GTI', 'ENERGIES'):
            continue
        arr = np.column_stack([hdu.data[name] for name in hdu.data.names])
        return i, arr
    raise ValueError("No image or table map data found in FITS file")


def write_map_data(result_hdul, ext_index, data):
    """Write summed map data back into the HDU it came from (image or table)."""
    if ext_index == 0:
        result_hdul[0].data = data
    else:
        table = result_hdul[ext_index].data
        for j, name in enumerate(table.names):
            table[name] = data[:, j]


def partial_sum_worker(file_slice):
    """Read a slice of FITS map files and return their pixel-wise sum.

    Handles both image maps (primary HDU) and HEALPix maps (binary table).
    Accumulates in float64 to avoid overflow/precision issues.
    """
    total = None
    for f in file_slice:
        with fits.open(f) as hdul:
            _, data = find_map_data(hdul)
            if total is None:
                total = data.astype(np.float64)
            else:
                total += data
    return total


def make_worker_env():
    """Build a subprocess environment with a private PFILES directory.

    Ftools lock and rewrite their .par files in the first (local) PFILES
    directory. With many workers sharing one directory (~/pfiles) they all
    contend on the same file locks, which serializes tool startup and can
    corrupt parameter files. Each pool worker process gets its own local
    directory instead; the system pfiles part (after ';') is inherited
    from the parent PFILES (which seed_system_pfiles points at a local
    mirror, so tool startup never reads .par files over the network).
    """
    env = os.environ.copy()
    worker_name = multiprocessing.current_process().name  # e.g. ForkPoolWorker-3
    pfiles_dir = os.path.join(tempfile.gettempdir(), "fermi_pfiles", worker_name)
    os.makedirs(pfiles_dir, exist_ok=True)

    # A run killed mid-write can leave truncated .par files behind (the dirs
    # are reused across runs); a zero-length par makes every later tool
    # startup in this worker fail with ape's 'Cannot open parameter file'.
    for stale in glob.glob(os.path.join(pfiles_dir, "*.par")):
        try:
            if os.path.getsize(stale) == 0:
                os.remove(stale)
        except OSError:
            pass

    parent_pfiles = env.get("PFILES", "")
    sys_part = parent_pfiles.split(";", 1)[1] if ";" in parent_pfiles else ""
    env["PFILES"] = f"{pfiles_dir};{sys_part}" if sys_part else pfiles_dir
    return env


def seed_system_pfiles(logger):
    """Mirror the system .par files to local disk and point PFILES at the mirror.

    Every gtool startup opens its .par file from the PFILES system directories.
    With the fermitools env on a network filesystem and very many workers
    (e.g. 128 cores x several tools per week), hundreds of near-simultaneous
    opens - plus NFS locking quirks - make ape fail with 'Cannot open
    parameter file' (Ape exception code 6). Serving the system pars from a
    local mirror takes the network out of tool startup entirely.

    Runs once in the parent (workers inherit the rewritten PFILES).
    """
    pfiles = os.environ.get("PFILES", "")
    if ";" not in pfiles:
        return
    local_part, sys_part = pfiles.split(";", 1)
    sys_dirs = [d for d in sys_part.split(":") if d and os.path.isdir(d)]
    if not sys_dirs:
        return

    seed_dir = os.path.join(tempfile.gettempdir(), "fermi_pfiles", "syspfiles_local")
    try:
        os.makedirs(seed_dir, exist_ok=True)
        n_copied = 0
        seen = set()
        for d in sys_dirs:
            for par in sorted(glob.glob(os.path.join(d, "*.par"))):
                name = os.path.basename(par)
                if name in seen:  # first dir wins, matching PFILES search order
                    continue
                seen.add(name)
                dest = os.path.join(seed_dir, name)
                if os.path.exists(dest) and os.path.getmtime(dest) >= os.path.getmtime(par):
                    continue
                # copy to temp + atomic replace so a concurrent pipeline
                # instance never reads a half-written par
                tmp_dest = f"{dest}.tmp{os.getpid()}"
                shutil.copy2(par, tmp_dest)
                os.replace(tmp_dest, dest)
                n_copied += 1
        if not seen:
            return
        os.environ["PFILES"] = f"{local_part};{seed_dir}"
        logger.log(f"System pfiles mirrored locally ({len(seen)} pars, {n_copied} updated) -> {seed_dir}")
    except Exception as e:
        logger.log(f"Could not mirror system pfiles locally ({e}) - keeping {sys_part}")


def preflight_par_check(config, logger):
    """Verify every pipeline tool can find its .par file with the exact PFILES
    the workers will use - BEFORE any copying/slicing/scheduling happens.

    A wrong PFILES system part makes every chunk fail with ape's 'Cannot open
    parameter file' (eFileNotFound) after minutes of setup; this catches it in
    milliseconds with a precise diagnosis instead. Returns True if all good.
    """
    tools = []
    for step in config.get('steps', []):
        cmd = step.get('command', '').strip()
        if cmd:
            tools.append(cmd.split()[0])
    for m in config.get('merging', []):
        if m.get('strategy') == 'hierarchical':
            tools.append(m.get('tool', 'gtltsum'))
        elif m.get('strategy') == 'ftmerge':
            tools.append('ftmerge')
    # De-dup, and skip non-Ftools commands (scripts have no .par files)
    tools = [t for t in dict.fromkeys(tools)
             if t not in ('python3', 'python', 'bash', 'sh')]

    env = make_worker_env()
    pfiles = env.get('PFILES', '')
    local_part, _, sys_part = pfiles.partition(';')
    search_dirs = [d for d in (local_part.split(':') + sys_part.split(':')) if d]

    missing = [t for t in tools
               if not any(os.path.exists(os.path.join(d, t + '.par')) for d in search_dirs)]

    logger.log(f"Worker PFILES: {pfiles}")
    if not missing:
        logger.log(f"Preflight: parameter files found for all tools ({', '.join(tools)}).")
        return True

    logger.error(f"Preflight: no .par file found for: {', '.join(missing)}")
    for d in search_dirs:
        state = 'MISSING DIR' if not os.path.isdir(d) else \
            f"{len(glob.glob(os.path.join(d, '*.par')))} .par files"
        logger.error(f"  PFILES dir {d}: {state}")
    logger.error("The PFILES system part does not reach the fermitools syspfiles. "
                 "Check 'fermi_base' in the config and the PFILES environment variable.")
    return False


# Marker for transient parameter-file failures under heavy startup bursts;
# these are worth retrying before failing the whole chunk.
APE_TRANSIENT_MARKER = "Cannot open parameter file"


def run_gtool_with_retry(cmd, env, max_tries=3):
    """subprocess.run a gtool command, retrying transient ape/pfiles errors.

    Raises subprocess.CalledProcessError after max_tries (or immediately for
    non-transient failures), matching plain subprocess.run(check=True).
    """
    for attempt in range(1, max_tries + 1):
        try:
            return subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True, env=env)
        except subprocess.CalledProcessError as e:
            if APE_TRANSIENT_MARKER not in (e.stderr or "") or attempt == max_tries:
                raise
            print(f"Transient parameter-file error (attempt {attempt}/{max_tries}), "
                  f"retrying: {cmd.split()[0] if cmd.split() else cmd}")
            time.sleep(2 * attempt)


def merge_batch_worker(args):
    """Merge one batch of files pairwise into batch_out using `tool`.

    Runs in a pool worker: batches within a merge stage are independent,
    so multiple batches execute concurrently.

    Returns (batch_out, None) on success or (None, error_message) on failure,
    so one bad batch doesn't abort the whole (possibly hour-long) merge. The
    tool's stderr is captured and returned so failures are diagnosable.
    """
    batch, batch_out, tool = args
    try:
        shutil.copy(batch[0], batch_out)
        if len(batch) == 1:
            return (batch_out, None)

        env = make_worker_env()
        for f in batch[1:]:
            temp_sum = batch_out + ".tmp"
            # clobber=yes so a leftover .tmp (e.g. from a previous aborted run)
            # doesn't make the tool refuse to write.
            cmd = f"{tool} infile1={batch_out} infile2={f} outfile={temp_sum} clobber=yes"
            try:
                run_gtool_with_retry(cmd, env)
            except subprocess.CalledProcessError as e:
                return (None, f"{cmd}\nSTDERR: {(e.stderr or '').strip()}")
            os.replace(temp_sum, batch_out)
        return (batch_out, None)
    except Exception as e:
        return (None, f"{tool} batch merge exception: {e}")


def setup_fermi_environment(logger, fermi_base=None):
    """Set up the Ftools environment (CALDB, FERMI_DIR, PFILES, PATH).

    fermi_base: root of the fermitools conda/micromamba env. Comes from the
    'fermi_base' key in the YAML config. If not set there, it is derived from
    the location of 'gtselect' on PATH (i.e. an already-activated fermi env).

    Env vars that are already set but point to nonexistent paths are treated
    as stale (e.g. exported by an old shell script for a different machine)
    and are overridden - this prevents every gtool from aborting with
    'File not found' when the hardcoded paths of another node leak in.
    """
    gtselect_path = shutil.which("gtselect")

    if not fermi_base and gtselect_path:
        # .../envs/fermi/bin/gtselect -> .../envs/fermi
        fermi_base = os.path.dirname(os.path.dirname(gtselect_path))
        logger.log(f"fermi_base not set in config; derived from gtselect location: {fermi_base}")
    elif fermi_base:
        fermi_base = os.path.expanduser(fermi_base)
        logger.log(f"Using fermi_base from config: {fermi_base}")

    if not fermi_base:
        logger.error("CRITICAL: fermi_base is not set in the config and 'gtselect' is not on PATH.")
        logger.error("Add to your YAML config:  fermi_base: \"/path/to/your/micromamba/envs/fermi\"")
        return

    if not os.path.exists(fermi_base):
        logger.error(f"CRITICAL: fermi_base does not exist: {fermi_base} - check the 'fermi_base' key in your config.")
        return

    env_updates = {
        "CALDB": f"{fermi_base}/share/fermitools/data/caldb",
        "CALDBCONFIG": f"{fermi_base}/share/fermitools/data/caldb/software/tools/caldb.config",
        "CALDBALIAS": f"{fermi_base}/share/fermitools/data/caldb/software/tools/alias_config.fits",
        "CALDBROOT": f"{fermi_base}/share/fermitools/data/caldb",
        "FERMI_DIR": f"{fermi_base}/share/fermitools",
        "FERMI_INST_DIR": f"{fermi_base}/share/fermitools",
        "HEADAS": f"{fermi_base}/heasoft" 
    }

    # Add fermi bins to PATH if missing
    if not gtselect_path or not shutil.which("farith"):
         extra_paths = [
             f"{fermi_base}/bin",
             f"{fermi_base}/heasoft/bin"
         ]
         for p in extra_paths:
             if os.path.exists(p) and p not in os.environ["PATH"]:
                 logger.log(f"Auto-adding {p} to PATH")
                 os.environ["PATH"] = p + os.pathsep + os.environ["PATH"]
         
         # Re-check
         gtselect_path = shutil.which("gtselect")

    # Set missing env vars, and CORRECT stale ones that point to nonexistent
    # paths (e.g. exported by a shell script hardcoded for a different machine)
    for key, val in env_updates.items():
        current = os.environ.get(key)
        if current is None:
            if os.path.exists(val) or key in ("CALDBCONFIG", "CALDBALIAS"):
                logger.log(f"Auto-setting {key}={val}")
                os.environ[key] = val
        elif not os.path.exists(current) and os.path.exists(val):
            logger.log(f"Overriding stale {key} (was {current}, path does not exist) -> {val}")
            os.environ[key] = val

    # Setup PFILES. Rebuild if missing, stale, or SHADOWED: a cluster-wide
    # HEASoft install can provide a PFILES system part whose dirs exist but
    # contain no gt*.par - then every gtool fails with ape's 'Cannot open
    # parameter file' (eFileNotFound). Existing dirs alone prove nothing;
    # gtselect.par is the sentinel that the fermitools pars are reachable.
    # (Interactive shells hide this: ~/pfiles holds learned copies of the
    # pars, but workers get a fresh private local dir and lose ~/pfiles.)
    pfiles_current = os.environ.get("PFILES", "")
    sys_part = pfiles_current.split(";", 1)[1] if ";" in pfiles_current else ""
    sys_dirs = [p for p in sys_part.split(":") if p]
    sys_dirs_exist = any(os.path.exists(p) for p in sys_dirs)
    has_gtool_pars = any(os.path.exists(os.path.join(p, "gtselect.par")) for p in sys_dirs)
    if not pfiles_current or not sys_dirs_exist or not has_gtool_pars:
        pfiles_local = os.path.expanduser("~/pfiles")
        os.makedirs(pfiles_local, exist_ok=True)
        sys_pfiles = f"{fermi_base}/heasoft/syspfiles:{fermi_base}/share/fermitools/syspfiles"
        # Keep any existing dirs (e.g. cluster HEASoft), searched AFTER the
        # fermitools ones so they cannot shadow the gtool pars.
        keep = ":".join(p for p in sys_dirs if os.path.exists(p))
        if keep:
            sys_pfiles = f"{sys_pfiles}:{keep}"
        if pfiles_current:
            logger.log(f"PFILES system part unusable for fermitools (was {pfiles_current}) - rebuilding")
        os.environ["PFILES"] = f"{pfiles_local}:.;{sys_pfiles}"
        logger.log(f"Auto-configured PFILES={os.environ['PFILES']}")


    logger.log(f"Environment Check: gtselect found at: {gtselect_path}")
    if not gtselect_path:
        logger.error("CRITICAL: 'gtselect' not found in PATH. Ftools environment likely not active.")
        logger.error(f"Current PATH: {os.environ.get('PATH')}")

def main():
    # Anchor the process at the repository root: post_processing commands are
    # run with shell=True and no cwd=, and log paths are relative, so without
    # this the pipeline only works when launched from the repo root.
    os.chdir(REPO_ROOT)

    parser = argparse.ArgumentParser(description="Parallel Ftools Runner")
    parser.add_argument("config", help="Path to YAML configuration file")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    args = parser.parse_args()

    logger = Logger("logs/runner.log")
    if args.dry_run:
        logger.log("Running in DRY-RUN mode")

    try:
        config = ConfigLoader.load(args.config, logger)
        setup_fermi_environment(logger, config.get('fermi_base'))
        raise_fd_limit(logger)
        seed_system_pfiles(logger)
        if not preflight_par_check(config, logger):
            sys.exit(1)

        # Run options come from the optional 'run:' section of the YAML config.
        # The only CLI flag is --dry-run (preview mode).
        run_opts = config.get('run', {}) or {}
        config['dry_run'] = args.dry_run
        config['skip_merge'] = run_opts.get('skip_merge', False)
        config['skip_post'] = run_opts.get('skip_post', False)
        config['selected_steps'] = run_opts.get('steps', None)  # None implies all steps

        if config['selected_steps']:
            logger.log(f"Selected steps: {config['selected_steps']}")

        scheduler = JobScheduler(config, logger)
        result = scheduler.run()

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        # Exit non-zero so run_background.sh / CI can tell a crash from a
        # clean finish (single_run.py already did this; this did not).
        sys.exit(1)


if __name__ == "__main__":
    main()



# NOTE on the gtltsum hierarchical merge:
# The batches within each merge stage now run in parallel across cores (see merge_hierarchical /
# merge_batch_worker), which removes most of the old ~7min serial bottleneck. The remaining
# sequential part is the pairwise merging *within* a batch (gtltsum can only combine two files
# at a time) and the final stages where few batches remain.
# DONE: the merge is now streamed during the main processing phase - finished chunk ltcubes are
# merged in batches of 8 by a side pool while processing continues (resources.stream_merge,
# default true). See JobScheduler.run(); batches are fixed by task id so output stays reproducible.
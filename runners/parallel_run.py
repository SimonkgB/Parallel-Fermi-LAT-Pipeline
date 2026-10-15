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

# Run inside the activated fermitools env (micromamba activate fermi), which
# provides the Fermi tools, CALDB/PFILES, omegaconf, astropy and numpy.
import numpy as np
from astropy.io import fits
from omegaconf import OmegaConf

# Root of the repository (this file lives in <repo>/runners/). Used both by the
# '${root:...}' config resolver and by the os.chdir() in the entry points, so
# that nothing depends on the directory the pipeline was launched from.

REPO_ROOT = Path(__file__).resolve().parent.parent


def raise_fd_limit(logger):
    """Raise the open-file soft limit to the hard limit.

    The pipeline legitimately wants many concurrent fds (hundreds of workers,
    each running an Ftool that opens several large FITS files, plus the merge
    pool). Distributions ship a conservative soft limit but a very
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
        # Every error() is counted: main() exits non-zero if any happened, so a
        # run that logged an ERROR can never look like a clean finish.
        self.n_errors = 0
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
        self.n_errors += 1
        self.log(f"ERROR: {message}")



# Top-level keys the runner understands. 'science_models' is never read - it is
# documentation for a likelihood stage that does not exist yet - but it is a
# legitimate part of the config, so it must not be flagged as a typo.
KNOWN_TOP_LEVEL_KEYS = {
    'vars', 'resources', 'input', 'spacecraft', 'common_files',
    'steps', 'merging', 'post_processing', 'science_models', 'run',
}

KNOWN_RESOURCE_KEYS = {
    'cores', 'ram_disk', 'ltcube_cache', 'slice_scfile',
}


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


WEEK_RE = re.compile(r"_w(\d+)_")


def parse_week_range(weeks):
    """'9-947' -> (9, 947); '' or None -> (None, None), meaning every week found."""
    if weeks is None:
        return None, None
    weeks = str(weeks).strip()
    if weeks == "":
        return None, None

    lo_text, _, hi_text = weeks.partition("-")
    lo = None
    hi = None
    if lo_text.strip() != "":
        lo = int(lo_text)
    if hi_text.strip() != "":
        hi = int(hi_text)
    return lo, hi


def week_number(path):
    """Week number from a weekly file name (..._w0123_...), or None."""
    match = WEEK_RE.search(Path(path).name)
    if match is None:
        return None
    return int(match.group(1))


def select_input_files(directory, pattern, weeks=""):
    """Input files matching `pattern`, restricted to the inclusive week range.

    This is the single source of truth for which weeks a run uses: both
    discover_files() and the '${week_tag:...}' name resolver call it, so the
    product file names can never disagree with the data that went into them.
    Files whose name carries no week number are kept only when no range is given.
    """
    lo, hi = parse_week_range(weeks)
    selected = []
    for path in sorted(Path(directory).glob(pattern)):
        week = week_number(path)
        if week is None:
            if lo is None and hi is None:
                selected.append(path)
            continue
        if lo is not None and week < lo:
            continue
        if hi is not None and week > hi:
            continue
        selected.append(path)
    return selected


def same_size_and_mtime(source, copy):
    """True if `copy` exists and has the size and mtime of `source`."""
    if not os.path.exists(copy):
        return False
    source_stat = os.stat(source)
    copy_stat = os.stat(copy)
    if source_stat.st_size != copy_stat.st_size:
        return False
    if int(source_stat.st_mtime) != int(copy_stat.st_mtime):
        return False
    return True


def energy_label(mev):
    """100 -> '100MeV', 316228 -> '316GeV'."""
    mev = float(mev)
    if mev >= 1000:
        return f"{mev / 1000:.0f}GeV"
    return f"{mev:.0f}MeV"


def resolve_root(*parts):
    """${root:data} -> <repo>/data, so config paths never depend on the cwd."""
    return os.path.normpath(str(REPO_ROOT.joinpath(*parts)))


def resolve_energy_tag(emin, emax, enumbins):
    """100, 316228, 28 -> '100MeV-316GeV-8pd' (bins per decade)."""
    import math
    bins_per_decade = float(enumbins) / math.log10(float(emax) / float(emin))
    return f"{energy_label(emin)}-{energy_label(emax)}-{bins_per_decade:.0f}pd"


def resolve_selection_tag(zmax):
    """90 -> 'zmax90'."""
    return f"zmax{float(zmax):g}"


def resolve_grid_tag(hpx_order, coordsys=""):
    """7 -> 'hpx128' (nside = 2^order)."""
    return f"hpx{2 ** int(hpx_order)}"


def resolve_week_tag(directory, pattern, weeks=""):
    """'w009-w947': the first and last week the run actually selects."""
    numbers = []
    for path in select_input_files(directory, pattern, weeks):
        week = week_number(path)
        if week is not None:
            numbers.append(week)
    if not numbers:
        return "wNONE"
    numbers.sort()
    return f"w{numbers[0]:03d}-w{numbers[-1]:03d}"


# Resolvers the configs use ('${root:...}', '${energy_tag:...}', ...). The four
# *_tag resolvers used to be referenced by the config but never registered, so
# loading it failed with "Unsupported interpolation type energy_tag".
CONFIG_RESOLVERS = {
    "root": resolve_root,
    "energy_tag": resolve_energy_tag,
    "selection_tag": resolve_selection_tag,
    "grid_tag": resolve_grid_tag,
    "week_tag": resolve_week_tag,
}


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
        for name, resolver in CONFIG_RESOLVERS.items():
            if not OmegaConf.has_resolver(name):
                OmegaConf.register_new_resolver(name, resolver)
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

    def keep_files(self):
        """True if the config asks to keep intermediate files.

        The configs call the key 'keep'; older code read 'keep_intermediate'.
        Both are honoured.
        """
        if self.config.get('keep', False):
            return True
        if self.config.get('keep_intermediate', False):
            return True
        return False

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
            if self.keep_files():
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
        # Finds all files matching the pattern in the input directory, restricted
        # to the optional input.weeks range (e.g. "9-947"). Use the range to leave
        # out the current week: its weekly file is still growing.
        weeks = self.config['input'].get('weeks', '')
        files = select_input_files(self.input_dir, self.pattern, weeks)
        if not files:
            message = f"No files found matching {self.pattern} in {self.input_dir}"
            if weeks:
                message += f" for weeks {weeks}"
            raise FileNotFoundError(message)
        if weeks:
            self.logger.log(f"Found {len(files)} input files (weeks {weeks}).")
        else:
            self.logger.log(f"Found {len(files)} input files.")
        return files

    def prepare_spacecraft(self, photon_files):
        """Merge the weekly spacecraft (FT2) files of the selected weeks.

        With a 'spacecraft:' section in the config, the run takes weekly FT2
        files exactly like the weekly photon files, and builds the merged file
        the tools need itself: lat_spacecraft_merged_w009-w947.fits in
        spacecraft.merged_dir, holding exactly the weeks of the photon files.
        The merged file records a fingerprint of its inputs (SC_HASH); a later
        run reuses it only if the fingerprint matches, otherwise rebuilds it.
        The result becomes common_files.scfile for the rest of the run.

        Without a 'spacecraft:' section, common_files.scfile is used as given.
        """
        sc_cfg = self.config.get('spacecraft')
        if not sc_cfg:
            return
        dry_run = self.config.get('dry_run', False)

        # Weekly spacecraft file of every week, by week number
        sc_by_week = {}
        for path in Path(sc_cfg['directory']).glob(sc_cfg.get('pattern', 'lat_spacecraft_weekly_w*.fits')):
            week = week_number(path)
            if week is None:
                continue
            if week in sc_by_week:
                raise RuntimeError(f"Two spacecraft files for week {week}: {sc_by_week[week]} and {path}")
            sc_by_week[week] = path

        # The same weeks as the photon files - every one must have its FT2 file
        week_file_pairs = []
        missing = []
        for photon_file in photon_files:
            week = week_number(photon_file)
            if week is None:
                raise RuntimeError(f"Cannot read a week number from {photon_file}; "
                                   f"weekly spacecraft matching needs weekly photon files")
            if week not in sc_by_week:
                missing.append(week)
                continue
            week_file_pairs.append((week, sc_by_week[week]))
        if missing:
            raise RuntimeError(f"No spacecraft file in {sc_cfg['directory']} for week(s) {missing}")

        # Sort by week so the merged rows are in time order
        week_file_pairs.sort()
        weeks_sorted = []
        files_sorted = []
        for week, path in week_file_pairs:
            weeks_sorted.append(week)
            files_sorted.append(path)

        merged_dir = sc_cfg.get('merged_dir') or self.config['vars']['data_dir']
        merged_name = f"lat_spacecraft_merged_w{weeks_sorted[0]:03d}-w{weeks_sorted[-1]:03d}.fits"
        merged_path = os.path.join(merged_dir, merged_name)
        fingerprint = spacecraft_fingerprint(files_sorted)

        if os.path.exists(merged_path) and read_sc_hash(merged_path) == fingerprint:
            self.logger.log(f"Spacecraft: reusing {merged_path} ({len(files_sorted)} weekly files, inputs unchanged)")
        elif dry_run:
            self.logger.log(f"[DryRun] Would merge {len(files_sorted)} weekly spacecraft files -> {merged_path}")
        else:
            if os.path.exists(merged_path):
                self.logger.log(f"Spacecraft: {merged_path} was built from other inputs - rebuilding")
            self.logger.log(f"Spacecraft: merging {len(files_sorted)} weekly files -> {merged_path}")
            t0 = time.time()
            os.makedirs(merged_dir, exist_ok=True)
            n_rows, n_dropped = merge_spacecraft_files(files_sorted, merged_path, fingerprint,
                                                       weeks_sorted[0], weeks_sorted[-1])
            self.logger.log(f"Spacecraft: {n_rows} rows written, {n_dropped} overlapping rows "
                            f"dropped at week boundaries, {time.time() - t0:.0f}s")

        if self.config.get('common_files') is None:
            self.config['common_files'] = {}
        if self.config['common_files'].get('scfile'):
            self.logger.log("WARNING: config: common_files.scfile is replaced by the merged "
                            "spacecraft file (the 'spacecraft:' section wins)")
        self.config['common_files']['scfile'] = merged_path

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
                    # Copy unless an IDENTICAL copy is already there. A file of the
                    # same name left by an earlier run may be an older spacecraft
                    # file; reusing it would silently cut the GTIs and livetime.
                    # copy2 keeps the mtime, so size + mtime identify the source.
                    if not dry_run:
                        if same_size_and_mtime(path, dest):
                            self.logger.log(f"  {filename}: reusing identical RAM-disk copy")
                        else:
                            if os.path.exists(dest):
                                self.logger.log(f"  {filename}: RAM-disk copy differs from the source - recopying")
                            shutil.copy2(path, dest)
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
        if not sc_src:
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
            with multiprocessing.Pool(n_workers) as pool:
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
            with multiprocessing.Pool(n_workers) as pool:
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
        self.prepare_spacecraft(files)
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
        if not self.config.get('dry_run', False):
            for m in self.config.get('merging', []):
                if m.get('strategy') == 'image_sum':
                    image_sum_keys.add(m['input_pattern'].strip("{}"))

        # Accumulator processes: summing a chunk into the running image_sum total
        # costs seconds for big cubes, and doing it in the main process would
        # serialize it - with many cores that becomes the throughput ceiling.
        # A few dedicated processes drain a queue of finished chunk paths instead
        # (chunks arrive slower than accumulators consume, so files don't pile up
        # on the RAM disk).
        accum_queue = None
        accum_procs = []
        if image_sum_keys:
            n_acc = max(1, min(8, self.cores))
            accum_queue = multiprocessing.Queue()
            for wid in range(n_acc):
                # daemon: if the run aborts, they must not keep the process alive
                # waiting on the queue
                p = multiprocessing.Process(target=accumulate_image_sum_worker,
                                            args=(accum_queue, self.working_dir, wid),
                                            daemon=True)
                p.start()
                accum_procs.append(p)
            self.logger.log(f"Started {n_acc} image_sum accumulator processes.")

        self.logger.log(f"Starting parallel execution on {self.cores} cores...")
        t_phase = time.time()
        results = []
        total_tasks = len(tasks)
        completed = 0
        contributors = {}     # output key -> set of chunk ids that produced it
        skipped_weeks = []    # weeks the tools reported as empty

        with multiprocessing.Pool(self.cores) as pool:
            # Use imap_unordered to track progress as tasks finish
            for result in pool.imap_unordered(execute_worker, tasks):
                if result.get('success'):
                    # Record which chunks produced which outputs BEFORE the
                    # image_sum outputs are handed off and cleared below; the
                    # merge phase checks that every merged product (counts,
                    # livetime) is built from the same set of weeks.
                    outputs = result.get('outputs', {})
                    for key, paths in outputs.items():
                        if paths:
                            contributors.setdefault(key, set()).add(result['id'])
                    for name in result.get('skipped', []):
                        skipped_weeks.append(name)

                    # Hand image_sum chunk outputs to the accumulators, which
                    # sum and delete them, and mark them consumed so the merge
                    # phase does not look for the deleted files.
                    for key in image_sum_keys:
                        for fp in outputs.get(key, []):
                            accum_queue.put((key, fp))
                        if key in outputs:
                            outputs[key] = []

                results.append(result)
                completed += 1
                percent = (completed / total_tasks) * 100
                self.logger.log(f"Progress: {completed}/{total_tasks} files ({percent:.1f}%)")
        phase_times['parallel processing'] = time.time() - t_phase

        if accum_procs:
            t_phase = time.time()
            self._drain_accumulators(accum_queue, accum_procs, image_sum_keys)
            phase_times['accumulator drain'] = time.time() - t_phase

        results.sort(key=lambda x: x['id'])

        hits = sum(r.get('ltcube_cache', [0, 0])[0] for r in results if r.get('success'))
        stored = sum(r.get('ltcube_cache', [0, 0])[1] for r in results if r.get('success'))
        if hits or stored:
            self.logger.log(f"Ltcube cache: {hits} hits, {stored} newly stored.")

        # Weeks the tools reported as having no usable data (e.g. LAT safe-hold
        # weeks). Legitimate, but they must be visible in runner.log, not only
        # in the workers' stdout.
        if skipped_weeks:
            skipped_weeks.sort()
            self.logger.log(f"{len(skipped_weeks)} week(s) skipped as empty (no good-quality time): "
                            + ", ".join(skipped_weeks))

        # Any failed chunk aborts the run before merging. A merged map that is
        # missing some weeks is not a smaller valid product: the counts and the
        # livetime/exposure would no longer describe the same data.
        failed = []
        for res in results:
            if not res['success']:
                failed.append(res)
        if failed:
            self.logger.error(f"{len(failed)} of {len(tasks)} chunks failed. Aborting before the merge phase.")
            for res in failed:
                self.logger.log(f"Chunk {res['id']} error: {res['error']}")
            raise RuntimeError(f"{len(failed)} chunks failed")

        # Every merged product must come from the same chunks (weeks).
        self._contributors = contributors
        if not self.config.get('selected_steps') and not self.config.get('dry_run', False):
            self.check_same_contributors(contributors)

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

    def check_same_contributors(self, contributors):
        """Raise if the merged products would come from different weeks.

        Each merge step names its chunk outputs (e.g. chunk_hpxcube for the
        counts, chunk_ltcube for the livetime). If a week produced one but not
        the other, the counts and the exposure would cover different time
        and every flux derived from them would be silently biased.
        """
        keys = []
        for merge_step in self.config.get('merging', []):
            keys.append(merge_step['input_pattern'].strip("{}"))
        if len(keys) < 2:
            return

        reference_key = keys[0]
        reference = contributors.get(reference_key, set())
        for key in keys[1:]:
            other = contributors.get(key, set())
            if other == reference:
                continue
            only_reference = sorted(reference - other)
            only_other = sorted(other - reference)
            self.logger.error(f"Merged products would cover different weeks: "
                              f"{len(reference)} chunks made {reference_key}, {len(other)} made {key}; "
                              f"only {reference_key}: {only_reference[:10]}, only {key}: {only_other[:10]}")
            raise RuntimeError(f"{reference_key} and {key} come from different chunks")
        self.logger.log(f"All {len(keys)} merged products come from the same {len(reference)} chunks.")

    def cleanup(self):
        """Remove the RAM disk / scratch dir (also called when a run aborts,
        so a failed run does not leave hundreds of GB in /dev/shm)."""
        self.ram_disk.cleanup()
        self.cleanup_workdir()

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
        # Respect the same keep flag as the RAM disk
        if self.ram_disk.keep_files():
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
                              f"image_sum totals are incomplete.")
            raise RuntimeError("image_sum accumulator died")

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
        # An accumulator that failed on a file only printed it and moved on, so
        # the total could be missing weeks. Compare with the chunks that made one.
        expected = len(getattr(self, '_contributors', {}).get(key, set()))
        if expected and acc['count'] != expected:
            self.logger.error(f"{key}: summed {acc['count']} chunk maps but {expected} chunks produced one")
            raise RuntimeError(f"image_sum of {key} is missing chunks")
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
        (e.g. chunk_hpxcube, chunk_ltcube), and merges each group using the
        strategy specified in the YAML config:
          image_sum     maps summed by the accumulator processes during the run
                        (counts); only the total is written here
          hierarchical  pairwise tree merge with a Fermi tool (livetime cubes, gtltsum)
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

            if strategy not in ('image_sum', 'hierarchical'):
                self.logger.error(f"Unknown merge strategy '{strategy}' in merge step {merge_step['name']} "
                                  f"(use image_sum or hierarchical)")
                raise RuntimeError(f"Unknown merge strategy: {strategy}")

            # image_sum chunks were accumulated during processing (files already
            # summed and deleted) - just write the running total.
            if strategy == 'image_sum' and not self.config.get('dry_run', False):
                if pattern_key in self._img_accum:
                    self.write_accumulated_image_sum(pattern_key, output_file)
                elif self.config.get('selected_steps'):
                    self.logger.log(f"Skipping merge step {merge_step['name']}: nothing accumulated "
                                    f"(expected during partial run).")
                else:
                    self.logger.error(f"Nothing was accumulated for merge step {merge_step['name']} "
                                      f"(looking for {pattern_key})")
                continue

            input_files = files_by_type.get(pattern_key, [])

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

            # Only 'hierarchical' gets here: image_sum was written above
            self.merge_hierarchical(input_files, output_file, tool=merge_step.get('tool', 'gtltsum'))

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
            self.logger.log(f"Merge stage {stage}: {len(current_files)} files -> {len(batches)} batches ({n_workers} in parallel)")

            if n_workers > 1:
                with multiprocessing.Pool(n_workers) as pool:
                    outcomes = pool.map(merge_batch_worker, batch_tasks)
            else:
                outcomes = [merge_batch_worker(t) for t in batch_tasks]

            # A failed batch aborts the merge. Skipping it (the old behaviour)
            # dropped up to 8 weeks from the livetime cube while the counts map
            # kept them, which biases the exposure low with only a log line.
            next_files = []
            n_failed = 0
            for out_path, err in outcomes:
                if err:
                    self.logger.error(f"Merge batch failed in stage {stage}: {err}")
                    n_failed += 1
                elif out_path:
                    next_files.append(out_path)
            if n_failed > 0:
                raise RuntimeError(f"{n_failed} merge batch(es) failed in stage {stage} for {output_file}")

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
                # Later steps use this step's output (flux needs the exposure),
                # so stop here instead of running them on a missing file.
                self.logger.error(f"Post-processing step {step['name']} failed: {e}")
                raise


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
    skipped = []          # weeks the tools reported as empty (logged by the parent)

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
                                    skipped.append(file_context['basename'])
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
                'timings': timings, 'ltcube_cache': cache_stats, 'skipped': skipped}

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


SC_HASH_KEY = "SC_HASH"


def spacecraft_fingerprint(paths):
    """Hash of the weekly FT2 inputs (name, size, mtime of each, in order).

    Stored in the merged file's header: a merged file is reused only when it
    was built from exactly these files, e.g. not after a weekly file was
    re-downloaded (the current week is reprocessed daily by the FSSC).
    """
    digest = hashlib.sha256()
    for path in paths:
        stat = os.stat(path)
        digest.update(f"{os.path.basename(path)}:{stat.st_size}:{int(stat.st_mtime)}\n".encode())
    return digest.hexdigest()[:32]


def read_sc_hash(path):
    """SC_HASH keyword of a merged spacecraft file, or None."""
    try:
        with fits.open(path) as hdul:
            return hdul[0].header.get(SC_HASH_KEY)
    except Exception:
        return None


def merge_spacecraft_files(paths, out_path, fingerprint, week_lo, week_hi):
    """Concatenate weekly FT2 files (in time order) into one merged FT2 file.

    Rows are taken in file order. A row that starts before the previous kept
    row stops (the same 30 s interval written in two adjacent weekly files)
    is dropped, so the merged table is strictly time ordered and never counts
    an interval twice - double rows would double that interval's livetime.

    Written to a temporary file and renamed, so a crash never leaves a
    half-written merged file that a later run could pick up.
    Returns (rows written, overlapping rows dropped).
    """
    pieces = []
    n_dropped = 0
    last_stop = None
    columns = None
    primary_header = None
    sc_header = None

    for path in paths:
        with fits.open(path, memmap=False) as hdul:
            sc_hdu = hdul['SC_DATA']
            if columns is None:
                columns = sc_hdu.columns
                primary_header = hdul[0].header.copy()
                sc_header = sc_hdu.header.copy()
            data = np.array(sc_hdu.data)

        start = data['START']
        if len(start) > 1 and np.any(np.diff(start) < 0):
            raise RuntimeError(f"{path}: SC_DATA rows are not in time order")

        if last_stop is None:
            keep = np.ones(len(start), dtype=bool)
        else:
            keep = start >= last_stop - 1e-3
        n_dropped += int(len(start) - keep.sum())
        data = data[keep]
        if len(data) == 0:
            continue
        pieces.append(data)
        last_stop = float(data['STOP'][-1])

    if not pieces:
        raise RuntimeError("No spacecraft rows to merge")
    merged = np.concatenate(pieces)

    # Keep the original column definitions (formats, units) and fill them
    sc_out = fits.BinTableHDU.from_columns(columns, header=sc_header, nrows=len(merged), name='SC_DATA')
    for name in merged.dtype.names:
        sc_out.data[name] = merged[name]

    tstart = float(merged['START'][0])
    tstop = float(merged['STOP'][-1])
    primary = fits.PrimaryHDU(header=primary_header)
    for header in (primary.header, sc_out.header):
        header['TSTART'] = tstart
        header['TSTOP'] = tstop
    primary.header[SC_HASH_KEY] = (fingerprint, 'hash of the weekly FT2 inputs')
    primary.header['WEEKMIN'] = (week_lo, 'first mission week merged')
    primary.header['WEEKMAX'] = (week_hi, 'last mission week merged')
    primary.header['NWEEKS'] = (len(paths), 'number of weekly FT2 files merged')

    tmp_path = f"{out_path}.tmp{os.getpid()}"
    fits.HDUList([primary, sc_out]).writeto(tmp_path, overwrite=True)
    os.replace(tmp_path, out_path)
    return len(merged), n_dropped


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


def check_fermi_environment(logger):
    """Stop early with a clear message if the fermitools env is not active.

    Otherwise every week would fail separately with a less obvious error.
    """
    gtselect_path = shutil.which("gtselect")
    if gtselect_path is None:
        logger.error("'gtselect' is not on PATH - activate the fermitools env first "
                     "(micromamba activate fermi)")
        return False
    logger.log(f"Fermi tools: {os.path.dirname(gtselect_path)}")
    return True


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

    scheduler = None
    try:
        config = ConfigLoader.load(args.config, logger)
        if not check_fermi_environment(logger):
            sys.exit(1)
        raise_fd_limit(logger)
        seed_system_pfiles(logger)

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
        # An aborted run must not leave its RAM disk (up to hundreds of GB of
        # /dev/shm) behind.
        if scheduler is not None:
            try:
                scheduler.cleanup()
            except Exception as cleanup_error:
                logger.log(f"Cleanup after the failure also failed: {cleanup_error}")
        # Exit non-zero so run_background.sh / CI can tell a crash from a
        # clean finish (single_run.py already did this; this did not).
        sys.exit(1)

    # A run that logged any ERROR did not produce trustworthy outputs, even if
    # it reached the end.
    if logger.n_errors > 0:
        logger.log(f"Finished with {logger.n_errors} error(s) - see logs/runner.log")
        sys.exit(1)


if __name__ == "__main__":
    main()


# NOTE on the gtltsum hierarchical merge:
# The batches within each merge stage run in parallel across cores (see merge_hierarchical /
# merge_batch_worker). The remaining sequential part is the pairwise merging *within* a batch
# (gtltsum can only combine two files at a time) and the final stages where few batches remain.
# Inputs are in chunk-id order, so batch composition and summation order (float32 livetime
# cubes) are reproducible.
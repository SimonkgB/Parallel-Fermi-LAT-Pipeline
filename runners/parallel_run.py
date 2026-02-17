#!/usr/bin/env python3
import os
import sys
import yaml
import glob
import shutil
import time
import subprocess
import multiprocessing
import argparse
from pathlib import Path
from datetime import datetime
import contextlib

# Import astropy only if needed for merging
try:
    from astropy.io import fits
    import numpy as np
    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False

class Logger:
    def __init__(self, log_file=None):
        """
        log_file: Path to log file [str]
        start_time: Start time [float]
        """
        self.log_file = log_file
        self.start_time = time.time()
        
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



class ConfigLoader:
    @staticmethod
    def load(path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)

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
        use_ram = self.ram_disk.setup()
        
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
                    if not os.path.exists(dest):
                         shutil.copy(path, dest) # Standard copy for single large file
                    common_file_context[key] = dest
                self.logger.log("Common files cached.")

            self.logger.log(f"Copying {len(files)} files to RAM disk using parallel rsync ({self.cores} workers)...")
            
            copy_tasks = []
            for f in files:
                dest = os.path.join(ram_input_dir, f.name)
                working_files.append(dest)
                # Only copy if destination doesn't exist or we want to force update (rsync handles this efficiently)
                copy_tasks.append((str(f), dest))
            
            # Run parallel copy
            with multiprocessing.Pool(self.cores) as pool:
                # The function copy_worker_rsync is defined below, this function copies the list of files paths to the RAM disk
                pool.map(copy_worker_rsync, copy_tasks)
                
            self.logger.log("RAM disk copy complete.")
            self.working_dir = self.ram_disk.path
        else:
            working_files = [str(f) for f in files]
            self.working_dir = os.getcwd() # Or a temp dir
            
            # For non-RAM disk, common files just point to their original location
            common_files = self.config.get('common_files', {})
            common_file_context = common_files.copy()
            
        return working_files, common_file_context

    def run(self):
        """Run the parallel Fermi pipeline.
        
        Orchestrates the entire pipeline: discovers files, sets up the environment,
        distributes work across cores, executes the pipeline steps, merges results,
        and cleans up.
        """
        files = self.discover_files()
        working_files, common_context = self.prepare_environment(files)
        
        # Distribute files evenly across available cores
        chunks = []
        num_files = len(working_files)
        chunk_size = (num_files + self.cores - 1) // self.cores
        
        for i in range(0, num_files, chunk_size):
            chunks.append(working_files[i:i + chunk_size])
            
        self.logger.log(f"Split {num_files} files into {len(chunks)} chunks for {self.cores} workers.")
        
        # Prepare Tasks
        tasks = []
        selected_steps = self.config.get('selected_steps', None)
        
        for i, chunk in enumerate(chunks):
            task = {
                'id': i,
                'files': chunk,
                'steps': self.config['steps'],
                'working_dir': self.working_dir,
                'dry_run': self.config.get('dry_run', False),
                'selected_steps': selected_steps,
                'common_context': common_context
            }
            tasks.append(task)


        self.logger.log(f"Starting parallel execution on {self.cores} cores...")
        results = []
        total_chunks = len(tasks)
        completed = 0
        
        with multiprocessing.Pool(self.cores) as pool:
            # Use imap_unordered to track progress as tasks finish
            for result in pool.imap_unordered(execute_worker, tasks):
                results.append(result)
                completed += 1
                percent = (completed / total_chunks) * 100
                self.logger.log(f"Progress: {completed}/{total_chunks} chunks ({percent:.1f}%) - Last: Chunk {result['id']}")

        results.sort(key=lambda x: x['id'])

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
                sys.exit(1)

        # Merge
        if not self.config.get('skip_merge', False):
            self.merge_results(results)
        else:
            self.logger.log("Skipping merge phase.")

        # Post-Processing
        if not self.config.get('skip_post', False):
            self.run_post_processing()
        else:
            self.logger.log("Skipping post-processing phase.")

        # Cleanup
        self.ram_disk.cleanup()
        self.logger.log("Pipeline complete!")

        # Performance Breakdown
        self.print_performance_stats(results, len(working_files))

    def print_performance_stats(self, results, total_files):
        # Aggregate
        step_times = {}
        for res in results:
            if not res['success']: continue
            worker_timings = res.get('timings', {})
            for step, duration in worker_timings.items():
                step_times[step] = step_times.get(step, 0.0) + duration
        
        if not step_times:
            return

        print("\n" + "="*40)
        print("PERFORMANCE BREAKDOWN (Cumulative CPU Time)")
        print("="*40)
        print(f"{'Step':<20} | {'Total Time (s)':<15} | {'Avg Time/File (s)':<15}")
        print("-" * 56)
        
        for step, total_duration in step_times.items():
            avg = total_duration / total_files if total_files > 0 else 0
            print(f"{step:<20} | {total_duration:<15.2f} | {avg:<15.4f}")
        print("-" * 56)
        print("(Note: Total Time is sum across all cores. Divide by core count for wall time estimate.)\n")

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
            
            base_data = result_hdul[0].data.copy()
            for f in files[1:]:
                with fits.open(f) as hdul:
                    base_data += hdul[0].data
            
            result_hdul[0].data = base_data
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
        """
        current_files = files
        stage = 0
        temp_dir = os.path.dirname(output_file)
        
        while len(current_files) > 1:
            stage += 1
            next_files = []
            batches = [current_files[i:i+8] for i in range(0, len(current_files), 8)]
            
            for i, batch in enumerate(batches):
                batch_out = os.path.join(temp_dir, f"temp_merge_stage{stage}_batch{i}.fits")
                if len(batch) == 1:
                    shutil.copy(batch[0], batch_out)
                    next_files.append(batch_out)
                    continue
                shutil.copy(batch[0], batch_out)
                for f in batch[1:]:
                    temp_sum = batch_out + ".tmp"
                    cmd = f"{tool} infile1={batch_out} infile2={f} outfile={temp_sum}"
                    subprocess.run(cmd, shell=True, check=True)
                    os.rename(temp_sum, batch_out)
                next_files.append(batch_out)
            if stage > 1:
                for f in current_files:
                    os.remove(f)
            current_files = next_files

        if current_files:
            os.rename(current_files[0], output_file)
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
    
    chunk_outputs = {} 
    timings = {} 
    
    try:
        worker_id = chunk_id
        common_context = task.get('common_context', {})
        
        for file_idx, input_file in enumerate(files):
            file_context = {
                'input': input_file,
                'worker_id': worker_id,
                'file_idx': file_idx,
                'chunk_id': chunk_id,
                'basename': os.path.basename(input_file).replace('.fits', '')
            }
            # Merge common files (e.g. {scfile}) into context
            file_context.update(common_context)
            
            for step in steps:
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
                        # Only print for the first file in the chunk to avoid log spam
                        if file_idx == 0:
                            print(f"[DryRun] Worker {chunk_id}: {cmd}")
                        # Register "fake" outputs for dry run
                        for key, path in step_paths:
                            if key not in chunk_outputs:
                                chunk_outputs[key] = []
                            chunk_outputs[key].append(path)
                    else:
                        try:
                            # Capture output to help debugging if it fails
                            proc = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                            
                            # If success, register outputs
                            for key, path in step_paths:
                                if key not in chunk_outputs:
                                    chunk_outputs[key] = []
                                chunk_outputs[key].append(path)

                        except subprocess.CalledProcessError as e:
                            # Check for benign errors like "No GTIs" which just means empty file
                            if "No GTIs found" in e.stderr:
                                print(f"Worker {chunk_id}: Skipping empty file {file_context['basename']} (No GTIs)")
                                # Do NOT register outputs, continue to next file/step
                                continue
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

        return {'id': chunk_id, 'success': True, 'outputs': chunk_outputs, 'timings': timings}

    except Exception as e:
        return {'id': chunk_id, 'success': False, 'error': str(e)}


def copy_worker_rsync(args):
    src, dest = args
    try:
        # -a: archive mode, --update: skip if newer exists
        subprocess.run(["rsync", "-a", "--update", src, dest], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error copying {src}: {e}")

def setup_fermi_environment(logger):
    # Ensure Fermi environment variables are set
    gtselect_path = shutil.which("gtselect")
    
    # HARDCODED: Change this if running from a different environment
    # Should point to the root of your fermitools conda/micromamba env
    fermi_base = "/net/hume.uio.no/uio/hume/student-u42/skgberg/micromamba/envs/fermi"
    
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

    # Set missing env vars
    for key, val in env_updates.items():
        if key not in os.environ:
             if os.path.exists(val) or key in ("CALDBCONFIG", "CALDBALIAS"):
                 logger.log(f"Auto-setting {key}={val}")
                 os.environ[key] = val

    # Setup PFILES
    if "PFILES" not in os.environ:
        pfiles_local = os.path.expanduser("~/pfiles")
        os.makedirs(pfiles_local, exist_ok=True)
        sys_pfiles = f"{fermi_base}/heasoft/syspfiles:{fermi_base}/share/fermitools/syspfiles"
        os.environ["PFILES"] = f"{pfiles_local}:.;{sys_pfiles}"
        logger.log(f"Auto-configured PFILES={os.environ['PFILES']}")


    logger.log(f"Environment Check: gtselect found at: {gtselect_path}")
    if not gtselect_path:
        logger.error("CRITICAL: 'gtselect' not found in PATH. Ftools environment likely not active.")
        logger.error(f"Current PATH: {os.environ.get('PATH')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel Ftools Runner")
    parser.add_argument("config", help="Path to YAML configuration file")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--steps", help="Comma-separated list of step names to run (e.g. gtselect,gtmktime)")
    parser.add_argument("--skip-merge", action="store_true", help="Skip the merging phase")
    parser.add_argument("--skip-post", action="store_true", help="Skip the post-processing phase")
    parser.add_argument("--keep-intermediate", action="store_true", help="Do not delete RAM disk / intermediate files after run")
    args = parser.parse_args()

    logger = Logger("runner.log") 
    if args.dry_run:
        logger.log("Running in DRY-RUN mode")

    setup_fermi_environment(logger)

    try:
        config = ConfigLoader.load(args.config)
        #import pdb; pdb.set_trace()  # <-- DEBUGGER: will pause here
        
        # Inject CLI args into config
        config['dry_run'] = args.dry_run
        config['skip_merge'] = args.skip_merge
        config['skip_post'] = args.skip_post
        config['keep_intermediate'] = args.keep_intermediate
        
        if args.steps:
            config['selected_steps'] = [s.strip() for s in args.steps.split(',')]
            logger.log(f"Selected steps: {config['selected_steps']}")
        else:
            config['selected_steps'] = None # Implies all
        
        scheduler = JobScheduler(config, logger)
        result = scheduler.run()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()



# TODO:
# In this current iteration gtltsum we do a hierarchical merge, which is not perfectly parallelized.
# Thus this is the biggest bottleneck in the pipeline, taking ~7min for teh 36files on a 36core machine.
# Initially it is not worth to fix, but if I were to use this software for a much larger dataset and more cores AND i use the software alot
# I would fix this by simply by using that we know the Avg Time / file = 41.42 s w could create a linear incremental number of fits file
# given to each core, then from when the first core with the least files have finished its job, set that core to work on gtltsum, and
# start using it to merge the completed sets from the other core. Estimated 14% performance increase for this explicit setup
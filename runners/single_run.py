#!/usr/bin/env python3
import sys
import os
import argparse
import time

# parallel_run.py lives in the same directory, so import it directly
from parallel_run import (JobScheduler, ConfigLoader, Logger, setup_fermi_environment,
                          seed_system_pfiles, preflight_par_check, raise_fd_limit,
                          REPO_ROOT)

def main():
    # See parallel_run.main(): config paths and logs are repo-root relative.
    os.chdir(REPO_ROOT)

    parser = argparse.ArgumentParser(description="Single-Core Fermi Runner (Baseline)")
    parser.add_argument("config", help="Path to YAML configuration file")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")

    args = parser.parse_args()

    logger = Logger("logs/single_runner.log")
    logger.log("Starting SINGLE CORE baseline runner...")
    logger.log("RAM Disk DISABLED (Simulating standard disk I/O)")

    try:
        config = ConfigLoader.load(args.config, logger)
        setup_fermi_environment(logger, config.get('fermi_base'))
        raise_fd_limit(logger)
        seed_system_pfiles(logger)
        if not preflight_par_check(config, logger):
            sys.exit(1)

        # Force single-core baseline: 1 core, no RAM disk (standard disk I/O)
        if 'resources' not in config:
            config['resources'] = {}
        config['resources']['cores'] = 1
        if 'ram_disk' not in config['resources']:
            config['resources']['ram_disk'] = {}
        config['resources']['ram_disk']['enabled'] = False

        # Run options come from the optional 'run:' section of the YAML config
        run_opts = config.get('run', {}) or {}
        config['dry_run'] = args.dry_run
        config['skip_merge'] = run_opts.get('skip_merge', False)
        config['skip_post'] = run_opts.get('skip_post', False)
        config['selected_steps'] = run_opts.get('steps', None)  # None implies all steps

        # Execute
        scheduler = JobScheduler(config, logger)
        
        start_time = time.time()
        scheduler.run()
        end_time = time.time()
        
        logger.log(f"Single core run complete in {end_time - start_time:.2f} seconds.")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

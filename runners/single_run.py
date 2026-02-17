#!/usr/bin/env python3
import sys
import os
import argparse
import time

from runners.parallel_run import JobScheduler, ConfigLoader, Logger, RAMDiskManager, setup_fermi_environment

def main():
    parser = argparse.ArgumentParser(description="Single-Core Fermi Runner (Baseline)")
    parser.add_argument("config", help="Path to YAML configuration file")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--no-ram-disk", action="store_true", help="Disable RAM disk (force disk I/O)")
    parser.add_argument("--steps", help="Comma-separated list of step names to run")
    parser.add_argument("--skip-merge", action="store_true", help="Skip merge phase")
    parser.add_argument("--skip-post", action="store_true", help="Skip post-processing")
    
    args = parser.parse_args()
    
    logger = Logger("single_runner.log")
    logger.log("Starting SINGLE CORE baseline runner...")

    setup_fermi_environment(logger)
    
    if args.no_ram_disk:
        logger.log("RAM Disk DISABLED (Simulating standard disk I/O)")
    
    try:
        config = ConfigLoader.load(args.config)
        
        # Override config for single core baseline
        if 'resources' not in config:
            config['resources'] = {}
        config['resources']['cores'] = 1
        
        if args.no_ram_disk:
            if 'resources' not in config: config['resources'] = {}
            if 'ram_disk' not in config['resources']: config['resources']['ram_disk'] = {}
            config['resources']['ram_disk']['enabled'] = False
            
        # Inject other CLI args (reuse logic)
        config['dry_run'] = args.dry_run
        config['skip_merge'] = args.skip_merge
        config['skip_post'] = args.skip_post
        
        if args.steps:
            config['selected_steps'] = [s.strip() for s in args.steps.split(',')]
        else:
            config['selected_steps'] = None

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

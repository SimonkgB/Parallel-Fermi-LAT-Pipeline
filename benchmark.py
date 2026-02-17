#!/usr/bin/env python3
import os
import sys
import time
import yaml
import shutil
import argparse
import subprocess
from pathlib import Path

def setup_benchmark_environment(original_config_path, num_files):
    """
    Creates a temporary setup for benchmarking:
    1. Reads original config.
    2. Creates a temp input directory with symlinks to a subset of files.
    3. Creates a temp config file pointing to this subset.
    """
    with open(original_config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    original_input_dir = config['input']['directory']
    pattern = config['input']['pattern']
    
    # 1. Create temp dirs
    bench_dir = os.path.abspath("benchmark_env")
    input_subset_dir = os.path.join(bench_dir, "input_subset")
    output_dir = os.path.join(bench_dir, "output")
    
    if os.path.exists(bench_dir):
        shutil.rmtree(bench_dir)
    os.makedirs(input_subset_dir)
    os.makedirs(output_dir)
    
    # 2. Link files
    source_path = Path(original_input_dir)
    
    files = sorted(source_path.glob(pattern))
    if not files:
        print(f"Error: No files found in {original_input_dir} matching {pattern}")
        sys.exit(1)
        
    subset = files[:num_files]
    print(f"Preparing benchmark with {len(subset)} files...")
    
    for f in subset:
        dest = os.path.join(input_subset_dir, f.name)
        os.symlink(os.path.abspath(f), dest)
        
    # 3. Create temp config
    bench_config = config.copy()
    bench_config['input']['directory'] = input_subset_dir
    # Ensure pattern matches the linked files
    bench_config['input']['pattern'] = pattern 
    
    # Redirect merge outputs to benchmark output dir
    for merge_step in bench_config.get('merging', []):
         filename = os.path.basename(merge_step['output_file'])
         merge_step['output_file'] = os.path.join(output_dir, filename)
         
    # Post-processing is skipped during benchmark (--skip-post)
    for post_step in bench_config.get('post_processing', []):
        pass

    bench_config_path = os.path.join(bench_dir, "benchmark.yaml")
    with open(bench_config_path, 'w') as f:
        yaml.dump(bench_config, f)
        
    return bench_config_path, bench_dir

def run_test(name, cmd_args, log_file):
    print(f"--- Running {name} ---")
    start = time.time()
    
    full_cmd = cmd_args + ["--skip-post"] 
    
    with open(log_file, 'w') as f:
        process = subprocess.Popen(full_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        for line in process.stdout:
            f.write(line)
            
        process.wait()
        
    end = time.time()
    duration = end - start
    
    if process.returncode != 0:
        print(f"Error: {name} failed! Check {log_file}")
        return None
        
    print(f"{name} completed in {duration:.2f}s")
    
    # Extract and show performance breakdown if present
    try:
        with open(log_file, 'r') as f:
            content = f.read()
            if "PERFORMANCE BREAKDOWN" in content:
                print("\n" + "-"*20 + " [Log Extract] " + "-"*20)
                # Extract from header to end or next section
                start_idx = content.find("PERFORMANCE BREAKDOWN") - 40 # Include the separator lines above it
                if start_idx < 0: start_idx = content.find("PERFORMANCE BREAKDOWN")
                
                # Print reasonable chunk
                snippet = content[start_idx:]
                print(snippet.strip())
                print("-" * 55 + "\n")
    except Exception as e:
        print(f"Could not extract performance stats: {e}")

    return duration

def main():
    parser = argparse.ArgumentParser(description="Fermi Pipeline Benchmark")
    parser.add_argument("--limit", type=int, default=10, help="Number of files to use for benchmark")
    parser.add_argument("--config", default="parallel_runner/fermi_pipeline.yaml", help="Base config file")
    parser.add_argument("--skip-single", action="store_true", help="Skip the single-core baseline run")
    args = parser.parse_args()
    
    # 1. Setup
    print("Setting up benchmark environment...")
    bench_config, bench_dir = setup_benchmark_environment(args.config, args.limit)
    
    # 2. Run Single Core (Baseline)
    t_single = None
    if not args.skip_single:
        # We create a run_background-like invocation but direct python
        single_log = "benchmark_single.log"
        single_cmd = [sys.executable, "single_runner/single_run.py", bench_config, "--no-ram-disk"]
        t_single = run_test("Single Core (No RAM Disk)", single_cmd, single_log)
    else:
        print("Skipping Single Core run (as requested).")
    
    # 3. Run Parallel
    parallel_log = "benchmark_parallel.log"
    parallel_cmd = [sys.executable, "parallel_runner/parallel_run.py", bench_config]
    t_parallel = run_test("Parallel (36 Cores + RAM Disk)", parallel_cmd, parallel_log)
    
    # 4. Report
    print("\n" + "="*40)
    print("BENCHMARK RESULTS")
    print("="*40)
    print(f"Files Processed: {args.limit}")
    
    if t_single:
        print(f"Single Core:   {t_single:.2f}s")
    if t_parallel:
        print(f"Parallel:      {t_parallel:.2f}s")
        
    if t_single and t_parallel:
        speedup = t_single / t_parallel
        print(f"Speedup:       {speedup:.2f}x")
    elif args.skip_single and t_parallel:
        # If user skipped single, maybe comparing to previous 1949s for 36 files?
        if args.limit == 36:
            baseline = 1949.71
            print("Compared to previous parallel run with spacecraft files outside RAM disk")
            print(f"Single Core:   {baseline:.2f}s (Reference)")
            speedup = baseline / t_parallel
            print(f"Speedup:       {speedup:.2f}x (Estimated)")
    else:
        print("Benchmark failed (one or both runs failed).")
        
    # Cleanup
    # shutil.rmtree(bench_dir) 
    print(f"\nBenchmark artifacts kept in {bench_dir}")

if __name__ == "__main__":
    main()

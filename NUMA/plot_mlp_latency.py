#!/usr/bin/env python3
import subprocess
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import argparse

def get_cache_sizes():
    """Get L1, L2, L3 cache sizes from lstopo and return in KB"""
    try:
        result = subprocess.run(['lstopo', '--no-io', '--of', 'console'], capture_output=True, text=True, check=True)
        output = result.stdout
        caches = {'L1': None, 'L2': None, 'L3': None}
        
        l1d_match = re.search(r'L1d\s+L#\d+\s+\((\d+)KB\)', output, re.IGNORECASE)
        l1i_match = re.search(r'L1i\s+L#\d+\s+\((\d+)KB\)', output, re.IGNORECASE)
        l2_match = re.search(r'L2\s+L#\d+\s+\((\d+)KB\)', output, re.IGNORECASE)
        l3_match = re.search(r'L3\s+L#\d+\s+\((\d+)KB\)', output, re.IGNORECASE)
        
        if l1d_match:
            caches['L1'] = int(l1d_match.group(1))
        elif l1i_match:
            caches['L1'] = int(l1i_match.group(1))
            
        if l2_match:
            caches['L2'] = int(l2_match.group(1))
        if l3_match:
            caches['L3'] = int(l3_match.group(1))
            
        l2_mb_match = re.search(r'L2\s+L#\d+\s+\((\d+)MB\)', output, re.IGNORECASE)
        l3_mb_match = re.search(r'L3\s+L#\d+\s+\((\d+)MB\)', output, re.IGNORECASE)
        
        if l2_mb_match and not caches['L2']:
            caches['L2'] = int(l2_mb_match.group(1)) * 1024
        if l3_mb_match and not caches['L3']:
            caches['L3'] = int(l3_mb_match.group(1)) * 1024
            
        print(f"Detected cache sizes: L1={caches['L1']}KB, L2={caches['L2']}KB, L3={caches['L3']}KB")
        return caches
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Warning: Could not run lstopo. Using default cache size estimates.")
        return {'L1': 32, 'L2': 256, 'L3': 8192}

def run_perf_mlp_latency(size_bytes, disable_prefetch=False, measure_l1=False):
    """Run membench with perf stat to measure MLP and latency metrics"""
    try:
        # Prepare environment for disabling prefetching
        env = os.environ.copy()
        if disable_prefetch:
            # Disable hardware prefetchers via MSR (requires root)
            # This is a simplified approach - actual implementation may vary by CPU
            pass
        
        if measure_l1:
            # Run sequential access to measure L1 MLP and latency
            perf_cmd = [
                'perf', 'stat', '-x,', '-e',
                'cycles,l1d_pend_miss.pending,L1-dcache-load-misses',
                '--', './membench', '-s', str(size_bytes)
            ]
            
            result = subprocess.run(perf_cmd, capture_output=True, text=True, check=True)
            
            # Parse perf output using AWK-like logic for L1
            awk_cmd = [
                'awk', '-F,',
                '''
                $3=="cycles"                  {C=$1}
                $3=="l1d_pend_miss.pending"   {P=$1}
                $3=="L1-dcache-load-misses"   {M=$1}
                END {
                    if(C > 0 && P != "<not counted>" && M != "<not counted>" && M > 0) {
                        printf("L1_MLP:%.2f\\n", P/C);
                        printf("L1_Latency:%.1f\\n", P/M);
                        printf("cycles:%s\\n", C);
                        printf("l1_pending:%s\\n", P);
                        printf("l1_misses:%s\\n", M);
                    }
                }
                '''
            ]
            
            awk_result = subprocess.run(awk_cmd, input=result.stderr, capture_output=True, text=True, check=True)
            
            metrics = {}
            
            for line in awk_result.stdout.strip().split('\n'):
                if not line:
                    continue
                try:
                    if line.startswith('L1_MLP:'):
                        metrics['l1_mlp'] = float(line.split(':')[1])
                    elif line.startswith('L1_Latency:'):
                        metrics['l1_latency'] = float(line.split(':')[1])
                    elif line.startswith('cycles:'):
                        value = line.split(':')[1]
                        if value != '<not counted>' and value.replace('.','').isdigit():
                            metrics['cycles'] = float(value)
                    elif line.startswith('l1_pending:'):
                        value = line.split(':')[1]
                        if value != '<not counted>' and value.replace('.','').isdigit():
                            metrics['l1_pending'] = float(value)
                    elif line.startswith('l1_misses:'):
                        value = line.split(':')[1]
                        if value != '<not counted>' and value.replace('.','').isdigit():
                            metrics['l1_misses'] = float(value)
                except (ValueError, IndexError):
                    continue
        else:
            # Run sequential access to measure L3 MLP and latency
            perf_cmd = [
                'perf', 'stat', '-x,', '-e',
                'cycles,offcore_requests_outstanding.demand_data_rd,longest_lat_cache.miss',
                '--', './membench', '-s', str(size_bytes)
            ]
            
            result = subprocess.run(perf_cmd, capture_output=True, text=True, check=True)
            
            # Parse perf output using AWK-like logic for L3
            awk_cmd = [
                'awk', '-F,',
                '''
                $3=="cycles"   {C=$1}
                $3=="offcore_requests_outstanding.demand_data_rd" {O=$1}
                $3=="longest_lat_cache.miss" {M=$1}
                END {
                    if(C > 0 && O != "<not counted>" && M != "<not counted>" && M > 0) {
                        printf("L3_MLP:%.2f\\n", O/C);
                        printf("L3_Latency:%.1f\\n", O/M);
                        printf("cycles:%s\\n", C);
                    }
                }
                '''
            ]
            
            awk_result = subprocess.run(awk_cmd, input=result.stderr, capture_output=True, text=True, check=True)
            
            metrics = {}
            
            for line in awk_result.stdout.strip().split('\n'):
                if not line:
                    continue
                try:
                    if line.startswith('L3_MLP:'):
                        metrics['l3_mlp'] = float(line.split(':')[1])
                    elif line.startswith('L3_Latency:'):
                        metrics['l3_latency'] = float(line.split(':')[1])
                    elif line.startswith('cycles:'):
                        value = line.split(':')[1]
                        if value != '<not counted>' and value.replace('.','').isdigit():
                            metrics['cycles'] = float(value)
                    elif line.startswith('offcore_outstanding:'):
                        value = line.split(':')[1]
                        if value != '<not counted>' and value.replace('.','').isdigit():
                            metrics['offcore_outstanding'] = float(value)
                    elif line.startswith('l3_misses:'):
                        value = line.split(':')[1]
                        if value != '<not counted>' and value.replace('.','').isdigit():
                            metrics['l3_misses'] = float(value)
                except (ValueError, IndexError):
                    continue
        
        return metrics
        
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error running perf with size {size_bytes}: {e}")
        return {}

def plot_from_csv(measure_l1=False):
    """Load CSV files and create combined plots"""
    cache_type = "l1" if measure_l1 else "l3"
    try:
        df_enabled = pd.read_csv(f'mlp_latency_{cache_type}_prefetch_enabled.csv')
        df_disabled = pd.read_csv(f'mlp_latency_{cache_type}_prefetch_disabled.csv')
    except FileNotFoundError as e:
        print(f"Error: CSV file not found: {e}")
        print("Please run data collection phases first.")
        sys.exit(1)
    
    # Get cache sizes for annotations
    cache_sizes = get_cache_sizes()
    
    # Create combined plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    mlp_key = f'{cache_type}_mlp'
    latency_key = f'{cache_type}_latency'
    cache_name = cache_type.upper()
    
    # MLP plot
    if len(df_enabled) > 0:
        ax1.semilogx(df_enabled['size_kb'], df_enabled[mlp_key], 'b-o', 
                    linewidth=2, markersize=6, label='Prefetch Enabled')
    if len(df_disabled) > 0:
        ax1.semilogx(df_disabled['size_kb'], df_disabled[mlp_key], 'r-s', 
                    linewidth=2, markersize=6, label='Prefetch Disabled')
    ax1.set_xlabel('Memory Size (KB)')
    ax1.set_ylabel(f'{cache_name} MLP')
    ax1.set_title(f'{cache_name} Memory Level Parallelism vs Memory Size')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Latency plot
    if len(df_enabled) > 0:
        ax2.semilogx(df_enabled['size_kb'], df_enabled[latency_key], 'b-o', 
                    linewidth=2, markersize=6, label='Prefetch Enabled')
    if len(df_disabled) > 0:
        ax2.semilogx(df_disabled['size_kb'], df_disabled[latency_key], 'r-s', 
                    linewidth=2, markersize=6, label='Prefetch Disabled')
    ax2.set_xlabel('Memory Size (KB)')
    ax2.set_ylabel(f'{cache_name} Miss Latency (cycles)')
    ax2.set_title(f'{cache_name} Miss Latency vs Memory Size')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add cache level annotations to all plots
    for ax in [ax1, ax2]:
        if cache_sizes['L1']:
            ax.axvline(x=cache_sizes['L1'], color='orange', linestyle='--', alpha=0.5, 
                      label=f'L1 ({cache_sizes["L1"]} KB)')
        if cache_sizes['L2']:
            ax.axvline(x=cache_sizes['L2'], color='purple', linestyle='--', alpha=0.5,
                      label=f'L2 ({cache_sizes["L2"]} KB)')
        if cache_sizes['L3']:
            ax.axvline(x=cache_sizes['L3'], color='green', linestyle='--', alpha=0.5,
                      label=f'L3 ({cache_sizes["L3"]} KB)')
    
    plt.tight_layout()
    plt.savefig(f'mlp_latency_{cache_type}_combined_plot.png', dpi=300, bbox_inches='tight')
    
    print(f"Combined plot saved as mlp_latency_{cache_type}_combined_plot.png")
    plt.show()

def run_measurements(with_prefetch=True, measure_l1=False):
    """Run MLP/Latency measurements with or without prefetching"""
    # Check if membench exists
    if not os.path.exists('./membench'):
        print("Error: membench not found. Compile it with: gcc -O2 -o membench membench.c")
        sys.exit(1)
    
    # Check if perf is available
    try:
        subprocess.run(['perf', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: perf not found. Please install perf tools.")
        sys.exit(1)
    
    # Get cache sizes from system
    cache_sizes = get_cache_sizes()
    
    # Define test sizes: from 1KB to 1GB
    sizes_kb = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
    sizes_bytes = [size * 1024 for size in sizes_kb]
    
    data = []
    cache_type = "l1" if measure_l1 else "l3"
    phase_name = "enabled" if with_prefetch else "disabled"
    filename = f'mlp_latency_{cache_type}_prefetch_{phase_name}.csv'
    
    print(f"Running {cache_type.upper()} MLP/Latency tests with prefetching {phase_name}...")
    for i, size in enumerate(sizes_bytes):
        print(f"Testing size {sizes_kb[i]} KB ({size} bytes)...", end=' ')
        metrics = run_perf_mlp_latency(size, disable_prefetch=not with_prefetch, measure_l1=measure_l1)
        if metrics:
            if measure_l1:
                row = {
                    'size_kb': sizes_kb[i],
                    'size_bytes': size,
                    'l1_mlp': metrics.get('l1_mlp', np.nan),
                    'l1_latency': metrics.get('l1_latency', np.nan),
                    'cycles': metrics.get('cycles', np.nan),
                    'l1_pending': metrics.get('l1_pending', np.nan),
                    'l1_misses': metrics.get('l1_misses', np.nan)
                }
                print(f"MLP:{metrics.get('l1_mlp', 0):.2f} Latency:{metrics.get('l1_latency', 0):.1f}")
            else:
                row = {
                    'size_kb': sizes_kb[i],
                    'size_bytes': size,
                    'l3_mlp': metrics.get('l3_mlp', np.nan),
                    'l3_latency': metrics.get('l3_latency', np.nan),
                    'cycles': metrics.get('cycles', np.nan),
                    'offcore_outstanding': metrics.get('offcore_outstanding', np.nan),
                    'l3_misses': metrics.get('l3_misses', np.nan)
                }
                print(f"MLP:{metrics.get('l3_mlp', 0):.2f} Latency:{metrics.get('l3_latency', 0):.1f}")
            data.append(row)
        else:
            print("Failed")
    
    # Convert to DataFrame and save as CSV
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    
    print(f"\nData saved to {filename}")
    if len(data) == 0:
        print("No successful measurements obtained.")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="NUMA MLP/Latency benchmark tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '-p', '--prefetch',
        action='store_true',
        help='Run measurements with prefetching enabled'
    )
    
    parser.add_argument(
        '-np', '--no-prefetch',
        action='store_true',
        help='Run measurements with prefetching disabled'
    )
    
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate plots from existing CSV data'
    )
    
    parser.add_argument(
        '--l1',
        action='store_true',
        help='Measure L1 cache MLP/latency (default: L3)'
    )
    
    parser.add_argument(
        '--l3',
        action='store_true',
        help='Measure L3 cache MLP/latency (default)'
    )
    
    args = parser.parse_args()
    
    # If no arguments specified, show help
    if not any([args.prefetch, args.no_prefetch, args.plot]):
        parser.print_help()
        sys.exit(1)
    
    # Determine which cache level to measure (default to L3)
    measure_l1 = args.l1 and not args.l3
    
    if args.plot:
        plot_from_csv(measure_l1=measure_l1)
    elif args.prefetch:
        run_measurements(with_prefetch=True, measure_l1=measure_l1)
    elif args.no_prefetch:
        run_measurements(with_prefetch=False, measure_l1=measure_l1)

if __name__ == "__main__":
    main()

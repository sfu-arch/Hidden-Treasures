#!/usr/bin/env python3
import subprocess
import re
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
# Can only monitor 2 out of 3 caches at the same time due to lack of performance counter events.
def get_cache_sizes():
    """Get L1, L2, L3 cache sizes from lstopo and return in KB"""
    try:
        result = subprocess.run(['lstopo', '--no-io', '--of', 'console'], capture_output=True, text=True, check=True)
        output = result.stdout
        caches = {'L1': None, 'L2': None, 'L3': None}
        
        # Parse cache information from lstopo output
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
            
        # Handle MB units as well
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

def run_perf_membench(size_bytes):
    """Run membench with perf stat and return miss rates and absolute miss counts"""
    try:
        perf_cmd = [
            'perf', 'stat', '-x,', '-e',
            'l2_rqsts.all_demand_references,l2_rqsts.all_demand_miss,LLC-loads,LLC-load-misses',
            '--', './membench', '-s', str(size_bytes)
        ]
        
        result = subprocess.run(perf_cmd, capture_output=True, text=True, check=True)
        # Print output for debugging
        print(result.stderr.strip())
        # Parse perf output using AWK-like logic
        awk_cmd = [
            'awk', '-F,',
            '''
            $3=="l2_rqsts.all_demand_references"   {l2=$1}
            $3=="l2_rqsts.all_demand_miss"         {l2m=$1}
            $3=="LLC-loads"                        {l3=$1}
            $3=="LLC-load-misses"                  {l3m=$1}
            END {
                if(l2 > 0 && l2m != "<not counted>" && l3m != "<not counted>") {
                    printf("L2_rate:%.2f\\n", 100*l2m/l2);
                    printf("L3_rate:%.2f\\n", 100*l3m/l2);
                    printf("L2_misses:%s\\n", l2m);
                    printf("L3_misses:%s\\n", l3m);
                }
            }
            '''
        ]
        
        awk_result = subprocess.run(awk_cmd, input=result.stderr, capture_output=True, text=True, check=True)
        
        miss_rates = {'L1': None, 'L2': None, 'L3': None}
        miss_counts = {'L1': None, 'L2': None, 'L3': None}
        
        for line in awk_result.stdout.strip().split('\n'):
            if not line:
                continue
            try:
                if line.startswith('L2_rate:'):
                    miss_rates['L2'] = float(line.split(':')[1])
                elif line.startswith('L3_rate:'):
                    miss_rates['L3'] = float(line.split(':')[1])
                elif line.startswith('L2_misses:'):
                    value = line.split(':')[1]
                    if value != '<not counted>' and value.isdigit():
                        miss_counts['L2'] = int(value)
                elif line.startswith('L3_misses:'):
                    value = line.split(':')[1]
                    if value != '<not counted>' and value.isdigit():
                        miss_counts['L3'] = int(value)
            except (ValueError, IndexError):
                continue
        
        return miss_rates, miss_counts
        
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error running perf with size {size_bytes}: {e}")
        return ({'L1': None, 'L2': None, 'L3': None}, 
                {'L1': None, 'L2': None, 'L3': None})

def main():
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
    
    l1_miss_rates = []
    l2_miss_rates = []
    l3_miss_rates = []
    l1_miss_counts = []
    l2_miss_counts = []
    l3_miss_counts = []
    actual_sizes = []
    
    print("Running miss rate tests...")
    for i, size in enumerate(sizes_bytes):
        print(f"Testing size {sizes_kb[i]} KB ({size} bytes)...", end=' ')
        miss_rates, miss_counts = run_perf_membench(size)
        
        if miss_rates['L2'] is not None:
            l2_miss_rates.append(miss_rates['L2'] if miss_rates['L2'] is not None else 0)
            l3_miss_rates.append(miss_rates['L3'] if miss_rates['L3'] is not None else 0)
            l2_miss_counts.append(miss_counts['L2'] if miss_counts['L2'] is not None else 0)
            l3_miss_counts.append(miss_counts['L3'] if miss_counts['L3'] is not None else 0)
            actual_sizes.append(sizes_kb[i])
            print(f"L2:{miss_rates['L2']:.1f}% L3:{miss_rates['L3']:.1f}%")
        else:
            print("Failed")
    
    if not l2_miss_rates:
        print("No successful measurements obtained.")
        sys.exit(1)
    
    # Create miss rate plot
    plt.figure(figsize=(12, 8))
    plt.semilogx(actual_sizes, l2_miss_rates, 'go-', linewidth=2, markersize=6, label='L2 Miss Rate')
    plt.semilogx(actual_sizes, l3_miss_rates, 'bo-', linewidth=2, markersize=6, label='L3 Miss Rate')
    
    plt.xlabel('Memory Size (KB)', fontsize=12)
    plt.ylabel('Miss Rate (%)', fontsize=12)
    plt.title('Cache Miss Rates vs Memory Size', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    
    # Add cache level annotations
    if cache_sizes['L1']:
        plt.axvline(x=cache_sizes['L1'], color='r', linestyle='--', alpha=0.7, 
                   label=f'L1 Cache ({cache_sizes["L1"]} KB)')
    if cache_sizes['L2']:
        plt.axvline(x=cache_sizes['L2'], color='orange', linestyle='--', alpha=0.7, 
                   label=f'L2 Cache ({cache_sizes["L2"]} KB)')
    if cache_sizes['L3']:
        plt.axvline(x=cache_sizes['L3'], color='purple', linestyle='--', alpha=0.7, 
                   label=f'L3 Cache ({cache_sizes["L3"]} KB)')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig('miss_rate_plot.png', dpi=300, bbox_inches='tight')
    
    # Create absolute miss count plot
    plt.figure(figsize=(12, 8))
    plt.loglog(actual_sizes, l2_miss_counts, 'go-', linewidth=2, markersize=6, label='L2 Miss Count')
    plt.loglog(actual_sizes, l3_miss_counts, 'bo-', linewidth=2, markersize=6, label='L3 Miss Count')
    
    plt.xlabel('Memory Size (KB)', fontsize=12)
    plt.ylabel('Absolute Miss Count', fontsize=12)
    plt.title('Cache Miss Counts vs Memory Size', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add cache level annotations
    if cache_sizes['L1']:
        plt.axvline(x=cache_sizes['L1'], color='r', linestyle='--', alpha=0.7, 
                   label=f'L1 Cache ({cache_sizes["L1"]} KB)')
    if cache_sizes['L2']:
        plt.axvline(x=cache_sizes['L2'], color='orange', linestyle='--', alpha=0.7, 
                   label=f'L2 Cache ({cache_sizes["L2"]} KB)')
    if cache_sizes['L3']:
        plt.axvline(x=cache_sizes['L3'], color='purple', linestyle='--', alpha=0.7, 
                   label=f'L3 Cache ({cache_sizes["L3"]} KB)')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig('miss_count_plot.png', dpi=300, bbox_inches='tight')
    
    print(f"\nPlots saved as miss_rate_plot.png and miss_count_plot.png")
    
    # Show the plots
    plt.show()

if __name__ == "__main__":
    main()



#!/usr/bin/env python3
import subprocess
import re
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def get_cache_sizes():
    """Get L1, L2, L3 cache sizes from lstopo and return in KB"""
    try:
        result = subprocess.run(['lstopo', '--no-io', '--of', 'console'], capture_output=True, text=True, check=True)
        output = result.stdout
        # print(output)
        caches = {'L1': None, 'L2': None, 'L3': None}
        
        # Parse cache information from lstopo output
        # Look for patterns like "L1d L#41 (32KB)" or "L2 L#41 (1024KB)"
        l1d_match = re.search(r'L1d\s+L#\d+\s+\((\d+)KB\)', output, re.IGNORECASE)
        l1i_match = re.search(r'L1i\s+L#\d+\s+\((\d+)KB\)', output, re.IGNORECASE)
        l2_match = re.search(r'L2\s+L#\d+\s+\((\d+)KB\)', output, re.IGNORECASE)
        l3_match = re.search(r'L3\s+L#\d+\s+\((\d+)KB\)', output, re.IGNORECASE)
        
        # Use L1d cache size (data cache) as the primary L1 size
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
        return {'L1': 32, 'L2': 256, 'L3': 8192}  # Default estimates

def run_membench(size_bytes):
    """Run membench with sequential scan for given size and return bandwidth in GB/s"""
    try:
        cmd = ['./membench', '-s', str(size_bytes)]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Parse output for bandwidth: "seq 12.3 GB/s (checksum=...)"
        match = re.search(r'seq ([\d.]+) GB/s', result.stdout)
        if match:
            return float(match.group(1))
        else:
            print(f"Warning: Could not parse bandwidth from: {result.stdout.strip()}")
            return None
    except subprocess.CalledProcessError as e:
        print(f"Error running membench with size {size_bytes}: {e}")
        return None
    except FileNotFoundError:
        print("Error: membench executable not found. Please compile it first.")
        sys.exit(1)

def main():
    # Check if membench exists
    if not os.path.exists('./membench'):
        print("Error: membench not found. Compile it with: gcc -O2 -o membench membench.c")
        sys.exit(1)
    
    # Get cache sizes from system
    cache_sizes = get_cache_sizes()
    
    # Define test sizes: from 1KB to 1GB
    sizes_kb = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
    sizes_bytes = [size * 1024 for size in sizes_kb]  # Convert KB to bytes
    
    bandwidths = []
    actual_sizes = []
    
    print("Running bandwidth tests...")
    for i, size in enumerate(sizes_bytes):
        print(f"Testing size {sizes_kb[i]} KB ({size} bytes)...", end=' ')
        bandwidth = run_membench(size)
        if bandwidth is not None:
            bandwidths.append(bandwidth)
            actual_sizes.append(sizes_kb[i])
            print(f"{bandwidth:.1f} GB/s")
        else:
            print("Failed")
    
    if not bandwidths:
        print("No successful measurements obtained.")
        sys.exit(1)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    plt.semilogx(actual_sizes, bandwidths, 'bo-', linewidth=2, markersize=6)
    plt.xlabel('Memory Size (KB)', fontsize=12)
    plt.ylabel('Bandwidth (GB/s)', fontsize=12)
    plt.title('Sequential Memory Access Bandwidth vs Memory Size', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add cache level annotations using detected sizes
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
    
    # Save the plot
    plt.savefig('bandwidth_plot.png', dpi=300, bbox_inches='tight')
    
    print(f"\nPlot saved as bandwidth_plot.png")
    print(f"Peak bandwidth: {max(bandwidths):.1f} GB/s at {actual_sizes[bandwidths.index(max(bandwidths))]} KB")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()

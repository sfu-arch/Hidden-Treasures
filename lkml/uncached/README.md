# Linux Kernel Memory Management & Cache Control Modules

A comprehensive educational project demonstrating memory management and CPU cache behavior in Linux kernel modules. This project provides three modules showcasing different memory allocation strategies, cache control techniques, and real-world kernel programming patterns.

## 📋 Table of Contents

- [Linux Kernel Memory Management \& Cache Control Modules](#linux-kernel-memory-management--cache-control-modules)
  - [📋 Table of Contents](#-table-of-contents)
  - [🚀 Quick Start](#-quick-start)
  - [📊 Project Overview](#-project-overview)
    - [What This Project Demonstrates](#what-this-project-demonstrates)
    - [Educational Goals](#educational-goals)
  - [🔧 Module Descriptions](#-module-descriptions)
    - [1. Static Cache Control (`uncached_mem`)](#1-static-cache-control-uncached_mem)
    - [2. Dynamic Cache Control (`dynamic_cache`)](#2-dynamic-cache-control-dynamic_cache)
    - [3. DMA/CMA Cache Control (`cma_cache`)](#3-dmacma-cache-control-cma_cache)
  - [📁 Project Structure](#-project-structure)
  - [⚙️ Installation \& Setup](#️-installation--setup)
    - [Prerequisites](#prerequisites)
    - [Building the Modules](#building-the-modules)
    - [CMA Setup (For Large Allocations)](#cma-setup-for-large-allocations)
  - [📖 Usage Guide](#-usage-guide)
    - [Static Cache Control Module](#static-cache-control-module)
    - [Dynamic Cache Control Module](#dynamic-cache-control-module)
    - [CMA Cache Control Module](#cma-cache-control-module)
  - [📈 Performance Analysis](#-performance-analysis)
    - [Expected Performance Results](#expected-performance-results)
    - [Performance Testing Commands](#performance-testing-commands)
  - [🔧 Troubleshooting](#-troubleshooting)
    - [Common Issues](#common-issues)
      - [1. Module Loading Failures](#1-module-loading-failures)
      - [2. Permission Denied Errors](#2-permission-denied-errors)
      - [3. CMA Allocation Failures](#3-cma-allocation-failures)
      - [4. Performance Inconsistencies](#4-performance-inconsistencies)
    - [Diagnostic Tools](#diagnostic-tools)
  - [🎓 Educational Resources](#-educational-resources)
    - [Learning Path](#learning-path)
    - [Key Concepts Demonstrated](#key-concepts-demonstrated)
      - [Memory Allocation APIs](#memory-allocation-apis)
      - [Cache Control](#cache-control)
      - [Kernel Interfaces](#kernel-interfaces)
      - [Performance Analysis](#performance-analysis)
    - [Further Reading](#further-reading)
  - [📄 License](#-license)
  - [📚 Version History](#-version-history)
  - [🔗 Related Documentation](#-related-documentation)



## 🚀 Quick Start

```bash
# 1. Build all modules
make all

# 2. Quick test - Static cache control
sudo insmod uncached_mem.ko
echo "0 4M" > /sys/kernel/uncached_mem/command
./timing_test

# 3. Quick test - Dynamic cache control  
sudo insmod dynamic_cache.ko
echo "alloc 8K" > /sys/kernel/dynamic_cache/command
./dynamic_test

# 4. Quick test - CMA large allocations
sudo insmod cma_cache.ko
echo "alloc 16M" > /sys/kernel/cma_cache/command
./cma_test
```

## 📊 Project Overview

### What This Project Demonstrates

- **Memory allocation strategies**: kmalloc, vmalloc, __get_free_pages, DMA coherent
- **Cache control mechanisms**: set_memory_uc(), set_memory_wb(), runtime cache toggling
- **Kernel-userspace interfaces**: sysfs, character devices, mmap
- **Performance analysis**: Cache vs uncached memory access patterns
- **Modern kernel development**: Error handling, resource management, documentation

### Educational Goals

After working with this project, you will understand:
- How CPU caches impact memory performance (50-70x difference!)
- Different kernel memory allocation APIs and their trade-offs
- Modern kernel module development with sysfs interfaces
- Memory mapping techniques for sharing kernel memory with userspace
- Real-world performance analysis and optimization techniques

## 🔧 Module Descriptions

### 1. Static Cache Control (`uncached_mem`)

**Purpose**: Basic cache behavior demonstration with variable-size allocation

**Key Features**:
- Variable size allocation: 4KB to 128MB
- Static cache state (set at allocation time)
- Automatic allocation method selection (kmalloc/vmalloc)
- sysfs interface with size parameter support

**Use Cases**:
- Understanding cache impact on performance
- Learning kernel memory allocation APIs
- Basic kernel module development patterns

**Interface**:
```bash
# Control: /sys/kernel/uncached_mem/command
# Device: /dev/uncached_mem
# Example: echo "0 4M" > command  # 4MB uncached
```

### 2. Dynamic Cache Control (`dynamic_cache`)

**Purpose**: Advanced per-page cache control with runtime manipulation

**Key Features**:
- Individual page cache control (4KB granularity)
- Runtime cache state toggling
- Variable-size block allocation (4KB to 64MB)
- Page pool management (1024 pages)
- Visual allocation mapping

**Use Cases**:
- Fine-grained cache research
- Understanding page-level memory management
- Performance testing with mixed cache states
- Advanced kernel programming techniques

**Interface**:
```bash
# Control: /sys/kernel/dynamic_cache/command  
# Device: /dev/dynamic_cache
# Examples: 
echo "alloc 16K" > command     # Allocate 4-page block
echo "uncache 5" > command     # Make page 5 uncached
echo "toggle 3" > command      # Toggle page 3 cache state
```

### 3. DMA/CMA Cache Control (`cma_cache`)

**Purpose**: Large contiguous memory allocation with DMA/CMA backend

**Key Features**:
- Large contiguous allocations: 1MB to 256MB
- DMA coherent memory (suitable for device drivers)
- Guaranteed physical contiguity
- NUMA-aware allocation support
- Cache control on large memory blocks

**Use Cases**:
- Device driver development
- Large buffer management
- Understanding CMA (Contiguous Memory Allocator)
- NUMA memory allocation patterns

**Interface**:
```bash
# Control: /sys/kernel/cma_cache/command
# Device: /dev/cma_cache  
# Examples:
echo "alloc 64M" > command     # Allocate 64MB contiguous block
echo "uncache 1" > command     # Make allocation 1 uncached
echo "numa 0" > command        # Target NUMA node 0
```

## 📁 Project Structure

```
uncached/
├── README.md                  # This file - comprehensive user guide
├── DESIGN.md                  # Technical design document & API reference
├── TROUBLESHOOTING.md         # Detailed troubleshooting guide
├── KERNEL_BUILD.md            # Kernel building guide with CMA support
├── Makefile                   # Build system for all modules and tests
│
├── uncached_mem.c             # Static cache control module
├── dynamic_cache.c            # Dynamic cache control module  
├── cma_cache.c                # DMA/CMA cache control module
│
├── timing_test.c              # Test program for uncached_mem
├── dynamic_test.c             # Test program for dynamic_cache
├── dynamic_size_test.c        # Variable size test for dynamic_cache
├── cma_test.c                 # Test program for cma_cache
├── cma_debug.c                # CMA diagnostic tool
│
└── run_dynamic_test.sh        # Automated testing script
```

## ⚙️ Installation & Setup

### Prerequisites

```bash
# Install kernel headers
sudo apt-get install linux-headers-$(uname -r)  # Ubuntu/Debian
sudo yum install kernel-devel                   # RHEL/CentOS
sudo dnf install kernel-devel                   # Fedora

# Install build tools
sudo apt-get install build-essential            # Ubuntu/Debian
sudo yum groupinstall "Development Tools"       # RHEL/CentOS
```

### Building the Modules

```bash
# Build all modules and test programs
make all

# Build individual components
make uncached_mem.ko        # Static cache module
make dynamic_cache.ko       # Dynamic cache module  
make cma_cache.ko           # CMA cache module
make timing_test            # Test programs
```

### CMA Setup (For Large Allocations)

The CMA module requires CMA memory to be available. Check current status:

```bash
# Check CMA availability
grep -E 'Cma(Total|Free)' /proc/meminfo
```

If CmaTotal is 0 or too small (< 64MB), configure GRUB:

```bash
# Edit GRUB configuration
sudo nano /etc/default/grub

# Add CMA parameter to GRUB_CMDLINE_LINUX:
GRUB_CMDLINE_LINUX="... cma=256M movable_node"

# Update GRUB and reboot
sudo update-grub && sudo reboot
```

**CMA Size Recommendations**:
- **Basic testing**: `cma=128M`
- **Large allocation testing**: `cma=256M`  
- **Stress testing**: `cma=512M` (8GB+ RAM systems)
- **NUMA systems**: `cma=256M@node0,256M@node1`

## 📖 Usage Guide

### Static Cache Control Module

**Load and test**:
```bash
# Load module
sudo insmod uncached_mem.ko

# Set permissions (one-time setup)
sudo chmod 666 /sys/kernel/uncached_mem/command /dev/uncached_mem

# Allocate memory and test
echo "0 4M" > /sys/kernel/uncached_mem/command    # 4MB uncached
echo "1 4M" > /sys/kernel/uncached_mem/command    # 4MB cached
./timing_test                                     # Run performance test

# Check status
cat /sys/kernel/uncached_mem/status

# Cleanup
echo "2" > /sys/kernel/uncached_mem/command       # Free all
sudo rmmod uncached_mem
```

**Size formats supported**: 4K, 8K, 1M, 512M, 1G (with K/M/G suffixes)

### Dynamic Cache Control Module

**Load and test**:
```bash
# Load module
sudo insmod dynamic_cache.ko
sudo chmod 666 /sys/kernel/dynamic_cache/command /dev/dynamic_cache

# Allocate pages and blocks
echo "alloc" > /sys/kernel/dynamic_cache/command      # Single page
echo "alloc 16K" > /sys/kernel/dynamic_cache/command  # 4-page block
echo "alloc 1M" > /sys/kernel/dynamic_cache/command   # 256-page block

# Control cache states
echo "uncache 0" > /sys/kernel/dynamic_cache/command  # Make page 0 uncached
echo "cache_block 2" > /sys/kernel/dynamic_cache/command  # Cache entire block 2
echo "toggle 5" > /sys/kernel/dynamic_cache/command   # Toggle page 5

# View allocation map
cat /sys/kernel/dynamic_cache/page_map
cat /sys/kernel/dynamic_cache/status

# Run comprehensive test
./dynamic_test

# Cleanup
sudo rmmod dynamic_cache
```

### CMA Cache Control Module

**Load and test**:
```bash
# Check CMA availability first
grep -E 'Cma(Total|Free)' /proc/meminfo

# Load module
sudo insmod cma_cache.ko
sudo chmod 666 /sys/kernel/cma_cache/command /dev/cma_cache

# Allocate large contiguous blocks
echo "alloc 8M" > /sys/kernel/cma_cache/command    # 8MB block
echo "alloc 32M" > /sys/kernel/cma_cache/command   # 32MB block
echo "alloc 128M" > /sys/kernel/cma_cache/command  # 128MB block

# Control cache attributes
echo "uncache 1" > /sys/kernel/cma_cache/command   # Make allocation 1 uncached
echo "toggle 2" > /sys/kernel/cma_cache/command    # Toggle allocation 2

# NUMA targeting (multi-node systems)
echo "numa 1" > /sys/kernel/cma_cache/command      # Target node 1

# Check detailed status (includes NUMA info)
cat /sys/kernel/cma_cache/status

# Run test suite
./cma_test basic        # Basic allocation tests
./cma_test performance  # Performance comparison
./cma_test stress       # Stress testing

# Cleanup  
echo "free 1" > /sys/kernel/cma_cache/command      # Free specific allocation
sudo rmmod cma_cache
```

## 📈 Performance Analysis

### Expected Performance Results

**Typical cache performance impact**:
- **Cached memory access**: ~1.3 ns per operation
- **Uncached memory access**: ~70 ns per operation  
- **Performance ratio**: 50-70x slower for uncached memory

**Example output from timing_test**:
```
=== PERFORMANCE SUMMARY ===
User space cached:     1.34 ns per access
Kernel uncached (mmap): 74.01 ns per access  
Kernel cached (mmap):   1.32 ns per access

Performance ratio: Uncached is 56.0x slower than cached
```

### Performance Testing Commands

```bash
# Static module performance test
./timing_test

# Dynamic module with different cache states
./dynamic_test
./dynamic_size_test

# CMA module performance comparison
./cma_test performance

# Custom performance testing
echo "alloc 1M" > /sys/kernel/dynamic_cache/command
echo "uncache 0" > /sys/kernel/dynamic_cache/command  # First page uncached
echo "cache 1" > /sys/kernel/dynamic_cache/command    # Second page cached
# Now mmap and test both pages to see performance difference
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Module Loading Failures

**Error**: `insmod: ERROR: could not insert module: Operation not permitted`

**Solutions**:
```bash
# Check kernel headers
ls /lib/modules/$(uname -r)/build

# Rebuild modules
make clean && make all

# Check dmesg for specific errors
dmesg | tail -10
```

#### 2. Permission Denied Errors

**Error**: `Permission denied` when writing to sysfs files

**Solution**:
```bash
# Set permissions for all modules
sudo chmod 666 /dev/uncached_mem /sys/kernel/uncached_mem/command
sudo chmod 666 /dev/dynamic_cache /sys/kernel/dynamic_cache/command  
sudo chmod 666 /dev/cma_cache /sys/kernel/cma_cache/command
```

#### 3. CMA Allocation Failures

**Error**: `Failed to allocate X bytes from DMA coherent memory`

**Diagnosis**:
```bash
# Check CMA status
grep -E 'Cma(Total|Free)' /proc/meminfo
./cma_debug  # Run diagnostic tool

# Check for memory fragmentation
cat /proc/buddyinfo
```

**Solutions**:
- Increase CMA pool size in GRUB: `cma=256M` 
- Add `movable_node` parameter to reduce fragmentation
- Try smaller allocation sizes
- Reboot to defragment memory

#### 4. Performance Inconsistencies

**Issue**: Performance results vary significantly between runs

**Solutions**:
- Disable CPU frequency scaling: `echo performance > /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor`
- Run tests multiple times and average results
- Ensure system is not under load during testing
- Use larger test datasets for more stable results

### Diagnostic Tools

```bash
# CMA diagnostic tool
make cma_debug
./cma_debug

# Check module status
cat /sys/kernel/*/status

# Monitor kernel messages
dmesg -w

# Check memory usage
free -h && cat /proc/meminfo | grep -E 'Cma|Available'
```

## 🎓 Educational Resources

### Learning Path

1. **Start with static module** (`uncached_mem`): Understand basic cache concepts
2. **Explore dynamic module** (`dynamic_cache`): Learn page-level control
3. **Study CMA module** (`cma_cache`): Understand large contiguous allocation

### Key Concepts Demonstrated

#### Memory Allocation APIs
- `kmalloc()` vs `vmalloc()` vs `__get_free_pages()`
- `dma_alloc_coherent()` for device-compatible memory
- Trade-offs: size limits, contiguity, performance

#### Cache Control
- `set_memory_uc()` and `set_memory_wb()` functions
- TLB flushing requirements
- Architecture-specific behavior

#### Kernel Interfaces
- Modern sysfs design patterns
- Character device implementation
- Memory mapping (`mmap`) techniques

#### Performance Analysis
- Measuring memory access latency
- Understanding cache hierarchy impact
- Quantifying performance trade-offs

### Further Reading

- **DESIGN.md**: Detailed technical implementation
- **KERNEL_BUILD.md**: Building custom kernels with CMA support
- **Intel optimization manual**: Cache optimization techniques
- **Linux Device Drivers book**: Advanced kernel programming

## 📄 License

GPL v2 - Compatible with Linux kernel licensing

## 📚 Version History

- **v3.0**: Added CMA/DMA module with NUMA support, comprehensive reorganization
- **v2.0**: Added dynamic cache control, sysfs interface, variable size support  
- **v1.0**: Basic uncached memory allocation with proc interface

---

## 🔗 Related Documentation

- **[DESIGN.md](DESIGN.md)**: Technical architecture and API reference
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)**: Detailed problem solving
- **[KERNEL_BUILD.md](KERNEL_BUILD.md)**: Building custom kernels with CMA

---

*This project is designed for educational purposes and demonstrates real-world kernel programming techniques. It provides hands-on experience with memory management, cache control, and performance analysis in Linux kernel development.*

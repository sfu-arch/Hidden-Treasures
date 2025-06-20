# Memory Management Kernel Modules for Cache Control Education

This project demonstrates comprehensive memory management techniques in Linux kernel modules, featuring three educational modules that showcase different aspects of memory allocation, cache control, and hardware interaction.

## Recent Updates (December 2024)

**🔧 Fixed CMA Test Allocation Issues**: 
- Resolved allocation ID tracking problems that caused "allocation not found" errors
- Improved test robustness with dynamic ID detection instead of hardcoded values
- Reduced default test sizes (1M, 2M, 4M) to prevent CMA memory exhaustion
- Added `cma_debug` tool for comprehensive CMA troubleshooting
- See `TROUBLESHOOTING.md` for detailed problem analysis and solutions

**✨ New Tools**:
- `cma_debug`: Diagnostic tool for CMA allocation issues
- Improved error handling and status reporting in all test programs

## Project Structure

```
uncached/
├── README.md              # This documentation file
├── DESIGN.md             # Detailed design explanation and API reference
├── TROUBLESHOOTING.md    # CMA allocation troubleshooting guide
├── KERNEL_BUILD.md       # Complete guide to building kernel with CMA support
├── uncached_mem.c        # Static cache control module with sysfs interface
├── dynamic_cache.c       # Dynamic per-page cache control module
├── cma_cache.c           # DMA/CMA-based large contiguous allocation module
├── Makefile             # Build configuration for all modules and test programs
├── timing_test.c        # Test program for uncached_mem module
├── dynamic_test.c       # Test program for dynamic_cache module
├── dynamic_size_test.c  # Variable size allocation test for dynamic_cache
├── cma_test.c           # Improved test program for cma_cache module
└── cma_debug.c          # CMA diagnostic and debugging tool
```

## Modules Overview

### 1. Static Cache Control (`uncached_mem`)
- **Purpose**: Basic cache behavior demonstration
- **Features**: Variable size allocation (4KB-128MB), static cache state
- **Use Case**: Understanding cache impact on performance

### 2. Dynamic Cache Control (`dynamic_cache`) 
- **Purpose**: Advanced per-page cache manipulation
- **Features**: Individual page cache control, variable-size blocks, runtime toggling
- **Use Case**: Fine-grained cache research and analysis

### 3. DMA/CMA Cache Control (`cma_cache`)
- **Purpose**: Large contiguous memory allocation with cache control
- **Features**: DMA coherent allocation (1MB-256MB), physical contiguity, cache attribute control
- **Use Case**: Device driver development, large buffer management

## Documentation

- **README.md**: User guide with installation, usage, and examples
- **DESIGN.md**: Technical design document explaining kernel APIs, architecture, gotchas, and educational insights for students who have completed an OS course

## Overview

The project consists of three main components:

1. **`uncached_mem.c`** - A kernel module that:
   - Allocates cached/uncached memory with **variable size support** (4KB to 128MB)
   - Uses **vmalloc for large allocations** (>1MB) and __get_free_pages for smaller ones
   - Provides **sysfs interface** at `/sys/kernel/uncached_mem/` for modern control
   - Creates **character device** `/dev/uncached_mem` for mmap functionality
   - Supports **size parameters** with K/M/G suffixes (e.g., "4K", "1M", "512M")
   - Handles **large allocations up to 1GB** using vmalloc approach
   - Maps memory to user space with proper caching attributes

2. **`timing_test.c`** - A user space test program that:
   - Uses **sysfs interface** for module control
   - Tests **variable allocation sizes** including large buffers
   - Directly accesses kernel-allocated memory via mmap
   - Demonstrates **size parameter usage** with different allocation methods
   - Provides nanosecond-precision timing measurements
   - Shows performance differences between allocation methods

3. **`Makefile`** - Build system with updated targets for the new interface

## Sysfs Interface

The module uses a modern sysfs interface instead of proc filesystem:

### Control Files
- **`/sys/kernel/uncached_mem/command`** - Main control interface
- **`/sys/kernel/uncached_mem/status`** - Current allocation status  
- **`/sys/kernel/uncached_mem/uncached_addr`** - Uncached buffer address
- **`/sys/kernel/uncached_mem/cached_addr`** - Cached buffer address
- **`/sys/kernel/uncached_mem/size_info`** - Size and allocation method info

### Device File
- **`/dev/uncached_mem`** - Character device for mmap functionality

## Commands and Size Support

The module accepts commands with optional size parameters:

### Command Format
```bash
echo "command [size]" > /sys/kernel/uncached_mem/command
```

### Available Commands
- **`0 [size]`** - `alloc_uc`: Allocate uncached memory
- **`1 [size]`** - `alloc_cached`: Allocate cached memory  
- **`2`** - `free`: Free all allocated memory

### Size Format
- **Bytes**: `4096`, `1048576`
- **With suffixes**: `4K`, `1M`, `512M`, `1G`
- **Default**: 4096 bytes if no size specified
- **Minimum**: 4096 bytes (page size)
- **Maximum**: 128MB (can be increased for larger allocations)

### Examples
```bash
# Allocate 8KB uncached memory
echo "0 8K" > /sys/kernel/uncached_mem/command

# Allocate 4MB cached memory  
echo "1 4M" > /sys/kernel/uncached_mem/command

# Allocate 1GB uncached (may require vmalloc)
echo "0 1G" > /sys/kernel/uncached_mem/command

# Free all allocated memory
echo "2" > /sys/kernel/uncached_mem/command
```

## Size Allocation Methods and Limitations

### Allocation Strategy
The module automatically chooses the best allocation method based on size:

| Size Range | Method | Notes |
|------------|--------|-------|
| 4KB - 1MB | `__get_free_pages` / `kmalloc` | Contiguous physical memory |
| 1MB - 128MB | `vmalloc` | Non-contiguous physical memory |
| >128MB | Not supported | Would require reserved memory |

### Large Allocation Support (>1MB)
For allocations larger than 1MB, the module uses `vmalloc()`:
- **Advantages**: Can allocate very large buffers (up to 1GB)
- **Limitations**: Non-contiguous physical memory
- **Uncached behavior**: Applied per-page using `set_memory_uc()`
- **mmap support**: Uses `remap_vmalloc_range()` for proper mapping

### 1GB Allocation Notes
When requesting 1GB of uncached memory:
```bash
echo "0 1G" > /sys/kernel/uncached_mem/command
```

**Expected behavior:**
- Uses vmalloc for allocation (non-contiguous physical memory)
- May take several seconds to complete
- System may experience temporary performance impact
- Success depends on available virtual memory
- Each page individually marked as uncached

**Potential issues:**
- **Memory fragmentation**: May fail on systems with limited memory
- **Performance impact**: Large allocations can impact system responsiveness  
- **Architecture limitations**: Some systems may not support large uncached regions
- **Virtual memory limits**: Depends on available kernel virtual address space

**Recommendations for large allocations:**
1. Test with smaller sizes first (256MB, 512MB)
2. Monitor system memory with `free -h` before attempting
3. Consider using reserved memory at boot time for guaranteed large allocations
4. Use dmesg to monitor allocation progress and any warnings

Building and Installation

### Prerequisites

- Linux kernel headers for your current kernel version
- GCC compiler
- Make utility
- Root access for module loading

```bash
# Install kernel headers (Ubuntu/Debian)
sudo apt-get install linux-headers-$(uname -r)

# Install kernel headers (RHEL/CentOS/Fedora)
sudo yum install kernel-devel-$(uname -r)
# or
sudo dnf install kernel-devel-$(uname -r)
```

### Building

```bash
# Build everything (kernel module + test program)
make all timing_test

# Build only the kernel module
make

# Build only the test program
make timing_test
```

## Usage

### 🚀 Quick Start for CMA Module (Most Common Use Case)

**⚠️ Prerequisites: Ensure CMA is enabled in your kernel**

If you get CMA allocation failures, you may need to enable/increase CMA memory:

```bash
# 1. Check if CMA is available
cat /proc/meminfo | grep -i cma
# Should show CmaTotal and CmaFree

# 2. If no CMA found, add kernel boot parameter:
sudo nano /etc/default/grub
# Add to GRUB_CMDLINE_LINUX_DEFAULT: "cma=128M"
# Example: GRUB_CMDLINE_LINUX_DEFAULT="quiet splash cma=128M"
sudo update-grub && sudo reboot

# 3. For detailed CMA setup, see CMA_KERNEL_SETUP.md
```

**Standard workflow:**

```bash
# 1. Build everything
make clean && make

# 2. Load the CMA module
sudo insmod cma_cache.ko
sudo chmod 666 /dev/cma_cache /sys/kernel/cma_cache/command

# 3. If having allocation issues, run diagnostics first:
./cma_debug cleanup   # Clean any leftover allocations
./cma_debug info      # Check system CMA information

# 4. Run the improved test program
./cma_test basic      # Basic functionality test
./cma_test performance # Performance comparison test

# 5. If you encounter allocation failures:
./cma_debug all       # Full diagnostic suite
```

**Note**: If you experience "allocation not found" errors, see `TROUBLESHOOTING.md` for detailed solutions.

### Basic Usage with Sysfs Interface

1. **Load the kernel module:**
   ```bash
   sudo insmod uncached_mem.ko
   ```

2. **Check available interfaces:**
   ```bash
   ls /sys/kernel/uncached_mem/
   # command  status  uncached_addr  cached_addr  size_info
   ls -l /dev/uncached_mem
   ```

3. **Check module status:**
   ```bash
   cat /sys/kernel/uncached_mem/status
   ```

4. **Allocate memory with size parameters:**
   ```bash
   # Allocate 8KB uncached memory
   echo "0 8K" | sudo tee /sys/kernel/uncached_mem/command
   
   # Allocate 4MB cached memory
   echo "1 4M" | sudo tee /sys/kernel/uncached_mem/command
   
   # Check current status
   cat /sys/kernel/uncached_mem/status
   
   # Free all memory
   echo "2" | sudo tee /sys/kernel/uncached_mem/command
   ```

5. **Run comprehensive timing tests:**
   ```bash
   ./timing_test
   ```

6. **Unload the kernel module:**
   ```bash
   sudo rmmod uncached_mem
   ```

### Running Tests Without Sudo

For convenience during testing, you can set permissions to allow non-root access to the module interfaces:

**Option 1: Set permissions temporarily (testing only)**
```bash
# Load module first
sudo insmod uncached_mem.ko

# Set permissions for both device and sysfs interfaces
sudo chmod 666 /dev/uncached_mem
sudo chmod 666 /sys/kernel/uncached_mem/command

# Now you can run tests without sudo
echo "0 4K" > /sys/kernel/uncached_mem/command
./timing_test
echo "2" > /sys/kernel/uncached_mem/command
```

**Option 2: Use group permissions**
```bash
# Check current permissions
ls -la /dev/uncached_mem
ls -la /sys/kernel/uncached_mem/command

# Add your user to appropriate group (usually root or adm)
sudo usermod -a -G adm $USER
# Log out and back in for group change to take effect
```

**Option 3: Use sudo only for module operations**
```bash
# Load module with sudo
sudo insmod uncached_mem.ko

# Set permissions once
sudo chmod 666 /dev/uncached_mem && sudo chmod 666 /sys/kernel/uncached_mem/command

# Run tests normally
make timing_test
./timing_test

# Cleanup with sudo
sudo rmmod uncached_mem
```

### Advanced Usage Examples

```bash
# Test different allocation sizes
echo "0 4K" > /sys/kernel/uncached_mem/command    # Small allocation (__get_free_pages)
echo "1 2M" > /sys/kernel/uncached_mem/command    # Large allocation (vmalloc)  
echo "2" > /sys/kernel/uncached_mem/command       # Free all

# Test very large allocation (may take time)
echo "0 512M" > /sys/kernel/uncached_mem/command  # 512MB uncached
echo "2" > /sys/kernel/uncached_mem/command       # Free

# Check allocation details
cat /sys/kernel/uncached_mem/size_info
cat /sys/kernel/uncached_mem/uncached_addr
cat /sys/kernel/uncached_mem/cached_addr
```

### Memory Access via mmap

The timing test program uses mmap to directly access kernel-allocated memory:
- **Device file**: `/dev/uncached_mem`
- **Offset 0**: Maps to uncached memory buffer
- **Offset 1**: Maps to cached memory buffer
- **Variable size**: Automatically detects allocated buffer size

### Convenient Make Targets

```bash
# Full automated test sequence
make full_test

# Test caching functionality specifically
make cache_test

# Load module
make load

# Unload module
make unload

# View module information
make info

# View kernel messages
make dmesg
```

### Manual Testing with Sysfs

You can interact with the kernel module directly through the sysfs interface:

```bash
# Check module status and help
cat /sys/kernel/uncached_mem/command  # Shows command help
cat /sys/kernel/uncached_mem/status   # Shows current allocations

# Allocate different sizes
echo "0 8192" | sudo tee /sys/kernel/uncached_mem/command     # 8KB uncached
echo "1 1M" | sudo tee /sys/kernel/uncached_mem/command       # 1MB cached

# Check allocation details  
cat /sys/kernel/uncached_mem/status
cat /sys/kernel/uncached_mem/size_info
cat /sys/kernel/uncached_mem/uncached_addr
cat /sys/kernel/uncached_mem/cached_addr

# Free all allocated memory
echo "2" | sudo tee /sys/kernel/uncached_mem/command
```

## Expected Performance Results

Typical performance measurements show significant differences based on allocation size and method:

### Small Allocations (4KB-1MB, using __get_free_pages)
- **User space cached**: ~1.3 ns per access
- **Kernel cached (mmap)**: ~1.3 ns per access
- **Kernel uncached (mmap)**: ~70 ns per access
- **Performance ratio**: Uncached is approximately **50-70x slower**

### Large Allocations (>1MB, using vmalloc)  
- **Kernel cached (vmalloc)**: ~1.5 ns per access
- **Kernel uncached (vmalloc)**: ~80-100 ns per access
- **Performance ratio**: Uncached is approximately **50-70x slower**
- **Additional overhead**: Slight increase due to non-contiguous memory

### Very Large Allocations (>100MB)
- **Allocation time**: May take several seconds
- **Memory pressure**: Can impact system performance during allocation
- **Success rate**: Depends on available memory and fragmentation

### DMA/CMA Allocation (cma_cache):
- Large block efficiency: High throughput for sequential access
- Physical contiguity: Zero fragmentation for device DMA
- Cache control overhead: Similar per-page costs
- Memory pressure: May trigger reclaim for large allocations
- **Recent improvement**: 33.6x performance difference (6.4 ns cached vs 215.8 ns uncached)

### 🔧 Recent CMA Improvements

The CMA module has been significantly improved to address common allocation issues:

**Problems Fixed:**
- ❌ "allocation not found" errors due to hardcoded allocation IDs
- ❌ Large allocation failures (16MB, 64MB) causing memory exhaustion  
- ❌ Inconsistent test results due to leftover allocations

**Solutions Implemented:**
- ✅ Dynamic allocation ID tracking - automatically detects successful allocations
- ✅ Conservative test sizes (1M, 2M, 4M) - more reliable on various systems
- ✅ Comprehensive diagnostic tool (`cma_debug`) - helps troubleshoot CMA issues
- ✅ Better error handling and cleanup - prevents test interference

**Performance Verification:**
The improved test now consistently shows the expected performance difference:
```
Cached memory:   6.43 ns/access
Uncached memory: 215.82 ns/access  
Performance ratio: 33.6x slower for uncached
```

For troubleshooting any remaining issues, see `TROUBLESHOOTING.md` or run `./cma_debug all`.

## Technical Details

### Memory Allocation with Variable Size Support

The kernel module supports two independent memory allocation types with sophisticated size handling:

1. **Uncached Memory** (Command `0 [size]`): 
   - **Small allocations** (<1MB): Uses `__get_free_pages()` with `set_memory_uc()`
   - **Large allocations** (≥1MB): Uses `vmalloc()` with per-page `set_memory_uc()`
   - **Automatic method selection** based on size for optimal success rate
   - **Size validation** with minimum 4KB and maximum 128MB limits
   - **Page alignment** for proper memory management

2. **Cached Memory** (Command `1 [size]`): 
   - **Small allocations** (≤4KB): Uses `kmalloc()`
   - **Large allocations** (>4KB): Uses `vmalloc()`
   - **Utilizes CPU cache hierarchy** for optimal performance
   - **Automatic method selection** for reliability

### Modern Sysfs Interface

The module provides a clean sysfs interface that:
- **Replaces proc filesystem** with modern kernel practices
- **Multiple attribute files** for different information types
- **Size parameter parsing** with K/M/G suffix support
- **Comprehensive status reporting** with allocation method details
- **Read-only information files** for monitoring

### Memory Mapping with Size Awareness

The module provides enhanced mmap functionality:
- **Character device interface** at `/dev/uncached_mem`
- **Variable buffer size support** automatically detected
- **Different mapping methods** for vmalloc vs __get_free_pages memory
- **Proper caching attribute preservation** in user space
- **Support for very large mappings** up to allocated buffer size

### Size Parameter Processing

The module includes robust size parsing:
```c
// Supported formats:
"4096"      // Bytes
"4K"        // Kilobytes  
"1M"        // Megabytes
"1G"        // Gigabytes (with limitations)
```

- **Suffix support**: K (1024), M (1024²), G (1024³)
- **Input validation**: Range checking and format validation
- **Page alignment**: Automatic rounding up to page boundaries
- **Error handling**: Clear error messages for invalid inputs

### Architecture Support and Limitations

- **x86/x86_64**: Full support with `set_memory_uc()` for uncached allocation
- **Other architectures**: Graceful fallback to normal allocation with warnings
- **Large allocation support**: vmalloc enables allocations beyond physical contiguity limits
- **Memory pressure handling**: Automatic fallback and clear error reporting

### Allocation Method Selection Logic

```
Size < 4KB:           Error (below minimum)
4KB ≤ Size ≤ 1MB:     __get_free_pages / kmalloc (contiguous)
1MB < Size ≤ 128MB:   vmalloc (non-contiguous)  
Size > 128MB:         Error (above maximum)
```

**Special case for 1GB requests:**
- Uses vmalloc approach
- May succeed depending on system memory
- Non-contiguous physical pages
- Each page individually marked uncached
- Potential for system impact during allocation

## Expected Results

When running the timing tests, you should observe:

- **User space cached**: ~1.3 ns per access
- **Kernel cached (mmap)**: ~1.3 ns per access  
- **Kernel uncached (mmap)**: ~70 ns per access
- **Performance difference**: 50-70x slower for uncached vs cached memory

## Troubleshooting

### Common Issues

1. **Module loading fails:**
   ```bash
   # Check kernel headers are installed
   ls /lib/modules/$(uname -r)/build
   
   # Check for compilation errors
   make clean && make all
   
   # Check kernel log for errors  
   dmesg | tail -20
   ```

2. **Sysfs interface not available:**
   ```bash
   # Check if interface exists
   ls /sys/kernel/uncached_mem/
   
   # If missing, check module load status
   lsmod | grep uncached_mem
   
   # Check permissions
   ls -l /sys/kernel/uncached_mem/
   ```

3. **Large allocation failures:**
   ```bash
   # Check available memory
   free -h
   cat /proc/meminfo | grep -E "(MemFree|MemAvailable)"
   
   # Try smaller size first
   echo "0 256M" > /sys/kernel/uncached_mem/command
   
   # Check kernel messages for details
   dmesg | tail -10
   ```

4. **Invalid command format:**
   ```bash
   # Correct format examples:
   echo "0 4K" > /sys/kernel/uncached_mem/command    # 4KB uncached
   echo "1 1M" > /sys/kernel/uncached_mem/command    # 1MB cached  
   echo "2" > /sys/kernel/uncached_mem/command       # Free all
   
   # Check command help:
   cat /sys/kernel/uncached_mem/command
   ```

5. **mmap failures:**
   ```bash
   # Ensure memory is allocated before mmap
   echo "0 4K" > /sys/kernel/uncached_mem/command
   ./timing_test
   
   # Check device file exists
   ls -l /dev/uncached_mem
   
   # Check device permissions
   sudo chmod 666 /dev/uncached_mem
   ```

6. **1GB allocation issues:**
   ```bash
   # Monitor memory during allocation
   watch -n 1 'free -h'
   
   # Check for vmalloc space
   cat /proc/vmallocinfo | tail -10
   
   # Try progressive sizes
   echo "0 256M" > /sys/kernel/uncached_mem/command  # Test 256MB first
   echo "2" > /sys/kernel/uncached_mem/command       # Free
   echo "0 512M" > /sys/kernel/uncached_mem/command  # Test 512MB
   echo "2" > /sys/kernel/uncached_mem/command       # Free  
   echo "0 1G" > /sys/kernel/uncached_mem/command    # Try 1GB
   ```

### Debugging

- **Kernel messages**: Use `dmesg | tail -20` to view allocation details and errors
- **Module status**: Check `/sys/kernel/uncached_mem/status` for current allocations
- **Size information**: Check `/sys/kernel/uncached_mem/size_info` for allocation methods
- **Module loading**: Verify with `lsmod | grep uncached_mem`
- **Memory monitoring**: Use `free -h` and `cat /proc/meminfo` during large allocations
- **Allocation progress**: Monitor dmesg output during large vmalloc operations

### Size Limit Recommendations

| System RAM | Recommended Max | Notes |
|------------|----------------|-------|
| <2GB | 64MB | Conservative limit |
| 2-4GB | 128MB | Default maximum |  
| 4-8GB | 256MB | Moderate usage |
| >8GB | 512MB-1GB | Monitor memory pressure |

### Performance Optimization

- **Test smaller sizes first** to verify module functionality
- **Monitor system load** during large allocations
- **Use appropriate allocation size** for your testing needs
- **Consider memory fragmentation** on long-running systems
- **Free allocations promptly** to reduce memory pressure

**For detailed technical explanations of gotchas and API limitations, see [DESIGN.md](DESIGN.md).**

## Educational Value

This project demonstrates:
- **Modern Linux kernel module development** with sysfs interface
- **Variable size memory management** and allocation strategies in kernel space
- **Large memory allocation techniques** using vmalloc for non-contiguous memory
- **CPU cache behavior and performance implications** with real measurements
- **Sysfs filesystem interface creation** and modern kernel practices
- **Character device creation** for mmap functionality
- **Memory mapping between kernel and user space** with size awareness
- **Architecture-specific memory caching control** with fallback handling
- **High-precision timing measurement techniques** for performance analysis
- **Real-world performance testing methodologies** with variable parameters
- **Size parameter parsing and validation** in kernel modules
- **Error handling and system resource management** in kernel code

**For detailed technical explanations, API usage, and gotchas, see [DESIGN.md](DESIGN.md).**

## Safety Notes

- **Memory allocation limits**: 128MB default maximum to prevent system instability
- **Large allocation warnings**: System impact notifications for >512MB allocations
- **Automatic cleanup**: Module properly handles cleanup on both normal and error exits
- **Memory leak prevention**: Tracking of allocation methods and sizes for proper cleanup
- **Size validation**: Input validation prevents invalid or dangerous allocations
- **Development environment**: Designed for development/testing, not production use
- **System monitoring**: Includes comprehensive status reporting for safe usage
- **Progressive testing**: Recommends testing smaller sizes before attempting large allocations

## Sample Output

```bash
$ ./timing_test
Memory Access Timing Test - Sysfs Interface with Variable Size Support
=====================================================================
Using sysfs interface: /sys/kernel/uncached_mem/command
Using device file for mmap: /dev/uncached_mem

--- Initial Module Status ---
Uncached Memory Module Status
============================
Uncached: not allocated at (null) (0 bytes)
Cached: not allocated at (null) (0 bytes)
Allocation methods: uncached=__get_free_pages, cached=kmalloc

--- Baseline: User Space Results ---
User cached (malloc): 1.34 ns per access (total: 8582064 ns)

--- Test 1: Kernel Uncached Memory via mmap ---
Allocating uncached memory with size 4096 bytes...
Successfully mapped uncached memory to user space at 0x7f02a8bbc000 (4096 bytes)
Kernel uncached (mmap): 74.01 ns per access (total: 473692380 ns)
Unmapped kernel memory

--- Test 2: Kernel Cached Memory via mmap ---
Allocating cached memory with size 4096 bytes...
Successfully mapped cached memory to user space at 0x7f02a8bbc000 (4096 bytes)
Kernel cached (mmap): 1.32 ns per access (total: 8462089 ns)
Unmapped kernel memory

--- Test 3: Large Allocation Test (1MB) ---
Testing 1MB uncached allocation...
Successfully mapped uncached memory to user space at 0x7f02a8bbc000 (1048576 bytes)
1MB uncached (mmap): 76.23 ns per access (total: 487851234 ns)
Unmapped kernel memory

=== PERFORMANCE SUMMARY ===
User space cached:     1.34 ns per access
Kernel uncached (mmap): 74.01 ns per access
Kernel cached (mmap):   1.32 ns per access

Performance ratio: Uncached is 56.0x slower than cached

--- Final Module Status ---
Uncached Memory Module Status
============================
Uncached: not allocated at (null) (0 bytes)
Cached: not allocated at (null) (0 bytes)
Allocation methods: uncached=vmalloc, cached=vmalloc

Sysfs Interface Usage:
- Commands: echo 'cmd [size]' > /sys/kernel/uncached_mem/command
- Status:   cat /sys/kernel/uncached_mem/status
- Examples: echo '0 1M' > command (1MB uncached)
           echo '1 512K' > command (512KB cached)
           echo '2' > command (free all)

This test uses sysfs interface with variable size support
Large allocations (>1MB) use vmalloc instead of __get_free_pages
Maximum allocation size: 128MB
```

## License

This project is released under the GPL license, compatible with the Linux kernel.

## Version History

- **v2.0**: Sysfs interface with variable size support, vmalloc for large allocations
- **v1.0**: Original proc-based interface with fixed 4KB allocations

## Interface Transition Notes

This version (v2.0) has migrated from the `/proc` filesystem interface to the modern `sysfs` interface:

### Changes from v1.0:
- **Control interface**: `/proc/uncached_mem` → `/sys/kernel/uncached_mem/command`
- **Status information**: Single proc read → Multiple sysfs attribute files
- **Size support**: Fixed 4KB → Variable size with K/M/G suffixes
- **Memory mapping**: Proc file mmap → Character device `/dev/uncached_mem`
- **Allocation methods**: Fixed __get_free_pages → Automatic method selection (vmalloc for large)

### Migration Guide:
```bash
# Old v1.0 interface:
echo "0" > /proc/uncached_mem
cat /proc/uncached_mem

# New v2.0 interface:  
echo "0 4K" > /sys/kernel/uncached_mem/command
cat /sys/kernel/uncached_mem/status
```

### Backward Compatibility:
- mmap functionality remains the same (offset 0/1 for uncached/cached)
- Command numbers unchanged (0=uncached, 1=cached, 2=free)
- Basic allocation behavior preserved for default sizes

---

# Dynamic Cache Control Module

## Overview

The `dynamic_cache` module provides advanced per-page cache control functionality, allowing individual pages to have their cache state toggled at runtime. This module is ideal for detailed cache behavior research and performance analysis.

## Key Features

- **Page-level cache control**: Individual pages can be cached/uncached dynamically
- **Variable size allocation**: Support for allocating 1 to many contiguous pages with size suffixes (K/M/G)
- **Block management**: Multi-page blocks with block-level operations (cache/uncache/free entire blocks)
- **Runtime cache toggling**: Change cache attributes after allocation
- **Page pool management**: Pre-allocated pool of 1024 pages (4MB total)
- **Visual page mapping**: See allocation state with visual indicators
- **Pattern testing**: Set and verify test patterns on specific pages
- **Comprehensive monitoring**: Real-time status and allocation tracking with block information

## Project Structure (Extended)

```
uncached/
├── README.md           # This documentation file
├── DESIGN.md          # Detailed design explanation (now includes dynamic_cache)
├── uncached_mem.c      # Static cache control kernel module
├── dynamic_cache.c     # Dynamic per-page cache control module  
├── Makefile           # Build system for both modules
├── timing_test.c      # Test program for uncached_mem module
├── dynamic_test.c     # Comprehensive test program for dynamic_cache module
└── dynamic_size_test.c # Variable size allocation test program
```

## Dynamic Cache Module Usage

### Loading the Module

```bash
# Build both modules
make all

# Load the dynamic cache module
sudo insmod dynamic_cache.ko

# Check if loaded successfully
lsmod | grep dynamic_cache
```

### Basic Operations

```bash
# Check module status
cat /sys/kernel/dynamic_cache/status

# Allocate a new page (returns page ID)
echo "alloc" > /sys/kernel/dynamic_cache/command

# Set page 0 as uncached
echo "uncache 0" > /sys/kernel/dynamic_cache/command

# Set page 0 as cached  
echo "cache 0" > /sys/kernel/dynamic_cache/command

# Toggle cache state of page 0
echo "toggle 0" > /sys/kernel/dynamic_cache/command

# Set test pattern on page 0
echo "pattern 0 AA" > /sys/kernel/dynamic_cache/command

# View page allocation map
cat /sys/kernel/dynamic_cache/page_map

# Free the page
echo "free 0" > /sys/kernel/dynamic_cache/command
```

### Variable Size Allocation

The dynamic_cache module supports allocating contiguous blocks of multiple pages:

```bash
# Single page allocation (default)
echo "alloc" > /sys/kernel/dynamic_cache/command          # 4K (1 page)
echo "alloc 4K" > /sys/kernel/dynamic_cache/command       # Explicit 4K

# Multi-page block allocation (returns block ID)  
echo "alloc 8K" > /sys/kernel/dynamic_cache/command       # 2 pages
echo "alloc 64K" > /sys/kernel/dynamic_cache/command      # 16 pages
echo "alloc 1M" > /sys/kernel/dynamic_cache/command       # 256 pages
echo "alloc 4194304" > /sys/kernel/dynamic_cache/command  # 4M in bytes

# Size specifications:
# - Raw bytes: automatically rounded up to page boundary
# - K suffix: kilobytes (e.g., 16K = 16384 bytes)
# - M suffix: megabytes (e.g., 2M = 2097152 bytes)  
# - G suffix: gigabytes (e.g., 1G = 1073741824 bytes)
# - Limits: 4K minimum, 64M maximum

# Check allocation status (shows blocks and individual pages)
cat /sys/kernel/dynamic_cache/status
```

**Block vs Page Operations:**
```bash
# Block operations (affect all pages in a block)
echo "cache_block 1" > /sys/kernel/dynamic_cache/command    # Cache entire block
echo "uncache_block 1" > /sys/kernel/dynamic_cache/command  # Uncache entire block  
echo "free_block 1" > /sys/kernel/dynamic_cache/command     # Free entire block

# Individual page operations (work within blocks too)
echo "cache 5" > /sys/kernel/dynamic_cache/command          # Cache specific page
echo "uncache 5" > /sys/kernel/dynamic_cache/command        # Uncache specific page
echo "free 5" > /sys/kernel/dynamic_cache/command           # Free specific page
```

**Example Status Output:**
```
Active blocks: 2

Active Blocks:
Block ID  Start  Pages  Size
--------  -----  -----  ----
       1      0      2    8K
       2      8     64  256K

Allocated Pages:
ID   Virtual     PFN        Block    Cache State
---  ----------  ---------  -------  -----------
  0  ffff888...  123456789        1  CACHED
  1  ffff888...  123456790        1  UNCACHED
  8  ffff888...  123456798        2  CACHED
...
```

### Permission Requirements (Dynamic Cache)

For running tests without sudo, set appropriate permissions:

```bash
# Set permissions for both device and sysfs interfaces
sudo chmod 666 /dev/dynamic_cache
sudo chmod 666 /sys/kernel/dynamic_cache/command

# Now you can run operations without sudo
echo "alloc" > /sys/kernel/dynamic_cache/command
./dynamic_test
```

### Sysfs Interface (Dynamic Cache)

- **`/sys/kernel/dynamic_cache/command`** - Main control interface
- **`/sys/kernel/dynamic_cache/status`** - Detailed allocation status
- **`/sys/kernel/dynamic_cache/page_map`** - Visual allocation map
- **`/dev/dynamic_cache`** - Character device for memory mapping

### Available Commands

| Command | Description | Example |
|---------|-------------|---------|
| `alloc` | Allocate a new page | `echo "alloc" > command` |
| `free <id>` | Free a specific page | `echo "free 5" > command` |
| `cache <id>` | Set page as cached | `echo "cache 3" > command` |
| `uncache <id>` | Set page as uncached | `echo "uncache 3" > command` |
| `toggle <id>` | Toggle cache state | `echo "toggle 3" > command` |
| `pattern <id> <val>` | Set test pattern | `echo "pattern 3 FF" > command` |

### Running Tests

```bash
# Build and run dynamic cache tests
make dynamic_test
sudo insmod dynamic_cache.ko
./dynamic_test

# Or use the automated test target
make dynamic_test_run

# Test both modules
make test_all
```

### Memory Mapping Individual Pages (Fixed Implementation)

The dynamic cache module allows mapping individual pages with specific cache attributes. **Important**: Use correct offset calculation for mmap.

#### Correct Usage Example:
```c
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>  // for getpagesize()

int fd = open("/dev/dynamic_cache", O_RDWR);
if (fd < 0) {
    perror("Failed to open device");
    return -1;
}

// CRITICAL: Calculate offset correctly
// For page N: offset = N * PAGE_SIZE (in bytes)
int page_id = 5;
off_t offset = page_id * getpagesize();  // 5 * 4096 = 20480

// Map page 5 with correct offset
void *addr = mmap(NULL, getpagesize(), PROT_READ|PROT_WRITE, MAP_SHARED, fd, offset);
if (addr == MAP_FAILED) {
    perror("mmap failed");
    close(fd);
    return -1;
}

printf("Page %d mapped at %p\n", page_id, addr);

// Use the memory
volatile uint64_t *data = (volatile uint64_t *)addr;
*data = 0x1234567890ABCDEF;
printf("Wrote: 0x%016lx\n", *data);

// Clean up
munmap(addr, getpagesize());
close(fd);
```

#### Common mmap Mistakes to Avoid:
```c
// ❌ WRONG: Don't pass page ID directly as offset
mmap(NULL, 4096, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 5);  // FAILS

// ✅ CORRECT: Calculate byte offset
mmap(NULL, 4096, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 5 * 4096);  // Works
```

### Complete Dynamic Cache Workflow

#### 1. Module Setup
```bash
# Build and load module
make all
sudo insmod dynamic_cache.ko

# Verify loading
dmesg | tail -5
ls -la /sys/kernel/dynamic_cache/
ls -la /dev/dynamic_cache
```

#### 2. Allocate and Configure Pages
```bash
# Allocate several pages
echo "alloc" > /sys/kernel/dynamic_cache/command  # Page 0
echo "alloc" > /sys/kernel/dynamic_cache/command  # Page 1
echo "alloc" > /sys/kernel/dynamic_cache/command  # Page 2

# Configure cache states
echo "uncache 0" > /sys/kernel/dynamic_cache/command  # Page 0 = uncached
echo "cache 1" > /sys/kernel/dynamic_cache/command    # Page 1 = cached
echo "toggle 2" > /sys/kernel/dynamic_cache/command   # Page 2 = uncached

# Set test patterns
echo "pattern 0 AA" > /sys/kernel/dynamic_cache/command
echo "pattern 1 55" > /sys/kernel/dynamic_cache/command
echo "pattern 2 FF" > /sys/kernel/dynamic_cache/command
```

#### 3. Monitor Status
```bash
# Check detailed status
cat /sys/kernel/dynamic_cache/status

# View allocation map (visual representation)
cat /sys/kernel/dynamic_cache/page_map
# Output: UCS... (U=uncached, C=cached, S=uncached, .=free)
```

#### 4. Test with User Program
```c
// test_multi_pages.c
#include <stdio.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stdint.h>

int main() {
    int fd = open("/dev/dynamic_cache", O_RDWR);
    
    // Map multiple pages with different cache states
    void *page0 = mmap(NULL, 4096, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0 * 4096);  // Uncached
    void *page1 = mmap(NULL, 4096, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 1 * 4096);  // Cached
    void *page2 = mmap(NULL, 4096, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 2 * 4096);  // Uncached
    
    volatile uint64_t *data0 = (volatile uint64_t *)page0;
    volatile uint64_t *data1 = (volatile uint64_t *)page1; 
    volatile uint64_t *data2 = (volatile uint64_t *)page2;
    
    // Test performance difference
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < 1000000; i++) {
        *data1 = i;  // Cached - should be fast
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double cached_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < 1000000; i++) {
        *data0 = i;  // Uncached - should be slow
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double uncached_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    printf("Cached time: %.6f sec\n", cached_time);
    printf("Uncached time: %.6f sec\n", uncached_time);
    printf("Ratio: %.1fx slower\n", uncached_time / cached_time);
    
    munmap(page0, 4096);
    munmap(page1, 4096);
    munmap(page2, 4096);
    close(fd);
}
```

#### 5. Runtime Cache Control
```bash
# Change cache state of mapped pages
echo "cache 0" > /sys/kernel/dynamic_cache/command    # Convert page 0 to cached
echo "uncache 1" > /sys/kernel/dynamic_cache/command  # Convert page 1 to uncached

# Test performance again - should see reversed results
```

#### 6. Cleanup
```bash
# Free specific pages
echo "free 0" > /sys/kernel/dynamic_cache/command
echo "free 1" > /sys/kernel/dynamic_cache/command
echo "free 2" > /sys/kernel/dynamic_cache/command

# Or unload module (auto-cleanup)
sudo rmmod dynamic_cache
```

### Variable Size Allocation Testing

The `dynamic_size_test.c` program demonstrates variable size allocation and block management:

```bash
# Build and run the size test
make dynamic_size_test
sudo insmod dynamic_cache.ko
sudo chmod 666 /dev/dynamic_cache /sys/kernel/dynamic_cache/command
./dynamic_size_test
```

**Example Test Sequence:**
```bash
# Test various block sizes
echo "alloc 4K" > /sys/kernel/dynamic_cache/command     # Single page
echo "alloc 64K" > /sys/kernel/dynamic_cache/command    # 16-page block  
echo "alloc 1M" > /sys/kernel/dynamic_cache/command     # 256-page block

# Check allocation status
cat /sys/kernel/dynamic_cache/status

# Block-level operations
echo "uncache_block 2" > /sys/kernel/dynamic_cache/command  # Uncache 1M block
echo "cache_block 2" > /sys/kernel/dynamic_cache/command    # Re-cache 1M block

# Individual page operations within blocks
echo "uncache 16" > /sys/kernel/dynamic_cache/command      # Uncache one page in 64K block

# Free blocks
echo "free_block 1" > /sys/kernel/dynamic_cache/command    # Free 64K block
echo "free_block 2" > /sys/kernel/dynamic_cache/command    # Free 1M block
```

# DMA/CMA Cache Control Module

## Overview

The `cma_cache` module demonstrates large contiguous memory allocation using DMA coherent allocation (which may use CMA backend when available). This module is ideal for understanding device driver memory management and situations requiring guaranteed physical contiguity.

## Key Features

- **Large contiguous allocations**: 1MB to 256MB blocks
- **DMA coherent memory**: Uses kernel's DMA allocation subsystem
- **Physical contiguity**: Guaranteed contiguous physical memory
- **Cache attribute control**: Set allocated blocks as cached or uncached
- **Device integration**: Demonstrates platform device and DMA API usage
- **Memory mapping**: Map large blocks to user space for testing

## CMA Module Usage

### Loading the Module

```bash
# Build all modules
make all

# Check CMA memory availability first (recommended for large allocations)
grep -E 'Cma(Total|Free)' /proc/meminfo
# If CmaTotal < 64MB, consider configuring GRUB (see CMA Configuration section below)

# Load the CMA cache module
sudo insmod cma_cache.ko

# Check if loaded successfully
lsmod | grep cma_cache
```

### Basic Operations

```bash
# Check module status
cat /sys/kernel/cma_cache/status

# Allocate DMA memory blocks (returns allocation ID)
echo "alloc 1M" > /sys/kernel/cma_cache/command     # 1MB block
echo "alloc 16M" > /sys/kernel/cma_cache/command    # 16MB block
echo "alloc 64M" > /sys/kernel/cma_cache/command    # 64MB block

# For larger allocations (>50MB), see "CMA Memory Configuration" section below
# if you encounter "Failed to allocate" errors

# Set cache attributes
echo "uncache 1" > /sys/kernel/cma_cache/command    # Set allocation 1 as uncached
echo "cache 1" > /sys/kernel/cma_cache/command      # Set allocation 1 as cached
echo "toggle 2" > /sys/kernel/cma_cache/command     # Toggle cache state

# Free allocations
echo "free 1" > /sys/kernel/cma_cache/command
echo "free 2" > /sys/kernel/cma_cache/command
```

### Size Specifications

```bash
# Various size formats supported
echo "alloc 1048576" > /sys/kernel/cma_cache/command  # 1MB in bytes
echo "alloc 4M" > /sys/kernel/cma_cache/command       # 4MB with suffix
echo "alloc 64M" > /sys/kernel/cma_cache/command      # 64MB with suffix
echo "alloc 1G" > /sys/kernel/cma_cache/command       # 1GB (if system supports)

# Size limits: 1MB minimum, 256MB maximum
# All sizes automatically aligned to page boundaries
```

### Permission Requirements (CMA Cache)

For running tests without sudo, set appropriate permissions:

```bash
sudo chmod 666 /dev/cma_cache
sudo chmod 666 /sys/kernel/cma_cache/command
```

### Status Display

The status interface provides comprehensive allocation information:

```
DMA Cache Control Status
========================
Total allocations: 3/32
Total allocated memory: 81920 bytes (80 MB)
Cached allocations: 2
Uncached allocations: 1

Active Allocations:
ID   Size       Pages  Virtual     Physical    Cache State
---  ---------  -----  ----------  ----------  -----------
  1      1024K    256  ffff888...  12345000    CACHED
  2     16384K   4096  ffff889...  15678000    UNCACHED  
  3     65536K  16384  ffff88a...  20000000    CACHED

Size limits: 1024K - 262144K
```

### Memory Mapping Large Blocks

```c
#include <sys/mman.h>
#include <fcntl.h>

// Open device file
int fd = open("/dev/cma_cache", O_RDWR);

// Map allocation 1 (1MB block)
// Note: offset = allocation_id * PAGE_SIZE for identification
void *addr = mmap(NULL, 1024*1024, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 1);

// Access the large contiguous memory block
volatile uint64_t *data = (volatile uint64_t *)addr;
for (int i = 0; i < 131072; i++) {  // 1MB / 8 bytes
    data[i] = 0x1234567890ABCDEF + i;
}

// Unmap and cleanup
munmap(addr, 1024*1024);
close(fd);
```

### CMA Test Program

```bash
# Build and run the CMA test program
make cma_test
sudo insmod cma_cache.ko
sudo chmod 666 /dev/cma_cache /sys/kernel/cma_cache/command

# Run different test modes
./cma_test basic       # Basic allocation and cache control tests
./cma_test performance # Performance comparison tests
./cma_test stress      # Stress testing with multiple large allocations
./cma_test all         # Run all test modes
```

**Example Test Output:**
```
CMA Cache Control Test Program
==============================

=== Basic Allocation Test ===
Allocating 1M...
Allocating 4M...
Allocating 16M...
Allocating 64M...

=== Cache Control Test ===
Setting allocation 1 as uncached...
Setting allocation 2 as uncached...
Toggling allocation 3 cache state...

=== Performance Test ===
Testing cached memory performance...
Testing uncached memory performance...

Performance Results (1000000 iterations):
Cached memory:   0.452000 seconds (452.00 ns/access)
Uncached memory: 2.840000 seconds (2840.00 ns/access)
Performance ratio: 6.3x slower for uncached
```

## Performance Comparison

All three modules provide performance insights for different use cases:

### Static Cache Control (uncached_mem)
- **Use case**: Basic cache behavior demonstration
- **Granularity**: Entire allocation (4KB to 128MB)
- **Performance**: One-time setup cost
- **Interface**: Simple alloc/free commands
- **Best for**: Educational understanding of cache impact

### Dynamic Cache Control (dynamic_cache)  
- **Use case**: Advanced cache research and analysis
- **Granularity**: Individual pages (4KB each) + variable-size blocks
- **Performance**: Runtime toggle costs (~100μs per page)
- **Interface**: Command-based with real-time control
- **Best for**: Fine-grained cache behavior research

### DMA/CMA Cache Control (cma_cache)
- **Use case**: Large contiguous memory management
- **Granularity**: Large blocks (1MB to 256MB)
- **Performance**: Guaranteed physical contiguity
- **Interface**: DMA allocation with cache control
- **Best for**: Device driver development, DMA buffer management

### Static Cache Control (uncached_mem)
- **Use case**: Basic cache behavior demonstration
- **Granularity**: Entire allocation (4KB-128MB)
- **Performance**: One-time setup cost
- **Interface**: Simple alloc/free commands

### Dynamic Cache Control (dynamic_cache)  
- **Use case**: Advanced cache research and analysis
- **Granularity**: Individual pages (4KB each)
- **Performance**: Runtime toggle costs (~100μs per page)
- **Interface**: Command-based with real-time control

### Typical Results
```
Static Allocation (uncached_mem):
- Cached access:    ~1.3 ns per operation
- Uncached access:  ~70 ns per operation
- Performance ratio: 50-70x slower for uncached

Dynamic Control (dynamic_cache):
- Cache toggle time: ~100 μs per page
- TLB flush impact: ~10 μs per page
- Per-operation timing: Similar to static

DMA/CMA Allocation (cma_cache):
- Large block efficiency: High throughput for sequential access
- Physical contiguity: Zero fragmentation for device DMA
- Cache control overhead: Similar per-page costs
- Memory pressure: May trigger reclaim for large allocations
- **Recent improvement**: 33.6x performance difference (6.4 ns cached vs 215.8 ns uncached)
```

## CMA Quick Start Guide

### For Large Allocation Testing (>50MB)

**1. Check current CMA:**
```bash
grep -E 'Cma(Total|Free)' /proc/meminfo
```

**2. If CmaTotal < 128MB, configure GRUB:**
```bash
sudo nano /etc/default/grub
# Add: GRUB_CMDLINE_LINUX="... cma=256M movable_node"
# Add: GRUB_CMDLINE_LINUX_DEFAULT="... cma=256M@node0 for NUMA
sudo update-grub && sudo reboot
```

**3. Test large allocations:**
```bash
make all && sudo insmod cma_cache.ko
sudo chmod 666 /dev/cma_cache /sys/kernel/cma_cache/command
echo "alloc 128M" > /sys/kernel/cma_cache/command
cat /sys/kernel/cma_cache/status
```

### NUMA Targeting (Multi-node systems)

**1. Check NUMA topology:**
```bash
ls /sys/devices/system/node/  # Shows available nodes
```

**2. Configure NUMA-specific CMA:**
```bash
# In GRUB: cma=256M@node0,256M@node1
```

### Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| "Failed to allocate X bytes" | CMA pool too small | Increase `cma=` in GRUB |
| Allocation on wrong NUMA node | No NUMA targeting | Use `cma=SIZE@nodeN` |
| Memory fragmentation | High system load | Add `movable_node` parameter |

# Kernel Module for Uncached Memory Allocation with Sysfs Interface

This project demonstrates the implementation of a Linux kernel module that allocates both cached and uncached memory with variable size support, providing a modern sysfs interface for control and direct user-space access via mmap for performance testing.

## Project Structure

```
uncached/
├── README.md           # This documentation file
├── DESIGN.md          # Detailed design explanation and API reference
├── uncached_mem.c      # Kernel module source code with sysfs interface
├── Makefile           # Build configuration for kernel module and test program
└── timing_test.c      # User space timing test program with sysfs support
```

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
- **Runtime cache toggling**: Change cache attributes after allocation
- **Page pool management**: Pre-allocated pool of 1024 pages (4MB total)
- **Visual page mapping**: See allocation state with visual indicators
- **Pattern testing**: Set and verify test patterns on specific pages
- **Comprehensive monitoring**: Real-time status and allocation tracking

## Project Structure (Extended)

```
uncached/
├── README.md           # This documentation file
├── DESIGN.md          # Detailed design explanation (now includes dynamic_cache)
├── uncached_mem.c      # Static cache control kernel module
├── dynamic_cache.c     # Dynamic per-page cache control module  
├── Makefile           # Build system for both modules
├── timing_test.c      # Test program for uncached_mem module
└── dynamic_test.c     # Comprehensive test program for dynamic_cache module
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
sudo rmmod dynamic_cache

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
    return 0;
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

## Performance Comparison

Both modules provide performance insights:

### Static Cache Control (uncached_mem)
- **Use case**: Basic cache behavior demonstration
- **Granularity**: Entire allocation (4KB to 128MB)
- **Performance**: One-time setup cost
- **Interface**: Simple alloc/free commands

### Dynamic Cache Control (dynamic_cache)  
- **Use case**: Advanced cache research and analysis
- **Granularity**: Individual pages (4KB each)
- **Performance**: Runtime toggle costs (~100μs per page)
- **Interface**: Command-based with real-time control

### Typical Results
```
Static Allocation:
- Cached access:    ~1.3 ns per operation
- Uncached access:  ~70 ns per operation
- Performance ratio: 50-70x slower for uncached

Dynamic Control:
- Cache toggle time: ~100 μs per page
- TLB flush impact: ~10 μs per page
- Per-operation timing: Similar to static
```

## Use Cases

### Educational
- **OS Courses**: Understand kernel memory management
- **System Performance**: Learn cache behavior impact
- **Kernel Programming**: Modern kernel module development

### Research
- **Cache Analysis**: Detailed per-page cache behavior study
- **Performance Tuning**: Identify cache-sensitive operations  
- **Algorithm Testing**: Compare performance with different cache states

### Development
- **Driver Development**: Test memory mapping strategies
- **Embedded Systems**: Optimize cache usage for specific workloads
- **Performance Debugging**: Isolate cache-related performance issues

## Build Targets

The Makefile supports both modules:

```bash
make all              # Build both kernel modules
make timing_test      # Build static cache test program
make dynamic_test     # Build dynamic cache test program
make test             # Test uncached_mem module
make dynamic_test_run # Test dynamic_cache module  
make test_all         # Test both modules
make clean           # Clean all build artifacts
```

## Troubleshooting (Updated with Fixes)

### mmap Issues (Fixed in Current Version)

#### Problem: mmap fails with "Invalid argument"
**Symptoms:**
- `mmap()` returns `MAP_FAILED` 
- `errno = 22` (EINVAL)
- No kernel error messages in dmesg

**Solution (Fixed):**
```c
// ❌ WRONG: Don't pass page index directly
mmap(NULL, 4096, PROT_READ|PROT_WRITE, MAP_SHARED, fd, page_id);

// ✅ CORRECT: Pass byte offset
mmap(NULL, 4096, PROT_READ|PROT_WRITE, MAP_SHARED, fd, page_id * 4096);
```

**Root Cause:** mmap offset parameter must be in bytes, not page units.

#### Problem: Kernel module compilation fails
**Solution:**
```bash
# Install kernel headers
sudo apt-get install linux-headers-$(uname -r)

# Clean and rebuild
make clean && make all
```

#### Problem: Module loading fails
**Symptoms:**
- `insmod` fails with "Operation not permitted"
- "Unknown symbol" errors in dmesg

**Solution:**
```bash
# Check kernel version compatibility
uname -r
ls /lib/modules/$(uname -r)/build

# Rebuild for current kernel
make clean && make all

# Load with debugging
sudo insmod dynamic_cache.ko
dmesg | tail -10
```

### Module Loading Issues
```bash
# Check kernel logs
dmesg | tail -20

# Verify kernel headers
ls /lib/modules/$(uname -r)/build

# Check module dependencies
modinfo dynamic_cache.ko
```

### Permission Issues
```bash
# Ensure proper permissions for sysfs files
ls -la /sys/kernel/dynamic_cache/

# Check device file permissions  
ls -la /dev/dynamic_cache

# Run tests with appropriate privileges
sudo ./dynamic_test
```

### Memory Mapping Debugging
```bash
# Enable kernel debugging
echo "file mm/mmap.c +p" > /sys/kernel/debug/dynamic_debug/control  # (if available)

# Check our module's debug output
dmesg | grep -E "(mmap|dynamic_cache)"

# Verify page allocation
cat /sys/kernel/dynamic_cache/status
```

### Performance Issues
```bash
# Check system load
top
cat /proc/meminfo

# Monitor cache behavior
perf stat -e cache-misses,cache-references ./your_test_program

# Check for memory fragmentation (if using large allocations)
cat /proc/buddyinfo
```

### Architecture Support
- Cache control requires x86 architecture
- Some features may not be available on all systems
- Check dmesg for architecture-specific warnings

## Known Issues and Workarounds

### 1. Device Permissions
**Issue**: `/dev/dynamic_cache` created with root-only permissions.
**Workaround**: Run tests with `sudo` or add udev rules.
```bash
# Temporary fix
sudo chmod 666 /dev/dynamic_cache

# Permanent fix: Add udev rule
echo 'KERNEL=="dynamic_cache", MODE="0666"' | sudo tee /etc/udev/rules.d/99-dynamic-cache.rules
```

### 2. Memory Fragmentation (Large Allocations)
**Issue**: vmalloc may fail for very large memory pools.
**Workaround**: Reduce `MAX_PAGES` in module source if needed.

### 3. Architecture Limitations
**Issue**: Cache control not available on all architectures.
**Expected Behavior**: Module loads but cache operations may be no-ops.

## Build Targets (Updated)

The Makefile supports both modules and all tests:

```bash
make all              # Build both kernel modules
make timing_test      # Build static cache test program
make dynamic_test     # Build dynamic cache test program
make test             # Test uncached_mem module
make dynamic_test_run # Test dynamic_cache module (fully automated)
make test_all         # Test both modules
make clean           # Clean all build artifacts

# Individual module control
make load             # Load uncached_mem
make load_dynamic     # Load dynamic_cache
make unload           # Unload uncached_mem
make unload_dynamic   # Unload dynamic_cache
```

## Testing Strategy

### Automated Testing
```bash
# Quick test
make dynamic_test_run

# Comprehensive test with both modules
make test_all

# Manual testing with custom parameters
sudo insmod dynamic_cache.ko
echo "alloc" > /sys/kernel/dynamic_cache/command
# ... custom tests ...
sudo rmmod dynamic_cache
```

### Manual Verification Steps
1. **Module Loading**: Check `lsmod` and `dmesg`
2. **Interface Creation**: Verify sysfs and device files exist
3. **Page Allocation**: Test `alloc` command and check status
4. **Cache Control**: Test `cache`/`uncache` commands
5. **Memory Mapping**: Test mmap with correct offsets
6. **Performance**: Measure cached vs uncached access times
7. **Cleanup**: Verify proper resource cleanup on unload

## License

GPL v2 - See individual source files for details.

## Educational Value

This project demonstrates:
- **Modern kernel module development** with sysfs interfaces
- **Advanced memory management** techniques and trade-offs
- **CPU cache behavior** and performance implications  
- **Architecture-specific programming** with portable APIs
- **Comprehensive testing** and debugging practices

For detailed technical explanations, see `DESIGN.md`.


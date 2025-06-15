# Kernel Module for Uncached Memory Allocation and Direct Access

This project demonstrates the implementation of a Linux kernel module that allocates both cached and uncached memory, providing direct user-space access via mmap for performance testing and comparison.

## Project Structure

```
Kernel-mod/
├── README.md           # This documentation file
├── uncached_mem.c      # Kernel module source code
├── Makefile           # Build configuration for kernel module and test program
└── timing_test.c      # User space timing test program with direct memory access
```

## Overview

The project consists of three main components:

1. **`uncached_mem.c`** - A kernel module that:
   - Allocates cached memory using `kmalloc()`
   - Allocates uncached memory using `__get_free_pages()` with `set_memory_uc()`
   - Provides mmap interface for direct user-space access to kernel memory
   - Uses integer commands for simple control interface
   - Supports simultaneous allocation of both memory types
   - Maps memory to user space with proper caching attributes

2. **`timing_test.c`** - A user space test program that:
   - Directly accesses kernel-allocated memory via mmap
   - Measures actual performance differences between cached/uncached memory
   - Tests both memory types simultaneously
   - Provides nanosecond-precision timing measurements
   - Demonstrates real cache behavior impact

3. **`Makefile`** - Build system that:
   - Compiles the kernel module
   - Builds the test program with real-time library support
   - Provides convenient targets for testing and module management

## Interface Commands

The module uses simple integer commands via `/proc/uncached_mem`:

- **0** - `alloc_uc`: Allocate uncached memory
- **1** - `alloc_cached`: Allocate cached memory  
- **2** - `free`: Free all allocated memory

## Building and Installation

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

### Basic Usage

1. **Load the kernel module:**
   ```bash
   sudo insmod uncached_mem.ko
   ```

2. **Check module status:**
   ```bash
   cat /proc/uncached_mem
   ```

3. **Allocate memory using integer commands:**
   ```bash
   # Allocate uncached memory
   echo "0" | sudo tee /proc/uncached_mem
   
   # Allocate cached memory
   echo "1" | sudo tee /proc/uncached_mem
   
   # Free all memory
   echo "2" | sudo tee /proc/uncached_mem
   ```

4. **Run comprehensive timing tests:**
   ```bash
   ./timing_test
   ```

5. **Unload the kernel module:**
   ```bash
   sudo rmmod uncached_mem
   ```

### Memory Access via mmap

The timing test program uses mmap to directly access kernel-allocated memory:
- **Offset 0**: Maps to uncached memory buffer
- **Offset 1**: Maps to cached memory buffer

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

### Manual Testing

You can interact with the kernel module directly through the proc interface:

```bash
# Check module status
cat /proc/uncached_mem

# Allocate uncached memory
echo "0" | sudo tee /proc/uncached_mem

# Allocate cached memory
echo "1" | sudo tee /proc/uncached_mem

# Check status after allocation
cat /proc/uncached_mem

# Free all allocated memory
echo "2" | sudo tee /proc/uncached_mem
```

## Expected Performance Results

Typical performance measurements show significant differences:

- **User space cached**: ~1.3 ns per access
- **Kernel cached (mmap)**: ~1.3 ns per access
- **Kernel uncached (mmap)**: ~70 ns per access

**Performance ratio**: Uncached memory is approximately **50-70x slower** than cached memory.

## Technical Details

### Memory Allocation

The kernel module supports two independent memory allocation types:

1. **Uncached Memory** (Command 0): Uses `__get_free_pages()` with `set_memory_uc()` to:
   - Allocate page-aligned memory
   - Mark pages as uncached using architecture-specific functions
   - Bypass CPU caches for direct memory access
   - Provide consistent uncached behavior across architectures

2. **Cached Memory** (Command 1): Uses `kmalloc()` to:
   - Allocate regular kernel memory with normal caching
   - Utilize CPU cache hierarchy for optimal performance
   - Provide standard kernel memory allocation behavior

### Memory Mapping

The module provides mmap functionality that:
- Maps kernel memory directly to user space
- Preserves caching attributes (cached vs uncached)
- Uses different offsets to select buffer type:
  - Offset 0: Uncached buffer
  - Offset 1: Cached buffer
- Enables direct user-space access to kernel-allocated memory

### Architecture Support

- **x86/x86_64**: Full support with `set_memory_uc()` for uncached allocation
- **Other architectures**: Graceful fallback to normal allocation with warnings
- **Compatibility**: Works with Linux kernel 5.15 and similar versions
   - Is coherent between CPU and DMA devices
   - Provides both virtual and physical addresses
   - Demonstrates genuine uncached memory access patterns

2. **Cached Memory**: Uses `kmalloc()` to allocate memory that:
   - Uses normal kernel memory allocation
   - Benefits from CPU cache hierarchy
   - Provides baseline cached performance for comparison
   - Demonstrates typical kernel memory behavior

### Caching Control

The module provides dynamic caching control through proc interface commands:
- `cache_on` - Switch to cached memory mode
### Interface Commands

The module accepts three integer commands:

- `0` (alloc_uc) - Allocate uncached memory
- `1` (alloc_cached) - Allocate cached memory
- `2` (free) - Free all allocated memory

### Timing Measurements

The test program measures:
- **Sequential access**: Memory access in cache-line-sized steps (64 bytes)
- **Direct kernel memory**: Actual cached vs uncached memory performance
- **High precision**: Nanosecond-level timing using `clock_gettime()`
- **Real cache impact**: Measures actual hardware cache behavior

### Key Features

1. **Direct Memory Access**: User space directly accesses kernel-allocated memory via mmap
2. **Simultaneous Allocation**: Both cached and uncached buffers can be allocated simultaneously
3. **Accurate Timing**: Measures real performance differences between cache behaviors
4. **Architecture Aware**: Properly handles different CPU architectures and kernel versions

### Limitations

1. **Root Privileges**: Requires root access for module loading and memory allocation
2. **Platform Dependency**: Uncached memory allocation varies between architectures
3. **Kernel Version**: Some features may require specific kernel versions or configurations

## Expected Results

When running the timing tests, you should observe:

- **User space cached**: ~1.3 ns per access
- **Kernel cached (mmap)**: ~1.3 ns per access  
- **Kernel uncached (mmap)**: ~70 ns per access
- **Performance difference**: 50-70x slower for uncached vs cached memory

## Troubleshooting

### Common Issues

1. **Module fails to load:**
   ```bash
   # Check kernel log for errors
   dmesg | tail -20
   
   # Verify kernel headers are installed
   ls /lib/modules/$(uname -r)/build
   ```

1. **Module loading fails:**
   ```bash
   # Check kernel headers are installed
   ls /lib/modules/$(uname -r)/build
   
   # Rebuild if necessary
   make clean && make all
   ```

2. **Invalid command format:**
   ```bash
   # Use integer commands only
   echo "0" | sudo tee /proc/uncached_mem  # NOT "alloc_uc"
   echo "1" | sudo tee /proc/uncached_mem  # NOT "alloc_cached"
   echo "2" | sudo tee /proc/uncached_mem  # NOT "free"
   ```

3. **mmap failed:**
   ```bash
   # Ensure memory is allocated before mmap
   echo "0" | sudo tee /proc/uncached_mem  # Allocate first
   ./timing_test                           # Then test
   ```

4. **Build errors:**
   ```bash
   # Clean and rebuild
   make clean
   make all
   ```

### Debugging

- Use `dmesg` to view kernel module messages and allocation details
- Check `/proc/uncached_mem` for current module status
- Verify module is loaded with `lsmod | grep uncached_mem`
- Monitor mmap operations with kernel messages

## Educational Value

This project demonstrates:
- Linux kernel module development
- Memory management and allocation in kernel space
- CPU cache behavior and performance implications
- Proc filesystem interface creation
- Memory mapping between kernel and user space
- Architecture-specific memory caching control
- High-precision timing measurement techniques
- Real-world performance testing methodologies

## Safety Notes

- Always unload the module when finished testing
- The module includes proper cleanup in the exit function
- Memory leaks are prevented by tracking allocated buffers independently
- Use in development/testing environments only
- Module properly handles both allocation types simultaneously
- Clean error handling for invalid commands and allocation failures

## Sample Output

```bash
$ ./timing_test
Memory Access Timing Test - Direct Kernel Memory Access via mmap
=================================================================
Commands: 0=alloc_uc, 1=alloc_cached, 2=free

--- Baseline: User Space Results ---
User cached (malloc): 1.34 ns per access (total: 8582064 ns)

--- Test 1: Kernel Uncached Memory via mmap ---
Successfully mapped uncached memory to user space at 0x7f02a8bbc000
Kernel uncached (mmap): 74.01 ns per access (total: 473692380 ns)
Unmapped kernel memory

--- Test 2: Kernel Cached Memory via mmap ---
Successfully mapped cached memory to user space at 0x7f02a8bbc000
Kernel cached (mmap): 1.32 ns per access (total: 8462089 ns)
Unmapped kernel memory

=== PERFORMANCE SUMMARY ===
User space cached:     1.34 ns per access
Kernel uncached (mmap): 74.01 ns per access
Kernel cached (mmap):   1.32 ns per access

Performance ratio: Uncached is 56.0x slower than cached
```

## License

This project is released under the GPL license, compatible with the Linux kernel.


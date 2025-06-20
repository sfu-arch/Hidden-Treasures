# Troubleshooting Guide

This guide provides solutions for common issues encountered with the Linux kernel memory management modules.

## Table of Contents

1. [Module Loading Issues](#module-loading-issues)
2. [Memory Allocation Failures](#memory-allocation-failures)
3. [Memory Mapping (mmap) Issues](#memory-mapping-mmap-issues)
4. [Permission and Access Issues](#permission-and-access-issues)
5. [Performance Issues](#performance-issues)
6. [Build and Compilation Issues](#build-and-compilation-issues)
7. [CMA-Specific Issues](#cma-specific-issues)
8. [NUMA-Related Issues](#numa-related-issues)
9. [System Stability Issues](#system-stability-issues)
10. [Testing and Debugging Tips](#testing-and-debugging-tips)

---

## Module Loading Issues

### Problem: Module loading fails with "Operation not permitted"
**Symptoms:**
```bash
sudo insmod module.ko
insmod: ERROR: could not insert module: Operation not permitted
```

**Causes & Solutions:**
1. **Secure Boot enabled:**
   ```bash
   # Check if Secure Boot is enabled
   mokutil --sb-state
   
   # Disable Secure Boot in BIOS/UEFI settings
   # Or sign the module (advanced)
   ```

2. **Missing kernel headers:**
   ```bash
   # Install kernel headers
   sudo apt-get install linux-headers-$(uname -r)
   
   # Verify headers exist
   ls /lib/modules/$(uname -r)/build
   ```

3. **Kernel version mismatch:**
   ```bash
   # Check kernel version
   uname -r
   
   # Rebuild modules for current kernel
   make clean && make all
   ```

### Problem: "Unknown symbol" errors in dmesg
**Symptoms:**
```bash
dmesg | tail
[12345.678] module: Unknown symbol in module, or unknown parameter
```

**Solutions:**
```bash
# Check module dependencies
modinfo module.ko

# Load dependent modules first
sudo modprobe cma

# Check for symbol conflicts
cat /proc/kallsyms | grep symbol_name
```

### Problem: Module loads but device files not created
**Symptoms:**
- `lsmod` shows module loaded
- `/dev/module_name` doesn't exist
- `/sys/kernel/module_name/` doesn't exist

**Solutions:**
```bash
# Check dmesg for errors
dmesg | grep module_name

# Check if udev is running
systemctl status udev

# Manually create device node (temporary fix)
sudo mknod /dev/module_name c MAJOR MINOR
sudo chmod 666 /dev/module_name
```

---

## Memory Allocation Failures

### Problem: "Failed to allocate X bytes from DMA coherent memory"
**Symptoms:**
```bash
echo "alloc 64M" > /sys/kernel/cma_cache/command
# dmesg shows: Failed to allocate 67108864 bytes from DMA coherent memory
```

**Root Cause Analysis:**
```bash
# Check CMA availability
grep -E 'Cma(Total|Free)' /proc/meminfo

# Check memory pressure
free -h
cat /proc/meminfo | grep -E '(MemAvailable|MemFree)'

# Check if CMA is allocating but DMA subsystem rejects it
dmesg | grep -E "(cma_alloc|Failed to allocate)" | tail -10
```

**Solutions:**

1. **Increase CMA pool size (permanent fix):**
   ```bash
   # Edit GRUB configuration
   sudo nano /etc/default/grub
   
   # Add to GRUB_CMDLINE_LINUX:
   GRUB_CMDLINE_LINUX="... cma=256M movable_node"
   
   # Update and reboot
   sudo update-grub
   sudo reboot
   ```

2. **DMA mask issue (64-bit systems):**
   ```bash
   # Check if allocations are above 4GB
   dmesg | grep "phys="
   # If physical addresses > 0x100000000, ensure module uses 64-bit DMA mask
   ```

3. **Temporary memory compaction:**
   ```bash
   # Force memory compaction
   echo 1 > /proc/sys/vm/compact_memory
   
   # Drop caches
   echo 3 > /proc/sys/vm/drop_caches
   ```

### Problem: Small allocations work, large ones fail
**Diagnosis:**
```bash
# Test allocation progression
echo "alloc 1M" > /sys/kernel/cma_cache/command    # Works
echo "alloc 4M" > /sys/kernel/cma_cache/command    # Works  
echo "alloc 16M" > /sys/kernel/cma_cache/command   # Fails

# Check fragmentation
cat /proc/buddyinfo
cat /proc/pagetypeinfo
```

**Solutions:**
1. **Enable movable pages:**
   ```bash
   # Add to GRUB: movable_node
   GRUB_CMDLINE_LINUX="... cma=256M movable_node"
   ```

2. **Reduce system memory pressure:**
   ```bash
   # Close unnecessary applications
   # Increase swap space
   sudo fallocate -l 2G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

### Problem: vmalloc allocation failures
**Symptoms:**
```bash
# For dynamic_cache or uncached_mem modules
echo "alloc 64M" > /sys/kernel/uncached_mem/command
# dmesg: vmalloc allocation failed
```

**Solutions:**
```bash
# Check vmalloc space
cat /proc/vmallocinfo | tail -10
grep VmallocTotal /proc/meminfo

# For 32-bit systems, reduce allocation sizes
# vmalloc space is very limited (~128MB total)

# For 64-bit systems, check for vmalloc parameter
# Add to GRUB: vmalloc=512M (if needed)
```

---

## Memory Mapping (mmap) Issues

### Problem: mmap fails with "Invalid argument" (EINVAL)
**Symptoms:**
```c
void *addr = mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, offset);
// Returns MAP_FAILED, errno = 22 (EINVAL)
```

**Root Causes & Solutions:**

1. **Incorrect offset calculation:**
   ```c
   // ❌ WRONG: Don't pass page index directly
   mmap(NULL, 4096, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 5);
   
   // ✅ CORRECT: Calculate byte offset
   mmap(NULL, 4096, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 5 * 4096);
   ```

2. **Memory not allocated:**
   ```bash
   # Ensure memory is allocated before mapping
   echo "alloc 4K" > /sys/kernel/dynamic_cache/command
   # Then map page 0
   ```

3. **Wrong device file:**
   ```bash
   # Ensure correct device file
   ls -la /dev/uncached_mem /dev/dynamic_cache /dev/cma_cache
   ```

### Problem: mmap succeeds but causes segmentation fault
**Symptoms:**
- mmap returns valid address
- Accessing memory causes SIGSEGV
- Kernel may show protection faults

**Debugging:**
```bash
# Check kernel logs
dmesg | grep -E "(segfault|protection fault|general protection)"

# Check VMA flags
cat /proc/PID/maps | grep module_address
```

**Solutions:**

1. **vmalloc memory mapping issue (common with dynamic_cache):**
   ```c
   // Module must use vm_insert_page() for vmalloc memory
   // Not remap_pfn_range()
   
   // In kernel module:
   page = vmalloc_to_page(addr);
   vm_insert_page(vma, user_addr, page);
   ```

2. **Cache attribute mismatch:**
   ```bash
   # Ensure cache attributes are set correctly
   echo "cache 0" > /sys/kernel/dynamic_cache/command
   # Then try mapping
   ```

### Problem: mmap performance is unexpectedly slow
**Diagnosis:**
```c
// Time memory access
struct timespec start, end;
clock_gettime(CLOCK_MONOTONIC, &start);
for (int i = 0; i < 1000000; i++) {
    *data = i;  // Should be ~1ns for cached, ~70ns for uncached
}
clock_gettime(CLOCK_MONOTONIC, &end);
```

**Solutions:**
```bash
# Check cache state
cat /sys/kernel/module/status

# Verify cache setting took effect
dmesg | grep "Set.*cached"

# Test different cache states
echo "cache ID" > command
echo "uncache ID" > command
```

---

## Permission and Access Issues

### Problem: Permission denied writing to sysfs files
**Symptoms:**
```bash
echo "alloc 4M" > /sys/kernel/cma_cache/command
bash: /sys/kernel/cma_cache/command: Permission denied
```

**Solutions:**
```bash
# Set permissions (temporary)
sudo chmod 666 /sys/kernel/cma_cache/command
sudo chmod 666 /dev/cma_cache

# Set permissions for all modules
sudo chmod 666 /dev/uncached_mem /sys/kernel/uncached_mem/command
sudo chmod 666 /dev/dynamic_cache /sys/kernel/dynamic_cache/command  
sudo chmod 666 /dev/cma_cache /sys/kernel/cma_cache/command

# Permanent solution: Add udev rules
echo 'KERNEL=="uncached_mem", MODE="0666"' | sudo tee /etc/udev/rules.d/99-memory-modules.rules
echo 'KERNEL=="dynamic_cache", MODE="0666"' | sudo tee -a /etc/udev/rules.d/99-memory-modules.rules
echo 'KERNEL=="cma_cache", MODE="0666"' | sudo tee -a /etc/udev/rules.d/99-memory-modules.rules
```

### Problem: sysfs files don't exist
**Symptoms:**
```bash
ls /sys/kernel/
# No module directories visible
```

**Solutions:**
```bash
# Check if module loaded successfully
lsmod | grep -E "(uncached_mem|dynamic_cache|cma_cache)"

# Check module initialization logs
dmesg | grep -E "(loading|failed)"

# If module loaded but no sysfs, check kobject creation
dmesg | grep kobject
```

---

## Performance Issues

### Problem: Unexpectedly slow memory access
**Expected Performance:**
- Cached memory: ~1-2 ns per access
- Uncached memory: ~50-100 ns per access

**Diagnosis:**
```c
// Simple performance test
volatile uint64_t *data = (volatile uint64_t *)mapped_address;
struct timespec start, end;

clock_gettime(CLOCK_MONOTONIC, &start);
for (int i = 0; i < 1000000; i++) {
    *data = i;
}
clock_gettime(CLOCK_MONOTONIC, &end);

double ns_per_access = ((end.tv_sec - start.tv_sec) * 1e9 + 
                       (end.tv_nsec - start.tv_nsec)) / 1000000.0;
printf("%.2f ns per access\n", ns_per_access);
```

**Solutions:**

1. **Cache state not applied:**
   ```bash
   # Verify cache state
   cat /sys/kernel/module/status
   
   # Check kernel logs for cache setting
   dmesg | grep -E "(cached|uncached)"
   ```

2. **TLB not flushed:**
   ```bash
   # Module should flush TLB after cache changes
   # Check if module properly calls flush_tlb_kernel_range()
   ```

3. **System under load:**
   ```bash
   # Check system load
   top
   cat /proc/loadavg
   
   # Run tests when system is idle
   ```

### Problem: Performance varies between runs
**Causes:**
- CPU frequency scaling
- Memory pressure
- Other system activity

**Solutions:**
```bash
# Disable CPU frequency scaling (for testing)
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Set CPU affinity
taskset -c 0 ./test_program

# Disable interrupts on test CPU (advanced)
echo 0 > /proc/irq/*/smp_affinity
```

---

## Build and Compilation Issues

### Problem: Compilation fails with "No rule to make target"
**Symptoms:**
```bash
make all
make: *** No rule to make target 'all'. Stop.
```

**Solutions:**
```bash
# Check Makefile exists and has correct targets
cat Makefile | grep -E "^(all|clean):"

# Verify kernel build directory
ls /lib/modules/$(uname -r)/build

# Clean and rebuild
make clean
make all
```

### Problem: "Unknown type name" or missing headers
**Symptoms:**
```c
error: unknown type name 'dma_addr_t'
error: implicit declaration of function 'dma_alloc_coherent'
```

**Solutions:**
```bash
# Check kernel headers
ls /lib/modules/$(uname -r)/build/include/

# Install missing headers
sudo apt-get install linux-headers-$(uname -r)

# For custom kernels, ensure headers match running kernel
uname -r
ls /lib/modules/$(uname -r)/
```

### Problem: Module compiles but symbols missing at runtime
**Solutions:**
```bash
# Check module info
modinfo module.ko

# Check for symbol dependencies
nm module.ko | grep ' U '

# Load prerequisite modules
sudo modprobe cma
sudo modprobe dma_mapping
```

---

## CMA-Specific Issues

### Problem: CMA allocations above 4GB rejected
**Symptoms:**
```bash
# dmesg shows successful CMA allocation but DMA coherent allocation fails
cma: cma_alloc(): returned 0000000086e9b1e0
Failed to allocate X bytes from DMA coherent memory
```

**Root Cause:** 32-bit DMA mask on 64-bit system

**Solution:**
The module should use 64-bit DMA mask:
```c
// In module initialization
ret = dma_set_mask_and_coherent(&pdev->dev, DMA_BIT_MASK(64));
if (ret) {
    // Fallback to 32-bit if needed
    ret = dma_set_mask_and_coherent(&pdev->dev, DMA_BIT_MASK(32));
}
```

### Problem: CMA region too small
**Check current CMA:**
```bash
grep -E 'Cma(Total|Free)' /proc/meminfo
# CmaTotal:          65536 kB  (64MB)
# CmaFree:           32768 kB  (32MB)
```

**Increase CMA size:**
```bash
# Edit GRUB
sudo nano /etc/default/grub

# Add larger CMA allocation
GRUB_CMDLINE_LINUX="... cma=256M movable_node"

# Apply changes
sudo update-grub
sudo reboot
```

### Problem: CMA allocation on wrong NUMA node
**Check allocation location:**
```bash
# Check NUMA topology
ls /sys/devices/system/node/

# See where CMA allocated
cat /sys/kernel/cma_cache/status | grep "NUMA"
```

**Target specific NUMA node:**
```bash
# At boot time
GRUB_CMDLINE_LINUX="... cma=256M@node0"

# At runtime
echo "numa 0" > /sys/kernel/cma_cache/command
```

---

## NUMA-Related Issues

### Problem: Module ignores NUMA node parameter
**Check NUMA support:**
```bash
# Check if NUMA is enabled in kernel
grep CONFIG_NUMA /boot/config-$(uname -r)

# Check available nodes
cat /sys/devices/system/node/online

# Check module NUMA support
cat /sys/kernel/cma_cache/status | grep -A5 "NUMA"
```

**Solutions:**
```bash
# For single-node systems, NUMA node 0 should work
echo "numa 0" > /sys/kernel/cma_cache/command

# For multi-node systems, check node availability
numactl --hardware  # if available
```

### Problem: NUMA strict mode failures
**Symptoms:**
```bash
# Module loaded with numa_strict=true
sudo insmod cma_cache.ko target_numa_node=1 numa_strict=true
# Allocations fail if node 1 unavailable
```

**Solutions:**
```bash
# Check which nodes are online
cat /sys/devices/system/node/online

# Use available node or disable strict mode
sudo rmmod cma_cache
sudo insmod cma_cache.ko target_numa_node=0 numa_strict=false
```

---

## System Stability Issues

### Problem: System freezes during large allocations
**Symptoms:**
- System becomes unresponsive during allocation
- High I/O wait times
- Eventually recovers or needs reboot

**Prevention:**
```bash
# Check available memory before large allocations
free -h

# Monitor during allocation
watch -n 1 'free -h && grep -E "Cma(Total|Free)" /proc/meminfo'

# Start with smaller allocations
echo "alloc 32M" > command  # Instead of 256M
```

**Recovery:**
```bash
# If system responsive, free allocations
echo "free 1" > /sys/kernel/cma_cache/command

# Force memory compaction
echo 1 > /proc/sys/vm/compact_memory

# In worst case, unload module
sudo rmmod module_name
```

### Problem: Kernel crashes or oops
**Symptoms:**
- Kernel panic
- Call traces in dmesg
- System reboot required

**Prevention:**
```bash
# Always check module status before operations
cat /sys/kernel/module/status

# Don't access memory after freeing
# Always unmap before freeing allocations

# Use debug kernel if available
```

**Debugging:**
```bash
# Check crash logs
dmesg | grep -A 20 -B 5 "BUG\|Oops\|panic"

# Save kernel logs
journalctl -k > kernel.log

# Enable additional debugging (if needed)
echo 1 > /proc/sys/kernel/panic_on_oops
```

---

## Testing and Debugging Tips

### Enable Verbose Logging
```bash
# Check module messages
dmesg -w | grep -E "(uncached_mem|dynamic_cache|cma_cache)"

# Enable dynamic debugging (if supported)
echo "file mm/mmap.c +p" > /sys/kernel/debug/dynamic_debug/control
```

### Systematic Testing Approach
```bash
# 1. Start with smallest allocations
echo "alloc 1M" > command

# 2. Gradually increase size
echo "alloc 4M" > command
echo "alloc 16M" > command

# 3. Test cache operations
echo "uncache 1" > command
echo "cache 1" > command

# 4. Test memory mapping
./test_program

# 5. Clean up
echo "free 1" > command
```

### Performance Testing
```bash
# Test cache performance difference
./timing_test          # For uncached_mem
./dynamic_test         # For dynamic_cache  
./cma_test performance # For cma_cache

# Expected results:
# Cached:   ~1-2 ns per access
# Uncached: ~50-100 ns per access
# Ratio:    30-70x slower for uncached
```

### Memory Usage Monitoring
```bash
# Monitor memory usage during testing
watch -n 1 'free -h && cat /proc/meminfo | grep -E "(Cma|Vmalloc)" && cat /proc/buddyinfo | head -3'

# Check for memory leaks
# (Memory usage should return to baseline after freeing allocations)
```

### Automated Testing
```bash
# Use make targets for testing
make test              # Test uncached_mem
make dynamic_test_run  # Test dynamic_cache
make cma_test_run      # Test cma_cache (if available)

# Run all tests
make test_all
```

---

## Quick Reference: Common Solutions

| Problem | Quick Solution |
|---------|---------------|
| Module won't load | `sudo apt-get install linux-headers-$(uname -r)` |
| Permission denied | `sudo chmod 666 /dev/module /sys/kernel/module/command` |
| Large allocation fails | Edit GRUB: `cma=256M movable_node` |
| mmap EINVAL | Check offset: `offset = page_id * PAGE_SIZE` |
| Slow performance | Verify cache state: `cat /sys/kernel/module/status` |
| System freeze | Start small: `echo "alloc 1M"` before larger sizes |
| Build fails | `make clean && make all` |
| CMA above 4GB fails | Module needs 64-bit DMA mask |

---

## Getting Help

1. **Check kernel logs:** `dmesg | tail -20`
2. **Check module status:** `cat /sys/kernel/module/status`
3. **Test with minimal case:** Start with 1MB allocations
4. **Compare with working examples:** Use provided test programs
5. **Check system resources:** `free -h` and `lsmod`

For persistent issues, collect this information:
- Kernel version: `uname -r`
- Memory info: `free -h && grep -E 'Cma(Total|Free)' /proc/meminfo`
- Module status: `cat /sys/kernel/module/status`
- Error messages: `dmesg | grep module_name`
- Test program output and any error messages

# Linux Kernel Memory Management and Cache Control Design

- [Linux Kernel Memory Management and Cache Control Design](#linux-kernel-memory-management-and-cache-control-design)
  - [Overview](#overview)
  - [Learning Objectives](#learning-objectives)
  - [Common Architecture and Design Principles](#common-architecture-and-design-principles)
    - [Shared Design Principles](#shared-design-principles)
  - [Memory Allocation Strategies](#memory-allocation-strategies)
    - [Size-Based Allocation Strategy (uncached\_mem)](#size-based-allocation-strategy-uncached_mem)
    - [Pool-Based Strategy (dynamic\_cache)](#pool-based-strategy-dynamic_cache)
    - [DMA-Coherent Strategy (cma\_cache)](#dma-coherent-strategy-cma_cache)
  - [Cache Control Mechanisms](#cache-control-mechanisms)
    - [Basic Cache Control](#basic-cache-control)
    - [Per-Module Cache Strategies](#per-module-cache-strategies)
    - [Performance Impact](#performance-impact)
  - [Module-Specific Implementations](#module-specific-implementations)
    - [1. Basic Allocation Module (uncached\_mem)](#1-basic-allocation-module-uncached_mem)
    - [2. Dynamic Cache Control Module (dynamic\_cache)](#2-dynamic-cache-control-module-dynamic_cache)
    - [3. Large Block Allocation Module (cma\_cache)](#3-large-block-allocation-module-cma_cache)
  - [Common Implementation Gotchas and Solutions](#common-implementation-gotchas-and-solutions)
    - [1. **Memory Mapping Strategy Selection** ⚠️ **CRITICAL**](#1-memory-mapping-strategy-selection-️-critical)
    - [2. **Cache Attribute Cleanup Order** ⚠️ **CRITICAL**](#2-cache-attribute-cleanup-order-️-critical)
    - [3. **mmap Offset Interpretation**](#3-mmap-offset-interpretation)
    - [4. **Size Validation and Alignment**](#4-size-validation-and-alignment)
    - [5. **Permission Requirements**](#5-permission-requirements)
  - [Performance Characteristics](#performance-characteristics)
    - [Typical Performance Results](#typical-performance-results)
    - [Memory Usage Efficiency](#memory-usage-efficiency)
  - [Educational Value and Applications](#educational-value-and-applications)
    - [**Core Concepts Demonstrated**](#core-concepts-demonstrated)
    - [**Real-World Applications**](#real-world-applications)
    - [**Development Skills Acquired**](#development-skills-acquired)
  - [Module Comparison](#module-comparison)
  - [Comparison with Other Modules](#comparison-with-other-modules)
  - [Kernel API Reference](#kernel-api-reference)
    - [Key API Differences](#key-api-differences)
  - [Best Practices Summary](#best-practices-summary)
  - [Kernel API Quick Reference](#kernel-api-quick-reference)
    - [Memory Allocation APIs](#memory-allocation-apis)
    - [Cache Control APIs](#cache-control-apis)
    - [Memory Mapping APIs](#memory-mapping-apis)
    - [Interface Setup APIs](#interface-setup-apis)
    - [🔧 **API Usage Gotchas**](#-api-usage-gotchas)
      - [Usage Examples](#usage-examples)
          - [For `__get_free_pages` Memory:](#for-__get_free_pages-memory)
          - [For `vmalloc` Memory:](#for-vmalloc-memory)
        - [Size-Based Allocation Strategy](#size-based-allocation-strategy)
        - [Contiguous Memory Mapping (`__get_free_pages`)](#contiguous-memory-mapping-__get_free_pages)
        - [Virtual Memory Mapping (`vmalloc`)](#virtual-memory-mapping-vmalloc)
      - [1. **Cleanup Order is Critical**](#1-cleanup-order-is-critical)
      - [2. **mmap Offset Interpretation**](#2-mmap-offset-interpretation)
      - [3. **Size Alignment**](#3-size-alignment)
      - [4. **Permission Requirements**](#4-permission-requirements)



## Overview

This document explains the design and implementation of three complementary Linux kernel modules that demonstrate different aspects of memory management and CPU cache behavior:

- **`uncached_mem`** - Basic cached/uncached memory allocation with variable sizes (4KB-128MB)
- **`dynamic_cache`** - Page-level dynamic cache control with a pre-allocated memory pool
- **`cma_cache`** - Large contiguous memory blocks (1MB-256MB) using DMA allocation

Together, these modules provide a comprehensive educational framework covering memory allocation strategies, cache control mechanisms, and user-kernel interfaces.

## Learning Objectives

After studying these modules, you should understand:
- **Memory Allocation**: Different kernel memory allocation APIs and their trade-offs
- **Cache Control**: CPU cache behavior and architecture-specific memory attributes  
- **Memory Mapping**: Strategies for mapping kernel memory to user space
- **Interface Design**: Modern sysfs and character device interfaces
- **Performance Analysis**: Measuring and understanding cache performance impact
- **Real-world Applications**: DMA-coherent memory management and device drivers

## Common Architecture and Design Principles

All three modules share a consistent dual-interface design pattern:

```
User Space                    Kernel Space
-----------                   -------------

┌─────────────────┐          ┌──────────────────────┐
│  User Program   │          │                      │
│                 │◄────────►│   Sysfs Interface    │
│ ./test_program  │          │ /sys/kernel/         │
│                 │          │   <module_name>/     │
└─────────────────┘          │                      │
         │                   │                      │
         │ mmap()            └──────────────────────┘
         ▼                             │
┌─────────────────┐                    │ Control
│   /dev/         │           ┌──────────────────────┐
│  <module_name>  │◄─────────►│  Kernel Module       │
│ (char device)   │           │                      │
└─────────────────┘           │ ┌──────────────────┐ │
                              │ │ Memory Manager   │ │
                              │ │                  │ │
                              │ │ Allocation       │ │
                              │ │ Strategy         │ │
                              │ └──────────────────┘ │
                              │                      │
                              │ ┌──────────────────┐ │
                              │ │ Cache Controller │ │
                              │ │                  │ │
                              │ │ set_memory_uc()  │ │
                              │ │ set_memory_wb()  │ │
                              │ └──────────────────┘ │
                              └──────────────────────┘
```

### Shared Design Principles

1. **Dual Interface Design**
   - **Sysfs for Control**: Modern, structured interface for commands and status
   - **Character Device for Data**: Efficient mmap access to allocated memory

2. **Consistent Command Interface**
   - Simple text-based commands via sysfs
   - Size specifications with K/M/G suffixes  
   - Comprehensive status reporting

3. **Robust Error Handling**
   - Input validation with clear error messages
   - Proper resource cleanup on failure
   - Architecture-specific capability detection

## Memory Allocation Strategies

The modules demonstrate three different approaches to kernel memory management:

### Size-Based Allocation Strategy (uncached_mem)

Automatically selects the best allocation method based on size requirements:

| Size Range | Method | Reason |
|------------|--------|---------|
| 4KB - 1MB | `__get_free_pages` / `kmalloc` | Contiguous physical memory, better for small allocations |
| 1MB - 128MB | `vmalloc` | Non-contiguous, handles memory fragmentation |
| >128MB | Rejected | System stability, configurable limit |

### Pool-Based Strategy (dynamic_cache)

Pre-allocates a fixed pool of pages for flexible management:
- 1024 pages (4MB) allocated at module load using `vmalloc`
- Individual pages allocated/freed on demand
- Per-page cache state control
- Support for multi-page contiguous blocks

### DMA-Coherent Strategy (cma_cache)

Uses DMA allocator for large contiguous blocks:
- 1MB to 256MB physically contiguous allocations
- `dma_alloc_coherent()` ensures DMA compatibility
- Platform device integration for proper DMA setup
- Suitable for real-world device driver scenarios

## Cache Control Mechanisms

All modules use the same fundamental cache control APIs but apply them differently:

### Basic Cache Control
```c
// Set memory as uncached (bypasses CPU cache)
int set_memory_uc(unsigned long addr, int numpages);

// Restore write-back caching  
int set_memory_wb(unsigned long addr, int numpages);

// Flush TLB entries for immediate effect
void flush_tlb_kernel_range(unsigned long start, unsigned long end);
```

### Per-Module Cache Strategies

**uncached_mem**: Sets entire allocations as cached or uncached at allocation time
```c
// Apply to entire allocation
set_memory_uc((unsigned long)buffer, num_pages);
```

**dynamic_cache**: Allows individual pages to be toggled at runtime
```c
// Per-page control with TLB flush
set_memory_uc((unsigned long)pages[idx].virt_addr, 1);
flush_tlb_kernel_range(addr, addr + PAGE_SIZE);
```

**cma_cache**: Controls cache state of large contiguous blocks
```c
// Apply to large DMA-coherent blocks
set_memory_uc((unsigned long)alloc->virt_addr, alloc->num_pages);
```

### Performance Impact
- **Cached memory**: ~1-10 ns per access
- **Uncached memory**: ~30-70 ns per access  
- **Performance ratio**: 5-70x slower for uncached memory
- **System impact**: Large uncached regions can affect overall performance

## Module-Specific Implementations

### 1. Basic Allocation Module (uncached_mem)

**Purpose**: Demonstrates fundamental cache behavior with variable-size allocations

**Key Features**:
- Size-adaptive allocation strategy (4KB - 128MB)
- Automatic method selection based on size
- Simple cached/uncached allocation modes
- Suitable for basic cache behavior demonstrations

**State Management**:
```c
static void *uncached_buffer = NULL;
static void *cached_buffer = NULL;
static size_t uncached_size = 0;
static size_t cached_size = 0;
static int uncached_is_vmalloc = 0;
static int cached_is_vmalloc = 0;
```

**Usage Example**:
```bash
echo "0 4M" > /sys/kernel/uncached_mem/command    # Allocate 4MB uncached
echo "1 1M" > /sys/kernel/uncached_mem/command    # Allocate 1MB cached
```

### 2. Dynamic Cache Control Module (dynamic_cache)

**Purpose**: Page-level dynamic cache control with runtime state changes

**Key Features**:
- Pre-allocated pool of 1024 pages (4MB)
- Individual page cache state control
- Multi-page block allocation support
- Runtime cache state toggling

**Enhanced Data Structures**:
```c
struct page_info {
    void *virt_addr;              // Virtual address
    struct page *page;            // Page structure pointer  
    unsigned long pfn;            // Page frame number
    int is_cached;                // Current cache state
    int allocated;                // Allocation status
    int block_id;                 // Block ID for multi-page allocations
};

struct page_block {
    int start_idx;                // Starting page index
    int num_pages;                // Number of pages in block
    int block_id;                 // Unique block identifier
    int allocated;                // Block allocation status
};
```

**Usage Examples**:
```bash
echo "alloc 4K" > /sys/kernel/dynamic_cache/command    # Allocate single page
echo "alloc 1M" > /sys/kernel/dynamic_cache/command    # Allocate 256-page block
echo "uncache 0" > /sys/kernel/dynamic_cache/command   # Set page 0 uncached
echo "toggle 0" > /sys/kernel/dynamic_cache/command    # Toggle page 0 cache state
echo "uncache_block 1" > /sys/kernel/dynamic_cache/command  # Set entire block uncached
```

**Critical Implementation Fix**: Uses `vm_insert_page()` instead of `remap_pfn_range()` for vmalloc memory mapping.

### 3. Large Block Allocation Module (cma_cache)

**Purpose**: DMA-coherent large contiguous memory allocation

**Key Features**:
- Large contiguous blocks (1MB - 256MB)
- DMA-coherent memory allocation
- Platform device integration
- Suitable for real-world device driver scenarios

**DMA Integration**:
```c
struct cma_allocation {
    void *virt_addr;                 // Virtual address
    dma_addr_t dma_handle;          // DMA/physical address
    size_t size;                    // Size in bytes
    int num_pages;                  // Number of pages
    int is_cached;                  // Cache state
    int allocated;                  // Allocation status
    int alloc_id;                   // Unique identifier
};

// Platform device for DMA operations
static struct platform_device *cma_pdev;
```

**Usage Examples**:
```bash
echo "alloc 16M" > /sys/kernel/cma_cache/command    # Allocate 16MB DMA block
echo "uncache 1" > /sys/kernel/cma_cache/command    # Set allocation 1 uncached
```

## Common Implementation Gotchas and Solutions

### 1. **Memory Mapping Strategy Selection** ⚠️ **CRITICAL**

**Problem**: Different memory types require different mapping functions.

**Solution**:
```c
// For __get_free_pages() or alloc_pages() - physically contiguous
ret = remap_pfn_range(vma, vma->vm_start, page_to_pfn(page), PAGE_SIZE, prot);

// For vmalloc() pages - virtually contiguous  
ret = vm_insert_page(vma, vma->vm_start, vmalloc_to_page(addr));

// For DMA coherent memory - physically contiguous
phys_addr = virt_to_phys(buffer);
pfn = phys_addr >> PAGE_SHIFT;
ret = remap_pfn_range(vma, vma->vm_start, pfn, size, prot);
```

### 2. **Cache Attribute Cleanup Order** ⚠️ **CRITICAL**

**Wrong**:
```c
free_pages((unsigned long)buffer, get_order(size));  // System crash!
```

**Correct**:
```c
set_memory_wb((unsigned long)buffer, num_pages);     // Restore caching first
free_pages((unsigned long)buffer, get_order(size));  // Then free
```

### 3. **mmap Offset Interpretation**

**User Space**: Offset in bytes
```c
mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, page_id * getpagesize());
```

**Kernel Space**: Automatically converted to pages
```c
int page_idx = vma->vm_pgoff;  // Kernel receives page index
```

### 4. **Size Validation and Alignment**

**Common Requirements**:
- Minimum size: 4KB (PAGE_SIZE)
- All sizes automatically page-aligned
- Architecture and system-specific maximum limits
- Proper error handling for invalid sizes

### 5. **Permission Requirements**

```bash
# Required for non-root testing:
sudo chmod 666 /sys/kernel/*/command
sudo chmod 666 /dev/*
```

## Performance Characteristics

### Typical Performance Results
- **Cached access**: ~1-10 ns per operation
- **Uncached access**: ~30-70 ns per operation  
- **Performance ratio**: 5-70x slower for uncached memory
- **Cache toggle time**: ~100 μs per page (dynamic_cache)
- **TLB flush impact**: ~10 μs per page
- **Allocation time**: Varies by size and method (see module-specific sections)

### Memory Usage Efficiency
- **Physical memory**: Direct mapping with minimal overhead
- **Virtual memory**: Efficient page-level management
- **Resource overhead**: Minimal per-allocation tracking structures
- **System impact**: Large allocations may trigger memory reclaim

## Educational Value and Applications

### **Core Concepts Demonstrated**
1. **Memory Allocation Strategies**: Comparing different kernel allocation APIs
2. **Cache Behavior Analysis**: Quantifying CPU cache performance impact
3. **Memory Mapping Techniques**: Different approaches for user-kernel memory sharing
4. **Interface Design**: Modern sysfs and character device patterns
5. **Resource Management**: Proper cleanup and error handling
6. **Architecture Considerations**: Platform-specific memory management

### **Real-World Applications**
- **Device Driver Development**: DMA buffer management and cache coherency
- **High-Performance Computing**: Memory optimization and access pattern analysis
- **Graphics and Media**: Large buffer allocation and management
- **Network Processing**: Packet buffer pools and zero-copy techniques
- **System Profiling**: Understanding memory subsystem behavior

### **Development Skills Acquired**
- Linux kernel module development best practices
- Memory subsystem internals and debugging
- User-space/kernel-space interface design
- Performance measurement and optimization
- System-level resource management

## Module Comparison

| Feature | uncached_mem | dynamic_cache | cma_cache |
|---------|-------------|---------------|-----------|
| **Purpose** | Basic cache demonstration | Page-level cache research | DMA-coherent large blocks |
| **Allocation Size** | 4KB - 128MB | Fixed pool (4MB) | 1MB - 256MB |
| **Granularity** | Entire allocation | Per-page (4KB) | Per-allocation |
| **Memory Type** | Various methods | vmalloc pool | DMA coherent |
| **Physical Contiguity** | Not guaranteed | Not guaranteed | Guaranteed |
| **Cache Control** | Static per alloc | Dynamic per page | Dynamic per block |
| **DMA Suitability** | Limited | No | Full |
| **Complexity** | Low | Medium | Medium-High |
| **Use Case** | Basic demonstration | Research tool | Real-world scenarios |

## Comparison with Other Modules

| Feature | uncached_mem | dynamic_cache | cma_cache |
|---------|-------------|---------------|-----------|
| **Allocation Size** | 4KB - 128MB | Fixed pool (4MB) | 1MB - 256MB |
| **Granularity** | Entire allocation | Per-page (4KB) | Per-allocation |
| **Memory Type** | Various methods | vmalloc pool | DMA coherent |
| **Physical Contiguity** | Not guaranteed | Not guaranteed | Guaranteed |
| **Cache Control** | Static per alloc | Dynamic per page | Dynamic per block |
| **DMA Suitability** | Limited | No | Full |
| **Main Allocation APIs** | `__get_free_pages()`<br>`vmalloc()`<br>`kmalloc()` | `vmalloc()`<br>`vmalloc_to_page()` | `dma_alloc_coherent()`<br>`platform_device_*()` |
| **Cache Control APIs** | `set_memory_uc()`<br>`set_memory_wb()` | `set_memory_uc()`<br>`set_memory_wb()`<br>`flush_tlb_*()` | `set_memory_uc()`<br>`set_memory_wb()` |
| **Memory Mapping APIs** | `remap_pfn_range()`<br>`vm_insert_page()` | `vm_insert_page()`<br>`vmalloc_to_page()` | `remap_pfn_range()`<br>`virt_to_phys()` |
| **Interface APIs** | `kobject_create()`<br>`sysfs_create_group()`<br>`cdev_init()` | `kobject_create()`<br>`sysfs_create_group()`<br>`cdev_init()` | `kobject_create()`<br>`sysfs_create_group()`<br>`cdev_init()` |
| **Use Case** | Basic demonstration | Research tool | Real-world scenarios |
| **Complexity** | Low | Medium | Medium-High |

The three modules together provide a comprehensive educational framework covering all aspects of Linux kernel memory management and cache control, from basic concepts to advanced real-world applications.

## Kernel API Reference

The following table provides a quick reference for all kernel APIs used across the three modules:

| API Function | Category | Description |
|--------------|----------|-------------|
| **Memory Allocation APIs** |
| `__get_free_pages(flags, order)` | Physical Memory | Allocates 2^order contiguous physical pages.<br>Returns virtual address, requires `free_pages()` cleanup. |
| `kmalloc(size, flags)` | Physical Memory | Allocates small physically contiguous memory.<br>Fast allocation, limited size, requires `kfree()` cleanup. |
| `vmalloc(size)` | Virtual Memory | Allocates virtually contiguous, physically fragmented memory.<br>Good for large allocations, requires `vfree()` cleanup. |
| `dma_alloc_coherent(dev, size, handle, flags)` | DMA Memory | Allocates DMA-coherent physically contiguous memory.<br>Returns both virtual and DMA addresses, requires `dma_free_coherent()`. |
| `free_pages(addr, order)` | Memory Cleanup | Frees memory allocated by `__get_free_pages()`.<br>Must restore cache attributes before calling. |
| `kfree(ptr)` | Memory Cleanup | Frees memory allocated by `kmalloc()`.<br>Pointer must be exactly as returned by `kmalloc()`. |
| `vfree(ptr)` | Memory Cleanup | Frees memory allocated by `vmalloc()`.<br>Automatically handles page-by-page cleanup. |
| `dma_free_coherent(dev, size, addr, handle)` | DMA Cleanup | Frees DMA memory allocated by `dma_alloc_coherent()`.<br>Must provide both virtual address and DMA handle. |
| **Memory Analysis APIs** |
| `vmalloc_to_page(addr)` | Page Conversion | Converts vmalloc virtual address to page structure.<br>Required for mapping vmalloc pages to user space. |
| `virt_to_phys(addr)` | Address Translation | Converts virtual address to physical address.<br>Works for directly mapped memory (not vmalloc). |
| `page_to_pfn(page)` | Page Conversion | Converts page structure to page frame number.<br>Used for memory mapping operations. |
| `get_order(size)` | Size Calculation | Calculates order (power of 2) for given size.<br>Required for `__get_free_pages()` allocation. |
| **Cache Control APIs** |
| `set_memory_uc(addr, numpages)` | Cache Control | Sets memory pages as uncached (bypasses CPU cache).<br>Dramatically reduces memory access performance (~50x slower). |
| `set_memory_wb(addr, numpages)` | Cache Control | Restores write-back caching for memory pages.<br>Must be called before freeing memory to avoid corruption. |
| `flush_tlb_kernel_range(start, end)` | TLB Management | Flushes TLB entries for specified kernel address range.<br>Required after changing page attributes for immediate effect. |
| **Memory Mapping APIs** |
| `remap_pfn_range(vma, addr, pfn, size, prot)` | Physical Mapping | Maps physically contiguous pages to user virtual memory.<br>Used for `__get_free_pages()` and DMA memory. |
| `vm_insert_page(vma, addr, page)` | Page Mapping | Maps individual page structure to user virtual memory.<br>Required for vmalloc pages (non-contiguous physical memory). |
| **Platform Device APIs** |
| `platform_device_alloc(name, id)` | Device Creation | Allocates platform device structure for DMA operations.<br>Required to obtain device structure for DMA allocation. |
| `platform_device_add(pdev)` | Device Registration | Registers platform device with kernel device model.<br>Makes device available for DMA operations. |
| `dma_set_mask_and_coherent(dev, mask)` | DMA Setup | Sets DMA addressing capabilities for device.<br>Configures which physical addresses device can access. |
| **Sysfs Interface APIs** |
| `kobject_create_and_add(name, parent)` | Sysfs Directory | Creates directory in sysfs filesystem.<br>Provides structured interface for kernel module parameters. |
| `sysfs_create_group(kobj, group)` | Sysfs Attributes | Creates group of attribute files in sysfs directory.<br>Allows user space to interact with kernel module. |
| `sysfs_remove_group(kobj, group)` | Sysfs Cleanup | Removes attribute group from sysfs.<br>Required for proper cleanup on module unload. |
| `kobject_put(kobj)` | Sysfs Cleanup | Decrements reference count and cleans up kobject.<br>Automatically removes sysfs directory when count reaches zero. |
| **Character Device APIs** |
| `alloc_chrdev_region(dev, first, count, name)` | Device Numbers | Dynamically allocates device major/minor numbers.<br>Preferred over static assignment for modern drivers. |
| `cdev_init(cdev, fops)` | Device Init | Initializes character device with file operations.<br>Links device structure to operation function pointers. |
| `cdev_add(cdev, dev, count)` | Device Registration | Adds character device to kernel device registry.<br>Makes device available for user space access. |
| `class_create(owner, name)` | Device Class | Creates device class for automatic /dev file creation.<br>Enables udev to automatically create device files. |
| `device_create(class, parent, devt, data, fmt, ...)` | Device File | Creates device file in /dev directory.<br>Provides user space access point to character device. |


### Key API Differences

| Module | Main Allocation APIs | Cache Control APIs | Memory Mapping APIs |
|--------|---------------------|-------------------|-------------------|
| **uncached_mem** | `__get_free_pages()`<br>`vmalloc()`<br>`kmalloc()` | `set_memory_uc()`<br>`set_memory_wb()` | `remap_pfn_range()`<br>`vm_insert_page()` |
| **dynamic_cache** | `vmalloc()`<br>`vmalloc_to_page()` | `set_memory_uc()`<br>`set_memory_wb()`<br>`flush_tlb_*()` | `vm_insert_page()`<br>`vmalloc_to_page()` |
| **cma_cache** | `dma_alloc_coherent()`<br>`platform_device_*()` | `set_memory_uc()`<br>`set_memory_wb()` | `remap_pfn_range()`<br>`virt_to_phys()` |

## Best Practices Summary

1. **Memory Allocation Strategy**
   - Choose API based on size and contiguity requirements
   - Validate all inputs and handle allocation failures gracefully
   - Consider memory fragmentation and system stability

2. **Cache Control Management**
   - Always restore cache attributes before freeing memory
   - Minimize cache attribute changes (expensive operations)
   - Handle architecture-specific limitations

3. **Memory Mapping Implementation**
   - Use `remap_pfn_range()` for physically contiguous memory
   - Use `vm_insert_page()` for vmalloc memory
   - Validate all page structures before mapping

4. **Interface Design**
   - Provide comprehensive status reporting for debugging
   - Use modern sysfs interfaces over legacy proc files
   - Implement proper error handling and resource cleanup

5. **Testing and Debugging**
   - Monitor system resources during large allocations
   - Check kernel logs for warnings and errors
   - Test edge cases and document limitations clearly

## Kernel API Quick Reference

### Memory Allocation APIs
```c
// Contiguous physical memory
void *__get_free_pages(gfp_t flags, unsigned int order);
void free_pages(unsigned long addr, unsigned int order);

// General kernel memory
void *kmalloc(size_t size, gfp_t flags);
void kfree(const void *ptr);

// Virtual memory (non-contiguous physical)
void *vmalloc(unsigned long size);
void vfree(const void *addr);

// DMA coherent memory
void *dma_alloc_coherent(struct device *dev, size_t size, 
                        dma_addr_t *handle, gfp_t flags);
void dma_free_coherent(struct device *dev, size_t size, void *cpu_addr, 
                      dma_addr_t handle);
```

### Cache Control APIs
```c
// Set memory as uncached
int set_memory_uc(unsigned long addr, int numpages);

// Restore write-back caching
int set_memory_wb(unsigned long addr, int numpages);

// Flush TLB entries
void flush_tlb_kernel_range(unsigned long start, unsigned long end);
```

### Memory Mapping APIs
```c
// Map physically contiguous pages to user VMA
int remap_pfn_range(struct vm_area_struct *vma, unsigned long addr,
                   unsigned long pfn, unsigned long size, pgprot_t prot);

// Map individual page to user VMA
int vm_insert_page(struct vm_area_struct *vma, unsigned long addr, 
                  struct page *page);

// Convert vmalloc address to page structure
struct page *vmalloc_to_page(const void *addr);

// Convert virtual to physical address
phys_addr_t virt_to_phys(volatile void *address);
```

### Interface Setup APIs
```c
// Sysfs interface
struct kobject *kobject_create_and_add(const char *name, struct kobject *parent);
int sysfs_create_group(struct kobject *kobj, const struct attribute_group *grp);

// Character device
int alloc_chrdev_region(dev_t *dev, unsigned baseminor, unsigned count, 
                       const char *name);
void cdev_init(struct cdev *cdev, const struct file_operations *fops);
int cdev_add(struct cdev *p, dev_t dev, unsigned count);
```

This comprehensive design document provides a complete understanding of Linux kernel memory management and cache control mechanisms through three complementary educational modules. Each module builds upon the previous concepts while introducing new techniques and real-world applications.



### 🔧 **API Usage Gotchas**

#### Usage Examples



###### For `__get_free_pages` Memory:
```c
// Get physical address and map
phys_addr = virt_to_phys(buffer_to_map);
pfn = phys_addr >> PAGE_SHIFT;
remap_pfn_range(vma, vma->vm_start, pfn, size, vma->vm_page_prot);
```

###### For `vmalloc` Memory:
```c
// Map page by page
for (offset = 0; offset < size; offset += PAGE_SIZE) {
    struct page *page = vmalloc_to_page((void *)(addr + offset));
    vm_insert_page(vma, user_addr + offset, page);
}
```


##### Size-Based Allocation Strategy

```c
if (size >= LARGE_ALLOC_THRESHOLD) {  // 1MB threshold
    // Use vmalloc for large allocations
    buffer = vmalloc(size);
    is_vmalloc = 1;
    
    // Set uncached per-page for vmalloc
    for (page_addr = addr; page_addr < end; page_addr += PAGE_SIZE) {
        struct page *page = vmalloc_to_page((void *)page_addr);
        set_memory_uc(pfn << PAGE_SHIFT, 1);
    }
} else {
    // Use __get_free_pages for small allocations  
    buffer = (void *)__get_free_pages(GFP_KERNEL, get_order(size));
    is_vmalloc = 0;
    
    // Set uncached for entire region
    set_memory_uc((unsigned long)buffer, num_pages);
}
```


##### Contiguous Memory Mapping (`__get_free_pages`)
```c
// Simple physical mapping
phys_addr = virt_to_phys(buffer_to_map);
pfn = phys_addr >> PAGE_SHIFT;
remap_pfn_range(vma, vma->vm_start, pfn, size, vma->vm_page_prot);
```

##### Virtual Memory Mapping (`vmalloc`)
```c
// Page-by-page mapping for non-contiguous memory
for (offset = 0; offset < size; offset += PAGE_SIZE) {
    struct page *page = vmalloc_to_page((void *)(addr + offset));
    vm_insert_page(vma, user_addr + offset, page);
}
```


#### 1. **Cleanup Order is Critical**
```c
// ❌ WRONG - Will cause system instability
free_pages((unsigned long)buffer, get_order(size));

// ✅ CORRECT - Restore caching first
set_memory_wb((unsigned long)buffer, num_pages);
free_pages((unsigned long)buffer, get_order(size));
```

#### 2. **mmap Offset Interpretation**
```c
// User space usage:
mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);         // Uncached buffer
mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, getpagesize()); // Cached buffer
```
**Gotcha**: Offset is multiplied by page size by the kernel

#### 3. **Size Alignment**
```c
// All sizes are automatically page-aligned:
size = PAGE_ALIGN(size);  // Rounds up to next page boundary
```

#### 4. **Permission Requirements**
```bash
# Sysfs files need write permissions for non-root users:
sudo chmod 666 /sys/kernel/uncached_mem/command
sudo chmod 666 /dev/uncached_mem
```


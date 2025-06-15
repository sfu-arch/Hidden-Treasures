# Linux Kernel Module Design: Uncached Memory Allocation

## Overview

This document explains the design and implementation of a Linux kernel module that demonstrates cache behavior by providing both cached and uncached memory allocation with variable size support. The module uses modern Linux kernel APIs and provides a sysfs interface for control.

## Learning Objectives

After studying this module, you should understand:
- Linux kernel module development practices
- Memory management in kernel space (kmalloc, vmalloc, __get_free_pages)
- CPU cache control and architecture-specific memory attributes
- Sysfs interface design and implementation
- Character device creation for mmap functionality
- Performance implications of cache behavior

## Architecture Overview

```
User Space                    Kernel Space
-----------                   -------------

┌─────────────────┐          ┌──────────────────────┐
│  User Program   │          │                      │
│                 │◄────────►│   Sysfs Interface    │
│ ./timing_test   │          │ /sys/kernel/         │
└─────────────────┘          │   uncached_mem/      │
         │                   │                      │
         │ mmap()            └──────────────────────┘
         ▼                             │
┌─────────────────┐                    │ Control
│   /dev/         │          ┌──────────────────────┐
│  uncached_mem   │◄────────►│  Kernel Module       │
│ (char device)   │          │                      │
└─────────────────┘          │ ┌──────────────────┐ │
                              │ │ Memory Allocator │ │
                              │ │                  │ │
                              │ │ Small: kmalloc/  │ │
                              │ │ __get_free_pages │ │
                              │ │                  │ │
                              │ │ Large: vmalloc   │ │
                              │ └──────────────────┘ │
                              │                      │
                              │ ┌──────────────────┐ │
                              │ │ Cache Control    │ │
                              │ │                  │ │
                              │ │ set_memory_uc()  │ │
                              │ │ set_memory_wb()  │ │
                              │ └──────────────────┘ │
                              └──────────────────────┘
```

## Core Design Principles

### 1. **Dual Interface Design**
- **Sysfs for Control**: Modern, clean interface for allocation commands
- **Character Device for Data**: Efficient mmap access to allocated memory

### 2. **Adaptive Allocation Strategy**
The module automatically selects the best allocation method based on size:

| Size Range | Method | Reason |
|------------|--------|---------|
| 4KB - 1MB | `__get_free_pages` / `kmalloc` | Contiguous physical memory, better for small allocations |
| 1MB - 128MB | `vmalloc` | Non-contiguous, handles memory fragmentation |
| >128MB | Rejected | System stability, configurable limit |

### 3. **State Management**
The module maintains separate state for uncached and cached allocations:
```c
// Global state variables
static void *uncached_buffer = NULL;
static void *cached_buffer = NULL;
static size_t uncached_size = 0;
static size_t cached_size = 0;
static int uncached_is_vmalloc = 0;
static int cached_is_vmalloc = 0;
```

## Key API Calls and Their Usage

### Memory Allocation APIs

#### 1. **`__get_free_pages()`** - Contiguous Physical Memory
```c
buffer = (void *)__get_free_pages(GFP_KERNEL, get_order(size));
```
- **Purpose**: Allocate contiguous physical memory pages
- **When used**: Small allocations (<1MB) 
- **Advantages**: Contiguous physical memory, efficient for DMA
- **Limitations**: Subject to memory fragmentation, limited to smaller sizes
- **Cleanup**: `free_pages()`

#### 2. **`vmalloc()`** - Virtual Memory Allocation
```c
buffer = vmalloc(size);
```
- **Purpose**: Allocate virtually contiguous memory (physically non-contiguous)
- **When used**: Large allocations (≥1MB)
- **Advantages**: Can allocate large amounts, less affected by fragmentation
- **Limitations**: Physically non-contiguous, slight performance overhead
- **Cleanup**: `vfree()`

#### 3. **`kmalloc()`** - Kernel Memory Allocation
```c
buffer = kmalloc(size, GFP_KERNEL);
```
- **Purpose**: General-purpose kernel memory allocation
- **When used**: Small cached allocations (≤4KB)
- **Advantages**: Fast, efficient for small allocations
- **Limitations**: Size limited, subject to fragmentation
- **Cleanup**: `kfree()`

### Cache Control APIs

#### 1. **`set_memory_uc()`** - Set Memory as Uncached
```c
int result = set_memory_uc((unsigned long)buffer, num_pages);
```
- **Purpose**: Mark memory pages as uncached (bypasses CPU caches)
- **Architecture**: x86/x86_64 specific
- **Performance**: Significantly slower access (~50-70x)
- **Use case**: Testing cache behavior, DMA coherency

#### 2. **`set_memory_wb()`** - Restore Write-Back Caching
```c
int result = set_memory_wb((unsigned long)buffer, num_pages);
```
- **Purpose**: Restore normal write-back caching before freeing memory
- **Critical**: Must be called before `free_pages()` to avoid system instability

### Sysfs Interface APIs

#### 1. **Kobject Management**
```c
// Create sysfs directory
uncached_kobj = kobject_create_and_add("uncached_mem", kernel_kobj);

// Create attribute group
sysfs_create_group(uncached_kobj, &attr_group);
```

#### 2. **Attribute Definitions**
```c
static struct kobj_attribute command_attr = __ATTR(command, 0664, command_show, command_store);
```
- **0664 permissions**: Owner/group read-write, others read-only
- **show function**: Handles read operations
- **store function**: Handles write operations

### Character Device APIs

#### 1. **Device Registration**
```c
// Allocate device number
alloc_chrdev_region(&dev_num, 0, 1, "uncached_mem");

// Initialize and add character device
cdev_init(&uncached_cdev, &device_fops);
cdev_add(&uncached_cdev, dev_num, 1);

// Create device class and device file
uncached_class = class_create(THIS_MODULE, "uncached_mem");
device_create(uncached_class, NULL, dev_num, NULL, "uncached_mem");
```

#### 2. **Memory Mapping APIs**

##### For `__get_free_pages` Memory:
```c
// Get physical address and map
phys_addr = virt_to_phys(buffer_to_map);
pfn = phys_addr >> PAGE_SHIFT;
remap_pfn_range(vma, vma->vm_start, pfn, size, vma->vm_page_prot);
```

##### For `vmalloc` Memory:
```c
// Map page by page
for (offset = 0; offset < size; offset += PAGE_SIZE) {
    struct page *page = vmalloc_to_page((void *)(addr + offset));
    vm_insert_page(vma, user_addr + offset, page);
}
```

## Memory Management Design

### Size-Based Allocation Strategy

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

### Memory Mapping Strategy

The module handles two different mapping scenarios:

#### Contiguous Memory Mapping (`__get_free_pages`)
```c
// Simple physical mapping
phys_addr = virt_to_phys(buffer_to_map);
pfn = phys_addr >> PAGE_SHIFT;
remap_pfn_range(vma, vma->vm_start, pfn, size, vma->vm_page_prot);
```

#### Virtual Memory Mapping (`vmalloc`)
```c
// Page-by-page mapping for non-contiguous memory
for (offset = 0; offset < size; offset += PAGE_SIZE) {
    struct page *page = vmalloc_to_page((void *)(addr + offset));
    vm_insert_page(vma, user_addr + offset, page);
}
```

## Size Limitations and Gotchas

### 🚨 **Critical Size Limitations**

#### 1. **Minimum Size: 4KB (PAGE_SIZE)**
```bash
echo "0 1K" > /sys/kernel/uncached_mem/command  # ❌ FAILS
echo "0 4K" > /sys/kernel/uncached_mem/command  # ✅ SUCCESS
```
**Reason**: Memory management works on page boundaries

#### 2. **Maximum Size: 128MB (Configurable)**
```bash
echo "0 200M" > /sys/kernel/uncached_mem/command  # ❌ FAILS  
echo "0 128M" > /sys/kernel/uncached_mem/command  # ✅ SUCCESS
```
**Reason**: System stability and virtual memory limitations

#### 3. **1MB Threshold Behavior**
```bash
echo "0 512K" > command    # Uses __get_free_pages
echo "0 2M" > command      # Uses vmalloc (different behavior!)
```

### ⚠️ **Memory Allocation Gotchas**

#### 1. **Fragmentation Issues**
- **`__get_free_pages`**: Requires contiguous physical memory
- **Large allocations may fail** even with available memory
- **Solution**: Module automatically switches to `vmalloc` for large sizes

#### 2. **Cache Control Limitations**
```c
// This may fail silently on some systems:
set_memory_uc((unsigned long)buffer, num_pages);
```
- **Architecture dependent**: Only works reliably on x86/x86_64
- **Large vmalloc regions**: May fail to set all pages as uncached
- **No error for partial failure**: Check dmesg for warnings

#### 3. **vmalloc Address Space Limits**
- **32-bit systems**: Very limited vmalloc space (~128MB total)
- **64-bit systems**: Much larger but still finite
- **Check available**: `cat /proc/vmallocinfo`

### 🔧 **API Usage Gotchas**

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

### 🎯 **Performance Gotchas**

#### 1. **Dramatic Performance Difference**
- **Cached memory**: ~1.3 ns per access
- **Uncached memory**: ~70 ns per access (50-70x slower!)
- **System impact**: Large uncached allocations can affect overall performance

#### 2. **Allocation Time**
- **Small allocations**: Near-instantaneous
- **Large allocations**: May take several seconds (especially >64MB)
- **System freezing**: Very large allocations may briefly impact system responsiveness

#### 3. **Memory Pressure**
- **Large allocations**: Can trigger memory reclaim
- **OOM conditions**: Very large allocations may trigger out-of-memory killer
- **Recommendation**: Monitor system memory before large allocations

### 🔍 **Debugging and Monitoring**

#### 1. **Check Allocation Success**
```bash
# Always verify allocation succeeded:
cat /sys/kernel/uncached_mem/status
dmesg | tail -10
```

#### 2. **Monitor Memory Usage**
```bash
# Before large allocations:
free -h
cat /proc/vmallocinfo | tail -10
```

#### 3. **Check Cache Setting Success**
```bash
# Look for warnings in kernel log:
dmesg | grep -i "uncached\|failed"
```

## Educational Insights

### 1. **Cache Behavior Demonstration**
This module provides concrete evidence of CPU cache impact:
- **Cache hits**: ~1.3 ns access time
- **Cache misses**: ~70 ns access time  
- **Real-world impact**: 50-70x performance difference

### 2. **Kernel Memory Management**
Shows three different allocation strategies and their trade-offs:
- **kmalloc**: Fast, limited size
- **__get_free_pages**: Contiguous physical, better for DMA  
- **vmalloc**: Large sizes, handles fragmentation

### 3. **Modern Kernel Interfaces**
Demonstrates modern Linux kernel practices:
- **Sysfs over proc**: Structured, per-attribute files
- **Proper error handling**: Input validation and cleanup
- **Resource management**: Automatic method selection

### 4. **System Programming Concepts**
- **Memory mapping**: Different strategies for different memory types
- **Hardware abstraction**: Architecture-specific cache control
- **Resource limits**: Balancing functionality with system stability

## Best Practices Learned

1. **Always validate input sizes** and provide clear error messages
2. **Use appropriate allocation method** based on size requirements  
3. **Restore memory attributes** before freeing to avoid system issues
4. **Provide comprehensive status reporting** for debugging
5. **Handle architecture differences** gracefully
6. **Test edge cases** and document limitations clearly
7. **Monitor system resources** during large allocations

This module serves as an excellent example of modern Linux kernel module development while providing practical insights into CPU cache behavior and memory management strategies.

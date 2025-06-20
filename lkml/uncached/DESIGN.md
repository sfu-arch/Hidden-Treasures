# Linux Kernel Module Design: Uncached Memory Allocation

> **Quick Reference**: For a comparison of all three modules (uncached_mem, dynamic_cache, cma_cache), see the [Module Comparison Table](README.md#-module-comparison) in README.md.

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
│                 │          │   uncached_mem/      │
└─────────────────┘          │                      │
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

---

# Dynamic Cache Control Module (dynamic_cache.c)

## Overview

The `dynamic_cache` module extends the cache behavior demonstration by providing **per-page dynamic cache control**. Unlike the static `uncached_mem` module that allocates entire regions as cached or uncached, this module allows individual pages to have their cache state toggled at runtime.

## Key Features

### 1. **Page-Level Cache Control**
- Individual pages can be set as cached or uncached dynamically
- Uses `set_memory_uc()` and `set_memory_wb()` for cache attribute changes
- Maintains per-page state tracking
- Supports cache state toggling

### 2. **Advanced Memory Management**
- Pre-allocates a pool of 1024 pages using vmalloc
- Tracks each page's virtual address, page structure, and PFN
- Manages allocation/deallocation of individual pages to users
- Automatic state restoration on cleanup

### 3. **Variable Size Block Allocation**
- Supports allocating contiguous blocks of multiple pages
- Size specification with K/M/G suffixes (e.g., "4K", "1M", "64M")
- Automatic page alignment and size validation (4K min, 64M max)
- Block-level operations (cache/uncache/free entire blocks)
- Mixed allocation model (single pages + multi-page blocks)

### 4. **Enhanced User Interface**
- **Command interface**: `echo "command args" > /sys/kernel/dynamic_cache/command`
- **Status monitoring**: Real-time view of all allocated pages
- **Page map visualization**: Visual representation of allocation state
- **Pattern testing**: Set and verify test patterns on pages

## Architecture

```
User Space                    Kernel Space
-----------                   -------------

┌─────────────────┐          ┌──────────────────────┐
│  Test Program   │          │  Sysfs Interface     │
│                 │◄────────►│ /sys/kernel/         │
│ ./dynamic_test  │          │   dynamic_cache/     │
└─────────────────┘          │                      │
         │                   │                      │
         │ mmap()            └──────────────────────┘
         ▼                   │
┌─────────────────┐          ┌──────┬───────────────┐
│   /dev/         │◄────────►│  Page Pool Manager   │
│ dynamic_cache   │          │                      │
└─────────────────┘          │ ┌──────────────────┐ │
                              │ │ Page Pool        │ │
                              │ │ (1024 pages)     │ │
                              │ │                  │ │
                              │ │ vmalloc region   │ │
                              │ └──────────────────┘ │
                              │                      │
                              │ ┌──────────────────┐ │
                              │ │ Page Tracking    │ │
                              │ │                  │ │
                              │ │ • virt_addr      │ │
                              │ │ • page struct    │ │
                              │ │ • pfn            │ │
                              │ │ • cache state    │ │
                              │ │ • allocation     │ │
                              │ │                  │ │
                              │ └──────────────────┘ │
                              │                      │
                              │ ┌──────────────────┐ │
                              │ │ Dynamic Cache    │ │
                              │ │ Control          │ │
                              │ │                  │ │
                              │ │ set_memory_uc()  │ │
                              │ │ set_memory_wb()  │ │
                              │ │ flush_tlb_*()    │ │
                              │ └──────────────────┘ │
                              └──────────────────────┘
```

## Usage Examples

### Basic Page Allocation and Cache Control

```bash
# Load the module
sudo insmod dynamic_cache.ko

# Check initial status
cat /sys/kernel/dynamic_cache/status

# Allocate a page (returns page ID)
echo "alloc" > /sys/kernel/dynamic_cache/command

# Set page 0 as uncached
echo "uncache 0" > /sys/kernel/dynamic_cache/command

# Set page 0 as cached
echo "cache 0" > /sys/kernel/dynamic_cache/command

# Toggle cache state
echo "toggle 0" > /sys/kernel/dynamic_cache/command

# Set test pattern
echo "pattern 0 AA" > /sys/kernel/dynamic_cache/command

# View allocation map
cat /sys/kernel/dynamic_cache/page_map

# Free the page
echo "free 0" > /sys/kernel/dynamic_cache/command
```

#### Permission Requirements
```bash
# Sysfs files need write permissions for non-root users:
sudo chmod 666 /sys/kernel/dynamic_cache/command
sudo chmod 666 /dev/dynamic_cache
```

### Variable Size Block Allocation

The dynamic_cache module supports allocating contiguous blocks of multiple pages with size specifications:

```bash
# Allocate specific sizes (returns block ID for multi-page allocations)
echo "alloc 4K" > /sys/kernel/dynamic_cache/command      # Single page (same as basic alloc)
echo "alloc 8K" > /sys/kernel/dynamic_cache/command      # 2 contiguous pages  
echo "alloc 1M" > /sys/kernel/dynamic_cache/command      # 256 contiguous pages
echo "alloc 2048" > /sys/kernel/dynamic_cache/command    # 2048 bytes = 1 page (rounded up)

# Supported size suffixes: K (kilobytes), M (megabytes), G (gigabytes)
# Size limits: 4K minimum, 64M maximum
# All sizes are automatically rounded up to page boundaries

# Check allocation status (shows both individual pages and blocks)
cat /sys/kernel/dynamic_cache/status

# Block operations work on all pages in the block
echo "uncache_block 1" > /sys/kernel/dynamic_cache/command  # Make entire block uncached
echo "cache_block 1" > /sys/kernel/dynamic_cache/command    # Make entire block cached
echo "free_block 1" > /sys/kernel/dynamic_cache/command     # Free entire block

# Individual page operations within blocks still work
echo "uncache 5" > /sys/kernel/dynamic_cache/command        # Make specific page uncached
echo "cache 5" > /sys/kernel/dynamic_cache/command          # Make specific page cached
```

#### Block Status Display

The status interface shows detailed information about allocated blocks:

```
Active Blocks:
Block ID  Start  Pages  Size
--------  -----  -----  ----
       1      0      2    8K
       2      5     64  256K

Allocated Pages:
ID   Virtual     PFN        Block    Cache State
---  ----------  ---------  -------  -----------
  0  ffff...001  123456789        1  UNCACHED
  1  ffff...002  123456790        1  CACHED
  5  ffff...006  123456794        2  UNCACHED
...
```

### Memory Mapping Individual Pages

```c
// Open device file
int fd = open("/dev/dynamic_cache", O_RDWR);

// Map page 5 (page ID is passed as offset)
void *addr = mmap(NULL, 4096, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 5);

// Access the memory
volatile uint64_t *data = (volatile uint64_t *)addr;
*data = 0x1234567890ABCDEF;

// Unmap
munmap(addr, 4096);
close(fd);
```

## Implementation Details

### Block Management for Variable Size Allocation

The module supports both single-page allocations and multi-page contiguous blocks:

```c
struct page_block {
    int start_idx;                    // Starting page index in the pool
    int num_pages;                    // Number of contiguous pages in this block
    int block_id;                     // Unique block identifier
    int allocated;                    // Whether this block is in use
};

static struct page_block blocks[MAX_PAGES]; // Block tracking for contiguous allocations
static int current_block_id = 1;             // Next block ID to assign
```

**Size Parsing and Validation:**
```c
static int parse_size_string(const char *str, size_t *size)
{
    // Supports: raw bytes, K/M/G suffixes
    // Example: "4K" -> 4096, "1M" -> 1048576
    // Automatic page alignment: PAGE_ALIGN(size)
    // Range validation: 4K minimum, 64M maximum
}
```

**Multi-Page Allocation Algorithm:**
```c
static int allocate_user_pages(int num_pages)
{
    // 1. Find contiguous sequence of free pages
    // 2. Allocate block tracking entry
    // 3. Mark individual pages with block_id
    // 4. Initialize block metadata
    // 5. Return block_id for block operations
}
```

**Block vs Page Operation Handling:**
- **Single page commands**: `alloc`, `cache/uncache N`, `free N`  
- **Block commands**: `alloc SIZE`, `cache/uncache_block ID`, `free_block ID`
- **Mixed mode**: Pages within blocks can still be individually controlled

### Page Pool Management

```c
struct page_info {
    void *virt_addr;              // Virtual address of the page
    struct page *page;            // Page structure pointer  
    unsigned long pfn;            // Page frame number
    int is_cached;                // Current cache state (1=cached, 0=uncached)
    int allocated;                // Whether this slot is in use
    int block_id;                 // Block ID if part of multi-page allocation (-1 for single)
};

static struct page_info pages[MAX_PAGES]; // Page tracking array
```

### Dynamic Cache Control

The module uses architecture-specific functions to control page cache attributes:

```c
static int set_page_cache_state(int page_idx, int cached)
{
    if (cached) {
        // Set page as cached (write-back)
        ret = set_memory_wb((unsigned long)pages[page_idx].virt_addr, 1);
    } else {
        // Set page as uncached
        ret = set_memory_uc((unsigned long)pages[page_idx].virt_addr, 1);
    }
    
    // Flush TLB to ensure changes take effect
    flush_tlb_kernel_range((unsigned long)pages[page_idx].virt_addr,
                          (unsigned long)pages[page_idx].virt_addr + PAGE_SIZE);
    return ret;
}
```

### Memory Mapping with Cache Attributes

```c
static int device_mmap(struct file *file, struct vm_area_struct *vma)
{
    int page_idx = vma->vm_pgoff; // Page index from offset
    struct page *page = pages[page_idx].page;
    
    // Set page protection based on cache state
    if (!pages[page_idx].is_cached) {
        vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot);
    }
    
    // For vmalloc pages, use vm_insert_page instead of remap_pfn_range
    return vm_insert_page(vma, vma->vm_start, page);
}
```

## Critical Implementation Gotchas and Fixes

### 1. **vmalloc Memory Mapping Issue** ⚠️

**Problem**: Initially used `remap_pfn_range()` to map vmalloc pages, which failed because vmalloc pages are not guaranteed to be physically contiguous.

**Symptoms**:
- mmap() returns success but pages aren't properly accessible
- Kernel may crash or corrupt memory on page access
- Inconsistent behavior across different memory allocations

**Wrong Implementation**:
```c
// DON'T DO THIS with vmalloc pages
pfn = pages[page_idx].pfn;
if (remap_pfn_range(vma, vma->vm_start, pfn, PAGE_SIZE, vma->vm_page_prot)) {
    return -EAGAIN;
}
```

**Correct Implementation**:
```c
// CORRECT: Use vm_insert_page for vmalloc pages
page = pages[page_idx].page;
ret = vm_insert_page(vma, vma->vm_start, page);
if (ret) {
    printk(KERN_ERR "vm_insert_page failed: %d\n", ret);
    return ret;
}
```

**Key Learning**: `remap_pfn_range()` is for physically contiguous memory (like `__get_free_pages()`), while `vm_insert_page()` is for individual pages from vmalloc.

### 2. **mmap Offset Interpretation** ⚠️

**Problem**: Confusion about how mmap offset parameter works in user space vs kernel space.

**Wrong User Code**:
```c
// Wrong: passing page index directly
mmap(NULL, 4096, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 1);  // FAILS
```

**Correct User Code**:
```c
// Correct: offset in bytes (page_index * PAGE_SIZE)
mmap(NULL, 4096, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 4096);  // Works
```

**Explanation**: 
- User space: offset is in **bytes**
- Kernel space: `vma->vm_pgoff` is automatically converted to **pages** by the kernel
- For page N: user passes `N * PAGE_SIZE`, kernel receives `N` in `vm_pgoff`

### 3. **VMA Flags for vmalloc Memory** ⚠️

**Problem**: Using inappropriate VMA flags that are meant for device memory.

**Wrong**:
```c
vma->vm_flags |= VM_IO | VM_DONTEXPAND | VM_DONTDUMP;  // VM_IO inappropriate
```

**Correct**:
```c
vma->vm_flags |= VM_DONTEXPAND | VM_DONTDUMP;  // No VM_IO for vmalloc
```

**Key Learning**: `VM_IO` flag is for actual hardware device memory, not for kernel vmalloc memory.

### 4. **Page Structure Validation** ⚠️

**Problem**: Not validating that vmalloc_to_page() succeeded before using the page structure.

**Defensive Implementation**:
```c
if (!pages[page_idx].page) {
    printk(KERN_ERR "Page %d has no page structure\n", page_idx);
    return -EINVAL;
}
```

### 5. **Test Program Page Allocation Tracking** ⚠️

**Problem**: Complex parsing logic that failed to correctly identify allocated page IDs.

**Wrong Approach**:
```c
// Complex status parsing that was unreliable
line = strtok(status_buffer, "\n");
// ... complex parsing logic that broke
```

**Simple Approach**:
```c
// Simple sequential tracking since module allocates pages in order
static int next_page_id = 0;
int page_id = num_allocated; // Use allocation count as page ID
```

## Educational Value - Advanced Concepts

### 1. **Memory Mapping Strategies**
Real-world demonstration of different memory mapping approaches:
- **`remap_pfn_range()`**: For physically contiguous memory
- **`vm_insert_page()`**: For individual pages (vmalloc, page cache)
- **Understanding when to use each approach**

### 2. **vmalloc vs. Physical Memory Management**
- **Virtual memory**: vmalloc provides virtual contiguity, not physical
- **Page structure tracking**: Using `vmalloc_to_page()` for individual page access
- **Memory pool design**: Pre-allocation vs. on-demand allocation strategies

### 3. **User-Kernel Interface Design**
- **Offset interpretation**: How user-space byte offsets become kernel page indices
- **Error handling**: Proper validation and error reporting across the interface
- **Device file semantics**: Character device behavior for memory mapping

### 3. **Runtime Cache Control**
- **Dynamic reconfiguration**: Changing memory attributes after allocation
- **State management**: Tracking complex per-page attributes
- **Performance implications**: Real-time cache behavior changes

### 4. **Comprehensive System Interface**
- **Multi-attribute sysfs**: Command, status, and visualization interfaces
- **Flexible command parsing**: String-based command interface
- **Device file integration**: Character device for memory mapping

## Performance Characteristics

The dynamic cache control allows for detailed performance analysis:

```
Typical Performance Results (After Fixes):
- Cached access:    ~1.3 ns per operation
- Uncached access:  ~70 ns per operation  
- Performance ratio: 50-70x slower for uncached
- Cache toggle time: ~100 μs per page
- TLB flush impact: ~10 μs per page
- mmap functionality: Fully working with both cache states
```

## Gotchas and Limitations (Updated with Fixes)

### 1. **Architecture Dependencies**
- Cache control functions may not be available on all architectures
- Some architectures may not support fine-grained cache control
- TLB flush behavior varies between CPU architectures
- **Fixed**: Module now gracefully handles architecture limitations with proper ifdefs

### 2. **Memory Pool Limitations**
- Fixed pool size (1024 pages = 4MB)
- vmalloc may not provide physically contiguous pages
- Cannot extend pool size dynamically
- **Important**: vmalloc pages require special mapping techniques (vm_insert_page)

### 3. **Performance Considerations**
- Cache attribute changes are expensive operations
- TLB flushes affect system-wide performance
- Frequent cache toggles can impact overall system performance
- **Measured Impact**: ~56x performance difference between cached/uncached in tests

### 4. **System Stability**
- Improper cache attribute management can cause system instability
- Memory must be restored to cached state before freeing
- Race conditions possible with concurrent access
- **Fixed**: Proper cleanup on module unload and signal handling in tests

### 5. **Memory Mapping Complexity** ⚠️ **CRITICAL**
- **vmalloc mapping**: Requires `vm_insert_page()`, not `remap_pfn_range()`
- **Offset calculation**: User space uses byte offsets, kernel receives page offsets
- **VMA flags**: Different flags needed for vmalloc vs device memory
- **Page validation**: Must verify vmalloc_to_page() success before mapping

### 6. **Testing and Debugging Challenges**
- Device permissions require root access for testing
- Kernel debugging requires dmesg monitoring
- mmap failures can be silent or misleading without proper error checking
- **Fixed**: Comprehensive test suite with proper error reporting

## Best Practices for Dynamic Cache Control (Updated)

1. **Minimize cache attribute changes** - They are expensive operations
2. **Batch operations** when possible to reduce TLB flush overhead
3. **Always restore cache state** before freeing memory
4. **Use appropriate synchronization** for multi-threaded access
5. **Monitor system performance** impact during testing
6. **Test on target architecture** - behavior varies significantly
7. **Handle architecture differences** gracefully in production code
8. **🆕 Use correct mapping functions** - `vm_insert_page()` for vmalloc, `remap_pfn_range()` for physical
9. **🆕 Validate page structures** before attempting to map them
10. **🆕 Implement comprehensive testing** with proper error handling and cleanup

## Development Lessons Learned

### **Memory Mapping API Selection**
```c
// For __get_free_pages() or alloc_pages() - physically contiguous
ret = remap_pfn_range(vma, vma->vm_start, page_to_pfn(page), PAGE_SIZE, prot);

// For vmalloc() pages - virtually contiguous
ret = vm_insert_page(vma, vma->vm_start, vmalloc_to_page(addr));
```

### **User-Space mmap Offset Calculation**
```c
// For page N:
int page_id = N;
off_t offset = page_id * getpagesize();  // Convert to bytes
void *addr = mmap(NULL, getpagesize(), PROT_READ|PROT_WRITE, MAP_SHARED, fd, offset);
```

### **Kernel Debugging Strategy**
```c
printk(KERN_INFO "mmap: size=%lu, offset=%lu, page_idx=%d\n", 
       size, vma->vm_pgoff, (int)vma->vm_pgoff);
// Always log key parameters for debugging
```

## Comparison: Static vs. Dynamic Cache Control (Updated)

| Feature | uncached_mem (Static) | dynamic_cache (Dynamic) |
|---------|----------------------|-------------------------|
| Cache Control | Allocation-time only | Runtime changeable |
| Granularity | Entire allocation | Individual pages |
| Memory Usage | Variable size (4KB-128MB) | Fixed pool (1024 pages) |
| Allocation Method | __get_free_pages/vmalloc | vmalloc only |
| Mapping Function | remap_pfn_range/vm_insert_page | vm_insert_page |
| Interface | Simple alloc/free | Command-based |
| Complexity | Low | High |
| Use Case | Basic cache demonstration | Advanced cache research |
| Performance Impact | One-time setup cost | Ongoing toggle costs |
| **mmap Support** | ✅ Fixed and working | ✅ Fixed and working |
| **Test Coverage** | Basic timing tests | Comprehensive test suite |

Both modules together provide a comprehensive understanding of Linux kernel memory management and CPU cache behavior, suitable for educational use and system performance research. The dynamic_cache module now includes important lessons about vmalloc memory mapping and proper user-kernel interface design.

---

# CMA Cache Control Module (cma_cache.c)

## Overview

The `cma_cache` module demonstrates **large contiguous memory allocation** using DMA coherent allocator with cache control capabilities. This module focuses on allocating multi-megabyte contiguous memory blocks that are suitable for DMA operations and provides dynamic cache control over these large allocations.

## Key Features

### 1. **Large Contiguous Memory Allocation**
- Uses `dma_alloc_coherent()` for guaranteed physically contiguous memory
- Leverages CMA (Contiguous Memory Allocator) backend when available
- Supports allocation sizes from 1MB to 256MB
- Maintains allocation tracking with unique IDs

### 2. **Variable Size with Suffix Support**
- Size specification with K/M/G suffixes (e.g., "4M", "64M", "1G")
- Automatic page alignment and size validation
- Flexible allocation patterns (1MB, 2MB, 4MB, etc.)
- Memory size limits: 1MB minimum, 256MB maximum

### 3. **Cache Control on Large Blocks**
- Per-allocation cache state control (cached/uncached/toggle)
- Uses `set_memory_uc()` and `set_memory_wb()` for cache attribute changes
- Maintains cache state across large memory regions
- Supports runtime cache state transitions

### 4. **DMA-Compatible Memory Management**
- Allocates DMA-coherent memory suitable for device operations
- Provides both virtual and physical (DMA) addresses
- Ensures memory is accessible from both CPU and potential DMA devices
- Proper cleanup with cache state restoration

## Architecture

```
User Space                    Kernel Space
-----------                   -------------

┌─────────────────┐          ┌──────────────────────┐
│   CMA Test      │          │  Sysfs Interface     │
│   Program       │◄────────►│ /sys/kernel/         │
│                 │          │   cma_cache/         │
│ ./cma_test      │          │                      │
└─────────────────┘          └──────────────────────┘
         │                   │
         │ mmap()            │
         ▼                   ▼
┌─────────────────┐          ┌──────────────────────┐
│   /dev/         │◄────────►│  DMA Allocation      │
│  cma_cache      │          │  Manager             │
└─────────────────┘          │                      │
                              │ ┌──────────────────┐ │
                              │ │ Platform Device  │ │
                              │ │ (for DMA ops)    │ │
                              │ └──────────────────┘ │
                              │                      │
                              │ ┌──────────────────┐ │
                              │ │ DMA Coherent     │ │
                              │ │ Memory Pool      │ │
                              │ │                  │ │
                              │ │ • 1MB - 256MB    │ │
                              │ │ • Physically     │ │
                              │ │   contiguous     │ │
                              │ │ • Cache control  │ │
                              │ └──────────────────┘ │
                              │                      │
                              │ ┌──────────────────┐ │
                              │ │ Allocation       │ │
                              │ │ Tracking         │ │
                              │ │                  │ │
                              │ │ • virt_addr      │ │
                              │ │ • dma_handle     │ │
                              │ │ • size           │ │
                              │ │ • cache_state    │ │
                              │ │ • alloc_id       │ │
                              │ └──────────────────┘ │
                              └──────────────────────┘
```

## Usage Examples

### Basic Large Block Allocation

```bash
# Load the module
sudo insmod cma_cache.ko

# Check initial status
cat /sys/kernel/cma_cache/status

# Allocate various sized blocks
echo "alloc 1M" > /sys/kernel/cma_cache/command     # 1MB block
echo "alloc 4M" > /sys/kernel/cma_cache/command     # 4MB block  
echo "alloc 16M" > /sys/kernel/cma_cache/command    # 16MB block
echo "alloc 64M" > /sys/kernel/cma_cache/command    # 64MB block

# Check allocation status
cat /sys/kernel/cma_cache/status
```

### Cache Control on Large Blocks

```bash
# Set different cache states
echo "uncache 1" > /sys/kernel/cma_cache/command    # Set 1MB block as uncached
echo "cache 2" > /sys/kernel/cma_cache/command      # Set 4MB block as cached
echo "toggle 3" > /sys/kernel/cma_cache/command     # Toggle 16MB block cache state

# View current states
cat /sys/kernel/cma_cache/status
```

#### Permission Requirements
```bash
# Sysfs files need write permissions for non-root users:
sudo chmod 666 /sys/kernel/cma_cache/command
sudo chmod 666 /dev/cma_cache
```

### Memory Mapping Large Blocks

```c
// Open device file
int fd = open("/dev/cma_cache", O_RDWR);

// Map allocation 1 (1MB block, allocation ID 1)
void *addr = mmap(NULL, 1024*1024, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 1 * getpagesize());

// Access the large contiguous memory block
volatile uint64_t *data = (volatile uint64_t *)addr;
for (int i = 0; i < 131072; i++) {  // 1MB / 8 bytes
    data[i] = 0x1234567890ABCDEF + i;
}

// Unmap and cleanup
munmap(addr, 1024*1024);
close(fd);
```

## Implementation Details

### DMA Allocation Management

```c
struct cma_allocation {
    void *virt_addr;                 // Virtual address of allocation
    dma_addr_t dma_handle;          // DMA address (physical address)
    size_t size;                    // Size in bytes
    int num_pages;                  // Number of pages
    int is_cached;                  // Cache state (1=cached, 0=uncached)
    int allocated;                  // Whether this slot is in use
    int alloc_id;                   // Unique allocation identifier
};

static struct cma_allocation allocations[MAX_CMA_ALLOCATIONS]; // Up to 32 allocations
```

### Platform Device for DMA Operations

```c
// Create platform device for DMA/CMA operations
cma_pdev = platform_device_alloc("cma_cache", -1);
platform_device_add(cma_pdev);

// Set DMA mask for the platform device
// If you set this to 32 bits, address ranges cannot exceed 4GB and you may run into issues with larger allocations
dma_set_mask_and_coherent(&cma_pdev->dev, DMA_BIT_MASK(64));
```

### Large Block Allocation Algorithm

```c
static int allocate_cma_memory(size_t size)
{
    // 1. Validate size (1MB min, 256MB max)
    // 2. Page-align the size
    // 3. Find free allocation slot
    // 4. Allocate using dma_alloc_coherent()
    // 5. Initialize allocation metadata
    // 6. Return allocation ID
    
    alloc->virt_addr = dma_alloc_coherent(&cma_pdev->dev, size, 
                                         &alloc->dma_handle, GFP_KERNEL);
}
```

### Cache Control on Large Regions

```c
static int set_cma_cache_state(struct cma_allocation *alloc, int cached)
{
    unsigned long virt = (unsigned long)alloc->virt_addr;
    int num_pages = alloc->num_pages;
    
    if (cached) {
        return set_memory_wb(virt, num_pages);  // Set as write-back cached
    } else {
        return set_memory_uc(virt, num_pages);  // Set as uncached
    }
}
```

### Status Display

The status interface provides comprehensive information about large allocations:

```
DMA Cache Control Status
========================
Total allocations: 4/32
Total allocated memory: 85196800 bytes (81.3 MB)
Cached allocations: 2
Uncached allocations: 2

Active Allocations:
ID   Size       Pages  Virtual     Physical    Cache State
---  ---------  -----  ----------  ----------  -----------
  1     1024K    256  ffff888...  0x81700000  UNCACHED
  2     4096K   1024  ffff888...  0x82000000  CACHED
  3    16384K   4096  ffff888...  0x83000000  UNCACHED
  4    65536K  16384  ffff888...  0x84000000  CACHED

Size limits: 1024K - 262144K
```

## Performance Characteristics

### Allocation Performance
- **Small blocks (1-4MB)**: Near-instantaneous allocation
- **Medium blocks (16-64MB)**: Allocation time 100-500ms
- **Large blocks (128-256MB)**: Allocation time 1-5 seconds
- **Physical contiguity**: Guaranteed across entire allocation

### Cache Performance Impact
- **Cached large blocks**: ~6-10 ns per access
- **Uncached large blocks**: ~30-70 ns per access  
- **Performance ratio**: 5-7x slower for uncached memory
- **Block size impact**: Larger blocks show more pronounced cache effects

### Memory Usage Efficiency
- **Physical memory**: Directly mapped, no fragmentation within blocks
- **Virtual memory**: Efficient mapping with proper cache attributes
- **DMA compatibility**: Zero-copy access for DMA devices
- **Resource overhead**: Minimal per-allocation tracking

## Educational Value

The CMA cache module provides insights into:

### **Advanced Memory Management**
- Large contiguous memory allocation techniques
- DMA-coherent memory management
- Physical memory layout and contiguity
- Platform device and DMA subsystem integration

### **Cache Behavior at Scale**
- Cache effects on large memory regions
- Performance scaling with memory size
- Cache coherency across large allocations
- Memory access pattern optimization

### **Real-World Applications**
- Device driver memory management
- Graphics and media buffer allocation
- Network packet buffer management
- High-performance computing memory pools

### **System Integration**
- Kernel-userspace memory sharing
- sysfs interface design patterns
- Character device implementation
- Resource management and cleanup

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

### Usage Notes

- **Memory allocation**: Choose API based on size and contiguity requirements
- **Cache control**: Always restore cache attributes before freeing memory  
- **Memory mapping**: Use `remap_pfn_range()` for physical memory, `vm_insert_page()` for vmalloc
- **Error handling**: All APIs can fail; always check return values
- **Cleanup order**: Reverse of initialization order to avoid resource leaks

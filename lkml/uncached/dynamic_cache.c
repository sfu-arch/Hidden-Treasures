#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/uaccess.h>
#include <linux/slab.h>
#include <linux/mm.h>
#include <linux/vmalloc.h>
#include <linux/gfp.h>
#include <linux/io.h>
#include <linux/kobject.h>
#include <linux/sysfs.h>
#include <linux/ctype.h>
#include <linux/cdev.h>
#include <linux/device.h>
#include <linux/mman.h>

// Include architecture-specific headers
#ifdef CONFIG_X86
#include <asm/cacheflush.h>
#include <asm/set_memory.h>
#include <asm/pgtable.h>
#include <asm/tlbflush.h>
#endif

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Your Name");
MODULE_DESCRIPTION("Dynamic page-level cache control using page protection mechanisms");
MODULE_VERSION("1.0");

#define MAX_PAGES 1024  // Maximum number of pages we can manage
#define PAGE_POOL_SIZE (MAX_PAGES * PAGE_SIZE)
#define MAX_ALLOCATION_SIZE (64UL * 1024 * 1024) // 64MB limit for dynamic cache

// Page allocation block structure for contiguous allocations
struct page_block {
    int start_idx;                    // Starting page index in the pool
    int num_pages;                    // Number of contiguous pages in this block
    int block_id;                     // Unique block identifier
    int allocated;                    // Whether this block is in use
};

// Page state tracking
struct page_info {
    void *virt_addr;              // Virtual address of the page
    struct page *page;            // Page structure pointer
    unsigned long pfn;            // Page frame number
    int is_cached;                // Current cache state (1=cached, 0=uncached)
    int allocated;                // Whether this slot is in use
    int block_id;                 // Which block this page belongs to (-1 if single page)
};

// Global state
static struct kobject *dynamic_kobj;
static dev_t dev_num;
static struct cdev dynamic_cdev;
static struct class *dynamic_class;
static struct device *dynamic_device;

// Memory pool management
static void *page_pool = NULL;           // Base address of our page pool
static struct page_info pages[MAX_PAGES]; // Page tracking array
static struct page_block blocks[MAX_PAGES]; // Block tracking for contiguous allocations
static int num_allocated_pages = 0;
static int next_block_id = 1;            // Auto-incrementing block ID
static DEFINE_MUTEX(page_mutex);         // Protect page operations

// Size parsing function with K/M/G suffix support (from uncached_mem.c)
static int parse_size_string(const char *str, size_t *size)
{
    char *endptr;
    unsigned long val;
    char suffix;
    
    val = simple_strtoul(str, &endptr, 10);
    
    if (endptr == str) {
        return -EINVAL;
    }
    
    if (*endptr != '\0') {
        if (strlen(endptr) == 1) {
            suffix = tolower(*endptr);
            switch (suffix) {
            case 'k':
                val *= 1024;
                break;
            case 'm':
                val *= 1024 * 1024;
                break;
            case 'g':
                val *= 1024 * 1024 * 1024;
                break;
            default:
                return -EINVAL;
            }
        } else {
            return -EINVAL;
        }
    }
    
    *size = val;
    return 0;
}

// Page allocation and management functions

static int allocate_page_pool(void)
{
    int i;
    unsigned long addr;
    
    // Allocate a large contiguous region
    page_pool = vmalloc(PAGE_POOL_SIZE);
    if (!page_pool) {
        printk(KERN_ERR "Failed to allocate page pool\n");
        return -ENOMEM;
    }
    
    // Initialize page tracking structures
    memset(pages, 0, sizeof(pages));
    memset(blocks, 0, sizeof(blocks));
    addr = (unsigned long)page_pool;
    
    for (i = 0; i < MAX_PAGES; i++) {
        pages[i].virt_addr = (void *)(addr + i * PAGE_SIZE);
        pages[i].page = vmalloc_to_page(pages[i].virt_addr);
        if (pages[i].page) {
            pages[i].pfn = page_to_pfn(pages[i].page);
        }
        pages[i].is_cached = 1;  // Start as cached
        pages[i].allocated = 0;   // Not allocated to user
        pages[i].block_id = -1;   // Not part of any block
        
        // Initialize block tracking
        blocks[i].start_idx = -1;
        blocks[i].num_pages = 0;
        blocks[i].block_id = -1;
        blocks[i].allocated = 0;
    }
    
    printk(KERN_INFO "Allocated page pool: %d pages at %p\n", MAX_PAGES, page_pool);
    return 0;
}

static void free_page_pool(void)
{
    int i;
    
    if (!page_pool)
        return;
        
    // Restore all pages to cached state before freeing
    for (i = 0; i < MAX_PAGES; i++) {
        if (pages[i].page && !pages[i].is_cached) {
            #ifdef CONFIG_X86
            set_memory_wb((unsigned long)pages[i].virt_addr, 1);
            #endif
        }
    }
    
    vfree(page_pool);
    page_pool = NULL;
    printk(KERN_INFO "Freed page pool\n");
}

// Dynamic cache control functions

static int set_page_cache_state(int page_idx, int cached)
{
    int ret = 0;
    
    if (page_idx < 0 || page_idx >= MAX_PAGES) {
        return -EINVAL;
    }
    
    if (!pages[page_idx].page) {
        return -EINVAL;
    }
    
    if (pages[page_idx].is_cached == cached) {
        return 0; // Already in desired state
    }
    
    #ifdef CONFIG_X86
    if (cached) {
        // Set page as cached (write-back)
        ret = set_memory_wb((unsigned long)pages[page_idx].virt_addr, 1);
        if (ret == 0) {
            pages[page_idx].is_cached = 1;
            printk(KERN_INFO "Page %d set to CACHED\n", page_idx);
        }
    } else {
        // Set page as uncached
        ret = set_memory_uc((unsigned long)pages[page_idx].virt_addr, 1);
        if (ret == 0) {
            pages[page_idx].is_cached = 0;
            printk(KERN_INFO "Page %d set to UNCACHED\n", page_idx);
        }
    }
    
    if (ret != 0 && ret != -ENOTSUPP) {
        printk(KERN_WARNING "Failed to change cache state for page %d: %d\n", page_idx, ret);
    }
    #else
    printk(KERN_WARNING "Cache control not supported on this architecture\n");
    ret = -ENOTSUPP;
    #endif
    
    // Flush TLB to ensure changes take effect
    // Note: flush_tlb_kernel_range may not be available on all kernels
    // Use __flush_tlb_one or similar if available
    #ifdef CONFIG_X86
    __flush_tlb_one_user((unsigned long)pages[page_idx].virt_addr);
    #endif
    
    return ret;
}

static int allocate_user_page(void)
{
    int i;
    
    mutex_lock(&page_mutex);
    
    // Find an unallocated page
    for (i = 0; i < MAX_PAGES; i++) {
        if (!pages[i].allocated) {
            pages[i].allocated = 1;
            num_allocated_pages++;
            mutex_unlock(&page_mutex);
            
            // Initialize page with test pattern
            memset(pages[i].virt_addr, 0xAA, PAGE_SIZE);
            
            printk(KERN_INFO "Allocated page %d to user (cached=%d)\n", i, pages[i].is_cached);
            return i;
        }
    }
    
    mutex_unlock(&page_mutex);
    return -ENOMEM; // No free pages
}

static int free_user_page(int page_idx)
{
    if (page_idx < 0 || page_idx >= MAX_PAGES) {
        return -EINVAL;
    }
    
    mutex_lock(&page_mutex);
    
    if (!pages[page_idx].allocated) {
        mutex_unlock(&page_mutex);
        return -EINVAL;
    }
    
    // Check if this page is part of a block
    if (pages[page_idx].block_id != -1) {
        mutex_unlock(&page_mutex);
        printk(KERN_WARNING "Cannot free individual page %d - it's part of block %d. Use 'free_block %d'\n", 
               page_idx, pages[page_idx].block_id, pages[page_idx].block_id);
        return -EPERM;
    }
    
    // Restore to cached state
    if (!pages[page_idx].is_cached) {
        set_page_cache_state(page_idx, 1);
    }
    
    pages[page_idx].allocated = 0;
    pages[page_idx].block_id = -1;
    num_allocated_pages--;
    
    mutex_unlock(&page_mutex);
    
    printk(KERN_INFO "Freed page %d\n", page_idx);
    return 0;
}

// Find contiguous free pages in the pool
static int find_contiguous_pages(int num_pages)
{
    int i, j, found;
    
    if (num_pages <= 0 || num_pages > MAX_PAGES) {
        return -1;
    }
    
    for (i = 0; i <= MAX_PAGES - num_pages; i++) {
        found = 1;
        for (j = 0; j < num_pages; j++) {
            if (pages[i + j].allocated) {
                found = 0;
                break;
            }
        }
        if (found) {
            return i; // Return starting index
        }
    }
    
    return -1; // No contiguous region found
}

// Allocate contiguous pages and assign them to a block
static int allocate_user_pages(int num_pages)
{
    int start_idx, i, block_idx;
    int current_block_id;
    
    if (num_pages <= 0) {
        return -EINVAL;
    }
    
    // Convert size to pages (already done by caller, but double-check)
    if (num_pages > MAX_PAGES) {
        printk(KERN_ERR "Requested %d pages exceeds maximum %d\n", num_pages, MAX_PAGES);
        return -EINVAL;
    }
    
    mutex_lock(&page_mutex);
    
    // Find contiguous free pages
    start_idx = find_contiguous_pages(num_pages);
    if (start_idx < 0) {
        mutex_unlock(&page_mutex);
        printk(KERN_WARNING "Cannot find %d contiguous free pages\n", num_pages);
        return -ENOMEM;
    }
    
    // Find a free block slot
    block_idx = -1;
    for (i = 0; i < MAX_PAGES; i++) {
        if (!blocks[i].allocated) {
            block_idx = i;
            break;
        }
    }
    
    if (block_idx < 0) {
        mutex_unlock(&page_mutex);
        printk(KERN_ERR "No free block slots available\n");
        return -ENOMEM;
    }
    
    current_block_id = next_block_id++;
    
    // Initialize the block
    blocks[block_idx].start_idx = start_idx;
    blocks[block_idx].num_pages = num_pages;
    blocks[block_idx].block_id = current_block_id;
    blocks[block_idx].allocated = 1;
    
    // Mark all pages in the block as allocated
    for (i = 0; i < num_pages; i++) {
        pages[start_idx + i].allocated = 1;
        pages[start_idx + i].block_id = current_block_id;
        // Initialize page with test pattern
        memset(pages[start_idx + i].virt_addr, 0xAA + (i % 16), PAGE_SIZE);
    }
    
    num_allocated_pages += num_pages;
    
    mutex_unlock(&page_mutex);
    
    printk(KERN_INFO "Allocated block %d: %d contiguous pages starting at page %d\n", 
           current_block_id, num_pages, start_idx);
    return current_block_id;
}

// Find block by block ID
static int find_block_by_id(int block_id)
{
    int i;
    
    for (i = 0; i < MAX_PAGES; i++) {
        if (blocks[i].allocated && blocks[i].block_id == block_id) {
            return i;
        }
    }
    return -1;
}

// Free a contiguous block
static int free_user_block(int block_id)
{
    int block_idx, i, start_idx, num_pages;
    
    mutex_lock(&page_mutex);
    
    block_idx = find_block_by_id(block_id);
    if (block_idx < 0) {
        mutex_unlock(&page_mutex);
        printk(KERN_WARNING "Block ID %d not found\n", block_id);
        return -ENOENT;
    }
    
    start_idx = blocks[block_idx].start_idx;
    num_pages = blocks[block_idx].num_pages;
    
    // Restore all pages in block to cached state and free them
    for (i = 0; i < num_pages; i++) {
        int page_idx = start_idx + i;
        if (!pages[page_idx].is_cached) {
            set_page_cache_state(page_idx, 1);
        }
        pages[page_idx].allocated = 0;
        pages[page_idx].block_id = -1;
    }
    
    // Free the block
    blocks[block_idx].allocated = 0;
    blocks[block_idx].start_idx = -1;
    blocks[block_idx].num_pages = 0;
    blocks[block_idx].block_id = -1;
    
    num_allocated_pages -= num_pages;
    
    mutex_unlock(&page_mutex);
    
    printk(KERN_INFO "Freed block %d (%d pages starting at %d)\n", block_id, num_pages, start_idx);
    return 0;
}

// Simple kernel-space string tokenizer
static char *simple_strtok(char *str, const char *delim, char **saveptr)
{
    char *start, *end;
    
    if (str != NULL) {
        *saveptr = str;
    }
    
    if (*saveptr == NULL) {
        return NULL;
    }
    
    // Skip leading delimiters
    start = *saveptr;
    while (*start && strchr(delim, *start)) {
        start++;
    }
    
    if (*start == '\0') {
        *saveptr = NULL;
        return NULL;
    }
    
    // Find end of token
    end = start;
    while (*end && !strchr(delim, *end)) {
        end++;
    }
    
    if (*end != '\0') {
        *end = '\0';
        *saveptr = end + 1;
    } else {
        *saveptr = NULL;
    }
    
    return start;
}

static ssize_t command_show(struct kobject *kobj, struct kobj_attribute *attr, char *buf)
{
    return sprintf(buf, "Dynamic Cache Control Commands:\n"
                        "alloc [size]             - Allocate page(s) (size: bytes, K, M, G suffix)\n"
                        "free <page_id>           - Free a single page\n"
                        "free_block <block_id>    - Free a contiguous block\n"
                        "cache <page_id>          - Set page as cached\n"
                        "uncache <page_id>        - Set page as uncached\n"
                        "toggle <page_id>         - Toggle cache state\n"
                        "pattern <page_id> <val>  - Set test pattern\n"
                        "\nExamples:\n"
                        "echo 'alloc' > command           # Allocate 1 page\n"
                        "echo 'alloc 8192' > command      # Allocate 2 pages (8KB)\n"
                        "echo 'alloc 1M' > command        # Allocate 256 pages (1MB)\n"
                        "echo 'free 5' > command          # Free single page 5\n"
                        "echo 'free_block 3' > command    # Free block 3\n"
                        "echo 'uncache 5' > command       # Set page 5 uncached\n"
                        "echo 'toggle 3' > command        # Toggle page 3 cache state\n");
}

static ssize_t command_store(struct kobject *kobj, struct kobj_attribute *attr,
                            const char *buf, size_t count)
{
    char cmd[64];
    char *token, *saveptr;
    char *cmd_name;
    int page_idx, value;
    int ret = 0;
    
    if (count >= sizeof(cmd)) {
        return -EINVAL;
    }
    
    strncpy(cmd, buf, count);
    cmd[count] = '\0';
    
    // Remove trailing newline
    if (count > 0 && cmd[count-1] == '\n') {
        cmd[count-1] = '\0';
    }
    
    // Parse command
    cmd_name = simple_strtok(cmd, " ", &saveptr);
    if (!cmd_name) {
        return -EINVAL;
    }
    
    if (strcmp(cmd_name, "alloc") == 0) {
        // Check if size parameter is provided
        token = simple_strtok(NULL, " ", &saveptr);
        
        if (token) {
            // Parse size with optional suffixes
            size_t size;
            int num_pages;
            
            if (parse_size_string(token, &size) != 0) {
                printk(KERN_WARNING "Invalid size format. Use bytes, or append K/M/G (e.g., 4K, 1M)\n");
                return -EINVAL;
            }
            
            // Validate size limits
            if (size < PAGE_SIZE) {
                printk(KERN_WARNING "Size too small, minimum is %lu bytes\n", PAGE_SIZE);
                return -EINVAL;
            }
            if (size > MAX_ALLOCATION_SIZE) {
                printk(KERN_WARNING "Size too large, maximum is %lu bytes\n", MAX_ALLOCATION_SIZE);
                return -EINVAL;
            }
            
            // Round up to page boundary and convert to pages
            size = PAGE_ALIGN(size);
            num_pages = size / PAGE_SIZE;
            
            if (num_pages == 1) {
                // Single page allocation
                ret = allocate_user_page();
                if (ret < 0) {
                    printk(KERN_WARNING "Failed to allocate page: %d\n", ret);
                    return ret;
                }
                printk(KERN_INFO "Allocated single page %d (%zu bytes)\n", ret, size);
            } else {
                // Multi-page allocation
                ret = allocate_user_pages(num_pages);
                if (ret < 0) {
                    printk(KERN_WARNING "Failed to allocate %d pages: %d\n", num_pages, ret);
                    return ret;
                }
                printk(KERN_INFO "Allocated block %d with %d pages (%zu bytes)\n", ret, num_pages, size);
            }
        } else {
            // Default single page allocation
            ret = allocate_user_page();
            if (ret < 0) {
                printk(KERN_WARNING "Failed to allocate page: %d\n", ret);
                return ret;
            }
            printk(KERN_INFO "Allocated single page %d (default size)\n", ret);
        }
        
    } else if (strcmp(cmd_name, "free") == 0) {
        token = simple_strtok(NULL, " ", &saveptr);
        if (!token || kstrtoint(token, 10, &page_idx) != 0) {
            return -EINVAL;
        }
        
        ret = free_user_page(page_idx);
        if (ret < 0) {
            return ret;
        }
        
    } else if (strcmp(cmd_name, "free_block") == 0) {
        int block_id;
        token = simple_strtok(NULL, " ", &saveptr);
        if (!token || kstrtoint(token, 10, &block_id) != 0) {
            return -EINVAL;
        }
        
        ret = free_user_block(block_id);
        if (ret < 0) {
            return ret;
        }
        
    } else if (strcmp(cmd_name, "cache") == 0) {
        token = simple_strtok(NULL, " ", &saveptr);
        if (!token || kstrtoint(token, 10, &page_idx) != 0) {
            return -EINVAL;
        }
        
        if (page_idx < 0 || page_idx >= MAX_PAGES || !pages[page_idx].allocated) {
            return -EINVAL;
        }
        
        ret = set_page_cache_state(page_idx, 1);
        if (ret < 0) {
            return ret;
        }
        
    } else if (strcmp(cmd_name, "uncache") == 0) {
        token = simple_strtok(NULL, " ", &saveptr);
        if (!token || kstrtoint(token, 10, &page_idx) != 0) {
            return -EINVAL;
        }
        
        if (page_idx < 0 || page_idx >= MAX_PAGES || !pages[page_idx].allocated) {
            return -EINVAL;
        }
        
        ret = set_page_cache_state(page_idx, 0);
        if (ret < 0) {
            return ret;
        }
        
    } else if (strcmp(cmd_name, "toggle") == 0) {
        token = simple_strtok(NULL, " ", &saveptr);
        if (!token || kstrtoint(token, 10, &page_idx) != 0) {
            return -EINVAL;
        }
        
        if (page_idx < 0 || page_idx >= MAX_PAGES || !pages[page_idx].allocated) {
            return -EINVAL;
        }
        
        ret = set_page_cache_state(page_idx, !pages[page_idx].is_cached);
        if (ret < 0) {
            return ret;
        }
        
    } else if (strcmp(cmd_name, "pattern") == 0) {
        token = simple_strtok(NULL, " ", &saveptr);
        if (!token || kstrtoint(token, 10, &page_idx) != 0) {
            return -EINVAL;
        }
        
        token = simple_strtok(NULL, " ", &saveptr);
        if (!token || kstrtoint(token, 16, &value) != 0) {
            return -EINVAL;
        }
        
        if (page_idx < 0 || page_idx >= MAX_PAGES || !pages[page_idx].allocated) {
            return -EINVAL;
        }
        
        memset(pages[page_idx].virt_addr, value & 0xFF, PAGE_SIZE);
        printk(KERN_INFO "Set pattern 0x%02x on page %d\n", value & 0xFF, page_idx);
        
    } else {
        return -EINVAL;
    }
    
    return count;
}

static ssize_t status_show(struct kobject *kobj, struct kobj_attribute *attr, char *buf)
{
    int i, len = 0;
    int cached_count = 0, uncached_count = 0;
    int active_blocks = 0;
    
    len += sprintf(buf + len, "Dynamic Cache Control Status\n");
    len += sprintf(buf + len, "===========================\n");
    len += sprintf(buf + len, "Total pages: %d\n", MAX_PAGES);
    len += sprintf(buf + len, "Allocated pages: %d\n", num_allocated_pages);
    len += sprintf(buf + len, "Page pool base: %p\n", page_pool);
    
    // Count active blocks
    for (i = 0; i < MAX_PAGES; i++) {
        if (blocks[i].allocated) {
            active_blocks++;
        }
    }
    len += sprintf(buf + len, "Active blocks: %d\n", active_blocks);
    
    // Show block information
    if (active_blocks > 0) {
        len += sprintf(buf + len, "\nActive Blocks:\n");
        len += sprintf(buf + len, "Block ID  Start  Pages  Size\n");
        len += sprintf(buf + len, "--------  -----  -----  ----\n");
        
        for (i = 0; i < MAX_PAGES && len < PAGE_SIZE - 200; i++) {
            if (blocks[i].allocated) {
                len += sprintf(buf + len, "%8d  %5d  %5d  %dK\n",
                              blocks[i].block_id, blocks[i].start_idx, 
                              blocks[i].num_pages, blocks[i].num_pages * 4);
            }
        }
    }
    
    len += sprintf(buf + len, "\nAllocated Pages:\n");
    len += sprintf(buf + len, "ID   Virtual     PFN        Block    Cache State\n");
    len += sprintf(buf + len, "---  ----------  ---------  -------  -----------\n");
    
    for (i = 0; i < MAX_PAGES && len < PAGE_SIZE - 100; i++) {
        if (pages[i].allocated) {
            if (pages[i].block_id == -1) {
                len += sprintf(buf + len, "%3d  %p  %09lx  %7s  %s\n", 
                              i, pages[i].virt_addr, pages[i].pfn,
                              "single",
                              pages[i].is_cached ? "CACHED" : "UNCACHED");
            } else {
                len += sprintf(buf + len, "%3d  %p  %09lx  %7d  %s\n", 
                              i, pages[i].virt_addr, pages[i].pfn,
                              pages[i].block_id,
                              pages[i].is_cached ? "CACHED" : "UNCACHED");
            }
            
            if (pages[i].is_cached) {
                cached_count++;
            } else {
                uncached_count++;
            }
        }
    }
    
    len += sprintf(buf + len, "\nSummary: %d cached, %d uncached\n", cached_count, uncached_count);
    
    return len;
}

static ssize_t page_map_show(struct kobject *kobj, struct kobj_attribute *attr, char *buf)
{
    int i, len = 0;
    
    len += sprintf(buf + len, "Page Allocation Map (C=cached, U=uncached, .=free):\n");
    
    for (i = 0; i < MAX_PAGES && len < PAGE_SIZE - 50; i++) {
        if (i % 64 == 0 && i > 0) {
            len += sprintf(buf + len, "\n");
        }
        if (i % 64 == 0) {
            len += sprintf(buf + len, "%4d: ", i);
        }
        
        if (!pages[i].allocated) {
            len += sprintf(buf + len, ".");
        } else if (pages[i].is_cached) {
            len += sprintf(buf + len, "C");
        } else {
            len += sprintf(buf + len, "U");
        }
    }
    len += sprintf(buf + len, "\n");
    
    return len;
}

// Attribute definitions
static struct kobj_attribute command_attr = __ATTR(command, 0664, command_show, command_store);
static struct kobj_attribute status_attr = __ATTR(status, 0444, status_show, NULL);
static struct kobj_attribute page_map_attr = __ATTR(page_map, 0444, page_map_show, NULL);

static struct attribute *attrs[] = {
    &command_attr.attr,
    &status_attr.attr,
    &page_map_attr.attr,
    NULL,
};

static struct attribute_group attr_group = {
    .attrs = attrs,
};

// Character device functions for mmap

static int device_open(struct inode *inode, struct file *file)
{
    return 0;
}

static int device_release(struct inode *inode, struct file *file)
{
    return 0;
}

static int device_mmap(struct file *file, struct vm_area_struct *vma)
{
    unsigned long size = vma->vm_end - vma->vm_start;
    unsigned long offset = vma->vm_pgoff;
    int page_idx = offset; // Page index is passed as offset
    struct page *page;
    int ret;
    
    printk(KERN_INFO "mmap request: size=%lu, offset=%lu, page_idx=%d\n", size, offset, page_idx);
    
    if (size != PAGE_SIZE) {
        printk(KERN_ERR "mmap size must be exactly one page (%lu bytes)\n", PAGE_SIZE);
        return -EINVAL;
    }
    
    if (page_idx < 0 || page_idx >= MAX_PAGES) {
        printk(KERN_ERR "Invalid page index: %d (offset was %lu)\n", page_idx, offset);
        return -EINVAL;
    }
    
    if (!pages[page_idx].allocated) {
        printk(KERN_ERR "Page %d not allocated\n", page_idx);
        return -EINVAL;
    }
    
    if (!pages[page_idx].page) {
        printk(KERN_ERR "Page %d has no page structure\n", page_idx);
        return -EINVAL;
    }
    
    page = pages[page_idx].page;
    
    // Set up VMA flags for vmalloc pages
    vma->vm_flags |= VM_DONTEXPAND | VM_DONTDUMP;
    
    // Set page protection based on cache state
    if (!pages[page_idx].is_cached) {
        vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot);
        printk(KERN_INFO "Mapping page %d as UNCACHED\n", page_idx);
    } else {
        printk(KERN_INFO "Mapping page %d as CACHED\n", page_idx);
    }
    
    // For vmalloc pages, use vm_insert_page instead of remap_pfn_range
    ret = vm_insert_page(vma, vma->vm_start, page);
    if (ret) {
        printk(KERN_ERR "vm_insert_page failed for page %d: %d\n", page_idx, ret);
        return ret;
    }
    
    printk(KERN_INFO "Successfully mapped page %d (pfn=0x%lx) to user space\n", 
           page_idx, page_to_pfn(page));
    return 0;
}

static const struct file_operations device_fops = {
    .open = device_open,
    .release = device_release,
    .mmap = device_mmap,
};

// Module initialization and cleanup

static int __init dynamic_cache_init(void)
{
    int ret;
    
    printk(KERN_INFO "Dynamic cache control module loading\n");
    
    // Allocate page pool
    ret = allocate_page_pool();
    if (ret) {
        return ret;
    }
    
    // Create sysfs interface
    dynamic_kobj = kobject_create_and_add("dynamic_cache", kernel_kobj);
    if (!dynamic_kobj) {
        printk(KERN_ERR "Failed to create sysfs kobject\n");
        free_page_pool();
        return -ENOMEM;
    }
    
    ret = sysfs_create_group(dynamic_kobj, &attr_group);
    if (ret) {
        printk(KERN_ERR "Failed to create sysfs attribute group\n");
        kobject_put(dynamic_kobj);
        free_page_pool();
        return ret;
    }
    
    // Create character device
    ret = alloc_chrdev_region(&dev_num, 0, 1, "dynamic_cache");
    if (ret < 0) {
        printk(KERN_ERR "Failed to allocate character device region\n");
        goto cleanup_sysfs;
    }
    
    cdev_init(&dynamic_cdev, &device_fops);
    ret = cdev_add(&dynamic_cdev, dev_num, 1);
    if (ret < 0) {
        printk(KERN_ERR "Failed to add character device\n");
        goto cleanup_chrdev;
    }
    
    dynamic_class = class_create(THIS_MODULE, "dynamic_cache");
    if (IS_ERR(dynamic_class)) {
        printk(KERN_ERR "Failed to create device class\n");
        ret = PTR_ERR(dynamic_class);
        goto cleanup_cdev;
    }
    
    dynamic_device = device_create(dynamic_class, NULL, dev_num, NULL, "dynamic_cache");
    if (IS_ERR(dynamic_device)) {
        printk(KERN_ERR "Failed to create device\n");
        ret = PTR_ERR(dynamic_device);
        goto cleanup_class;
    }
    
    printk(KERN_INFO "Dynamic cache control module loaded successfully\n");
    printk(KERN_INFO "Sysfs interface: /sys/kernel/dynamic_cache/\n");
    printk(KERN_INFO "Device file: /dev/dynamic_cache\n");
    printk(KERN_INFO "Page pool: %d pages allocated\n", MAX_PAGES);
    
    return 0;
    
cleanup_class:
    class_destroy(dynamic_class);
cleanup_cdev:
    cdev_del(&dynamic_cdev);
cleanup_chrdev:
    unregister_chrdev_region(dev_num, 1);
cleanup_sysfs:
    sysfs_remove_group(dynamic_kobj, &attr_group);
    kobject_put(dynamic_kobj);
    free_page_pool();
    return ret;
}

static void __exit dynamic_cache_exit(void)
{
    int i;
    
    // Free all allocated pages
    for (i = 0; i < MAX_PAGES; i++) {
        if (pages[i].allocated) {
            free_user_page(i);
        }
    }
    
    // Clean up device and sysfs
    if (dynamic_device) {
        device_destroy(dynamic_class, dev_num);
    }
    if (dynamic_class) {
        class_destroy(dynamic_class);
    }
    cdev_del(&dynamic_cdev);
    unregister_chrdev_region(dev_num, 1);
    
    sysfs_remove_group(dynamic_kobj, &attr_group);
    kobject_put(dynamic_kobj);
    
    free_page_pool();
    
    printk(KERN_INFO "Dynamic cache control module unloaded\n");
}

module_init(dynamic_cache_init);
module_exit(dynamic_cache_exit);

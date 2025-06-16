/*
 * CMA Cache Control Module
 * 
 * This kernel module demonstrates:
 * - DMA coherent allocation for large contiguous memory blocks
 * - Uses CMA backend when available for guaranteed physical contiguity
 * - Cache attribute control on DMA-allocated memory
 * - Memory mapping large contiguous blocks to userspace with different cache attributes
 * - Educational use for understanding large contiguous memory allocation and cache control
 *
 * Features:
 * - Variable size DMA allocation (1MB to 256MB)
 * - Per-allocation cache control (cached/uncached)
 * - sysfs interface for control and monitoring
 * - Character device for mmap operations
 * - Comprehensive status reporting
 *
 * License: GPL v2
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/slab.h>
#include <linux/mm.h>
#include <linux/dma-mapping.h>
#include <linux/device.h>
#include <linux/cdev.h>
#include <linux/fs.h>
#include <linux/uaccess.h>
#include <linux/mman.h>
#include <linux/kobject.h>
#include <linux/sysfs.h>
#include <linux/string.h>
#include <linux/mutex.h>
#include <linux/platform_device.h>
#include <linux/page-flags.h>
#include <linux/ctype.h>
#include <asm/set_memory.h>

#define MODULE_NAME "cma_cache"
#define MAX_CMA_ALLOCATIONS 32
#define MIN_ALLOCATION_SIZE (1024 * 1024)     // 1MB minimum
#define MAX_ALLOCATION_SIZE (256 * 1024 * 1024) // 256MB maximum

MODULE_LICENSE("GPL v2");
MODULE_AUTHOR("Educational Module");
MODULE_DESCRIPTION("DMA coherent memory allocation with cache control");
MODULE_VERSION("1.0");

// DMA allocation tracking structure
struct cma_allocation {
    void *virt_addr;                 // Virtual address of allocation
    dma_addr_t dma_handle;          // DMA address (physical address)
    size_t size;                    // Size in bytes
    int num_pages;                  // Number of pages
    int is_cached;                  // Cache state (1=cached, 0=uncached)
    int allocated;                  // Whether this slot is in use
    int alloc_id;                   // Unique allocation identifier
};

// Global state
static struct platform_device *cma_pdev;
static struct cma_allocation allocations[MAX_CMA_ALLOCATIONS];
static int num_allocations = 0;
static int current_alloc_id = 1;
static DEFINE_MUTEX(cma_mutex);

// Character device
static struct cdev cma_cdev;
static dev_t cma_devno;
static struct class *cma_class;
static struct device *cma_device;

// sysfs kobject
static struct kobject *cma_kobj;

// Helper functions

// Parse size string with K/M/G suffixes
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

// Find allocation by ID
static struct cma_allocation *find_allocation(int alloc_id)
{
    int i;
    
    for (i = 0; i < MAX_CMA_ALLOCATIONS; i++) {
        if (allocations[i].allocated && allocations[i].alloc_id == alloc_id) {
            return &allocations[i];
        }
    }
    return NULL;
}

// Set cache attributes for CMA allocation
static int set_cma_cache_state(struct cma_allocation *alloc, int cached)
{
    int ret;
    unsigned long virt = (unsigned long)alloc->virt_addr;
    int num_pages = alloc->num_pages;
    
    if (cached) {
        // Set as cached (write-back)
        ret = set_memory_wb(virt, num_pages);
        if (ret) {
            printk(KERN_ERR "Failed to set memory as cached: %d\n", ret);
            return ret;
        }
    } else {
        // Set as uncached
        ret = set_memory_uc(virt, num_pages);
        if (ret) {
            printk(KERN_ERR "Failed to set memory as uncached: %d\n", ret);
            return ret;
        }
    }
    
    alloc->is_cached = cached;
    printk(KERN_INFO "Set allocation %d (%zu bytes) as %s\n", 
           alloc->alloc_id, alloc->size, cached ? "cached" : "uncached");
    
    return 0;
}

// DMA allocation function
static int allocate_cma_memory(size_t size)
{
    struct cma_allocation *alloc;
    int i;
    int num_pages;
    
    // Find free slot
    for (i = 0; i < MAX_CMA_ALLOCATIONS; i++) {
        if (!allocations[i].allocated) {
            break;
        }
    }
    
    if (i >= MAX_CMA_ALLOCATIONS) {
        printk(KERN_ERR "No free allocation slots\n");
        return -ENOMEM;
    }
    
    alloc = &allocations[i];
    
    // Validate and align size
    if (size < MIN_ALLOCATION_SIZE) {
        printk(KERN_ERR "Size too small, minimum is %u bytes\n", MIN_ALLOCATION_SIZE);
        return -EINVAL;
    }
    
    if (size > MAX_ALLOCATION_SIZE) {
        printk(KERN_ERR "Size too large, maximum is %u bytes\n", MAX_ALLOCATION_SIZE);
        return -EINVAL;
    }
    
    // Round up to page boundary
    size = PAGE_ALIGN(size);
    num_pages = size / PAGE_SIZE;
    
    // Allocate using DMA coherent allocation (uses CMA backend when available)
    alloc->virt_addr = dma_alloc_coherent(&cma_pdev->dev, size, 
                                         &alloc->dma_handle, GFP_KERNEL);
    
    if (!alloc->virt_addr) {
        printk(KERN_ERR "Failed to allocate %zu bytes from DMA coherent memory\n", size);
        return -ENOMEM;
    }
    
    // Initialize allocation metadata
    alloc->size = size;
    alloc->num_pages = num_pages;
    alloc->is_cached = 1;  // DMA coherent memory is initially cached
    alloc->allocated = 1;
    alloc->alloc_id = current_alloc_id++;
    
    num_allocations++;
    
    printk(KERN_INFO "Allocated DMA coherent memory: ID=%d, size=%zu bytes (%d pages), "
           "virt=%p, phys=%pad\n", 
           alloc->alloc_id, alloc->size, alloc->num_pages, 
           alloc->virt_addr, &alloc->dma_handle);
    
    return alloc->alloc_id;
}

// Free DMA allocation
static int free_cma_memory(int alloc_id)
{
    struct cma_allocation *alloc;
    
    alloc = find_allocation(alloc_id);
    if (!alloc) {
        printk(KERN_ERR "Allocation %d not found\n", alloc_id);
        return -ENOENT;
    }
    
    // Restore memory to cached state before freeing
    if (!alloc->is_cached) {
        set_memory_wb((unsigned long)alloc->virt_addr, alloc->num_pages);
    }
    
    // Free the DMA coherent memory
    dma_free_coherent(&cma_pdev->dev, alloc->size, alloc->virt_addr, alloc->dma_handle);
    
    printk(KERN_INFO "Freed DMA allocation %d (%zu bytes)\n", alloc->alloc_id, alloc->size);
    
    // Clear allocation metadata
    memset(alloc, 0, sizeof(*alloc));
    num_allocations--;
    
    return 0;
}

// sysfs interface

static ssize_t command_show(struct kobject *kobj, struct kobj_attribute *attr, char *buf)
{
    return sprintf(buf, 
                   "DMA Cache Control Commands:\n"
                   "alloc <size>             - Allocate DMA coherent memory (size: bytes, K, M, G suffix)\n"
                   "free <id>                - Free DMA allocation by ID\n"
                   "cache <id>               - Set allocation as cached\n"
                   "uncache <id>             - Set allocation as uncached\n"
                   "toggle <id>              - Toggle cache state\n"
                   "\n"
                   "Examples:\n"
                   "echo 'alloc 4M' > command       # Allocate 4MB\n"
                   "echo 'uncache 1' > command      # Set allocation 1 as uncached\n"
                   "echo 'free 1' > command         # Free allocation 1\n");
}

static ssize_t command_store(struct kobject *kobj, struct kobj_attribute *attr,
                            const char *buf, size_t count)
{
    char cmd[64];
    char *cmd_name, *token, *cmd_ptr;
    int ret = 0;
    size_t size;
    int alloc_id;
    struct cma_allocation *alloc;
    
    if (count >= sizeof(cmd)) {
        return -EINVAL;
    }
    
    strncpy(cmd, buf, count);
    cmd[count] = '\0';
    
    // Remove trailing newline
    if (count > 0 && cmd[count-1] == '\n') {
        cmd[count-1] = '\0';
    }
    
    mutex_lock(&cma_mutex);
    
    // Parse command using strsep
    cmd_ptr = cmd;
    cmd_name = strsep(&cmd_ptr, " ");
    if (!cmd_name) {
        ret = -EINVAL;
        goto out;
    }
    
    if (strcmp(cmd_name, "alloc") == 0) {
        token = strsep(&cmd_ptr, " ");
        if (!token) {
            printk(KERN_ERR "alloc command requires size parameter\n");
            ret = -EINVAL;
            goto out;
        }
        
        if (parse_size_string(token, &size) != 0) {
            printk(KERN_ERR "Invalid size format. Use bytes, or append K/M/G (e.g., 4M, 16M)\n");
            ret = -EINVAL;
            goto out;
        }
        
        ret = allocate_cma_memory(size);
        if (ret < 0) {
            goto out;
        }
        
        printk(KERN_INFO "DMA allocation successful: ID=%d\n", ret);
        
    } else if (strcmp(cmd_name, "free") == 0) {
        token = strsep(&cmd_ptr, " ");
        if (!token || kstrtoint(token, 10, &alloc_id) != 0) {
            ret = -EINVAL;
            goto out;
        }
        
        ret = free_cma_memory(alloc_id);
        
    } else if (strcmp(cmd_name, "cache") == 0) {
        token = strsep(&cmd_ptr, " ");
        if (!token || kstrtoint(token, 10, &alloc_id) != 0) {
            ret = -EINVAL;
            goto out;
        }
        
        alloc = find_allocation(alloc_id);
        if (!alloc) {
            ret = -ENOENT;
            goto out;
        }
        
        ret = set_cma_cache_state(alloc, 1);
        
    } else if (strcmp(cmd_name, "uncache") == 0) {
        token = strsep(&cmd_ptr, " ");
        if (!token || kstrtoint(token, 10, &alloc_id) != 0) {
            ret = -EINVAL;
            goto out;
        }
        
        alloc = find_allocation(alloc_id);
        if (!alloc) {
            ret = -ENOENT;
            goto out;
        }
        
        ret = set_cma_cache_state(alloc, 0);
        
    } else if (strcmp(cmd_name, "toggle") == 0) {
        token = strsep(&cmd_ptr, " ");
        if (!token || kstrtoint(token, 10, &alloc_id) != 0) {
            ret = -EINVAL;
            goto out;
        }
        
        alloc = find_allocation(alloc_id);
        if (!alloc) {
            ret = -ENOENT;
            goto out;
        }
        
        ret = set_cma_cache_state(alloc, !alloc->is_cached);
        
    } else {
        printk(KERN_ERR "Unknown command: %s\n", cmd_name);
        ret = -EINVAL;
    }
    
out:
    mutex_unlock(&cma_mutex);
    
    if (ret < 0) {
        return ret;
    }
    
    return count;
}

static ssize_t status_show(struct kobject *kobj, struct kobj_attribute *attr, char *buf)
{
    int i, len = 0;
    int cached_count = 0, uncached_count = 0;
    size_t total_allocated = 0;
    
    mutex_lock(&cma_mutex);
    
    len += sprintf(buf + len, "DMA Cache Control Status\n");
    len += sprintf(buf + len, "========================\n");
    len += sprintf(buf + len, "Total allocations: %d/%d\n", num_allocations, MAX_CMA_ALLOCATIONS);
    
    // Calculate totals
    for (i = 0; i < MAX_CMA_ALLOCATIONS; i++) {
        if (allocations[i].allocated) {
            total_allocated += allocations[i].size;
            if (allocations[i].is_cached) {
                cached_count++;
            } else {
                uncached_count++;
            }
        }
    }
    
    len += sprintf(buf + len, "Total allocated memory: %zu bytes (%zu MB)\n", 
                   total_allocated, total_allocated / (1024 * 1024));
    len += sprintf(buf + len, "Cached allocations: %d\n", cached_count);
    len += sprintf(buf + len, "Uncached allocations: %d\n", uncached_count);
    
    if (num_allocations > 0) {
        len += sprintf(buf + len, "\nActive Allocations:\n");
        len += sprintf(buf + len, "ID   Size       Pages  Virtual     Physical    Cache State\n");
        len += sprintf(buf + len, "---  ---------  -----  ----------  ----------  -----------\n");
        
        for (i = 0; i < MAX_CMA_ALLOCATIONS && len < PAGE_SIZE - 200; i++) {
            if (allocations[i].allocated) {
                len += sprintf(buf + len, "%3d  %7zuK  %5d  %p  %pad  %s\n",
                              allocations[i].alloc_id,
                              allocations[i].size / 1024,
                              allocations[i].num_pages,
                              allocations[i].virt_addr,
                              &allocations[i].dma_handle,
                              allocations[i].is_cached ? "CACHED" : "UNCACHED");
            }
        }
    }
    
    len += sprintf(buf + len, "\nSize limits: %uK - %uK\n", 
                   MIN_ALLOCATION_SIZE / 1024, MAX_ALLOCATION_SIZE / 1024);
    
    mutex_unlock(&cma_mutex);
    
    return len;
}

static struct kobj_attribute command_attribute = __ATTR(command, 0664, command_show, command_store);
static struct kobj_attribute status_attribute = __ATTR(status, 0444, status_show, NULL);

static struct attribute *attrs[] = {
    &command_attribute.attr,
    &status_attribute.attr,
    NULL,
};

static struct attribute_group attr_group = {
    .attrs = attrs,
};

// Character device operations for mmap

static int cma_device_open(struct inode *inode, struct file *file)
{
    return 0;
}

static int cma_device_release(struct inode *inode, struct file *file)
{
    return 0;
}

static int cma_device_mmap(struct file *file, struct vm_area_struct *vma)
{
    struct cma_allocation *alloc;
    unsigned long offset, size;
    int alloc_id;
    int ret;
    
    // Extract allocation ID from offset (passed as alloc_id * PAGE_SIZE)
    offset = vma->vm_pgoff << PAGE_SHIFT;
    alloc_id = offset / PAGE_SIZE;
    size = vma->vm_end - vma->vm_start;
    
    mutex_lock(&cma_mutex);
    
    alloc = find_allocation(alloc_id);
    if (!alloc) {
        printk(KERN_ERR "mmap: allocation %d not found\n", alloc_id);
        mutex_unlock(&cma_mutex);
        return -ENOENT;
    }
    
    if (size > alloc->size) {
        printk(KERN_ERR "mmap: requested size %lu exceeds allocation size %zu\n", 
               size, alloc->size);
        mutex_unlock(&cma_mutex);
        return -EINVAL;
    }
    
    // Map the physical pages with appropriate cache attributes
    // For DMA coherent memory, respect the current cache state
    if (!alloc->is_cached) {
        vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot);
    }
    
    ret = remap_pfn_range(vma, vma->vm_start, 
                         alloc->dma_handle >> PAGE_SHIFT,
                         size, vma->vm_page_prot);
    
    if (ret) {
        printk(KERN_ERR "mmap: remap_pfn_range failed: %d\n", ret);
        mutex_unlock(&cma_mutex);
        return ret;
    }
    
    printk(KERN_INFO "mmap: mapped allocation %d (%zu bytes) to user space\n", 
           alloc_id, size);
    
    mutex_unlock(&cma_mutex);
    return 0;
}

static const struct file_operations cma_fops = {
    .owner = THIS_MODULE,
    .open = cma_device_open,
    .release = cma_device_release,
    .mmap = cma_device_mmap,
};

// Module initialization and cleanup

static int __init cma_cache_init(void)
{
    int ret;
    
    printk(KERN_INFO "DMA Cache Control Module loading...\n");
    
    // Initialize allocation tracking
    memset(allocations, 0, sizeof(allocations));
    
    // Create platform device for CMA/DMA operations
    cma_pdev = platform_device_alloc("cma_cache", -1);
    if (!cma_pdev) {
        printk(KERN_ERR "Failed to allocate platform device\n");
        return -ENOMEM;
    }
    
    ret = platform_device_add(cma_pdev);
    if (ret) {
        printk(KERN_ERR "Failed to add platform device: %d\n", ret);
        platform_device_put(cma_pdev);
        return ret;
    }
    
    // Set up DMA mask for the platform device
    ret = dma_set_mask_and_coherent(&cma_pdev->dev, DMA_BIT_MASK(32));
    if (ret) {
        printk(KERN_ERR "Failed to set DMA mask: %d\n", ret);
        goto err_platform;
    }
    
    // Create character device
    ret = alloc_chrdev_region(&cma_devno, 0, 1, MODULE_NAME);
    if (ret) {
        printk(KERN_ERR "Failed to allocate char device region: %d\n", ret);
        goto err_platform;
    }
    
    cdev_init(&cma_cdev, &cma_fops);
    cma_cdev.owner = THIS_MODULE;
    
    ret = cdev_add(&cma_cdev, cma_devno, 1);
    if (ret) {
        printk(KERN_ERR "Failed to add char device: %d\n", ret);
        goto err_chrdev;
    }
    
    // Create device class
    cma_class = class_create(THIS_MODULE, MODULE_NAME);
    if (IS_ERR(cma_class)) {
        ret = PTR_ERR(cma_class);
        printk(KERN_ERR "Failed to create device class: %d\n", ret);
        goto err_cdev;
    }
    
    // Create device node
    cma_device = device_create(cma_class, NULL, cma_devno, NULL, MODULE_NAME);
    if (IS_ERR(cma_device)) {
        ret = PTR_ERR(cma_device);
        printk(KERN_ERR "Failed to create device: %d\n", ret);
        goto err_class;
    }
    
    // Create sysfs interface
    cma_kobj = kobject_create_and_add("cma_cache", kernel_kobj);
    if (!cma_kobj) {
        printk(KERN_ERR "Failed to create kobject\n");
        ret = -ENOMEM;
        goto err_device;
    }
    
    ret = sysfs_create_group(cma_kobj, &attr_group);
    if (ret) {
        printk(KERN_ERR "Failed to create sysfs group: %d\n", ret);
        goto err_kobject;
    }
    
    printk(KERN_INFO "DMA Cache Control Module loaded successfully\n");
    printk(KERN_INFO "Device: /dev/%s\n", MODULE_NAME);
    printk(KERN_INFO "sysfs: /sys/kernel/cma_cache/\n");
    
    return 0;
    
err_kobject:
    kobject_put(cma_kobj);
err_device:
    device_destroy(cma_class, cma_devno);
err_class:
    class_destroy(cma_class);
err_cdev:
    cdev_del(&cma_cdev);
err_chrdev:
    unregister_chrdev_region(cma_devno, 1);
err_platform:
    platform_device_unregister(cma_pdev);
    return ret;
}

static void __exit cma_cache_exit(void)
{
    int i;
    
    printk(KERN_INFO "DMA Cache Control Module unloading...\n");
    
    // Free all remaining allocations
    mutex_lock(&cma_mutex);
    for (i = 0; i < MAX_CMA_ALLOCATIONS; i++) {
        if (allocations[i].allocated) {
            printk(KERN_INFO "Cleaning up allocation %d\n", allocations[i].alloc_id);
            
            // Restore memory to cached state before freeing
            if (!allocations[i].is_cached) {
                set_memory_wb((unsigned long)allocations[i].virt_addr, 
                             allocations[i].num_pages);
            }
            
            dma_free_coherent(&cma_pdev->dev, allocations[i].size, 
                             allocations[i].virt_addr, allocations[i].dma_handle);
        }
    }
    mutex_unlock(&cma_mutex);
    
    // Remove sysfs interface
    sysfs_remove_group(cma_kobj, &attr_group);
    kobject_put(cma_kobj);
    
    // Remove device
    device_destroy(cma_class, cma_devno);
    class_destroy(cma_class);
    cdev_del(&cma_cdev);
    unregister_chrdev_region(cma_devno, 1);
    
    // Remove platform device
    platform_device_unregister(cma_pdev);
    
    printk(KERN_INFO "DMA Cache Control Module unloaded\n");
}

module_init(cma_cache_init);
module_exit(cma_cache_exit);

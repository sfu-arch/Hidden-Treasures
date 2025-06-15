#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/uaccess.h>
#include <linux/slab.h>
#include <linux/mm.h>
#include <linux/vmalloc.h>
#include <linux/gfp.h>
#include <linux/io.h>
#include <asm/io.h>
#include <linux/timekeeping.h>
#include <linux/kobject.h>
#include <linux/sysfs.h>
#include <linux/ctype.h>

// Include architecture-specific headers only if available
#ifdef CONFIG_X86
#include <asm/cacheflush.h>
#include <asm/set_memory.h>
#endif

// Compatibility for older kernels or systems without set_memory functions
//#ifndef set_memory_uc
//#define set_memory_uc(addr, numpages) (-ENOTSUPP)
//#endif

//#ifndef set_memory_wb
//#define set_memory_wb(addr, numpages) (0)
//#endif

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Your Name");
MODULE_DESCRIPTION("Kernel module for uncached memory allocation with sysfs interface and variable size support");
MODULE_VERSION("2.0");

#define BUFFER_SIZE 4096
#define MAX_ALLOCATION_SIZE (128UL * 1024 * 1024) // 128MB limit
#define LARGE_ALLOC_THRESHOLD (1024 * 1024)       // 1MB threshold for vmalloc

static struct kobject *uncached_kobj;
static void *uncached_buffer = NULL;
static void *cached_buffer = NULL;
static int uncached_allocated = 0;
static int cached_allocated = 0;
static size_t uncached_size = 0;
static size_t cached_size = 0;
static int uncached_is_vmalloc = 0; // Track allocation method
static int cached_is_vmalloc = 0;

// Helper function to parse size strings with K/M/G suffixes
static int parse_size_string(const char *str, size_t *size)
{
    char *endptr;
    unsigned long val;
    
    val = simple_strtoul(str, &endptr, 10);
    
    if (endptr == str) {
        return -EINVAL; // No digits found
    }
    
    // Handle suffixes
    if (*endptr != '\0') {
        switch (tolower(*endptr)) {
        case 'k':
            val *= 1024;
            break;
        case 'm':
            val *= 1024 * 1024;
            break;
        case 'g':
            val *= 1024UL * 1024 * 1024;
            break;
        default:
            return -EINVAL; // Invalid suffix
        }
        endptr++;
        
        // Make sure nothing follows the suffix
        if (*endptr != '\0') {
            return -EINVAL;
        }
    }
    
    *size = val;
    return 0;
}

static void *allocate_uncached_memory(size_t size)
{
    void *buffer;
    
    if (size > MAX_ALLOCATION_SIZE) {
        printk(KERN_ERR "Allocation size too large: %zu (max %lu)\n", size, MAX_ALLOCATION_SIZE);
        return NULL;
    }
    
    // For large allocations, use vmalloc instead of __get_free_pages
    if (size >= LARGE_ALLOC_THRESHOLD) {
        printk(KERN_INFO "Using vmalloc for large uncached allocation (%zu bytes)\n", size);
        
        buffer = vmalloc(size);
        if (!buffer) {
            printk(KERN_ERR "vmalloc failed for size %zu\n", size);
            return NULL;
        }
        
        uncached_is_vmalloc = 1;
        
        // Try to set pages as uncached
        #ifdef CONFIG_X86
        {
            unsigned long addr = (unsigned long)buffer;
            unsigned long end = addr + size;
            unsigned long page_addr;
            int total_pages = 0;
            int failed_pages = 0;
            
            // Set each page as uncached
            for (page_addr = addr; page_addr < end; page_addr += PAGE_SIZE) {
                struct page *page = vmalloc_to_page((void *)page_addr);
                if (page) {
                    unsigned long pfn = page_to_pfn(page);
                    int result = set_memory_uc(pfn << PAGE_SHIFT, 1);
                    if (result != 0 && result != -ENOTSUPP) {
                        failed_pages++;
                    }
                    total_pages++;
                }
            }
            
            if (failed_pages > 0) {
                printk(KERN_WARNING "Failed to set %d/%d pages as uncached\n", 
                       failed_pages, total_pages);
            } else {
                printk(KERN_INFO "Successfully set %d pages as uncached\n", total_pages);
            }
        }
        #endif
        
    } else {
        // Use __get_free_pages for smaller allocations
        buffer = (void *)__get_free_pages(GFP_KERNEL, get_order(size));
        if (!buffer) {
            printk(KERN_ERR "Failed to allocate pages for uncached memory\n");
            return NULL;
        }
        
        uncached_is_vmalloc = 0;
        
        #ifdef CONFIG_X86
        {
            int result = set_memory_uc((unsigned long)buffer, (size + PAGE_SIZE - 1) >> PAGE_SHIFT);
            if (result == -ENOTSUPP) {
                printk(KERN_INFO "set_memory_uc not supported on this kernel, using normal allocation\n");
            } else if (result != 0) {
                printk(KERN_WARNING "Could not set memory as uncached (error %d), using normal allocation\n", result);
            } else {
                printk(KERN_INFO "Successfully set memory as uncached\n");
            }
        }
        #else
        printk(KERN_INFO "Uncached memory not supported on this architecture, using normal memory\n");
        #endif
    }
    
    printk(KERN_INFO "Allocated uncached memory at %p (size %zu, method: %s)\n", 
           buffer, size, uncached_is_vmalloc ? "vmalloc" : "__get_free_pages");
    
    return buffer;
}

// Function to allocate cached memory with size parameter
static void *allocate_cached_memory(size_t size)
{
    void *buffer;
    
    if (size > MAX_ALLOCATION_SIZE) {
        printk(KERN_ERR "Allocation size too large: %zu (max %lu)\n", size, MAX_ALLOCATION_SIZE);
        return NULL;
    }
    
    // For large allocations, use vmalloc
    if (size > PAGE_SIZE) {
        buffer = vmalloc(size);
        cached_is_vmalloc = 1;
        printk(KERN_INFO "Allocated cached memory using vmalloc\n");
    } else {
        // Allocate regular kernel memory (cached)
        buffer = kmalloc(size, GFP_KERNEL);
        cached_is_vmalloc = 0;
        printk(KERN_INFO "Allocated cached memory using kmalloc\n");
    }
    
    if (!buffer) {
        printk(KERN_ERR "Failed to allocate cached memory (size %zu)\n", size);
        return NULL;
    }
    
    printk(KERN_INFO "Allocated cached memory at %p (size %zu, method: %s)\n", 
           buffer, size, cached_is_vmalloc ? "vmalloc" : "kmalloc");
    
    return buffer;
}

// Function to free uncached memory
static void free_uncached_memory(void *buffer, size_t size)
{
    if (!buffer) return;
    
    if (uncached_is_vmalloc) {
        #ifdef CONFIG_X86
        // Restore write-back caching for vmalloc pages
        unsigned long addr = (unsigned long)buffer;
        unsigned long end = addr + size;
        unsigned long page_addr;
        
        for (page_addr = addr; page_addr < end; page_addr += PAGE_SIZE) {
            struct page *page = vmalloc_to_page((void *)page_addr);
            if (page) {
                unsigned long pfn = page_to_pfn(page);
                set_memory_wb(pfn << PAGE_SHIFT, 1);
            }
        }
        #endif
        
        vfree(buffer);
        printk(KERN_INFO "Freed vmalloc uncached memory\n");
        
    } else {
        #ifdef CONFIG_X86
        // Restore normal caching before freeing (if supported)
        int result = set_memory_wb((unsigned long)buffer, (size + PAGE_SIZE - 1) >> PAGE_SHIFT);
        if (result != 0 && result != -ENOTSUPP) {
            printk(KERN_WARNING "Failed to restore write-back caching (error %d)\n", result);
        }
        #endif
        
        // Free the pages
        free_pages((unsigned long)buffer, get_order(size));
        printk(KERN_INFO "Freed pages uncached memory\n");
    }
    
    uncached_is_vmalloc = 0;
}

// Function to free cached memory
static void free_cached_memory(void *buffer, size_t size)
{
    if (!buffer) return;
    
    if (cached_is_vmalloc) {
        vfree(buffer);
        printk(KERN_INFO "Freed vmalloc cached memory\n");
    } else {
        kfree(buffer);
        printk(KERN_INFO "Freed kmalloc cached memory\n");
    }
    
    cached_is_vmalloc = 0;
}

// Sysfs command interface 
static ssize_t command_show(struct kobject *kobj, struct kobj_attribute *attr, char *buf)
{
    return sprintf(buf, "Commands: 0 [size] = alloc_uc, 1 [size] = alloc_cached, 2 = free\n"
                        "Size format: bytes, or with suffix K/M/G (e.g., 4K, 1M, 512M)\n"
                        "Examples: echo '0 1M' > command, echo '1 4K' > command, echo '2' > command\n"
                        "Note: Allocations >128MB not supported. Large allocations may fail.\n");
}

static ssize_t command_store(struct kobject *kobj, struct kobj_attribute *attr,
                            const char *buf, size_t count)
{
    char command[64];
    int cmd_int;
    size_t size = BUFFER_SIZE; // default size
    char *size_str;
    
    if (count >= sizeof(command)) {
        return -EINVAL;
    }
    
    strncpy(command, buf, count);
    command[count] = '\0';
    
    // Remove trailing newline if present
    if (count > 0 && command[count-1] == '\n') {
        command[count-1] = '\0';
    }
    
    // Check if command contains size parameter (format: "cmd size")
    size_str = strchr(command, ' ');
    
    if (size_str) {
        *size_str = '\0'; // null terminate command part
        size_str++; // move to size part
        
        // Parse size with optional suffixes (K, M, G)
        if (parse_size_string(size_str, &size) != 0) {
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
        
        // Round up to page boundary
        size = PAGE_ALIGN(size);
    }
    
    // Convert command to integer
    if (kstrtoint(command, 10, &cmd_int) != 0) {
        printk(KERN_WARNING "Invalid command format. Use: command [size]\n");
        printk(KERN_INFO "Examples: '0 4096', '1 1M', '2'\n");
        return -EINVAL;
    }
    
    switch (cmd_int) {
    case 0: // alloc_uc (allocate uncached memory)
        if (uncached_allocated) {
            printk(KERN_WARNING "Uncached buffer already allocated\n");
            return count;
        }
        
        uncached_buffer = allocate_uncached_memory(size);
        if (!uncached_buffer) {
            return -ENOMEM;
        }
        memset(uncached_buffer, 0xAA, size);
        uncached_allocated = 1;
        uncached_size = size;
        printk(KERN_INFO "Allocated uncached memory buffer (%zu bytes)\n", size);
        break;
        
    case 1: // alloc_cached (allocate cached memory)
        if (cached_allocated) {
            printk(KERN_WARNING "Cached buffer already allocated\n");
            return count;
        }
        
        cached_buffer = allocate_cached_memory(size);
        if (!cached_buffer) {
            return -ENOMEM;
        }
        memset(cached_buffer, 0xAA, size);
        cached_allocated = 1;
        cached_size = size;
        printk(KERN_INFO "Allocated cached memory buffer (%zu bytes)\n", size);
        break;
        
    case 2: // free (free all allocated memory)
        if (cached_buffer) {
            free_cached_memory(cached_buffer, cached_size);
            cached_buffer = NULL;
            cached_allocated = 0;
            cached_size = 0;
        }
        
        if (uncached_buffer) {
            free_uncached_memory(uncached_buffer, uncached_size);
            uncached_buffer = NULL;
            uncached_allocated = 0;
            uncached_size = 0;
        }
        
        printk(KERN_INFO "Freed all buffers\n");
        break;
        
    default:
        printk(KERN_WARNING "Unknown command: %d\n", cmd_int);
        printk(KERN_INFO "Valid commands: 0=alloc_uc, 1=alloc_cached, 2=free\n");
        return -EINVAL;
    }
    
    return count;
}

// Sysfs status interface  
static ssize_t status_show(struct kobject *kobj, struct kobj_attribute *attr, char *buf)
{
    return sprintf(buf, "Uncached Memory Module Status\n"
                        "============================\n"
                        "Uncached: %s at %p (%zu bytes)\n"
                        "Cached: %s at %p (%zu bytes)\n"
                        "Allocation methods: uncached=%s, cached=%s\n",
                   uncached_allocated ? "ALLOCATED" : "not allocated", uncached_buffer, uncached_size,
                   cached_allocated ? "ALLOCATED" : "not allocated", cached_buffer, cached_size,
                   uncached_is_vmalloc ? "vmalloc" : "__get_free_pages",
                   cached_is_vmalloc ? "vmalloc" : "kmalloc");
}

// Buffer addresses (read-only)
static ssize_t uncached_addr_show(struct kobject *kobj, struct kobj_attribute *attr, char *buf)
{
    if (uncached_allocated)
        return sprintf(buf, "%p\n", uncached_buffer);
    else
        return sprintf(buf, "not allocated\n");
}

static ssize_t cached_addr_show(struct kobject *kobj, struct kobj_attribute *attr, char *buf)
{
    if (cached_allocated)
        return sprintf(buf, "%p\n", cached_buffer);
    else
        return sprintf(buf, "not allocated\n");
}

// Size information
static ssize_t size_info_show(struct kobject *kobj, struct kobj_attribute *attr, char *buf)
{
    return sprintf(buf, "Uncached size: %zu bytes\n"
                        "Cached size: %zu bytes\n"
                        "Maximum allocation: %lu bytes (%lu MB)\n"
                        "Large allocation threshold: %d bytes (%d MB)\n",
                   uncached_size, cached_size, 
                   MAX_ALLOCATION_SIZE, MAX_ALLOCATION_SIZE / (1024*1024),
                   LARGE_ALLOC_THRESHOLD, LARGE_ALLOC_THRESHOLD / (1024*1024));
}

// Attribute definitions
static struct kobj_attribute command_attr = __ATTR(command, 0664, command_show, command_store);
static struct kobj_attribute status_attr = __ATTR(status, 0444, status_show, NULL);
static struct kobj_attribute uncached_addr_attr = __ATTR(uncached_addr, 0444, uncached_addr_show, NULL);
static struct kobj_attribute cached_addr_attr = __ATTR(cached_addr, 0444, cached_addr_show, NULL);
static struct kobj_attribute size_info_attr = __ATTR(size_info, 0444, size_info_show, NULL);

static struct attribute *attrs[] = {
    &command_attr.attr,
    &status_attr.attr,
    &uncached_addr_attr.attr,
    &cached_addr_attr.attr,
    &size_info_attr.attr,
    NULL,
};

static struct attribute_group attr_group = {
    .attrs = attrs,
};

// mmap operation to map kernel memory to user space
static int proc_mmap(struct file *file, struct vm_area_struct *vma)
{
    unsigned long size = vma->vm_end - vma->vm_start;
    void *buffer_to_map = NULL;
    size_t buffer_size = 0;
    unsigned long phys_addr;
    unsigned long pfn;
    int is_uncached = 0;
    int is_vmalloc_buffer = 0;
    
    printk(KERN_INFO "mmap called, size: %lu bytes, offset: 0x%lx\n", size, vma->vm_pgoff);
    
    // Use offset to determine which buffer to map
    // offset 0 = uncached buffer, offset 1 = cached buffer
    if (vma->vm_pgoff == 0) {
        // Map uncached buffer
        if (!uncached_allocated || !uncached_buffer) {
            printk(KERN_ERR "Uncached buffer not allocated for mmap\n");
            return -ENOMEM;
        }
        buffer_to_map = uncached_buffer;
        buffer_size = uncached_size;
        is_uncached = 1;
        is_vmalloc_buffer = uncached_is_vmalloc;
    } else if (vma->vm_pgoff == 1) {
        // Map cached buffer  
        if (!cached_allocated || !cached_buffer) {
            printk(KERN_ERR "Cached buffer not allocated for mmap\n");
            return -ENOMEM;
        }
        buffer_to_map = cached_buffer;
        buffer_size = cached_size;
        is_uncached = 0;
        is_vmalloc_buffer = cached_is_vmalloc;
    } else {
        printk(KERN_ERR "Invalid mmap offset. Use 0 for uncached, 1 for cached\n");
        return -EINVAL;
    }
    
    if (size > buffer_size) {
        printk(KERN_ERR "mmap size too large (%lu > %zu)\n", size, buffer_size);
        return -EINVAL;
    }
    
    // For vmalloc memory, we need to handle mapping differently
    if (is_vmalloc_buffer) {
        unsigned long addr = (unsigned long)buffer_to_map;
        unsigned long user_addr = vma->vm_start;
        unsigned long offset;
        int ret;
        
        printk(KERN_INFO "Mapping vmalloc %s buffer (%zu bytes) using page-by-page mapping\n", 
               is_uncached ? "uncached" : "cached", buffer_size);
        
        // Set up the VMA for vmalloc
        vma->vm_flags |= VM_IO | VM_DONTEXPAND | VM_DONTDUMP;
        
        // For uncached vmalloc memory, disable caching in user space mapping
        if (is_uncached) {
            vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot);
            printk(KERN_INFO "Mapping vmalloc as uncached memory\n");
        } else {
            printk(KERN_INFO "Mapping vmalloc as cached memory\n");
        }
        
        // Map page by page for vmalloc memory
        for (offset = 0; offset < size; offset += PAGE_SIZE) {
            struct page *page = vmalloc_to_page((void *)(addr + offset));
            if (!page) {
                printk(KERN_ERR "vmalloc_to_page failed at offset %lu\n", offset);
                return -EFAULT;
            }
            
            ret = vm_insert_page(vma, user_addr + offset, page);
            if (ret) {
                printk(KERN_ERR "vm_insert_page failed at offset %lu: %d\n", offset, ret);
                return ret;
            }
        }
        
        // For uncached vmalloc memory, disable caching in user space mapping too
        if (is_uncached) {
            vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot);
            printk(KERN_INFO "Mapping vmalloc memory as uncached\n");
        } else {
            printk(KERN_INFO "Mapping vmalloc memory as cached\n");
        }
        
    } else {
        // For __get_free_pages or kmalloc memory
        phys_addr = virt_to_phys(buffer_to_map);
        pfn = phys_addr >> PAGE_SHIFT;
        
        printk(KERN_INFO "Mapping %s buffer at virtual %p, physical 0x%lx, pfn 0x%lx\n", 
               is_uncached ? "uncached" : "cached", buffer_to_map, phys_addr, pfn);
        
        // Set up the VMA
        vma->vm_flags |= VM_IO | VM_DONTEXPAND | VM_DONTDUMP;
        
        // Reset offset to 0 for actual mapping
        vma->vm_pgoff = 0;
        
        // For uncached memory, disable caching in user space mapping too
        if (is_uncached) {
            vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot);
            printk(KERN_INFO "Mapping as uncached memory\n");
        } else {
            printk(KERN_INFO "Mapping as cached memory\n");
        }
        
        // Map the pages
        if (remap_pfn_range(vma, vma->vm_start, pfn, size, vma->vm_page_prot)) {
            printk(KERN_ERR "remap_pfn_range failed\n");
            return -EAGAIN;
        }
    }
    
    printk(KERN_INFO "Successfully mapped %s memory to user space\n", 
           is_uncached ? "uncached" : "cached");
    
    return 0;
}

// Add character device support for mmap functionality
#include <linux/cdev.h>
#include <linux/device.h>

static dev_t dev_num;
static struct cdev uncached_cdev;
static struct class *uncached_class;
static struct device *uncached_device;

static int device_open(struct inode *inode, struct file *file)
{
    return 0;
}

static int device_release(struct inode *inode, struct file *file)
{
    return 0;
}

// Device file operations (mainly for mmap)
static const struct file_operations device_fops = {
    .open = device_open,
    .release = device_release,
    .mmap = proc_mmap,  // Reuse the mmap function
};

// Module initialization
static int __init uncached_mem_init(void)
{
    int ret;
    
    printk(KERN_INFO "Uncached memory module with sysfs interface loading\n");
    
    // Create sysfs interface
    uncached_kobj = kobject_create_and_add("uncached_mem", kernel_kobj);
    if (!uncached_kobj) {
        printk(KERN_ERR "Failed to create sysfs kobject\n");
        return -ENOMEM;
    }
    
    ret = sysfs_create_group(uncached_kobj, &attr_group);
    if (ret) {
        printk(KERN_ERR "Failed to create sysfs attribute group\n");
        kobject_put(uncached_kobj);
        return ret;
    }
    
    // Create character device for mmap functionality
    ret = alloc_chrdev_region(&dev_num, 0, 1, "uncached_mem");
    if (ret < 0) {
        printk(KERN_ERR "Failed to allocate character device region\n");
        goto cleanup_sysfs;
    }
    
    cdev_init(&uncached_cdev, &device_fops);
    ret = cdev_add(&uncached_cdev, dev_num, 1);
    if (ret < 0) {
        printk(KERN_ERR "Failed to add character device\n");
        goto cleanup_chrdev;
    }
    
    uncached_class = class_create(THIS_MODULE, "uncached_mem");
    if (IS_ERR(uncached_class)) {
        printk(KERN_ERR "Failed to create device class\n");
        ret = PTR_ERR(uncached_class);
        goto cleanup_cdev;
    }
    
    uncached_device = device_create(uncached_class, NULL, dev_num, NULL, "uncached_mem");
    if (IS_ERR(uncached_device)) {
        printk(KERN_ERR "Failed to create device\n");
        ret = PTR_ERR(uncached_device);
        goto cleanup_class;
    }
    
    printk(KERN_INFO "Uncached memory module loaded successfully\n");
    printk(KERN_INFO "Sysfs interface: /sys/kernel/uncached_mem/\n");
    printk(KERN_INFO "Device file: /dev/uncached_mem (for mmap)\n");
    printk(KERN_INFO "Control: echo 'command [size]' > /sys/kernel/uncached_mem/command\n");
    printk(KERN_INFO "Status: cat /sys/kernel/uncached_mem/status\n");
    
    return 0;
    
cleanup_class:
    class_destroy(uncached_class);
cleanup_cdev:
    cdev_del(&uncached_cdev);
cleanup_chrdev:
    unregister_chrdev_region(dev_num, 1);
cleanup_sysfs:
    sysfs_remove_group(uncached_kobj, &attr_group);
    kobject_put(uncached_kobj);
    return ret;
}

// Module cleanup
static void __exit uncached_mem_exit(void)
{
    // Free any allocated memory
    if (cached_buffer) {
        free_cached_memory(cached_buffer, cached_size);
        cached_buffer = NULL;
        cached_allocated = 0;
        cached_size = 0;
    }
    
    if (uncached_buffer) {
        free_uncached_memory(uncached_buffer, uncached_size);
        uncached_buffer = NULL;
        uncached_allocated = 0;
        uncached_size = 0;
    }
    
    // Remove device and sysfs
    if (uncached_device) {
        device_destroy(uncached_class, dev_num);
    }
    if (uncached_class) {
        class_destroy(uncached_class);
    }
    cdev_del(&uncached_cdev);
    unregister_chrdev_region(dev_num, 1);
    
    sysfs_remove_group(uncached_kobj, &attr_group);
    kobject_put(uncached_kobj);
    
    printk(KERN_INFO "Uncached memory module unloaded\n");
}

module_init(uncached_mem_init);
module_exit(uncached_mem_exit);

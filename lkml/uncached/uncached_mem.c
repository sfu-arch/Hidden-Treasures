#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/proc_fs.h>
#include <linux/uaccess.h>
#include <linux/slab.h>
#include <linux/mm.h>
#include <linux/vmalloc.h>
#include <linux/gfp.h>
#include <linux/io.h>
#include <asm/io.h>
#include <linux/timekeeping.h>

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
MODULE_DESCRIPTION("Kernel module for uncached memory allocation and mmap access");
MODULE_VERSION("1.0");

#define PROC_NAME "uncached_mem"
#define BUFFER_SIZE 4096

static struct proc_dir_entry *proc_entry;
static void *uncached_buffer = NULL;
static void *cached_buffer = NULL;
static int uncached_allocated = 0;
static int cached_allocated = 0;

// Function to allocate uncached memory
static void *allocate_uncached_memory(size_t size)
{
    void *buffer;
    
    // Use __get_free_pages for better control over caching
    buffer = (void *)__get_free_pages(GFP_KERNEL, get_order(size));
    if (!buffer) {
        printk(KERN_ERR "Failed to allocate pages for uncached memory\n");
        return NULL;
    }
    
    // Try to set memory as uncached if the function is available
    // This is architecture dependent and may not work on all systems
    #ifdef CONFIG_X86
    {
        int result = set_memory_uc((unsigned long)buffer, (size + PAGE_SIZE - 1) >> PAGE_SHIFT);
        if (result == -ENOTSUPP) {
            printk(KERN_INFO "set_memory_uc not supported on this kernel, using normal allocation\n");
        } else if (result != 0) {
            printk(KERN_WARNING "Could not set memory as uncached (error %d), using normal allocation\n", result);
            // Continue anyway - the memory will be cached but still functional
        } else {
            printk(KERN_INFO "Successfully set memory as uncached\n");
        }
    }
    #else
    printk(KERN_INFO "Uncached memory not supported on this architecture, using normal memory\n");
    #endif
    
    printk(KERN_INFO "Allocated uncached memory at %p (size %zu)\n", buffer, size);
    
    return buffer;
}

// Function to allocate cached memory
static void *allocate_cached_memory(size_t size)
{
    void *buffer;
    
    // Allocate regular kernel memory (cached)
    buffer = kmalloc(size, GFP_KERNEL);
    if (!buffer) {
        printk(KERN_ERR "Failed to allocate cached memory\n");
        return NULL;
    }
    
    printk(KERN_INFO "Allocated cached memory at %p\n", buffer);
    
    return buffer;
}

// Function to free uncached memory
static void free_uncached_memory(void *buffer, size_t size)
{
    if (buffer) {
        #ifdef CONFIG_X86
        // Restore normal caching before freeing (if supported)
        int result = set_memory_wb((unsigned long)buffer, (size + PAGE_SIZE - 1) >> PAGE_SHIFT);
        if (result != 0 && result != -ENOTSUPP) {
            printk(KERN_WARNING "Failed to restore write-back caching (error %d)\n", result);
        }
        #endif
        
        // Free the pages
        free_pages((unsigned long)buffer, get_order(size));
        
        printk(KERN_INFO "Freed uncached memory\n");
    }
}

// Function to free cached memory
static void free_cached_memory(void *buffer)
{
    if (buffer) {
        kfree(buffer);
        printk(KERN_INFO "Freed cached memory\n");
    }
}

// Proc file read operation
static ssize_t proc_read(struct file *file, char __user *buffer, size_t count, loff_t *pos)
{
    char output[512];
    int len;
    
    if (*pos > 0) {
        return 0; // EOF
    }
    
    len = snprintf(output, sizeof(output), 
                  "Uncached Memory Module\n"
                  "======================\n"
                  "Uncached buffer: %s at %p\n"
                  "Cached buffer: %s at %p\n"
                  "Buffer size: %d bytes\n"
                  "\nCommands:\n"
                  "  0 - alloc_uc (allocate uncached memory)\n"
                  "  1 - alloc_cached (allocate cached memory)\n"
                  "  2 - free (free all allocated memory)\n",
                  uncached_allocated ? "ALLOCATED" : "NOT ALLOCATED",
                  uncached_buffer,
                  cached_allocated ? "ALLOCATED" : "NOT ALLOCATED", 
                  cached_buffer,
                  BUFFER_SIZE);
    
    if (len > count) {
        len = count;
    }
    
    if (copy_to_user(buffer, output, len)) {
        return -EFAULT;
    }
    
    *pos += len;
    return len;
}

// Proc file write operation
static ssize_t proc_write(struct file *file, const char __user *buffer, size_t count, loff_t *pos)
{
    char command[32];
    int cmd_int;
    
    if (count >= sizeof(command)) {
        return -EINVAL;
    }
    
    if (copy_from_user(command, buffer, count)) {
        return -EFAULT;
    }
    
    command[count] = '\0';
    
    // Remove trailing newline if present
    if (count > 0 && command[count-1] == '\n') {
        command[count-1] = '\0';
    }
    
    // Convert command to integer
    if (kstrtoint(command, 10, &cmd_int) != 0) {
        printk(KERN_WARNING "Invalid command format. Use integers: 0, 1, or 2\n");
        printk(KERN_INFO "Commands: 0=alloc_uc, 1=alloc_cached, 2=free\n");
        return -EINVAL;
    }
    
    switch (cmd_int) {
    case 0: // alloc_uc (allocate uncached memory)
        if (uncached_allocated) {
            printk(KERN_WARNING "Uncached buffer already allocated\n");
            return count;
        }
        
        uncached_buffer = allocate_uncached_memory(BUFFER_SIZE);
        if (!uncached_buffer) {
            return -ENOMEM;
        }
        // Initialize buffer with test pattern
        memset(uncached_buffer, 0xAA, BUFFER_SIZE);
        uncached_allocated = 1;
        printk(KERN_INFO "Allocated uncached memory buffer\n");
        break;
        
    case 1: // alloc_cached (allocate cached memory)
        if (cached_allocated) {
            printk(KERN_WARNING "Cached buffer already allocated\n");
            return count;
        }
        
        cached_buffer = allocate_cached_memory(BUFFER_SIZE);
        if (!cached_buffer) {
            return -ENOMEM;
        }
        // Initialize buffer with test pattern
        memset(cached_buffer, 0xAA, BUFFER_SIZE);
        cached_allocated = 1;
        printk(KERN_INFO "Allocated cached memory buffer\n");
        break;
        
    case 2: // free (free all allocated memory)
        if (!uncached_allocated && !cached_allocated) {
            printk(KERN_WARNING "No buffers to free\n");
            return count;
        }
        
        if (cached_buffer) {
            free_cached_memory(cached_buffer);
            cached_buffer = NULL;
            cached_allocated = 0;
        }
        
        if (uncached_buffer) {
            free_uncached_memory(uncached_buffer, BUFFER_SIZE);
            uncached_buffer = NULL;
            uncached_allocated = 0;
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

// mmap operation to map kernel memory to user space
static int proc_mmap(struct file *file, struct vm_area_struct *vma)
{
    unsigned long size = vma->vm_end - vma->vm_start;
    void *buffer_to_map = NULL;
    unsigned long phys_addr;
    unsigned long pfn;
    int is_uncached = 0;
    
    printk(KERN_INFO "mmap called, size: %lu bytes, offset: 0x%lx\n", size, vma->vm_pgoff);
    
    if (size > BUFFER_SIZE) {
        printk(KERN_ERR "mmap size too large (%lu > %d)\n", size, BUFFER_SIZE);
        return -EINVAL;
    }
    
    // Use offset to determine which buffer to map
    // offset 0 = uncached buffer, offset 1 = cached buffer
    if (vma->vm_pgoff == 0) {
        // Map uncached buffer
        if (!uncached_allocated || !uncached_buffer) {
            printk(KERN_ERR "Uncached buffer not allocated for mmap\n");
            return -ENOMEM;
        }
        buffer_to_map = uncached_buffer;
        is_uncached = 1;
    } else if (vma->vm_pgoff == 1) {
        // Map cached buffer  
        if (!cached_allocated || !cached_buffer) {
            printk(KERN_ERR "Cached buffer not allocated for mmap\n");
            return -ENOMEM;
        }
        buffer_to_map = cached_buffer;
        is_uncached = 0;
    } else {
        printk(KERN_ERR "Invalid mmap offset. Use 0 for uncached, 1 for cached\n");
        return -EINVAL;
    }
    
    // Get physical address and page frame number
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
    
    printk(KERN_INFO "Successfully mapped %s memory to user space\n", 
           is_uncached ? "uncached" : "cached");
    
    return 0;
}

// Proc file operations
static const struct proc_ops proc_fops = {
    .proc_read = proc_read,
    .proc_write = proc_write,
    .proc_mmap = proc_mmap,
};

// Module initialization
static int __init uncached_mem_init(void)
{
    printk(KERN_INFO "Uncached memory module loaded\n");
    
    // Create proc entry
    proc_entry = proc_create(PROC_NAME, 0666, NULL, &proc_fops);
    if (!proc_entry) {
        printk(KERN_ERR "Failed to create proc entry\n");
        return -ENOMEM;
    }
    
    printk(KERN_INFO "Proc entry created: /proc/%s\n", PROC_NAME);
    return 0;
}

// Module cleanup
static void __exit uncached_mem_exit(void)
{
    // Free any allocated memory
    if (cached_buffer) {
        free_cached_memory(cached_buffer);
        cached_buffer = NULL;
        cached_allocated = 0;
    }
    
    if (uncached_buffer) {
        free_uncached_memory(uncached_buffer, BUFFER_SIZE);
        uncached_buffer = NULL;
        uncached_allocated = 0;
    }
    
    // Remove proc entry
    if (proc_entry) {
        proc_remove(proc_entry);
    }
    
    printk(KERN_INFO "Uncached memory module unloaded\n");
}

module_init(uncached_mem_init);
module_exit(uncached_mem_exit);

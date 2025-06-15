#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <time.h>
#include <sys/mman.h>
#include <errno.h>

#define SYSFS_COMMAND "/sys/kernel/uncached_mem/command"
#define SYSFS_STATUS "/sys/kernel/uncached_mem/status"
#define DEVICE_FILE "/dev/uncached_mem"
#define TEST_SIZE 4096
#define NUM_ITERATIONS 100000

// Function to get current time in nanoseconds
static long long get_time_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

// Simplified timing test function focused on ns measurements
static long long measure_memory_access(void *buffer, const char *test_type)
{
    volatile char *ptr = (volatile char *)buffer;
    long long start_time, end_time;
    int i, j;
    
    // Warm up
    for (i = 0; i < TEST_SIZE; i++) {
        ptr[i] = 0x55;
    }
    
    // Measure access time
    start_time = get_time_ns();
    for (i = 0; i < NUM_ITERATIONS; i++) {
        for (j = 0; j < TEST_SIZE; j += 64) { // Cache line steps
            volatile char dummy = ptr[j];
            (void)dummy;
        }
    }
    end_time = get_time_ns();
    
    long long total_time = end_time - start_time;
    double ns_per_access = (double)total_time / (NUM_ITERATIONS * (TEST_SIZE / 64));
    
    printf("%s: %.2f ns per access (total: %lld ns)\n", test_type, ns_per_access, total_time);
    return total_time;
}

// Function to map kernel memory to user space with buffer type
static void *map_kernel_memory(int buffer_type, size_t size)
{
    int fd;
    void *mapped_addr;
    off_t offset = buffer_type; // 0 for uncached, 1 for cached
    
    fd = open(DEVICE_FILE, O_RDWR);
    if (fd < 0) {
        perror("Failed to open device file for mmap");
        return NULL;
    }
    
    mapped_addr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, offset * getpagesize());
    if (mapped_addr == MAP_FAILED) {
        perror("mmap failed");
        close(fd);
        return NULL;
    }
    
    close(fd);
    printf("Successfully mapped %s memory to user space at %p (%zu bytes)\n", 
           buffer_type == 0 ? "uncached" : "cached", mapped_addr, size);
    return mapped_addr;
}

// Function to unmap kernel memory
static void unmap_kernel_memory(void *addr, size_t size)
{
    if (addr && addr != MAP_FAILED) {
        munmap(addr, size);
        printf("Unmapped kernel memory\n");
    }
}

// Function to send command to kernel module via sysfs
static int send_command(const char *cmd_str)
{
    int fd = open(SYSFS_COMMAND, O_WRONLY);
    if (fd < 0) {
        perror("Failed to open sysfs command file");
        return -1;
    }
    
    if (write(fd, cmd_str, strlen(cmd_str)) < 0) {
        perror("Failed to write command");
        close(fd);
        return -1;
    }
    
    close(fd);
    return 0;
}

// Function to read status from sysfs
static void read_status(void)
{
    int fd = open(SYSFS_STATUS, O_RDONLY);
    char buffer[1024];
    ssize_t bytes_read;
    
    if (fd < 0) {
        perror("Failed to open sysfs status file");
        return;
    }
    
    bytes_read = read(fd, buffer, sizeof(buffer) - 1);
    if (bytes_read > 0) {
        buffer[bytes_read] = '\0';
        printf("%s", buffer);
    }
    
    close(fd);
}

int main(void)
{
    void *cached_buffer;
    void *kernel_mapped_addr;
    long long cached_time, uncached_time, kernel_cached_time;
    size_t test_size = TEST_SIZE;
    
    printf("Memory Access Timing Test - Sysfs Interface with Variable Size Support\n");
    printf("=====================================================================\n");
    printf("Using sysfs interface: %s\n", SYSFS_COMMAND);
    printf("Using device file for mmap: %s\n", DEVICE_FILE);
    
    // Check if kernel module is loaded
    if (access(SYSFS_COMMAND, F_OK) != 0) {
        printf("Error: Kernel module not loaded or sysfs interface not available\n");
        printf("Load with: sudo insmod uncached_mem.ko\n");
        printf("Check: ls /sys/kernel/uncached_mem/\n");
        return 1;
    }
    
    // Read initial status
    printf("\n--- Initial Module Status ---\n");
    read_status();
    
    // Test 1: User space cached memory baseline
    cached_buffer = malloc(test_size);
    if (!cached_buffer) return 1;
    
    printf("\n--- Baseline: User Space Results ---\n");
    cached_time = measure_memory_access(cached_buffer, "User cached (malloc)");
    
    // Test 2: Kernel uncached memory via mmap (with size parameter)
    printf("\n--- Test 1: Kernel Uncached Memory via mmap ---\n");
    printf("Allocating uncached memory with size %zu bytes...\n", test_size);
    
    char cmd_buffer[64];
    snprintf(cmd_buffer, sizeof(cmd_buffer), "0 %zu", test_size);
    if (send_command(cmd_buffer) != 0) {
        printf("Failed to send allocation command\n");
        free(cached_buffer);
        return 1;
    }
    
    kernel_mapped_addr = map_kernel_memory(0, test_size); // map uncached buffer
    if (kernel_mapped_addr) {
        uncached_time = measure_memory_access(kernel_mapped_addr, "Kernel uncached (mmap)");
        unmap_kernel_memory(kernel_mapped_addr, test_size);
    } else {
        printf("Failed to map uncached memory\n");
        uncached_time = 0;
    }
    
    // Test 3: Kernel cached memory via mmap (with size parameter)
    printf("\n--- Test 2: Kernel Cached Memory via mmap ---\n");
    printf("Allocating cached memory with size %zu bytes...\n", test_size);
    
    snprintf(cmd_buffer, sizeof(cmd_buffer), "1 %zu", test_size);
    if (send_command(cmd_buffer) != 0) {
        printf("Failed to send allocation command\n");
        free(cached_buffer);
        return 1;
    }
    
    kernel_mapped_addr = map_kernel_memory(1, test_size); // map cached buffer
    if (kernel_mapped_addr) {
        kernel_cached_time = measure_memory_access(kernel_mapped_addr, "Kernel cached (mmap)");
        unmap_kernel_memory(kernel_mapped_addr, test_size);
    } else {
        printf("Failed to map cached memory\n");
        kernel_cached_time = 0;
    }
    
    // Clean up
    send_command("2"); // free all buffers
    
    // Test 4: Large allocation test (demonstrate size parameter)
    printf("\n--- Test 3: Large Allocation Test (1MB) ---\n");
    size_t large_size = 1024 * 1024; // 1MB
    
    printf("Testing 1MB uncached allocation...\n");
    if (send_command("0 1M") == 0) {
        kernel_mapped_addr = map_kernel_memory(0, large_size);
        if (kernel_mapped_addr) {
            // Quick test with larger buffer
            long long large_time = measure_memory_access(kernel_mapped_addr, "1MB uncached (mmap)");
            unmap_kernel_memory(kernel_mapped_addr, large_size);
        }
        send_command("2"); // free
    } else {
        printf("Failed to allocate 1MB uncached memory\n");
    }
    
    // Test 5: Cache-defeating pattern for comparison
    printf("\n--- Comparison: Cache-Defeating Pattern ---\n");
    void *large_buffer = malloc(test_size * 64);
    if (large_buffer) {
        volatile char *ptr = (volatile char *)large_buffer;
        long long start_time = get_time_ns();
        
        for (int i = 0; i < NUM_ITERATIONS; i++) {
            for (int j = 0; j < test_size; j += 4096) {
                volatile char dummy = ptr[j + (i % 64) * test_size];
                (void)dummy;
            }
        }
        
        long long end_time = get_time_ns();
        long long cache_defeat_time = end_time - start_time;
        long ns_per_access = cache_defeat_time / (NUM_ITERATIONS * (test_size / 4096));
        
        printf("Cache-defeating access: %ld ns per access (total: %lld ns)\n", 
               ns_per_access, cache_defeat_time);
        
        free(large_buffer);
    }
    
    // Performance summary
    printf("\n=== PERFORMANCE SUMMARY ===\n");
    if (cached_time > 0) {
        printf("User space cached:     %.2f ns per access\n", 
               (double)cached_time / (NUM_ITERATIONS * (test_size / 64)));
    }
    if (uncached_time > 0) {
        printf("Kernel uncached (mmap): %.2f ns per access\n", 
               (double)uncached_time / (NUM_ITERATIONS * (test_size / 64)));
    }
    if (kernel_cached_time > 0) {
        printf("Kernel cached (mmap):   %.2f ns per access\n", 
               (double)kernel_cached_time / (NUM_ITERATIONS * (test_size / 64)));
    }
    
    if (uncached_time > 0 && kernel_cached_time > 0) {
        double ratio = (double)uncached_time / kernel_cached_time;
        printf("\nPerformance ratio: Uncached is %.1fx slower than cached\n", ratio);
    }
    
    printf("\n--- Final Module Status ---\n");
    read_status();
    
    printf("\nSysfs Interface Usage:\n");
    printf("- Commands: echo 'cmd [size]' > %s\n", SYSFS_COMMAND);
    printf("- Status:   cat %s\n", SYSFS_STATUS);
    printf("- Examples: echo '0 1M' > command (1MB uncached)\n");
    printf("           echo '1 512K' > command (512KB cached)\n");
    printf("           echo '2' > command (free all)\n");
    
    printf("\nThis test uses sysfs interface with variable size support\n");
    printf("Large allocations (>1MB) use vmalloc instead of __get_free_pages\n");
    printf("Maximum allocation size: 128MB\n");
    
    free(cached_buffer);
    return 0;
}

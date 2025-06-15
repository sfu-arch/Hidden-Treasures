#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <time.h>
#include <sys/mman.h>
#include <errno.h>

#define PROC_FILE "/proc/uncached_mem"
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
static void *map_kernel_memory(int buffer_type)
{
    int fd;
    void *mapped_addr;
    off_t offset = buffer_type; // 0 for uncached, 1 for cached
    
    fd = open(PROC_FILE, O_RDWR);
    if (fd < 0) {
        perror("Failed to open proc file for mmap");
        return NULL;
    }
    
    mapped_addr = mmap(NULL, TEST_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, offset * getpagesize());
    if (mapped_addr == MAP_FAILED) {
        perror("mmap failed");
        close(fd);
        return NULL;
    }
    
    close(fd);
    printf("Successfully mapped %s memory to user space at %p\n", 
           buffer_type == 0 ? "uncached" : "cached", mapped_addr);
    return mapped_addr;
}

// Function to unmap kernel memory
static void unmap_kernel_memory(void *addr)
{
    if (addr && addr != MAP_FAILED) {
        munmap(addr, TEST_SIZE);
        printf("Unmapped kernel memory\n");
    }
}

// ...existing code...
// Function to send integer command to kernel module
static int send_command(int cmd)
{
    int fd = open(PROC_FILE, O_WRONLY);
    char cmd_str[16];
    
    if (fd < 0) return -1;
    
    snprintf(cmd_str, sizeof(cmd_str), "%d", cmd);
    write(fd, cmd_str, strlen(cmd_str));
    close(fd);
    return 0;
}

int main(void)
{
    void *cached_buffer;
    void *kernel_mapped_addr;
    long long cached_time, uncached_time, kernel_cached_time;
    
    printf("Memory Access Timing Test - Direct Kernel Memory Access via mmap\n");
    printf("=================================================================\n");
    printf("Commands: 0=alloc_uc, 1=alloc_cached, 2=free\n");
    
    // Check if kernel module is loaded
    if (access(PROC_FILE, F_OK) != 0) {
        printf("Error: Kernel module not loaded\n");
        printf("Load with: sudo insmod uncached_mem.ko\n");
        return 1;
    }
    
    // Test 1: User space cached memory baseline
    cached_buffer = malloc(TEST_SIZE);
    if (!cached_buffer) return 1;
    
    printf("\n--- Baseline: User Space Results ---\n");
    cached_time = measure_memory_access(cached_buffer, "User cached (malloc)");
    
    // Test 2: Kernel uncached memory via mmap
    printf("\n--- Test 1: Kernel Uncached Memory via mmap ---\n");
    send_command(0); // alloc_uc
    
    kernel_mapped_addr = map_kernel_memory(0); // map uncached buffer
    if (kernel_mapped_addr) {
        uncached_time = measure_memory_access(kernel_mapped_addr, "Kernel uncached (mmap)");
        unmap_kernel_memory(kernel_mapped_addr);
    } else {
        printf("Failed to map uncached memory\n");
        uncached_time = 0;
    }
    
    // Test 3: Kernel cached memory via mmap
    printf("\n--- Test 2: Kernel Cached Memory via mmap ---\n");
    send_command(1); // alloc_cached
    
    kernel_mapped_addr = map_kernel_memory(1); // map cached buffer
    if (kernel_mapped_addr) {
        kernel_cached_time = measure_memory_access(kernel_mapped_addr, "Kernel cached (mmap)");
        unmap_kernel_memory(kernel_mapped_addr);
    } else {
        printf("Failed to map cached memory\n");
        kernel_cached_time = 0;
    }
    
    send_command(2); // free all buffers
    
    // Test 4: Cache-defeating pattern for comparison
    printf("\n--- Comparison: Cache-Defeating Pattern ---\n");
    void *large_buffer = malloc(TEST_SIZE * 64);
    if (large_buffer) {
        volatile char *ptr = (volatile char *)large_buffer;
        long long start_time = get_time_ns();
        
        for (int i = 0; i < NUM_ITERATIONS; i++) {
            for (int j = 0; j < TEST_SIZE; j += 4096) {
                volatile char dummy = ptr[j + (i % 64) * TEST_SIZE];
                (void)dummy;
            }
        }
        
        long long end_time = get_time_ns();
        long long cache_defeat_time = end_time - start_time;
        long ns_per_access = cache_defeat_time / (NUM_ITERATIONS * (TEST_SIZE / 4096));
        
        printf("Cache-defeating access: %ld ns per access (total: %lld ns)\n", 
               ns_per_access, cache_defeat_time);
        
        free(large_buffer);
    }
    
    // Performance summary
    printf("\n=== PERFORMANCE SUMMARY ===\n");
    if (cached_time > 0) {
        printf("User space cached:     %.2f ns per access\n", 
               (double)cached_time / (NUM_ITERATIONS * (TEST_SIZE / 64)));
    }
    if (uncached_time > 0) {
        printf("Kernel uncached (mmap): %.2f ns per access\n", 
               (double)uncached_time / (NUM_ITERATIONS * (TEST_SIZE / 64)));
    }
    if (kernel_cached_time > 0) {
        printf("Kernel cached (mmap):   %.2f ns per access\n", 
               (double)kernel_cached_time / (NUM_ITERATIONS * (TEST_SIZE / 64)));
    }
    
    if (uncached_time > 0 && kernel_cached_time > 0) {
        double ratio = (double)uncached_time / kernel_cached_time;
        printf("\nPerformance ratio: Uncached is %.1fx slower than cached\n", ratio);
    }
    
    printf("\nThis test directly accesses kernel-allocated memory via mmap\n");
    printf("Uncached memory should show significantly higher access times\n");
    
    free(cached_buffer);
    return 0;
}

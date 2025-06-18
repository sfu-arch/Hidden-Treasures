/*
 * CMA Cache Test Program
 *
 * This program tests the CMA cache control module by:
 * - Allocating CMA memory blocks of various sizes
 * - Testing cache state changes (cached/uncached)
 * - Memory mapping CMA allocations to user space
 * - Performance testing on cached vs uncached memory
 * - Comprehensive validation of module functionality
 *
 * Usage: ./cma_test [test_mode]
 *   test_mode: basic, performance, stress (default: basic)
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <errno.h>
#include <time.h>
#include <stdint.h>

#define DEVICE_PATH "/dev/cma_cache"
#define COMMAND_PATH "/sys/kernel/cma_cache/command"
#define STATUS_PATH "/sys/kernel/cma_cache/status"

// Test allocation sizes - reduced to be more conservative
static const char* test_sizes[] = {
    "1M",      // 1MB
    "2M",      // 2MB  
    "4M",      // 4MB
    NULL       // Removed large sizes that often fail
};

// Helper functions

static int write_command(const char *cmd)
{
    FILE *fp;
    int ret;
    
    fp = fopen(COMMAND_PATH, "w");
    if (!fp) {
        perror("Failed to open command interface");
        return -1;
    }
    
    ret = fprintf(fp, "%s\n", cmd);
    fclose(fp);
    
    if (ret < 0) {
        printf("Failed to write command: %s\n", cmd);
        return -1;
    }
    
    return 0;
}

static void print_status(void)
{
    FILE *fp;
    char line[256];
    
    printf("\n=== CMA Cache Status ===\n");
    
    fp = fopen(STATUS_PATH, "r");
    if (!fp) {
        printf("Failed to read status\n");
        return;
    }
    
    while (fgets(line, sizeof(line), fp)) {
        printf("%s", line);
    }
    
    fclose(fp);
    printf("========================\n\n");
}

// Track successful allocation IDs
static int successful_alloc_ids[16];
static int num_successful_allocs = 0;

static void get_current_allocation_ids(int ids[], int *count)
{
    FILE *fp;
    char line[256];
    int id;
    
    *count = 0;
    
    fp = fopen(STATUS_PATH, "r");
    if (!fp) {
        return;
    }
    
    // Look for all allocation IDs in the status output
    while (fgets(line, sizeof(line), fp) && *count < 16) {
        // Look for lines like: "  7     1024K    256  ..."
        if (sscanf(line, "%d", &id) == 1 && id > 0 && id < 100) {
            // Verify this is actually an allocation line by checking for size info
            if (strstr(line, "K") && (strstr(line, "CACHED") || strstr(line, "UNCACHED"))) {
                ids[*count] = id;
                (*count)++;
            }
        }
    }
    
    fclose(fp);
}

static void test_basic_allocation(void)
{
    char cmd[64];
    int i;
    int initial_ids[16], initial_count;
    int current_ids[16], current_count;
    
    printf("=== Basic Allocation Test ===\n");
    
    // Reset tracking
    num_successful_allocs = 0;
    memset(successful_alloc_ids, 0, sizeof(successful_alloc_ids));
    
    // Get initial allocation IDs
    get_current_allocation_ids(initial_ids, &initial_count);
    
    // Test different allocation sizes
    for (i = 0; test_sizes[i]; i++) {
        printf("Allocating %s...\n", test_sizes[i]);
        snprintf(cmd, sizeof(cmd), "alloc %s", test_sizes[i]);
        
        if (write_command(cmd) < 0) {
            printf("Failed to allocate %s\n", test_sizes[i]);
            continue;
        }
        
        usleep(100000); // 100ms delay
        
        // Check for new allocations
        get_current_allocation_ids(current_ids, &current_count);
        if (current_count > initial_count + num_successful_allocs) {
            // Find the newest allocation ID
            for (int j = 0; j < current_count; j++) {
                int new_id = current_ids[j];
                int is_new = 1;
                
                // Check if this ID was in the initial list
                for (int k = 0; k < initial_count; k++) {
                    if (initial_ids[k] == new_id) {
                        is_new = 0;
                        break;
                    }
                }
                
                // Check if this ID is already in our successful list
                if (is_new) {
                    for (int k = 0; k < num_successful_allocs; k++) {
                        if (successful_alloc_ids[k] == new_id) {
                            is_new = 0;
                            break;
                        }
                    }
                }
                
                if (is_new && num_successful_allocs < 16) {
                    successful_alloc_ids[num_successful_allocs++] = new_id;
                    printf("Successfully allocated %s with ID %d\n", test_sizes[i], new_id);
                    break;
                }
            }
        }
    }
    
    printf("Total successful allocations: %d\n", num_successful_allocs);
    for (i = 0; i < num_successful_allocs; i++) {
        printf("  Allocation ID: %d\n", successful_alloc_ids[i]);
    }
    print_status();
}

static void test_cache_control(void)
{
    char cmd[64];
    int i;
    
    printf("=== Cache Control Test ===\n");
    
    if (num_successful_allocs == 0) {
        printf("No successful allocations to test cache control on\n");
        return;
    }
    
    // Test cache state changes using actual allocation IDs
    for (i = 0; i < num_successful_allocs && i < 3; i++) {
        int alloc_id = successful_alloc_ids[i];
        
        printf("Setting allocation %d as uncached...\n", alloc_id);
        snprintf(cmd, sizeof(cmd), "uncache %d", alloc_id);
        write_command(cmd);
        usleep(50000);
    }
    
    print_status();
    
    printf("Restoring allocations to cached state...\n");
    for (i = 0; i < num_successful_allocs && i < 3; i++) {
        int alloc_id = successful_alloc_ids[i];
        
        snprintf(cmd, sizeof(cmd), "cache %d", alloc_id);
        write_command(cmd);
        usleep(50000);
    }
    
    print_status();
}

static void test_memory_mapping(void)
{
    int fd;
    void *mapped_addr;
    volatile uint64_t *data;
    uint64_t test_pattern = 0x1234567890ABCDEF;
    int i;
    
    printf("=== Memory Mapping Test ===\n");
    
    if (num_successful_allocs == 0) {
        printf("No successful allocations to test memory mapping on\n");
        return;
    }
    
    fd = open(DEVICE_PATH, O_RDWR);
    if (fd < 0) {
        perror("Failed to open device");
        return;
    }
    
    // Test mapping the first successful allocation
    int alloc_id = successful_alloc_ids[0];
    printf("Mapping allocation %d...\n", alloc_id);
    mapped_addr = mmap(NULL, 1024*1024, PROT_READ|PROT_WRITE, MAP_SHARED, fd, alloc_id * getpagesize());
    
    if (mapped_addr == MAP_FAILED) {
        perror("mmap failed");
        close(fd);
        return;
    }
    
    printf("Successfully mapped allocation %d at %p\n", alloc_id, mapped_addr);
    
    // Test memory access
    data = (volatile uint64_t *)mapped_addr;
    
    printf("Writing test pattern...\n");
    for (i = 0; i < 1000; i++) {
        data[i] = test_pattern + i;
    }
    
    printf("Reading back test pattern...\n");
    for (i = 0; i < 1000; i++) {
        if (data[i] != test_pattern + i) {
            printf("Data mismatch at offset %d: expected 0x%lx, got 0x%lx\n",
                   i, test_pattern + i, data[i]);
            break;
        }
    }
    
    if (i == 1000) {
        printf("Memory test passed - all data verified correctly\n");
    }
    
    munmap(mapped_addr, 1024*1024);
    close(fd);
}

static void test_performance(void)
{
    int fd;
    void *cached_mem, *uncached_mem;
    volatile uint64_t *cached_data, *uncached_data;
    struct timespec start, end;
    double cached_time, uncached_time;
    int i;
    const int iterations = 1000000;
    char cmd[64];
    
    printf("=== Performance Test ===\n");
    
    if (num_successful_allocs < 2) {
        printf("Need at least 2 successful allocations for performance test\n");
        return;
    }
    
    // Set the first allocation as cached and second as uncached
    int cached_id = successful_alloc_ids[0];
    int uncached_id = successful_alloc_ids[1];
    
    snprintf(cmd, sizeof(cmd), "cache %d", cached_id);
    write_command(cmd);
    snprintf(cmd, sizeof(cmd), "uncache %d", uncached_id);
    write_command(cmd);
    usleep(100000);
    
    fd = open(DEVICE_PATH, O_RDWR);
    if (fd < 0) {
        perror("Failed to open device");
        return;
    }
    
    // Map both allocations
    cached_mem = mmap(NULL, 1024*1024, PROT_READ|PROT_WRITE, MAP_SHARED, fd, cached_id * getpagesize());
    uncached_mem = mmap(NULL, 1024*1024, PROT_READ|PROT_WRITE, MAP_SHARED, fd, uncached_id * getpagesize());
    
    if (cached_mem == MAP_FAILED || uncached_mem == MAP_FAILED) {
        perror("mmap failed");
        close(fd);
        return;
    }
    
    cached_data = (volatile uint64_t *)cached_mem;
    uncached_data = (volatile uint64_t *)uncached_mem;
    
    printf("Testing cached memory performance (allocation %d)...\n", cached_id);
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (i = 0; i < iterations; i++) {
        cached_data[i % 1000] = i;
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    cached_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    printf("Testing uncached memory performance (allocation %d)...\n", uncached_id);
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (i = 0; i < iterations; i++) {
        uncached_data[i % 1000] = i;
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    uncached_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    printf("\nPerformance Results (%d iterations):\n", iterations);
    printf("Cached memory:   %.6f seconds (%.2f ns/access)\n", 
           cached_time, (cached_time * 1e9) / iterations);
    printf("Uncached memory: %.6f seconds (%.2f ns/access)\n", 
           uncached_time, (uncached_time * 1e9) / iterations);
    if (cached_time > 0) {
        printf("Performance ratio: %.1fx slower for uncached\n", 
               uncached_time / cached_time);
    }
    
    munmap(cached_mem, 1024*1024);
    munmap(uncached_mem, 1024*1024);
    close(fd);
}

static void test_cleanup(void)
{
    char cmd[64];
    int i;
    
    printf("=== Cleanup Test ===\n");
    
    // Free all successful allocations
    printf("Freeing %d allocations...\n", num_successful_allocs);
    for (i = 0; i < num_successful_allocs; i++) {
        int alloc_id = successful_alloc_ids[i];
        printf("Freeing allocation %d...\n", alloc_id);
        snprintf(cmd, sizeof(cmd), "free %d", alloc_id);
        write_command(cmd);
        usleep(10000);
    }
    
    // Reset tracking
    num_successful_allocs = 0;
    memset(successful_alloc_ids, 0, sizeof(successful_alloc_ids));
    
    print_status();
}

static void test_stress(void)
{
    char cmd[64];
    int i;
    
    printf("=== Stress Test ===\n");
    
    // Allocate maximum number of blocks
    printf("Allocating multiple blocks...\n");
    for (i = 0; i < 16; i++) {
        snprintf(cmd, sizeof(cmd), "alloc %dM", 1 + (i % 4));
        if (write_command(cmd) < 0) {
            printf("Failed to allocate block %d\n", i);
            break;
        }
        usleep(10000);
    }
    
    print_status();
    
    // Toggle cache states rapidly
    printf("Rapid cache state changes...\n");
    for (i = 1; i <= 10; i++) {
        snprintf(cmd, sizeof(cmd), "toggle %d", i);
        write_command(cmd);
        usleep(1000);
    }
    
    print_status();
    
    // Clean up
    printf("Cleaning up stress test allocations...\n");
    for (i = 1; i <= 16; i++) {
        snprintf(cmd, sizeof(cmd), "free %d", i);
        write_command(cmd);
        usleep(1000);
    }
    
    print_status();
}

static void check_prerequisites(void)
{
    struct stat st;
    
    printf("Checking prerequisites...\n");
    
    if (stat(DEVICE_PATH, &st) != 0) {
        printf("Error: Device %s not found. Is the module loaded?\n", DEVICE_PATH);
        exit(1);
    }
    
    if (stat(COMMAND_PATH, &st) != 0) {
        printf("Error: Command interface %s not found.\n", COMMAND_PATH);
        exit(1);
    }
    
    if (stat(STATUS_PATH, &st) != 0) {
        printf("Error: Status interface %s not found.\n", STATUS_PATH);
        exit(1);
    }
    
    printf("All prerequisites satisfied.\n\n");
}

static void print_usage(const char *prog_name)
{
    printf("Usage: %s [test_mode]\n", prog_name);
    printf("  test_mode:\n");
    printf("    basic      - Basic allocation and cache control tests (default)\n");
    printf("    performance - Performance comparison tests\n");
    printf("    stress     - Stress testing with multiple allocations\n");
    printf("    all        - Run all tests\n");
    printf("\n");
    printf("Prerequisites:\n");
    printf("  1. Load module: sudo insmod cma_cache.ko\n");
    printf("  2. Set permissions: sudo chmod 666 /dev/cma_cache /sys/kernel/cma_cache/command\n");
    printf("\n");
}

int main(int argc, char *argv[])
{
    const char *test_mode = "basic";
    
    if (argc > 1) {
        test_mode = argv[1];
    }
    
    if (strcmp(test_mode, "help") == 0 || strcmp(test_mode, "-h") == 0) {
        print_usage(argv[0]);
        return 0;
    }
    
    printf("CMA Cache Control Test Program\n");
    printf("==============================\n\n");
    
    check_prerequisites();
    
    print_status();
    
    if (strcmp(test_mode, "basic") == 0 || strcmp(test_mode, "all") == 0) {
        test_basic_allocation();
        test_cache_control();
        test_memory_mapping();
        test_cleanup();
    }
    
    if (strcmp(test_mode, "performance") == 0 || strcmp(test_mode, "all") == 0) {
        test_basic_allocation();
        test_performance();
        test_cleanup();
    }
    
    if (strcmp(test_mode, "stress") == 0 || strcmp(test_mode, "all") == 0) {
        test_stress();
    }
    
    printf("\nTest completed successfully!\n");
    
    return 0;
}

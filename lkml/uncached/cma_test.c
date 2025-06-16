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

// Test allocation sizes
static const char* test_sizes[] = {
    "1M",      // 1MB
    "4M",      // 4MB  
    "16M",     // 16MB
    "64M",     // 64MB
    NULL
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

static void test_basic_allocation(void)
{
    char cmd[64];
    int i;
    
    printf("=== Basic Allocation Test ===\n");
    
    // Test different allocation sizes
    for (i = 0; test_sizes[i]; i++) {
        printf("Allocating %s...\n", test_sizes[i]);
        snprintf(cmd, sizeof(cmd), "alloc %s", test_sizes[i]);
        
        if (write_command(cmd) < 0) {
            printf("Failed to allocate %s\n", test_sizes[i]);
            continue;
        }
        
        usleep(100000); // 100ms delay
    }
    
    print_status();
}

static void test_cache_control(void)
{
    printf("=== Cache Control Test ===\n");
    
    // Test cache state changes
    printf("Setting allocation 1 as uncached...\n");
    write_command("uncache 1");
    usleep(50000);
    
    printf("Setting allocation 2 as uncached...\n");
    write_command("uncache 2");
    usleep(50000);
    
    printf("Toggling allocation 3 cache state...\n");
    write_command("toggle 3");
    usleep(50000);
    
    print_status();
    
    printf("Restoring allocations to cached state...\n");
    write_command("cache 1");
    write_command("cache 2");
    write_command("cache 3");
    usleep(50000);
    
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
    
    fd = open(DEVICE_PATH, O_RDWR);
    if (fd < 0) {
        perror("Failed to open device");
        return;
    }
    
    // Test mapping allocation 1 (should be 1MB)
    printf("Mapping allocation 1...\n");
    mapped_addr = mmap(NULL, 1024*1024, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 1 * getpagesize());
    
    if (mapped_addr == MAP_FAILED) {
        perror("mmap failed");
        close(fd);
        return;
    }
    
    printf("Successfully mapped allocation 1 at %p\n", mapped_addr);
    
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
    
    printf("=== Performance Test ===\n");
    
    // Set the allocations that were just created as cached/uncached
    // In makefile run, these will be 3 and 4; in standalone run, they'll be 1 and 2
    // Let's try both possibilities to be robust
    write_command("cache 1");
    write_command("uncache 2");
    write_command("cache 3"); 
    write_command("uncache 4");
    usleep(100000);
    
    fd = open(DEVICE_PATH, O_RDWR);
    if (fd < 0) {
        perror("Failed to open device");
        return;
    }
    
    // Try mapping allocations 1 and 2 first, then 3 and 4 as fallback
    cached_mem = mmap(NULL, 1024*1024, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 1 * getpagesize());
    if (cached_mem == MAP_FAILED) {
        // Try allocation 3 instead
        cached_mem = mmap(NULL, 1024*1024, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 3 * getpagesize());
    }
    
    uncached_mem = mmap(NULL, 1024*1024, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 2 * getpagesize());
    if (uncached_mem == MAP_FAILED) {
        // Try allocation 4 instead
        uncached_mem = mmap(NULL, 1024*1024, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 4 * getpagesize());
    }
    
    if (cached_mem == MAP_FAILED || uncached_mem == MAP_FAILED) {
        perror("mmap failed");
        close(fd);
        return;
    }
    
    cached_data = (volatile uint64_t *)cached_mem;
    uncached_data = (volatile uint64_t *)uncached_mem;
    
    printf("Testing cached memory performance...\n");
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (i = 0; i < iterations; i++) {
        cached_data[i % 1000] = i;
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    cached_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    printf("Testing uncached memory performance...\n");
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
    printf("Performance ratio: %.1fx slower for uncached\n", 
           uncached_time / cached_time);
    
    munmap(cached_mem, 1024*1024);
    munmap(uncached_mem, 1024*1024);
    close(fd);
}

static void test_cleanup(void)
{
    printf("=== Cleanup Test ===\n");
    
    // Free all allocations
    printf("Freeing all allocations...\n");
    write_command("free 1");
    write_command("free 2");
    write_command("free 3");
    write_command("free 4");
    usleep(100000);
    
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

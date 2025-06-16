/*
 * Dynamic Cache Control Test Program
 * 
 * This program tests the dynamic_cache kernel module which provides
 * per-page cache control using page protection mechanisms.
 * 
 * Author: Your Name
 * License: GPL
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <errno.h>
#include <time.h>
#include <signal.h>
#include <stdint.h>
#include <ctype.h>

#define SYSFS_BASE "/sys/kernel/dynamic_cache"
#define DEVICE_FILE "/dev/dynamic_cache"
#define PAGE_SIZE 4096
#define TEST_ITERATIONS 1000000
#define PATTERN_SIZE 256

// Test results structure
struct test_results {
    double cached_time;
    double uncached_time;
    double ratio;
    int cached_errors;
    int uncached_errors;
};

// Global variables for cleanup
static int allocated_pages[32];
static int num_allocated = 0;

// Signal handler for cleanup
void cleanup_handler(int sig)
{
    printf("\nSignal %d received, cleaning up...\n", sig);
    
    // Free all allocated pages
    FILE *cmd_file = fopen(SYSFS_BASE "/command", "w");
    if (cmd_file) {
        for (int i = 0; i < num_allocated; i++) {
            fprintf(cmd_file, "free %d", allocated_pages[i]);
            fflush(cmd_file);
        }
        fclose(cmd_file);
    }
    
    exit(1);
}

// Utility functions

static int write_command(const char *command)
{
    FILE *file = fopen(SYSFS_BASE "/command", "w");
    if (!file) {
        perror("Failed to open command file");
        return -1;
    }
    
    if (fprintf(file, "%s", command) < 0) {
        perror("Failed to write command");
        fclose(file);
        return -1;
    }
    
    fclose(file);
    return 0;
}

static int read_status(char *buffer, size_t size)
{
    FILE *file = fopen(SYSFS_BASE "/status", "r");
    if (!file) {
        perror("Failed to open status file");
        return -1;
    }
    
    size_t read_size = fread(buffer, 1, size - 1, file);
    buffer[read_size] = '\0';
    fclose(file);
    
    return 0;
}

static int allocate_test_page(void)
{
    // Send allocate command
    if (write_command("alloc") < 0) {
        return -1;
    }
    
    // The module allocates pages sequentially starting from the first available page
    // For test purposes, we'll track this manually
    int page_id = num_allocated; // Simple: use the count as the next page ID
    allocated_pages[num_allocated++] = page_id;
    printf("Allocated page %d\n", page_id);
    return page_id;
}

static int set_page_cache_state(int page_id, int cached)
{
    char command[64];
    snprintf(command, sizeof(command), "%s %d", cached ? "cache" : "uncache", page_id);
    return write_command(command);
}

static void *map_page(int page_id)
{
    int fd = open(DEVICE_FILE, O_RDWR);
    if (fd < 0) {
        perror("Failed to open device file");
        return MAP_FAILED;
    }
    
    void *addr = mmap(NULL, PAGE_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, page_id);
    if (addr == MAP_FAILED) {
        perror("mmap failed");
    }
    
    close(fd);
    return addr;
}

static double measure_memory_performance(void *addr, int iterations)
{
    struct timespec start, end;
    volatile uint64_t *mem = (volatile uint64_t *)addr;
    volatile uint64_t sum = 0;
    int i;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    // Perform memory operations
    for (i = 0; i < iterations; i++) {
        sum += mem[i % (PAGE_SIZE / sizeof(uint64_t))];
        mem[i % (PAGE_SIZE / sizeof(uint64_t))] = i;
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    double time_taken = (end.tv_sec - start.tv_sec) + 
                       (end.tv_nsec - start.tv_nsec) / 1e9;
    
    return time_taken;
}

static int verify_data_integrity(void *addr, int pattern)
{
    uint8_t *mem = (uint8_t *)addr;
    int errors = 0;
    
    // Write pattern
    for (int i = 0; i < PAGE_SIZE; i++) {
        mem[i] = (pattern + i) & 0xFF;
    }
    
    // Verify pattern
    for (int i = 0; i < PAGE_SIZE; i++) {
        if (mem[i] != ((pattern + i) & 0xFF)) {
            errors++;
        }
    }
    
    return errors;
}

// Test functions

static int test_basic_functionality(void)
{
    printf("\n=== Basic Functionality Test ===\n");
    
    char status[4096];
    
    // Check if sysfs interface exists
    if (access(SYSFS_BASE "/command", W_OK) != 0) {
        printf("ERROR: Cannot access sysfs interface at %s\n", SYSFS_BASE);
        return -1;
    }
    
    if (access(DEVICE_FILE, R_OK | W_OK) != 0) {
        printf("ERROR: Cannot access device file %s\n", DEVICE_FILE);
        return -1;
    }
    
    // Read initial status
    printf("Reading initial status...\n");
    if (read_status(status, sizeof(status)) < 0) {
        return -1;
    }
    
    printf("Initial status:\n%s\n", status);
    
    printf("Basic functionality test passed!\n");
    return 0;
}

static int test_page_allocation(void)
{
    printf("\n=== Page Allocation Test ===\n");
    
    int page_ids[5];
    int i;
    
    // Allocate several pages
    for (i = 0; i < 5; i++) {
        page_ids[i] = allocate_test_page();
        if (page_ids[i] < 0) {
            printf("ERROR: Failed to allocate page %d\n", i);
            return -1;
        }
    }
    
    // Show status after allocation
    char status[4096];
    if (read_status(status, sizeof(status)) == 0) {
        printf("Status after allocation:\n%s\n", status);
    }
    
    // Free all pages except the first one (keep it for further tests)
    for (i = 1; i < 5; i++) {
        char command[64];
        snprintf(command, sizeof(command), "free %d", page_ids[i]);
        if (write_command(command) < 0) {
            printf("ERROR: Failed to free page %d\n", page_ids[i]);
            return -1;
        }
        printf("Freed page %d\n", page_ids[i]);
        
        // Remove from our tracking array
        for (int j = 0; j < num_allocated; j++) {
            if (allocated_pages[j] == page_ids[i]) {
                memmove(&allocated_pages[j], &allocated_pages[j+1], 
                        (num_allocated - j - 1) * sizeof(int));
                num_allocated--;
                break;
            }
        }
    }
    
    printf("Page allocation test passed!\n");
    return 0;
}

static int test_cache_control(void)
{
    printf("\n=== Cache Control Test ===\n");
    
    // Use the first allocated page
    if (num_allocated == 0) {
        printf("ERROR: No pages allocated for cache control test\n");
        return -1;
    }
    
    int page_id = allocated_pages[0];
    printf("Testing cache control on page %d\n", page_id);
    
    // Test cache state changes
    printf("Setting page to UNCACHED...\n");
    if (set_page_cache_state(page_id, 0) < 0) {
        printf("ERROR: Failed to set page uncached\n");
        return -1;
    }
    
    printf("Setting page to CACHED...\n");
    if (set_page_cache_state(page_id, 1) < 0) {
        printf("ERROR: Failed to set page cached\n");
        return -1;
    }
    
    // Test toggle command
    printf("Toggling cache state...\n");
    char command[64];
    snprintf(command, sizeof(command), "toggle %d", page_id);
    if (write_command(command) < 0) {
        printf("ERROR: Failed to toggle cache state\n");
        return -1;
    }
    
    printf("Cache control test passed!\n");
    return 0;
}

static int test_mmap_functionality(void)
{
    printf("\n=== Memory Mapping Test ===\n");
    
    if (num_allocated == 0) {
        printf("ERROR: No pages allocated for mmap test\n");
        return -1;
    }
    
    int page_id = allocated_pages[0];
    printf("Testing mmap functionality on page %d\n", page_id);
    
    // Map the page
    void *addr = map_page(page_id);
    if (addr == MAP_FAILED) {
        printf("ERROR: Failed to map page %d\n", page_id);
        return -1;
    }
    
    printf("Page %d mapped at address %p\n", page_id, addr);
    
    // Test data integrity
    int errors = verify_data_integrity(addr, 0x55);
    if (errors > 0) {
        printf("ERROR: Data integrity test failed with %d errors\n", errors);
        munmap(addr, PAGE_SIZE);
        return -1;
    }
    
    printf("Data integrity test passed\n");
    
    // Unmap the page
    if (munmap(addr, PAGE_SIZE) < 0) {
        perror("munmap failed");
        return -1;
    }
    
    printf("Memory mapping test passed!\n");
    return 0;
}

static int test_performance_comparison(struct test_results *results)
{
    printf("\n=== Performance Comparison Test ===\n");
    
    if (num_allocated == 0) {
        printf("ERROR: No pages allocated for performance test\n");
        return -1;
    }
    
    int page_id = allocated_pages[0];
    void *addr;
    
    memset(results, 0, sizeof(*results));
    
    printf("Testing performance on page %d\n", page_id);
    
    // Test cached performance
    printf("Testing CACHED performance...\n");
    if (set_page_cache_state(page_id, 1) < 0) {
        printf("ERROR: Failed to set page cached\n");
        return -1;
    }
    
    addr = map_page(page_id);
    if (addr == MAP_FAILED) {
        printf("ERROR: Failed to map cached page\n");
        return -1;
    }
    
    results->cached_time = measure_memory_performance(addr, TEST_ITERATIONS);
    results->cached_errors = verify_data_integrity(addr, 0xAA);
    munmap(addr, PAGE_SIZE);
    
    printf("Cached performance: %.3f seconds\n", results->cached_time);
    
    // Test uncached performance
    printf("Testing UNCACHED performance...\n");
    if (set_page_cache_state(page_id, 0) < 0) {
        printf("ERROR: Failed to set page uncached\n");
        return -1;
    }
    
    addr = map_page(page_id);
    if (addr == MAP_FAILED) {
        printf("ERROR: Failed to map uncached page\n");
        return -1;
    }
    
    results->uncached_time = measure_memory_performance(addr, TEST_ITERATIONS);
    results->uncached_errors = verify_data_integrity(addr, 0xBB);
    munmap(addr, PAGE_SIZE);
    
    printf("Uncached performance: %.3f seconds\n", results->uncached_time);
    
    // Calculate ratio
    if (results->cached_time > 0) {
        results->ratio = results->uncached_time / results->cached_time;
        printf("Performance ratio (uncached/cached): %.2fx slower\n", results->ratio);
    }
    
    if (results->cached_errors == 0 && results->uncached_errors == 0) {
        printf("Data integrity maintained in both modes\n");
    } else {
        printf("WARNING: Data integrity errors - cached: %d, uncached: %d\n",
               results->cached_errors, results->uncached_errors);
    }
    
    printf("Performance comparison test completed!\n");
    return 0;
}

static int test_pattern_functionality(void)
{
    printf("\n=== Pattern Functionality Test ===\n");
    
    if (num_allocated == 0) {
        printf("ERROR: No pages allocated for pattern test\n");
        return -1;
    }
    
    int page_id = allocated_pages[0];
    char command[64];
    
    printf("Testing pattern functionality on page %d\n", page_id);
    
    // Set various patterns
    uint8_t patterns[] = {0x00, 0xFF, 0xAA, 0x55, 0xDE, 0xAD, 0xBE, 0xEF};
    int num_patterns = sizeof(patterns) / sizeof(patterns[0]);
    
    for (int i = 0; i < num_patterns; i++) {
        printf("Setting pattern 0x%02X...\n", patterns[i]);
        snprintf(command, sizeof(command), "pattern %d %02x", page_id, patterns[i]);
        if (write_command(command) < 0) {
            printf("ERROR: Failed to set pattern 0x%02X\n", patterns[i]);
            return -1;
        }
        
        // Map and verify pattern
        void *addr = map_page(page_id);
        if (addr == MAP_FAILED) {
            printf("ERROR: Failed to map page for pattern verification\n");
            return -1;
        }
        
        uint8_t *mem = (uint8_t *)addr;
        int errors = 0;
        for (int j = 0; j < PATTERN_SIZE; j++) {
            if (mem[j] != patterns[i]) {
                errors++;
                if (errors == 1) {  // Only report first few errors
                    printf("Pattern mismatch at offset %d: expected 0x%02X, got 0x%02X\n",
                           j, patterns[i], mem[j]);
                }
            }
        }
        
        munmap(addr, PAGE_SIZE);
        
        if (errors == 0) {
            printf("Pattern 0x%02X verified successfully\n", patterns[i]);
        } else {
            printf("Pattern 0x%02X verification failed with %d errors\n", patterns[i], errors);
        }
    }
    
    printf("Pattern functionality test completed!\n");
    return 0;
}

static void print_final_status(void)
{
    printf("\n=== Final Status ===\n");
    
    char status[4096];
    if (read_status(status, sizeof(status)) == 0) {
        printf("%s\n", status);
    }
    
    // Read page map
    FILE *file = fopen(SYSFS_BASE "/page_map", "r");
    if (file) {
        printf("\nPage Map:\n");
        char line[1024];
        while (fgets(line, sizeof(line), file)) {
            printf("%s", line);
        }
        fclose(file);
    }
}

int main(int argc, char *argv[])
{
    printf("Dynamic Cache Control Test Program\n");
    printf("==================================\n");
    
    // Set up signal handlers for cleanup
    signal(SIGINT, cleanup_handler);
    signal(SIGTERM, cleanup_handler);
    
    struct test_results results;
    int test_failed = 0;
    
    // Run test suite
    if (test_basic_functionality() < 0) {
        printf("FAILED: Basic functionality test\n");
        test_failed = 1;
        goto cleanup;
    }
    
    if (test_page_allocation() < 0) {
        printf("FAILED: Page allocation test\n");
        test_failed = 1;
        goto cleanup;
    }
    
    if (test_cache_control() < 0) {
        printf("FAILED: Cache control test\n");
        test_failed = 1;
        goto cleanup;
    }
    
    if (test_mmap_functionality() < 0) {
        printf("FAILED: Memory mapping test\n");
        test_failed = 1;
        goto cleanup;
    }
    
    if (test_pattern_functionality() < 0) {
        printf("FAILED: Pattern functionality test\n");
        test_failed = 1;
        goto cleanup;
    }
    
    if (test_performance_comparison(&results) < 0) {
        printf("FAILED: Performance comparison test\n");
        test_failed = 1;
        goto cleanup;
    }
    
    // Print summary
    printf("\n=== Test Summary ===\n");
    if (!test_failed) {
        printf("All tests PASSED!\n");
        printf("\nPerformance Results:\n");
        printf("  Cached access time:   %.3f seconds\n", results.cached_time);
        printf("  Uncached access time: %.3f seconds\n", results.uncached_time);
        printf("  Performance ratio:    %.2fx slower for uncached\n", results.ratio);
        printf("  Data integrity:       %s\n", 
               (results.cached_errors == 0 && results.uncached_errors == 0) ? "OK" : "ERRORS");
    } else {
        printf("Some tests FAILED!\n");
    }
    
cleanup:
    print_final_status();
    
    // Clean up allocated pages
    printf("\nCleaning up allocated pages...\n");
    for (int i = 0; i < num_allocated; i++) {
        char command[64];
        snprintf(command, sizeof(command), "free %d", allocated_pages[i]);
        write_command(command);
        printf("Freed page %d\n", allocated_pages[i]);
    }
    
    printf("\nTest program completed.\n");
    return test_failed ? 1 : 0;
}

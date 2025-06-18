/*
 * CMA Debug Tool
 * 
 * This tool helps diagnose CMA allocation issues by:
 * - Checking available CMA memory
 * - Testing allocation sizes incrementally
 * - Cleaning up leftover allocations
 * - Providing detailed error information
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <sys/stat.h>
#include <errno.h>

#define COMMAND_PATH "/sys/kernel/cma_cache/command"
#define STATUS_PATH "/sys/kernel/cma_cache/status"

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

static void cleanup_all_allocations(void)
{
    char cmd[64];
    int i;
    
    printf("=== Cleaning up all allocations ===\n");
    
    // Try to free allocations 1-32
    for (i = 1; i <= 32; i++) {
        snprintf(cmd, sizeof(cmd), "free %d", i);
        write_command(cmd);
        usleep(10000);
    }
    
    print_status();
}

static void check_cma_info(void)
{
    FILE *fp;
    char line[256];
    
    printf("=== System CMA Information ===\n");
    
    // Check /proc/meminfo for CMA info
    fp = fopen("/proc/meminfo", "r");
    if (fp) {
        while (fgets(line, sizeof(line), fp)) {
            if (strstr(line, "Cma")) {
                printf("%s", line);
            }
        }
        fclose(fp);
    }
    
    // Check dmesg for CMA initialization
    printf("\n=== CMA initialization messages ===\n");
    system("dmesg | grep -i cma | tail -10");
    
    printf("\n");
}

static void test_incremental_allocation(void)
{
    char cmd[64];
    int size_mb;
    int success_count = 0;
    int alloc_ids[16];
    
    printf("=== Incremental Allocation Test ===\n");
    
    // Test allocation sizes from 1MB up to 16MB
    for (size_mb = 1; size_mb <= 16; size_mb++) {
        printf("Testing %dMB allocation...\n", size_mb);
        snprintf(cmd, sizeof(cmd), "alloc %dM", size_mb);
        
        if (write_command(cmd) == 0) {
            printf("  SUCCESS: %dMB allocated\n", size_mb);
            success_count++;
            usleep(100000);
            print_status();
        } else {
            printf("  FAILED: %dMB allocation failed\n", size_mb);
            break;
        }
    }
    
    printf("Successfully allocated %d blocks\n", success_count);
    
    // Clean up
    printf("Cleaning up test allocations...\n");
    cleanup_all_allocations();
}

static void print_usage(const char *prog_name)
{
    printf("Usage: %s [command]\n", prog_name);
    printf("Commands:\n");
    printf("  status    - Show current allocation status\n");
    printf("  cleanup   - Clean up all allocations\n");
    printf("  info      - Show system CMA information\n");
    printf("  test      - Run incremental allocation test\n");
    printf("  all       - Run all diagnostic commands\n");
    printf("\n");
}

int main(int argc, char *argv[])
{
    const char *cmd = "all";
    
    if (argc > 1) {
        cmd = argv[1];
    }
    
    printf("CMA Debug Tool\n");
    printf("==============\n\n");
    
    // Check if module is loaded
    struct stat st;
    if (stat(COMMAND_PATH, &st) != 0) {
        printf("Error: CMA cache module not loaded or accessible\n");
        printf("Try: sudo insmod cma_cache.ko\n");
        exit(1);
    }
    
    if (strcmp(cmd, "status") == 0 || strcmp(cmd, "all") == 0) {
        print_status();
    }
    
    if (strcmp(cmd, "cleanup") == 0 || strcmp(cmd, "all") == 0) {
        cleanup_all_allocations();
    }
    
    if (strcmp(cmd, "info") == 0 || strcmp(cmd, "all") == 0) {
        check_cma_info();
    }
    
    if (strcmp(cmd, "test") == 0 || strcmp(cmd, "all") == 0) {
        test_incremental_allocation();
    }
    
    if (strcmp(cmd, "help") == 0 || strcmp(cmd, "-h") == 0) {
        print_usage(argv[0]);
    }
    
    return 0;
}

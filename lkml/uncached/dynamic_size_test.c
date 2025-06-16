#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <errno.h>

#define SYSFS_COMMAND "/sys/kernel/dynamic_cache/command"
#define SYSFS_STATUS "/sys/kernel/dynamic_cache/status"
#define DEVICE_PATH "/dev/dynamic_cache"

void test_allocation(const char *size_spec, const char *description) {
    FILE *cmd_file;
    char command[256];
    
    printf("\n=== Test: %s ===\n", description);
    
    // Allocate with specified size
    cmd_file = fopen(SYSFS_COMMAND, "w");
    if (!cmd_file) {
        perror("Failed to open command interface");
        return;
    }
    
    if (size_spec) {
        snprintf(command, sizeof(command), "alloc %s", size_spec);
    } else {
        strcpy(command, "alloc");
    }
    
    printf("Command: %s\n", command);
    fprintf(cmd_file, "%s\n", command);
    fclose(cmd_file);
    
    // Check dmesg for allocation result
    printf("Check dmesg output above for allocation result\n");
    sleep(1); // Give time for kernel message
}

void show_status() {
    FILE *status_file;
    char buffer[8192];
    size_t n;
    
    printf("\n=== Current Status ===\n");
    
    status_file = fopen(SYSFS_STATUS, "r");
    if (!status_file) {
        perror("Failed to open status interface");
        return;
    }
    
    while ((n = fread(buffer, 1, sizeof(buffer) - 1, status_file)) > 0) {
        buffer[n] = '\0';
        printf("%s", buffer);
    }
    
    fclose(status_file);
}

void test_free_block(int block_id) {
    FILE *cmd_file;
    char command[256];
    
    printf("\n=== Freeing Block %d ===\n", block_id);
    
    cmd_file = fopen(SYSFS_COMMAND, "w");
    if (!cmd_file) {
        perror("Failed to open command interface");
        return;
    }
    
    snprintf(command, sizeof(command), "free_block %d", block_id);
    printf("Command: %s\n", command);
    fprintf(cmd_file, "%s\n", command);
    fclose(cmd_file);
    
    printf("Check dmesg output above for free result\n");
    sleep(1);
}

void test_cache_control_on_block(int page_idx) {
    FILE *cmd_file;
    char command[256];
    
    printf("\n=== Testing Cache Control on Page %d ===\n", page_idx);
    
    cmd_file = fopen(SYSFS_COMMAND, "w");
    if (!cmd_file) {
        perror("Failed to open command interface");
        return;
    }
    
    // Test uncache
    snprintf(command, sizeof(command), "uncache %d", page_idx);
    printf("Command: %s\n", command);
    fprintf(cmd_file, "%s\n", command);
    fflush(cmd_file);
    sleep(1);
    
    // Test cache
    snprintf(command, sizeof(command), "cache %d", page_idx);
    printf("Command: %s\n", command);
    fprintf(cmd_file, "%s\n", command);
    
    fclose(cmd_file);
    sleep(1);
}

int main() {
    printf("Dynamic Cache Variable Size Allocation Test\n");
    printf("===========================================\n");
    
    printf("Testing allocation of different sizes in dynamic_cache module\n");
    printf("Make sure the module is loaded: sudo insmod dynamic_cache.ko\n");
    printf("Monitor kernel messages: sudo dmesg -w (in another terminal)\n");
    printf("Note: This test requires write access to sysfs files\n");
    printf("Run with: sudo ./dynamic_size_test or use chmod 666 on sysfs files\n\n");
    
    // Initial status
    show_status();
    
    // Test 1: Default single page allocation
    test_allocation(NULL, "Default single page (4KB)");
    show_status();
    
    // Test 2: Explicit single page
    test_allocation("4096", "Explicit single page (4096 bytes)");
    show_status();
    
    // Test 3: Two pages
    test_allocation("8K", "Two pages (8KB)");
    show_status();
    
    // Test 4: Large allocation with suffix
    test_allocation("64K", "Large allocation (64KB = 16 pages)");
    show_status();
    
    // Test 5: Megabyte allocation
    test_allocation("1M", "1 MB allocation (256 pages)");
    show_status();
    
    // Test cache control on pages within blocks
    printf("\n=== Testing Cache Control on Block Pages ===\n");
    test_cache_control_on_block(0);  // Should be a single page
    test_cache_control_on_block(1);  // Should be another single page
    test_cache_control_on_block(2);  // Should be first page of 2-page block
    test_cache_control_on_block(3);  // Should be second page of 2-page block
    
    show_status();
    
    // Test freeing blocks (block IDs start from 1)
    printf("\n=== Testing Block Freeing ===\n");
    test_free_block(3);  // Free the 64K block
    show_status();
    
    test_free_block(4);  // Free the 1M block  
    show_status();
    
    test_free_block(2);  // Free the 8K block
    show_status();
    
    // Try to free individual pages (should fail for block pages, succeed for single pages)
    printf("\n=== Testing Individual Page Freeing ===\n");
    FILE *cmd_file = fopen(SYSFS_COMMAND, "w");
    if (cmd_file) {
        printf("Trying to free page 0 (single page - should succeed)\n");
        fprintf(cmd_file, "free 0\n");
        fflush(cmd_file);
        sleep(1);
        
        printf("Trying to free page 1 (single page - should succeed)\n");
        fprintf(cmd_file, "free 1\n");
        fclose(cmd_file);
        sleep(1);
    }
    
    // Final status
    show_status();
    
    printf("\nTest completed. Check dmesg output for detailed allocation information.\n");
    return 0;
}

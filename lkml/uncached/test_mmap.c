#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <stdint.h>
#include <errno.h>
#include <string.h>

int main() {
    printf("Testing mmap with both cached and uncached pages...\n");
    
    // Test page 0 (uncached) - open first fd
    int fd0 = open("/dev/dynamic_cache", O_RDWR);
    if (fd0 < 0) {
        perror("Failed to open device for page 0");
        return 1;
    }
    printf("Device fd0: %d\n", fd0);
    
    printf("Attempting to map page 0...\n");
    void *addr0 = mmap(NULL, 4096, PROT_READ|PROT_WRITE, MAP_SHARED, fd0, 0);
    if (addr0 == MAP_FAILED) {
        printf("mmap page 0 failed: %s (errno=%d)\n", strerror(errno), errno);
        close(fd0);
        return 1;
    }
    printf("Page 0 (uncached) mapped at %p\n", addr0);
    
    // Test page 1 (cached) - open second fd
    int fd1 = open("/dev/dynamic_cache", O_RDWR);
    if (fd1 < 0) {
        perror("Failed to open device for page 1");
        munmap(addr0, 4096);
        close(fd0);
        return 1;
    }
    printf("Device fd1: %d\n", fd1);
    
    printf("Attempting to map page 1...\n");  
    void *addr1 = mmap(NULL, 4096, PROT_READ|PROT_WRITE, MAP_SHARED, fd1, 4096);  // Page 1 at offset 4096
    if (addr1 == MAP_FAILED) {
        printf("mmap page 1 failed: %s (errno=%d)\n", strerror(errno), errno);
        munmap(addr0, 4096);
        close(fd0);
        close(fd1);
        return 1;
    }
    printf("Page 1 (cached) mapped at %p\n", addr1);
    
    // Write test data
    volatile uint64_t *data0 = (volatile uint64_t *)addr0;
    volatile uint64_t *data1 = (volatile uint64_t *)addr1;
    
    *data0 = 0xDEADBEEFCAFEBABE;
    *data1 = 0x123456789ABCDEF0;
    
    printf("Wrote test data:\n");
    printf("  Page 0 (uncached): 0x%016lx\n", *data0);
    printf("  Page 1 (cached):   0x%016lx\n", *data1);
    
    // Verify data
    if (*data0 == 0xDEADBEEFCAFEBABE && *data1 == 0x123456789ABCDEF0) {
        printf("Data verification: PASSED\n");
    } else {
        printf("Data verification: FAILED\n");
    }
    
    // Clean up
    munmap(addr0, 4096);
    munmap(addr1, 4096);
    close(fd0);
    close(fd1);
    
    printf("mmap test completed successfully!\n");
    return 0;
}

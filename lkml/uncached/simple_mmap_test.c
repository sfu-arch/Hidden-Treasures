#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <stdint.h>

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <page_id>\n", argv[0]);
        return 1;
    }
    
    int page_id = atoi(argv[1]);
    printf("Testing mmap for page %d\n", page_id);
    
    int fd = open("/dev/dynamic_cache", O_RDWR);
    if (fd < 0) {
        perror("Failed to open device");
        return 1;
    }
    
    void *addr = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, page_id);
    if (addr == MAP_FAILED) {
        perror("mmap failed");
        close(fd);
        return 1;
    }
    
    printf("Successfully mapped page %d at address %p\n", page_id, addr);
    
    // Test basic read/write
    volatile uint64_t *data = (volatile uint64_t *)addr;
    uint64_t test_value = 0x1234567890ABCDEF;
    
    *data = test_value;
    uint64_t read_value = *data;
    
    if (read_value == test_value) {
        printf("Data integrity test passed: wrote 0x%llx, read 0x%llx\n", 
               (unsigned long long)test_value, (unsigned long long)read_value);
    } else {
        printf("Data integrity test failed: wrote 0x%llx, read 0x%llx\n", 
               (unsigned long long)test_value, (unsigned long long)read_value);
    }
    
    munmap(addr, 4096);
    close(fd);
    
    return 0;
}

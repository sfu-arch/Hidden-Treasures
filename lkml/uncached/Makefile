# Makefile for memory management kernel modules

# Module names
MODULE_NAME = uncached_mem
DYNAMIC_MODULE_NAME = dynamic_cache
CMA_MODULE_NAME = cma_cache

# Source files
obj-m += $(MODULE_NAME).o $(DYNAMIC_MODULE_NAME).o $(CMA_MODULE_NAME).o

# Kernel build directory (automatically detected)
KERNEL_BUILD := /lib/modules/$(shell uname -r)/build

# Current directory
PWD := $(shell pwd)

# Default target
all:
	$(MAKE) -C $(KERNEL_BUILD) M=$(PWD) modules

# Clean target
clean:
	$(MAKE) -C $(KERNEL_BUILD) M=$(PWD) clean
	rm -f timing_test dynamic_test dynamic_size_test cma_test cma_debug

# Install module (requires root privileges)
install:
	$(MAKE) -C $(KERNEL_BUILD) M=$(PWD) modules_install
	depmod -a

# Load module
load:
	sudo insmod $(MODULE_NAME).ko

# Load dynamic cache module
load_dynamic:
	sudo insmod $(DYNAMIC_MODULE_NAME).ko

# Load CMA cache module
load_cma:
	sudo insmod $(CMA_MODULE_NAME).ko

# Unload module
unload:
	sudo rmmod $(MODULE_NAME)

# Unload dynamic cache module
unload_dynamic:
	sudo rmmod $(DYNAMIC_MODULE_NAME)

# Unload CMA cache module
unload_cma:
	sudo rmmod $(CMA_MODULE_NAME)

# Build user space test program
timing_test: timing_test.c
	gcc -o timing_test timing_test.c -lrt

# Build dynamic cache test program
dynamic_test: dynamic_test.c
	gcc -o dynamic_test dynamic_test.c -lrt

# Build dynamic cache size test program
dynamic_size_test: dynamic_size_test.c
	gcc -o dynamic_size_test dynamic_size_test.c

# Build CMA cache test program
cma_test: cma_test.c
	gcc -o cma_test cma_test.c -lrt

# Build CMA debug tool  
cma_debug: cma_debug.c
	gcc -o cma_debug cma_debug.c

# Run tests (requires module to be loaded)
test: timing_test
	./timing_test

# Full test sequence
full_test: all timing_test
	@echo "Building and testing uncached memory module..."
	@echo "1. Loading module..."
	-sudo rmmod $(MODULE_NAME) 2>/dev/null || true
	sudo insmod $(MODULE_NAME).ko
	@echo "2. Running timing tests..."
	./timing_test
	@echo "3. Unloading module..."
	sudo rmmod $(MODULE_NAME)

# Test caching functionality specifically
cache_test: all timing_test
	@echo "Testing caching functionality..."
	@echo "1. Loading module..."
	-sudo rmmod $(MODULE_NAME) 2>/dev/null || true
	sudo insmod $(MODULE_NAME).ko
	@echo "2. Testing cache controls..."
	@echo "Initial status:"
	@cat /proc/$(MODULE_NAME)
	@echo "Enabling cache..."
	@echo "cache_on" | sudo tee /proc/$(MODULE_NAME)
	@cat /proc/$(MODULE_NAME)
	@echo "Allocating cached memory..."
	@echo "alloc" | sudo tee /proc/$(MODULE_NAME)
	@cat /proc/$(MODULE_NAME)
	@echo "Freeing memory..."
	@echo "free" | sudo tee /proc/$(MODULE_NAME)
	@echo "Disabling cache..."
	@echo "cache_off" | sudo tee /proc/$(MODULE_NAME)
	@cat /proc/$(MODULE_NAME)
	@echo "3. Running full timing test..."
	./timing_test
	@echo "4. Unloading module..."
	sudo rmmod $(MODULE_NAME)

# Show module info
info:
	modinfo $(MODULE_NAME).ko

# Show kernel messages related to this module
dmesg:
	dmesg | grep -i "uncached\|$(MODULE_NAME)\|dynamic" | tail -20

# Test dynamic cache module
dynamic_test_run: all dynamic_test
	@echo "Testing dynamic cache control module..."
	@echo "1. Loading dynamic cache module..."
	-sudo rmmod $(DYNAMIC_MODULE_NAME) 2>/dev/null || true
	sudo insmod $(DYNAMIC_MODULE_NAME).ko
	@echo "2. Running dynamic cache tests..."
	./dynamic_test
	@echo "3. Unloading dynamic cache module..."
	sudo rmmod $(DYNAMIC_MODULE_NAME)

# Test CMA cache module
cma_test_run: all cma_test
	@echo "Testing CMA cache control module..."
	@echo "1. Loading CMA cache module..."
	-sudo rmmod $(CMA_MODULE_NAME) 2>/dev/null || true
	sudo insmod $(CMA_MODULE_NAME).ko
	@echo "2. Setting permissions..."
	sudo chmod 666 /dev/$(CMA_MODULE_NAME) /sys/kernel/$(CMA_MODULE_NAME)/command
	@echo "3. Running CMA cache tests..."
	./cma_test basic
	@echo "4. Running performance tests..."
	./cma_test performance
	@echo "5. Unloading CMA cache module..."
	sudo rmmod $(CMA_MODULE_NAME)

# Test all modules
test_all: all timing_test dynamic_test cma_test
	@echo "Testing all modules..."
	@echo "=== Testing uncached_mem module ==="
	$(MAKE) full_test
	@echo ""
	@echo "=== Testing dynamic_cache module ==="
	$(MAKE) dynamic_test_run
	@echo ""
	@echo "=== Testing cma_cache module ==="
	$(MAKE) cma_test_run

.PHONY: all clean install load unload test full_test cache_test info dmesg load_dynamic unload_dynamic dynamic_test_run load_cma unload_cma cma_test_run test_all

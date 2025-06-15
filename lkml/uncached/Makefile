# Makefile for uncached memory kernel module

# Module name
MODULE_NAME = uncached_mem

# Source files
obj-m += $(MODULE_NAME).o

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
	rm -f timing_test

# Install module (requires root privileges)
install:
	$(MAKE) -C $(KERNEL_BUILD) M=$(PWD) modules_install
	depmod -a

# Load module
load:
	sudo insmod $(MODULE_NAME).ko

# Unload module
unload:
	sudo rmmod $(MODULE_NAME)

# Build user space test program
timing_test: timing_test.c
	gcc -o timing_test timing_test.c -lrt

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
	dmesg | grep -i "uncached\|$(MODULE_NAME)" | tail -20

.PHONY: all clean install load unload test full_test cache_test info dmesg

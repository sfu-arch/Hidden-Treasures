#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

/**
 * @file cuda_helpers.h
 * @brief Essential CUDA utility macros and functions for error checking and debugging
 * 
 * This file provides comprehensive CUDA error checking, device management,
 * and debugging utilities for the convolution lab project.
 */

// =============================================================================
// CUDA Error Checking Macros
// =============================================================================

/**
 * @brief Check CUDA runtime API calls for errors
 * @param call The CUDA runtime API call to check
 * 
 * Usage: CUDA_CHECK(cudaMalloc(&ptr, size));
 */
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "❌ CUDA Error at %s:%d\n", __FILE__, __LINE__); \
            fprintf(stderr, "   Function: %s\n", #call); \
            fprintf(stderr, "   Error: %s (%d)\n", cudaGetErrorString(err), err); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

/**
 * @brief Check for CUDA kernel launch errors
 * Should be called immediately after kernel launch
 * 
 * Usage: 
 * my_kernel<<<grid, block>>>(...);
 * CUDA_CHECK_KERNEL();
 */
#define CUDA_CHECK_KERNEL() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "❌ CUDA Kernel Launch Error at %s:%d\n", __FILE__, __LINE__); \
            fprintf(stderr, "   Error: %s (%d)\n", cudaGetErrorString(err), err); \
            exit(EXIT_FAILURE); \
        } \
        CUDA_CHECK(cudaDeviceSynchronize()); \
    } while(0)

/**
 * @brief Check for CUDA kernel execution errors (async version)
 * Use this for performance testing when you don't want to synchronize immediately
 */
#define CUDA_CHECK_KERNEL_ASYNC() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "❌ CUDA Kernel Launch Error at %s:%d\n", __FILE__, __LINE__); \
            fprintf(stderr, "   Error: %s (%d)\n", cudaGetErrorString(err), err); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// =============================================================================
// Device Information Functions
// =============================================================================

/**
 * @brief Print comprehensive CUDA device information
 * @param device_id Device ID to query (default: 0)
 */
inline void print_cuda_device_info(int device_id = 0) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    
    printf("🚀 CUDA Device Information:\n");
    printf("   Device Name: %s\n", prop.name);
    printf("   Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("   Global Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("   Shared Memory per Block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("   Registers per Block: %d\n", prop.regsPerBlock);
    printf("   Warp Size: %d\n", prop.warpSize);
    printf("   Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("   Max Block Dims: (%d, %d, %d)\n", 
           prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("   Max Grid Dims: (%d, %d, %d)\n",
           prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("   Memory Clock Rate: %.2f GHz\n", prop.memoryClockRate / 1e6);
    printf("   Memory Bus Width: %d bits\n", prop.memoryBusWidth);
    printf("   Peak Memory Bandwidth: %.2f GB/s\n",
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    printf("   Multiprocessors: %d\n", prop.multiProcessorCount);
    printf("   Max Threads per Multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("   L2 Cache Size: %.2f MB\n", prop.l2CacheSize / (1024.0 * 1024.0));
    printf("\n");
}

/**
 * @brief Check if CUDA is available and working
 * @return true if CUDA is available, false otherwise
 */
inline bool check_cuda_availability() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    
    if (err != cudaSuccess || device_count == 0) {
        printf("❌ CUDA is not available or no CUDA devices found\n");
        if (err != cudaSuccess) {
            printf("   Error: %s\n", cudaGetErrorString(err));
        }
        return false;
    }
    
    printf("✅ CUDA is available with %d device(s)\n", device_count);
    return true;
}

/**
 * @brief Initialize CUDA device and print information
 * @param device_id Device ID to use (default: 0)
 * @return true if initialization successful
 */
inline bool initialize_cuda_device(int device_id = 0) {
    if (!check_cuda_availability()) {
        return false;
    }
    
    CUDA_CHECK(cudaSetDevice(device_id));
    print_cuda_device_info(device_id);
    
    // Warm up the device
    CUDA_CHECK(cudaFree(0));
    
    return true;
}

// =============================================================================
// Memory Management Helpers
// =============================================================================

/**
 * @brief Allocate CUDA device memory with error checking
 * @param ptr Pointer to device memory pointer
 * @param size Size in bytes
 * @param description Optional description for debugging
 */
template<typename T>
inline void cuda_malloc_checked(T** ptr, size_t size, const char* description = nullptr) {
    CUDA_CHECK(cudaMalloc((void**)ptr, size));
    if (description) {
        printf("✅ Allocated %.2f MB for %s\n", size / (1024.0 * 1024.0), description);
    }
}

/**
 * @brief Free CUDA device memory with error checking
 * @param ptr Device memory pointer
 * @param description Optional description for debugging
 */
template<typename T>
inline void cuda_free_checked(T* ptr, const char* description = nullptr) {
    if (ptr != nullptr) {
        CUDA_CHECK(cudaFree(ptr));
        if (description) {
            printf("✅ Freed memory for %s\n", description);
        }
    }
}

/**
 * @brief Copy memory from host to device with error checking
 * @param dst Device pointer
 * @param src Host pointer
 * @param size Size in bytes
 * @param description Optional description for debugging
 */
template<typename T>
inline void cuda_memcpy_h2d_checked(T* dst, const T* src, size_t size, const char* description = nullptr) {
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
    if (description) {
        printf("✅ Copied %.2f MB from host to device (%s)\n", size / (1024.0 * 1024.0), description);
    }
}

/**
 * @brief Copy memory from device to host with error checking
 * @param dst Host pointer
 * @param src Device pointer
 * @param size Size in bytes
 * @param description Optional description for debugging
 */
template<typename T>
inline void cuda_memcpy_d2h_checked(T* dst, const T* src, size_t size, const char* description = nullptr) {
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
    if (description) {
        printf("✅ Copied %.2f MB from device to host (%s)\n", size / (1024.0 * 1024.0), description);
    }
}

// =============================================================================
// Kernel Launch Helpers
// =============================================================================

/**
 * @brief Calculate optimal 1D grid and block dimensions
 * @param total_threads Total number of threads needed
 * @param block_size Desired block size (will be adjusted if needed)
 * @return dim3 grid size
 */
inline dim3 calculate_grid_1d(int total_threads, int block_size = 256) {
    // Ensure block size is reasonable
    block_size = min(block_size, 1024);
    
    int grid_size = (total_threads + block_size - 1) / block_size;
    return dim3(grid_size);
}

/**
 * @brief Calculate optimal 2D grid and block dimensions
 * @param width Image width
 * @param height Image height
 * @param block_x Block width (default: 16)
 * @param block_y Block height (default: 16)
 * @return pair of (grid_dim, block_dim)
 */
inline std::pair<dim3, dim3> calculate_grid_2d(int width, int height, int block_x = 16, int block_y = 16) {
    // Ensure block dimensions are reasonable
    block_x = min(block_x, 32);
    block_y = min(block_y, 32);
    
    int grid_x = (width + block_x - 1) / block_x;
    int grid_y = (height + block_y - 1) / block_y;
    
    return std::make_pair(dim3(grid_x, grid_y), dim3(block_x, block_y));
}

// =============================================================================
// Performance Measurement Helpers
// =============================================================================

/**
 * @brief Simple CUDA event-based timer
 */
class CudaTimer {
private:
    cudaEvent_t start_event, stop_event;
    bool timing_active;

public:
    CudaTimer() : timing_active(false) {
        CUDA_CHECK(cudaEventCreate(&start_event));
        CUDA_CHECK(cudaEventCreate(&stop_event));
    }
    
    ~CudaTimer() {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }
    
    void start() {
        CUDA_CHECK(cudaEventRecord(start_event));
        timing_active = true;
    }
    
    float stop() {
        if (!timing_active) {
            fprintf(stderr, "Warning: Timer was not started\n");
            return 0.0f;
        }
        
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        
        float elapsed_ms;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event));
        
        timing_active = false;
        return elapsed_ms;
    }
};

// =============================================================================
// Debug and Profiling Helpers
// =============================================================================

#ifdef DEBUG
    #define CUDA_DEBUG_PRINT(fmt, ...) printf("🐛 DEBUG: " fmt "\n", ##__VA_ARGS__)
#else
    #define CUDA_DEBUG_PRINT(fmt, ...)
#endif

/**
 * @brief Print memory usage information
 */
inline void print_memory_usage() {
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    
    size_t used_mem = total_mem - free_mem;
    printf("📊 GPU Memory Usage:\n");
    printf("   Total: %.2f GB\n", total_mem / (1024.0 * 1024.0 * 1024.0));
    printf("   Used:  %.2f GB (%.1f%%)\n", 
           used_mem / (1024.0 * 1024.0 * 1024.0),
           100.0 * used_mem / total_mem);
    printf("   Free:  %.2f GB (%.1f%%)\n",
           free_mem / (1024.0 * 1024.0 * 1024.0),
           100.0 * free_mem / total_mem);
}

/**
 * @brief Check if we have sufficient GPU memory for allocation
 * @param required_bytes Required memory in bytes
 * @return true if sufficient memory available
 */
inline bool check_memory_availability(size_t required_bytes) {
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    
    if (required_bytes > free_mem) {
        printf("❌ Insufficient GPU memory:\n");
        printf("   Required: %.2f GB\n", required_bytes / (1024.0 * 1024.0 * 1024.0));
        printf("   Available: %.2f GB\n", free_mem / (1024.0 * 1024.0 * 1024.0));
        return false;
    }
    
    return true;
}

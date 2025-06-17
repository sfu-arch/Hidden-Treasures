#include "conv_naive.h"
#include "../common/cuda_helpers.h"
#include "../common/timer.h"
#include <iostream>

namespace gpu {

/**
 * @brief Naive CUDA kernel for 2D convolution
 * 
 * This is the most straightforward implementation with one thread per output pixel.
 * Each thread independently computes one output value by applying the full kernel.
 * 
 * Memory Access Pattern:
 * - Each thread accesses global memory for input and kernel data
 * - No memory coalescing optimization
 * - High memory bandwidth requirements due to redundant accesses
 * 
 * Educational Focus:
 * - Basic CUDA programming concepts
 * - Thread indexing and boundary checking
 * - Global memory access patterns
 * - Performance baseline for comparison
 */
__global__ void convolution2D_naive_kernel(const float* __restrict__ input, 
                                          float* __restrict__ output,
                                          int width, int height,
                                          const float* __restrict__ kernel,
                                          int kernel_size) {
    // Calculate thread's output pixel coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Boundary check - ensure thread is within image bounds
    if (x >= width || y >= height) {
        return;
    }
    
    int half_kernel = kernel_size / 2;
    float sum = 0.0f;
    
    // Apply convolution kernel
    // Each thread independently reads from global memory
    for (int ky = -half_kernel; ky <= half_kernel; ky++) {
        for (int kx = -half_kernel; kx <= half_kernel; kx++) {
            // Calculate input coordinates with kernel offset
            int input_y = y + ky;
            int input_x = x + kx;
            
            // Handle boundary conditions with zero padding
            if (input_y >= 0 && input_y < height && 
                input_x >= 0 && input_x < width) {
                
                // Global memory access - potential for optimization
                int input_idx = input_y * width + input_x;
                int kernel_idx = (ky + half_kernel) * kernel_size + (kx + half_kernel);
                
                // Accumulate weighted sum
                sum += input[input_idx] * kernel[kernel_idx];
            }
        }
    }
    
    // Write result to global memory
    output[y * width + x] = sum;
}

void convolution2D_naive(const float* input, float* output, int width, int height,
                        const float* kernel, int kernel_size) {
    
    // Device memory pointers
    float *d_input, *d_output, *d_kernel;
    
    // Calculate memory sizes
    size_t input_size = width * height * sizeof(float);
    size_t output_size = width * height * sizeof(float);
    size_t kernel_size_bytes = kernel_size * kernel_size * sizeof(float);
    
    Timer timer;
    timer.start();
    
    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_input, input_size));
    CUDA_CHECK(cudaMalloc(&d_output, output_size));
    CUDA_CHECK(cudaMalloc(&d_kernel, kernel_size_bytes));
    
    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_input, input, input_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kernel, kernel, kernel_size_bytes, cudaMemcpyHostToDevice));
    
    // Configure kernel launch parameters
    // Using 16x16 thread blocks for good occupancy on most GPUs
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    
    // Print launch configuration for educational purposes
    std::cout << "Naive GPU Launch Config:" << std::endl;
    std::cout << "  Grid: (" << gridSize.x << ", " << gridSize.y << ")" << std::endl;
    std::cout << "  Block: (" << blockSize.x << ", " << blockSize.y << ")" << std::endl;
    std::cout << "  Total threads: " << gridSize.x * gridSize.y * blockSize.x * blockSize.y << std::endl;
    
    // Launch kernel
    convolution2D_naive_kernel<<<gridSize, blockSize>>>(
        d_input, d_output, width, height, d_kernel, kernel_size);
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    
    // Wait for kernel completion
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(output, d_output, output_size, cudaMemcpyDeviceToHost));
    
    timer.stop();
    
    // Clean up device memory
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_kernel));
    
    // Print performance metrics
    double elapsed_ms = timer.last_measurement();
    std::cout << "Naive GPU Convolution took: " << elapsed_ms << " ms" << std::endl;
    
    // Calculate and display performance metrics
    printGPUPerformanceMetrics(width, height, kernel_size, elapsed_ms, "Naive");
}

void printGPUPerformanceMetrics(int width, int height, int kernel_size, 
                               double time_ms, const char* kernel_name) {
    // Calculate theoretical metrics
    long long total_operations = (long long)width * height * kernel_size * kernel_size * 2;
    double gflops = (total_operations / (time_ms / 1000.0)) / 1e9;
    
    // Estimate memory bandwidth (conservative)
    long long input_reads = (long long)width * height * kernel_size * kernel_size;
    long long kernel_reads = (long long)width * height * kernel_size * kernel_size;
    long long output_writes = (long long)width * height;
    long long total_bytes = (input_reads + kernel_reads + output_writes) * sizeof(float);
    double bandwidth_gb_s = (total_bytes / (time_ms / 1000.0)) / (1024.0 * 1024.0 * 1024.0);
    
    std::cout << kernel_name << " GPU Performance Metrics:" << std::endl;
    std::cout << "  Execution Time: " << time_ms << " ms" << std::endl;
    std::cout << "  Throughput: " << gflops << " GFLOPS" << std::endl;
    std::cout << "  Memory Bandwidth: " << bandwidth_gb_s << " GB/s" << std::endl;
    std::cout << "  Pixels/second: " << (width * height) / (time_ms / 1000.0) << std::endl;
    
    // Calculate operational intensity for roofline analysis
    double operational_intensity = (double)total_operations / total_bytes;
    std::cout << "  Operational Intensity: " << operational_intensity << " FLOP/Byte" << std::endl;
}

double calculateSpeedup(double cpu_time_ms, double gpu_time_ms) {
    return cpu_time_ms / gpu_time_ms;
}

void analyzeMemoryAccess(int width, int height, int kernel_size) {
    std::cout << "\nMemory Access Analysis (Naive Implementation):" << std::endl;
    
    long long input_accesses_per_thread = kernel_size * kernel_size;
    long long total_threads = (long long)width * height;
    long long total_input_accesses = input_accesses_per_thread * total_threads;
    
    std::cout << "  Input accesses per thread: " << input_accesses_per_thread << std::endl;
    std::cout << "  Total input accesses: " << total_input_accesses << std::endl;
    std::cout << "  Memory reuse factor: " << (double)total_input_accesses / (width * height) << "x" << std::endl;
    
    // Analyze coalescing potential
    std::cout << "  Coalescing: Poor - threads access scattered memory locations" << std::endl;
    std::cout << "  Cache efficiency: Low - no spatial/temporal locality exploitation" << std::endl;
    std::cout << "  Optimization potential: High - significant room for improvement" << std::endl;
}

} // namespace gpu

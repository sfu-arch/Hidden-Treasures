#ifndef CONV_NAIVE_H
#define CONV_NAIVE_H

namespace gpu {

/**
 * @brief Naive CUDA implementation of 2D convolution
 * 
 * This implementation serves as the baseline for performance comparison.
 * Each thread computes one output pixel independently, accessing global
 * memory directly without any optimization.
 * 
 * Characteristics:
 * - One thread per output pixel
 * - Direct global memory access
 * - No memory coalescing optimization
 * - No shared memory usage
 * - Simple and straightforward implementation
 * 
 * Educational Purpose:
 * - Demonstrates basic CUDA programming concepts
 * - Provides performance baseline for comparison
 * - Shows impact of memory access patterns
 * 
 * @param input Input image data in device memory
 * @param output Output image data in device memory (pre-allocated)
 * @param width Image width in pixels
 * @param height Image height in pixels
 * @param kernel Convolution kernel in device memory
 * @param kernel_size Kernel size (assumed square, odd dimension)
 */
void convolution2D_naive(const float* input, float* output, int width, int height,
                        const float* kernel, int kernel_size);

/**
 * @brief Print comprehensive GPU performance metrics
 * 
 * Calculates and displays various performance indicators including:
 * - Execution time
 * - GFLOPS (throughput)
 * - Memory bandwidth utilization
 * - Operational intensity for roofline analysis
 * 
 * @param width Image width
 * @param height Image height
 * @param kernel_size Kernel size
 * @param time_ms Execution time in milliseconds
 * @param kernel_name Name of the kernel implementation
 */
void printGPUPerformanceMetrics(int width, int height, int kernel_size, 
                               double time_ms, const char* kernel_name);

/**
 * @brief Calculate speedup compared to CPU implementation
 * @param cpu_time_ms CPU execution time in milliseconds
 * @param gpu_time_ms GPU execution time in milliseconds
 * @return Speedup factor (cpu_time / gpu_time)
 */
double calculateSpeedup(double cpu_time_ms, double gpu_time_ms);

/**
 * @brief Analyze memory access patterns for educational purposes
 * 
 * Provides detailed analysis of memory access characteristics:
 * - Number of memory accesses per thread
 * - Memory reuse patterns
 * - Coalescing efficiency
 * - Cache utilization
 * 
 * @param width Image width
 * @param height Image height
 * @param kernel_size Kernel size
 */
void analyzeMemoryAccess(int width, int height, int kernel_size);

} // namespace gpu

#endif // CONV_NAIVE_H

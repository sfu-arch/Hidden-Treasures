#include "conv_cpu.h"
#include "../common/timer.h"
#include <iostream>
#include <cstring>
#include <algorithm>
#include <omp.h>

namespace cpu {

void convolution2D(const float* input, float* output, int width, int height,
                   const float* kernel, int kernel_size, ConvolutionMode mode) {
    Timer timer;
    timer.start();
    
    int half_kernel = kernel_size / 2;
    
    // Initialize output to zero
    std::fill(output, output + width * height, 0.0f);
    
    switch (mode) {
        case ConvolutionMode::SEQUENTIAL:
            convolution2D_sequential(input, output, width, height, kernel, kernel_size);
            break;
        case ConvolutionMode::OPENMP:
            convolution2D_openmp(input, output, width, height, kernel, kernel_size);
            break;
        case ConvolutionMode::OPTIMIZED:
            convolution2D_optimized(input, output, width, height, kernel, kernel_size);
            break;
    }
    
    timer.stop();
    std::cout << "CPU Convolution (" << getModeString(mode) << ") took: " 
              << timer.getElapsedMs() << " ms" << std::endl;
}

void convolution2D_sequential(const float* input, float* output, int width, int height,
                             const float* kernel, int kernel_size) {
    int half_kernel = kernel_size / 2;
    
    // Iterate through each output pixel
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum = 0.0f;
            
            // Apply kernel
            for (int ky = -half_kernel; ky <= half_kernel; ky++) {
                for (int kx = -half_kernel; kx <= half_kernel; kx++) {
                    int input_y = y + ky;
                    int input_x = x + kx;
                    
                    // Handle boundary conditions (zero padding)
                    if (input_y >= 0 && input_y < height && 
                        input_x >= 0 && input_x < width) {
                        int input_idx = input_y * width + input_x;
                        int kernel_idx = (ky + half_kernel) * kernel_size + (kx + half_kernel);
                        sum += input[input_idx] * kernel[kernel_idx];
                    }
                }
            }
            
            output[y * width + x] = sum;
        }
    }
}

void convolution2D_openmp(const float* input, float* output, int width, int height,
                         const float* kernel, int kernel_size) {
    int half_kernel = kernel_size / 2;
    
    // Parallelize outer loop with OpenMP
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum = 0.0f;
            
            // Apply kernel
            for (int ky = -half_kernel; ky <= half_kernel; ky++) {
                for (int kx = -half_kernel; kx <= half_kernel; kx++) {
                    int input_y = y + ky;
                    int input_x = x + kx;
                    
                    // Handle boundary conditions (zero padding)
                    if (input_y >= 0 && input_y < height && 
                        input_x >= 0 && input_x < width) {
                        int input_idx = input_y * width + input_x;
                        int kernel_idx = (ky + half_kernel) * kernel_size + (kx + half_kernel);
                        sum += input[input_idx] * kernel[kernel_idx];
                    }
                }
            }
            
            output[y * width + x] = sum;
        }
    }
}

void convolution2D_optimized(const float* input, float* output, int width, int height,
                            const float* kernel, int kernel_size) {
    int half_kernel = kernel_size / 2;
    
    // Cache-friendly implementation with loop reordering and blocking
    const int BLOCK_SIZE = 64; // Tune for cache size
    
    #pragma omp parallel for
    for (int by = 0; by < height; by += BLOCK_SIZE) {
        for (int bx = 0; bx < width; bx += BLOCK_SIZE) {
            int end_y = std::min(by + BLOCK_SIZE, height);
            int end_x = std::min(bx + BLOCK_SIZE, width);
            
            for (int y = by; y < end_y; y++) {
                for (int x = bx; x < end_x; x++) {
                    float sum = 0.0f;
                    
                    // Apply kernel with optimized memory access
                    for (int ky = -half_kernel; ky <= half_kernel; ky++) {
                        int input_y = y + ky;
                        if (input_y >= 0 && input_y < height) {
                            for (int kx = -half_kernel; kx <= half_kernel; kx++) {
                                int input_x = x + kx;
                                if (input_x >= 0 && input_x < width) {
                                    int input_idx = input_y * width + input_x;
                                    int kernel_idx = (ky + half_kernel) * kernel_size + (kx + half_kernel);
                                    sum += input[input_idx] * kernel[kernel_idx];
                                }
                            }
                        }
                    }
                    
                    output[y * width + x] = sum;
                }
            }
        }
    }
}

float calculateGFLOPS(int width, int height, int kernel_size, double time_ms) {
    // Each output pixel requires kernel_size^2 multiply-adds
    long long operations = (long long)width * height * kernel_size * kernel_size * 2; // 2 ops per MAC
    double time_s = time_ms / 1000.0;
    return (operations / time_s) / 1e9;
}

double calculateMemoryBandwidth(int width, int height, int kernel_size, double time_ms) {
    // Input: width * height * sizeof(float)
    // Kernel: kernel_size * kernel_size * sizeof(float) (accessed multiple times)
    // Output: width * height * sizeof(float)
    // Approximate memory access (conservative estimate)
    long long input_accesses = (long long)width * height * kernel_size * kernel_size;
    long long kernel_accesses = (long long)width * height * kernel_size * kernel_size;
    long long output_accesses = (long long)width * height;
    
    long long total_bytes = (input_accesses + kernel_accesses + output_accesses) * sizeof(float);
    double time_s = time_ms / 1000.0;
    return (total_bytes / time_s) / (1024.0 * 1024.0 * 1024.0); // GB/s
}

const char* getModeString(ConvolutionMode mode) {
    switch (mode) {
        case ConvolutionMode::SEQUENTIAL: return "Sequential";
        case ConvolutionMode::OPENMP: return "OpenMP";
        case ConvolutionMode::OPTIMIZED: return "Optimized";
        default: return "Unknown";
    }
}

void printPerformanceMetrics(int width, int height, int kernel_size, double time_ms) {
    float gflops = calculateGFLOPS(width, height, kernel_size, time_ms);
    double bandwidth = calculateMemoryBandwidth(width, height, kernel_size, time_ms);
    
    std::cout << "Performance Metrics:" << std::endl;
    std::cout << "  Time: " << time_ms << " ms" << std::endl;
    std::cout << "  GFLOPS: " << gflops << std::endl;
    std::cout << "  Memory Bandwidth: " << bandwidth << " GB/s" << std::endl;
    std::cout << "  Pixels/second: " << (width * height) / (time_ms / 1000.0) << std::endl;
}

} // namespace cpu

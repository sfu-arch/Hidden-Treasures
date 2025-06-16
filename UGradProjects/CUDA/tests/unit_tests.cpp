#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <random>

// Include implementations to test
#include "../src/cpu/conv_cpu.h"
#include "../src/gpu/conv_naive.h"
#include "../src/common/image_io.h"
#include "../src/common/cuda_helpers.h"

/**
 * @brief Unit tests for CUDA Convolution Lab
 * 
 * These tests verify the correctness of all convolution implementations
 * and provide a foundation for performance regression testing.
 * 
 * Test Categories:
 * - Basic functionality tests
 * - Edge case handling
 * - Cross-platform consistency
 * - Performance regression detection
 */

class ConvolutionTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize test data
        width = 32;
        height = 32;
        kernel_size = 5;
        
        // Allocate test arrays
        input.resize(width * height);
        output_cpu.resize(width * height);
        output_gpu.resize(width * height);
        kernel.resize(kernel_size * kernel_size);
        
        // Generate deterministic test data
        generateTestData();
    }
    
    void TearDown() override {
        // Cleanup if needed
    }
    
    void generateTestData() {
        // Generate reproducible test pattern
        std::mt19937 gen(12345); // Fixed seed for reproducibility
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);
        
        // Fill input with random values
        for (size_t i = 0; i < input.size(); i++) {
            input[i] = dis(gen);
        }
        
        // Generate Gaussian kernel
        image_io::generateKernel(kernel.data(), kernel_size, image_io::KernelType::GAUSSIAN);
    }
    
    bool compareResults(const float* result1, const float* result2, 
                       int size, float tolerance = 1e-5f) {
        for (int i = 0; i < size; i++) {
            if (std::abs(result1[i] - result2[i]) > tolerance) {
                return false;
            }
        }
        return true;
    }
    
    // Test data
    int width, height, kernel_size;
    std::vector<float> input, output_cpu, output_gpu, kernel;
};

// Basic functionality tests
TEST_F(ConvolutionTest, CPUSequentialBasic) {
    cpu::convolution2D_sequential(input.data(), output_cpu.data(), 
                                 width, height, kernel.data(), kernel_size);
    
    // Check that output is not all zeros (assuming non-zero input and kernel)
    bool has_nonzero = false;
    for (float val : output_cpu) {
        if (val != 0.0f) {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero) << "CPU sequential output should not be all zeros";
}

TEST_F(ConvolutionTest, CPUOpenMPBasic) {
    cpu::convolution2D_openmp(input.data(), output_cpu.data(), 
                             width, height, kernel.data(), kernel_size);
    
    // Check that output is not all zeros
    bool has_nonzero = false;
    for (float val : output_cpu) {
        if (val != 0.0f) {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero) << "CPU OpenMP output should not be all zeros";
}

TEST_F(ConvolutionTest, CPUOptimizedBasic) {
    cpu::convolution2D_optimized(input.data(), output_cpu.data(), 
                                width, height, kernel.data(), kernel_size);
    
    // Check that output is not all zeros
    bool has_nonzero = false;
    for (float val : output_cpu) {
        if (val != 0.0f) {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero) << "CPU optimized output should not be all zeros";
}

TEST_F(ConvolutionTest, GPUNaiveBasic) {
    // Skip if CUDA not available
    if (!cuda::checkCudaCapability()) {
        GTEST_SKIP() << "CUDA not available";
    }
    
    gpu::convolution2D_naive(input.data(), output_gpu.data(), 
                            width, height, kernel.data(), kernel_size);
    
    // Check that output is not all zeros
    bool has_nonzero = false;
    for (float val : output_gpu) {
        if (val != 0.0f) {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero) << "GPU naive output should not be all zeros";
}

// Cross-implementation consistency tests
TEST_F(ConvolutionTest, CPUImplementationsConsistent) {
    std::vector<float> output_seq(width * height);
    std::vector<float> output_omp(width * height);
    std::vector<float> output_opt(width * height);
    
    // Run all CPU implementations
    cpu::convolution2D_sequential(input.data(), output_seq.data(), 
                                 width, height, kernel.data(), kernel_size);
    cpu::convolution2D_openmp(input.data(), output_omp.data(), 
                             width, height, kernel.data(), kernel_size);
    cpu::convolution2D_optimized(input.data(), output_opt.data(), 
                                width, height, kernel.data(), kernel_size);
    
    // Compare results
    EXPECT_TRUE(compareResults(output_seq.data(), output_omp.data(), width * height))
        << "Sequential and OpenMP results should match";
    EXPECT_TRUE(compareResults(output_seq.data(), output_opt.data(), width * height))
        << "Sequential and optimized results should match";
}

TEST_F(ConvolutionTest, CPUGPUConsistent) {
    // Skip if CUDA not available
    if (!cuda::checkCudaCapability()) {
        GTEST_SKIP() << "CUDA not available";
    }
    
    // Run CPU reference implementation
    cpu::convolution2D_sequential(input.data(), output_cpu.data(), 
                                 width, height, kernel.data(), kernel_size);
    
    // Run GPU implementation
    gpu::convolution2D_naive(input.data(), output_gpu.data(), 
                            width, height, kernel.data(), kernel_size);
    
    // Compare results with slightly relaxed tolerance for GPU
    EXPECT_TRUE(compareResults(output_cpu.data(), output_gpu.data(), 
                              width * height, 1e-4f))
        << "CPU and GPU results should match within tolerance";
}

// Edge case tests
TEST_F(ConvolutionTest, SmallKernel) {
    // Test with 3x3 kernel
    int small_kernel_size = 3;
    std::vector<float> small_kernel(small_kernel_size * small_kernel_size);
    image_io::generateKernel(small_kernel.data(), small_kernel_size, 
                           image_io::KernelType::IDENTITY);
    
    cpu::convolution2D_sequential(input.data(), output_cpu.data(), 
                                 width, height, small_kernel.data(), small_kernel_size);
    
    // Identity kernel should produce output very close to input
    EXPECT_TRUE(compareResults(input.data(), output_cpu.data(), 
                              width * height, 0.1f))
        << "Identity kernel should preserve input (approximately)";
}

TEST_F(ConvolutionTest, LargeKernel) {
    // Test with 15x15 kernel
    int large_kernel_size = 15;
    std::vector<float> large_kernel(large_kernel_size * large_kernel_size);
    std::vector<float> large_output(width * height);
    
    image_io::generateKernel(large_kernel.data(), large_kernel_size, 
                           image_io::KernelType::GAUSSIAN);
    
    cpu::convolution2D_sequential(input.data(), large_output.data(), 
                                 width, height, large_kernel.data(), large_kernel_size);
    
    // Should complete without errors
    SUCCEED() << "Large kernel convolution completed successfully";
}

TEST_F(ConvolutionTest, SmallImage) {
    // Test with very small image
    int small_width = 8;
    int small_height = 8;
    std::vector<float> small_input(small_width * small_height, 1.0f);
    std::vector<float> small_output(small_width * small_height);
    
    cpu::convolution2D_sequential(small_input.data(), small_output.data(), 
                                 small_width, small_height, kernel.data(), kernel_size);
    
    // Should complete without errors
    SUCCEED() << "Small image convolution completed successfully";
}

// Performance regression tests
TEST_F(ConvolutionTest, PerformanceRegression) {
    // Test with larger image for performance measurement
    int perf_width = 512;
    int perf_height = 512;
    std::vector<float> perf_input(perf_width * perf_height, 1.0f);
    std::vector<float> perf_output(perf_width * perf_height);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    cpu::convolution2D_sequential(perf_input.data(), perf_output.data(), 
                                 perf_width, perf_height, kernel.data(), kernel_size);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Performance threshold (adjust based on expected performance)
    EXPECT_LT(duration.count(), 5000) << "CPU convolution should complete within 5 seconds";
}

// Utility function tests
TEST_F(ConvolutionTest, PerformanceMetricsCalculation) {
    double gflops = cpu::calculateGFLOPS(width, height, kernel_size, 100.0);
    EXPECT_GT(gflops, 0.0) << "GFLOPS calculation should be positive";
    
    double bandwidth = cpu::calculateMemoryBandwidth(width, height, kernel_size, 100.0);
    EXPECT_GT(bandwidth, 0.0) << "Memory bandwidth calculation should be positive";
}

TEST_F(ConvolutionTest, ImageIOFunctions) {
    // Test image comparison function
    std::vector<float> identical(100, 1.0f);
    std::vector<float> different(100, 2.0f);
    
    EXPECT_TRUE(image_io::compareImages(identical.data(), identical.data(), 10, 10, 1e-6f))
        << "Identical images should compare as equal";
    EXPECT_FALSE(image_io::compareImages(identical.data(), different.data(), 10, 10, 1e-6f))
        << "Different images should compare as unequal";
}

// Test fixtures for different kernel types
class KernelTypeTest : public ConvolutionTest,
                      public ::testing::WithParamInterface<image_io::KernelType> {
};

TEST_P(KernelTypeTest, DifferentKernelTypes) {
    image_io::KernelType kernel_type = GetParam();
    
    // Generate kernel of specified type
    image_io::generateKernel(kernel.data(), kernel_size, kernel_type);
    
    // Run convolution
    cpu::convolution2D_sequential(input.data(), output_cpu.data(), 
                                 width, height, kernel.data(), kernel_size);
    
    // Should complete without errors
    SUCCEED() << "Convolution with kernel type " << static_cast<int>(kernel_type) 
              << " completed successfully";
}

INSTANTIATE_TEST_SUITE_P(
    AllKernelTypes,
    KernelTypeTest,
    ::testing::Values(
        image_io::KernelType::IDENTITY,
        image_io::KernelType::GAUSSIAN,
        image_io::KernelType::SHARPEN,
        image_io::KernelType::EDGE_DETECT
    )
);

// Main function for running tests
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    
    // Print system information
    std::cout << "Running CUDA Convolution Lab Unit Tests" << std::endl;
    std::cout << "=======================================" << std::endl;
    
    // Check CUDA availability
    if (cuda::checkCudaCapability()) {
        std::cout << "CUDA is available - GPU tests will be included" << std::endl;
        cuda::printDeviceInfo();
    } else {
        std::cout << "CUDA not available - GPU tests will be skipped" << std::endl;
    }
    
    return RUN_ALL_TESTS();
}

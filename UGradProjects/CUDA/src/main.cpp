#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <cmath>

// Common utilities
#include "common/utils.h"
#include "common/timer.h"
#include "common/image_io.h"
#include "common/cuda_helpers.h"

// CPU implementations
#include "cpu/conv_cpu.h"

// GPU implementations
#include "gpu/conv_naive.h"
// #include "gpu/conv_coalesced.h"    // To be implemented
// #include "gpu/conv_shared.h"       // To be implemented
// #include "gpu/conv_constant.h"     // To be implemented
// #include "gpu/conv_texture.h"      // To be implemented

/**
 * @brief CUDA Convolution Lab - Main Driver Program
 * 
 * This program demonstrates progressive optimization of 2D convolution
 * from naive CPU implementation to highly optimized GPU kernels.
 * 
 * Educational Objectives:
 * - Compare CPU vs GPU performance
 * - Understand memory hierarchy optimization
 * - Measure impact of different optimization strategies
 * - Analyze performance using roofline model
 */

void printUsage(const char* program_name) {
    std::cout << "CUDA Convolution Lab - Performance Comparison Tool\n" << std::endl;
    std::cout << "Usage: " << program_name << " [options]" << std::endl;
    std::cout << "\nOptions:" << std::endl;
    std::cout << "  -w, --width <size>     Image width (default: 1024)" << std::endl;
    std::cout << "  -h, --height <size>    Image height (default: 1024)" << std::endl;
    std::cout << "  -k, --kernel <size>    Kernel size (default: 5)" << std::endl;
    std::cout << "  -i, --input <file>     Input image file (optional)" << std::endl;
    std::cout << "  -o, --output <dir>     Output directory (default: ./output)" << std::endl;
    std::cout << "  -t, --test <name>      Test kernel name (default: all)" << std::endl;
    std::cout << "  -v, --verbose          Enable verbose output" << std::endl;
    std::cout << "  -c, --cpu-only         Run CPU tests only" << std::endl;
    std::cout << "  -g, --gpu-only         Run GPU tests only" << std::endl;
    std::cout << "  --help                 Show this help message" << std::endl;
    std::cout << "\nSupported kernels:" << std::endl;
    std::cout << "  cpu-seq, cpu-omp, cpu-opt, gpu-naive" << std::endl;
    std::cout << "  (more GPU kernels available in full implementation)" << std::endl;
    std::cout << "\nExamples:" << std::endl;
    std::cout << "  " << program_name << " --width 2048 --height 2048 --kernel 7" << std::endl;
    std::cout << "  " << program_name << " --input data/input/lena.png --test gpu-naive" << std::endl;
    std::cout << "  " << program_name << " --cpu-only --verbose" << std::endl;
}

void runCPUBenchmarks(const std::vector<float>& input, std::vector<float>& output,
                     int width, int height, const std::vector<float>& kernel,
                     int kernel_size, bool verbose) {
    
    std::cout << "\n=== CPU Benchmarks ===" << std::endl;
    
    if (verbose) {
        std::cout << "Running CPU convolution implementations..." << std::endl;
        std::cout << "Image size: " << width << "x" << height << std::endl;
        std::cout << "Kernel size: " << kernel_size << "x" << kernel_size << std::endl;
    }
    
    // Sequential implementation
    {
        Timer timer;
        timer.start();
        cpu::convolution2D(input.data(), output.data(), width, height, 
                          kernel.data(), kernel_size, cpu::ConvolutionMode::SEQUENTIAL);
        timer.stop();
        
        if (verbose) {
            cpu::printPerformanceMetrics(width, height, kernel_size, timer.getElapsedMs());
        }
    }
    
    // OpenMP implementation
    {
        std::vector<float> output_omp(width * height);
        Timer timer;
        timer.start();
        cpu::convolution2D(input.data(), output_omp.data(), width, height,
                          kernel.data(), kernel_size, cpu::ConvolutionMode::OPENMP);
        timer.stop();
        
        if (verbose) {
            cpu::printPerformanceMetrics(width, height, kernel_size, timer.getElapsedMs());
        }
    }
    
    // Optimized implementation
    {
        std::vector<float> output_opt(width * height);
        Timer timer;
        timer.start();
        cpu::convolution2D(input.data(), output_opt.data(), width, height,
                          kernel.data(), kernel_size, cpu::ConvolutionMode::OPTIMIZED);
        timer.stop();
        
        if (verbose) {
            cpu::printPerformanceMetrics(width, height, kernel_size, timer.getElapsedMs());
        }
    }
}

void runGPUBenchmarks(const std::vector<float>& input, std::vector<float>& output,
                     int width, int height, const std::vector<float>& kernel,
                     int kernel_size, bool verbose) {
    
    std::cout << "\n=== GPU Benchmarks ===" << std::endl;
    
    // Check CUDA availability
    if (!cuda::checkCudaCapability()) {
        std::cerr << "CUDA not available or insufficient capability" << std::endl;
        return;
    }
    
    if (verbose) {
        cuda::printDeviceInfo();
        std::cout << "Running GPU convolution implementations..." << std::endl;
    }
    
    // Naive GPU implementation
    {
        std::vector<float> output_gpu(width * height);
        Timer timer;
        timer.start();
        gpu::convolution2D_naive(input.data(), output_gpu.data(), width, height,
                                kernel.data(), kernel_size);
        timer.stop();
        
        // Verify correctness against CPU result
        bool correct = image_io::compareImages(output.data(), output_gpu.data(), 
                                             width, height, 1e-4f);
        std::cout << "GPU Naive result: " << (correct ? "CORRECT" : "INCORRECT") << std::endl;
        
        if (verbose) {
            gpu::analyzeMemoryAccess(width, height, kernel_size);
        }
    }
    
    // TODO: Add other GPU implementations
    // - Coalesced memory access
    // - Shared memory tiling
    // - Constant memory
    // - Texture memory
    // - Multiple optimizations combined
}

void saveResults(const std::vector<float>& input, const std::vector<float>& output,
                int width, int height, const std::string& output_dir) {
    
    std::cout << "\nSaving results to " << output_dir << std::endl;
    
    // Save input image (if not from file)
    std::string input_path = output_dir + "/input.png";
    if (image_io::saveImage(input.data(), width, height, input_path)) {
        std::cout << "Saved input image: " << input_path << std::endl;
    }
    
    // Save output image
    std::string output_path = output_dir + "/output.png";
    if (image_io::saveImage(output.data(), width, height, output_path)) {
        std::cout << "Saved output image: " << output_path << std::endl;
    }
}

int main(int argc, char* argv[]) {
    // Default parameters
    int width = 1024;
    int height = 1024;
    int kernel_size = 5;
    std::string input_file = "";
    std::string output_dir = "./output";
    std::string test_kernel = "all";
    bool verbose = false;
    bool cpu_only = false;
    bool gpu_only = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--help") {
            printUsage(argv[0]);
            return 0;
        }
        else if ((arg == "-w" || arg == "--width") && i + 1 < argc) {
            width = std::atoi(argv[++i]);
        }
        else if ((arg == "-h" || arg == "--height") && i + 1 < argc) {
            height = std::atoi(argv[++i]);
        }
        else if ((arg == "-k" || arg == "--kernel") && i + 1 < argc) {
            kernel_size = std::atoi(argv[++i]);
        }
        else if ((arg == "-i" || arg == "--input") && i + 1 < argc) {
            input_file = argv[++i];
        }
        else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            output_dir = argv[++i];
        }
        else if ((arg == "-t" || arg == "--test") && i + 1 < argc) {
            test_kernel = argv[++i];
        }
        else if (arg == "-v" || arg == "--verbose") {
            verbose = true;
        }
        else if (arg == "-c" || arg == "--cpu-only") {
            cpu_only = true;
        }
        else if (arg == "-g" || arg == "--gpu-only") {
            gpu_only = true;
        }
        else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            printUsage(argv[0]);
            return 1;
        }
    }
    
    // Validate parameters
    if (kernel_size % 2 == 0) {
        std::cerr << "Kernel size must be odd" << std::endl;
        return 1;
    }
    
    if (width <= 0 || height <= 0 || kernel_size <= 0) {
        std::cerr << "Invalid dimensions" << std::endl;
        return 1;
    }
    
    std::cout << "CUDA Convolution Lab" << std::endl;
    std::cout << "===================" << std::endl;
    std::cout << "Image size: " << width << "x" << height << std::endl;
    std::cout << "Kernel size: " << kernel_size << "x" << kernel_size << std::endl;
    
    // Prepare input data
    std::vector<float> input(width * height);
    std::vector<float> output(width * height);
    
    if (!input_file.empty()) {
        std::cout << "Loading input image: " << input_file << std::endl;
        if (!image_io::loadImage(input_file, input.data(), width, height)) {
            std::cerr << "Failed to load input image" << std::endl;
            return 1;
        }
    } else {
        std::cout << "Generating test pattern..." << std::endl;
        image_io::generateTestPattern(input.data(), width, height, 
                                    image_io::TestPattern::CHECKERBOARD);
    }
    
    // Prepare convolution kernel
    std::vector<float> kernel(kernel_size * kernel_size);
    image_io::generateKernel(kernel.data(), kernel_size, image_io::KernelType::GAUSSIAN);
    
    if (verbose) {
        std::cout << "\nKernel (center 3x3):" << std::endl;
        image_io::printKernel(kernel.data(), kernel_size, 3);
    }
    
    // Create output directory
    utils::createDirectory(output_dir);
    
    // Run benchmarks
    if (!gpu_only) {
        runCPUBenchmarks(input, output, width, height, kernel, kernel_size, verbose);
    }
    
    if (!cpu_only) {
        runGPUBenchmarks(input, output, width, height, kernel, kernel_size, verbose);
    }
    
    // Save results
    saveResults(input, output, width, height, output_dir);
    
    std::cout << "\nBenchmark completed successfully!" << std::endl;
    std::cout << "Check " << output_dir << " for output images" << std::endl;
    
    return 0;
}

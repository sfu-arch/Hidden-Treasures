#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <cmath>
#include <iomanip>

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

void runImageProcessingBenchmark(const std::string& input_file, const std::string& output_dir,
                                 const std::string& kernel_file, bool cpu_only, bool gpu_only, bool verbose) {
    
    std::cout << "\n=== Image Processing Benchmark ===" << std::endl;
    std::cout << "Input image: " << input_file << std::endl;
    std::cout << "Kernel file: " << kernel_file << std::endl;
    std::cout << "Output directory: " << output_dir << std::endl;
    
    // Create output directory
    createDirectory(output_dir);
    
    // Load input image
    ImageData input_image = load_image(input_file, true); // Force grayscale
    if (!input_image.is_valid()) {
        std::cerr << "❌ Error: Failed to load input image" << std::endl;
        return;
    }
    
    // Load convolution kernel
    auto kernel_data = load_kernel(kernel_file);
    if (kernel_data.empty()) {
        std::cerr << "❌ Error: Failed to load kernel from " << kernel_file << std::endl;
        return;
    }
    
    int kernel_size = static_cast<int>(std::sqrt(kernel_data.size()));
    
    if (verbose) {
        std::cout << "Image dimensions: " << input_image.width << "x" << input_image.height << std::endl;
        std::cout << "Kernel size: " << kernel_size << "x" << kernel_size << std::endl;
        std::cout << "Processing mode: ";
        if (cpu_only) std::cout << "CPU only";
        else if (gpu_only) std::cout << "GPU only";
        else std::cout << "CPU + GPU comparison";
        std::cout << std::endl;
    }
    
    // Prepare output image
    ImageData output_image(input_image.width, input_image.height, 1);
    
    // Run CPU implementation for reference
    std::vector<float> cpu_result;
    double cpu_time = 0.0;
    
    if (!gpu_only) {
        std::cout << "\n--- CPU Processing ---" << std::endl;
        cpu_result.resize(input_image.width * input_image.height);
        
        Timer timer;
        timer.start();
        cpu::convolution2D(input_image.raw_data(), cpu_result.data(),
                          input_image.width, input_image.height,
                          kernel_data.data(), kernel_size,
                          cpu::ConvolutionMode::OPTIMIZED);
        timer.stop();
        cpu_time = timer.last_measurement();
        
        std::cout << "CPU convolution took: " << cpu_time << " ms" << std::endl;
        
        if (verbose) {
            cpu::printPerformanceMetrics(input_image.width, input_image.height, 
                                       kernel_size, cpu_time);
        }
        
        // Save CPU result
        ImageData cpu_output(input_image.width, input_image.height, 1);
        cpu_output.data = cpu_result;
        save_image(cpu_output, output_dir + "/cpu_result.png");
    }
    
    // Run GPU implementation
    if (!cpu_only) {
        std::cout << "\n--- GPU Processing ---" << std::endl;
        
        Timer timer;
        timer.start();
        gpu::convolution2D_naive(input_image.raw_data(), output_image.raw_data(),
                                 input_image.width, input_image.height,
                                 kernel_data.data(), kernel_size);
        timer.stop();
        double gpu_time = timer.last_measurement();
        
        std::cout << "GPU convolution took: " << gpu_time << " ms" << std::endl;
        
        if (verbose) {
            gpu::printGPUPerformanceMetrics(input_image.width, input_image.height,
                                           kernel_size, gpu_time, "GPU Naive");
        }
        
        // Save GPU result
        save_image(output_image, output_dir + "/gpu_result.png");
        
        // Compare with CPU if both were run
        if (!gpu_only && !cpu_result.empty()) {
            bool correct = compare_arrays(cpu_result.data(), output_image.raw_data(),
                                        input_image.width, input_image.height, 1e-3f);
            
            std::cout << "\nGPU result: " << (correct ? "CORRECT" : "INCORRECT") << std::endl;
            
            if (cpu_time > 0.0) {
                double speedup = cpu_time / gpu_time;
                std::cout << "GPU speedup: " << std::fixed << std::setprecision(2) 
                         << speedup << "x" << std::endl;
            }
        }
    }
    
    std::cout << "\n✅ Image processing completed!" << std::endl;
    std::cout << "Results saved to: " << output_dir << std::endl;
}

void runSyntheticBenchmark(int width, int height, int kernel_size, 
                          const std::string& output_dir, bool cpu_only, bool gpu_only, bool verbose) {
    
    std::cout << "\n=== Synthetic Data Benchmark ===" << std::endl;
    
    if (verbose) {
        std::cout << "Creating synthetic test data..." << std::endl;
        std::cout << "Image size: " << width << "x" << height << std::endl;
        std::cout << "Kernel size: " << kernel_size << "x" << kernel_size << std::endl;
    }
    
    // Create synthetic input data
    std::vector<float> input(width * height);
    std::vector<float> output(width * height);
    
    // Initialize with a test pattern
    for (int i = 0; i < width * height; ++i) {
        input[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    
    // Create a simple kernel (like Gaussian blur)
    std::vector<float> kernel(kernel_size * kernel_size);
    float sum = 0.0f;
    for (int i = 0; i < kernel_size * kernel_size; ++i) {
        kernel[i] = 1.0f;
        sum += kernel[i];
    }
    // Normalize kernel
    for (int i = 0; i < kernel_size * kernel_size; ++i) {
        kernel[i] /= sum;
    }
    
    // Run CPU implementation
    double cpu_time = 0.0;
    std::vector<float> cpu_result;
    
    if (!gpu_only) {
        std::cout << "\n--- CPU Processing ---" << std::endl;
        cpu_result.resize(width * height);
        
        Timer timer;
        timer.start();
        cpu::convolution2D(input.data(), cpu_result.data(), width, height, 
                          kernel.data(), kernel_size, cpu::ConvolutionMode::OPTIMIZED);
        timer.stop();
        cpu_time = timer.last_measurement();
        
        std::cout << "CPU convolution took: " << cpu_time << " ms" << std::endl;
        
        if (verbose) {
            cpu::printPerformanceMetrics(width, height, kernel_size, cpu_time);
        }
    }
    
    // Run GPU implementation
    if (!cpu_only) {
        std::cout << "\n--- GPU Processing ---" << std::endl;
        
        Timer timer;
        timer.start();
        gpu::convolution2D_naive(input.data(), output.data(), width, height,
                                 kernel.data(), kernel_size);
        timer.stop();
        double gpu_time = timer.last_measurement();
        
        std::cout << "GPU convolution took: " << gpu_time << " ms" << std::endl;
        
        if (verbose) {
            gpu::printGPUPerformanceMetrics(width, height, kernel_size, gpu_time, "GPU Naive");
        }
        
        // Compare with CPU if both were run
        if (!gpu_only && !cpu_result.empty()) {
            bool correct = compare_arrays(cpu_result.data(), output.data(),
                                        width, height, 1e-3f);
            
            std::cout << "\nGPU result: " << (correct ? "CORRECT" : "INCORRECT") << std::endl;
            
            if (cpu_time > 0.0) {
                double speedup = cpu_time / gpu_time;
                std::cout << "GPU speedup: " << std::fixed << std::setprecision(2) 
                         << speedup << "x" << std::endl;
            }
        }
    }
}

int main(int argc, char* argv[]) {
    // Default parameters
    int width = 1024;
    int height = 1024;
    int kernel_size = 5;
    std::string input_file = "";
    std::string kernel_file = "";
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
        else if ((arg == "--kernel-file") && i + 1 < argc) {
            kernel_file = argv[++i];
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
        std::cerr << "❌ Error: Kernel size must be odd" << std::endl;
        return 1;
    }
    
    if (width <= 0 || height <= 0 || kernel_size <= 0) {
        std::cerr << "❌ Error: Invalid dimensions" << std::endl;
        return 1;
    }
    
    std::cout << "\n🚀 CUDA Convolution Lab - Image Processing Benchmark" << std::endl;
    std::cout << "====================================================" << std::endl;
    
    try {
        // Check if we're processing a real image or using synthetic data
        if (!input_file.empty()) {
            // Real image processing mode
            if (kernel_file.empty()) {
                // Default to Sobel X edge detection if no kernel specified
                kernel_file = "data/kernels/sobel_x_3x3.txt";
                std::cout << "Using default Sobel X edge detection kernel" << std::endl;
            }
            
            runImageProcessingBenchmark(input_file, output_dir, kernel_file, 
                                       cpu_only, gpu_only, verbose);
        } else {
            // Synthetic data benchmark mode
            std::cout << "Mode: Synthetic data benchmark" << std::endl;
            std::cout << "Image size: " << width << "x" << height << std::endl;
            std::cout << "Kernel size: " << kernel_size << "x" << kernel_size << std::endl;
            
            runSyntheticBenchmark(width, height, kernel_size, output_dir, 
                                 cpu_only, gpu_only, verbose);
        }
        
        std::cout << "\n✅ Benchmark completed successfully!" << std::endl;
        std::cout << "Results saved to: " << output_dir << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

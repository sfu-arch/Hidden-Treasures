#pragma once

#include <chrono>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <cmath>

/**
 * @file timer.h
 * @brief High-precision timing utilities for performance measurement
 * 
 * Provides both CPU and GPU timing capabilities with statistical analysis
 * for benchmarking convolution implementations.
 */

// =============================================================================
// CPU Timer Class
// =============================================================================

/**
 * @brief High-precision CPU timer using std::chrono
 */
class CPUTimer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
    std::vector<double> measurements;
    std::string timer_name;
    bool is_running;

public:
    explicit CPUTimer(const std::string& name = "Timer") 
        : timer_name(name), is_running(false) {}

    /**
     * @brief Start timing
     */
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
        is_running = true;
    }

    /**
     * @brief Stop timing and return elapsed time in milliseconds
     * @return Elapsed time in milliseconds
     */
    double stop() {
        if (!is_running) {
            std::cerr << "Warning: Timer '" << timer_name << "' was not started" << std::endl;
            return 0.0;
        }

        end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
        double elapsed_ms = duration.count() / 1e6;
        
        measurements.push_back(elapsed_ms);
        is_running = false;
        
        return elapsed_ms;
    }

    /**
     * @brief Get the last measurement without adding a new one
     * @return Last elapsed time in milliseconds
     */
    double last_measurement() const {
        if (measurements.empty()) {
            return 0.0;
        }
        return measurements.back();
    }

    /**
     * @brief Clear all measurements
     */
    void clear() {
        measurements.clear();
    }

    /**
     * @brief Get number of measurements taken
     */
    size_t count() const {
        return measurements.size();
    }

    /**
     * @brief Calculate statistics from all measurements
     */
    struct Statistics {
        double mean;
        double min;
        double max;
        double std_dev;
        double median;
        size_t count;
    };

    Statistics get_statistics() const {
        if (measurements.empty()) {
            return {0.0, 0.0, 0.0, 0.0, 0.0, 0};
        }

        Statistics stats;
        stats.count = measurements.size();
        
        // Calculate mean
        stats.mean = std::accumulate(measurements.begin(), measurements.end(), 0.0) / measurements.size();
        
        // Find min and max
        auto minmax = std::minmax_element(measurements.begin(), measurements.end());
        stats.min = *minmax.first;
        stats.max = *minmax.second;
        
        // Calculate standard deviation
        double variance = 0.0;
        for (double measurement : measurements) {
            variance += (measurement - stats.mean) * (measurement - stats.mean);
        }
        variance /= measurements.size();
        stats.std_dev = std::sqrt(variance);
        
        // Calculate median
        std::vector<double> sorted_measurements = measurements;
        std::sort(sorted_measurements.begin(), sorted_measurements.end());
        if (sorted_measurements.size() % 2 == 0) {
            stats.median = (sorted_measurements[sorted_measurements.size()/2 - 1] + 
                           sorted_measurements[sorted_measurements.size()/2]) / 2.0;
        } else {
            stats.median = sorted_measurements[sorted_measurements.size()/2];
        }
        
        return stats;
    }

    /**
     * @brief Print statistics in a formatted table
     */
    void print_statistics() const {
        auto stats = get_statistics();
        
        std::cout << "\n📊 Timer Statistics: " << timer_name << std::endl;
        std::cout << std::string(50, '=') << std::endl;
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Measurements:     " << stats.count << std::endl;
        std::cout << "Mean:            " << std::setw(8) << stats.mean << " ms" << std::endl;
        std::cout << "Median:          " << std::setw(8) << stats.median << " ms" << std::endl;
        std::cout << "Min:             " << std::setw(8) << stats.min << " ms" << std::endl;
        std::cout << "Max:             " << std::setw(8) << stats.max << " ms" << std::endl;
        std::cout << "Std Dev:         " << std::setw(8) << stats.std_dev << " ms" << std::endl;
        std::cout << "Coefficient of Variation: " << std::setw(5) << std::setprecision(1) 
                  << (stats.std_dev / stats.mean * 100) << "%" << std::endl;
        std::cout << std::string(50, '=') << std::endl;
    }
};

// =============================================================================
// RAII Timer for Automatic Measurement
// =============================================================================

/**
 * @brief RAII timer that automatically starts on construction and stops on destruction
 */
class ScopedTimer {
private:
    CPUTimer& timer;
    std::string operation_name;

public:
    ScopedTimer(CPUTimer& t, const std::string& operation = "Operation") 
        : timer(t), operation_name(operation) {
        timer.start();
    }

    ~ScopedTimer() {
        double elapsed = timer.stop();
        std::cout << "⏱️  " << operation_name << " completed in " 
                  << std::fixed << std::setprecision(3) << elapsed << " ms" << std::endl;
    }
};

// Macro for easy scoped timing
#define TIME_OPERATION(timer, name) ScopedTimer _scoped_timer(timer, name)

// =============================================================================
// Performance Analysis Functions
// =============================================================================

/**
 * @brief Calculate performance metrics for convolution operations
 * @param time_ms Execution time in milliseconds
 * @param width Image width
 * @param height Image height
 * @param kernel_size Convolution kernel size
 * @return Performance metrics structure
 */
struct ConvolutionMetrics {
    double time_ms;
    double gflops;
    double bandwidth_gb_s;
    double pixels_per_second;
    long long total_operations;
    long long total_bytes;
};

inline ConvolutionMetrics calculate_convolution_metrics(
    double time_ms, int width, int height, int kernel_size) {
    
    ConvolutionMetrics metrics;
    metrics.time_ms = time_ms;
    
    // Calculate total operations and bytes
    long long total_pixels = static_cast<long long>(width) * height;
    long long ops_per_pixel = static_cast<long long>(kernel_size) * kernel_size;
    metrics.total_operations = total_pixels * ops_per_pixel * 2; // multiply-add
    
    // Estimate memory accesses (input reads + output write)
    long long bytes_per_pixel = sizeof(float) * (ops_per_pixel + 1);
    metrics.total_bytes = total_pixels * bytes_per_pixel;
    
    // Calculate performance metrics
    double time_s = time_ms / 1000.0;
    metrics.gflops = (metrics.total_operations / time_s) / 1e9;
    metrics.bandwidth_gb_s = (metrics.total_bytes / time_s) / 1e9;
    metrics.pixels_per_second = total_pixels / time_s;
    
    return metrics;
}

/**
 * @brief Print convolution performance metrics in a formatted table
 */
inline void print_convolution_metrics(const ConvolutionMetrics& metrics, const std::string& implementation_name) {
    std::cout << "\n🚀 Performance Metrics: " << implementation_name << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Execution Time:      " << std::setw(10) << metrics.time_ms << " ms" << std::endl;
    std::cout << "Throughput:          " << std::setw(10) << metrics.gflops << " GFLOPS" << std::endl;
    std::cout << "Memory Bandwidth:    " << std::setw(10) << metrics.bandwidth_gb_s << " GB/s" << std::endl;
    std::cout << "Pixels/Second:       " << std::setw(10) << std::scientific << metrics.pixels_per_second << std::endl;
    std::cout << std::fixed;
    std::cout << "Total Operations:    " << metrics.total_operations << std::endl;
    std::cout << "Total Memory Access: " << std::setprecision(1) << metrics.total_bytes / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
}

/**
 * @brief Compare two implementations and print speedup analysis
 */
inline void compare_implementations(const ConvolutionMetrics& baseline, const ConvolutionMetrics& optimized,
                                   const std::string& baseline_name, const std::string& optimized_name) {
    
    double speedup = baseline.time_ms / optimized.time_ms;
    double gflops_improvement = optimized.gflops / baseline.gflops;
    double bandwidth_improvement = optimized.bandwidth_gb_s / baseline.bandwidth_gb_s;
    
    std::cout << "\n📈 Performance Comparison" << std::endl;
    std::cout << std::string(50, '=') << std::endl;
    std::cout << "Baseline:        " << baseline_name << std::endl;
    std::cout << "Optimized:       " << optimized_name << std::endl;
    std::cout << std::string(50, '-') << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Speedup:         " << std::setw(8) << speedup << "x" << std::endl;
    std::cout << "GFLOPS Ratio:    " << std::setw(8) << gflops_improvement << "x" << std::endl;
    std::cout << "Bandwidth Ratio: " << std::setw(8) << bandwidth_improvement << "x" << std::endl;
    std::cout << std::string(50, '=') << std::endl;
    
    // Performance analysis
    if (speedup > 1.5) {
        std::cout << "✅ Significant improvement achieved!" << std::endl;
    } else if (speedup > 1.1) {
        std::cout << "⚠️  Moderate improvement - consider further optimization" << std::endl;
    } else if (speedup < 0.9) {
        std::cout << "❌ Performance regression detected!" << std::endl;
    } else {
        std::cout << "➡️  Similar performance - optimization may not be effective" << std::endl;
    }
}

// =============================================================================
// Benchmarking Utilities
// =============================================================================

/**
 * @brief Run multiple iterations and collect statistics
 * @param func Function to benchmark (should return execution time in ms)
 * @param iterations Number of iterations to run
 * @param warmup_iterations Number of warmup iterations (not counted)
 * @return Statistics from timing measurements
 */
template<typename Func>
CPUTimer::Statistics benchmark_function(Func func, int iterations = 10, int warmup_iterations = 3) {
    CPUTimer timer("Benchmark");
    
    // Warmup iterations
    for (int i = 0; i < warmup_iterations; ++i) {
        func();
    }
    
    // Actual benchmark iterations
    for (int i = 0; i < iterations; ++i) {
        timer.start();
        func();
        timer.stop();
    }
    
    return timer.get_statistics();
}

/**
 * @brief Validate timing consistency and warn about outliers
 */
inline void validate_timing_consistency(const CPUTimer::Statistics& stats, double outlier_threshold = 2.0) {
    double cv = stats.std_dev / stats.mean;
    
    std::cout << "\n🔍 Timing Validation" << std::endl;
    std::cout << std::string(30, '-') << std::endl;
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "Coefficient of Variation: " << cv * 100 << "%" << std::endl;
    
    if (cv < 0.05) {
        std::cout << "✅ Excellent timing consistency" << std::endl;
    } else if (cv < 0.10) {
        std::cout << "✅ Good timing consistency" << std::endl;
    } else if (cv < 0.20) {
        std::cout << "⚠️  Moderate timing variation - consider more iterations" << std::endl;
    } else {
        std::cout << "❌ High timing variation - results may be unreliable" << std::endl;
        std::cout << "   Consider: longer warmup, fewer background processes, or more iterations" << std::endl;
    }
    
    // Check for outliers
    double outlier_range = stats.mean + outlier_threshold * stats.std_dev;
    if (stats.max > outlier_range) {
        std::cout << "⚠️  Potential outliers detected (max: " << std::setprecision(3) 
                  << stats.max << " ms vs expected: " << outlier_range << " ms)" << std::endl;
    }
}

#ifndef CONV_CPU_H
#define CONV_CPU_H

namespace cpu {

/**
 * @brief Convolution execution modes for CPU implementation
 */
enum class ConvolutionMode {
    SEQUENTIAL,  ///< Single-threaded implementation
    OPENMP,      ///< Multi-threaded OpenMP implementation
    OPTIMIZED    ///< Cache-optimized blocked implementation
};

/**
 * @brief Performs 2D convolution on CPU with specified mode
 * @param input Input image data (row-major order)
 * @param output Output image data (row-major order, pre-allocated)
 * @param width Image width in pixels
 * @param height Image height in pixels
 * @param kernel Convolution kernel (row-major order)
 * @param kernel_size Kernel size (assumed square, odd dimension)
 * @param mode Execution mode (sequential, OpenMP, or optimized)
 */
void convolution2D(const float* input, float* output, int width, int height,
                   const float* kernel, int kernel_size, ConvolutionMode mode = ConvolutionMode::SEQUENTIAL);

/**
 * @brief Sequential (single-threaded) 2D convolution implementation
 * @param input Input image data
 * @param output Output image data (pre-allocated)
 * @param width Image width in pixels
 * @param height Image height in pixels
 * @param kernel Convolution kernel
 * @param kernel_size Kernel size (odd dimension)
 */
void convolution2D_sequential(const float* input, float* output, int width, int height,
                             const float* kernel, int kernel_size);

/**
 * @brief OpenMP parallelized 2D convolution implementation
 * @param input Input image data
 * @param output Output image data (pre-allocated)
 * @param width Image width in pixels
 * @param height Image height in pixels
 * @param kernel Convolution kernel
 * @param kernel_size Kernel size (odd dimension)
 */
void convolution2D_openmp(const float* input, float* output, int width, int height,
                         const float* kernel, int kernel_size);

/**
 * @brief Cache-optimized blocked 2D convolution implementation
 * @param input Input image data
 * @param output Output image data (pre-allocated)
 * @param width Image width in pixels
 * @param height Image height in pixels
 * @param kernel Convolution kernel
 * @param kernel_size Kernel size (odd dimension)
 */
void convolution2D_optimized(const float* input, float* output, int width, int height,
                            const float* kernel, int kernel_size);

/**
 * @brief Calculate theoretical GFLOPS for convolution operation
 * @param width Image width
 * @param height Image height
 * @param kernel_size Kernel size
 * @param time_ms Execution time in milliseconds
 * @return GFLOPS (Giga Floating Point Operations Per Second)
 */
float calculateGFLOPS(int width, int height, int kernel_size, double time_ms);

/**
 * @brief Calculate memory bandwidth utilization
 * @param width Image width
 * @param height Image height
 * @param kernel_size Kernel size
 * @param time_ms Execution time in milliseconds
 * @return Memory bandwidth in GB/s
 */
double calculateMemoryBandwidth(int width, int height, int kernel_size, double time_ms);

/**
 * @brief Get string representation of convolution mode
 * @param mode Convolution execution mode
 * @return String description of the mode
 */
const char* getModeString(ConvolutionMode mode);

/**
 * @brief Print comprehensive performance metrics
 * @param width Image width
 * @param height Image height
 * @param kernel_size Kernel size
 * @param time_ms Execution time in milliseconds
 */
void printPerformanceMetrics(int width, int height, int kernel_size, double time_ms);

} // namespace cpu

#endif // CONV_CPU_H

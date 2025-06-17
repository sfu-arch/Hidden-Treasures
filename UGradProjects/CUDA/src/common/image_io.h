#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <iostream>
#include <memory>
#include <fstream>

/**
 * @file image_io.h
 * @brief Image loading, saving, and manipulation utilities for the convolution lab
 * 
 * Provides OpenCV-based image I/O with float format conversion and validation
 * specifically designed for GPU convolution operations.
 */

// =============================================================================
// Image Data Structure
// =============================================================================

/**
 * @brief Container for image data in float format suitable for GPU processing
 */
struct ImageData {
    std::vector<float> data;    ///< Image data in row-major order
    int width;                  ///< Image width in pixels
    int height;                 ///< Image height in pixels
    int channels;               ///< Number of channels (1=grayscale, 3=RGB, 4=RGBA)
    
    ImageData() : width(0), height(0), channels(0) {}
    
    ImageData(int w, int h, int c) : width(w), height(h), channels(c) {
        data.resize(w * h * c);
    }
    
    /**
     * @brief Get total number of pixels
     */
    size_t pixel_count() const { return width * height; }
    
    /**
     * @brief Get total number of elements (pixels * channels)
     */
    size_t element_count() const { return width * height * channels; }
    
    /**
     * @brief Get memory size in bytes
     */
    size_t memory_size() const { return element_count() * sizeof(float); }
    
    /**
     * @brief Check if image data is valid
     */
    bool is_valid() const {
        return width > 0 && height > 0 && channels > 0 && 
               data.size() == element_count();
    }
    
    /**
     * @brief Get pixel value at specified coordinates and channel
     */
    float& at(int x, int y, int channel = 0) {
        return data[y * width * channels + x * channels + channel];
    }
    
    const float& at(int x, int y, int channel = 0) const {
        return data[y * width * channels + x * channels + channel];
    }
    
    /**
     * @brief Get raw data pointer for GPU operations
     */
    float* raw_data() { return data.data(); }
    const float* raw_data() const { return data.data(); }
};

// =============================================================================
// Image Loading Functions
// =============================================================================

/**
 * @brief Load image from file and convert to float format
 * @param filename Path to image file
 * @param force_grayscale Convert to grayscale if true
 * @return ImageData structure with loaded image
 */
inline ImageData load_image(const std::string& filename, bool force_grayscale = false) {
    ImageData image;
    
    // Load image using OpenCV
    cv::Mat cv_image;
    if (force_grayscale) {
        cv_image = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    } else {
        cv_image = cv::imread(filename, cv::IMREAD_COLOR);
    }
    
    if (cv_image.empty()) {
        std::cerr << "❌ Error: Could not load image '" << filename << "'" << std::endl;
        return image;
    }
    
    // Convert to float format [0.0, 1.0]
    cv::Mat float_image;
    cv_image.convertTo(float_image, CV_32F, 1.0/255.0);
    
    // Extract image properties
    image.width = float_image.cols;
    image.height = float_image.rows;
    image.channels = float_image.channels();
    
    // Copy data to our structure
    image.data.resize(image.element_count());
    std::memcpy(image.data.data(), float_image.data, image.memory_size());
    
    std::cout << "✅ Loaded image: " << filename << std::endl;
    std::cout << "   Dimensions: " << image.width << "x" << image.height 
              << " (" << image.channels << " channels)" << std::endl;
    std::cout << "   Memory size: " << image.memory_size() / (1024.0 * 1024.0) << " MB" << std::endl;
    
    return image;
}

/**
 * @brief Create a test pattern image for debugging
 * @param width Image width
 * @param height Image height
 * @param pattern_type Type of test pattern (0=checkerboard, 1=gradient, 2=random)
 * @return Generated test image
 */
inline ImageData create_test_image(int width, int height, int pattern_type = 0) {
    ImageData image(width, height, 1); // Grayscale test pattern
    
    switch (pattern_type) {
        case 0: // Checkerboard pattern
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    bool checker = ((x / 32) + (y / 32)) % 2;
                    image.at(x, y) = checker ? 1.0f : 0.0f;
                }
            }
            break;
            
        case 1: // Gradient pattern
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    image.at(x, y) = static_cast<float>(x) / width;
                }
            }
            break;
            
        case 2: // Random pattern
            std::srand(42); // Fixed seed for reproducibility
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    image.at(x, y) = static_cast<float>(std::rand()) / RAND_MAX;
                }
            }
            break;
    }
    
    std::cout << "✅ Created test image: " << width << "x" << height 
              << " (pattern type " << pattern_type << ")" << std::endl;
    
    return image;
}

// =============================================================================
// Image Saving Functions
// =============================================================================

/**
 * @brief Save image to file
 * @param image Image data to save
 * @param filename Output filename
 * @param quality JPEG quality (0-100) for JPEG files
 * @return true if successful
 */
inline bool save_image(const ImageData& image, const std::string& filename, int quality = 95) {
    if (!image.is_valid()) {
        std::cerr << "❌ Error: Invalid image data for saving" << std::endl;
        return false;
    }
    
    // Convert from float [0.0, 1.0] to uint8 [0, 255]
    cv::Mat cv_image;
    cv::Mat float_image(image.height, image.width, 
                        image.channels == 1 ? CV_32F : CV_32FC3,
                        const_cast<float*>(image.raw_data()));
    
    float_image.convertTo(cv_image, CV_8U, 255.0);
    
    // Set JPEG quality if saving JPEG
    std::vector<int> compression_params;
    if (filename.find(".jpg") != std::string::npos || filename.find(".jpeg") != std::string::npos) {
        compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
        compression_params.push_back(quality);
    }
    
    bool success = cv::imwrite(filename, cv_image, compression_params);
    
    if (success) {
        std::cout << "✅ Saved image: " << filename << std::endl;
    } else {
        std::cerr << "❌ Error: Could not save image '" << filename << "'" << std::endl;
    }
    
    return success;
}

// =============================================================================
// Image Comparison and Validation
// =============================================================================

/**
 * @brief Compare two images and calculate difference metrics
 * @param img1 First image
 * @param img2 Second image
 * @param tolerance Maximum allowed difference per pixel
 * @return true if images are similar within tolerance
 */
inline bool compare_images(const ImageData& img1, const ImageData& img2, float tolerance = 1e-3f) {
    if (img1.width != img2.width || img1.height != img2.height || img1.channels != img2.channels) {
        std::cerr << "❌ Image comparison failed: Dimension mismatch" << std::endl;
        return false;
    }
    
    float max_diff = 0.0f;
    float mse = 0.0f;
    size_t different_pixels = 0;
    
    for (size_t i = 0; i < img1.element_count(); ++i) {
        float diff = std::abs(img1.data[i] - img2.data[i]);
        max_diff = std::max(max_diff, diff);
        mse += diff * diff;
        
        if (diff > tolerance) {
            different_pixels++;
        }
    }
    
    mse /= img1.element_count();
    float rmse = std::sqrt(mse);
    float psnr = 20.0f * std::log10(1.0f / rmse);
    
    std::cout << "\n📊 Image Comparison Results:" << std::endl;
    std::cout << "   Max difference:     " << std::scientific << max_diff << std::endl;
    std::cout << "   RMSE:               " << rmse << std::endl;
    std::cout << "   PSNR:               " << std::fixed << std::setprecision(2) << psnr << " dB" << std::endl;
    std::cout << "   Different pixels:   " << different_pixels << " / " << img1.pixel_count() 
              << " (" << std::setprecision(1) << 100.0 * different_pixels / img1.pixel_count() << "%)" << std::endl;
    
    bool similar = (max_diff <= tolerance);
    if (similar) {
        std::cout << "✅ Images are similar within tolerance (" << tolerance << ")" << std::endl;
    } else {
        std::cout << "❌ Images differ by more than tolerance (" << tolerance << ")" << std::endl;
    }
    
    return similar;
}

/**
 * @brief Create difference image for visual inspection
 * @param img1 First image
 * @param img2 Second image
 * @param scale Scale factor for difference visualization
 * @return Difference image (amplified for visibility)
 */
inline ImageData create_difference_image(const ImageData& img1, const ImageData& img2, float scale = 10.0f) {
    if (img1.width != img2.width || img1.height != img2.height || img1.channels != img2.channels) {
        std::cerr << "❌ Cannot create difference image: Dimension mismatch" << std::endl;
        return ImageData();
    }
    
    ImageData diff_image(img1.width, img1.height, img1.channels);
    
    for (size_t i = 0; i < img1.element_count(); ++i) {
        float diff = std::abs(img1.data[i] - img2.data[i]) * scale;
        diff_image.data[i] = std::min(diff, 1.0f); // Clamp to [0, 1]
    }
    
    return diff_image;
}

/**
 * @brief Compare two raw float arrays (for backward compatibility)
 * @param data1 First array
 * @param data2 Second array
 * @param width Image width
 * @param height Image height
 * @param tolerance Comparison tolerance
 * @return True if arrays are similar within tolerance
 */
inline bool compare_arrays(const float* data1, const float* data2, 
                          int width, int height, float tolerance = 1e-4f) {
    int total_elements = width * height;
    for (int i = 0; i < total_elements; ++i) {
        if (std::abs(data1[i] - data2[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

// =============================================================================
// Convolution Kernel Loading
// =============================================================================

/**
 * @brief Load convolution kernel from text file
 * @param filename Path to kernel file
 * @return 2D vector containing kernel values
 */
inline std::vector<std::vector<float>> load_kernel(const std::string& filename) {
    std::vector<std::vector<float>> kernel;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "❌ Error: Could not open kernel file '" << filename << "'" << std::endl;
        return kernel;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        std::vector<float> row;
        std::istringstream iss(line);
        float value;
        
        while (iss >> value) {
            row.push_back(value);
        }
        
        if (!row.empty()) {
            kernel.push_back(row);
        }
    }
    
    if (!kernel.empty()) {
        std::cout << "✅ Loaded kernel: " << filename << " (" 
                  << kernel.size() << "x" << kernel[0].size() << ")" << std::endl;
    }
    
    return kernel;
}

/**
 * @brief Create standard convolution kernels
 * @param type Kernel type ("blur", "sharpen", "edge", "sobel_x", "sobel_y")
 * @param size Kernel size (3, 5, 7, etc.) - must be odd
 * @return 2D vector containing kernel values
 */
inline std::vector<std::vector<float>> create_standard_kernel(const std::string& type, int size = 3) {
    std::vector<std::vector<float>> kernel;
    
    if (size % 2 == 0) {
        std::cerr << "❌ Error: Kernel size must be odd" << std::endl;
        return kernel;
    }
    
    if (type == "blur") {
        // Gaussian-like blur kernel
        kernel.resize(size, std::vector<float>(size));
        float sum = 0.0f;
        int center = size / 2;
        
        for (int y = 0; y < size; ++y) {
            for (int x = 0; x < size; ++x) {
                float dist = std::sqrt((x - center) * (x - center) + (y - center) * (y - center));
                float value = std::exp(-dist * dist / (2.0f * 1.0f * 1.0f));
                kernel[y][x] = value;
                sum += value;
            }
        }
        
        // Normalize
        for (int y = 0; y < size; ++y) {
            for (int x = 0; x < size; ++x) {
                kernel[y][x] /= sum;
            }
        }
        
    } else if (type == "sharpen") {
        kernel = {{0, -1, 0}, {-1, 5, -1}, {0, -1, 0}};
        
    } else if (type == "edge") {
        kernel = {{-1, -1, -1}, {-1, 8, -1}, {-1, -1, -1}};
        
    } else if (type == "sobel_x") {
        kernel = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
        
    } else if (type == "sobel_y") {
        kernel = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
        
    } else {
        std::cerr << "❌ Error: Unknown kernel type '" << type << "'" << std::endl;
        return kernel;
    }
    
    std::cout << "✅ Created " << type << " kernel (" << size << "x" << size << ")" << std::endl;
    return kernel;
}

/**
 * @brief Save kernel to text file
 * @param kernel 2D kernel values
 * @param filename Output filename
 * @return true if successful
 */
inline bool save_kernel(const std::vector<std::vector<float>>& kernel, const std::string& filename) {
    std::ofstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "❌ Error: Could not create kernel file '" << filename << "'" << std::endl;
        return false;
    }
    
    for (const auto& row : kernel) {
        for (size_t i = 0; i < row.size(); ++i) {
            file << std::fixed << std::setprecision(6) << row[i];
            if (i < row.size() - 1) file << " ";
        }
        file << std::endl;
    }
    
    std::cout << "✅ Saved kernel: " << filename << std::endl;
    return true;
}

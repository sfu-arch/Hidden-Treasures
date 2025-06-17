#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <fstream>
#include <filesystem>

/**
 * @file utils.h
 * @brief General utility functions for the CUDA convolution lab
 * 
 * Provides command-line parsing, logging, formatting, and other
 * general-purpose utilities used throughout the project.
 */

// =============================================================================
// Command Line Argument Parsing
// =============================================================================

/**
 * @brief Simple command line argument parser
 */
class ArgumentParser {
private:
    std::map<std::string, std::string> args;
    std::vector<std::string> positional_args;
    std::string program_name;

public:
    ArgumentParser(int argc, char* argv[]) {
        if (argc > 0) {
            program_name = argv[0];
        }
        
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            
            if (arg.substr(0, 2) == "--") {
                // Long option: --key=value or --key value
                size_t eq_pos = arg.find('=');
                if (eq_pos != std::string::npos) {
                    std::string key = arg.substr(2, eq_pos - 2);
                    std::string value = arg.substr(eq_pos + 1);
                    args[key] = value;
                } else {
                    std::string key = arg.substr(2);
                    if (i + 1 < argc && argv[i + 1][0] != '-') {
                        args[key] = argv[++i];
                    } else {
                        args[key] = "true"; // Flag without value
                    }
                }
            } else if (arg.substr(0, 1) == "-" && arg.length() > 1) {
                // Short option: -k value or -flag
                std::string key = arg.substr(1);
                if (i + 1 < argc && argv[i + 1][0] != '-') {
                    args[key] = argv[++i];
                } else {
                    args[key] = "true"; // Flag without value
                }
            } else {
                // Positional argument
                positional_args.push_back(arg);
            }
        }
    }

    /**
     * @brief Check if argument exists
     */
    bool has(const std::string& key) const {
        return args.find(key) != args.end();
    }

    /**
     * @brief Get string argument with default value
     */
    std::string get(const std::string& key, const std::string& default_value = "") const {
        auto it = args.find(key);
        return (it != args.end()) ? it->second : default_value;
    }

    /**
     * @brief Get integer argument with default value
     */
    int get_int(const std::string& key, int default_value = 0) const {
        auto it = args.find(key);
        if (it != args.end()) {
            return std::stoi(it->second);
        }
        return default_value;
    }

    /**
     * @brief Get float argument with default value
     */
    float get_float(const std::string& key, float default_value = 0.0f) const {
        auto it = args.find(key);
        if (it != args.end()) {
            return std::stof(it->second);
        }
        return default_value;
    }

    /**
     * @brief Get boolean argument (checks for flag presence or "true"/"false")
     */
    bool get_bool(const std::string& key, bool default_value = false) const {
        auto it = args.find(key);
        if (it != args.end()) {
            std::string value = it->second;
            std::transform(value.begin(), value.end(), value.begin(), ::tolower);
            return value == "true" || value == "1" || value == "yes";
        }
        return default_value;
    }

    /**
     * @brief Get positional arguments
     */
    const std::vector<std::string>& get_positional() const {
        return positional_args;
    }

    /**
     * @brief Print help message
     */
    void print_help(const std::string& description = "") const {
        std::cout << "Usage: " << program_name << " [OPTIONS]" << std::endl;
        if (!description.empty()) {
            std::cout << "\n" << description << std::endl;
        }
        std::cout << "\nCommon Options:" << std::endl;
        std::cout << "  --input FILE        Input image file" << std::endl;
        std::cout << "  --output FILE       Output image file" << std::endl;
        std::cout << "  --kernel FILE       Convolution kernel file" << std::endl;
        std::cout << "  --implementation NAME  Implementation to use (cpu, naive, coalesced, shared, etc.)" << std::endl;
        std::cout << "  --benchmark         Run performance benchmark" << std::endl;
        std::cout << "  --iterations N      Number of benchmark iterations (default: 10)" << std::endl;
        std::cout << "  --verbose           Enable verbose output" << std::endl;
        std::cout << "  --help              Show this help message" << std::endl;
    }
};

// =============================================================================
// Logging and Output Utilities
// =============================================================================

/**
 * @brief Simple logging levels
 */
enum class LogLevel {
    DEBUG = 0,
    INFO = 1,
    WARNING = 2,
    ERROR = 3
};

/**
 * @brief Global log level (can be set via environment or command line)
 */
static LogLevel g_log_level = LogLevel::INFO;

/**
 * @brief Set global logging level
 */
inline void set_log_level(LogLevel level) {
    g_log_level = level;
}

/**
 * @brief Log message with specified level
 */
template<typename... Args>
void log_message(LogLevel level, const std::string& format, Args... args) {
    if (level < g_log_level) return;
    
    // Get current time
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    // Format timestamp
    std::cout << "[" << std::put_time(std::localtime(&time_t), "%H:%M:%S");
    std::cout << "." << std::setfill('0') << std::setw(3) << ms.count() << "] ";
    
    // Add level prefix
    switch (level) {
        case LogLevel::DEBUG:   std::cout << "🐛 DEBUG: "; break;
        case LogLevel::INFO:    std::cout << "ℹ️  INFO:  "; break;
        case LogLevel::WARNING: std::cout << "⚠️  WARN:  "; break;
        case LogLevel::ERROR:   std::cout << "❌ ERROR: "; break;
    }
    
    // Print formatted message
    printf(format.c_str(), args...);
    std::cout << std::endl;
}

// Convenience macros for logging
#define LOG_DEBUG(fmt, ...) log_message(LogLevel::DEBUG, fmt, ##__VA_ARGS__)
#define LOG_INFO(fmt, ...) log_message(LogLevel::INFO, fmt, ##__VA_ARGS__)
#define LOG_WARNING(fmt, ...) log_message(LogLevel::WARNING, fmt, ##__VA_ARGS__)
#define LOG_ERROR(fmt, ...) log_message(LogLevel::ERROR, fmt, ##__VA_ARGS__)

// =============================================================================
// String and Formatting Utilities
// =============================================================================

/**
 * @brief Format file size in human-readable format
 */
inline std::string format_file_size(size_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int unit_index = 0;
    double size = static_cast<double>(bytes);
    
    while (size >= 1024.0 && unit_index < 4) {
        size /= 1024.0;
        unit_index++;
    }
    
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << size << " " << units[unit_index];
    return oss.str();
}

/**
 * @brief Format execution time in human-readable format
 */
inline std::string format_time(double milliseconds) {
    if (milliseconds < 1.0) {
        return std::to_string(static_cast<int>(milliseconds * 1000)) + " μs";
    } else if (milliseconds < 1000.0) {
        return std::to_string(static_cast<int>(milliseconds)) + " ms";
    } else {
        return std::to_string(milliseconds / 1000.0) + " s";
    }
}

/**
 * @brief Format throughput values
 */
inline std::string format_throughput(double value, const std::string& unit) {
    const char* prefixes[] = {"", "K", "M", "G", "T"};
    int prefix_index = 0;
    
    while (value >= 1000.0 && prefix_index < 4) {
        value /= 1000.0;
        prefix_index++;
    }
    
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << value << " " 
        << prefixes[prefix_index] << unit;
    return oss.str();
}

/**
 * @brief Create a formatted table row
 */
inline std::string table_row(const std::vector<std::string>& columns, 
                            const std::vector<int>& widths) {
    std::ostringstream oss;
    for (size_t i = 0; i < columns.size() && i < widths.size(); ++i) {
        oss << "| " << std::setw(widths[i]) << std::left << columns[i] << " ";
    }
    oss << "|";
    return oss.str();
}

/**
 * @brief Create a table separator line
 */
inline std::string table_separator(const std::vector<int>& widths) {
    std::ostringstream oss;
    for (int width : widths) {
        oss << "+" << std::string(width + 2, '-');
    }
    oss << "+";
    return oss.str();
}

// =============================================================================
// Mathematical Utilities
// =============================================================================

/**
 * @brief Calculate percentage with proper formatting
 */
inline std::string format_percentage(double value, double total) {
    if (total == 0.0) return "N/A";
    double percentage = (value / total) * 100.0;
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1) << percentage << "%";
    return oss.str();
}

/**
 * @brief Round up to nearest power of 2
 */
inline int next_power_of_2(int value) {
    if (value <= 0) return 1;
    int power = 1;
    while (power < value) {
        power <<= 1;
    }
    return power;
}

/**
 * @brief Round up to nearest multiple
 */
inline int round_up_to_multiple(int value, int multiple) {
    return ((value + multiple - 1) / multiple) * multiple;
}

/**
 * @brief Calculate operational intensity for roofline analysis
 */
inline double calculate_operational_intensity(long long operations, long long bytes) {
    if (bytes == 0) return 0.0;
    return static_cast<double>(operations) / static_cast<double>(bytes);
}

// =============================================================================
// Progress and Status Display
// =============================================================================

/**
 * @brief Simple progress bar
 */
class ProgressBar {
private:
    int total;
    int current;
    int width;
    std::string description;

public:
    ProgressBar(int total_items, int bar_width = 50, const std::string& desc = "Progress")
        : total(total_items), current(0), width(bar_width), description(desc) {}

    void update(int completed) {
        current = completed;
        display();
    }

    void increment() {
        current++;
        display();
    }

    void finish() {
        current = total;
        display();
        std::cout << std::endl;
    }

private:
    void display() {
        double progress = static_cast<double>(current) / total;
        int filled = static_cast<int>(progress * width);
        
        std::cout << "\r" << description << ": [";
        std::cout << std::string(filled, '█');
        std::cout << std::string(width - filled, '░');
        std::cout << "] " << std::setw(3) << static_cast<int>(progress * 100) << "% ";
        std::cout << "(" << current << "/" << total << ")";
        std::cout.flush();
    }
};

// =============================================================================
// System and Environment Utilities
// =============================================================================

/**
 * @brief Get environment variable with default value
 */
inline std::string get_env_var(const std::string& name, const std::string& default_value = "") {
    const char* value = std::getenv(name.c_str());
    return value ? std::string(value) : default_value;
}

/**
 * @brief Check if file exists
 */
inline bool file_exists(const std::string& filename) {
    std::ifstream file(filename);
    return file.good();
}

/**
 * @brief Get file extension
 */
inline std::string get_file_extension(const std::string& filename) {
    size_t pos = filename.find_last_of('.');
    if (pos != std::string::npos) {
        return filename.substr(pos + 1);
    }
    return "";
}

/**
 * @brief Create output filename based on input and suffix
 */
inline std::string create_output_filename(const std::string& input_filename, 
                                         const std::string& suffix) {
    size_t pos = input_filename.find_last_of('.');
    if (pos != std::string::npos) {
        return input_filename.substr(0, pos) + "_" + suffix + 
               input_filename.substr(pos);
    }
    return input_filename + "_" + suffix;
}

/**
 * @brief Create a directory if it doesn't exist
 * @param path Directory path to create
 * @return True if directory was created or already exists
 */
inline bool createDirectory(const std::string& path) {
    try {
        std::filesystem::create_directories(path);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to create directory: " << path << " - " << e.what() << std::endl;
        return false;
    }
}

// =============================================================================
// Configuration and Settings
// =============================================================================

/**
 * @brief Simple configuration manager
 */
class Config {
private:
    std::map<std::string, std::string> settings;

public:
    void set(const std::string& key, const std::string& value) {
        settings[key] = value;
    }

    template<typename T>
    void set(const std::string& key, const T& value) {
        std::ostringstream oss;
        oss << value;
        settings[key] = oss.str();
    }

    std::string get_string(const std::string& key, const std::string& default_value = "") const {
        auto it = settings.find(key);
        return (it != settings.end()) ? it->second : default_value;
    }

    int get_int(const std::string& key, int default_value = 0) const {
        auto it = settings.find(key);
        if (it != settings.end()) {
            try {
                return std::stoi(it->second);
            } catch (...) {
                return default_value;
            }
        }
        return default_value;
    }

    bool get_bool(const std::string& key, bool default_value = false) const {
        auto it = settings.find(key);
        if (it != settings.end()) {
            std::string value = it->second;
            std::transform(value.begin(), value.end(), value.begin(), ::tolower);
            return value == "true" || value == "1" || value == "yes";
        }
        return default_value;
    }

    void print_all() const {
        std::cout << "📋 Configuration Settings:" << std::endl;
        for (const auto& pair : settings) {
            std::cout << "   " << pair.first << " = " << pair.second << std::endl;
        }
    }
};

// Global configuration instance
extern Config g_config;

// =============================================================================
// Kernel Loading Functions
// =============================================================================

/**
 * @brief Load convolution kernel from text file
 * @param filename Path to kernel file
 * @return Vector containing kernel data in row-major order
 * 
 * File format:
 * Line 1: kernel_size (integer)
 * Next kernel_size lines: kernel_size float values per line
 */
inline std::vector<float> load_kernel(const std::string& filename) {
    std::vector<float> kernel;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "❌ Error: Could not open kernel file '" << filename << "'" << std::endl;
        return kernel;
    }
    
    std::string line;
    int kernel_size = 0;
    int line_count = 0;
    
    while (std::getline(file, line)) {
        // Skip comments and empty lines
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        if (kernel_size == 0) {
            // First non-comment line should be the kernel size
            kernel_size = std::stoi(line);
            kernel.reserve(kernel_size * kernel_size);
            continue;
        }
        
        // Parse kernel values
        std::istringstream iss(line);
        float value;
        while (iss >> value) {
            kernel.push_back(value);
        }
        
        line_count++;
        if (line_count >= kernel_size) {
            break;
        }
    }
    
    file.close();
    
    // Validate kernel
    if (kernel.size() != static_cast<size_t>(kernel_size * kernel_size)) {
        std::cerr << "❌ Error: Invalid kernel file format. Expected " 
                  << kernel_size * kernel_size << " values, got " << kernel.size() << std::endl;
        kernel.clear();
        return kernel;
    }
    
    std::cout << "✅ Loaded " << kernel_size << "x" << kernel_size 
              << " kernel from " << filename << std::endl;
    
    return kernel;
}

/**
 * @brief Create a standard convolution kernel
 * @param type Kernel type ("gaussian", "sobel_x", "sobel_y", "laplacian")
 * @param size Kernel size (must be odd)
 * @return Vector containing kernel data
 */
inline std::vector<float> create_standard_kernel(const std::string& type, int size = 3) {
    std::vector<float> kernel;
    
    if (type == "gaussian" && size == 3) {
        kernel = {
            1.0f/16, 2.0f/16, 1.0f/16,
            2.0f/16, 4.0f/16, 2.0f/16,
            1.0f/16, 2.0f/16, 1.0f/16
        };
    }
    else if (type == "sobel_x" && size == 3) {
        kernel = {
            -1.0f, 0.0f, 1.0f,
            -2.0f, 0.0f, 2.0f,
            -1.0f, 0.0f, 1.0f
        };
    }
    else if (type == "sobel_y" && size == 3) {
        kernel = {
            -1.0f, -2.0f, -1.0f,
             0.0f,  0.0f,  0.0f,
             1.0f,  2.0f,  1.0f
        };
    }
    else if (type == "laplacian" && size == 3) {
        kernel = {
             0.0f, -1.0f,  0.0f,
            -1.0f,  4.0f, -1.0f,
             0.0f, -1.0f,  0.0f
        };
    }
    else {
        std::cerr << "❌ Error: Unknown kernel type '" << type 
                  << "' or unsupported size " << size << std::endl;
    }
    
    return kernel;
}

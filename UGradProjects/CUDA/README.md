# 🚀 CUDA Convolution Lab: Mining Memory Bandwidth

**From Naïve to Near-Peak Performance on Modern GPUs**

This project provides a comprehensive learning experience in GPU memory optimization through 2D image convolution implementations. Students will progress from basic CUDA kernels to highly optimized implementations that approach theoretical memory bandwidth limits.

## 🎯 Project Overview

Transform a simple 2D convolution from a slow CPU implementation to a high-performance GPU kernel achieving 30x+ speedup through systematic memory hierarchy optimization.

### Learning Objectives
- Master GPU memory hierarchy (Global → Shared → Constant → Texture)
- Understand memory coalescing and bank conflicts
- Implement advanced optimization techniques (async memory ops, occupancy tuning)
- Use professional profiling tools (Nsight Compute)
- Analyze performance with roofline models

## 🛠️ Prerequisites

**Hardware Requirements:**
- NVIDIA GPU with compute capability 6.0+ (GTX 1060 or newer)
- 8GB+ system RAM
- 20GB+ free disk space

**Software Requirements:**
- CUDA Toolkit 12.0+ (includes nvcc compiler)
- NVIDIA Driver (latest stable)
- Nsight Compute (for profiling)
- CMake 3.18+ or Make
- C++17 compatible compiler

**Skills Prerequisites:**
- C/C++ programming (pointers, arrays, memory management)
- Basic linear algebra (matrix operations)
- Command line usage
- Basic debugging skills

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Verify CUDA installation
nvcc --version
nvidia-smi

# Clone and build project
git clone <your-repo-url>
cd cuda-convolution-lab

### 1. Environment Setup

```bash
# Verify CUDA installation
nvcc --version
nvidia-smi

# Clone and build project
git clone <your-repo-url>
cd cuda-convolution-lab

# Build with CMake (recommended)
mkdir Build && cd Build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=89  # For RTX 4090
make

# Or build with Make
make all
```

### 2. Run Test Program

```bash
# Test basic functionality (CPU only to avoid GPU hangs)
./Build/conv_benchmark --width 512 --height 512 --kernel 3 --cpu-only --verbose

# Test GPU functionality (small size first)
./Build/conv_benchmark --width 256 --height 256 --kernel 3 --gpu-only --verbose

# Run comprehensive benchmark
./Build/conv_benchmark --width 1024 --height 1024 --kernel 5 --verbose
```

### 3. Real Image Processing

The project now supports real image processing using OpenCV with edge detection kernels:

```bash
# Process real images with Sobel edge detection
./Build/conv_benchmark --input data/input/test_image.png --kernel-file data/kernels/sobel_x_3x3.txt --verbose

# Use Sobel Y for vertical edge detection
./Build/conv_benchmark --input data/input/gradient.png --kernel-file data/kernels/sobel_y_3x3.txt --verbose

# Compare CPU vs GPU with real images
./Build/conv_benchmark --input data/input/test_image.png --kernel-file data/kernels/sobel_x_3x3.txt --output ./results --verbose
```

### 4. Profile Performance

```bash
# Profile with automated script
./tools/profile.sh --mode summary --size 512 --kernel 3

# View profiling results
ls profiling_results/
cat profiling_results/profiling_report.md
```

## 🖼️ Image Processing Features

### Supported Image Operations

The convolution lab now supports real image processing with **OpenCV integration**, enabling practical computer vision applications alongside performance optimization learning.

#### Edge Detection Filters
- **Sobel X Filter** (`data/kernels/sobel_x_3x3.txt`): Detects horizontal edges
- **Sobel Y Filter** (`data/kernels/sobel_y_3x3.txt`): Detects vertical edges  
- **Custom Kernels**: Add your own `.txt` files with space-separated values

#### Image Processing Workflow
1. **Load** - Supports common formats (PNG, JPG, BMP) via OpenCV
2. **Convert** - Automatic conversion to float32 [0.0-1.0] for GPU processing
3. **Process** - Apply convolution using CPU or GPU implementations
4. **Save** - Output results in PNG format with automatic scaling
5. **Compare** - Built-in CPU/GPU result validation with PSNR metrics

### Command Line Options

#### Basic Usage
```bash
# Synthetic benchmark mode (original functionality)
./conv_benchmark [options]

# Real image processing mode  
./conv_benchmark --input <image> --kernel-file <kernel> [options]
```

#### Core Options
- `--input <file>` - Input image file (PNG/JPG/BMP)
- `--kernel-file <file>` - Convolution kernel file (.txt format)
- `--output <dir>` - Output directory (default: ./output)
- `--verbose` - Detailed performance metrics and debugging info

#### Processing Control
- `--cpu-only` - Run only CPU implementation
- `--gpu-only` - Run only GPU implementation  
- `--width <int>` - Image width for synthetic mode (default: 1024)
- `--height <int>` - Image height for synthetic mode (default: 1024)
- `--kernel <int>` - Kernel size for synthetic mode (default: 5)

#### Examples

**Edge Detection on Real Images:**
```bash
# Horizontal edge detection
./conv_benchmark --input data/input/test_image.png \
                 --kernel-file data/kernels/sobel_x_3x3.txt \
                 --output ./edge_results --verbose

# Vertical edge detection  
./conv_benchmark --input data/input/gradient.png \
                 --kernel-file data/kernels/sobel_y_3x3.txt \
                 --output ./edge_results --verbose

# GPU-only processing for performance testing
./conv_benchmark --input data/input/test_image.png \
                 --kernel-file data/kernels/sobel_x_3x3.txt \
                 --gpu-only --verbose
```

**Synthetic Benchmarks (Original Mode):**
```bash
# Large synthetic benchmark
./conv_benchmark --width 2048 --height 2048 --kernel 7 --verbose

# CPU baseline measurement
./conv_benchmark --width 1024 --height 1024 --kernel 5 --cpu-only --verbose
```

### Output Files

When processing real images, the program generates:
- `cpu_result.png` - CPU convolution result (if CPU processing enabled)
- `gpu_result.png` - GPU convolution result (if GPU processing enabled)
- Console output with performance metrics, speedup analysis, and correctness validation

### Sample Kernels

**Sobel X (Horizontal Edges):**
```
-1 0 1
-2 0 2  
-1 0 1
```

**Sobel Y (Vertical Edges):**
```
-1 -2 -1
 0  0  0
 1  2  1
```

**Custom Kernel Format:**
Create `.txt` files with space or newline-separated float values in row-major order:
```
0.1 0.2 0.1
0.2 0.4 0.2
0.1 0.2 0.1
```

## 📁 Project Structure

```
cuda-convolution-lab/
├── README.md                          # Project overview and quick start
├── DESIGN.md                          # 📋 Detailed design document and roadmap
├── Makefile                           # Build automation
├── CMakeLists.txt                     # CMake build system
├── profiling_results/                 # 📊 Generated profiling data and reports
│   ├── profiling_report.md            # Executive summary
│   ├── roofline_analysis.txt          # Performance bounds analysis
│   ├── memory_analysis.md             # Memory optimization recommendations
│   └── *.nsys-rep                     # Nsight Systems timeline data
├── data/
│   ├── input/                         # 🖼️ Test images for real image processing
│   │   ├── test_image.png             # ✅ Synthetic test pattern (checkerboard)
│   │   ├── gradient.png               # ✅ Gradient test image
│   │   └── README.md                  # Image format and usage guidelines
│   ├── kernels/                       # 🔧 Convolution filter definitions  
│   │   ├── identity_3x3.txt           # ✅ Identity filter (no change)
│   │   ├── gaussian_5x5.txt           # ✅ Gaussian blur kernel
│   │   ├── sobel_x_3x3.txt            # ✅ Sobel X (horizontal edges)
│   │   ├── sobel_y_3x3.txt            # ✅ Sobel Y (vertical edges)  
│   │   └── README.md                  # Kernel format specifications
│   └── reference/                     # Expected output images
├── src/
│   ├── common/                        # ✅ Shared utilities and headers
│   │   ├── cuda_helpers.h             # CUDA error checking macros
│   │   ├── timer.h                    # Performance timing utilities
│   │   ├── image_io.h                 # ✅ OpenCV-based image I/O and processing
│   │   └── utils.h                    # ✅ Kernel loading and utility functions
│   ├── cpu/                           # ✅ CPU reference implementations
│   │   └── conv_cpu.cpp               # Sequential/OpenMP/Optimized versions
│   ├── gpu/                           # GPU kernel implementations
│   │   ├── conv_naive.cu              # ✅ Basic GPU kernel (needs correctness fix)
│   │   ├── conv_coalesced.cu          # ⏳ Memory coalescing optimization
│   │   ├── conv_shared.cu             # ⏳ Shared memory tiling
│   │   ├── conv_constant.cu           # ⏳ Constant memory optimization
│   │   ├── conv_texture.cu            # ⏳ Texture memory optimization
│   │   └── conv_async.cu              # ⏳ Asynchronous memory operations
│   └── main.cpp                       # ✅ Enhanced driver with real image processing
├── tests/
│   ├── unit_tests.cpp                 # Correctness verification
│   └── performance_tests.sh           # Automated benchmarking
├── tools/
│   ├── profile.sh                     # ✅ Automated profiling script
│   ├── roofline.py                    # ⏳ Performance analysis scripts
│   └── visualize.py                   # ⏳ Results plotting
└── docs/
    ├── lab_report_template.md         # Report structure guide
    ├── optimization_guide.md          # Step-by-step optimization hints
    └── profiling_guide.md             # Nsight Compute tutorial
```

**Legend**: ✅ Implemented | ⏳ Planned | ⚠️ Needs Fix

## 🎯 Implementation Milestones

### Phase 1: Foundation (Weeks 1-2)
- [x] **OpenCV Integration**: Real image I/O with format conversion
- [x] **Image Processing Mode**: Edge detection with Sobel kernels  
- [x] **Enhanced CLI**: Support for real images and kernel files
- [ ] **CPU Baseline**: Reference implementation with timing
- [ ] **Naive GPU**: Basic kernel, one thread per pixel
- [ ] **Coalesced Access**: Memory access pattern optimization

**Expected Performance**: 2-10x speedup over CPU

### Phase 2: Memory Hierarchy (Weeks 3-4)
- [ ] **Shared Memory Tiling**: 32x32 tiles with halo regions
- [ ] **Constant Memory**: Filter weights in constant cache
- [ ] **Texture Memory**: 2D spatial locality optimization

**Expected Performance**: 10-25x speedup over CPU

### Phase 3: Advanced Optimization (Weeks 5-6)
- [ ] **Async Memory**: Double-buffered pipelines
- [ ] **Occupancy Tuning**: Register and block size optimization
- [ ] **Performance Analysis**: Roofline model and bottleneck identification

**Expected Performance**: 25-40x speedup over CPU

## 🔬 Profiling and Analysis

### Key Performance Metrics

```bash
# Memory bandwidth utilization
ncu --metrics dram__throughput.avg.pct_of_peak ./conv_benchmark

# Memory coalescing efficiency
ncu --metrics smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct ./conv_benchmark

# Occupancy analysis
ncu --metrics sm__warps_active.avg.pct_of_peak ./conv_benchmark
```

### Performance Targets

| Implementation | Speedup | Memory Efficiency | L1 Hit Rate |
|---------------|---------|-------------------|-------------|
| CPU Baseline  | 1.0x    | ~10%              | ~90%        |
| Naive GPU     | 2-5x    | ~20%              | ~30%        |
| Coalesced     | 5-10x   | ~60%              | ~40%        |
| Shared Memory | 10-20x  | ~70%              | ~60%        |
| Advanced      | 25-40x  | ~85%              | ~80%        |

## 📊 **Current Status** (Updated June 17, 2025)

### ✅ **New Features Completed**
- **Real Image Processing**: OpenCV integration for PNG/JPG/BMP images
- **Edge Detection**: Sobel X and Y kernels for computer vision applications  
- **Enhanced CLI**: Support for `--input` and `--kernel-file` parameters
- **Result Validation**: CPU/GPU comparison with PSNR metrics
- **Test Images**: Generated synthetic test patterns for validation

### System Configuration
- **GPU**: NVIDIA GeForce RTX 4090 (Compute Capability 8.9)
- **Memory**: 24.56 GB total, 23.84 GB free
- **CUDA**: Version 12.6.77, Driver 560.35.03
- **Peak Memory Bandwidth**: ~1008 GB/s theoretical

### Performance Baseline (256x256 image, 3x3 kernel)

| Implementation | Execution Time | Throughput | Memory BW | Speedup |
|---------------|----------------|------------|-----------|---------|
| CPU Sequential | 1.24 ms | 3.81 GFLOPS | 14.97 GB/s | 1.0x |
| CPU OpenMP | 8.11 ms | 0.58 GFLOPS | 2.29 GB/s | 0.15x |
| CPU Optimized | 5.88 ms | 0.80 GFLOPS | 3.16 GB/s | 0.21x |
| **GPU Naive** | **80.27 ms** | **0.015 GFLOPS** | **0.058 GB/s** | **0.015x** |

### 🚨 **Key Findings from Profiling**
- **Critical Issue**: GPU implementation is **60x slower** than CPU!
- **Root Cause**: Memory allocation dominates 99.8% of execution time
- **Kernel Performance**: Actual compute is very fast (2.272 µs)
- **Memory Utilization**: Only 0.006% of peak bandwidth used
- **Optimization Potential**: 1000x+ improvement possible

### Profiling Data (Nsight Systems Analysis)
```
Time Distribution:
- cudaMalloc: 99.8% (88.14 ms)
- cudaMemcpy: 0.1% (0.105 ms) 
- Kernel execution: 0.001% (0.002 ms)
- cudaFree: 0.04% (0.038 ms)

Memory Transfers:
- Host-to-Device: 13.25 KB
- Device-to-Host: 12.51 KB
- Total: 25.76 KB

Kernel Metrics:
- Grid: (16,16) blocks
- Block size: (16,16) threads  
- Total threads: 65,536
- Execution time: 2.272 µs
```

### 🎯 **Immediate Optimization Targets**
1. **Memory Management**: Pre-allocate GPU memory to eliminate malloc overhead
2. **Memory Coalescing**: Reorganize access patterns for 50-100x bandwidth improvement
3. **Shared Memory**: Implement tiling for 10-20x compute efficiency
4. **Constant Memory**: Store kernel in constant cache for 2-3x access speed
5. **Algorithm Correctness**: Fix the "INCORRECT" result bug in GPU implementation

## 🛠️ Common Issues and Solutions

### Environment Setup
```bash
# If CUDA not found
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# If compilation fails
# Check CUDA compute capability matches your GPU
nvcc --help | grep -A 10 "gpu-architecture"
```

### Runtime Errors
```cpp
// Always check CUDA errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", \
                   __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)
```

## 📚 Documentation

### 📋 **[DESIGN.md](./DESIGN.md)** - Comprehensive Technical Documentation
- **Architecture Overview**: System design and component relationships
- **Implementation Status**: Current progress and completion status  
- **Performance Analysis**: Detailed profiling results and bottleneck analysis
- **Optimization Roadmap**: Phase-by-phase improvement strategy
- **Technical Specifications**: Build requirements and API details

### 📊 **[Profiling Results](./profiling_results/)** - Live Performance Data
- **Executive Summary**: `profiling_report.md` 
- **Memory Analysis**: `memory_analysis.md` with optimization recommendations
- **Roofline Analysis**: `roofline_analysis.txt` with performance bounds
- **Timeline Data**: `.nsys-rep` files for Nsight Systems visualization

### Learning Resources

### Essential Reading
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) - Chapters 3-6
- [Nsight Compute User Guide](https://docs.nvidia.com/nsight-compute/)
- "Programming Massively Parallel Processors" by Kirk & Hwu

### Video Tutorials
- [Udacity Intro to Parallel Programming](https://www.udacity.com/course/intro-to-parallel-programming--cs344)
- [NVIDIA Developer YouTube Channel](https://www.youtube.com/c/NVIDIADeveloper)

### Community Support
- [NVIDIA Developer Forums](https://forums.developer.nvidia.com/c/accelerated-computing/cuda/)
- [Stack Overflow CUDA Tag](https://stackoverflow.com/questions/tagged/cuda)

## 🎓 Assessment and Deliverables

### Code Deliverables
- [ ] All optimization implementations working correctly
- [ ] Comprehensive test suite with correctness verification
- [ ] Professional code documentation and comments
- [ ] Performance measurement and analysis scripts

### Analysis Report
- [ ] Performance progression analysis (5-10 pages)
- [ ] Roofline model and bottleneck identification
- [ ] Optimization methodology and lessons learned
- [ ] Professional visualizations and charts

### Success Criteria
- **Correctness**: All implementations produce identical results
- **Performance**: Achieve target speedups for each optimization level
- **Analysis**: Demonstrate understanding of GPU memory hierarchy
- **Documentation**: Professional-quality code and report

## 🤝 Contributing

This is an educational project template. Students should:
1. Fork the repository for their own implementation
2. Follow the milestone progression systematically
3. Document their optimization journey
4. Share insights and lessons learned

## 📄 License

Educational use only. See LICENSE file for details.

---

**Ready to start optimizing?** Begin with the CPU baseline implementation and work your way through each optimization milestone. Remember: the goal is not just performance, but understanding *why* each optimization works!

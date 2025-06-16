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

# Build with Make
make all

# Or build with CMake
mkdir build && cd build
cmake ..
make
```

### 2. Run Test Program

```bash
# Test CUDA setup
./test_cuda

# Run basic convolution benchmark
./conv_benchmark --input data/input/lenna_512x512.png --kernel data/kernels/blur_3x3.txt
```

### 3. Profile Performance

```bash
# Profile with Nsight Compute
./tools/profile.sh
```

## 📁 Project Structure

```
cuda-convolution-lab/
├── README.md                          # This file
├── Makefile                           # Build automation
├── CMakeLists.txt                     # Alternative build system
├── data/
│   ├── input/                         # Test images (add your own PNGs)
│   ├── kernels/                       # Convolution filter definitions
│   └── reference/                     # Expected output images
├── src/
│   ├── common/                        # Shared utilities and headers
│   │   ├── cuda_helpers.h             # CUDA error checking macros
│   │   ├── timer.h                    # Performance timing utilities
│   │   ├── image_io.h                 # Image loading/saving functions
│   │   └── utils.h                    # General utility functions
│   ├── cpu/
│   │   └── conv_cpu.cpp               # CPU reference implementation
│   ├── gpu/
│   │   ├── conv_naive.cu              # Basic GPU kernel
│   │   ├── conv_coalesced.cu          # Memory coalescing optimization
│   │   ├── conv_shared.cu             # Shared memory tiling
│   │   ├── conv_constant.cu           # Constant memory optimization
│   │   ├── conv_texture.cu            # Texture memory optimization
│   │   └── conv_async.cu              # Asynchronous memory operations
│   └── main.cpp                       # Driver program and benchmarking
├── tests/
│   ├── unit_tests.cpp                 # Correctness verification
│   └── performance_tests.sh           # Automated benchmarking
├── tools/
│   ├── profile.sh                     # Nsight Compute automation
│   ├── roofline.py                    # Performance analysis scripts
│   └── visualize.py                   # Results plotting
└── docs/
    ├── lab_report_template.md         # Report structure guide
    ├── optimization_guide.md          # Step-by-step optimization hints
    └── profiling_guide.md             # Nsight Compute tutorial
```

## 🎯 Implementation Milestones

### Phase 1: Foundation (Weeks 1-2)
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

## 📚 Learning Resources

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


**"From Naïve to Near-Peak Performance on RTX 4090"**

## 🔍 Project Overview

Transform a simple **2D image convolution** from a slow, basic implementation into a high-performance GPU kernel that approaches the theoretical memory bandwidth of modern GPUs. Through hands-on optimization, you'll master the GPU memory hierarchy and learn to extract maximum performance from parallel hardware.

[Intro to matrix algorithms](https://www.cs.sfu.ca/~ashriram/Courses/CS7ARCH/hw/hw4.html)

**Why This Matters**: In the AI and HPC era, understanding how to optimize memory access patterns is crucial for performance-critical applications. This project builds the foundation for working with neural networks, scientific computing, and any GPU-accelerated workload.

---

## 🎓 Prerequisites and Getting Started Guide

### 📋 Skills Assessment

Before starting this project, you should be comfortable with:
- [ ] **C/C++ programming** (pointers, arrays, basic memory management)
- [ ] **Basic linear algebra** (matrix operations, convolution concept)
- [ ] **Command line usage** (compiling programs, running executables)
- [ ] **Debugging fundamentals** (reading error messages, using print statements)

**If you need review**: Complete a C++ refresher before proceeding.

### 🛠️ Development Environment Setup

**Hardware Requirements:**
- NVIDIA GPU with compute capability 6.0+ (GTX 1060 or newer)
- 8GB+ system RAM
- 20GB+ free disk space

**Software Stack:**
```bash
# On SFU GPU machine
 ssh cs-arch-29.cmpt.sfu.ca
module load CUDA/LIB/12.6
```

**IDE Recommendations:**
- **VS Code** with CUDA extension for syntax highlighting
- **Nsight Systems/Compute** for profiling and debugging
- **Git** for version control

### 🚦 Environment Verification

Test your setup with this simple program:
```cpp
// test_cuda.cu
#include <cuda_runtime.h>
#include <iostream>

__global__ void hello() {
    printf("Hello from GPU thread %d\n", threadIdx.x);
}

int main() {
    hello<<<1, 5>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

Compile and run:
```bash
nvcc test_cuda.cu -o test_cuda
./test_cuda
```

**Expected output**: "Hello from GPU thread 0" through "Hello from GPU thread 4"

---

## 📚 Background Knowledge Building

### Week 1: GPU Architecture Fundamentals

**Learning Goals:**
- Understand the difference between CPU and GPU architecture
- Learn the CUDA programming model
- Grasp basic concepts: threads, blocks, grids

**Study Materials:**
1. **Read**: CUDA C++ Programming Guide, Chapter 1-2 (2-3 hours)
2. **Watch**: ["Intro to Parallel Programming" by Udacity]() - Lessons 1-2 (2 hours)
3. **Practice**: Write simple kernels (vector addition, element-wise operations)

**Hands-On Exercise:**
```cpp
// Practice: Implement vector addition on GPU
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
```

### Week 2: Memory Hierarchy Deep Dive

**Learning Goals:**
- Understand GPU memory types and their characteristics
- Learn about memory coalescing and bank conflicts
- Grasp the importance of memory bandwidth vs. compute

**Study Materials:**
1. **Read**: CUDA Programming Guide, Chapter 5 (Memory) - 3-4 hours
2. **Watch**: ["GPU Memory Hierarchy Explained"](https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/)
3. **Experiment**: Write programs that access memory in different patterns

**Key Concepts to Master:**
- **Global Memory**: Large but slow, shared across all threads
- **Shared Memory**: Fast but small, shared within a block
- **Constant Memory**: Read-only, cached, good for broadcasting
- **Texture Memory**: 2D spatial locality, hardware interpolation

**Memory Hierarchy Diagram to Annotate:**
```
┌─────────────────┐
│   Registers     │ ← 32-bit values per thread
├─────────────────┤
│  L1 Cache       │ ← ~128KB per SM
│  Shared Memory  │ ← Programmer controlled
├─────────────────┤
│     L2 Cache    │ ← 40-108MB (depending on GPU)
├─────────────────┤
│ Global Memory   │ ← GB/s bandwidth, ms latency
│   (GDDR6X)      │
└─────────────────┘
```

---

## 🎯 Detailed Implementation Milestones

### Phase 1: Foundation Building (Weeks 3-4)

| Milestone | Implementation | Expected Performance | Key Learning |
|-----------|---------------|---------------------|--------------|
| **CPU Baseline** | Reference convolution in C++ | 1.0x (baseline) | Algorithm correctness, timing methodology |
| **Naive GPU** | One thread per pixel, global memory only | 2-5x speedup | GPU programming model, memory latency |
| **Coalesced Access** | Stride-aware indexing for cache efficiency | 5-10x speedup | Memory coalescing, cache line utilization |

### Phase 2: Memory Hierarchy Optimization (Weeks 5-6)

| Milestone | Implementation | Expected Performance | Key Learning |
|-----------|---------------|---------------------|--------------|
| **Shared Memory Tiling** | 32x32 tiles with halo regions | 10-20x speedup | On-chip memory management, bank conflicts |
| **Constant Memory** | Filter weights in constant cache | 15-25x speedup | Broadcast-friendly access patterns |
| **Texture Memory** | 2D spatial locality optimization | 20-30x speedup | Hardware-accelerated interpolation |

### Phase 3: Advanced Techniques (Weeks 7-8)

| Milestone | Implementation | Expected Performance | Key Learning |
|-----------|---------------|---------------------|--------------|
| **Async Memory Ops** | Double-buffered cp.async pipelines | 25-35x speedup | Latency hiding, pipeline parallelism |
| **Occupancy Tuning** | Register optimization, block sizing | 30-40x speedup | Resource utilization, wavefront scheduling |

---

## 📁 Complete Starter Repository Structure

```
cuda-convolution-lab/
├── README.md                          # Setup and build instructions
├── Makefile                           # Build automation
├── CMakeLists.txt                     # Alternative build system
├── data/
│   ├── input/
│   │   ├── lenna_512x512.png         # Test images
│   │   ├── nature_1920x1080.png
│   │   └── random_4096x4096.png
│   ├── kernels/
│   │   ├── blur_3x3.txt              # Filter definitions
│   │   ├── sharpen_3x3.txt
│   │   └── edge_sobel_3x3.txt
│   └── reference/                     # Expected outputs
├── src/
│   ├── common/
│   │   ├── utils.h                    # Common utilities
│   │   ├── timer.h                    # CUDA/CPU timing
│   │   ├── image_io.h                 # PNG/JPEG loading
│   │   └── cuda_helpers.h             # Error checking macros
│   ├── cpu/
│   │   └── conv_cpu.cpp               # CPU reference implementation
│   ├── gpu/
│   │   ├── conv_naive.cu              # Basic GPU kernel
│   │   ├── conv_coalesced.cu          # Memory coalescing optimization
│   │   ├── conv_shared.cu             # Shared memory tiling
│   │   ├── conv_constant.cu           # Constant memory for filters
│   │   ├── conv_texture.cu            # Texture memory binding
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

---

## 🛠️ Debugging and Troubleshooting

### Common CUDA Errors and Solutions

**Error: "CUDA out of memory"**
```cpp
// Problem: Allocating too much memory
// Solution: Check allocation sizes and free unused memory

cudaError_t err = cudaMalloc(&d_data, size);
if (err != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
    // Handle gracefully - reduce problem size or use streaming
}
```

**Error: "Invalid device pointer"**
```cpp
// Problem: Passing host pointer to kernel or vice versa
// Solution: Use unified memory or explicit transfers

// Wrong:
float* h_data = new float[size];
kernel<<<blocks, threads>>>(h_data); // Host pointer to kernel!

// Right:
float* d_data;
cudaMalloc(&d_data, size * sizeof(float));
cudaMemcpy(d_data, h_data, size * sizeof(float), cudaMemcpyHostToDevice);
kernel<<<blocks, threads>>>(d_data);
```

**Development Best Practices:**
```cpp
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// Usage:
CUDA_CHECK(cudaMalloc(&d_data, size));
CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));
```

---

## 🔬 Profiling and Performance Analysis

### Essential Nsight Compute Metrics

**Key Performance Indicators:**
```bash
# Memory bandwidth utilization
ncu --metrics dram__throughput.avg.pct_of_peak ./conv_benchmark

# Memory coalescing efficiency  
ncu --metrics smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct ./conv_benchmark

# Occupancy analysis
ncu --metrics sm__warps_active.avg.pct_of_peak ./conv_benchmark

# L1 cache hit rate
ncu --metrics l1tex__hit_rate.pct ./conv_benchmark
```

**Performance Targets by Implementation:**

| Implementation | Speedup vs CPU | Memory Efficiency | L1 Hit Rate |
|---------------|-----------------|-------------------|-------------|
| CPU Baseline  | 1.0x           | ~10%              | ~90%        |
| Naive GPU     | 2-5x           | ~20%              | ~30%        |
| Coalesced     | 5-10x          | ~60%              | ~40%        |
| Shared Memory | 10-20x         | ~70%              | ~60%        |
| Constant Mem  | 15-25x         | ~75%              | ~70%        |
| Texture Mem   | 20-30x         | ~80%              | ~75%        |
| Async/Advanced| 25-40x         | ~85%              | ~80%        |

---

## Lab Report Requirements

### Technical Analysis Components

**1. Performance Methodology (2-3 pages)**
- Hardware specifications and test environment
- Timing methodology and measurement accuracy
- Test image characteristics and kernel sizes
- Baseline performance establishment

**2. Optimization Analysis (4-6 pages)**
- Step-by-step optimization implementation
- Performance improvement quantification
- Bottleneck identification and resolution
- Memory hierarchy utilization analysis

**3. Roofline Model Analysis (1-2 pages)**
```cpp
// Calculate operational intensity for roofline analysis
float ops_per_pixel = kernel_size * kernel_size;
float bytes_per_pixel = sizeof(float) * (kernel_size * kernel_size + 1);
float operational_intensity = ops_per_pixel / bytes_per_pixel;

printf("Operational Intensity: %.2f FLOP/byte\n", operational_intensity);
```

### Visual Documentation Requirements

**Performance Charts:**
- Speedup progression across implementations
- Memory bandwidth utilization trends
- Occupancy vs. performance correlation
- Roofline model positioning

**Code Examples:**
- Before/after optimization comparisons
- Key kernel implementations with annotations
- Profiling command examples and results interpretation

---

## 🎯 Success Criteria and Portfolio Value

### Achievement Levels

**Minimum Viable**
- Working CPU and basic GPU implementations
- 5x+ speedup demonstration
- Basic profiling and analysis
- Functional code with documentation

**Target Performance**
- All core optimizations implemented
- 20x+ speedup achieved
- Comprehensive profiling analysis
- Professional documentation quality

**Exceptional Achievement**
- 30x+ speedup approaching theoretical limits
- Advanced optimization techniques
- Detailed roofline and bottleneck analysis
- Research-quality documentation and insights

### Career and Academic Value

**Industry Skills Demonstrated:**
- **Performance Engineering**: Critical for AI/ML companies
- **Systems Programming**: Essential for backend/infrastructure roles
- **Parallel Computing**: Growing importance across all tech sectors
- **Profiling/Debugging**: Core skill for senior engineering positions

**Graduate School Preparation:**
- Research methodology and technical writing
- Performance analysis and optimization
- Independent learning and problem-solving
- Advanced systems programming experience

**Resume Impact:**
- "Achieved 30x GPU acceleration through memory hierarchy optimization"
- "Proficient in CUDA programming and performance analysis tools"
- "Experience with high-performance computing and parallel algorithms"

This comprehensive project provides practical experience with the same optimization techniques used in production AI frameworks, scientific computing applications, and high-performance graphics systems.

---

## Additional Resources

### Essential Documentation
- **CUDA C++ Programming Guide**: Complete reference for CUDA programming
- **Nsight Compute User Guide**: Profiling and optimization methodology  
- **NVIDIA Developer Blog**: Latest optimization techniques and case studies

### Community and Support
- **NVIDIA Developer Forums**: Official support and community discussions
- **Stack Overflow CUDA tag**: Programming questions and solutions
- **GPU Computing Academic Resources**: Research papers and advanced techniques

### Recommended Reading
- **"Programming Massively Parallel Processors"** by Kirk & Hwu
- **"Professional CUDA C Programming"** by John Cheng
- **NVIDIA GTC Conference Presentations** on memory optimization

This project framework provides students with a structured path from basic GPU programming to advanced performance optimization, building skills directly applicable to modern AI, scientific computing, and high-performance application development.

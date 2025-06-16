# CUDA Convolution Lab - Optimization Guide

This guide provides detailed explanations of GPU optimization strategies for 2D convolution, progressing from naive implementations to highly optimized kernels.

## Table of Contents

1. [Understanding the Problem](#understanding-the-problem)
2. [Memory Hierarchy on GPU](#memory-hierarchy-on-gpu)
3. [Optimization Strategy Roadmap](#optimization-strategy-roadmap)
4. [Implementation Details](#implementation-details)
5. [Performance Analysis](#performance-analysis)
6. [Common Pitfalls](#common-pitfalls)

## Understanding the Problem

### 2D Convolution Basics

2D convolution is a fundamental operation in image processing and computer vision:

```
output[y][x] = Σ Σ input[y+j][x+i] * kernel[j][i]
               j i
```

**Computational Characteristics:**
- **Compute intensity**: 2 FLOPs per output pixel per kernel element
- **Memory access pattern**: Multiple reads per output pixel
- **Parallelism**: Independent computation per output pixel
- **Memory reuse**: High spatial locality in input data

### Performance Challenges

1. **Memory bandwidth limitation**: Each output pixel requires multiple input reads
2. **Cache inefficiency**: Scattered memory access patterns
3. **Low arithmetic intensity**: More memory operations than compute operations
4. **Resource utilization**: Balancing occupancy with resource usage

## Memory Hierarchy on GPU

Understanding GPU memory hierarchy is crucial for optimization:

### Memory Types and Characteristics

| Memory Type | Bandwidth | Latency | Size | Scope | Caching |
|-------------|-----------|---------|------|-------|---------|
| **Global Memory** | ~1TB/s | 400-800 cycles | GB | All threads | L2 cache |
| **Shared Memory** | ~19TB/s | 1-32 cycles | 48-164KB | Thread block | N/A |
| **Constant Memory** | ~1TB/s | 1-10 cycles | 64KB | All threads | Constant cache |
| **Texture Memory** | ~1TB/s | Variable | N/A | All threads | Texture cache |
| **Registers** | ~40TB/s | 1 cycle | 64KB | Thread | N/A |

### Memory Access Patterns

**Coalesced Access:**
```cpp
// Good: consecutive threads access consecutive memory
tid = threadIdx.x + blockIdx.x * blockDim.x;
data = input[tid];  // Coalesced

// Bad: scattered access pattern
data = input[tid * stride];  // Non-coalesced if stride > 1
```

**Bank Conflicts in Shared Memory:**
```cpp
// Good: no bank conflicts
shared_data[threadIdx.x] = data;

// Bad: bank conflicts
shared_data[threadIdx.x * 2] = data;  // 2-way bank conflict
```

## Optimization Strategy Roadmap

### Phase 1: Naive Implementation (Baseline)

**Characteristics:**
- One thread per output pixel
- Direct global memory access
- No memory optimization

**Expected Performance:**
- Low memory bandwidth utilization (~10-20%)
- Poor cache efficiency
- Simple implementation for correctness verification

**Code Pattern:**
```cpp
__global__ void conv_naive(float* input, float* output, float* kernel, 
                          int width, int height, int kernel_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        float sum = 0.0f;
        for (int ky = 0; ky < kernel_size; ky++) {
            for (int kx = 0; kx < kernel_size; kx++) {
                // Direct global memory access
                sum += input[(y+ky)*width + (x+kx)] * kernel[ky*kernel_size + kx];
            }
        }
        output[y*width + x] = sum;
    }
}
```

### Phase 2: Coalesced Memory Access

**Objective:** Improve global memory bandwidth utilization

**Strategy:**
- Organize thread blocks for sequential memory access
- Ensure consecutive threads access consecutive memory locations

**Expected Improvement:** 2-4x speedup

**Implementation Notes:**
```cpp
// Ensure proper thread block organization
dim3 blockSize(16, 16);  // 256 threads per block
dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
              (height + blockSize.y - 1) / blockSize.y);
```

### Phase 3: Shared Memory Tiling

**Objective:** Reduce global memory accesses through data reuse

**Strategy:**
- Load image tiles into shared memory
- Reuse shared data across threads in the same block
- Handle boundary conditions carefully

**Expected Improvement:** 3-8x speedup over naive

**Key Concepts:**
```cpp
__global__ void conv_shared(float* input, float* output, float* kernel,
                           int width, int height, int kernel_size) {
    extern __shared__ float tile[];
    
    // Calculate tile size including padding for kernel
    int tile_size = blockDim.x + kernel_size - 1;
    
    // Load tile into shared memory (with padding)
    // ... tile loading code ...
    
    __syncthreads();
    
    // Compute convolution using shared memory
    // ... convolution code ...
}
```

**Optimization Considerations:**
- **Tile size selection**: Balance shared memory usage with reuse
- **Padding strategy**: Handle kernel overlap between tiles
- **Bank conflict avoidance**: Organize shared memory access patterns

### Phase 4: Constant Memory Optimization

**Objective:** Optimize kernel access through constant memory

**Strategy:**
- Store convolution kernel in constant memory
- Leverage constant cache for broadcast reads

**Expected Improvement:** 1.5-2x speedup for kernel access

**Implementation:**
```cpp
__constant__ float const_kernel[MAX_KERNEL_SIZE * MAX_KERNEL_SIZE];

// Copy kernel to constant memory
cudaMemcpyToSymbol(const_kernel, host_kernel, kernel_size_bytes);
```

### Phase 5: Texture Memory Optimization

**Objective:** Leverage 2D spatial locality and hardware interpolation

**Strategy:**
- Bind input image to texture memory
- Utilize 2D texture cache for spatial locality
- Handle boundary conditions with texture addressing

**Expected Improvement:** 1.5-3x speedup for input access

**Implementation:**
```cpp
texture<float, 2, cudaReadModeElementType> tex_input;

// Bind texture
cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
cudaBindTexture2D(0, tex_input, d_input, channelDesc, width, height, pitch);

// Access in kernel
float value = tex2D(tex_input, x + kx, y + ky);
```

### Phase 6: Advanced Optimizations

**Multiple Outputs per Thread:**
- Compute multiple output pixels per thread
- Improve arithmetic intensity
- Reduce memory access overhead

**Kernel Fusion:**
- Combine multiple operations in single kernel
- Reduce memory bandwidth requirements
- Improve cache utilization

**Asynchronous Processing:**
- Overlap computation with memory transfers
- Use CUDA streams for concurrent execution
- Hide memory latency

## Implementation Details

### Tile Size Selection

**Factors to Consider:**
1. **Shared memory capacity**: Must fit within 48-164KB limit
2. **Occupancy**: More threads vs. more resources per thread
3. **Kernel size**: Larger kernels require more padding
4. **Computational efficiency**: Balance setup cost with computation

**Recommended Tile Sizes:**

| Kernel Size | Tile Size | Shared Memory Usage | Occupancy |
|-------------|-----------|---------------------|-----------|
| 3x3 | 16x16 | ~1.4KB | High |
| 5x5 | 16x16 | ~1.6KB | High |
| 7x7 | 14x14 | ~1.8KB | Good |
| 11x11 | 12x12 | ~2.5KB | Fair |

### Boundary Handling Strategies

**Zero Padding:**
```cpp
float getValue(float* input, int x, int y, int width, int height) {
    if (x >= 0 && x < width && y >= 0 && y < height) {
        return input[y * width + x];
    }
    return 0.0f;  // Zero padding
}
```

**Clamping:**
```cpp
float getValue(float* input, int x, int y, int width, int height) {
    x = max(0, min(x, width - 1));
    y = max(0, min(y, height - 1));
    return input[y * width + x];
}
```

**Mirroring:**
```cpp
float getValue(float* input, int x, int y, int width, int height) {
    if (x < 0) x = -x;
    if (x >= width) x = 2 * width - x - 1;
    if (y < 0) y = -y;
    if (y >= height) y = 2 * height - y - 1;
    return input[y * width + x];
}
```

## Performance Analysis

### Roofline Analysis

**Operational Intensity Calculation:**
```
OI = FLOPs / Bytes Accessed
   = (width * height * kernel_size² * 2) / 
     (width * height * kernel_size² * 4 + width * height * 4)
   ≈ 2 / (4 * kernel_size² / (width * height) + 4)
```

For large images: OI ≈ 0.5 FLOP/Byte

**Memory Bandwidth Requirements:**
- Input accesses: width × height × kernel_size² × 4 bytes
- Kernel accesses: width × height × kernel_size² × 4 bytes  
- Output writes: width × height × 4 bytes

**Performance Bounds:**
- Memory bound: BW_memory × OI
- Compute bound: Peak_FLOPS

### Profiling Checklist

**Key Metrics to Monitor:**
1. **Kernel execution time**
2. **Memory throughput** (GB/s)
3. **Achieved occupancy** (%)
4. **Cache hit rates** (L1, L2, constant, texture)
5. **Memory efficiency** (coalescing %)
6. **Compute utilization** (%)

**Tools:**
- NVIDIA Nsight Compute for detailed metrics
- NVIDIA Nsight Systems for timeline analysis
- nvprof for quick profiling (legacy)

## Common Pitfalls

### Memory-Related Issues

**Bank Conflicts:**
```cpp
// Bad: causes bank conflicts
__shared__ float tile[16][16];
tile[threadIdx.y][threadIdx.x * 2] = data;  // 2-way conflict

// Good: no bank conflicts
tile[threadIdx.y][threadIdx.x] = data;
```

**Non-Coalesced Access:**
```cpp
// Bad: strided access pattern
int idx = threadIdx.x * stride;
data = input[idx];

// Good: consecutive access
int idx = threadIdx.x;
data = input[idx];
```

**Shared Memory Overuse:**
```cpp
// Bad: exceeds shared memory limit
__shared__ float huge_tile[64][64];  // 16KB, may limit occupancy

// Good: balanced resource usage
__shared__ float tile[16][16];       // 1KB, good occupancy
```

### Algorithmic Issues

**Incorrect Boundary Handling:**
```cpp
// Bad: may access out-of-bounds memory
for (int ky = 0; ky < kernel_size; ky++) {
    int y_idx = y + ky;  // No bounds checking
    data = input[y_idx * width + x];
}

// Good: proper bounds checking
for (int ky = 0; ky < kernel_size; ky++) {
    int y_idx = y + ky;
    if (y_idx >= 0 && y_idx < height) {
        data = input[y_idx * width + x];
    }
}
```

**Inefficient Thread Organization:**
```cpp
// Bad: poor occupancy
dim3 blockSize(8, 8);   // Only 64 threads per block

// Good: better occupancy
dim3 blockSize(16, 16); // 256 threads per block
```

### Performance Issues

**Resource Overuse:**
- Too many registers per thread
- Excessive shared memory usage
- Suboptimal block size selection

**Insufficient Parallelism:**
- Small problem sizes
- Poor load balancing
- Synchronization bottlenecks

### Debugging Strategies

**Correctness Verification:**
1. Start with CPU reference implementation
2. Test with simple kernels (identity, box filter)
3. Verify boundary conditions separately
4. Use small problem sizes for debugging

**Performance Debugging:**
1. Profile each optimization incrementally
2. Compare against theoretical peaks
3. Analyze memory access patterns
4. Monitor occupancy and resource usage

## Conclusion

Optimizing CUDA convolution requires understanding:
1. **Memory hierarchy** and access patterns
2. **Thread organization** and resource utilization
3. **Algorithmic considerations** for parallelization
4. **Profiling techniques** for performance analysis

The progression from naive to optimized implementations demonstrates the impact of memory hierarchy exploitation, with potential speedups of 10-20x over naive GPU implementations and 50-200x over CPU implementations.

Success requires balancing multiple factors:
- Memory bandwidth vs. compute utilization
- Resource usage vs. occupancy
- Implementation complexity vs. performance gains
- Portability vs. architecture-specific optimization

Continue to the [Profiling Guide](profiling_guide.md) for detailed performance analysis techniques.

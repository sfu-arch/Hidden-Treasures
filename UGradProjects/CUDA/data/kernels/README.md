# Convolution Kernels for CUDA Lab

This directory contains predefined convolution kernels for testing different image processing operations.

## Available Kernels

### Basic Kernels

#### Identity (3x3)
```
0 0 0
0 1 0
0 0 0
```
- **Purpose**: No-op convolution (should preserve input)
- **Use case**: Testing correctness and performance baseline

#### Box Blur (5x5)
```
1/25 1/25 1/25 1/25 1/25
1/25 1/25 1/25 1/25 1/25
1/25 1/25 1/25 1/25 1/25
1/25 1/25 1/25 1/25 1/25
1/25 1/25 1/25 1/25 1/25
```
- **Purpose**: Simple averaging filter
- **Use case**: Basic smoothing, easy to verify results

#### Gaussian Blur (5x5, σ=1.0)
```
0.003 0.013 0.022 0.013 0.003
0.013 0.059 0.097 0.059 0.013
0.022 0.097 0.159 0.097 0.022
0.013 0.059 0.097 0.059 0.013
0.003 0.013 0.022 0.013 0.003
```
- **Purpose**: Smooth blurring with natural falloff
- **Use case**: Noise reduction, general smoothing

### Edge Detection Kernels

#### Sobel X (3x3)
```
-1  0  1
-2  0  2
-1  0  1
```
- **Purpose**: Horizontal edge detection
- **Use case**: Feature detection, gradient computation

#### Sobel Y (3x3)
```
-1 -2 -1
 0  0  0
 1  2  1
```
- **Purpose**: Vertical edge detection
- **Use case**: Feature detection, gradient computation

#### Laplacian (3x3)
```
 0 -1  0
-1  4 -1
 0 -1  0
```
- **Purpose**: General edge detection
- **Use case**: Edge enhancement, sharpening

### Enhancement Kernels

#### Sharpen (3x3)
```
 0 -1  0
-1  5 -1
 0 -1  0
```
- **Purpose**: Image sharpening
- **Use case**: Detail enhancement

#### Unsharp Mask (5x5)
```
-1/256 -4/256  -6/256  -4/256 -1/256
-4/256 -16/256 -24/256 -16/256 -4/256
-6/256 -24/256 476/256 -24/256 -6/256
-4/256 -16/256 -24/256 -16/256 -4/256
-1/256 -4/256  -6/256  -4/256 -1/256
```
- **Purpose**: Advanced sharpening
- **Use case**: Professional image enhancement

### Large Kernels for Performance Testing

#### Gaussian 11x11 (σ=2.0)
- **Purpose**: Large smoothing kernel
- **Use case**: Performance testing with larger memory footprint

#### Gaussian 15x15 (σ=3.0)
- **Purpose**: Very large smoothing kernel
- **Use case**: Stress testing memory bandwidth

## File Formats

Kernels are stored in simple text format:
- First line: kernel size (assumed square)
- Following lines: kernel values in row-major order
- Values separated by spaces or tabs
- Comments start with '#'

Example file format:
```
# Gaussian 3x3 kernel, sigma=1.0
3
0.075 0.124 0.075
0.124 0.204 0.124
0.075 0.124 0.075
```

## Usage

Load kernels in your program:
```cpp
// C++ example
std::vector<float> kernel;
int kernel_size;
if (image_io::loadKernel("data/kernels/gaussian_5x5.txt", kernel, kernel_size)) {
    // Use kernel for convolution
}
```

Or generate kernels programmatically:
```cpp
// Generate Gaussian kernel
std::vector<float> kernel(5 * 5);
image_io::generateKernel(kernel.data(), 5, image_io::KernelType::GAUSSIAN);
```

## Performance Characteristics

### Memory Usage by Kernel Size

| Kernel Size | Elements | Memory (bytes) | Cache Efficiency |
|-------------|----------|----------------|------------------|
| 3x3 | 9 | 36 | Excellent |
| 5x5 | 25 | 100 | Good |
| 7x7 | 49 | 196 | Fair |
| 11x11 | 121 | 484 | Poor |
| 15x15 | 225 | 900 | Very Poor |

### Computational Complexity

For an NxN image and KxK kernel:
- **Operations**: N² × K² multiply-add operations
- **Memory accesses**: ~N² × K² reads + N² writes
- **Operational intensity**: 2 FLOPs per memory access (for large images)

### Optimization Considerations

1. **Small kernels (3x3, 5x5)**: 
   - Fit easily in constant memory
   - Good cache locality
   - Suitable for all optimization strategies

2. **Medium kernels (7x7, 9x9)**:
   - May exceed constant memory limits
   - Shared memory becomes more beneficial
   - Consider kernel separability

3. **Large kernels (11x11+)**:
   - Require careful memory management
   - Shared memory tiling essential
   - Consider FFT-based convolution for very large kernels

## Adding Custom Kernels

To add your own kernels:

1. Create a text file with the kernel values
2. Follow the format specification above
3. Place in this directory
4. Test with small images first
5. Update this README with kernel description

## References

- [Image Processing Kernels](https://en.wikipedia.org/wiki/Kernel_(image_processing))
- [Gaussian Kernels](https://homepages.inf.ed.ac.uk/rbf/HIPR2/gsmooth.htm)
- [Edge Detection Operators](https://homepages.inf.ed.ac.uk/rbf/HIPR2/sobel.htm)

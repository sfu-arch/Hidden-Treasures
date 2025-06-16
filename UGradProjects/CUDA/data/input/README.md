# Sample Images for CUDA Convolution Lab

This directory contains sample images for testing the convolution implementations.

## Included Images

### Test Patterns
- `checkerboard_512.png` - 512x512 checkerboard pattern for testing
- `gradient_1024.png` - 1024x1024 gradient pattern
- `noise_256.png` - 256x256 random noise pattern

### Real Images
- `lena_512.png` - Classic Lena test image (512x512)
- `barbara_512.png` - Barbara test image with fine details
- `cameraman_256.png` - Cameraman test image (256x256)

## Image Formats

All images are provided in PNG format with:
- 8-bit grayscale
- No compression artifacts
- Square dimensions for consistent testing

## Usage

The main program can automatically generate test patterns if no input image is specified:

```bash
# Use automatic test pattern
./cuda_convolution --width 1024 --height 1024

# Use specific input image
./cuda_convolution --input data/input/lena_512.png
```

## Adding New Images

To add your own test images:

1. Convert to grayscale PNG format
2. Ensure power-of-2 dimensions for optimal GPU performance
3. Place in this directory
4. Update this README with the new image description

## Performance Considerations

Image sizes and their expected performance characteristics:

| Size | Pixels | Memory (MB) | Expected GPU Time (ms) |
|------|--------|-------------|------------------------|
| 256x256 | 65K | 0.25 | <1 |
| 512x512 | 262K | 1.0 | 1-5 |
| 1024x1024 | 1M | 4.0 | 5-20 |
| 2048x2048 | 4M | 16.0 | 20-80 |
| 4096x4096 | 16M | 64.0 | 80-300 |

Note: Actual performance depends on GPU architecture and optimization level.

#!/bin/bash

# CUDA Convolution Lab - Performance Testing Script
# This script runs comprehensive performance tests across different
# image sizes, kernel sizes, and implementations

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default parameters
EXECUTABLE="./build/cuda_convolution"
OUTPUT_DIR="./performance_results"
VERBOSE=false
GPU_ONLY=false
CPU_ONLY=false

# Test configurations
IMAGE_SIZES=(256 512 1024 2048 4096)
KERNEL_SIZES=(3 5 7 9 11 15)
WARMUP_RUNS=2
BENCHMARK_RUNS=5

print_header() {
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}    CUDA Convolution Lab - Performance Tests    ${NC}"
    echo -e "${BLUE}================================================${NC}"
    echo ""
}

print_usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -e, --executable <path>   Path to convolution executable (default: $EXECUTABLE)"
    echo "  -o, --output <dir>        Output directory for results (default: $OUTPUT_DIR)"
    echo "  -v, --verbose             Enable verbose output"
    echo "  -c, --cpu-only            Run CPU tests only"
    echo "  -g, --gpu-only            Run GPU tests only"
    echo "  -s, --size <size>         Test single image size only"
    echo "  -k, --kernel <size>       Test single kernel size only"
    echo "  -r, --runs <num>          Number of benchmark runs (default: $BENCHMARK_RUNS)"
    echo "  -h, --help                Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --verbose --output ./my_results"
    echo "  $0 --gpu-only --size 2048 --kernel 7"
    echo "  $0 --cpu-only --runs 10"
}

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if executable exists
    if [ ! -f "$EXECUTABLE" ]; then
        log_error "Executable not found: $EXECUTABLE"
        echo "Please build the project first:"
        echo "  make"
        echo "or"
        echo "  mkdir build && cd build && cmake .. && make"
        exit 1
    fi
    
    # Check if executable is runnable
    if [ ! -x "$EXECUTABLE" ]; then
        log_error "Executable is not executable: $EXECUTABLE"
        exit 1
    fi
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    if [ $? -ne 0 ]; then
        log_error "Failed to create output directory: $OUTPUT_DIR"
        exit 1
    fi
    
    log_info "Prerequisites check passed"
}

run_warmup() {
    local size=$1
    local kernel=$2
    
    if [ "$VERBOSE" = true ]; then
        log_info "Running warmup for size=${size}, kernel=${kernel}..."
    fi
    
    for ((i=1; i<=WARMUP_RUNS; i++)); do
        $EXECUTABLE --width $size --height $size --kernel $kernel \
                   --output "$OUTPUT_DIR/warmup" >/dev/null 2>&1
    done
}

run_benchmark() {
    local size=$1
    local kernel=$2
    local test_type=$3
    local result_file=$4
    
    log_info "Testing ${test_type}: ${size}x${size} image, ${kernel}x${kernel} kernel"
    
    # Run warmup
    run_warmup $size $kernel
    
    # Prepare command arguments
    local cmd_args="--width $size --height $size --kernel $kernel --output $OUTPUT_DIR/temp"
    
    if [ "$test_type" = "CPU" ]; then
        cmd_args="$cmd_args --cpu-only"
    elif [ "$test_type" = "GPU" ]; then
        cmd_args="$cmd_args --gpu-only"
    fi
    
    if [ "$VERBOSE" = true ]; then
        cmd_args="$cmd_args --verbose"
    fi
    
    # Run benchmark iterations
    local total_time=0
    local best_time=999999
    local worst_time=0
    
    echo "# Benchmark Results: $test_type" >> "$result_file"
    echo "# Image Size: ${size}x${size}" >> "$result_file"
    echo "# Kernel Size: ${kernel}x${kernel}" >> "$result_file"
    echo "# Timestamp: $(date)" >> "$result_file"
    echo "Run,Time(ms),GFLOPS,Bandwidth(GB/s)" >> "$result_file"
    
    for ((i=1; i<=BENCHMARK_RUNS; i++)); do
        if [ "$VERBOSE" = true ]; then
            echo "  Run $i/$BENCHMARK_RUNS..."
        fi
        
        # Run the benchmark and capture output
        local output=$($EXECUTABLE $cmd_args 2>&1)
        local exit_code=$?
        
        if [ $exit_code -ne 0 ]; then
            log_error "Benchmark run $i failed with exit code $exit_code"
            continue
        fi
        
        # Extract timing information (this would need to be adapted based on actual output format)
        local time_ms=$(echo "$output" | grep -o "took: [0-9.]*" | grep -o "[0-9.]*" | head -1)
        local gflops=$(echo "$output" | grep -o "GFLOPS: [0-9.]*" | grep -o "[0-9.]*" | head -1)
        local bandwidth=$(echo "$output" | grep -o "Bandwidth: [0-9.]*" | grep -o "[0-9.]*" | head -1)
        
        if [ -n "$time_ms" ]; then
            echo "$i,$time_ms,$gflops,$bandwidth" >> "$result_file"
            
            # Update statistics
            total_time=$(echo "$total_time + $time_ms" | bc -l)
            if (( $(echo "$time_ms < $best_time" | bc -l) )); then
                best_time=$time_ms
            fi
            if (( $(echo "$time_ms > $worst_time" | bc -l) )); then
                worst_time=$time_ms
            fi
        fi
    done
    
    # Calculate average
    local avg_time=$(echo "scale=3; $total_time / $BENCHMARK_RUNS" | bc -l)
    
    echo "" >> "$result_file"
    echo "# Summary Statistics" >> "$result_file"
    echo "Average Time: $avg_time ms" >> "$result_file"
    echo "Best Time: $best_time ms" >> "$result_file"
    echo "Worst Time: $worst_time ms" >> "$result_file"
    echo "" >> "$result_file"
    
    if [ "$VERBOSE" = true ]; then
        echo "  Average: ${avg_time}ms, Best: ${best_time}ms, Worst: ${worst_time}ms"
    fi
}

run_scalability_test() {
    log_info "Running scalability tests..."
    
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local scalability_file="$OUTPUT_DIR/scalability_${timestamp}.csv"
    
    echo "ImageSize,KernelSize,TestType,AvgTime(ms),BestTime(ms),GFLOPS,Speedup" > "$scalability_file"
    
    for size in "${IMAGE_SIZES[@]}"; do
        for kernel in "${KERNEL_SIZES[@]}"; do
            # Skip very large combinations that might be too slow
            if [ $size -gt 2048 ] && [ $kernel -gt 9 ]; then
                log_warning "Skipping ${size}x${size} with ${kernel}x${kernel} kernel (too large)"
                continue
            fi
            
            local result_file="$OUTPUT_DIR/result_${size}_${kernel}_${timestamp}.csv"
            
            if [ "$CPU_ONLY" = false ]; then
                run_benchmark $size $kernel "GPU" "$result_file"
            fi
            
            if [ "$GPU_ONLY" = false ]; then
                run_benchmark $size $kernel "CPU" "$result_file"
            fi
        done
    done
    
    log_info "Scalability test completed. Results saved to $scalability_file"
}

run_roofline_analysis() {
    log_info "Running roofline analysis..."
    
    # This would require additional implementation in the main program
    # to output operational intensity and peak performance data
    
    local roofline_file="$OUTPUT_DIR/roofline_data.csv"
    echo "Implementation,OperationalIntensity,AchievedGFLOPS,MemoryBandwidth" > "$roofline_file"
    
    # Run tests with specific configurations for roofline analysis
    for size in 1024 2048; do
        for kernel in 5 11; do
            log_info "Collecting roofline data for ${size}x${size}, kernel ${kernel}x${kernel}"
            
            # This would need integration with the main program to output
            # the required metrics in a parseable format
            $EXECUTABLE --width $size --height $size --kernel $kernel \
                       --output "$OUTPUT_DIR/roofline_temp" --verbose \
                       >> "$roofline_file" 2>&1
        done
    done
    
    log_info "Roofline analysis data collected in $roofline_file"
}

generate_report() {
    log_info "Generating performance report..."
    
    local report_file="$OUTPUT_DIR/performance_report.md"
    local timestamp=$(date)
    
    cat > "$report_file" << EOF
# CUDA Convolution Lab - Performance Report

Generated: $timestamp

## System Information

$(uname -a)

## Test Configuration

- Executable: $EXECUTABLE
- Warmup runs: $WARMUP_RUNS
- Benchmark runs: $BENCHMARK_RUNS
- Image sizes tested: ${IMAGE_SIZES[*]}
- Kernel sizes tested: ${KERNEL_SIZES[*]}

## Results Summary

### Performance Data Files

EOF

    # List all CSV files generated
    for file in "$OUTPUT_DIR"/*.csv; do
        if [ -f "$file" ]; then
            echo "- $(basename "$file")" >> "$report_file"
        fi
    done
    
    cat >> "$report_file" << EOF

### Key Findings

(This section would be populated by analyzing the results)

1. **GPU vs CPU Performance**: 
   - Analysis of speedup across different problem sizes

2. **Memory Hierarchy Impact**:
   - Effect of different optimization strategies

3. **Scalability Analysis**:
   - Performance scaling with image and kernel sizes

### Recommendations

1. Use GPU implementations for images larger than NxN pixels
2. Optimal kernel sizes for different use cases
3. Memory optimization strategies

## Next Steps

- Implement additional GPU optimizations (shared memory, texture memory)
- Profile with NVIDIA NSight for detailed analysis
- Compare with vendor-optimized libraries (cuDNN, OpenCV)

EOF

    log_info "Performance report generated: $report_file"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--executable)
            EXECUTABLE="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -c|--cpu-only)
            CPU_ONLY=true
            shift
            ;;
        -g|--gpu-only)
            GPU_ONLY=true
            shift
            ;;
        -s|--size)
            IMAGE_SIZES=("$2")
            shift 2
            ;;
        -k|--kernel)
            KERNEL_SIZES=("$2")
            shift 2
            ;;
        -r|--runs)
            BENCHMARK_RUNS="$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Main execution
main() {
    print_header
    
    log_info "Starting performance testing..."
    log_info "Output directory: $OUTPUT_DIR"
    log_info "Benchmark runs per test: $BENCHMARK_RUNS"
    
    check_prerequisites
    
    # Run the test suite
    run_scalability_test
    
    # Additional analyses
    if [ "$VERBOSE" = true ]; then
        run_roofline_analysis
    fi
    
    # Generate final report
    generate_report
    
    log_info "Performance testing completed!"
    log_info "Results available in: $OUTPUT_DIR"
}

# Run main function
main "$@"

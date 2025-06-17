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
EXECUTABLE="./Build/conv_benchmark"
OUTPUT_DIR="./performance_results"
VERBOSE=false
GPU_ONLY=false
CPU_ONLY=false

# Test configurations (adjusted for current hardware capabilities)
IMAGE_SIZES=(256 512 1024 2048)
KERNEL_SIZES=(3 5 7 9 11)
WARMUP_RUNS=3
BENCHMARK_RUNS=5

# Profiling integration
PROFILING_ENABLED=false
PROFILE_SCRIPT="./tools/profile.sh"

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
    echo "  -c, --cpu-only            Run CPU performance tests only"
    echo "  -g, --gpu-only            Run GPU performance tests only"
    echo "  -s, --size <size>         Test single image size only"
    echo "  -k, --kernel <size>       Test single kernel size only"
    echo "  -r, --runs <num>          Number of benchmark runs (default: $BENCHMARK_RUNS)"
    echo "  -p, --profile             Enable detailed profiling with Nsight tools"
    echo "  --quick                   Run quick test with small sizes only"
    echo "  --correctness             Run correctness verification tests (always runs both CPU+GPU)"
    echo "  -h, --help                Show this help message"
    echo ""
    echo "Note: --cpu-only and --gpu-only affect performance benchmarks only."
    echo "      Correctness tests always run both CPU and GPU for comparison."
    echo ""
    echo "Examples:"
    echo "  $0 --verbose --output ./my_results"
    echo "  $0 --gpu-only --size 1024 --kernel 5 --profile"
    echo "  $0 --quick --correctness"
    echo "  $0 --cpu-only --runs 10"
    echo "  $0 --gpu-only --correctness  # Still runs CPU+GPU for correctness"
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
        echo "  mkdir Build && cd Build"
        echo "  cmake .. -DCMAKE_CUDA_ARCHITECTURES=89"
        echo "  make"
        exit 1
    fi
    
    # Check if executable is runnable
    if [ ! -x "$EXECUTABLE" ]; then
        log_error "Executable is not executable: $EXECUTABLE"
        exit 1
    fi
    
    # Check CUDA setup
    if ! command -v nvidia-smi &> /dev/null; then
        log_warning "nvidia-smi not found. GPU tests may fail."
    elif [ "$GPU_ONLY" = true ] || [ "$GPU_ONLY" = false ]; then
        # Quick GPU check
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits > /dev/null 2>&1
        if [ $? -ne 0 ]; then
            log_warning "GPU not accessible. GPU tests may fail."
        else
            log_info "GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
        fi
    fi
    
    # Check profiling tools if profiling enabled
    if [ "$PROFILING_ENABLED" = true ]; then
        if [ ! -f "$PROFILE_SCRIPT" ]; then
            log_warning "Profiling script not found: $PROFILE_SCRIPT"
            log_warning "Disabling profiling for this run"
            PROFILING_ENABLED=false
        elif ! command -v nsys &> /dev/null; then
            log_warning "Nsight Systems (nsys) not found. Profiling may be limited."
        fi
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
    local successful_runs=0
    
    echo "# Benchmark Results: $test_type" >> "$result_file"
    echo "# Image Size: ${size}x${size}" >> "$result_file"
    echo "# Kernel Size: ${kernel}x${kernel}" >> "$result_file"
    echo "# Timestamp: $(date)" >> "$result_file"
    echo "Run,Time(ms),GFLOPS,Bandwidth(GB/s),Status" >> "$result_file"
    
    for ((i=1; i<=BENCHMARK_RUNS; i++)); do
        if [ "$VERBOSE" = true ]; then
            echo "  Run $i/$BENCHMARK_RUNS..."
        fi

        echo "$EXECUTABLE $cmd_args"
        # Run the benchmark and capture output
        
        local output=$($EXECUTABLE $cmd_args 2>&1)
        local exit_code=$?
        
        if [ $exit_code -ne 0 ]; then
            log_error "Benchmark run $i failed with exit code $exit_code"
            echo "$i,ERROR,ERROR,ERROR,FAILED" >> "$result_file"
            continue
        fi
        
        # Extract timing information from current output format
        local time_ms=$(echo "$output" | grep -o "took: [0-9.]*" | grep -o "[0-9.]*" | head -1)
        local gflops=$(echo "$output" | grep -o "Throughput: [0-9.]*" | grep -o "[0-9.]*" | head -1)
        local bandwidth=$(echo "$output" | grep -o "Memory Bandwidth: [0-9.]*" | grep -o "[0-9.]*" | head -1)
        
        # Check for correctness
        local status="SUCCESS"
        if echo "$output" | grep -q "INCORRECT"; then
            status="INCORRECT"
            log_warning "Run $i produced incorrect results"
        fi
        
        if [ -n "$time_ms" ] && [ "$time_ms" != "0" ]; then
            echo "$i,$time_ms,$gflops,$bandwidth,$status" >> "$result_file"
            
            # Update statistics only for successful runs
            if [ "$status" = "SUCCESS" ]; then
                total_time=$(echo "$total_time + $time_ms" | bc -l)
                successful_runs=$((successful_runs + 1))
                
                if (( $(echo "$time_ms < $best_time" | bc -l) )); then
                    best_time=$time_ms
                fi
                if (( $(echo "$time_ms > $worst_time" | bc -l) )); then
                    worst_time=$time_ms
                fi
            fi
        else
            echo "$i,ERROR,ERROR,ERROR,PARSE_ERROR" >> "$result_file"
            log_warning "Failed to parse timing data from run $i"
        fi
    done
    
    # Calculate statistics
    if [ $successful_runs -gt 0 ]; then
        local avg_time=$(echo "scale=3; $total_time / $successful_runs" | bc -l)
        
        echo "" >> "$result_file"
        echo "# Summary Statistics" >> "$result_file"
        echo "Successful Runs: $successful_runs/$BENCHMARK_RUNS" >> "$result_file"
        echo "Average Time: $avg_time ms" >> "$result_file"
        echo "Best Time: $best_time ms" >> "$result_file"
        echo "Worst Time: $worst_time ms" >> "$result_file"
        echo "" >> "$result_file"
        
        if [ "$VERBOSE" = true ]; then
            echo "  Average: ${avg_time}ms, Best: ${best_time}ms, Worst: ${worst_time}ms ($successful_runs/$BENCHMARK_RUNS successful)"
        fi
    else
        log_error "All benchmark runs failed for ${test_type} ${size}x${size} kernel ${kernel}"
        echo "ERROR: All runs failed" >> "$result_file"
    fi
}

run_correctness_test() {
    log_info "Running correctness verification tests..."
    
    # Correctness verification always needs both CPU and GPU results for comparison
    # This is independent of --gpu-only or --cpu-only flags which only affect performance benchmarks
    if [ "$GPU_ONLY" = true ]; then
        log_info "Note: Running both CPU and GPU for correctness verification despite --gpu-only flag"
    elif [ "$CPU_ONLY" = true ]; then
        log_info "Note: Running both CPU and GPU for correctness verification despite --cpu-only flag"
    fi
    
    local correctness_file="$OUTPUT_DIR/correctness_results.txt"
    echo "CUDA Convolution Lab - Correctness Test Results" > "$correctness_file"
    echo "Generated: $(date)" >> "$correctness_file"
    echo "" >> "$correctness_file"
    
    local test_sizes=(256 512)
    local test_kernels=(3 5)
    local total_tests=0
    local passed_tests=0
    
    for size in "${test_sizes[@]}"; do
        for kernel in "${test_kernels[@]}"; do
            total_tests=$((total_tests + 1))
            
            log_info "Testing correctness: ${size}x${size} image, ${kernel}x${kernel} kernel"
            
            # Always run with both CPU and GPU for correctness comparison
            # This ensures we can verify GPU results against CPU reference
            local output=$($EXECUTABLE --width $size --height $size --kernel $kernel \
                          --output "$OUTPUT_DIR/correctness_temp" 2>&1)
            local exit_code=$?
            
            echo "Test ${total_tests}: ${size}x${size} image, ${kernel}x${kernel} kernel" >> "$correctness_file"
            
            if [ $exit_code -eq 0 ]; then
                # Look for the correctness comparison in the output
                if echo "$output" | grep -q "GPU.*result: CORRECT"; then
                    echo "  Status: PASSED (GPU matches CPU reference)" >> "$correctness_file"
                    passed_tests=$((passed_tests + 1))
                    log_info "  ✅ PASSED"
                elif echo "$output" | grep -q "GPU.*result: INCORRECT"; then
                    echo "  Status: FAILED (GPU does not match CPU reference)" >> "$correctness_file"
                    log_error "  ❌ FAILED - Incorrect result"
                else
                    echo "  Status: UNKNOWN (Could not determine result from output)" >> "$correctness_file"
                    log_warning "  ⚠️ UNKNOWN - Could not determine result"
                    
                    # Add debug information to the file
                    if [ "$VERBOSE" = true ]; then
                        echo "  Debug output:" >> "$correctness_file"
                        echo "$output" | head -10 >> "$correctness_file"
                    fi
                fi
            else
                echo "  Status: ERROR (Exit code: $exit_code)" >> "$correctness_file"
                log_error "  ❌ ERROR - Exit code: $exit_code"
                
                # Add error output for debugging
                if [ "$VERBOSE" = true ]; then
                    echo "  Error output:" >> "$correctness_file"
                    echo "$output" >> "$correctness_file"
                fi
            fi
            
            echo "" >> "$correctness_file"
        done
    done
    
    echo "Summary: $passed_tests/$total_tests tests passed" >> "$correctness_file"
    log_info "Correctness testing completed: $passed_tests/$total_tests tests passed"
    log_info "Detailed results saved to: $correctness_file"
}

run_profiling_analysis() {
    log_info "Running profiling analysis..."
    
    if [ "$PROFILING_ENABLED" = false ]; then
        log_warning "Profiling not enabled. Use --profile flag to enable."
        return
    fi
    
    if [ ! -f "$PROFILE_SCRIPT" ]; then
        log_error "Profiling script not found: $PROFILE_SCRIPT"
        return
    fi
    
    # Run profiling on a representative test case
    local profile_size=512
    local profile_kernel=5
    
    log_info "Running detailed profiling: ${profile_size}x${profile_size} image, ${profile_kernel}x${profile_kernel} kernel"
    
    # Debug: Show the exact command being executed
    local profile_cmd="$PROFILE_SCRIPT --executable \"$EXECUTABLE\" --mode detailed --size $profile_size --kernel $profile_kernel"
    if [ "$VERBOSE" = true ]; then
        log_info "Profile command: $profile_cmd"
    fi
    
    # Run the profiling script with proper executable path using bash explicitly
    bash "$PROFILE_SCRIPT" --executable "$EXECUTABLE" --mode detailed --size $profile_size --kernel $profile_kernel
    
    if [ $? -eq 0 ]; then
        log_info "Profiling completed successfully"
        
        # Copy profiling results to performance results directory
        if [ -d "./profiling_results" ]; then
            cp -r ./profiling_results/* "$OUTPUT_DIR/" 2>/dev/null || true
            log_info "Profiling results copied to: $OUTPUT_DIR"
        fi
    else
        log_error "Profiling failed"
    fi
}

run_quick_test() {
    log_info "Running quick performance test..."
    
    # Override test configurations for quick testing
    local quick_sizes=(256 512)
    local quick_kernels=(3 5)
    local original_runs=$BENCHMARK_RUNS
    BENCHMARK_RUNS=3
    
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local quick_file="$OUTPUT_DIR/quick_test_${timestamp}.csv"
    
    echo "ImageSize,KernelSize,TestType,AvgTime(ms),Status" > "$quick_file"
    
    for size in "${quick_sizes[@]}"; do
        for kernel in "${quick_kernels[@]}"; do
            local result_file="$OUTPUT_DIR/quick_${size}_${kernel}_${timestamp}.csv"
            
            if [ "$CPU_ONLY" = false ]; then
                run_benchmark $size $kernel "GPU" "$result_file"
                
                # Extract average time from result file
                local gpu_avg=$(grep "Average Time:" "$result_file" | cut -d: -f2 | xargs)
                local gpu_status="SUCCESS"
                if grep -q "INCORRECT" "$result_file"; then
                    gpu_status="INCORRECT"
                elif grep -q "ERROR" "$result_file"; then
                    gpu_status="ERROR"
                fi
                echo "$size,$kernel,GPU,$gpu_avg,$gpu_status" >> "$quick_file"
            fi
            
            if [ "$GPU_ONLY" = false ]; then
                run_benchmark $size $kernel "CPU" "$result_file"
                
                # Extract average time from result file
                local cpu_avg=$(grep "Average Time:" "$result_file" | cut -d: -f2 | xargs)
                echo "$size,$kernel,CPU,$cpu_avg,SUCCESS" >> "$quick_file"
            fi
        done
    done
    
    # Restore original benchmark runs
    BENCHMARK_RUNS=$original_runs
    
    log_info "Quick test completed. Results saved to: $quick_file"
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

**Hardware:**
$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "GPU information not available")

**Software:**
- CUDA Version: $(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release //' | cut -d, -f1 || echo "Not available")
- System: $(uname -a)

## Test Configuration

- Executable: $EXECUTABLE
- Warmup runs: $WARMUP_RUNS
- Benchmark runs: $BENCHMARK_RUNS
- Image sizes tested: ${IMAGE_SIZES[*]}
- Kernel sizes tested: ${KERNEL_SIZES[*]}
- CPU only: $CPU_ONLY
- GPU only: $GPU_ONLY
- Profiling enabled: $PROFILING_ENABLED

## Current Implementation Status

### ✅ Implemented Components
- **CPU Reference**: Sequential, OpenMP, and optimized implementations
- **GPU Naive**: Basic one-thread-per-pixel kernel (⚠️ correctness issues detected)
- **Build System**: CMake with CUDA 12.6 support
- **Profiling Infrastructure**: Nsight Systems/Compute integration

### 🚨 Known Issues  
- **GPU Correctness**: GPU implementation produces incorrect results
- **Memory Allocation**: 99.8% of GPU time spent in cudaMalloc (major bottleneck)
- **Memory Bandwidth**: Using only 0.006% of theoretical peak bandwidth

### ⏳ Planned Optimizations
- Memory coalescing optimization
- Shared memory tiling  
- Constant memory for kernel storage
- Texture memory optimization
- Asynchronous memory operations

## Results Summary

### Performance Data Files

EOF

    # List all CSV files generated
    for file in "$OUTPUT_DIR"/*.csv; do
        if [ -f "$file" ]; then
            echo "- [$(basename "$file")]($(basename "$file"))" >> "$report_file"
        fi
    done
    
    # Add correctness results if available
    if [ -f "$OUTPUT_DIR/correctness_results.txt" ]; then
        echo "- [correctness_results.txt](correctness_results.txt)" >> "$report_file"
    fi
    
    cat >> "$report_file" << EOF

### Key Findings

EOF

    # Analyze quick test results if available
    local latest_quick=$(ls -t "$OUTPUT_DIR"/quick_test_*.csv 2>/dev/null | head -1)
    if [ -n "$latest_quick" ] && [ -f "$latest_quick" ]; then
        cat >> "$report_file" << EOF
#### Performance Comparison (from quick test)

| Configuration | CPU Time (ms) | GPU Time (ms) | GPU Speedup | GPU Status |
|---------------|---------------|---------------|-------------|------------|
EOF
        
        # Extract data from quick test results
        while IFS=, read -r size kernel testtype avgtime status; do
            if [[ "$size" =~ ^[0-9]+$ ]] && [ "$testtype" = "GPU" ]; then
                local cpu_time=$(grep "$size,$kernel,CPU" "$latest_quick" | cut -d, -f4)
                local gpu_time="$avgtime"
                local gpu_status="$status"
                
                if [ -n "$cpu_time" ] && [ "$cpu_time" != "ERROR" ] && [ "$gpu_time" != "ERROR" ]; then
                    local speedup=$(echo "scale=2; $cpu_time / $gpu_time" | bc -l 2>/dev/null || echo "N/A")
                    echo "| ${size}x${size}, ${kernel}x${kernel} | $cpu_time | $gpu_time | ${speedup}x | $gpu_status |" >> "$report_file"
                fi
            fi
        done < "$latest_quick"
    fi
    
    cat >> "$report_file" << EOF

#### Analysis Summary

1. **GPU Performance Issues**: 
   - Current GPU implementation is significantly slower than CPU
   - Root cause: Memory allocation overhead dominates execution time
   - Correctness issues prevent meaningful performance optimization

2. **Immediate Priorities**:
   - Fix GPU algorithm correctness (boundary conditions, indexing)
   - Pre-allocate GPU memory to eliminate malloc overhead
   - Implement memory coalescing for bandwidth efficiency

3. **Optimization Potential**:
   - **1000x improvement** possible through memory optimization
   - Current bandwidth utilization: <0.01% of theoretical peak
   - Target: 50-80% bandwidth utilization with proper coalescing

### Next Steps

1. **Phase 1**: Fix correctness and memory allocation issues
2. **Phase 2**: Implement coalesced memory access patterns  
3. **Phase 3**: Add shared memory tiling optimization
4. **Phase 4**: Integrate constant and texture memory optimizations

### Profiling Integration

EOF

    if [ "$PROFILING_ENABLED" = true ]; then
        cat >> "$report_file" << EOF
Detailed profiling data available in:
- Timeline analysis: \`nsys_*.nsys-rep\` files  
- Kernel metrics: \`ncu_*.log\` files
- Memory analysis: \`memory_analysis.md\`
- Roofline analysis: \`roofline_analysis.txt\`

Use NVIDIA Nsight Compute/Systems GUI for detailed visualization.
EOF
    else
        cat >> "$report_file" << EOF
Profiling was not enabled for this run. Use \`--profile\` flag for detailed analysis.

To run profiling separately:
\`\`\`bash
./tools/profile.sh --mode detailed --size 512 --kernel 5
\`\`\`
EOF
    fi
    
    cat >> "$report_file" << EOF

---
**Generated by**: CUDA Convolution Lab Performance Testing Framework  
**Version**: 1.0 (Updated June 16, 2025)
EOF

    log_info "Performance report generated: $report_file"
}

# Additional test modes
QUICK_TEST=false
CORRECTNESS_TEST=false

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
        -p|--profile)
            PROFILING_ENABLED=true
            shift
            ;;
        --quick)
            QUICK_TEST=true
            shift
            ;;
        --correctness)
            CORRECTNESS_TEST=true
            shift
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
    
    # Show configuration
    if [ "$VERBOSE" = true ]; then
        log_info "Configuration:"
        log_info "  Executable: $EXECUTABLE"
        log_info "  Benchmark runs: $BENCHMARK_RUNS"
        log_info "  CPU only: $CPU_ONLY"
        log_info "  GPU only: $GPU_ONLY"
        log_info "  Profiling: $PROFILING_ENABLED"
        log_info "  Quick test: $QUICK_TEST"
        log_info "  Correctness test: $CORRECTNESS_TEST"
        log_info "  Image sizes: ${IMAGE_SIZES[*]}"
        log_info "  Kernel sizes: ${KERNEL_SIZES[*]}"
    fi
    
    check_prerequisites
    
    # Run correctness test if requested
    if [ "$CORRECTNESS_TEST" = true ]; then
        run_correctness_test
    fi
    
    # Run appropriate test suite
    if [ "$QUICK_TEST" = true ]; then
        run_quick_test
    else
        # Run the full test suite
        run_scalability_test
    fi
    
    # Run profiling analysis if enabled
    if [ "$PROFILING_ENABLED" = true ] && [ "$QUICK_TEST" = false ]; then
        run_profiling_analysis
    fi
    
    # Generate final report
    generate_report
    
    log_info "Performance testing completed!"
    log_info "Results available in: $OUTPUT_DIR"
    
    # Show key findings summary
    if [ -f "$OUTPUT_DIR/quick_test_"*.csv ] || [ -f "$OUTPUT_DIR/scalability_"*.csv ]; then
        echo ""
        log_info "Key Results Summary:"
        
        # Look for GPU vs CPU comparison in quick test results
        local latest_quick=$(ls -t "$OUTPUT_DIR"/quick_test_*.csv 2>/dev/null | head -1)
        if [ -n "$latest_quick" ] && [ -f "$latest_quick" ]; then
            echo "From quick test results:"
            
            # Extract representative GPU and CPU times
            local gpu_time=$(grep "256,3,GPU" "$latest_quick" | cut -d, -f4 | head -1)
            local cpu_time=$(grep "256,3,CPU" "$latest_quick" | cut -d, -f4 | head -1)
            
            if [ -n "$gpu_time" ] && [ -n "$cpu_time" ] && [ "$gpu_time" != "ERROR" ] && [ "$cpu_time" != "ERROR" ]; then
                local speedup=$(echo "scale=2; $cpu_time / $gpu_time" | bc -l 2>/dev/null || echo "N/A")
                echo "  CPU time (256x256, 3x3): ${cpu_time} ms"
                echo "  GPU time (256x256, 3x3): ${gpu_time} ms"
                echo "  GPU speedup: ${speedup}x"
            fi
        fi
        
        # Check for correctness issues
        if [ -f "$OUTPUT_DIR/correctness_results.txt" ]; then
            local passed=$(grep "Summary:" "$OUTPUT_DIR/correctness_results.txt" | grep -o "[0-9]*\/[0-9]*" | head -1)
            if [ -n "$passed" ]; then
                echo "  Correctness: $passed tests passed"
            fi
        fi
    fi
}

# Run main function
main "$@"

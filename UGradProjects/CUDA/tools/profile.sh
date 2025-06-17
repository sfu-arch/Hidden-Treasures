#!/bin/bash

# CUDA Convolution Lab - GPU Profiling Script
# This script runs detailed profiling using NVIDIA NSight tools
# and generates comprehensive performance analysis reports

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default parameters
EXECUTABLE="./Build/conv_benchmark"
OUTPUT_DIR="./profiling_results"
PROFILE_MODE="summary"
IMAGE_SIZE=1024
KERNEL_SIZE=7

# Profiling tools
NVPROF="nsys"
NSIGHT_COMPUTE="ncu"
NSIGHT_SYSTEMS="nsys"

print_header() {
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}    CUDA Convolution Lab - GPU Profiling       ${NC}"
    echo -e "${BLUE}================================================${NC}"
    echo ""
}

print_usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -e, --executable <path>   Path to convolution executable"
    echo "  -o, --output <dir>        Output directory for profiling results"
    echo "  -m, --mode <mode>         Profiling mode: summary|detailed|memory|compute"
    echo "  -s, --size <size>         Image size for profiling (default: $IMAGE_SIZE)"
    echo "  -k, --kernel <size>       Kernel size for profiling (default: $KERNEL_SIZE)"
    echo "  -t, --tool <tool>         Profiling tool: nvprof|ncu|nsys|all"
    echo "  -h, --help                Show this help message"
    echo ""
    echo "Profiling modes:"
    echo "  summary     - Quick overview of kernel performance"
    echo "  detailed    - Comprehensive analysis with all metrics"
    echo "  memory      - Focus on memory bandwidth and access patterns"
    echo "  compute     - Focus on compute utilization and efficiency"
    echo ""
    echo "Examples:"
    echo "  $0 --mode detailed --size 2048"
    echo "  $0 --tool ncu --mode memory"
    echo "  $0 --tool all --size 1024 --kernel 5"
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
    log_info "Checking profiling prerequisites..."
    
    # Check if executable exists
    if [ ! -f "$EXECUTABLE" ]; then
        log_error "Executable not found: $EXECUTABLE"
        exit 1
    fi
    
    # Check for CUDA profiling tools
    local tools_found=0
    
    if command -v $NVPROF &> /dev/null; then
        log_info "Found nvprof (legacy profiler)"
        tools_found=$((tools_found + 1))
    fi
    
    if command -v $NSIGHT_COMPUTE &> /dev/null; then
        log_info "Found NVIDIA Nsight Compute"
        tools_found=$((tools_found + 1))
    fi
    
    if command -v $NSIGHT_SYSTEMS &> /dev/null; then
        log_info "Found NVIDIA Nsight Systems"
        tools_found=$((tools_found + 1))
    fi
    
    if [ $tools_found -eq 0 ]; then
        log_error "No CUDA profiling tools found. Please install NVIDIA NSight tools."
        exit 1
    fi
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    
    log_info "Prerequisites check passed"
}

profile_with_nvprof() {
    local mode=$1
    local output_prefix="$OUTPUT_DIR/nvprof_${mode}_${IMAGE_SIZE}_k${KERNEL_SIZE}"
    
    log_info "Profiling with nvprof (mode: $mode)..."
    
    case $mode in
        "summary")
            $NVPROF --print-gpu-summary \
                    --log-file "${output_prefix}.log" \
                    $EXECUTABLE  --width $IMAGE_SIZE --height $IMAGE_SIZE --kernel $KERNEL_SIZE
            ;;
        "detailed")
            $NVPROF --print-gpu-summary \
                    --print-gpu-trace \
                    --print-api-trace \
                    --export-profile "${output_prefix}.nvvp" \
                    --log-file "${output_prefix}.log" \
                    $EXECUTABLE --gpu-only --width $IMAGE_SIZE --height $IMAGE_SIZE --kernel $KERNEL_SIZE
            ;;
        "memory")
            $NVPROF --print-gpu-summary \
                    --events global_load,global_store,shared_load,shared_store \
                    --metrics gld_efficiency,gst_efficiency,gld_throughput,gst_throughput \
                    --log-file "${output_prefix}.log" \
                    $EXECUTABLE --gpu-only --width $IMAGE_SIZE --height $IMAGE_SIZE --kernel $KERNEL_SIZE
            ;;
        "compute")
            $NVPROF --print-gpu-summary \
                    --metrics achieved_occupancy,sm_efficiency,flop_count_sp \
                    --log-file "${output_prefix}.log" \
                    $EXECUTABLE --gpu-only --width $IMAGE_SIZE --height $IMAGE_SIZE --kernel $KERNEL_SIZE
            ;;
    esac
    
    log_info "nvprof profiling completed. Results saved to ${output_prefix}.*"
}

profile_with_ncu() {
    local mode=$1
    local output_prefix="$OUTPUT_DIR/ncu_${mode}_${IMAGE_SIZE}_k${KERNEL_SIZE}"
    
    log_info "Profiling with NVIDIA Nsight Compute (mode: $mode)..."
    
    case $mode in
        "summary")
            $NSIGHT_COMPUTE --metrics gpu__time_duration.sum \
                           --csv \
                           --log-file "${output_prefix}.log" \
                           $EXECUTABLE --gpu-only --width $IMAGE_SIZE --height $IMAGE_SIZE --kernel $KERNEL_SIZE
            ;;
        "detailed")
            $NSIGHT_COMPUTE --set full \
                           --export "${output_prefix}" \
                           --log-file "${output_prefix}.log" \
                           $EXECUTABLE --gpu-only --width $IMAGE_SIZE --height $IMAGE_SIZE --kernel $KERNEL_SIZE
            ;;
        "memory")
            $NSIGHT_COMPUTE --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__throughput.avg.pct_of_peak_sustained_elapsed,\
lts__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__data_pipe_lsu_wavefronts_mem_shared.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_global.sum \
                           --csv \
                           --log-file "${output_prefix}.log" \
                           $EXECUTABLE --gpu-only --width $IMAGE_SIZE --height $IMAGE_SIZE --kernel $KERNEL_SIZE
            ;;
        "compute")
            $NSIGHT_COMPUTE --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,\
sm__sass_thread_inst_executed_op_fadd_pred_on.sum,\
sm__sass_thread_inst_executed_op_fmul_pred_on.sum \
                           --csv \
                           --log-file "${output_prefix}.log" \
                           $EXECUTABLE --gpu-only --width $IMAGE_SIZE --height $IMAGE_SIZE --kernel $KERNEL_SIZE
            ;;
    esac
    
    log_info "Nsight Compute profiling completed. Results saved to ${output_prefix}.*"
}

profile_with_nsys() {
    local mode=$1
    local output_prefix="$OUTPUT_DIR/nsys_${mode}_${IMAGE_SIZE}_k${KERNEL_SIZE}"
    
    log_info "Profiling with NVIDIA Nsight Systems (mode: $mode)..."
    
    case $mode in
        "summary"|"detailed")
            $NSIGHT_SYSTEMS profile \
                           --trace cuda,nvtx \
                           --output "${output_prefix}" \
                           --stats true \
                           $EXECUTABLE --gpu-only --width $IMAGE_SIZE --height $IMAGE_SIZE --kernel $KERNEL_SIZE
            ;;
        "memory"|"compute")
            $NSIGHT_SYSTEMS profile \
                           --trace cuda,nvtx,osrt \
                           --output "${output_prefix}" \
                           --stats true \
                           --capture-range cudaProfilerApi \
                           $EXECUTABLE --gpu-only --width $IMAGE_SIZE --height $IMAGE_SIZE --kernel $KERNEL_SIZE
            ;;
    esac
    
    log_info "Nsight Systems profiling completed. Results saved to ${output_prefix}.*"
}

analyze_roofline() {
    log_info "Performing roofline analysis..."
    
    local roofline_file="$OUTPUT_DIR/roofline_analysis.txt"
    
    cat > "$roofline_file" << EOF
# CUDA Convolution Lab - Roofline Analysis
# Generated: $(date)

## System Information
$(nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv)

## Theoretical Peak Performance
EOF
    
    # Get GPU specifications (this would need to be expanded based on specific GPU)
    local gpu_info=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
    echo "GPU: $gpu_info" >> "$roofline_file"
    
    # Calculate theoretical peaks (example values - should be GPU-specific)
    cat >> "$roofline_file" << EOF

## Convolution Analysis

Image Size: ${IMAGE_SIZE}x${IMAGE_SIZE}
Kernel Size: ${KERNEL_SIZE}x${KERNEL_SIZE}

### Operational Intensity Calculation
- Total FLOPs: $(( IMAGE_SIZE * IMAGE_SIZE * KERNEL_SIZE * KERNEL_SIZE * 2 ))
- Memory Access: $(( IMAGE_SIZE * IMAGE_SIZE * KERNEL_SIZE * KERNEL_SIZE * 4 + IMAGE_SIZE * IMAGE_SIZE * 4 )) bytes
- Operational Intensity: (calculated from profiling data)

### Performance Bounds
- Memory Bound: (calculated from memory bandwidth)
- Compute Bound: (calculated from peak FLOPS)

### Optimization Recommendations
(Generated based on profiling results)

EOF
    
    log_info "Roofline analysis saved to $roofline_file"
}

generate_memory_analysis() {
    log_info "Generating memory access pattern analysis..."
    
    local memory_file="$OUTPUT_DIR/memory_analysis.md"
    
    cat > "$memory_file" << EOF
# Memory Access Pattern Analysis

## Naive Implementation Analysis

### Global Memory Access Pattern
- Each thread accesses multiple global memory locations
- Access pattern: scattered (non-coalesced)
- Memory reuse: minimal

### Optimization Opportunities

1. **Coalesced Access**
   - Organize threads to access contiguous memory
   - Expected improvement: 2-4x bandwidth utilization

2. **Shared Memory**
   - Cache frequently accessed data in shared memory
   - Reduce global memory accesses
   - Expected improvement: 3-8x for appropriate tile sizes

3. **Constant Memory**
   - Store convolution kernel in constant memory
   - Broadcast read access pattern
   - Expected improvement: 1.5-2x for kernel access

4. **Texture Memory**
   - Utilize 2D spatial locality for image data
   - Hardware interpolation and caching
   - Expected improvement: 1.5-3x for scattered access

## Memory Hierarchy Utilization

| Memory Type | Bandwidth | Latency | Size | Current Usage |
|-------------|-----------|---------|------|---------------|
| Global      | ~1TB/s    | 400-800 cycles | GB | Heavy |
| Shared      | ~19TB/s   | 1-32 cycles | 48-164KB | None |
| Constant    | ~1TB/s    | 1-10 cycles | 64KB | None |
| Texture     | ~1TB/s    | Variable | N/A | None |

EOF
    
    log_info "Memory analysis saved to $memory_file"
}

generate_report() {
    log_info "Generating comprehensive profiling report..."
    
    local report_file="$OUTPUT_DIR/profiling_report.md"
    local timestamp=$(date)
    
    cat > "$report_file" << EOF
# CUDA Convolution Lab - Profiling Report

Generated: $timestamp
Image Size: ${IMAGE_SIZE}x${IMAGE_SIZE}
Kernel Size: ${KERNEL_SIZE}x${KERNEL_SIZE}

## Executive Summary

This report provides detailed performance analysis of the CUDA convolution implementations
using NVIDIA profiling tools.

## System Configuration

### Hardware
$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv)

### Software
- CUDA Version: $(nvcc --version | grep "release" | cut -d' ' -f5-6)
- Driver Version: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader)

## Profiling Results

### Performance Summary
(Results would be extracted from profiling tool outputs)

### Kernel Analysis
1. **Execution Time**: 
2. **Memory Throughput**: 
3. **Compute Utilization**: 
4. **Occupancy**: 

### Memory Access Patterns
1. **Global Memory Efficiency**: 
2. **Cache Hit Rates**: 
3. **Bandwidth Utilization**: 

## Optimization Recommendations

### Immediate Improvements
1. Implement coalesced memory access
2. Add shared memory tiling
3. Use constant memory for convolution kernel

### Advanced Optimizations
1. Texture memory for image data
2. Multiple kernel fusion
3. Asynchronous memory transfers

### Expected Performance Gains
- Coalesced access: 2-4x improvement
- Shared memory: 3-8x improvement  
- Constant memory: 1.5-2x improvement
- Combined optimizations: 10-20x potential improvement

## Roofline Analysis

(Include roofline plot and analysis)

## Comparison with Theoretical Peaks

(Compare achieved performance with hardware limits)

## Next Steps

1. Implement suggested optimizations
2. Re-profile optimized versions
3. Compare with library implementations (cuDNN)
4. Analyze scaling across different GPU architectures

EOF
    
    log_info "Profiling report generated: $report_file"
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
        -m|--mode)
            PROFILE_MODE="$2"
            shift 2
            ;;
        -s|--size)
            IMAGE_SIZE="$2"
            shift 2
            ;;
        -k|--kernel)
            KERNEL_SIZE="$2"
            shift 2
            ;;
        -t|--tool)
            PROFILE_TOOL="$2"
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
    
    log_info "Starting GPU profiling..."
    log_info "Image size: ${IMAGE_SIZE}x${IMAGE_SIZE}"
    log_info "Kernel size: ${KERNEL_SIZE}x${KERNEL_SIZE}"
    log_info "Profile mode: $PROFILE_MODE"
    log_info "Output directory: $OUTPUT_DIR"
    
    check_prerequisites
    
    # Run profiling based on selected tool
    case ${PROFILE_TOOL:-"all"} in
        "nvprof")
            profile_with_nvprof "$PROFILE_MODE"
            ;;
        "ncu")
            profile_with_ncu "$PROFILE_MODE"
            ;;
        "nsys")
            profile_with_nsys "$PROFILE_MODE"
            ;;
        "all")
            if command -v $NVPROF &> /dev/null; then
                profile_with_nvprof "$PROFILE_MODE"
            fi
            if command -v $NSIGHT_COMPUTE &> /dev/null; then
                profile_with_ncu "$PROFILE_MODE"
            fi
            if command -v $NSIGHT_SYSTEMS &> /dev/null; then
                profile_with_nsys "$PROFILE_MODE"
            fi
            ;;
    esac
    
    # Generate analysis reports
    analyze_roofline
    generate_memory_analysis
    generate_report
    
    log_info "Profiling completed!"
    log_info "Results and analysis available in: $OUTPUT_DIR"
    
    echo ""
    echo "Next steps:"
    echo "1. Review the profiling report: $OUTPUT_DIR/profiling_report.md"
    echo "2. Analyze detailed profiles with NVIDIA Nsight Compute/Systems GUI"
    echo "3. Implement suggested optimizations"
    echo "4. Re-run profiling to measure improvements"
}

# Run main function
main "$@"

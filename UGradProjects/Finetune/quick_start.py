#!/usr/bin/env python3
"""
Quick Start Script for CPU Memory Optimization Research Project
Run this script to verify your setup and begin exploring the CPU benchmarks
"""

import numpy as np
import time
import subprocess
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

try:
    from cpu_memory_benchmark_suite import CPUBenchmarkSuite, CPUMemoryOptimizationTrainer
    print("✓ Successfully imported cpu_memory_benchmark_suite")
except ImportError as e:
    print(f"✗ Failed to import cpu_memory_benchmark_suite: {e}")
    sys.exit(1)

def check_environment():
    """Verify the development environment is properly configured"""
    print("🔍 Environment Check")
    print("=" * 50)
    
    # Python version
    print(f"Python version: {sys.version}")
    
    # NumPy and core libraries
    print(f"NumPy version: {np.__version__}")
    
    try:
        import scipy
        print(f"SciPy version: {scipy.__version__}")
    except ImportError:
        print("⚠️  SciPy not available")
    
    # Check CPU information
    try:
        result = subprocess.run(['lscpu'], capture_output=True, text=True)
        if result.returncode == 0:
            print("\nCPU Information:")
            for line in result.stdout.split('\n'):
                if any(keyword in line for keyword in ['Model name', 'CPU(s)', 'Thread', 'Cache']):
                    print(f"  {line}")
    except:
        print("Could not retrieve CPU information")
    
    # Check compilers
    print("\nCompiler Check:")
    for compiler in ['gcc', 'g++', 'clang', 'clang++']:
        try:
            result = subprocess.run([compiler, '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                version_line = result.stdout.split('\n')[0]
                print(f"  ✓ {compiler}: {version_line}")
        except FileNotFoundError:
            print(f"  ✗ {compiler}: Not found")
    
    # Check profiling tools
    print("\nProfileing Tools:")
    for tool in ['perf', 'valgrind']:
        try:
            result = subprocess.run([tool, '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"  ✓ {tool}: Available")
        except FileNotFoundError:
            print(f"  ✗ {tool}: Not found")
    
    # Check BLAS libraries
    print("\nBLAS Library Check:")
    try:
        import numpy.distutils.system_info as sysinfo
        blas_info = sysinfo.get_info('blas')
        if blas_info:
            print(f"  ✓ BLAS library detected")
            print(f"    Libraries: {blas_info.get('libraries', 'Unknown')}")
        else:
            print("  ⚠️  No optimized BLAS library detected")
    except:
        print("  ⚠️  Could not check BLAS configuration")
    
    # Test basic CPU computation
    try:
        size = 1000
        A = np.random.rand(size, size)
        B = np.random.rand(size, size)
        
        start_time = time.time()
        C = np.dot(A, B)
        end_time = time.time()
        
        computation_time = end_time - start_time
        flops = 2 * size**3  # Matrix multiplication FLOPs
        gflops = flops / (computation_time * 1e9)
        
        print(f"\nBasic Performance Test:")
        print(f"  Matrix multiplication ({size}x{size}): {computation_time:.3f} seconds")
        print(f"  Performance: {gflops:.1f} GFLOPS")
        
    except Exception as e:
        print(f"  ✗ Basic computation test failed: {e}")
    
    print()

def run_benchmark_demo():
    """Run a quick demonstration of the CPU benchmark suite"""
    print("🚀 CPU Benchmark Suite Demo")
    print("=" * 50)
    
    # Initialize benchmark suite
    suite = CPUBenchmarkSuite()
    print(f"Loaded {len(suite.benchmarks)} benchmarks:")
    
    for i, benchmark in enumerate(suite.benchmarks):
        print(f"  {i+1}. {benchmark.name}")
        print(f"      Type: {benchmark.optimization_type}")
        print(f"      Difficulty: Level {benchmark.difficulty_level}")
    
    print()
    
    # Run a simple benchmark test
    print("Running Data Structure Optimization Benchmark...")
    benchmark = suite.benchmarks[0]  # Data structure benchmark
    
    # Generate test data
    inputs = benchmark.generate_input_data(size_factor=1)
    print(f"Input data shapes:")
    for key, array in inputs.items():
        if hasattr(array, 'shape'):
            print(f"  {key}: {array.shape} ({array.dtype})")
        else:
            print(f"  {key}: {array}")
    
    # Time reference implementation
    start_time = time.time()
    result = benchmark.reference_implementation(inputs)
    ref_time = time.time() - start_time
    
    print(f"Reference execution time: {ref_time*1000:.2f} ms")
    print(f"Output shape: {result.shape}")
    print(f"Sample output: {result[:5]} ...")
    
    # Show optimization template
    print("\nOptimization Challenge:")
    print("-" * 30)
    template = benchmark.extract_optimization_template()
    template_preview = template[:800] + "..." if len(template) > 800 else template
    print(template_preview)
    
    print()

def demonstrate_cache_analysis():
    """Show cache behavior analysis capabilities"""
    print("📊 Cache Behavior Analysis Demo")
    print("=" * 50)
    
    print("Demonstrating cache-friendly vs cache-unfriendly access patterns...")
    
    # Test different matrix access patterns
    size = 2048
    matrix = np.random.rand(size, size).astype(np.float32)
    
    print(f"\nTesting {size}x{size} matrix access patterns:")
    
    # Row-major access (cache-friendly)
    start_time = time.time()
    row_sum = 0.0
    for i in range(size):
        for j in range(size):
            row_sum += matrix[i, j]
    row_time = time.time() - start_time
    
    # Column-major access (cache-unfriendly)
    start_time = time.time()
    col_sum = 0.0
    for j in range(size):
        for i in range(size):
            col_sum += matrix[i, j]
    col_time = time.time() - start_time
    
    print(f"  Row-major access (cache-friendly):    {row_time:.3f} seconds")
    print(f"  Column-major access (cache-unfriendly): {col_time:.3f} seconds")
    print(f"  Performance ratio: {col_time/row_time:.2f}x slower")
    
    # Matrix multiplication comparison
    A = np.random.rand(512, 512).astype(np.float32)
    B = np.random.rand(512, 512).astype(np.float32)
    
    # NumPy optimized (should use BLAS)
    start_time = time.time()
    C_optimized = np.dot(A, B)
    optimized_time = time.time() - start_time
    
    # Naive implementation
    start_time = time.time()
    C_naive = np.zeros((512, 512), dtype=np.float32)
    for i in range(512):
        for j in range(512):
            for k in range(512):
                C_naive[i, j] += A[i, k] * B[k, j]
    naive_time = time.time() - start_time
    
    print(f"\nMatrix Multiplication Comparison (512x512):")
    print(f"  Optimized (NumPy/BLAS): {optimized_time:.3f} seconds")
    print(f"  Naive implementation:   {naive_time:.3f} seconds")
    print(f"  Speedup: {naive_time/optimized_time:.1f}x")
    
    print()

def show_optimization_examples():
    """Show examples of CPU memory optimizations"""
    print("⚡ CPU Memory Optimization Examples")
    print("=" * 50)
    
    print("Example 1: Data Structure Layout Optimization")
    print("-" * 50)
    
    aos_code = """
    // Array of Structures (AoS) - potentially cache-unfriendly
    struct Point3D {
        float x, y, z;
    };
    
    std::vector<Point3D> points(n);
    
    // Computing distances - poor spatial locality
    for (int i = 0; i < n; ++i) {
        float dist = sqrt(points[i].x * points[i].x + 
                         points[i].y * points[i].y + 
                         points[i].z * points[i].z);
        distances[i] = dist;
    }
    """
    
    soa_code = """
    // Structure of Arrays (SoA) - cache-friendly
    struct Points3D {
        std::vector<float> x, y, z;
    };
    
    Points3D points(n);
    
    // Computing distances - excellent spatial locality + vectorizable
    for (int i = 0; i < n; ++i) {
        float dist = sqrt(points.x[i] * points.x[i] + 
                         points.y[i] * points.y[i] + 
                         points.z[i] * points.z[i]);
        distances[i] = dist;
    }
    """
    
    print("Array of Structures (AoS):")
    print(aos_code)
    print("\nStructure of Arrays (SoA) - Optimized:")
    print(soa_code)
    
    print("Example 2: Matrix Tiling for Cache Efficiency")
    print("-" * 50)
    
    tiling_code = """
    // Cache-oblivious matrix multiplication with multi-level tiling
    void optimized_gemm(float* A, float* B, float* C, int N) {
        const int L1_TILE = 64;   // Fit in L1 cache
        const int L2_TILE = 256;  // Fit in L2 cache
        
        for (int ii = 0; ii < N; ii += L2_TILE) {
            for (int jj = 0; jj < N; jj += L2_TILE) {
                for (int kk = 0; kk < N; kk += L2_TILE) {
                    // L2 tiles
                    for (int i = ii; i < min(ii + L2_TILE, N); i += L1_TILE) {
                        for (int j = jj; j < min(jj + L2_TILE, N); j += L1_TILE) {
                            for (int k = kk; k < min(kk + L2_TILE, N); k += L1_TILE) {
                                // L1 tiles with register blocking
                                micro_kernel(A, B, C, i, j, k, L1_TILE);
                            }
                        }
                    }
                }
            }
        }
    }
    """
    
    print(tiling_code)
    print()

def show_training_pipeline():
    """Demonstrate the training pipeline setup for CPU optimization"""
    print("🎯 CPU Optimization Training Pipeline Demo")
    print("=" * 50)
    
    # Initialize trainer
    trainer = CPUMemoryOptimizationTrainer()
    
    # Generate training prompts
    prompts = trainer.generate_training_prompts()
    print(f"Generated {len(prompts)} training prompts")
    
    # Show example prompt
    if prompts:
        example = prompts[0]
        print(f"\nExample prompt for '{example['benchmark_name']}':")
        print(f"Optimization type: {example['optimization_type']}")
        print(f"Difficulty level: {example['difficulty']}")
        print("-" * 40)
        prompt_preview = example['prompt'][:600] + "..." if len(example['prompt']) > 600 else example['prompt']
        print(prompt_preview)
    
    print("\nCPU Training pipeline features:")
    print("- Multi-turn refinement with cache efficiency feedback")
    print("- CPU-specific reward functions (cache miss rates, IPC)")
    print("- Data structure and algorithm optimization")
    print("- Matrix tiling and blocking optimization")
    print("- Loop transformation and vectorization")
    print("- Halide DSL schedule optimization")
    print()
    
    # Show optimization types
    opt_types = set(p["optimization_type"] for p in prompts)
    print(f"Optimization types covered: {opt_types}")
    print()

def main():
    """Main entry point for the CPU optimization quick start demo"""
    print("🔬 CPU Memory Optimization Research Project")
    print("Quick Start Demo and Environment Verification")
    print("=" * 60)
    print()
    
    # Run all demonstration functions
    try:
        check_environment()
        run_benchmark_demo()
        demonstrate_cache_analysis()
        show_optimization_examples()
        show_training_pipeline()
        
        print("🎉 CPU Optimization Quick Start Complete!")
        print("=" * 50)
        print("Next steps:")
        print("1. Explore the CPU benchmark suite in detail")
        print("2. Run individual benchmarks with different parameters")
        print("3. Implement custom CPU memory benchmarks")
        print("4. Begin fine-tuning experiments on CPU optimization")
        print("5. Check out notebooks/getting_started.ipynb for interactive exploration")
        print()
        print("For detailed guidance, see:")
        print("- Fine-tune-Memory-Optimization-Research.md (research plan)")
        print("- TASK_CHECKLIST.md (implementation tasks)")
        print("- cpu_memory_benchmark_suite.py (core framework)")
        print("- cpp_code/ directory for C++ examples")
        print()
        print("Compile and run the C++ example:")
        print("  cd cpp_code && make && ./example_benchmark")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        print("Check your environment setup and dependencies.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

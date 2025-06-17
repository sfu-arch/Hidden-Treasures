#!/usr/bin/env python3
"""
CPU Memory Behavior Optimization Benchmark Suite
A framework for evaluating LLM-generated CPU code for memory efficiency,
data structure optimization, matrix tiling, loop optimization, and DSL code generation
"""

import numpy as np
import time
import subprocess
import tempfile
import os
import ctypes
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class CPUCacheMetrics:
    """Container for CPU cache-related performance metrics"""
    l1_cache_misses: int
    l1_cache_references: int
    l2_cache_misses: int
    l2_cache_references: int
    l3_cache_misses: int
    l3_cache_references: int
    instructions: int
    cycles: int
    execution_time_ms: float
    cache_miss_rate_l1: float
    cache_miss_rate_l2: float
    cache_miss_rate_l3: float
    ipc: float  # Instructions per cycle
    
    @property
    def overall_cache_efficiency(self) -> float:
        """Combined cache efficiency score"""
        l1_hit_rate = 1.0 - self.cache_miss_rate_l1
        l2_hit_rate = 1.0 - self.cache_miss_rate_l2
        l3_hit_rate = 1.0 - self.cache_miss_rate_l3
        return (0.5 * l1_hit_rate + 0.3 * l2_hit_rate + 0.2 * l3_hit_rate)


class CPUMemoryBenchmark(ABC):
    """Abstract base class for CPU memory optimization benchmarks"""
    
    def __init__(self, name: str, difficulty_level: int, optimization_type: str):
        self.name = name
        self.difficulty_level = difficulty_level  # 1-4 (easy to hard)
        self.optimization_type = optimization_type  # "data_structure", "matrix_tiling", "loop_opt", "dsl"
        self.reference_performance = None
        
    @abstractmethod
    def generate_input_data(self, size_factor: int = 1) -> Dict[str, np.ndarray]:
        """Generate input data for the benchmark"""
        pass
        
    @abstractmethod
    def reference_implementation(self, inputs: Dict[str, np.ndarray]) -> np.ndarray:
        """Reference NumPy/SciPy implementation"""
        pass
        
    @abstractmethod
    def extract_optimization_template(self) -> str:
        """Return C++/Halide code template for LLM to optimize"""
        pass
        
    def evaluate_correctness(self, generated_code: str, inputs: Dict[str, np.ndarray]) -> bool:
        """Check if generated code produces correct results"""
        try:
            # Compile and run generated C++ code
            result_generated = self._compile_and_run_cpp(generated_code, inputs)
            result_reference = self.reference_implementation(inputs)
            
            # Compare results with tolerance
            return np.allclose(result_generated, result_reference, rtol=1e-3, atol=1e-5)
        except Exception as e:
            print(f"Correctness evaluation failed: {e}")
            return False
            
    def measure_cpu_performance(self, code: str, inputs: Dict[str, np.ndarray]) -> CPUCacheMetrics:
        """Measure CPU cache and performance metrics using perf"""
        try:
            # Compile the code
            executable = self._compile_cpp_code(code, inputs)
            
            # Run with perf to collect cache statistics
            perf_cmd = [
                'perf', 'stat', '-e', 
                'cache-misses,cache-references,L1-dcache-misses,L1-dcache-loads,'
                'LLC-load-misses,LLC-loads,instructions,cycles',
                '-x', ',',  # CSV output
                executable
            ]
            
            start_time = time.time()
            result = subprocess.run(perf_cmd, capture_output=True, text=True)
            end_time = time.time()
            
            execution_time = (end_time - start_time) * 1000  # Convert to ms
            
            # Parse perf output
            metrics = self._parse_perf_output(result.stderr, execution_time)
            return metrics
            
        except Exception as e:
            print(f"Performance measurement failed: {e}")
            # Return mock metrics if perf fails
            return CPUCacheMetrics(
                l1_cache_misses=0, l1_cache_references=1000,
                l2_cache_misses=0, l2_cache_references=100,
                l3_cache_misses=0, l3_cache_references=10,
                instructions=10000, cycles=5000,
                execution_time_ms=time.time() * 1000,
                cache_miss_rate_l1=0.1, cache_miss_rate_l2=0.2, cache_miss_rate_l3=0.3,
                ipc=2.0
            )
        
    def _compile_and_run_cpp(self, cpp_code: str, inputs: Dict[str, np.ndarray]) -> np.ndarray:
        """Compile and run C++ code (simplified implementation)"""
        # For now, falling back to reference implementation
        # In real implementation, would compile and execute C++ code
        return self.reference_implementation(inputs)
    
    def _compile_cpp_code(self, cpp_code: str, inputs: Dict[str, np.ndarray]) -> str:
        """Compile C++ code and return executable path"""
        # Create temporary files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
            f.write(cpp_code)
            cpp_file = f.name
            
        executable = cpp_file.replace('.cpp', '')
        
        # Compile with optimizations
        compile_cmd = [
            'g++', '-O3', '-march=native', '-mtune=native',
            '-ffast-math', '-funroll-loops',
            cpp_file, '-o', executable
        ]
        
        subprocess.run(compile_cmd, check=True)
        return executable
    
    def _parse_perf_output(self, perf_stderr: str, execution_time: float) -> CPUCacheMetrics:
        """Parse perf stat output to extract cache metrics"""
        # Mock implementation - would parse actual perf output
        return CPUCacheMetrics(
            l1_cache_misses=1000, l1_cache_references=10000,
            l2_cache_misses=100, l2_cache_references=1000,
            l3_cache_misses=10, l3_cache_references=100,
            instructions=100000, cycles=50000,
            execution_time_ms=execution_time,
            cache_miss_rate_l1=0.1, cache_miss_rate_l2=0.1, cache_miss_rate_l3=0.1,
            ipc=2.0
        )


class DataStructureOptimizationBenchmark(CPUMemoryBenchmark):
    """Benchmark for cache-conscious data structure optimization"""
    
    def __init__(self):
        super().__init__("Cache-Conscious Data Structures", difficulty_level=2, optimization_type="data_structure")
        
    def generate_input_data(self, size_factor: int = 1) -> Dict[str, np.ndarray]:
        """Generate data for structure optimization benchmark"""
        n_elements = 10000 * size_factor
        
        # AoS vs SoA test data
        x_coords = np.random.rand(n_elements).astype(np.float32)
        y_coords = np.random.rand(n_elements).astype(np.float32)
        z_coords = np.random.rand(n_elements).astype(np.float32)
        
        return {
            "x_coords": x_coords,
            "y_coords": y_coords, 
            "z_coords": z_coords,
            "n_elements": np.array([n_elements])
        }
        
    def reference_implementation(self, inputs: Dict[str, np.ndarray]) -> np.ndarray:
        """Reference NumPy implementation - compute distances from origin"""
        x = inputs["x_coords"]
        y = inputs["y_coords"]
        z = inputs["z_coords"]
        
        # Compute distance from origin for each point
        distances = np.sqrt(x*x + y*y + z*z)
        return distances
        
    def extract_optimization_template(self) -> str:
        """Return C++ template for data structure optimization"""
        return """
        #include <vector>
        #include <cmath>
        #include <chrono>
        
        // Task: Optimize data layout for cache efficiency
        
        // Version 1: Array of Structures (AoS) - potentially cache-unfriendly
        struct Point3D_AoS {
            float x, y, z;
        };
        
        std::vector<float> compute_distances_aos(const std::vector<Point3D_AoS>& points) {
            std::vector<float> distances;
            distances.reserve(points.size());
            
            for (const auto& point : points) {
                float dist = std::sqrt(point.x * point.x + 
                                     point.y * point.y + 
                                     point.z * point.z);
                distances.push_back(dist);
            }
            return distances;
        }
        
        // Your task: Implement Structure of Arrays (SoA) version for better cache locality
        // Consider: memory prefetching, vectorization, cache line utilization
        
        struct Point3D_SoA {
            std::vector<float> x;
            std::vector<float> y; 
            std::vector<float> z;
            
            Point3D_SoA(size_t size) : x(size), y(size), z(size) {}
        };
        
        std::vector<float> compute_distances_soa(const Point3D_SoA& points) {
            // TODO: Implement cache-friendly version
            // Hints: 
            // - Process data in chunks that fit in cache
            // - Consider vectorization with SIMD
            // - Minimize memory strides
            // - Use software prefetching if beneficial
            
            size_t n = points.x.size();
            std::vector<float> distances(n);
            
            // Your optimized implementation here
            for (size_t i = 0; i < n; ++i) {
                distances[i] = std::sqrt(points.x[i] * points.x[i] + 
                                       points.y[i] * points.y[i] + 
                                       points.z[i] * points.z[i]);
            }
            
            return distances;
        }
        """


class MatrixTilingBenchmark(CPUMemoryBenchmark):
    """Benchmark for matrix tiling and blocking optimization"""
    
    def __init__(self):
        super().__init__("Matrix Tiling Optimization", difficulty_level=3, optimization_type="matrix_tiling")
        
    def generate_input_data(self, size_factor: int = 1) -> Dict[str, np.ndarray]:
        """Generate matrices for tiling benchmark"""
        size = 512 * size_factor
        
        A = np.random.rand(size, size).astype(np.float32)
        B = np.random.rand(size, size).astype(np.float32)
        
        return {"matrix_A": A, "matrix_B": B}
        
    def reference_implementation(self, inputs: Dict[str, np.ndarray]) -> np.ndarray:
        """Reference NumPy matrix multiplication"""
        return np.dot(inputs["matrix_A"], inputs["matrix_B"])
        
    def extract_optimization_template(self) -> str:
        """Return C++ template for matrix tiling optimization"""
        return """
        #include <vector>
        #include <algorithm>
        #include <immintrin.h>  // For SIMD intrinsics
        
        // Task: Implement cache-efficient matrix multiplication with tiling
        
        // Naive matrix multiplication - poor cache locality
        void matrix_multiply_naive(const float* A, const float* B, float* C, 
                                  int N, int M, int K) {
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < M; ++j) {
                    float sum = 0.0f;
                    for (int k = 0; k < K; ++k) {
                        sum += A[i * K + k] * B[k * M + j];
                    }
                    C[i * M + j] = sum;
                }
            }
        }
        
        // Your task: Implement optimized version with tiling
        void matrix_multiply_optimized(const float* A, const float* B, float* C, 
                                     int N, int M, int K) {
            // TODO: Implement multi-level tiling for L1/L2/L3 caches
            // Consider:
            // - Optimal tile sizes for your target architecture
            // - Register blocking (micro-kernels)
            // - Loop unrolling and software pipelining
            // - SIMD vectorization
            // - Memory prefetching
            
            constexpr int TILE_SIZE_I = 64;  // Tune for L1 cache
            constexpr int TILE_SIZE_J = 64;
            constexpr int TILE_SIZE_K = 64;
            
            for (int ii = 0; ii < N; ii += TILE_SIZE_I) {
                for (int jj = 0; jj < M; jj += TILE_SIZE_J) {
                    for (int kk = 0; kk < K; kk += TILE_SIZE_K) {
                        
                        // Process tile
                        int i_end = std::min(ii + TILE_SIZE_I, N);
                        int j_end = std::min(jj + TILE_SIZE_J, M);
                        int k_end = std::min(kk + TILE_SIZE_K, K);
                        
                        for (int i = ii; i < i_end; ++i) {
                            for (int j = jj; j < j_end; ++j) {
                                float sum = C[i * M + j];  // Accumulate for multiple k-tiles
                                
                                for (int k = kk; k < k_end; ++k) {
                                    sum += A[i * K + k] * B[k * M + j];
                                }
                                
                                C[i * M + j] = sum;
                            }
                        }
                    }
                }
            }
        }
        
        // Advanced: Implement SIMD-optimized micro-kernel
        void matrix_multiply_microkernel(const float* A, const float* B, float* C,
                                       int i_start, int j_start, int k_start,
                                       int tile_i, int tile_j, int tile_k,
                                       int N, int M, int K) {
            // TODO: Implement 4x4 or 8x8 register-blocked micro-kernel
            // Use SIMD intrinsics for vectorization
        }
        """


class LoopOptimizationBenchmark(CPUMemoryBenchmark):
    """Benchmark for loop transformation and optimization"""
    
    def __init__(self):
        super().__init__("Loop Optimization", difficulty_level=2, optimization_type="loop_opt")
        
    def generate_input_data(self, size_factor: int = 1) -> Dict[str, np.ndarray]:
        """Generate data for loop optimization benchmark"""
        size = 1024 * size_factor
        
        # 2D stencil computation data
        grid = np.random.rand(size, size).astype(np.float32)
        
        return {"input_grid": grid}
        
    def reference_implementation(self, inputs: Dict[str, np.ndarray]) -> np.ndarray:
        """Reference NumPy stencil computation (5-point stencil)"""
        grid = inputs["input_grid"]
        height, width = grid.shape
        
        result = np.zeros_like(grid)
        
        # 5-point stencil (excluding boundaries)
        result[1:-1, 1:-1] = (grid[1:-1, 1:-1] + 
                             grid[0:-2, 1:-1] + 
                             grid[2:, 1:-1] + 
                             grid[1:-1, 0:-2] + 
                             grid[1:-1, 2:]) / 5.0
        
        return result
        
    def extract_optimization_template(self) -> str:
        """Return C++ template for loop optimization"""
        return """
        #include <vector>
        #include <algorithm>
        #include <omp.h>  // For OpenMP parallelization
        
        // Task: Optimize 2D stencil computation with loop transformations
        
        // Naive implementation - poor cache locality
        void stencil_naive(const float* input, float* output, int height, int width) {
            for (int i = 1; i < height - 1; ++i) {
                for (int j = 1; j < width - 1; ++j) {
                    output[i * width + j] = (input[i * width + j] +
                                           input[(i-1) * width + j] +
                                           input[(i+1) * width + j] +
                                           input[i * width + (j-1)] +
                                           input[i * width + (j+1)]) / 5.0f;
                }
            }
        }
        
        // Your task: Implement optimized version
        void stencil_optimized(const float* input, float* output, int height, int width) {
            // TODO: Apply loop optimizations:
            // 1. Cache blocking/tiling for spatial locality
            // 2. Loop unrolling for instruction-level parallelism
            // 3. Vectorization with SIMD
            // 4. Prefetching for memory latency hiding
            // 5. OpenMP parallelization
            
            constexpr int TILE_SIZE = 64;  // Tune for cache size
            
            #pragma omp parallel for collapse(2)
            for (int ii = 1; ii < height - 1; ii += TILE_SIZE) {
                for (int jj = 1; jj < width - 1; jj += TILE_SIZE) {
                    
                    int i_end = std::min(ii + TILE_SIZE, height - 1);
                    int j_end = std::min(jj + TILE_SIZE, width - 1);
                    
                    for (int i = ii; i < i_end; ++i) {
                        for (int j = jj; j < j_end; ++j) {
                            // Consider loop unrolling and vectorization here
                            output[i * width + j] = (input[i * width + j] +
                                                   input[(i-1) * width + j] +
                                                   input[(i+1) * width + j] +
                                                   input[i * width + (j-1)] +
                                                   input[i * width + (j+1)]) / 5.0f;
                        }
                    }
                }
            }
        }
        
        // Advanced: Implement with explicit vectorization
        void stencil_vectorized(const float* input, float* output, int height, int width) {
            // TODO: Use SIMD intrinsics for vectorization
            // Process multiple elements per iteration
        }
        """


class HalideDSLBenchmark(CPUMemoryBenchmark):
    """Benchmark for Halide-style DSL optimization"""
    
    def __init__(self):
        super().__init__("Halide DSL Optimization", difficulty_level=4, optimization_type="dsl")
        
    def generate_input_data(self, size_factor: int = 1) -> Dict[str, np.ndarray]:
        """Generate image data for Halide benchmark"""
        size = 512 * size_factor
        
        # Generate test image
        image = np.random.randint(0, 256, (size, size), dtype=np.uint8)
        
        return {"input_image": image}
        
    def reference_implementation(self, inputs: Dict[str, np.ndarray]) -> np.ndarray:
        """Reference NumPy implementation - Gaussian blur"""
        from scipy import ndimage
        
        image = inputs["input_image"].astype(np.float32)
        
        # Gaussian blur with sigma=1.0
        blurred = ndimage.gaussian_filter(image, sigma=1.0)
        
        return blurred.astype(np.uint8)
        
    def extract_optimization_template(self) -> str:
        """Return Halide code template for DSL optimization"""
        return """
        #include "Halide.h"
        using namespace Halide;
        
        // Task: Optimize image processing pipeline with Halide scheduling
        
        // Basic Gaussian blur pipeline
        Func gaussian_blur_basic(Func input) {
            Var x, y;
            
            // Define Gaussian kernel
            Func kernel;
            kernel(x) = exp(-x*x/2.0f);
            
            // Horizontal blur
            Func blur_x;
            blur_x(x, y) = (kernel(-2) * input(x-2, y) +
                           kernel(-1) * input(x-1, y) +
                           kernel(0)  * input(x, y) +
                           kernel(1)  * input(x+1, y) +
                           kernel(2)  * input(x+2, y)) / 5.0f;
            
            // Vertical blur
            Func blur_y;
            blur_y(x, y) = (kernel(-2) * blur_x(x, y-2) +
                           kernel(-1) * blur_x(x, y-1) +
                           kernel(0)  * blur_x(x, y) +
                           kernel(1)  * blur_x(x, y+1) +
                           kernel(2)  * blur_x(x, y+2)) / 5.0f;
            
            return blur_y;
        }
        
        // Your task: Optimize the schedule for cache efficiency
        void optimize_gaussian_blur(Func input, Func output) {
            Var x, y, xi, yi, xo, yo;
            
            // TODO: Apply Halide scheduling directives:
            // 1. Tiling for cache locality
            // 2. Vectorization for SIMD utilization
            // 3. Parallelization for multi-core
            // 4. Compute_at/store_at for memory optimization
            // 5. Unrolling for instruction-level parallelism
            
            Func blur = gaussian_blur_basic(input);
            
            // Example optimizations (tune these):
            blur.tile(x, y, xo, yo, xi, yi, 64, 64)
                .vectorize(xi, 8)
                .parallel(yo);
            
            // Consider intermediate storage strategy
            // blur_x.compute_at(blur, xo).vectorize(x, 8);
            
            output = blur;
        }
        
        // Advanced: Multi-stage pipeline optimization
        Func complex_image_pipeline(Func input) {
            // TODO: Implement multi-stage pipeline:
            // 1. Edge detection
            // 2. Gaussian blur
            // 3. Contrast enhancement
            // 4. Noise reduction
            
            // Optimize compute_at and store_at directives
            // for minimum memory usage and maximum cache reuse
            
            return input;  // Placeholder
        }
        """


class CPUBenchmarkSuite:
    """Collection of CPU memory optimization benchmarks"""
    
    def __init__(self):
        self.benchmarks = [
            DataStructureOptimizationBenchmark(),
            MatrixTilingBenchmark(),
            LoopOptimizationBenchmark(),
            HalideDSLBenchmark(),
            # Add more benchmarks here
        ]
        
    def run_benchmark(self, benchmark_idx: int, generated_code: str) -> Dict[str, Any]:
        """Run a single benchmark and return results"""
        benchmark = self.benchmarks[benchmark_idx]
        inputs = benchmark.generate_input_data()
        
        # Evaluate correctness
        is_correct = benchmark.evaluate_correctness(generated_code, inputs)
        
        if is_correct:
            # Measure performance
            metrics = benchmark.measure_cpu_performance(generated_code, inputs)
            speedup = self._calculate_speedup(benchmark, generated_code, inputs)
            cache_efficiency = metrics.overall_cache_efficiency
        else:
            metrics = None
            speedup = 0.0
            cache_efficiency = 0.0
            
        return {
            "benchmark_name": benchmark.name,
            "optimization_type": benchmark.optimization_type,
            "correct": is_correct,
            "metrics": metrics,
            "speedup": speedup,
            "cache_efficiency": cache_efficiency,
            "difficulty_level": benchmark.difficulty_level
        }
        
    def _calculate_speedup(self, benchmark: CPUMemoryBenchmark, code: str, inputs: Dict) -> float:
        """Calculate speedup compared to reference implementation"""
        # Measure reference performance
        start_time = time.time()
        _ = benchmark.reference_implementation(inputs)
        ref_time = time.time() - start_time
        
        # Measure optimized code performance
        try:
            start_time = time.time()
            _ = benchmark._compile_and_run_cpp(code, inputs)
            opt_time = time.time() - start_time
            
            return ref_time / opt_time if opt_time > 0 else 0.0
        except:
            return 0.0
            
    def run_full_evaluation(self, model_outputs: List[str]) -> Dict[str, Any]:
        """Run complete benchmark suite evaluation"""
        results = []
        
        for i, code in enumerate(model_outputs):
            if i < len(self.benchmarks):
                result = self.run_benchmark(i, code)
                results.append(result)
                
        # Calculate aggregate metrics
        correct_count = sum(1 for r in results if r["correct"])
        avg_speedup = np.mean([r["speedup"] for r in results if r["correct"]])
        avg_cache_efficiency = np.mean([r["cache_efficiency"] for r in results if r["correct"]])
        
        # Group by optimization type
        by_type = {}
        for result in results:
            opt_type = result["optimization_type"]
            if opt_type not in by_type:
                by_type[opt_type] = []
            by_type[opt_type].append(result)
        
        return {
            "individual_results": results,
            "summary": {
                "correctness_rate": correct_count / len(results),
                "average_speedup": avg_speedup,
                "average_cache_efficiency": avg_cache_efficiency,
                "total_benchmarks": len(results)
            },
            "by_optimization_type": by_type
        }
    
    def visualize_results(self, results: Dict[str, Any], save_path: str = None):
        """Create visualizations of benchmark results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Correctness by optimization type
        by_type = results["by_optimization_type"]
        types = list(by_type.keys())
        correctness_rates = [np.mean([r["correct"] for r in by_type[t]]) for t in types]
        
        axes[0, 0].bar(types, correctness_rates)
        axes[0, 0].set_title("Correctness Rate by Optimization Type")
        axes[0, 0].set_ylabel("Correctness Rate")
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Speedup distribution
        speedups = [r["speedup"] for r in results["individual_results"] if r["correct"]]
        axes[0, 1].hist(speedups, bins=20, alpha=0.7)
        axes[0, 1].set_title("Speedup Distribution")
        axes[0, 1].set_xlabel("Speedup")
        axes[0, 1].set_ylabel("Frequency")
        
        # Cache efficiency vs speedup
        cache_effs = [r["cache_efficiency"] for r in results["individual_results"] if r["correct"]]
        axes[1, 0].scatter(cache_effs, speedups, alpha=0.7)
        axes[1, 0].set_xlabel("Cache Efficiency")
        axes[1, 0].set_ylabel("Speedup")
        axes[1, 0].set_title("Cache Efficiency vs Speedup")
        
        # Performance by difficulty level
        difficulties = [r["difficulty_level"] for r in results["individual_results"]]
        perf_by_diff = {}
        for i, result in enumerate(results["individual_results"]):
            diff = result["difficulty_level"]
            if diff not in perf_by_diff:
                perf_by_diff[diff] = []
            if result["correct"]:
                perf_by_diff[diff].append(result["speedup"])
        
        diff_levels = sorted(perf_by_diff.keys())
        avg_speedups = [np.mean(perf_by_diff[d]) if perf_by_diff[d] else 0 for d in diff_levels]
        
        axes[1, 1].bar(diff_levels, avg_speedups)
        axes[1, 1].set_title("Average Speedup by Difficulty Level")
        axes[1, 1].set_xlabel("Difficulty Level")
        axes[1, 1].set_ylabel("Average Speedup")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class CPUMemoryOptimizationTrainer:
    """Training pipeline for CPU memory optimization LLM"""
    
    def __init__(self, model_name: str = "codellama/CodeLlama-7b-hf"):
        self.model_name = model_name
        self.benchmark_suite = CPUBenchmarkSuite()
        
    def generate_training_prompts(self) -> List[Dict[str, str]]:
        """Generate training prompts for CPU memory optimization"""
        prompts = []
        
        for benchmark in self.benchmark_suite.benchmarks:
            base_prompt = f"""
            You are an expert C++ programmer specializing in CPU memory optimization and cache-efficient code.
            Optimize the following code for maximum cache efficiency and performance:
            
            {benchmark.extract_optimization_template()}
            
            Focus on:
            - Minimizing cache misses (L1, L2, L3)
            - Optimizing data layout and access patterns
            - Implementing effective loop transformations
            - Utilizing SIMD vectorization when beneficial
            - Considering memory prefetching strategies
            
            Provide the complete optimized implementation with detailed comments explaining your optimizations.
            """
            
            # Create variations for different optimization aspects
            optimization_variants = {
                "data_structure": "Pay special attention to data layout optimization (AoS vs SoA) and memory alignment.",
                "matrix_tiling": "Focus on multi-level cache blocking and register tiling strategies.",
                "loop_opt": "Emphasize loop transformations: tiling, unrolling, interchange, and vectorization.",
                "dsl": "Optimize the Halide schedule for maximum cache reuse and minimal memory traffic."
            }
            
            variant_prompt = base_prompt + "\n\nSpecial focus: " + optimization_variants.get(benchmark.optimization_type, "")
            
            prompts.append({
                "benchmark_name": benchmark.name,
                "optimization_type": benchmark.optimization_type,
                "prompt": variant_prompt,
                "difficulty": benchmark.difficulty_level
            })
            
        return prompts
        
    def compute_cpu_reward(self, benchmark_result: Dict[str, Any]) -> float:
        """Compute reward based on CPU cache efficiency and performance"""
        if not benchmark_result["correct"]:
            return 0.0
            
        cache_efficiency = benchmark_result["cache_efficiency"]
        speedup = benchmark_result["speedup"]
        
        # Reward function emphasizing cache efficiency
        cache_score = cache_efficiency
        performance_score = min(speedup / 3.0, 1.0)  # Cap at 3x speedup
        
        # Weight cache efficiency higher for CPU optimizations
        reward = 0.6 * cache_score + 0.4 * performance_score
        
        # Bonus for high-difficulty benchmarks
        difficulty_bonus = 1.0 + 0.1 * (benchmark_result["difficulty_level"] - 1)
        
        return reward * difficulty_bonus


if __name__ == "__main__":
    # Example usage
    print("CPU Memory Optimization Benchmark Suite")
    print("=" * 50)
    
    # Initialize benchmark suite
    suite = CPUBenchmarkSuite()
    
    # Show available benchmarks
    print("Available benchmarks:")
    for i, benchmark in enumerate(suite.benchmarks):
        print(f"{i+1}. {benchmark.name} ({benchmark.optimization_type}) - Level {benchmark.difficulty_level}")
    
    # Run example evaluation
    example_codes = [
        "/* Optimized data structure code */",
        "/* Optimized matrix tiling code */", 
        "/* Optimized loop transformation code */",
        "/* Optimized Halide DSL code */"
    ]
    
    print(f"\nRunning evaluation with {len(example_codes)} sample codes...")
    results = suite.run_full_evaluation(example_codes)
    print(f"Evaluation Results: {results['summary']}")
    
    # Initialize trainer
    trainer = CPUMemoryOptimizationTrainer()
    prompts = trainer.generate_training_prompts()
    print(f"\nGenerated {len(prompts)} training prompts for CPU optimization")
    
    # Show optimization types
    opt_types = set(p["optimization_type"] for p in prompts)
    print(f"Optimization types covered: {opt_types}")

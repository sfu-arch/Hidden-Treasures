cmake_minimum_required(VERSION 3.18)
project(cuda_convolution_lab LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui)

# Set compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -fopenmp")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -lineinfo")

# Set CUDA architecture - adjust for your GPU
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)  # RTX 40 series (RTX 4090)
    # Uncomment appropriate line for your GPU:
    # set(CMAKE_CUDA_ARCHITECTURES 61)  # GTX 10 series
    # set(CMAKE_CUDA_ARCHITECTURES 86)  # RTX 30 series  
    # set(CMAKE_CUDA_ARCHITECTURES 75)  # RTX 20 series
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)

# Define source files
set(COMMON_SOURCES
    src/main.cpp
)

set(CPU_SOURCES
    src/cpu/conv_cpu.cpp
)

set(GPU_SOURCES
    src/gpu/conv_naive.cu
)

set(TEST_SOURCES
    tests/unit_tests.cpp
)

# Create test CUDA file if it doesn't exist
set(TEST_CUDA_FILE ${CMAKE_CURRENT_SOURCE_DIR}/test_cuda.cu)
if(NOT EXISTS ${TEST_CUDA_FILE})
    file(WRITE ${TEST_CUDA_FILE}
"#include <cuda_runtime.h>
#include <iostream>

__global__ void hello() {
    printf(\"Hello from GPU thread %d\\n\", threadIdx.x);
}

int main() {
    hello<<<1, 5>>>();
    cudaDeviceSynchronize();
    std::cout << \"CUDA test completed successfully!\" << std::endl;
    return 0;
}
")
endif()

# Main benchmark executable
add_executable(conv_benchmark
    ${COMMON_SOURCES}
    ${CPU_SOURCES}
    ${GPU_SOURCES}
)

# Simple CUDA test executable
add_executable(test_cuda test_cuda.cu)

# Unit test executable
add_executable(unit_tests
    ${TEST_SOURCES}
    ${CPU_SOURCES}
    ${GPU_SOURCES}
)

# Link libraries
target_link_libraries(conv_benchmark 
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    stdc++fs
)

target_link_libraries(test_cuda 
    ${CUDA_LIBRARIES}
)

target_link_libraries(unit_tests
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    stdc++fs
)

# Set properties for CUDA compilation
set_target_properties(conv_benchmark PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

set_target_properties(test_cuda PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
)

set_target_properties(unit_tests PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Add custom targets for common tasks
add_custom_target(check-cuda
    COMMAND nvcc --version
    COMMAND nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
    COMMENT "Checking CUDA environment"
)

add_custom_target(benchmark
    COMMAND conv_benchmark --benchmark --verbose
    DEPENDS conv_benchmark
    COMMENT "Running performance benchmarks"
)

add_custom_target(profile
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tools/profile.sh
    DEPENDS conv_benchmark
    COMMENT "Profiling with Nsight Compute"
)

# Print configuration information
message(STATUS "CUDA Convolution Lab Configuration:")
message(STATUS "  CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "  CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  OpenCV_VERSION: ${OpenCV_VERSION}")
message(STATUS "  CUDA_VERSION: ${CUDA_VERSION}")

# Installation rules
install(TARGETS conv_benchmark test_cuda unit_tests
    RUNTIME DESTINATION bin
)

install(DIRECTORY data/
    DESTINATION share/cuda-convolution-lab/data
)

install(DIRECTORY tools/
    DESTINATION share/cuda-convolution-lab/tools
    FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                     GROUP_READ GROUP_EXECUTE
                     WORLD_READ WORLD_EXECUTE
)

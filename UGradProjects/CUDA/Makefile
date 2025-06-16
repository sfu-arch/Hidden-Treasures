# CUDA Convolution Lab Makefile
# Builds all CPU and GPU implementations with proper optimization flags

# Compiler and tool configuration
NVCC = nvcc
CXX = g++
PYTHON = python3

# CUDA configuration - adjust compute capability for your GPU
CUDA_ARCH = sm_75  # RTX 20/30 series (adjust for your GPU: sm_61 for GTX 10xx, sm_86 for RTX 30xx, sm_89 for RTX 40xx)
CUDA_VERSION = $(shell nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')

# Directory structure
SRC_DIR = src
BUILD_DIR = build
DATA_DIR = data
TOOLS_DIR = tools
COMMON_DIR = $(SRC_DIR)/common
CPU_DIR = $(SRC_DIR)/cpu  
GPU_DIR = $(SRC_DIR)/gpu
TEST_DIR = tests

# Compiler flags
NVCC_FLAGS = -std=c++17 -O3 -arch=$(CUDA_ARCH) -lineinfo
NVCC_FLAGS += -Xcompiler -Wall,-Wextra,-O3,-fopenmp
NVCC_FLAGS += -I$(COMMON_DIR) -I/usr/include/opencv4
CXX_FLAGS = -std=c++17 -O3 -Wall -Wextra -fopenmp -I$(COMMON_DIR)

# Libraries
LIBS = -lcuda -lcudart -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui
LIBS += -lstdc++fs  # For filesystem operations

# Source files
CPU_SOURCES = $(CPU_DIR)/conv_cpu.cpp
GPU_SOURCES = $(wildcard $(GPU_DIR)/*.cu)
COMMON_SOURCES = $(SRC_DIR)/main.cpp
TEST_SOURCES = $(TEST_DIR)/unit_tests.cpp

# Object files
CPU_OBJECTS = $(CPU_SOURCES:%.cpp=$(BUILD_DIR)/%.o)
GPU_OBJECTS = $(GPU_SOURCES:%.cu=$(BUILD_DIR)/%.o)
COMMON_OBJECTS = $(COMMON_SOURCES:%.cpp=$(BUILD_DIR)/%.o)
TEST_OBJECTS = $(TEST_SOURCES:%.cpp=$(BUILD_DIR)/%.o)

# Executables
MAIN_EXEC = conv_benchmark
TEST_EXEC = test_cuda
UNIT_TEST_EXEC = unit_tests

# Default target
.PHONY: all clean test profile setup help

all: setup $(MAIN_EXEC) $(TEST_EXEC) $(UNIT_TEST_EXEC)

# Create build directories
setup:
	@echo "🛠️  Setting up build environment..."
	@mkdir -p $(BUILD_DIR)/$(SRC_DIR)
	@mkdir -p $(BUILD_DIR)/$(CPU_DIR)
	@mkdir -p $(BUILD_DIR)/$(GPU_DIR)
	@mkdir -p $(BUILD_DIR)/$(TEST_DIR)
	@echo "✅ Build directories created"

# Main benchmark executable
$(MAIN_EXEC): $(COMMON_OBJECTS) $(CPU_OBJECTS) $(GPU_OBJECTS)
	@echo "🔗 Linking main benchmark executable..."
	$(NVCC) $(NVCC_FLAGS) $^ -o $@ $(LIBS)
	@echo "✅ Built $(MAIN_EXEC)"

# Simple CUDA test executable
$(TEST_EXEC): $(BUILD_DIR)/test_cuda.o
	@echo "🔗 Linking CUDA test executable..."
	$(NVCC) $(NVCC_FLAGS) $^ -o $@ $(LIBS)
	@echo "✅ Built $(TEST_EXEC)"

# Unit test executable
$(UNIT_TEST_EXEC): $(TEST_OBJECTS) $(CPU_OBJECTS) $(GPU_OBJECTS)
	@echo "🔗 Linking unit test executable..."
	$(NVCC) $(NVCC_FLAGS) $^ -o $@ $(LIBS)
	@echo "✅ Built $(UNIT_TEST_EXEC)"

# Compile CUDA source files
$(BUILD_DIR)/%.o: %.cu
	@echo "🔥 Compiling CUDA: $<"
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Compile C++ source files
$(BUILD_DIR)/%.o: %.cpp
	@echo "⚙️  Compiling C++: $<"
	@mkdir -p $(dir $@)
	$(CXX) $(CXX_FLAGS) -c $< -o $@

# Simple CUDA test program
$(BUILD_DIR)/test_cuda.o: test_cuda.cu
	@echo "🔥 Compiling CUDA test..."
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Create simple CUDA test file if it doesn't exist
test_cuda.cu:
	@echo "📝 Creating simple CUDA test file..."
	@echo '#include <cuda_runtime.h>' > test_cuda.cu
	@echo '#include <iostream>' >> test_cuda.cu
	@echo '' >> test_cuda.cu
	@echo '__global__ void hello() {' >> test_cuda.cu
	@echo '    printf("Hello from GPU thread %d\\n", threadIdx.x);' >> test_cuda.cu
	@echo '}' >> test_cuda.cu
	@echo '' >> test_cuda.cu
	@echo 'int main() {' >> test_cuda.cu
	@echo '    hello<<<1, 5>>>();' >> test_cuda.cu
	@echo '    cudaDeviceSynchronize();' >> test_cuda.cu
	@echo '    std::cout << "CUDA test completed successfully!" << std::endl;' >> test_cuda.cu
	@echo '    return 0;' >> test_cuda.cu
	@echo '}' >> test_cuda.cu

# Test targets
test: $(TEST_EXEC) $(UNIT_TEST_EXEC)
	@echo "🧪 Running CUDA environment test..."
	./$(TEST_EXEC)
	@echo "🧪 Running unit tests..."
	./$(UNIT_TEST_EXEC)

# Profiling target
profile: $(MAIN_EXEC)
	@echo "📊 Starting profiling session..."
	@if [ -f "$(TOOLS_DIR)/profile.sh" ]; then \
		chmod +x $(TOOLS_DIR)/profile.sh; \
		$(TOOLS_DIR)/profile.sh; \
	else \
		echo "⚠️  Profiling script not found. Creating basic profile..."; \
		ncu --target-processes all --kernel-name-base function ./$(MAIN_EXEC); \
	fi

# Performance benchmark
benchmark: $(MAIN_EXEC)
	@echo "🏃‍♂️ Running performance benchmarks..."
	@./$(MAIN_EXEC) --benchmark --verbose

# Clean build artifacts
clean:
	@echo "🧹 Cleaning build artifacts..."
	rm -rf $(BUILD_DIR)
	rm -f $(MAIN_EXEC) $(TEST_EXEC) $(UNIT_TEST_EXEC)
	rm -f test_cuda.cu
	rm -f *.nsys-rep *.ncu-rep
	@echo "✅ Clean completed"

# Development helpers
debug: NVCC_FLAGS += -g -G -DDEBUG
debug: CXX_FLAGS += -g -DDEBUG
debug: all

release: NVCC_FLAGS += -DNDEBUG
release: CXX_FLAGS += -DNDEBUG
release: all

# Check CUDA environment
check-cuda:
	@echo "🔍 Checking CUDA environment..."
	@echo "CUDA Version: $(CUDA_VERSION)"
	@echo "Compute Capability: $(CUDA_ARCH)"
	@nvcc --version
	@nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv

# Install dependencies (Ubuntu/Debian)
install-deps:
	@echo "📦 Installing dependencies..."
	sudo apt update
	sudo apt install -y libopencv-dev cmake build-essential
	@echo "✅ Dependencies installed"

# Generate compilation database for language servers
compile-commands:
	@echo "📋 Generating compile commands..."
	bear -- make clean all

# Help target
help:
	@echo "🚀 CUDA Convolution Lab - Available targets:"
	@echo ""
	@echo "  all             - Build all executables (default)"
	@echo "  setup           - Create build directories"
	@echo "  test            - Run CUDA environment and unit tests"
	@echo "  benchmark       - Run performance benchmarks"
	@echo "  profile         - Profile with Nsight Compute"
	@echo "  clean           - Remove build artifacts"
	@echo "  debug           - Build with debug information"
	@echo "  release         - Build optimized release version"
	@echo "  check-cuda      - Verify CUDA environment"
	@echo "  install-deps    - Install system dependencies"
	@echo "  help            - Show this help message"
	@echo ""
	@echo "📋 Configuration:"
	@echo "  CUDA_ARCH       = $(CUDA_ARCH)"
	@echo "  CUDA_VERSION    = $(CUDA_VERSION)"
	@echo ""
	@echo "💡 Quick start:"
	@echo "  make check-cuda    # Verify CUDA setup"
	@echo "  make test          # Test basic functionality"
	@echo "  make benchmark     # Run performance tests"

# Phony targets to avoid conflicts with file names
.PHONY: all clean test profile benchmark debug release check-cuda install-deps compile-commands help setup

# Print variables for debugging
print-%:
	@echo $* = $($*)

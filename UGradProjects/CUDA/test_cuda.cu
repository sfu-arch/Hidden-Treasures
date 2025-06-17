#include <cuda_runtime.h>
#include <iostream>

__global__ void hello() {
    printf("Hello from GPU thread %d\n", threadIdx.x);
}

int main() {
    hello<<<1, 5>>>();
    cudaDeviceSynchronize();
    std::cout << "CUDA test completed successfully!" << std::endl;
    return 0;
}

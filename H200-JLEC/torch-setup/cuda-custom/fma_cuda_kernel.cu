// Fused multiply-add: y = x * w + b
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

template <typename scalar_t>
__global__ void fma_kernel(const scalar_t* __restrict__ x,
                           const scalar_t* __restrict__ w,
                           const scalar_t* __restrict__ b,
                           scalar_t* __restrict__ y,
                           int64_t n) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (i < n) y[i] = x[i] * w[i] + b[i];
}

torch::Tensor fma_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && b.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(x.sizes() == w.sizes() && x.sizes() == b.sizes(), "shape mismatch");

  auto x_contig = x.contiguous();
  auto w_contig = w.contiguous();
  auto b_contig = b.contiguous();

  auto y = torch::empty_like(x_contig);
  const int64_t n = x_contig.numel();
  const int threads = 256;
  const int64_t blocks = (n + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      x_contig.scalar_type(),
      "fma_cuda",
      [&] {
        fma_kernel<scalar_t><<<blocks, threads>>>(
            x_contig.data_ptr<scalar_t>(),
            w_contig.data_ptr<scalar_t>(),
            b_contig.data_ptr<scalar_t>(),
            y.data_ptr<scalar_t>(),
            n);
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

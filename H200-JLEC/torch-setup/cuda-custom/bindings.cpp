#include <torch/extension.h>

torch::Tensor fma_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b);

torch::Tensor fma_op(torch::Tensor x, torch::Tensor w, torch::Tensor b) {
  return fma_cuda(x, w, b);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fma", &fma_op, "Fused multiply-add (CUDA)");
}

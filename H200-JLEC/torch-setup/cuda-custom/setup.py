from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="fused_ops",
    version="0.1.0",
    ext_modules=[
        CUDAExtension(
            name="fused_ops_cuda",
            sources=["bindings.cpp", "fma_cuda_kernel.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math", "-diag-suppress=3189"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)

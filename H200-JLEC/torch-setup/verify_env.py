#!/usr/bin/env python3
"""Verify the llmdev environment end-to-end on H200 NVL.

Run inside the llmdev venv:  python verify_env.py
Checks: torch+CUDA, GPU/NVLink topology, Triton kernel correctness,
and live nvcc compilation of a custom CUDA op (proves the toolchain).
"""
import subprocess
import sys


def hr(title):
    print(f"\n{'='*60}\n{title}\n{'='*60}")


def check_torch():
    hr("torch / CUDA")
    import torch
    print("torch        :", torch.__version__)
    print("torch.cuda   :", torch.version.cuda)
    print("cuDNN        :", torch.backends.cudnn.version())
    print("NCCL         :", ".".join(map(str, torch.cuda.nccl.version())))
    print("device count :", torch.cuda.device_count())
    assert torch.cuda.is_available(), "CUDA not available to torch"
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU{i}: {p.name}  sm_{p.major}{p.minor}  {p.total_memory/1e9:.0f} GB")
    assert torch.cuda.get_device_capability(0) == (9, 0), \
        "Expected Hopper sm_90 (H200)"
    return torch


def check_topology():
    hr("NVLink topology (read NV# = bridged pair; SYS/PHB = PCIe)")
    try:
        out = subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True)
        print(out)
        if "NV" not in out:
            print("WARNING: no NVLink (NV#) links seen — check bridges / `nvidia-smi nvlink --status`")
    except Exception as e:  # noqa: BLE001
        print("could not run nvidia-smi topo -m:", e)


def check_triton(torch):
    hr("Triton kernel (softmax) vs torch")
    import triton
    import triton.language as tl
    print("triton:", triton.__version__)

    @triton.jit
    def _softmax(out_ptr, in_ptr, in_s, out_s, n_cols, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < n_cols
        x = tl.load(in_ptr + row * in_s + cols, mask=mask, other=-float("inf"))
        x = x - tl.max(x, axis=0)
        num = tl.exp(x)
        y = num / tl.sum(num, axis=0)
        tl.store(out_ptr + row * out_s + cols, y, mask=mask)

    x = torch.randn(2048, 781, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(x.shape[1])
    _softmax[(x.shape[0],)](out, x, x.stride(0), out.stride(0), x.shape[1], BLOCK=BLOCK)
    torch.testing.assert_close(out, torch.softmax(x, axis=1), rtol=1e-4, atol=1e-4)
    print("Triton softmax matches torch  ✓")


def check_cuda_extension(torch):
    hr("Live CUDA op compile via nvcc (load_inline)")
    from torch.utils.cpp_extension import load_inline
    cuda_src = r"""
    #include <torch/extension.h>
    __global__ void addk(float* x, float k, int n){
        int i = blockIdx.x*blockDim.x + threadIdx.x;
        if (i < n) x[i] += k;
    }
    torch::Tensor add_scalar(torch::Tensor x, double k){
        TORCH_CHECK(x.is_cuda(), "x must be CUDA");
        auto y = x.clone();
        int n = y.numel();
        addk<<<(n+255)/256, 256>>>(y.data_ptr<float>(), (float)k, n);
        return y;
    }
    """
    cpp_src = "torch::Tensor add_scalar(torch::Tensor x, double k);"
    mod = load_inline(name="verify_addk", cpp_sources=cpp_src,
                      cuda_sources=cuda_src, functions=["add_scalar"],
                      with_cuda=True, verbose=False)
    x = torch.randn(4096, device="cuda")
    torch.testing.assert_close(mod.add_scalar(x, 2.5), x + 2.5)
    print("nvcc-compiled custom op runs and is correct  ✓")


def check_flash():
    hr("FlashAttention")
    try:
        import flash_attn
        print("flash-attn:", flash_attn.__version__, " ✓")
    except Exception as e:  # noqa: BLE001
        print("flash-attn not importable:", e)


def main():
    torch = check_torch()
    check_topology()
    check_triton(torch)
    check_cuda_extension(torch)
    check_flash()
    print("\nAll core checks passed.\n")


if __name__ == "__main__":
    sys.exit(main())

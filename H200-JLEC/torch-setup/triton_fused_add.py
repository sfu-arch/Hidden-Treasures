"""A real Triton kernel: fused (x * scale + residual) with vectorized I/O,
autotuned over block size. Pattern you'll reuse for fused LLM kernels.

    python examples/triton_fused_add.py
"""
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[triton.Config({"BLOCK": b}, num_warps=w)
             for b in (1024, 2048, 4096) for w in (4, 8)],
    key=["n_elements"],
)
@triton.jit
def _fused_scale_add(x_ptr, r_ptr, out_ptr, scale, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    r = tl.load(r_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x * scale + r, mask=mask)


def fused_scale_add(x, r, scale):
    out = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)  # noqa: E731
    _fused_scale_add[grid](x, r, out, scale, n)
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    r = torch.randn_like(x)
    out = fused_scale_add(x, r, 1.7)
    torch.testing.assert_close(out, x * 1.7 + r, rtol=2e-2, atol=2e-2)
    print("correctness ✓")

    ms = triton.testing.do_bench(lambda: fused_scale_add(x, r, 1.7))
    gbps = 3 * x.numel() * x.element_size() / (ms * 1e-3) / 1e9  # 2 read + 1 write
    print(f"{ms:.3f} ms   ~{gbps:.0f} GB/s effective")

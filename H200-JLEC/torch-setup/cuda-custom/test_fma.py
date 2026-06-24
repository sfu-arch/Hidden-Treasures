import torch
import fused_ops_cuda


def main() -> None:
    assert torch.cuda.is_available(), "CUDA is not available"

    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        x = torch.randn(1 << 20, device="cuda", dtype=dtype)
        w = torch.randn_like(x)
        b = torch.randn_like(x)

        y = fused_ops_cuda.fma(x, w, b)
        ref = x * w + b

        torch.testing.assert_close(y, ref, rtol=2e-2, atol=2e-2)
        print(f"{dtype}: OK")

    print("fused_ops_cuda.fma verified on", torch.cuda.get_device_name(0))


if __name__ == "__main__":
    main()

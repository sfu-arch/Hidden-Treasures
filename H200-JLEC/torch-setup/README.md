# H200 NVL — LLM Dev / Train / Serve Environment

A reproducible setup for **custom ops (Triton + CUDA C++)**, **LLM fine-tuning /
training**, and **serving**, built on top of your *managed* CUDA toolkit and a
PCIe **H200 NVL** box (Hopper sm_90, bridged NVLink pairs, no NVSwitch).

## Why three environments, not one

| env | what's in it | torch |
|-----|--------------|-------|
| `llmdev`  | Triton, CUDA C++ extensions, transformers/peft/trl/deepspeed, flash-attn | **yours**, pinned to system CUDA |
| `vllm`    | vLLM inference server | vLLM's own |
| `sglang`  | SGLang inference server | SGLang's own |

vLLM and SGLang each **hard-pin a specific torch build**. If you install them
next to your dev torch they'll overwrite it — and every custom `.so` you
compiled is ABI-locked to the torch it was built against, so it would then fail
to load. Keeping serving in separate venvs means your kernels and your servers
never fight. This is the single most important structural decision here.

## Layout

```
setup-dev.sh                 # build the llmdev env (detects your CUDA -> matched torch)
setup-serve.sh               # build vllm + sglang envs (isolated)
verify_env.py                # torch+CUDA, NVLink topo, Triton kernel, live nvcc compile
examples/
  triton_fused_add.py        # autotuned Triton kernel (correctness + bandwidth)
  sft_lora.py                # topology-aware LoRA SFT skeleton
  cuda_ext/                  # full setuptools CUDA op (build + test)
    fma_cuda_kernel.cu
    bindings.cpp
    setup.py
    test_fma.py
```

## Quick start

```bash
srun -p profali -A profali --gres=shard:1 -c 8 --pty bash

# 0. Make sure your managed CUDA is on PATH (nvcc --version works) and topology is sane:
nvidia-smi topo -m            # note which GPU indices read NV# (a bridged pair)

# 1. Dev/train env (auto-matches torch to your nvcc)
bash setup-dev.sh
source ~/envs/llmdev/bin/activate
python verify_env.py          # should end with "All core checks passed."

# 2. Prove the custom-op toolchain end-to-end
cd cuda-custom
TORCH_CUDA_ARCH_LIST=9.0 uv pip install -e . --no-build-isolation   ∞
python3 test_fma.py

# 3. Hugging face fine-tuning example. Data parallel training across NVLink pairs, tensor/pipeline parallelism inside pairs.
hf auth # Choose login using token, Get token from https://huggingface.co/settings/tokens
python sft_lora.py # Mistral-7B example, swap MODEL as needed. Note the launch pattern for NVLink topology.



# 3. Serving envs (separate)
bash setup-serve.sh
```

## The CUDA-matching rule (the part your *managed* CUDA actually affects)

For **running** torch, the wheel is self-contained and your system CUDA is
irrelevant. For **building custom CUDA ops**, the system `nvcc` compiles against
torch's headers, so `nvcc`'s version must match `torch.version.cuda` — same
major, ideally same minor. `setup-dev.sh` picks the torch wheel channel straight
from `nvcc --version` to keep them aligned. Override with
`TORCH_CHANNEL=cu128 bash setup-dev.sh` if you want a specific one.

Triton needs no toolkit — it JIT-compiles via LLVM at runtime, and ships *inside*
the torch wheel. Do **not** `pip install triton` separately; that desyncs it from
torch and breaks `torch.compile`/inductor.

## Topology rules for this box (bridged NVLink, no switch)

NVLink only spans a physical bridge. Anything crossing pairs goes over PCIe.
`nvidia-smi topo -m` is ground truth: `NV#` = NVLink pair, `SYS/PHB/PXB` = PCIe.

- **Training:** keep tensor/pipeline parallel *inside* a bridged pair; spread
  data-parallel / FSDP replicas *across* pairs. `examples/sft_lora.py` shows the
  launch pattern (`CUDA_VISIBLE_DEVICES=<bridged pair>`).
- **Serving:** one TP group per bridged pair; scale out with one replica per pair
  behind a router, never one TP group spanning pairs. See `setup-serve.sh`.
- **Cross-pair P2P** may be blocked by IOMMU/ACS, forcing NCCL through host
  memory. If cross-pair collectives are slow, check
  `sudo lspci -vvv | grep -i acsctl` and disable ACS in BIOS for PCIe P2P.

## FlashAttention on H200

`setup-dev.sh` installs FlashAttention-2 (works, `attn_implementation=
"flash_attention_2"`). For peak Hopper throughput build **FlashAttention-3**:

```bash
git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention/hopper && python setup.py install
```

## Serving cheatsheet

```bash
# vLLM on bridged pair 0,1
source ~/envs/vllm/bin/activate
CUDA_VISIBLE_DEVICES=0,1 vllm serve <model> --tensor-parallel-size 2 --port 8000

# SGLang on bridged pair 0,1
source ~/envs/sglang/bin/activate
CUDA_VISIBLE_DEVICES=0,1 python -m sglang.launch_server --model-path <model> --tp 2 --port 30000
```

## Notes / assumptions

- Pins are intentionally light; the resolver pulls current mutually-compatible
  versions. If you need a frozen set, `uv pip freeze > requirements.lock` after a
  good build and reuse it.
- Python 3.12 is the default (broadest wheel coverage). Override with `PYVER=3.11`.
- If you later add TensorRT-LLM or Torch-TensorRT, give it **its own** env too and
  pin torch to that release's support matrix — same isolation logic as vLLM/SGLang.

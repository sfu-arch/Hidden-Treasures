#!/usr/bin/env bash
# ============================================================================
# llmdev — custom ops (Triton + CUDA C++) + LLM fine-tuning/training
# Uses YOUR managed CUDA toolkit (nvcc) so compiled kernels match it.
# Target: H200 NVL (PCIe), Hopper sm_90, Ubuntu 24.04.
# ============================================================================
set -euo pipefail

ENV_DIR="${ENV_DIR:-$HOME/envs/llmdev}"
PYVER="${PYVER:-3.12}"

# --- 1. The system CUDA toolkit decides the torch wheel ---------------------
# For custom CUDA extensions, torch.version.cuda must match your nvcc
# (same major; same minor is ideal). We pick the wheel channel from nvcc.
if ! command -v nvcc >/dev/null 2>&1; then
  echo "ERROR: nvcc not on PATH. Load your managed CUDA module first" >&2
  echo "       (e.g. 'module load cuda/12.8' or export PATH=/usr/local/cuda/bin:\$PATH)" >&2
  exit 1
fi
CUDA_MM="$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')"
echo ">> system CUDA toolkit: $CUDA_MM"

# Override anytime with:  TORCH_CHANNEL=cu128 ./setup-dev.sh
if [[ -z "${TORCH_CHANNEL:-}" ]]; then
  case "$CUDA_MM" in
    13.0)      TORCH_CHANNEL=cu130 ;;
    12.9)      TORCH_CHANNEL=cu129 ;;
    12.8)      TORCH_CHANNEL=cu128 ;;
    12.6|12.7) TORCH_CHANNEL=cu126 ;;
    12.4|12.5) TORCH_CHANNEL=cu124 ;;
    *) echo "Unmapped CUDA $CUDA_MM — set TORCH_CHANNEL=cuXXX manually" >&2; exit 1 ;;
  esac
fi
echo ">> torch wheel channel: $TORCH_CHANNEL"
# If torch's published CUDA != your nvcc minor, extension builds still work via
# CUDA minor-version compatibility, but exact match avoids surprises.

# --- 2. uv (fast, reproducible) ---------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # shellcheck disable=SC1090
  source "$HOME/.local/bin/env" 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"
fi

uv venv "$ENV_DIR" --python "$PYVER"
# shellcheck disable=SC1090
source "$ENV_DIR/bin/activate"

# --- 3. torch matched to the toolkit (Triton ships inside the wheel) --------
# NOTE: do NOT 'pip install triton' separately — torch pins a specific Triton
# and a standalone install will break torch.compile / inductor.
uv pip install torch torchvision torchaudio \
  --index-url "https://download.pytorch.org/whl/$TORCH_CHANNEL"

# --- 4. native-build toolchain for custom ops -------------------------------
uv pip install ninja cmake packaging wheel setuptools pybind11

# Build only for Hopper -> fast compiles, smaller .so. Persist into the env.
if ! grep -q TORCH_CUDA_ARCH_LIST "$ENV_DIR/bin/activate"; then
  echo 'export TORCH_CUDA_ARCH_LIST="9.0"' >> "$ENV_DIR/bin/activate"
fi
export TORCH_CUDA_ARCH_LIST="9.0"

# --- 5. training / fine-tuning stack ----------------------------------------
uv pip install \
  transformers datasets tokenizers accelerate peft trl deepspeed \
  bitsandbytes safetensors sentencepiece protobuf einops \
  liger-kernel wandb hf-transfer

# --- 6. FlashAttention-2 (needs torch + ninja already present) --------------
uv pip install flash-attn --no-build-isolation

cat <<EOF

============================================================
 llmdev ready at: $ENV_DIR
   source $ENV_DIR/bin/activate
   python verify_env.py
------------------------------------------------------------
 Optional FlashAttention-3 (Hopper-optimized, best on H200):
   git clone https://github.com/Dao-AILab/flash-attention
   cd flash-attention/hopper && python setup.py install
============================================================
EOF

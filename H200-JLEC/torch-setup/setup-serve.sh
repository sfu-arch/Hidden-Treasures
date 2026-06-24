#!/usr/bin/env bash
# ============================================================================
# Serving envs — vLLM and SGLang, each ISOLATED.
# They pin their own torch build; never install them into llmdev.
# ============================================================================
set -euo pipefail
PYVER="${PYVER:-3.12}"

command -v uv >/dev/null 2>&1 || { echo "install uv first (run setup-dev.sh)"; exit 1; }

# --- vLLM --------------------------------------------------------------------
uv venv "$HOME/envs/vllm" --python "$PYVER"
# shellcheck disable=SC1090
source "$HOME/envs/vllm/bin/activate"
uv pip install vllm            # pulls its own matched torch + flashinfer
deactivate
echo ">> vLLM env: $HOME/envs/vllm"

# --- SGLang ------------------------------------------------------------------
uv venv "$HOME/envs/sglang" --python "$PYVER"
# shellcheck disable=SC1090
source "$HOME/envs/sglang/bin/activate"
uv pip install "sglang[all]"   # pulls torch + flashinfer attention kernels
deactivate
echo ">> SGLang env: $HOME/envs/sglang"

cat <<'EOF'

============================================================
 Serving on H200 NVL pairs — KEEP TENSOR-PARALLEL INSIDE A BRIDGE.
 Read the bridged GPU indices from `nvidia-smi topo -m` (NV# pairs),
 then pin TP to exactly those two so NCCL rides NVLink, not PCIe.

 vLLM (one replica on the bridged pair 0,1):
   CUDA_VISIBLE_DEVICES=0,1 vllm serve <model> \
     --tensor-parallel-size 2 --port 8000

 SGLang (same idea):
   CUDA_VISIBLE_DEVICES=0,1 python -m sglang.launch_server \
     --model-path <model> --tp 2 --port 30000

 Scale out across pairs = one replica PER bridged pair behind a router,
 NOT one TP group spanning pairs (that would cross PCIe and stall).
   pair A: CUDA_VISIBLE_DEVICES=0,1  --port 8000
   pair B: CUDA_VISIBLE_DEVICES=2,3  --port 8001   # then load-balance
============================================================
EOF

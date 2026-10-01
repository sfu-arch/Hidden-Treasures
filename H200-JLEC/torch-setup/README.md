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


## SPGEMM or SPMM

Install uv, install requirements.txt


Relevant files
- download_datasets.py, gnn.py, spmm_mtx.py

```
# on cs-arch-32.cmpt.sfu.ca
 module load LIB/CUDA/13.0
python3 download_datasets.py --task spgemm --all
python3 download_datasets.py --task spmm --all
python3 download_datasets.py --task gnn --all
python gnn.py timing --root . --all --F 256
python3 spmm_mtx.py timing --ms-root HS --all
python3 spmm_mtx.py timing --ms-root MS --all
```

### gnn.py vs spmm_mtx.py — same kernel, different semantics

Both scripts call `torch.sparse.mm(CSR, dense)`, which hits the identical
cuSPARSE code path. What differs is what the operands mean and how they're
built:

- **gnn.py — message passing on a graph.** `A` is the graph adjacency:
  `edge_index` symmetrized (both directions concatenated), duplicates merged,
  all values forced to 1.0 (structure-only). Always square
  (`num_nodes x num_nodes`). `H` is a dense node-feature matrix
  `(num_nodes, F)`. So `A @ H` computes, per node, the unweighted sum of its
  neighbors' feature vectors — the aggregate step of one GCN-style layer
  (minus normalization and the weight matrix). The irregular, usually
  power-law degree distribution of these graphs is the workload's defining
  feature.
- **spmm_mtx.py — generic SpMM on SuiteSparse matrices.** `A` is the matrix
  as the dataset provides it: possibly rectangular, not symmetrized by the
  script (only the `.mtx` parser mirrors entries the file declares
  symmetric), real values when a raw `.mtx` is present. `B` is a random
  `(cols, K)` operand with no graph meaning. This measures SpMM as a
  numerical kernel on scientific-computing sparsity patterns ("regular"
  mesh/banded vs "power-law", per the `degree_type` tag).


### What cuSPARSE actually does underneath

`torch.sparse.mm(A_csr, H)` launches
two kernels — `cusparse::csr_partition_kernel` then
`cusparse::csrmm_alg2_kernel<..., long, long, float, ...>` — i.e. the
generic-API `cusparseSpMM` with **CSR_ALG2**, int64 indices, fp32 values.

**Dispatch.** PyTorch wraps the tensors in generic-API descriptors:
`cusparseCreateCsr` with `CUSPARSE_INDEX_64I` (PyTorch keeps int64 indices)
and `cusparseCreateDnMat` with row-major order (PyTorch dense layout), then
`cusparseSpMM_bufferSize` + workspace alloc + `cusparseSpMM` with ALG2 — the
variant NVIDIA recommends for row-major dense operands (default ALG1 is
tuned for column-major).

**Kernel 1 — `csr_partition_kernel` (load balancing).** Scans `indptr` and
partitions work into tiles of roughly equal *nonzero count* rather than
equal row count, writing tile boundaries into the workspace. Each thread
block gets similar work regardless of row length; long hub rows in power-law
graphs are split across blocks instead of serializing one block.

**Kernel 2 — `csrmm_alg2_kernel` (the SpMM).** Each block processes its nnz
partition against a tile of the F columns of H. Threads lie along
consecutive columns of H/C, so reads of `H[j,:]` segments and writes of
`C[i,:]` are coalesced. The sparse row's `(col_index, value)` pairs stream
sequentially through shared memory. For each nonzero `(i,j)` the kernel
gathers row `H[j,:]` — the random access whose locality is dictated entirely
by the graph's column ordering. Products accumulate in registers; each
`C[i,:]` is written once (beta=0, C never read).

** Roofline

 Per nonzero: `2F` flops vs 4 B value + 8 B
int64 column index + up to `4F` B of gathered H-row on an L2 miss. With no
reuse that is < 0.5 flop/byte — deep in the bandwidth-bound region. The only
lever is **L2 reuse of H rows**: well-ordered mesh-like matrices (MS /
"regular") hit lines already resident; power-law graphs with scattered
neighborhoods thrash L2 and measured DRAM bytes approach the worst case.
That single effect — L2 hit rate on the H gather — separates the dataset
clusters far more than flop throughput. Note also that cuSPARSE accepts
int32 indices (half the index traffic), but PyTorch sparse CSR standardizes
on int64.

### SpMM vs SpGEMM — what changes when H is sparse too

`torch.sparse.mm` dispatches on operand layout; the dense-H case above is
what makes it SpMM. The cases are entirely different algorithms:

- **H dense → SpMM** (`csrmm_alg2`, both scripts here): output is dense with
  known shape, allocated up front; cost is exactly `2*nnz(A)*F` flops;
  performance governed by L2 reuse of H gathers. The right primitive for GNN
  message passing, since node features genuinely are dense.
- **H sparse → SpGEMM** (`cusparseSpGEMM`, benchmarked by spgemm_mtx.py):
  output nnz is unknown until computed, so cuSPARSE runs a multi-phase
  scheme (symbolic/work-estimation pass, allocation, numeric pass). The
  inner loop is not gather-accumulate but *merging of sparse rows* — for
  each nonzero `A[i,k]`, row `B[k,:]` is fetched and merged into row `i` via
  hash tables / sorted merges in shared memory. Index-matching work
  dominates; much of the runtime produces no flops. Flop count is
  `2 * sum(products)`, which can wildly exceed output nnz — compression
  ratio becomes a first-order performance variable that doesn't exist in
  SpMM. Throughput lands an order of magnitude or more below SpMM on the
  same matrix (compare spgemm_mtx.csv vs spmm_mtx.csv on shared datasets).
- **A dense too → GEMM**: `torch.matmul` goes to cuBLAS, tensor cores light
  up, and you get the roofline's compute ceiling (the `gemm` anchor).

Hierarchy on the same hardware: dense GEMM (tensor cores, compute-bound) >>
SpMM (CUDA cores, bandwidth-bound on gathers) >> SpGEMM (CUDA cores, bound
by index-matching and irregular merges). Densifying A to use tensor cores
only pays off above roughly 10–20% density — far denser than any of these
graphs.


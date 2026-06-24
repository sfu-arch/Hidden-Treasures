This document was put together to help students get up and running on the shared H200 node in the JLEC. It was authored by Arrvindh Shriraman (one of the first users; **NOT THE SUPPORT STAFF**). If you want any specific upgrades or adding users send email to research-support@sfu.ca.  In general, this is a **self managed machine** and a **shared machine (meaning multiple labs and students/lab shared the GPUs)**. Meaning if your code is bad (e.g., OOM error in a model) it can affect others. So follow the instructions carefully below. 
- [Running JOBS](#running-jobs)
  - [Submitting work — `srun` / `salloc` / `sbatch`](#submitting-work--srun--salloc--sbatch)
  - [Watching the queue — `squeue`](#watching-the-queue--squeue)
  - [Inspecting a job — `scontrol show job`](#inspecting-a-job--scontrol-show-job)
  - [Managing your own jobs](#managing-your-own-jobs)
  - [Status \& history](#status--history)
  - [Two things that trip people up](#two-things-that-trip-people-up)
  - [Why three environments, not one](#why-three-environments-not-one)
 - [Toch Setup](#torch-setup)
  - [Quick start](#quick-start)
  - [The CUDA-matching rule (the part your *managed* CUDA actually affects)](#the-cuda-matching-rule-the-part-your-managed-cuda-actually-affects)
  - [Topology rules for this box (bridged NVLink, no switch)](#topology-rules-for-this-box-bridged-nvlink-no-switch)
  - [FlashAttention on H200](#flashattention-on-h200)
  - [Serving cheatsheet](#serving-cheatsheet)


# Running JOBS

**Rule #1: Unless otherwise spcified, you always have to use the following commands as soon as you ssh into the machine to run anything (including an interactive session)**. This is because the machine is shared across multiple labs and users, and you need to request resources (GPUs, CPU cores, memory) for your jobs. The Slurm workload manager handles this resource allocation and job scheduling.

**Overview.** Each professors lab gets an exclusive slice of the machine: "GPU die" H200 (141GB VRAM, 96 cores, 256GB). This resource is shared amongst students in the same lab (8 shards currently; or simultaneous use by 8 students). 


Quick reference for running jobs on the shared H200-JLEC node.

Examples use the **`profali`** lab — **swap in your own lab name** everywhere
(`profali`, `profrichard`, `profivan`, `profashriram`). Your *partition* and your
*account* have the same name.

> **Golden rule:** always match `-p <lab>` **and** `-A <lab>`. The partition checks
> your **account**, not your username — a mismatch is denied.

---

## Submitting work — `srun` / `salloc` / `sbatch`

srun and salloc are for interactive sessions, while sbatch is for batch jobs. The --gres flag specifies the type and amount of resources you want to request (e.g., shards or GPUs). The --cpus-per-task flag specifies how many CPU cores you want to allocate for your job. The --pty bash at the end of srun commands opens an interactive bash shell on the allocated resources.

```bash
# You have to mail ashriram@sfu.ca to be added to the partition first. Mail him/or better yet. msg on teams after you can ssh in.
# Interactive shell, a 1/4-die slice (2 of 8 shards) + 8 CPUs — coexists with labmates
srun -p profali -A profali --gres=shard:2 --cpus-per-task=8 --pty bash

# Bigger interactive session: the whole  gpu worth of shards
# Only allocates. Follow up with srun
salloc -p profali -A profali --gres=shard:8 --cpus-per-task=28
srun ./a.out
# Exclusive WHOLE GPU (blocks other shard jobs on that die)
srun -p profali -A profali --gres=gpu:1 --pty bash

# One quick command on a slice
srun -p profali -A profali --gres=shard:2 -c 8 nvidia-smi -L

# Batch job with wall-time + name
sbatch -p profali -A profali --gres=shard:4 --cpus-per-task=16 \
       --time=04:00:00 -J train train.sh
# Example train.sh script
#!/bin/bash
# Example batch script for training a model on the shared H200 node.
#SBATCH --gres=shard:4              # Request 4 shards (half a GPU die)
#SBATCH --cpus-per-task=16          # Request 16 CPU cores
#SBATCH --time=04:00:00             # Set a time limit for the job
#SBATCH -J train                   # Set a job name (train)
#SBATCH -o train.out               # Redirect output to a file
# Load any necessary modules or activate your environment here
# e.g., module load cuda/11.8
# Run your training command
python train.py
#

```

If your lab's default partition/account is configured, you can drop `-p`/`-A`:

```bash
srun --gres=shard:2 -c 8 --pty bash
```

**How much to ask for:**

| Request | What you get | Shareable? |
|---|---|---|
| `--gres=shard:2` | ~1/4 of the die | yes — 6 shards free for labmates |
| `--gres=shard:8` | the whole die's shard budget | no — consumes the lab's cap |
| `--gres=gpu:h200:1` | the whole die, exclusive | no — blocks all shard jobs on it |

---

## Watching the queue — `squeue`

```bash
squeue -u $USER                       # my jobs
squeue --me -t RUNNING                # only my running jobs
squeue -p profali                # everything on my lab's partition

# Why am I pending? (%R = reason)
squeue --me -t PENDING -o '%.8i %.9P %.8j %.8T %R'

# Shard/GPU allocation per job (%b = GRES binding)
squeue -p profali -o '%.8i %.8u %.9P %.8T %.12b %.10M %R'
```

**Common pending reasons:**

- `QOSGrpGpuLimit` — the lab's GPU/shard budget is full; wait for a labmate to finish.
- `Resources` — no free shards on the die right now.
- `Priority` — a higher-priority job is queued ahead of you.

---

## Inspecting a job — `scontrol show job`

```bash
scontrol show job 18                  # full detail for one job
scontrol show job -d 18               # add scheduling / GRES detail
scontrol show job 18 | grep -iE 'JobState|Reason|AllocTRES|NodeList|RunTime'
```

**What to read in the output:**

- `JobState` / `Reason` — RUNNING, or why it's PENDING.
- `AllocTRES` — what you actually got, e.g. `cpu=8,mem=16G,gres/shard=2`.
- `NodeList` — which die your job landed on (`profali` → `/dev/nvidia2`).
- `TimeLimit` / `RunTime` — how long it can run vs. has run.

---

## Managing your own jobs

```bash
scancel 18                            # cancel one job
scancel -u $USER                      # cancel ALL my jobs
scancel -u $USER -t PENDING           # cancel only my queued jobs
scontrol hold 18                      # pause a pending job
scontrol release 18                   # resume it
```

---

## Status & history

```bash
sinfo -p profali                 # is the lab partition up? nodes idle/alloc?
sinfo -N -o '%n %P %t %G'             # per-node state + GRES (shards/gpu)

# live shard usage on the die
scontrol show node profali | grep -iE 'Gres|AllocTRES|State'

# my job history today
sacct -u $USER --starttime today \
  -o JobID,JobName,Partition,AllocTRES%30,State,Elapsed

# accounting for one job (memory high-water mark, etc.)
sacct -j 18 -o JobID,State,Elapsed,MaxRSS,AllocTRES%40
```

---

## Two things that trip people up

1. **Match `-p` and `-A`.** The partition's `AllowAccounts` checks the *account*,
   not your username. Use `-p profali -A profali`. Mismatch → access denied.

2. **`shard` ≠ isolation.** `--gres=shard:N` shares the die; `nvidia-smi` still shows
   the *whole* H200's memory and utilization regardless of how many shards you asked
   for. Shards are scheduler bookkeeping, not a hardware partition — don't be alarmed
   that you "see" all 141 GB. You can still OOM a labmate on the same die, so be
   considerate. For guaranteed exclusive access, use `--gres=gpu:h200:1`.


# Torch setup.

This machine does not come preinstalled with pytorch and other bells and whistles for ML research. You have to setup your own environment. Using scripts below. A reproducible setup for **custom ops (Triton + CUDA C++)**, **LLM fine-tuning /
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

# Layout

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
# 0. Make sure your managed CUDA is on PATH (nvcc --version works) and topology is sane:
module load LIB/CUDA/13.0 # Machine also supports 12.6 and older. If you need specific version in modules ask research-support@sfu.ca

nvidia-smi topo -m            # note which GPU indices read NV# (a bridged pair)

# 1. Dev/train env (auto-matches torch to your nvcc)
# We will be using uv as our manager. It is a fast pip replacement. 
# It creates a virtual environment. This installs the correct TORCH version
bash setup-dev.sh
source ~/envs/llmdev/bin/activate
python verify_env.py          # should end with "All core checks passed."

# 2. Prove the custom-op toolchain end-to-end
cd examples && TORCH_CUDA_ARCH_LIST=9.0 uv pip install -e . && python test_fma.py

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

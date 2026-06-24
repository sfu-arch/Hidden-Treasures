"""Minimal LoRA SFT skeleton, topology-aware for H200 NVL pairs.

Launch on ONE bridged pair (NVLink) — read the NV# pair from `nvidia-smi topo -m`:

    CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
        --num_processes 2 --multi_gpu \
        examples/sft_lora.py

For a model that fits on one GPU, prefer DDP (one replica per GPU) and let each
bridged pair hold a replica. For a model that must shard, FSDP within the
bridged pair keeps the heavy all-gather/reduce-scatter on NVLink. Sharding
ACROSS pairs pushes that traffic onto PCIe — avoid unless the model demands it.
"""
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

MODEL = "mistralai/Mistral-7B-v0.1"   # swap as needed

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",   # FA2; use FA3 build for max Hopper perf
)

peft_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

ds = load_dataset("tatsu-lab/alpaca", split="train[:2%]")

cfg = SFTConfig(
    output_dir="out/sft-lora",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    bf16=True,
    gradient_checkpointing=True,
    num_train_epochs=1,
    logging_steps=10,
    max_length=2048,
    # FSDP within the bridged pair (set via accelerate config or these flags):
    # fsdp="full_shard auto_wrap",
    # fsdp_config={"transformer_layer_cls_to_wrap": "LlamaDecoderLayer"},
)

trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds,
                     processing_class=tok, peft_config=peft_cfg)
trainer.train()
trainer.save_model("out/sft-lora")

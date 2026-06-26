#!/bin/bash
export WANDB_RUN_ID=flickr192_optimized_x0pred_v1_bf16
export WANDB_RESUME=allow
cd /mnt/zone_a/PQDiff/PQDiff
source .venv/bin/activate
# bitsandbytes (8-bit AdamW) needs torch's pip-shipped CUDA runtime libs on LD_LIBRARY_PATH explicitly.
export LD_LIBRARY_PATH="$(pwd)/.venv/lib/python3.13/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH"
# Before launching: check GPU1 has ~3.8GB+ free (nvidia-smi) — it was shared with another job
# (pct_lft_newloss.py, ~3.5GB) as of 2026-06-24. Verified fitting alongside it with use_ema=False +
# adamw8bit + micro_batch_size=2 (~340MB margin — tight but tested through a full eval/sampling cycle).
# See improvement.md §5.6 for the full memory-optimization story.
CUDA_VISIBLE_DEVICES=1 ACCELERATE_MIXED_PRECISION=bf16 python train_ldm.py \
    --config=configs/flickr192_optimized.py
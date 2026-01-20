#!/bin/bash
wandb_name=2gpu_debug

export WANDB_NAME=$wandb_name
export WANDB_PROJECT=nanogpt_ES_claude

uv run torchrun \
    --standalone \
    --nproc_per_node=2 \
    train.py \
    --batch_size 128

echo "run.sh has finished."

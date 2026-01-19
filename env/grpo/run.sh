#!/bin/bash
wandb_name=b200test
timeout=1h

export VLLM_USE_V1=0
timeout $timeout uv run python grpo.py \
    --learning_rate 1e-5 \
    --grpo_steps 20 \
    --group_size 8 \
    --rollout_subset_size 64 \
    --eval_epochs 2 \
    --train_steps_per_rollout 1 \
    --gradient_accumulation_steps 32 \
    --batch_size 2 \
    --cliprange 0.2 \
    --loss_type grpo_clip \
    --wandb_name $wandb_name

echo "Experiment finished successfully!"

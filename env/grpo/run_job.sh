#!/bin/bash
wandb_name=b200test
timeout=1h

export VLLM_USE_V1=0
timeout $timeout uv run   \
    --project . \
    --default-index https://pypi.org/simple \
    --index https://download.pytorch.org/whl/cu128 \
    --index-strategy unsafe-best-match \
    python grpo.py \
        --learning_rate 1e-5 \
        --grpo_steps 20 \
        --group_size 8 \
        --rollout_subset_size 128 \
        --eval_epochs 2 \
        --train_steps_per_rollout 1 \
        --gradient_accumulation_steps 16 \
        --batch_size 4 \
        --cliprange 0.2 \
        --loss_type grpo_clip \
        --wandb_name $wandb_name

echo "Experiment finished successfully!"
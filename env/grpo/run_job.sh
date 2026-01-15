#!/bin/bash
wandb_name=b200test
timeout=1h

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --wandb_name) wandb_name="$2"; shift 2 ;;
        --timeout) timeout="$2"; shift 2 ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
done

# Set memory limit: 225GB (90% of 250GB to leave headroom)
# ulimit -v sets virtual memory limit in KB
ulimit -v $((225 * 1024 * 1024))

aws --endpoint-url https://conductor.data.apple.com s3 sync \
    s3://afm-bastion/zitong_yang2/research-agent/grpo/MATH/ \
    MATH/

aws --endpoint-url https://conductor.data.apple.com s3 sync \
    s3://afm-bastion/zitong_yang2/research-agent/cache/huggingface/hub/models--Qwen--Qwen2.5-Math-1.5B/ \
    ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-Math-1.5B/

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
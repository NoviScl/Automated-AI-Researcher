#!/bin/bash
wandb_name=8xh100dev
timeout=2h
# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --wandb_name) wandb_name="$2"; shift 2 ;;
        --timeout) timeout="$2"; shift 2 ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
done

# Check if fineweb10B directory exists and has data
if [ -d "fineweb10B" ] && [ "$(ls -A fineweb10B)" ]; then
    echo "fineweb10B directory exists and contains data, skipping download"
else
    echo "fineweb10B directory does not exist or is empty, need to download data"
    # Check if aws CLI is available
    if command -v aws &> /dev/null; then
        echo "AWS CLI found, downloading data from S3"
        aws --endpoint-url https://conductor.data.apple.com s3 sync \
            s3://afm-bastion/zitong_yang2/research-agent/nanogpt/fineweb10B/ \
            fineweb10B/
    else
        echo "AWS CLI not found, generating data using fineweb.py"
        uv run python fineweb.py
    fi
fi

export WANDB_NAME=$wandb_name
export WANDB_PROJECT=nanogpt_ES_claude

if ! timeout $timeout uv run torchrun \
    --standalone \
    --nproc_per_node=8 \
        train.py; then
    echo "train.py failed with exit code $?. The script will continue."
fi

echo "run_job.sh has finished."
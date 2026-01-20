# Research Environments for Automated AI Researcher

We share the research environments used in our paper **Towards Execution-Grounded Automated AI Researcher**.

## Environments

### Post-training environment: GRPO on math reasoning
`env/grpo`: This is a stand-alone directory that implements the GRPO algorithm from scratch and finetunes a Qwen2.5-Math-1.5B model on the MATH dataset. To run the baseline GRPO algorithm, use `cd env/grpo && bash run_job.sh`. This training script runs on a single B200 GPU and contains that default hyper-parameters that we give to the LLM agent (including both the ideator model and executor model). 

We perform the model training and rollout sampling on one single GPU in this GRPO implementation. If you have a GPU with less memory than B200, we also provide another training script `env/grpo/run.sh`. We have tested it on one single A100 (80GB). 

Assuming you have logged into your Wandb account, the training script will automatically log the experiment in your Wandb account with specified project name and run name.  

### Pre-training environment: nanoGPT on FineWeb
`env/nanogpt`: This is a stand-alone directory that implements the nanoGPT baseline to pretrain GPT-2 model on the FineWeb dataset. To run the nanoGPT environment, first download the fineweb data using `cd env/nanogpt && uv run python fineweb.py`, and then run the training command `cd env/nanogpt && bash run_job.sh`. Note that `run_job.sh` is the default script we used during our experiments, where each run has to use 8 GPUs. 

If you want to do a quick debugging, we provide `env/nanogpt/run.sh` that is tested on 2 H200s. You can also adjust `--nproc_per_node=2` and `--batch_size 128` based on how many GPUs you are actually using. Again, all experiment logs will be automatically logged on your Wandb account. 

Before running, make sure to change `input_bin` and `input_val_bin` in `env/nanogpt/train.py` to the actual directories that you store the downloaded FineWeb data. 

## Evolutionary Search Scaffold

`agent/` implements our execution-guided evolutionary search scaffold. Running `agent/full_pipeline.py` will run the full pipeline that: generates ideas, generates the code diffs to implement the ideas, patches the code diffs into the environments and uploads the codebases. Our automated executor will then allocates the codebases to available GPUs and executes the training jobs.

## Idea Trajectories 

We share the full idea trajectories from our evolutionary search experiments. 

| Run                       | Trajectory Link                                                                                     |
|---------------------------|-----------------------------------------------------------------------------------------------------|
| Claude-4.5-Opus on NanoGPT | [HF Link](https://huggingface.co/datasets/codasci/search_es_pre_claude_4_5_opus)                   |
| Claude-4.5-Sonnet on NanoGPT | [HF Link](https://huggingface.co/datasets/codasci/search_es_pre_claude_4_5_sonnet)                   |
| GPT-5 on NanoGPT | [HF Link](https://huggingface.co/datasets/codasci/search_es_pre_gpt5)                   |
| Claude-4.5-Opus on GRPO | [HF Link](https://huggingface.co/datasets/codasci/search_es_post_claude_4_5_opus)                   |
| Claude-4.5-Sonnet on GRPO | [HF Link](https://huggingface.co/datasets/codasci/search_es_post_claude_4_5_sonnet)                   |
| GPT-5 on GRPO | [HF Link](https://huggingface.co/datasets/codasci/search_es_post_gpt5)                   |

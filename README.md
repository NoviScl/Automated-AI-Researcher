# Research Environments for Automated AI Researcher

We share the research environments used in our paper **Towards Execution-Grounded Automated AI Researcher**.

### Environments

```env/grpo```: This is a stand-alone directory that implements the GRPO algorithm from scratch and finetunes a Qwen2.5-Math-1.5B model on the MATH dataset.

```env/nanogpt```: This is a stand-alone directory that implements the NanoGPT baseline to pretrain GPT-2 model on the FineWeb dataset.

### Evolutionary Search Scaffold

```agent/``` implements our execution-guided evolutionary search scaffold. Running ```agent/full_pipeline.py``` will run the full pipeline that: generates ideas, generates the code diffs to implement the ideas, patches the code diffs into the environments and uploads the codebases. Our automated executor will then allocates the codebases to available GPUs and executes the training jobs.

### Idea Trajectories 

We share the full idea trajectories from our evolutionary search experiments. 

| Run                       | Trajectory Link                                                                                     |
|---------------------------|-----------------------------------------------------------------------------------------------------|
| Claude-4.5-Opus on NanoGPT | [HF Link](https://huggingface.co/datasets/codasci/search_es_pre_claude_4_5_opus)                   |
| Claude-4.5-Sonnet on NanoGPT | [HF Link](https://huggingface.co/datasets/codasci/search_es_pre_claude_4_5_sonnet)                   |
| GPT-5 on NanoGPT | [HF Link](https://huggingface.co/datasets/codasci/search_es_pre_gpt5)                   |
| Claude-4.5-Opus on GRPO | [HF Link](https://huggingface.co/datasets/codasci/search_es_post_claude_4_5_opus)                   |
| Claude-4.5-Sonnet on GRPO | [HF Link](https://huggingface.co/datasets/codasci/search_es_post_claude_4_5_sonnet)                   |
| GPT-5 on GRPO | [HF Link](https://huggingface.co/datasets/codasci/search_es_post_gpt5)                   |
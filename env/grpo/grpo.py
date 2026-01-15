from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import tokenize_prompt_and_output, get_response_log_probs
from sample import load_prompt_template, load_dataset_with_prompt_template, sample_rollout
from drgrpo_grader import r1_zero_reward_fn_train
from evaluate import r1_zero_reward_fn_eval, evaluate_vllm
from grpo_utils import compute_group_normalized_rewards, grpo_microbatch_train_step
from torch.utils.data import DataLoader, Dataset
import torch
import json
import argparse
from tqdm import tqdm
from vllm import LLM, SamplingParams
import wandb
import random

def load_policy_into_vllm_instance(policy, llm):
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())

class MathDataset(Dataset):
    def __init__(self, input_ids, labels, response_mask):
        self.input_ids = input_ids
        self.labels = labels
        self.response_mask = response_mask

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.labels[idx], self.response_mask[idx]

def load_model_and_tokenizer(model_path = "Qwen/Qwen2.5-Math-1.5B", tokenizer_path = "Qwen/Qwen2.5-Math-1.5B"):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return model, tokenizer

def load_dataset(dataset_path = "MATH/train.jsonl"):
    with open(dataset_path, 'r') as f:
        dataset = [json.loads(line) for line in f]
    return dataset

def tokenize_dataset(dataset, tokenizer):
    prompts = [example["prompt"] for example in dataset]
    outputs = [example["response"] for example in dataset]
    tokenized_dataset = tokenize_prompt_and_output(prompts, outputs, tokenizer)
    return tokenized_dataset

def create_data_loader(dataset, batch_size = 8, shuffle = True):
    dataset = MathDataset(dataset["input_ids"], dataset["labels"], dataset["response_mask"])
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True)
    return data_loader

def evaluate_model(policy_model, vllm_model, eval_prompts, eval_answers, eval_sampling_params, output_path = None):
    load_policy_into_vllm_instance(policy_model, vllm_model)
    metrics = evaluate_vllm(vllm_model, r1_zero_reward_fn_eval, eval_prompts, eval_answers, eval_sampling_params, output_path=output_path)
    return metrics

def train_loop(model, train_prompts, train_answers, learning_rate, grpo_steps, train_steps_per_rollout, output_dir, batch_size, gradient_accumulation_steps = 4, group_size = 2, rollout_subset_size = 256, device = "cuda", logging_steps = 20, saving_steps = 4000, eval_epochs = 5, eval_prompts = None, eval_answers = None, sampling_params = None, eval_vllm_model = None, cliprange = 0.2, loss_type = "reinforce_with_baseline"):
    model.to(device)
    training_steps = grpo_steps
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0, betas=(0.9, 0.95))
    global_step = 0  # Initialize global step counter

    for epoch in range(grpo_steps):
        model.train()
        eval_mean_reward = None

        # Evaluate on validation set every 5 epochs
        if epoch % eval_epochs == 0 and eval_prompts is not None and eval_answers is not None:
            model.eval()
            print("\nEvaluating on validation set at epoch: ", epoch)
            eval_results = evaluate_model(model, eval_vllm_model, eval_prompts, eval_answers, eval_sampling_params)
            eval_mean_reward = sum(result["reward"]["reward"] for result in eval_results) / len(eval_results)

            model.train()

        ## load the current policy model to vllm for sampling rollouts
        load_policy_into_vllm_instance(model, vllm_model)

        ## sample rollouts
        print ("Sampling rollouts for epoch: ", epoch)
        rollout_prompts, rollout_answers, rollout_responses, rollout_rewards = sample_rollout(vllm_model, r1_zero_reward_fn_train, train_prompts, train_answers, G=group_size, eval_sampling_params=eval_sampling_params, subset_size=rollout_subset_size, return_rewards=True, batch_size=512)
        # Randomly sample 2 rollouts to print
        indices = random.sample(range(len(rollout_prompts)), 2)
        print ("Example rollouts:")
        for idx in indices:
            print(f"\nRollout {idx}:")
            print(f"Prompt: {rollout_prompts[idx]}")
            print(f"Response: {rollout_responses[idx]}")
            print(f"Reward: {rollout_rewards[idx]}")
            print(f"Ground truth: {rollout_answers[idx]}")
        rollout_tokenized = tokenize_prompt_and_output(rollout_prompts, rollout_responses, tokenizer)
        rollout_data_loader = create_data_loader(rollout_tokenized, batch_size=batch_size, shuffle=False)

        # Get old policy log probs batch by batch to avoid OOM
        # print ("Getting old policy log probs")
        old_log_probs_list = []
        with torch.no_grad():
            for batch in rollout_data_loader:
                input_ids, labels, response_mask = [t.to(device) for t in batch]
                old_response_log_probs = get_response_log_probs(
                    model,
                    input_ids,
                    labels,
                    return_token_entropy=False,
                    no_grad=True
                )
                old_log_probs_list.append(old_response_log_probs["log_probs"])

                # Clean up memory after each batch
                del old_response_log_probs
                torch.cuda.empty_cache()

            # Concatenate all batches
            old_log_probs = torch.cat(old_log_probs_list, dim=0)
            del old_log_probs_list
            torch.cuda.empty_cache()

        # Compute advantages using group normalization - no gradients needed
        with torch.no_grad():
            advantages, raw_rewards, metadata = compute_group_normalized_rewards(
                reward_fn=r1_zero_reward_fn_train,
                rollout_responses=rollout_responses,
                repeated_ground_truths=rollout_answers,
                group_size=group_size,
                advantage_eps=1e-6,
                normalize_by_std=True
            )
            advantages = advantages.to(device)

        # Log raw rewards statistics
        print("\nGRPO epoch: ", epoch)
        print(f"Mean reward: {metadata['mean_reward']:.4f}")

        # Log reward metrics to wandb
        if eval_mean_reward is not None:
            wandb.log({
                "eval/mean_reward": eval_mean_reward,
                "train/mean_reward": metadata["mean_reward"],
            }, step=global_step)
        else:
            wandb.log({
                "train/mean_reward": metadata["mean_reward"],
            }, step=global_step)


        ## train on this rollout batch for train_steps_per_rollout steps
        for train_step in range(train_steps_per_rollout):
            # Process each batch
            for batch_idx, batch in tqdm(enumerate(rollout_data_loader)):
                global_step += 1
                input_ids, labels, response_mask = [t.to(device) for t in batch]

                # Get current policy log probs (with gradients)
                response_log_probs = get_response_log_probs(
                    model,
                    input_ids,
                    labels,
                    return_token_entropy=True,
                    no_grad=False
                )
                policy_log_probs = response_log_probs["log_probs"]
                entropy = response_log_probs["token_entropy"]

                # Calculate data index for advantages/old_log_probs
                batch_idx_total = batch_idx * batch_size
                batch_advantages = advantages[batch_idx_total : batch_idx_total + batch_size].unsqueeze(-1)  # Add dimension to get (batch_size, 1)
                batch_old_log_probs = old_log_probs[batch_idx_total : batch_idx_total + batch_size]

                loss, metadata = grpo_microbatch_train_step(
                    policy_log_probs=policy_log_probs,
                    response_mask=response_mask,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    loss_type=loss_type,
                    advantages=batch_advantages,
                    old_log_probs=batch_old_log_probs,
                    cliprange=cliprange
                )

                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                    # Log training metrics to wandb
                    wandb.log({
                        "train/loss": loss.item(),
                    }, step=global_step)
                    # print ("Global Step: ", global_step, "Loss: ", loss.item(), "Entropy: ", entropy.mean().item(), "Clip fraction: ", metadata.get("clip_fraction", 0.0))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--tokenizer_path", type=str, default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--train_dataset_path", type=str, default="MATH/train.jsonl")
    parser.add_argument("--eval_dataset_path", type=str, default="MATH/test.jsonl")
    parser.add_argument("--output_dir", type=str, default="ckpts/")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--grpo_steps", type=int, default=200)
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--rollout_subset_size", type=int, default=256)
    parser.add_argument("--eval_epochs", type=int, default=2)
    parser.add_argument("--train_steps_per_rollout", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--cliprange", type=float, default=0.2)
    parser.add_argument("--loss_type", type=str, default="grpo_clip")
    parser.add_argument("--wandb_project", type=str, default="grpo-math-no-example-prompt")
    parser.add_argument("--wandb_name", type=str, default="grpo_clip_1")
    args = parser.parse_args()

    print("Full list of args:", vars(args))

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script")

    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        config={
            "learning_rate": args.learning_rate,
            "grpo_steps": args.grpo_steps,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "cliprange": args.cliprange,
            "model_path": args.model_path,
            "train_dataset_path": args.train_dataset_path,
            "eval_dataset_path": args.eval_dataset_path,
        }
    )

    prompt_template = load_prompt_template()
    vllm_model = LLM(model=args.model_path, tokenizer=args.tokenizer_path, gpu_memory_utilization=0.55)
    eval_prompts, eval_answers = load_dataset_with_prompt_template(prompt_template, dataset_path=args.eval_dataset_path)
    eval_sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"], include_stop_str_in_output=True
    )

    # Initialize training model on first GPU
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.tokenizer_path)
    train_prompts, train_answers = load_dataset_with_prompt_template(prompt_template, dataset_path=args.train_dataset_path)
    train_loop(
        model,
        train_prompts,
        train_answers,
        args.learning_rate,
        args.grpo_steps,
        args.train_steps_per_rollout,
        args.output_dir,
        args.batch_size,
        args.gradient_accumulation_steps,
        args.group_size,
        args.rollout_subset_size,
        eval_epochs=args.eval_epochs,
        eval_prompts=eval_prompts,
        eval_answers=eval_answers,
        sampling_params=eval_sampling_params,
        eval_vllm_model=vllm_model,
        cliprange=args.cliprange,
        loss_type=args.loss_type
    )


    # Cleanup distributed resources
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    # Clean up CUDA memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Close wandb run
    wandb.finish()


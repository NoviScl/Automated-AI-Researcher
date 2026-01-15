from vllm import LLM, SamplingParams
import json
from tqdm import tqdm
import random

def load_prompt_template(prompt_path = "prompts/r1_zero.prompt"):
    with open(prompt_path, 'r') as f:
        prompt_template = f.read()
    return prompt_template

def get_model_and_sampling_params(model_path = "Qwen/Qwen2.5-Math-1.5B", tokenizer_path = "Qwen/Qwen2.5-Math-1.5B"):
    # Create an LLM.
    llm = LLM(model=model_path, tokenizer=tokenizer_path)

    # Create a sampling params object, stopping generation on newline.
    sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"]
    )
    sampling_params.include_stop_str_in_output = True

    return llm, sampling_params

def load_dataset_with_prompt_template(prompt_template, dataset_path="MATH/test.jsonl"):
    with open(dataset_path, 'r') as f:
        dataset = [json.loads(line) for line in f]
    print(f"Loaded {len(dataset)} examples from {dataset_path}")

    prompts = [prompt_template.format(question=example["problem"]) for example in dataset]
    answers = [example["answer"] for example in dataset]
    return prompts, answers


def sample_rollout(
    vllm_model,
    reward_fn,
    prompts,
    answers,
    G,
    eval_sampling_params,
    output_path = None,
    subset_size = None,
    return_rewards = False,
    batch_size = 64
):
    ## sample G answers per prompt, and keep those that are correct
    results = []
    if subset_size is not None:
        # Randomly sample subset_size examples instead of taking first subset_size
        indices = random.sample(range(len(prompts)), subset_size)
        prompts = [prompts[i] for i in indices]
        answers = [answers[i] for i in indices]

    # Create batched prompts by repeating each prompt G times
    batched_prompts = []
    batched_answers = []
    batched_responses = []
    batched_rewards = []
    for prompt, answer in zip(prompts, answers):
        batched_prompts.extend([prompt] * G)
        batched_answers.extend([answer] * G)

    # Process in batches to avoid OOM
    all_outputs = []
    for i in range(0, len(batched_prompts), batch_size):
        batch_prompts = batched_prompts[i:i + batch_size]
        batch_outputs = vllm_model.generate(batch_prompts, eval_sampling_params)
        all_outputs.extend(batch_outputs)

    # Process results
    total_rewards = 0
    for output, answer in tqdm(zip(all_outputs, batched_answers)):
        generated_answer = output.outputs[0].text
        reward = reward_fn(generated_answer, answer)
        if return_rewards:
            batched_responses.append(generated_answer)
            batched_rewards.append(reward["reward"])
            total_rewards += reward["reward"]
        elif reward["reward"] == 1:
            total_rewards += 1
            dp = {}
            dp["prompt"] = output.prompt
            dp["response"] = generated_answer
            results.append(dp)

    print (f"Accuracy of sampled rollouts: {total_rewards}/{len(batched_prompts)} = {total_rewards / len(batched_prompts) * 100}%")

    if output_path is not None:
        with open(output_path, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')

    if return_rewards:
        return batched_prompts, batched_answers, batched_responses, batched_rewards
    return results


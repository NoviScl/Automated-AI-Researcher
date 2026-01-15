import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.api import apiqa
import os
from retry import retry
import json
import shutil
import subprocess
from pathlib import Path
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import re
import random
from agent.evolutionary_search import *
random.seed(42)


# @retry(tries=3, delay=2)
def agent_call_idea_simple(num_ideas = 10, cache_file = "agent/all_ideas.json", env_dir = "env_grpo", model_name = "gpt-5"):
    context = context_prompt(base_dir=env_dir)
    system_message = f"""
    You are a research scientist that helps me decide what are some promising experiments to run to improve the performance of the model.
    """

    if "grpo" in env_dir:
        ### prompt v1
        # prompt = f"""
        # Below is the list of all code files in this codebase including the example script to launch a job:
        # {context}\n\n
        # I want you to think of {num_ideas} experiments to run that involve modifying the codebase and running the corresponding experiments.
        # I'm looking for experiments that involve some targeted changes to the codebase, for example, changing only a few specific functions to implement a new idea or hypothesis.
        # You can include ideas that might improve the performance of the model, for example, changing the loss function from token-level to sequence-level or re-designing the training reward function. Or you can also include some exploratory ideas that help you better understand the model and codebase's behavior, such as changing the loss type from grpo_clip to reinforce_with_baseline.
        # These are examples of good experiments because they only involve changing a few lines of code in the reward function but still can offer some interesting insights and possible empirical improvements.
        # Now generate the list of {num_ideas} simple and diverse experiments. For each experiment, include two sub-sections: (1) one-sentence description of the experiment; (2) brief summary of the code changes needed. Format each block with the following tags:
        # [Experiment] ...
        # [Code Changes] ...
        # Add a tag [End] after each experiment.
        # Note that each experiment should only involve a single experiment (rather than doing multiple runs and making comparisons). If it involves changing any hyperparameters, you should specify the new values of the hyperparameters.
        # Also note that you are not allowed to change any part of the evaluation logic, including the evaluation data and the evaluation metrics; and you are not allowed to change the eval frequency or the hard time limit set in the run_job.sh script.
        # Now return the list of experiments with no other text.
        # """

        ### prompt v2
        # prompt = f"""
        # Below is the list of all code files in this codebase including the example script to launch a job:
        # {context}\n\n
        # I want you to think of {num_ideas} experiments to run that involve modifying the codebase and running the corresponding experiments.
        # I'm looking for experiments that involve some targeted changes to the codebase, for example, changing a few specific functions to implement a new idea or hypothesis.
        # You should focus on novel training algorithms that might improve the performance of the model. This can include anything such as modifying the loss function, designing new training rewards, implementing new training curriculum, or anything else that is well-motivated and makes sense.
        # For example, some known ideas include changing the GRPO loss function from token-level to sequence-level, setting a different clip range for the upper bound, or removing the normalization in the advantage computation. You should come up with ideas in this flavor but different from these examples. You can also be more bold and replace GRPO altogether.
        # Note that you should come up with generalizable and scalable ideas that can transfer to other models and datasets, which means you probably shouldn't generate ideas that are too specific to the current setup, such as massively tuning the hyperparameters.
        # Now generate the list of {num_ideas} diverse experiments. For each experiment, include two sub-sections: (1) a short description of the idea; (2) a brief summary of the code changes needed. Format each block with the following tags:
        # [Experiment] ...
        # [Code Changes] ...
        # Add a tag [End] after each experiment.
        # Note that each experiment should only involve a single experiment (rather than doing multiple runs and making comparisons). If it involves setting any hyperparameters, you should specify the new values of the hyperparameters.
        # Also note that you are not allowed to change any part of the evaluation logic, including the evaluation data and the evaluation metrics; and you are not allowed to change the eval frequency or the hard time limit set in the run_job.sh script.
        # Now return the list of experiments with no other text.
        # """

        ### prompt v3
        prompt = f"""
        Below is the list of all code files in this codebase including the example script to launch a job:
        {context}\n\n
        I want you to think of {num_ideas} experiments to run that involve modifying the codebase and running the corresponding experiments.
        I'm looking for experiments that involve some targeted changes to the codebase to implement a new research idea.
        You should focus on novel and simple training algorithms that can improve the performance of the model. This can include interventions like modifying the loss function, designing additional training rewards, implementing a new training curriculum, or anything else that is well-motivated and makes sense.
        For example, some simple ideas include changing the GRPO loss function from token-level to sequence-level, setting a different clip range for the upper bound, removing the normalization in the advantage computation, or designing a problem currimulum based on the advantage distribution. You should come up with ideas in this flavor but different from these examples.
        We prefer simple ideas that can be implemented by changing just a few functions in the codebase, because usually simple ideas are likely to generalize.
        Note that you should come up with generalizable and scalable ideas that can transfer to other models and datasets, and you shouldn't generate ideas that are too specific to the current setup, such as massively tuning the hyperparameters.
        Now generate the list of {num_ideas} diverse and simple experiments. For each experiment, include two sub-sections: (1) a short description of the idea; (2) a brief summary of the code changes needed. Format each block with the following tags:
        [Experiment] ...
        [Code Changes] ...
        Add a tag [End] after each experiment.
        Note that each experiment should only involve a single experiment (rather than doing multiple runs and making comparisons). If it involves setting any hyperparameters, you should specify the new values of the hyperparameters.
        Also note that you are not allowed to change any part of the evaluation logic, including the evaluation data and the evaluation metrics; and you are not allowed to change the eval frequency or the hard time limit set in the run_job.sh script.
        Now return the list of experiments with no other text.
        """
    elif "nanogpt" in env_dir:
        prompt = f"""
        Below is the list of all code files in this codebase including the example script to launch a job:
        {context}\n\n
        I want you to think of {num_ideas} novel and simple ideas that involve modifying the codebase and running the corresponding experiments.
        I'm looking for experiments that involve some targeted changes to the codebase, for example, changing a few specific functions to implement a new idea or hypothesis.
        You should include ideas that can improve the performance of the model or make the model train faster, for example, with a better architecture, learning rate schedule, optimizer, etc.
        Now generate the list of {num_ideas} diverse experiments. For each experiment, include two sub-sections: (1) a short description of the idea; (2) a brief summary of the code changes needed. Format each block with the following tags:
        [Experiment] ...
        [Code Changes] ...
        Add a tag [End] after each experiment.
        Note that each experiment should only involve a single experiment (rather than doing multiple runs and making comparisons). If it involves setting any hyperparameters, you should specify the new values of the hyperparameters.
        Also note that you are not allowed to change any part of the evaluation logic, including the evaluation data and the evaluation metrics; and you are not allowed to change the eval frequency or the hard time limit set in the script.
        We have implemented functions like `forward_with_cache` and `forward_safe` in the codebase, these are meant to preserve the autoregressive nature of the model and avoid leaking information from future tokens during the validation step. You should not break the autoregressive nature of these functions.
        External imports are not supported so you should suggest experiments that can be implemented by directly changing a few functions in the codebase.
        You should absolutely not change the loss function part: loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1); or the validation hyperparameters including val_loss_every and val_tokens (we only do the valiation at the end of the training). You are not allowed to change the time limit in "if elapsed_time_seconds > 1500:" either. These must be left unchanged for fair comparison!
        Now return the list of experiments with no other text.
        """
    # elif "nanogpt" in env_dir:
    #     prompt = f"""
    #     Below is the list of all code files in this codebase including the example script to launch a job:
    #     {context}\n\n
    #     I want you to think of {num_ideas} experiments to run that involve modifying the codebase and running the corresponding experiments.
    #     I'm looking for experiments that involve some targeted changes to the codebase, for example, changing a few specific functions to implement a new idea or hypothesis.
    #     You should include ideas that might improve the performance of the model or make the model train faster, for example, with a better architecture, learning rate schedule, or optimizer.
    #     Now generate the list of {num_ideas} diverse experiments. For each experiment, include two sub-sections: (1) a short description of the idea; (2) a brief summary of the code changes needed. Format each block with the following tags:
    #     [Experiment] ...
    #     [Code Changes] ...
    #     Add a tag [End] after each experiment.
    #     Note that each experiment should only involve a single experiment (rather than doing multiple runs and making comparisons). If it involves setting any hyperparameters, you should specify the new values of the hyperparameters.
    #     Also note that you are not allowed to change any part of the evaluation logic, including the evaluation data and the evaluation metrics; and you are not allowed to change the eval frequency or the hard time limit set in the run_job.sh script.
    #     You should absolutely not change the loss function part: loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1); or the validation hyperparameters including val_loss_every and val_tokens. These must be left unchanged for fair comparison!
    #     Now return the list of experiments with no other text.
    #     """

    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            cache_lst = json.load(f)
        cache_str = "\n\n".join(cache_lst)
        prompt += f"""
        Avoid any similar ideas to the following experiments that have been proposed before:
        {cache_str}
        """
    response = apiqa(prompt, model_name, system_message, claude_thinking_mode=True, claude_thinking_budget=3000, temperature=1.0)
    if len(response) == 2:
        thinking, response = response
    else:
        thinking = None
    # print (prompt)
    # print (thinking)
    print (response)
    response = strip_response(response)
    response_lst = response.split("[End]")
    response_lst = [
        r[r.find("[Experiment]"):].strip()
        for r in response_lst
        if r.strip() and "[Experiment]" in r
    ]
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            cache_lst = json.load(f)
        cache_lst.extend(response_lst)
    else:
        cache_lst = response_lst
    with open(cache_file, "w") as f:
        json.dump(cache_lst, f, indent=4)
    return response, thinking


@retry(tries=3, delay=2)
def agent_call_idea_update(num_ideas = 10, prev_ideas_file = "ideas/ideas_for_testing.json", prev_training_logs = "training_logs/epoch0/", top_k=10, cache_file = "ideas/ideas_for_testing_update.json", env_dir = "env_grpo", model_name = "gpt-5"):
    '''
    Read the previous batch of ideas, take the top-performing ideas, shove them into the prompt, and generate a new batch of ideas.
    '''
    context = context_prompt(base_dir=env_dir)
    system_message = f"""
    You are a research scientist that helps me decide what are some promising experiments to run to improve the performance of the model.
    """

    ## read the previous ideas
    with open(prev_ideas_file, "r") as f:
        prev_ideas_lst = json.load(f)
    prev_ideas_dict = {}
    for idx, idea in enumerate(prev_ideas_lst):
        prev_ideas_dict[f"idea_{idx}"] = idea

    ## read the previous training logs and take the top-performing ideas, with randomization
    with open(os.path.join(prev_training_logs, "ranked_ideas.json"), "r") as f:
        prev_ranked_list = json.load(f)
    prev_ideas_prompt = ""
    # Sample top_k ideas from the top (top_k * 2) ideas
    top_candidates = prev_ranked_list[: top_k * 2]
    if len(top_candidates) <= top_k:
        sampled_ideas = top_candidates
    else:
        sampled_ideas = random.sample(top_candidates, top_k)
    for idea_dict in sampled_ideas:
        idea_key, reward_value = next(iter(idea_dict.items()))
        prev_ideas_prompt += "Idea: " + prev_ideas_dict[idea_key] + "\n"
        prev_ideas_prompt += "Final Eval Accuracy: " + str(reward_value) + "\n\n"

    prompt = f"""
    Below is the list of all code files in this codebase including the example script to launch a job:
    {context}\n\n
    I want you to think of {num_ideas} experiments to run that involve modifying the codebase and running the corresponding experiments.
    I'm looking for experiments that involve some targeted changes to the codebase, for example, changing only a few specific functions to implement a new idea or hypothesis.
    Below is the list of previous experiments that have been proposed and their final accuracy:
    {prev_ideas_prompt}
    As a reference, the baseline accuracy of running the original codebase without any modifications is about 0.48.
    Now synthesize insights from the previous experiments and generate {num_ideas} new experiments that are different from the previous experiments and will likely work better.
    For each experiment, include two sub-sections: (1) concise description of the experiment; (2) brief summary of the code changes needed. Format each block with the following tags:
    [Experiment] ...
    [Code Changes] ...
    Add a tag [End] after each experiment.
    Note that each experiment should only involve a single experiment (rather than doing multiple runs and making comparisons). If it involves changing any hyperparameters, you should specify the new values of the hyperparameters.
    Also note that you are not allowed to change any part of the evaluation logic, including the evaluation data and the evaluation metrics.
    Now return the list of experiments with no other text.
    """
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            cache_lst = json.load(f)
        cache_str = "\n\n".join(cache_lst)
        prompt += f"""
        Avoid any similar ideas to the following experiments that have been proposed before:
        {cache_str}
        """
    (thinking, response) = apiqa(prompt, model_name, system_message, claude_thinking_mode=True, claude_thinking_budget=3000, temperature=1.0)
    # print (thinking)
    # print (response)
    response = strip_response(response)
    response_lst = response.split("[End]")
    response_lst = [
        r[r.find("[Experiment]"):].strip()
        for r in response_lst
        if r.strip() and "[Experiment]" in r
    ]
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            cache_lst = json.load(f)
        cache_lst.extend(response_lst)
    else:
        cache_lst = response_lst
    with open(cache_file, "w") as f:
        json.dump(cache_lst, f, indent=4)
    return response, thinking


def code_diff_fixer(diff_file="diffs/code_diff_idea_0.diff"):
    """
    Reads a unified diff file, checks each hunk header for correct line counts,
    and rewrites the diff file with corrected hunk headers if needed.
    """

    with open(diff_file, "r") as f:
        lines = f.readlines()

    output_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Look for hunk header
        if line.startswith('@@'):
            # Parse the hunk header
            m = re.match(r"@@ -(\d+),?(\d*) \+(\d+),?(\d*) @@", line)
            if not m:
                output_lines.append(line)
                i += 1
                continue
            old_start = int(m.group(1))
            old_count = int(m.group(2)) if m.group(2) else 1
            new_start = int(m.group(3))
            new_count = int(m.group(4)) if m.group(4) else 1

            # Collect hunk lines
            hunk_lines = []
            i += 1
            while i < len(lines) and not lines[i].startswith('@@') and not lines[i].startswith('--- ') and not lines[i].startswith('+++ '):
                hunk_lines.append(lines[i])
                i += 1

            # Count lines for old and new files
            actual_old = 0
            actual_new = 0
            for hline in hunk_lines:
                if hline.startswith('-'):
                    actual_old += 1
                elif hline.startswith('+'):
                    actual_new += 1
                else:
                    actual_old += 1
                    actual_new += 1

            # If the counts are wrong, fix the header
            if old_count != actual_old or new_count != actual_new:
                new_header = f"@@ -{old_start},{actual_old} +{new_start},{actual_new} @@\n"
                output_lines.append(new_header)
            else:
                output_lines.append(line)
            output_lines.extend(hunk_lines)
        else:
            output_lines.append(line)
            i += 1

    # Write back the fixed diff
    with open(diff_file, "w") as f:
        f.writelines(output_lines)

def generate_code_diff(idea_idx = 6, base_dir = "env_grpo", variant_dir = "repo_variants_testing", idea_file = "agent/all_ideas.json", diff_dir = "diffs_testing", prev_diff_file = None, prev_diff_error = None, idea_lst = None, model_name = "gpt-5"):
    if idea_lst is not None:
        idea_str = idea_lst[idea_idx]
    else:
        with open(idea_file, "r") as f:
            cache_lst = json.load(f)
        idea_str = cache_lst[idea_idx]
    context = context_prompt(base_dir=base_dir)

    system_message = f"""
    You are a research scientist that can help me implement the given research idea in the codebase.
    """
    prompt = f"""
    Below is the list of all code files in this codebase including the example script to launch a job:
    {context}\n\n
    Below is the experiment that you need to implement:
    {idea_str}\n\n
    I want you to understand the problem and generate the code changes needed to implement the experiment.
    In your response, generate all the code changes in the standard unified diff format as used in git without any other text. This includes any change to the training script too if needed.
    Try not to import external libraries because they may not be installed in the execution environment; instead, implement the code changes in the codebase itself.
    Make sure the diff file is valid and the counts in the headers must match the actual number of lines the hunk contains after accounting for context, so that it can be applied to the original codebase. You cannot have is a hunk (@@ ... @@) floating without its preceding file metadata. Try to include a few context lines before and after each hunk to make the diff file more robust.
    """

    if "grpo" in base_dir.lower():
        prompt += f"""
        \nNote that you are absolutely not allowed to change the evaluation logic, including the eval data and eval metrics (the reward function used to evaluate the model). If you need to change the reward function, you should implement a new reward function only for training and use the original reward function for evaluation.
        Do not change the wandb name or project name.
        """
    elif "nanogpt" in base_dir.lower():
        prompt += f"""
        \nYou are not allowed to change the loss function part: loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1); or the validation hyperparameters including val_loss_every and val_tokens. You are not allowed to change the time limit of 1500 seconds in "if elapsed_time_seconds > 1500:" either. These must be left unchanged for fair comparison!
        We have implemented functions like `forward_with_cache` and `forward_safe` in the codebase, these are meant to preserve the autoregressive nature of the model and avoid leaking information from future tokens during the validation step. You should not break the autoregressive nature of these functions.
        Avoid any possible case of leaking information from future tokens that breaks the autoregressive nature of the model. For example, you should not mix in any future tokens into the current token's representation or normalize across the entire sequence.
        Do not change the wandb name or project name.
        """
    # elif "nanogpt" in base_dir.lower():
    #     prompt += f"""
    #     \nYou are not allowed to change the loss function part: loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1); or the validation hyperparameters including val_loss_every and val_tokens. These must be left unchanged for fair comparison!
    #     Do not change the wandb name or project name.
    #     """

    if prev_diff_file is not None:
        prev_diff_str = diff_prompt(prev_diff_file)
        prompt += f"""
        You have generated the following diff file before:
        {prev_diff_str}
        """
    if prev_diff_error is not None:
        prompt += f"""
        But it generated the following error message:
        {prev_diff_error}
        In the output, fix the errors and generate a new diff file. One common source of error is that you might have messed up the number of lines in the hunk, so make sure to count carefully how many lines are covered by each hunk.
        """


    prompt += "\nNow return a single diff file containing all the code changes with no other text so that I can run patch to apply the diff to the codebase."

    if "gpt" in model_name.lower():
        prompt += " The hearder should always start with something like --- xxx.py\n+++ xxx.py where xxx is the actual file name that the hunk is modifying. Make sure the first line of the hunk body corresponds to the line number specified in the hunk header. You can include a few more unchanged context lines to make the diff file more robust."

    response = apiqa(prompt, model_name, system_message, claude_thinking_mode=True, claude_thinking_budget=3000, temperature=1.0, max_tokens=10000)
    if len(response) == 2:
        thinking, response = response
    else:
        thinking = ""

    response = strip_response(response, token="```diff")

    ## edit the headers
    variant_dir = f"{variant_dir}/idea_{idea_idx}"
    response_lines = response.splitlines()
    for i, line in enumerate(response_lines):
        if line.startswith("--- "):
            filename = line.split('/')[-1]
            response_lines[i] = f"--- {variant_dir}/{filename}"
        elif line.startswith("+++ "):
            filename = line.split('/')[-1]
            response_lines[i] = f"+++ {variant_dir}/{filename}"
    response = "\n".join(response_lines)
    return thinking, response


def apply_code_diff(main_repo_dir = "env_grpo", new_repo_dir = "repo_variants/idea_0", diff_file = "diffs/code_diff_idea_0.diff"):
    main_repo_dir = Path(main_repo_dir).resolve()
    new_repo_dir = Path(new_repo_dir).resolve()

    # Remove target directory if it already exists
    if new_repo_dir.exists():
        shutil.rmtree(new_repo_dir)

    # Copy the repo (including working tree)
    shutil.copytree(main_repo_dir, new_repo_dir, symlinks=True)

    def run(cmd, cwd=new_repo_dir, check=True):
        result = subprocess.run(
            cmd,
            cwd=cwd,
            text=True,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if check and result.returncode != 0:
            raise subprocess.CalledProcessError(
                result.returncode, cmd, output=result.stdout, stderr=result.stderr
            )
        return result

    try:
        # Use patch to apply the diff in the new target repo
        run(f"patch -p0 < {str(diff_file)}", cwd=".")
    except subprocess.CalledProcessError as e:
        # If patch fails, raise an error with details
        raise RuntimeError(
            f"Failed to apply diff with patch:\n{e.stderr or e.output}"
        )
    return new_repo_dir


def apply_diff_dir(main_repo_dir="cs336_alignment", new_repo_dir="cs336_alignment_idea_6", diff_dir="diffs"):
    """
    Apply all .diff files in diff_dir to a copy of main_repo_dir, creating new_repo_dir.
    The .diff files are applied in sorted order.
    """
    main_repo_dir = Path(main_repo_dir).resolve()
    new_repo_dir = Path(new_repo_dir).resolve()
    diff_dir = Path(diff_dir).resolve()

    if not main_repo_dir.is_dir():
        raise FileNotFoundError(f"Original repo directory not found: {main_repo_dir}")
    if not diff_dir.is_dir():
        raise FileNotFoundError(f"Diff directory not found: {diff_dir}")

    # Remove target directory if it already exists
    if new_repo_dir.exists():
        shutil.rmtree(new_repo_dir)

    # Copy the repo contents directly (not as a subdirectory)
    # First create the target directory
    new_repo_dir.mkdir(parents=True, exist_ok=True)
    # Copy contents of main_repo_dir into new_repo_dir
    for item in main_repo_dir.iterdir():
        if item.is_dir():
            shutil.copytree(item, new_repo_dir / item.name, symlinks=True)
        else:
            shutil.copy2(item, new_repo_dir)

    def run(cmd, cwd=new_repo_dir, check=True):
        result = subprocess.run(
            cmd,
            cwd=cwd,
            text=True,
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if check and result.returncode != 0:
            raise subprocess.CalledProcessError(
                result.returncode, cmd, output=result.stdout, stderr=result.stderr
            )
        return result

    # Find all .diff files in diff_dir, sort them
    diff_files = sorted(diff_dir.glob("*.diff"))
    if not diff_files:
        raise FileNotFoundError(f"No .diff files found in {diff_dir}")

    for diff_file in diff_files:
        try:
            run(["git", "apply", "--stat", str(diff_file)])
            run(["git", "apply", "--check", "--allow-empty", str(diff_file)])
            run(["git", "apply", "--allow-empty", str(diff_file)])
            print (f"Applied diff {diff_file} successfully!")
        except subprocess.CalledProcessError as e:
            # fallback with rejects
            fallback = run(["git", "apply", "--reject", "--whitespace=fix", "--allow-empty", str(diff_file)], check=False)
            if fallback.returncode != 0:
                raise RuntimeError(
                    f"Failed to apply diff {diff_file}; primary error:\n{e.stderr or e.output}\n"
                    f"Fallback also failed:\n{fallback.stderr or fallback.stdout}"
                )
            else:
                # if .rej files exist, inform user
                rej_files = list(new_repo_dir.rglob("*.rej"))
                if rej_files:
                    raise RuntimeError(
                        f"Diff {diff_file} applied with rejects; need manual resolution. .rej files: {', '.join(str(p) for p in rej_files)}"
                    )
    return new_repo_dir


def feedback_loop(idea_idx = 6, idea_file = "agent/all_ideas.json", main_repo_dir = "cs336_alignment", new_repo_dir = "cs336_alignment_idea_6", diff_file = "diffs/code_diff_idea_6.diff", diff_error_file = "diffs/code_diff_error_idea_6.txt"):
    ## iteratively refine the code diff if not applied successfully
    max_trials = 10
    trial_num = 0
    success = False
    while not success and trial_num < max_trials:
        trial_num += 1
        print (f"Trial {trial_num}...")
        ## generate the code diff
        if trial_num == 1:
            response, thinking = generate_code_diff(idea_idx=idea_idx, base_dir="env_grpo", idea_file=idea_file)
        else:
            response, thinking = generate_code_diff(idea_idx=idea_idx, base_dir="env_grpo", idea_file=idea_file, previous_diff_file=diff_file, previous_diff_error=diff_error_file)
        with open(diff_file, "w") as f:
            f.write(response)
        print (f"Diff file generated: {diff_file}")

        try:
            apply_code_diff(main_repo_dir=main_repo_dir, new_repo_dir=new_repo_dir, diff_file=diff_file)
            success = True
            print (f"Trial {trial_num} of code diff generation succeeded!")
        except RuntimeError as e:
            ## log the code diff merge error
            with open(diff_error_file, "w") as log_file:
                log_file.write(str(e) + "\n")
            print (f"Trial {trial_num} of code diff generation failed!")


def _generate_code_diff_parallel_helper(idea_idx, max_trials, env_dir, repo_dir, idea_file, diffs_dir, idea_lst = None, model_name = "gpt-5"):
    print (f"Generating code diff for idea {idea_idx}...")
    trial_num = 0
    error_msg = None
    while trial_num < max_trials:
        trial_num += 1
        print (f"Trial {trial_num}...")
        try:
            if trial_num > 1 and error_msg is not None:
                _, response = generate_code_diff(idea_idx=idea_idx, base_dir=env_dir, variant_dir=repo_dir, idea_file=idea_file, diff_dir=diffs_dir, prev_diff_file=f"{diffs_dir}/code_diff_idea_{idea_idx}.diff", prev_diff_error=error_msg, idea_lst=idea_lst, model_name=model_name)
            else:
                _, response = generate_code_diff(idea_idx=idea_idx, base_dir=env_dir, variant_dir=repo_dir, idea_file=idea_file, diff_dir=diffs_dir, idea_lst=idea_lst, model_name=model_name)
            with open(f"{diffs_dir}/code_diff_idea_{idea_idx}.diff", "w") as f:
                f.write(response)
            code_diff_fixer(diff_file=f"{diffs_dir}/code_diff_idea_{idea_idx}.diff")

            new_repo_dir = f"{repo_dir}/idea_{idea_idx}"
            diff_file = f"{diffs_dir}/code_diff_idea_{idea_idx}.diff"
            apply_code_diff(main_repo_dir=env_dir, new_repo_dir=new_repo_dir, diff_file=diff_file)
            print (f"Trial {trial_num} of code diff generation succeeded!")
            break
        except Exception as e:
            error_msg = str(e)
            print (f"Trial {trial_num} of code diff generation failed with error: {error_msg}")

            new_repo_dir = f"{repo_dir}/idea_{idea_idx}"
            if os.path.exists(new_repo_dir):
                shutil.rmtree(new_repo_dir)

def generate_code_diff_parallel(max_trials = 10, diffs_dir = "diffs_epoch1_parallel_corrected", repo_dir = "repo_variants_epoch1_parallel_corrected", env_dir = "env_grpo", idea_file = "ideas/all_ideas_0826_epoch1.json", idea_lst = None, model_name = "gpt-5", total_workers = 10):
    os.makedirs(diffs_dir, exist_ok=True)
    os.makedirs(repo_dir, exist_ok=True)
    if idea_lst is not None:
        ideas = idea_lst
    else:
        with open(idea_file, "r") as f:
            ideas = json.load(f)
    total_ideas = len(ideas)

    map_func = partial(_generate_code_diff_parallel_helper, max_trials=max_trials, env_dir=env_dir, repo_dir=repo_dir, idea_file=idea_file, diffs_dir=diffs_dir, idea_lst=idea_lst, model_name=model_name)
    print(f"Using {total_workers} workers")
    with ThreadPoolExecutor(max_workers=total_workers) as executor:
        executor.map(map_func, range(total_ideas))

def agent_call_idea(num_ideas = 200, cache_file = "ideas/all_ideas_claude_epoch2.json", run_name = "GRPO-env-test", epoch_num = 1, prev_ideas_file = "ideas/all_ideas_0826_epoch1.json", prev_training_logs = "training_logs/epoch1_retrywrapper/", top_k=20, sample_k=100, env_dir = "env_grpo", model_name = "gpt-5"):
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)

    if epoch_num == 0:
        ideas_per_batch = 10
        num_batches = num_ideas // ideas_per_batch
        for i in tqdm(range(num_batches)):
            agent_call_idea_simple(num_ideas=ideas_per_batch, cache_file=cache_file, env_dir=env_dir, model_name=model_name)
    else:
        agent_call_idea_evolutionary(total_num_ideas = num_ideas, run_name = run_name, epoch_num = epoch_num, top_k=top_k, sample_k=sample_k, cache_file = cache_file, env_dir=env_dir, model_name=model_name)

    return


if __name__ == "__main__":
    # ## generate the experiments
    # for i in tqdm(range(2)):
    #     response, thinking = agent_call_idea_simple(num_ideas=10, cache_file="agent/all_ideas_0826_epoch1.json")

    ## generate the code diff for each experiment
    # generate_code_diff_parallel(max_trials=10, diffs_dir="diffs_epoch1_parallel_corrected", repo_dir="repo_variants_epoch1_parallel_corrected", env_dir="env_grpo", idea_file="ideas/all_ideas_0826_epoch1.json")


    # ## generate the updated ideas
    # for i in tqdm(range(20)):
    #     agent_call_idea_update(num_ideas = 10, prev_ideas_file = "ideas/all_ideas_0826_epoch1.json", prev_training_logs = "training_logs/epoch1_retrywrapper/", top_k=20, cache_file = "ideas/all_ideas_claude_epoch2.json")

    # generate_code_diff_parallel(max_trials=10, diffs_dir="diffs_claude_epoch2", repo_dir="repo_variants_claude_epoch2", env_dir="env_grpo", idea_file="ideas/all_ideas_claude_epoch2.json")

    apply_code_diff(main_repo_dir = "env/grpo", new_repo_dir = "repo_variants_api_testing/idea_1", diff_file = "diffs_api_testing/code_diff_idea_1.diff")
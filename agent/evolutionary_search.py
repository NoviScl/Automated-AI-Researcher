import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.api import apiqa
from retry import retry
import json
from tqdm import tqdm
import re
import math
import random
random.seed(42)


def context_prompt(base_dir = "env_grpo"):
    prompt_parts = []

    for root, dirs, files in os.walk(base_dir):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for file in files:
            if file.endswith(".py") or file.endswith(".sh"):
                file_path = os.path.join(root, file)
                if file == "evaluate.py":
                    prompt_parts.append(f"===== {file_path} =====\nThis file is reserved for evaluating the model. You not allowed to read or modify this file.\n")
                elif file == "fineweb.py" or file == "run.sh":
                    continue ## skip the data processing script or debug script
                else:
                    print (file)
                    with open(file_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                    numbered_content = "".join(f"{i+1}: {line}" for i, line in enumerate(lines))
                    prompt_parts.append(f"===== {file_path} =====\n{numbered_content}\n")

    return "\n".join(prompt_parts)

def diff_prompt(diff_file = "diffs/code_diff_idea_0.diff"):
    with open(diff_file, "r") as f:
        diff_lines = f.readlines()
    # Prepend line numbers to each line
    diff_str_with_lines = "".join(f"{i+1}: {line}" for i, line in enumerate(diff_lines))
    return diff_str_with_lines

def strip_response(response, token="```diff"):
    if response.startswith(token):
        response = response.split(token)[1].strip()
    if response.endswith("```"):
        response = response.split("```")[0].strip()
    return response

def filter_log(log_lines):
    filtered_lines = []
    for line in log_lines.split("\n"):
        if "it/s" not in line and "wandb:" not in line and "WARNING:" not in line:
            filtered_lines.append(line)
    return "\n".join(filtered_lines)

def log_prompt():
    log_dir = "logs"
    log_parts = []
    log_files = []
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if file.endswith(".log"):
                file_path = os.path.join(root, file)
                log_files.append(file_path)
    # Sort log files in sequential order
    log_files.sort()
    for file_path in log_files:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        log_parts.append(f"===== {file_path} =====\n{filter_log(content)}\n")
    return "\n".join(log_parts)


def update_database(run_name = "GRPO-env-test", epoch_num = 0):
    '''
    gather all ideas and training logs and update the database
    '''

    ## database should be a ranked list of all ideas under this run
    if os.path.exists(f"ideas_{run_name}/database.json"):
        with open(f"ideas_{run_name}/database.json", "r") as f:
            database_lst = json.load(f)
    else:
        database_lst = []

    ## load ideas
    with open(f"ideas_{run_name}/ideas_epoch{epoch_num}.json", "r") as f:
        ideas_lst = json.load(f)

    ## read ranked ideas directly from ranked_ideas.json instead of recomputing
    base_idea_dir = f"training_logs_{run_name}/epoch{epoch_num}/"
    ranked_path = os.path.join(base_idea_dir, "ranked_ideas.json")
    if os.path.exists(ranked_path):
        with open(ranked_path, "r") as f:
            ranked_ideas_list = json.load(f)
        for idea_dict in ranked_ideas_list:
            if not idea_dict:
                continue
            idea_key, reward_value = next(iter(idea_dict.items()))
            m = re.search(r"idea_(\d+)", idea_key)
            if not m:
                continue
            idea_id = int(m.group(1))
            if 0 <= idea_id < len(ideas_lst):
                if "grpo" in run_name.lower():
                    idea_dp = {
                        "epoch": epoch_num,
                        "idea_id": idea_id,
                        "idea": ideas_lst[idea_id],
                        "best_eval_accuracy": reward_value if reward_value is not None else -999,
                    }
                elif "nanogpt" in run_name.lower():
                    ## skip reward hacking examples
                    if reward_value is None or (isinstance(reward_value, float) and math.isnan(reward_value)) or reward_value < 2.5:
                        continue
                    idea_dp = {
                        "epoch": epoch_num,
                        "idea_id": idea_id,
                        "idea": ideas_lst[idea_id],
                        "lowest_val_loss": reward_value if reward_value is not None else -999,
                    }
                # elif "nanogpt" in run_name.lower():
                #     idea_dp = {
                #         "epoch": epoch_num,
                #         "idea_id": idea_id,
                #         "idea": ideas_lst[idea_id],
                #         "fastest_time_to_target": reward_value if reward_value is not None else -999,
                #     }
                database_lst.append(idea_dp)

    # Remove duplicates: keep only the best entry for each (epoch, idea_id) pair
    unique_dict = {}
    for entry in database_lst:
        key = (entry["epoch"], entry["idea_id"])
        # Determine the metric key and comparison direction based on run_name
        if "grpo" in run_name.lower():
            metric_key = "best_eval_accuracy"
            # For GRPO, keep the entry with the highest accuracy
            if key not in unique_dict or entry[metric_key] > unique_dict[key][metric_key]:
                unique_dict[key] = entry
        elif "nanogpt" in run_name.lower():
            metric_key = "lowest_val_loss"
            # For NanoGPT, ignore entries with NaN values, and keep the entry with the lowest validation loss
            metric_val = entry[metric_key]
            # Exclude entries with NaN metric
            if metric_val is not None and not (isinstance(metric_val, float) and math.isnan(metric_val)):
                if key not in unique_dict or metric_val < unique_dict[key][metric_key]:
                    unique_dict[key] = entry
        # elif "nanogpt" in run_name.lower():
        #     metric_key = "fastest_time_to_target"
        #     # For NanoGPT, keep the entry with the lowest positive time_to_target (ignore -999 or <=0)
        #     current_val = entry[metric_key]
        #     prev_val = unique_dict[key][metric_key] if key in unique_dict else None
        #     # Only consider positive values as valid
        #     if current_val is not None and current_val > 0:
        #         if (
        #             key not in unique_dict
        #             or prev_val is None
        #             or prev_val <= 0
        #             or current_val < prev_val
        #         ):
        #             unique_dict[key] = entry
        #     # If no positive value has been seen yet, allow -999 as fallback
        #     elif key not in unique_dict:
        #         unique_dict[key] = entry
        else:
            # Default: just keep the first entry
            if key not in unique_dict:
                unique_dict[key] = entry

    deduped_database_lst = list(unique_dict.values())

    # For GRPO, sort descending by best_eval_accuracy
    # For NanoGPT, push non-positive to end, then sort ascending by fastest_time_to_target
    if "grpo" in run_name.lower():
        ranked_database = sorted(
            deduped_database_lst,
            key=lambda x: x["best_eval_accuracy"],
            reverse=True
        )
    elif "nanogpt" in run_name.lower():
        ranked_database = sorted(
            deduped_database_lst,
            key=lambda x: float("inf") if x["lowest_val_loss"] is None or x["lowest_val_loss"] <= 0 else x["lowest_val_loss"]
        )
    # elif "nanogpt" in run_name.lower():
    #     ranked_database = sorted(
    #         deduped_database_lst,
    #         key=lambda x: float("inf") if x["fastest_time_to_target"] is None or x["fastest_time_to_target"] <= 0 else x["fastest_time_to_target"]
    #     )
    else:
        # Default: sort by epoch and idea_id
        ranked_database = sorted(
            deduped_database_lst,
            key=lambda x: (x["epoch"], x["idea_id"])
        )

    with open(f"ideas_{run_name}/database.json", "w") as f:
        json.dump(ranked_database, f, indent=4)



@retry(tries=3, delay=2)
def agent_call_idea_evolutionary_exploit(num_ideas = 10, idea_database = "ideas_GRPO-env-test/database.json", top_k=10, cache_file = "ideas_GRPO-env-test/ideas_epoch1_evolutionary.json", env_dir = "env_grpo", model_name = "gpt-5"):
    '''
    Read the previous batches of ideas, take the top-performing ideas, shove them into the prompt, and generate a new batch of ideas.
    '''
    context = context_prompt(base_dir=env_dir)
    system_message = f"""
    You are a research scientist that helps me decide what are some promising experiments to run to improve the performance of the model.
    """

    ## read the previous ideas
    with open(idea_database, "r") as f:
        idea_database_lst = json.load(f)

    if "grpo" in env_dir.lower():
        baseline_threshold = 0.49
    # elif "nanogpt_1gpu" in env_dir.lower():
    #     baseline_threshold = 3.63
    elif "nanogpt" in env_dir.lower():
        baseline_threshold = 3.255

    ## get all ideas above the baseline threshold
    if "grpo" in env_dir.lower():
        top_candidates = [idea_dict for idea_dict in idea_database_lst if idea_dict["best_eval_accuracy"] > baseline_threshold]
    elif "nanogpt" in env_dir.lower():
        top_candidates = [idea_dict for idea_dict in idea_database_lst if idea_dict["lowest_val_loss"] > 0 and idea_dict["lowest_val_loss"] < baseline_threshold]
    # elif "nanogpt" in env_dir.lower():
    #     top_candidates = [idea_dict for idea_dict in idea_database_lst if idea_dict["fastest_time_to_target"] > 0 and idea_dict["fastest_time_to_target"] < baseline_threshold]

    prev_ideas_prompt = ""
    # Sample top_k ideas from the candidates as the parent ideas
    if len(top_candidates) <= top_k:
        sampled_ideas = top_candidates
        random.shuffle(sampled_ideas)
    else:
        sampled_ideas = random.sample(top_candidates, top_k)

    if "grpo" in env_dir.lower():
        for idea_dict in sampled_ideas:
            prev_ideas_prompt += "Idea: " + idea_dict["idea"] + "\n"
            prev_ideas_prompt += "Eval Accuracy: " + str(idea_dict["best_eval_accuracy"]) + "\n\n"
    elif "nanogpt" in env_dir.lower():
        for idea_dict in sampled_ideas:
            prev_ideas_prompt += "Idea: " + idea_dict["idea"] + "\n"
            prev_ideas_prompt += "Final Validation Loss: " + str(idea_dict["lowest_val_loss"]) + "\n\n"
    # elif "nanogpt" in env_dir.lower():
    #     for idea_dict in sampled_ideas:
    #         prev_ideas_prompt += "Idea: " + idea_dict["idea"] + "\n"
    #         if idea_dict["fastest_time_to_target"] < 0:
    #             prev_ideas_prompt += "Fastest Time to Target: Either failed to execute or just never reached the target loss of 3.28 before timeout.\n\n"
    #         else:
    #             prev_ideas_prompt += "Fastest Time to Target: " + str(idea_dict["fastest_time_to_target"]) + "\n\n"

    if "grpo" in env_dir.lower():
        # prompt = f"""
        # Below is the list of all code files in this codebase including the example script to launch a job:
        # {context}\n\n
        # Below is the list of successful ideas that have been tried and their final accuracy:
        # {prev_ideas_prompt}
        # They all beat the baseline accuracy of {baseline_threshold} achieved by running the original codebase without any modifications.
        # Now I want you to generate {num_ideas} ideas that directly improve upon these successful ideas.
        # You should exploit good practices from the previous experiments, for example, by combining successful ideas, making refinement to them, etc.
        # The goal is to generate {num_ideas} ideas that are even better than the previous ideas.
        # For each experiment, include two sub-sections: (1) concise description of the experiment; (2) brief summary of the code changes needed. Format each block with the following tags:
        # [Experiment] ...
        # [Code Changes] ...
        # Add a tag [End] after each experiment.
        # Make sure to elaborate on the full experiment details because the experiment executor does not have context of prior experiments.
        # If the experiment involves changing any hyperparameters, you should specify the new values of the hyperparameters.
        # Also note that you are not allowed to change any part of the evaluation logic, including the evaluation data and the evaluation metrics; and you are not allowed to change the eval frequency or the hard time limit set in the run_job.sh script.
        # """
        prompt = f"""
        Below is the list of all code files in this codebase including the example script to launch a job:
        {context}\n\n
        Below is the list of successful ideas that have been tried and their final accuracy:
        {prev_ideas_prompt}
        They all beat the baseline accuracy of {baseline_threshold} achieved by running the original codebase without any modifications.
        Now I want you to generate {num_ideas} ideas that directly improve upon these successful ideas.
        You should exploit good practices from the previous experiments, for example, by combining successful ideas, making refinement to them, etc.
        However, make sure to focus on novel training algorithms that can improve the performance of the model rather than just tweaking the hyperparameters. Avoid just naively stacking many known tricks together; instead, synthesize elegant research ideas that leverage the positive insights from the previous experiments.
        The goal is to generate {num_ideas} ideas that are even better than the previous ideas.
        For each experiment, include two sub-sections: (1) concise description of the experiment; (2) brief summary of the code changes needed. Format each block with the following tags:
        [Experiment] ...
        [Code Changes] ...
        Add a tag [End] after each experiment.
        Make sure to elaborate on the full experiment details because the experiment executor does not have context of prior experiments.
        If the experiment involves changing any hyperparameters, you should specify the new values of the hyperparameters.
        Also note that you are not allowed to change any part of the evaluation logic, including the evaluation data and the evaluation metrics; and you are not allowed to change the eval frequency or the hard time limit set in the run_job.sh script.
        """
    elif "nanogpt" in env_dir.lower():
        prompt = f"""
        I'm trying to optimize the training for the nanoGPT speedrun. The goal is to reach the lowest possible validation loss within one hour of wall clock time.
        Below is the list of all code files in this codebase including the example script to launch a job:
        {context}\n\n
        Below is the list of positive ideas that have been tried and their lowest validation loss in the end:
        {prev_ideas_prompt}
        They all beat the baseline code that can reach the final validation loss of {baseline_threshold}.
        Now I want you to generate {num_ideas} ideas that directly improve upon these successful ideas.
        You should exploit good practices from the previous experiments, for example, by combining successful ideas, making refinement to them, etc.
        The goal is to generate {num_ideas} ideas that are even better than the previous ideas.
        For each experiment, include two sub-sections: (1) concise description of the experiment; (2) brief summary of the code changes needed. Format each block with the following tags:
        [Experiment] ...
        [Code Changes] ...
        Add a tag [End] after each experiment.
        Make sure to elaborate on the full experiment details because the experiment executor does not have context of prior experiments.
        If the experiment involves changing any hyperparameters, you should specify the new values of the hyperparameters.
        External imports are not supported so you should suggest experiments that can be implemented by directly changing a few functions in the codebase.
        Also note that you are not allowed to change any part of the evaluation logic, including the evaluation data and the evaluation metrics; and you are not allowed to change the eval frequency or the hard time limit set in the run_job.sh script.
        You should absolutely not change the loss function part: loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1); or the validation hyperparameters including val_loss_every and val_tokens (we only do the valiation at the end of the training). You are not allowed to change the time limit of 3600 seconds in "if elapsed_time_seconds > 3600:" either. These must be left unchanged for fair comparison!
        We have implemented functions like `forward_with_cache` and `forward_safe` in the codebase, these are meant to preserve the autoregressive nature of the model and avoid leaking information from future tokens during the validation step. You should not break the autoregressive nature of these functions.
        Also avoid any possible case of leaking information from future tokens that breaks the autoregressive nature of the model. For example, you should not mix in any future tokens into the current token's representation or normalize across the entire sequence.
        """
    # elif "nanogpt" in env_dir.lower():
    #     prompt = f"""
    #     I'm trying to optimize the training speed for the nanoGPT speedrun. The goal is to reach the target loss of 3.28 as fast as possible.
    #     Below is the list of all code files in this codebase including the example script to launch a job:
    #     {context}\n\n
    #     Below is the list of positive ideas that have been tried and their fastest time to reach the target:
    #     {prev_ideas_prompt}
    #     They all beat the baseline code that can reach the target loss in {baseline_threshold} ms.
    #     Now I want you to generate {num_ideas} ideas that directly improve upon these successful ideas.
    #     You should exploit good practices from the previous experiments, for example, by combining successful ideas, making refinement to them, etc.
    #     The goal is to generate {num_ideas} ideas that are even better than the previous ideas.
    #     For each experiment, include two sub-sections: (1) concise description of the experiment; (2) brief summary of the code changes needed. Format each block with the following tags:
    #     [Experiment] ...
    #     [Code Changes] ...
    #     Add a tag [End] after each experiment.
    #     Make sure to elaborate on the full experiment details because the experiment executor does not have context of prior experiments.
    #     If the experiment involves changing any hyperparameters, you should specify the new values of the hyperparameters.
    #     Also note that you are not allowed to change any part of the evaluation logic, including the evaluation data and the evaluation metrics; and you are not allowed to change the eval frequency or the hard time limit set in the run_job.sh script.
    #     You should absolutely not change the loss function part: loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1); or the validation hyperparameters including val_loss_every and val_tokens. These must be left unchanged for fair comparison!
    #     """

    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            cache_lst = json.load(f)
        cache_str = "\n\n".join(cache_lst)
        prompt += f"""
        Avoid any similar ideas to the following experiments that have been proposed before:
        {cache_str}
        """
    response = apiqa(prompt, model_name, system_message, claude_thinking_mode=True, claude_thinking_budget=1500, temperature=1.0)
    if len(response) == 2:
        thinking, response = response
    else:
        thinking = ""
    # print (prompt)
    print (thinking)
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
def agent_call_idea_evolutionary_explore(num_ideas = 10, idea_database = "ideas_GRPO-env-test/database.json", sample_k=100, cache_file = "ideas_GRPO-env-test/ideas_epoch1_evolutionary.json", env_dir = "env_grpo", model_name = "gpt-5"):
    '''
    Read the previous batches of ideas, take the top-performing ideas, shove them into the prompt, and generate a new batch of ideas.
    '''
    context = context_prompt(base_dir=env_dir)
    system_message = f"""
    You are a research scientist that helps me decide what are some promising experiments to run to improve the performance of the model.
    """

    ## read the previous ideas and randomly sample top_k for context
    with open(idea_database, "r") as f:
        idea_database_lst = json.load(f)
    if len(idea_database_lst) <= sample_k:
        sampled_ideas = idea_database_lst
        random.shuffle(sampled_ideas)
    else:
        sampled_ideas = random.sample(idea_database_lst, sample_k)

    prev_ideas_prompt = ""
    if "grpo" in env_dir.lower():
        baseline_threshold = 0.49
        for idea_dict in sampled_ideas:
            prev_ideas_prompt += "Idea: " + idea_dict["idea"] + "\n"
            if idea_dict["best_eval_accuracy"] < 0:
                prev_ideas_prompt += "Eval Accuracy: Failed to implement or execute\n\n"
            else:
                prev_ideas_prompt += "Eval Accuracy: " + str(idea_dict["best_eval_accuracy"]) + "\n\n"
    elif "nanogpt" in env_dir.lower():
        baseline_threshold = 3.255
        for idea_dict in sampled_ideas:
            prev_ideas_prompt += "Idea: " + idea_dict["idea"] + "\n"
            if idea_dict["lowest_val_loss"] < 0:
                prev_ideas_prompt += "Lowest Validation Loss: Failed to implement or execute\n\n"
            else:
                prev_ideas_prompt += "Lowest Validation Loss: " + str(idea_dict["lowest_val_loss"]) + "\n\n"
    # elif "nanogpt" in env_dir.lower():
    #     baseline_threshold = 2116778
    #     for idea_dict in sampled_ideas:
    #         prev_ideas_prompt += "Idea: " + idea_dict["idea"] + "\n"
    #         if idea_dict["fastest_time_to_target"] < 0:
    #             prev_ideas_prompt += "Fastest Time to Target: Either failed to execute or just never reached the target loss of 3.28 before timeout.\n\n"
    #         else:
    #             prev_ideas_prompt += "Fastest Time to Target: " + str(idea_dict["fastest_time_to_target"]) + "\n\n"


    if "grpo" in env_dir.lower():
        # prompt = f"""
        # Below is the list of all code files in this codebase including the example script to launch a job:
        # {context}\n\n
        # Below is the list of all ideas that have been tried and their final accuracy:
        # {prev_ideas_prompt}
        # For your reference, the baseline accuracy of running the original codebase without any modifications is {baseline_threshold}.
        # Now I want you to generate {num_ideas} brand new ideas that are different from all the previous ideas.
        # Make sure to leverage some insights from the previous ideas by avoiding patterns that often lead to negative results.
        # You can also try out random ideas that have nothing to do with the previous ideas but you think are worth trying.
        # For each experiment, include two sub-sections: (1) concise description of the experiment; (2) brief summary of the code changes needed. Format each block with the following tags:
        # [Experiment] ...
        # [Code Changes] ...
        # Add a tag [End] after each experiment.
        # If the experiment involves changing any hyperparameters, you should specify the new values of the hyperparameters.
        # Also note that you are not allowed to change any part of the evaluation logic, including the evaluation data and the evaluation metrics; and you are not allowed to change the eval frequency or the hard time limit set in the run_job.sh script.
        # """
        prompt = f"""
        Below is the list of all code files in this codebase including the example script to launch a job:
        {context}\n\n
        Below is the list of all ideas that have been tried and their final accuracy:
        {prev_ideas_prompt}
        For your reference, the baseline accuracy of running the original codebase without any modifications is {baseline_threshold}.
        Now I want you to generate {num_ideas} brand new ideas that are different from all the previous ideas.
        Make sure to leverage some insights from the previous ideas by avoiding patterns that often lead to negative results.
        You can also try out random ideas that have nothing to do with the previous ideas but you think are worth trying.
        You should focus on novel training algorithms that might improve the performance of the model. This can include anything such as modifying the loss function, designing new training rewards, implementing new training curriculum, or anything else that is well-motivated and makes sense.
        Note that you should come up with generalizable and scalable ideas that can transfer to other models and datasets, which means you probably shouldn't generate ideas that are too specific to the current setup, such as massively tuning the hyperparameters.
        For each experiment, include two sub-sections: (1) concise description of the experiment; (2) brief summary of the code changes needed. Format each block with the following tags:
        [Experiment] ...
        [Code Changes] ...
        Add a tag [End] after each experiment.
        If the experiment involves changing any hyperparameters, you should specify the new values of the hyperparameters.
        Also note that you are not allowed to change any part of the evaluation logic, including the evaluation data and the evaluation metrics; and you are not allowed to change the eval frequency or the hard time limit set in the run_job.sh script.
        """
    elif "nanogpt" in env_dir.lower():
        prompt = f"""
        I'm trying to optimize the training loss for the nanoGPT speedrun. The goal is to reach the lowest possible validation loss within one hour of wall clock time.
        Below is the list of all code files in this codebase including the example script to launch a job:
        {context}\n\n
        Below is the list of all ideas that have been tried and their validation loss in the end:
        {prev_ideas_prompt}
        For your reference, the baseline code can reach the a final validation loss of {baseline_threshold}.
        Now I want you to generate {num_ideas} brand new ideas that are different from all the previous ideas.
        You should explore novel ideas that have not been well-established in pretraining research yet. I'm looking for some clean and simple ideas instead of stacking many known tricks together or simply tweaking the hyperparameters.
        For each experiment, include two sub-sections: (1) concise description of the experiment; (2) brief summary of the code changes needed. Format each block with the following tags:
        [Experiment] ...
        [Code Changes] ...
        Add a tag [End] after each experiment.
        If the experiment involves changing any hyperparameters, you should specify the new values of the hyperparameters.
        External imports are not supported so you should suggest experiments that can be implemented by directly changing a few functions in the codebase.
        Also note that you are not allowed to change any part of the evaluation logic, including the evaluation data and the evaluation metrics; and you are not allowed to change the eval frequency or the hard time limit set in the run_job.sh script.
        You should absolutely not change the loss function part: loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1); or the validation hyperparameters including val_loss_every and val_tokens (we only do the valiation at the end of the training). You are not allowed to change the time limit of 3600 seconds in "if elapsed_time_seconds > 3600:" either. These must be left unchanged for fair comparison!
        Also avoid any possible case of leaking information from future tokens that breaks the autoregressive nature of the model. For example, you should not mix in any future tokens into the current token's representation or normalize across the entire sequence.
        We have implemented functions like `forward_with_cache` and `forward_safe` in the codebase, these are meant to preserve the autoregressive nature of the model and avoid leaking information from future tokens during the validation step. You should not break the autoregressive nature of these functions.
        """
    # elif "nanogpt" in env_dir.lower():
    #     prompt = f"""
    #     I'm trying to optimize the training speed for the nanoGPT speedrun. The goal is to reach the target loss of 3.28 as fast as possible.
    #     Below is the list of all code files in this codebase including the example script to launch a job:
    #     {context}\n\n
    #     Below is the list of all ideas that have been tried and their fastest time to reach the target:
    #     {prev_ideas_prompt}
    #     For your reference, the baseline code can reach the target loss in {baseline_threshold} ms.
    #     Now I want you to generate {num_ideas} brand new ideas that are different from all the previous ideas.
    #     Make sure to leverage some insights from the previous ideas by avoiding patterns that often lead to negative results.
    #     You can also try out random ideas that have nothing to do with the previous ideas but you think are worth trying.
    #     For each experiment, include two sub-sections: (1) concise description of the experiment; (2) brief summary of the code changes needed. Format each block with the following tags:
    #     [Experiment] ...
    #     [Code Changes] ...
    #     Add a tag [End] after each experiment.
    #     If the experiment involves changing any hyperparameters, you should specify the new values of the hyperparameters.
    #     Also note that you are not allowed to change any part of the evaluation logic, including the evaluation data and the evaluation metrics; and you are not allowed to change the eval frequency or the hard time limit set in the run_job.sh script.
    #     You should absolutely not change the loss function part: loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1); or the validation hyperparameters including val_loss_every and val_tokens. These must be left unchanged for fair comparison!
    #     """

    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            cache_lst = json.load(f)
        cache_str = "\n\n".join(cache_lst)
        prompt += f"""
        Avoid any similar ideas to the following experiments that have been proposed before:
        {cache_str}
        """
    response = apiqa(prompt, model_name, system_message, claude_thinking_mode=True, claude_thinking_budget=1500, temperature=1.0)
    if len(response) == 2:
        thinking, response = response
    else:
        thinking = ""
    # print (prompt)
    print (thinking)
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

def agent_call_idea_evolutionary(total_num_ideas = 200, run_name = "GRPO-env-test", epoch_num = 1, top_k=10, sample_k=100, cache_file = "ideas_GRPO-env-test/ideas_epoch1_evolutionary.json", env_dir = "env_grpo", model_name = "gpt-5"):
    ## update the database
    update_database(run_name = run_name, epoch_num = epoch_num - 1)

    ## decide the distribution of exploit and explore
    max_exploit_ratio = min(0.5 + 0.1 * (epoch_num // 2), 0.8)
    # max_exploit_ratio = 0.5
    num_exploit = int(total_num_ideas * max_exploit_ratio)
    num_exploit = (num_exploit // 10) * 10
    num_explore = total_num_ideas - num_exploit

    ## generate the exploit ideas
    ideas_per_batch = 10
    num_batches = num_exploit // ideas_per_batch
    for i in tqdm(range(num_batches)):
        agent_call_idea_evolutionary_exploit(num_ideas = ideas_per_batch, idea_database = f"ideas_{run_name}/database.json", top_k=top_k, cache_file = cache_file, env_dir=env_dir, model_name=model_name)

    ## generate the explore ideas
    num_batches = num_explore // ideas_per_batch
    for i in tqdm(range(num_batches)):
        agent_call_idea_evolutionary_explore(num_ideas = ideas_per_batch, idea_database = f"ideas_{run_name}/database.json", sample_k=sample_k, cache_file = cache_file, env_dir=env_dir, model_name=model_name)



if __name__ == "__main__":
    update_database(run_name = "nanogpt_safe_full_bsz80", epoch_num = 11)

    # agent_call_idea_evolutionary_exploit(num_ideas = 10, idea_database = "ideas_GRPO-env-test/database.json", baseline_threshold = 0.49, top_k=10, cache_file = "ideas_GRPO-env-test/ideas_epoch1_evolutionary.json")
    # agent_call_idea_evolutionary_explore(num_ideas = 10, idea_database = "ideas_GRPO-env-test/database.json", baseline_threshold = 0.49, cache_file = "ideas_GRPO-env-test/ideas_epoch1_evolutionary.json")
    # agent_call_idea_evolutionary(total_num_ideas = 200, run_name = "GRPO-env-test", epoch_num = 0, baseline_threshold = 0.49, top_k=10, sample_k=100, cache_file = "ideas_GRPO-env-test/ideas_epoch1.json")

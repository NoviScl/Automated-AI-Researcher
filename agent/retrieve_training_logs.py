import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wandb
import fnmatch
import os
import json
import time

def _retry_wandb(call, desc: str, max_attempts: int = 9, base_delay: float = 2.0):
    """Retry a callable with exponential backoff, for transient WANDB/network errors."""
    for attempt in range(1, max_attempts + 1):
        try:
            return call()
        except Exception as e:  # noqa: BLE001 - robust against wandb/request exceptions
            if attempt == max_attempts:
                print(f"W&B {desc} failed after {attempt} attempts: {e}")
                raise
            delay = base_delay * (2 ** (attempt - 1))
            print(
                f"W&B {desc} error: {e}. Retrying in {int(delay)}s (attempt {attempt}/{max_attempts})"
            )
            time.sleep(delay)


def extract_metrics(series, metric_name):
    """
    Given a pandas Series (e.g., run.history()["eval/mean_reward"]),
    return a list of dicts with {"step": idx, "reward": value} for non-NaN values.
    """
    # Drop NaN values and keep index
    filtered = series.dropna()
    return [{"step": int(idx), metric_name: float(val)} for idx, val in filtered.items()]

def extract_metrics_nanogpt(lines, target_loss = 3.28):
    eval_rewards = []
    time_to_target = -999
    for line in lines:
        if "val_loss:" in line:
            parts = line.split(" val_loss:")
            step = parts[0].split("/")[0].replace("step:", "").strip()
            val_loss = parts[1].split("train_time:")[0].strip()
            train_time_ms = parts[1].split("train_time:")[1].strip().replace("ms", "")
            val_loss_f = float(val_loss)
            train_time_i = int(train_time_ms)
            eval_rewards.append({"step": int(step), "val_loss": val_loss_f, "train_time_ms": train_time_i})
            if val_loss_f <= target_loss and time_to_target == -999:
                time_to_target = train_time_i
    return eval_rewards, time_to_target

def get_run_name(run_name, epoch_num, idea_number = None):
    # if "grpo" in run_name.lower() or "nanogpt_1gpu" in run_name.lower():
    #     machine = "b200"
    # elif "nanogpt" in run_name.lower():
    #     machine = "h100"
    # else:
    #     raise ValueError(f"Unknown run name: {run_name}")
    #     machine = "h100"

    ## all experiments are on b200 now 
    machine = "b200"
    if idea_number is None:
        name_pattern = f"{run_name}_epoch{epoch_num}_{machine}_idea_*"
    else:
        name_pattern = f"{run_name}_epoch{epoch_num}_{machine}_idea_{idea_number}"
    return name_pattern

def retrieve_training_logs(run_name, epoch_num, env_dir="env_nanogpt", entity="hashimoto-group", project="nanogpt-training", target_loss = 3.28):
    api = wandb.Api(timeout=300)
    name_pattern = get_run_name(run_name, epoch_num)
    print(f"name_pattern: {name_pattern}")
    runs_with_output_log = []

    # Fetch all runs in the project (you can also pass filters to reduce results)
    runs = api.runs(f"{entity}/{project}")
    logs_dir = f"training_logs_{run_name}/epoch{epoch_num}/"
    os.makedirs(logs_dir, exist_ok=True)

    # For ranking ideas by final eval_reward
    # Use a dict to avoid duplicates, always keep the latest run for each idea
    idea_final_rewards = {}

    # To ensure we keep the latest run, process runs in order of increasing created_at (oldest to newest)
    # so the latest run for each idea will overwrite previous ones
    # sorted_runs = sorted(
    #     [run for run in runs if fnmatch.fnmatch(run.name, name_pattern)],
    #     key=lambda r: getattr(r, "created_at", 0) or 0
    # )
    sorted_runs = _retry_wandb(
        lambda: sorted(
            [
                run
                for run in api.runs(f"{entity}/{project}")
                if fnmatch.fnmatch(run.name, name_pattern)
            ],
            key=lambda r: getattr(r, "created_at", 0) or 0,
        ),
        "list/sort runs",
    )

    for run in sorted_runs:
        print(f"Matched run: {run.name} (id={run.id})")
        idea_name = run.name.split("_idea_")[-1]
        idea_dir = f"idea_{idea_name}"
        os.makedirs(os.path.join(logs_dir, idea_dir), exist_ok=True)

        # download the full experiment logs if they exist
        # the log is only available after the run finished or failed
        output_log_path = os.path.join(logs_dir, idea_dir, "output.log")
        try:
            run.file("output.log").download(root=f"{logs_dir}/{idea_dir}", replace=True)
            print("  -> downloaded output.log")
            if idea_dir not in runs_with_output_log:
                runs_with_output_log.append(idea_dir)
        except Exception as e:
            print("  -> no output.log found:", e)
            continue

        # extract the metrics from the run
        if "grpo" in run_name.lower():
            ## for grpo, we will extract the metrics from the run history
            try:
                history = run.history()
                eval_rewards = extract_metrics(history["eval/mean_reward"], "eval_reward")
                train_rewards = extract_metrics(history["train/mean_reward"], "train_reward")
            except Exception as e:
                print(f"Error extracting metrics for {run.name}: {e}")
                eval_rewards = []
                train_rewards = []
        elif "nanogpt" in run_name.lower():
            ## for nanogpt, we will infer the metrics from the output.log
            # read the output.log
            with open(output_log_path, "r") as f:
                lines = f.readlines()
            eval_rewards, time_to_target = extract_metrics_nanogpt(lines, target_loss)

        # save the metrics to a json file
        with open(os.path.join(logs_dir, idea_dir, "metrics.json"), "w") as f:
            if "grpo" in run_name.lower():
                json.dump({"eval_rewards": eval_rewards, "train_rewards": train_rewards}, f)
            elif "nanogpt" in run_name.lower():
                json.dump({"eval_rewards": eval_rewards, "time_to_target": time_to_target}, f)

        # For ranking: get the final eval_reward if available
        if eval_rewards:
            if "grpo" in run_name.lower():
                final_eval_reward = max([r["eval_reward"] for r in eval_rewards])
            elif "nanogpt" in run_name.lower():
                final_eval_reward = min([r["val_loss"] for r in eval_rewards])
            # elif "nanogpt" in run_name.lower():
            #     final_eval_reward = time_to_target
        else:
            final_eval_reward = -999.0  # or None, but -inf sorts to bottom

        # Always overwrite with the latest run for this idea
        idea_final_rewards["idea_" + idea_name] = final_eval_reward


    # After processing all runs, rank the ideas by best eval_reward and save
    # note that this is only a snapshot of the best ideas at the time of retrieval
    # the best ideas may change over time as all jobs are finished
    if idea_final_rewards:
        # Sort descending by reward
        # Sort with mode-specific ordering
        if "grpo" in run_name.lower():
            ranked_ideas = sorted(
                idea_final_rewards.items(),
                key=lambda x: x[1],
                reverse=True
            )
        else:
            # For NanoGPT, push non-positive sentinel values (e.g., -999) to the end, then sort ascending
            ranked_ideas = sorted(
                idea_final_rewards.items(),
                key=lambda x: float("inf") if x[1] <= 0 else x[1]
            )
        ranked_ideas_dicts = [{idea: reward} for idea, reward in ranked_ideas]
        with open(os.path.join(logs_dir, "ranked_ideas.json"), "w") as f:
            json.dump(ranked_ideas_dicts, f, indent=4)

        # Compute and print how many ideas got an eval_reward > 0
        num_ideas_with_positive_eval = sum(1 for idea, reward in idea_final_rewards.items() if reward > 0)
        print(f"Number of ideas with best eval_reward > 0: {num_ideas_with_positive_eval}")
    else:
        num_ideas_with_positive_eval = 0
        ranked_ideas_dicts = []

    return len(runs_with_output_log), ranked_ideas_dicts

if __name__ == "__main__":
    run_name = "nanogpt_finaleval"
    epoch_num = 0
    num_ideas, ranked_ideas_dicts = retrieve_training_logs(run_name, epoch_num, env_dir="env/nanogpt", entity="hashimoto-group", project="nanogpt-training", target_loss=3.28)
    print(f"Number of ideas: {num_ideas}")
    print(f"Ranked ideas: {ranked_ideas_dicts}")
import datasets
from datasets import Dataset
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import retry
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.api import apiqa
import random 
from tqdm import tqdm
import json

random.seed(2025)

def sample_ideas(sample_n = 30):
    models = ["gpt5", "claude_4_5_sonnet", "claude_4_5_opus"]
    for env in ["pre", "post"]:
        all_ideas = []
        for model in models:
            dataset_name = f"search_es_{env}_{model}"
            print (f"Processing {dataset_name}")
            dataset = datasets.load_dataset(f"codasci/{dataset_name}", split="train")
            for epoch in range(10):
                filtered_rows = [
                    row for row in dataset
                    if row.get("epoch") == epoch and row.get("result") is not None
                ]
                print (f"Epoch {epoch}: {len(filtered_rows)} executed ideas")
                sampled_rows = random.sample(filtered_rows, min(len(filtered_rows), sample_n))
                all_ideas.extend(sampled_rows)

        dataset = Dataset.from_list(all_ideas)
        dataset.push_to_hub(f"codasci/nosearch_{env}_sampled")
        print (f"Uploaded {len(all_ideas)} ideas to HuggingFace")
        
    return 

def analyze_code_diff(env="post", top_n=5):
    import scipy.stats

    models = ["gpt5", "claude_4_5_sonnet", "claude_4_5_opus"]
    model_colors = {
        "gpt5": "tab:blue",
        "claude_4_5_sonnet": "tab:orange",
        "claude_4_5_opus": "tab:green"
    }
    all_ideas = []
    model_epoch_linelengths = {model: {epoch: [] for epoch in range(10)} for model in models}
    model_epoch_rewards = {model: {epoch: [] for epoch in range(10)} for model in models}

    # For collecting (lines, accuracy) per model for correlation and binning
    model_lines_acc = {model: [] for model in models}

    for model in models:
        dataset_name = f"search_es_{env}_{model}"
        print(f"Processing {dataset_name}")
        dataset = datasets.load_dataset(f"codasci/{dataset_name}", split="train")
        for epoch in range(10):
            filtered_rows = [
                row for row in dataset
                if row.get("epoch") == epoch
                and row.get("result") is not None
                and (
                    "accuracy" not in row["result"]
                    or row["result"].get("accuracy", 0) >= 0.1
                )
            ]
            print(f"Epoch {epoch}: {len(filtered_rows)} executed ideas")
            all_ideas.extend(filtered_rows)

            for row in filtered_rows:
                num_lines = row["code_diff"].count('\n') + 1 if "code_diff" in row and isinstance(row["code_diff"], str) else 0
                model_epoch_linelengths[model][epoch].append(num_lines)
                if "result" in row and row["result"] is not None:
                    acc = row["result"].get("accuracy", 0)
                    model_epoch_rewards[model][epoch].append(acc)
                    # Collect for bin plot/correlation if meaningful
                    model_lines_acc[model].append((num_lines, acc))

    print(f"Total {len(all_ideas)} ideas")

    # ---- Plotting the average code_diff line count per epoch for each model ----
    epochs = list(range(10))
    plt.figure(figsize=(8, 5))
    for model in models:
        avgs = []
        for epoch in epochs:
            lines = model_epoch_linelengths[model][epoch]
            avg_lines = np.mean(lines) if lines else 0
            avgs.append(avg_lines)
        plt.plot(epochs, avgs, label=model, color=model_colors[model], marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Average Lines of Code in code_diff')
    plt.title(f'Average code_diff size by model per epoch (env={env})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"code_diff_size_by_model_per_epoch_{env}.pdf", format='pdf', dpi=300, bbox_inches='tight')
    plt.close()

    # ---- Additional Plot: Average reward per epoch for each model ----
    plt.figure(figsize=(8, 5))
    for model in models:
        avgs = []
        for epoch in epochs:
            rewards = model_epoch_rewards[model][epoch]
            avg_reward = np.mean(rewards) if rewards else 0
            avgs.append(avg_reward)
        plt.plot(epochs, avgs, label=model, color=model_colors[model], marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Average Reward (accuracy)')
    plt.title(f'Average reward (accuracy) by model per epoch (env={env})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"avg_reward_by_model_per_epoch_{env}.pdf", format='pdf', dpi=300, bbox_inches='tight')
    plt.close()

    # ---- New Plot: Grouped bar plot of mean accuracy for lines-of-code bins ----

    n_bins = 5
    binned_avgs = {}
    bin_edges_per_model = {}

    # First, for each model get all num_lines, to compute global bin edges (per model)
    for model in models:
        lines = [item[0] for item in model_lines_acc[model]]
        if lines:
            min_lines, max_lines = min(lines), max(lines)
            if min_lines == max_lines:  # Trivial case: only one line length across all
                bin_edges = np.linspace(min_lines, max_lines+1, n_bins+1)
            else:
                bin_edges = np.linspace(min_lines, max_lines, n_bins+1)
        else:
            bin_edges = np.arange(0, n_bins+1)
        bin_edges_per_model[model] = bin_edges

    # For grouped bar plot, make bins for each model, store average accuracy per bin
    for model in models:
        lines_accs = model_lines_acc[model]
        bin_edges = bin_edges_per_model[model]
        bin_indices_to_accs = {i: [] for i in range(n_bins)}
        for num_lines, acc in lines_accs:
            # bin index: [bin_edges[i], bin_edges[i+1])
            # if num_lines==max, put into last bin
            bin_idx = np.digitize(num_lines, bin_edges, right=False) - 1
            if bin_idx < 0:
                bin_idx = 0
            elif bin_idx >= n_bins:
                bin_idx = n_bins-1
            bin_indices_to_accs[bin_idx].append(acc)
        bin_means = []
        for i in range(n_bins):
            accs = bin_indices_to_accs[i]
            mean_acc = np.mean(accs) if accs else 0
            bin_means.append(mean_acc)
        binned_avgs[model] = bin_means

    # Bar plot
    x = np.arange(n_bins)  # bin positions
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, model in enumerate(models):
        # Offset for each model's bars: -width, 0, +width
        ax.bar(x + (i-1)*width, binned_avgs[model], width=width, color=model_colors[model], label=model)
    # x ticks: show bin ranges as labels (for one model e.g. gpt5, since bin edges vary per model)
    reference_model = models[0]
    reference_edges = bin_edges_per_model[reference_model]
    bin_labels = []
    for i in range(n_bins):
        left = int(reference_edges[i])
        right = int(reference_edges[i+1]) - 1 if i != n_bins-1 else int(reference_edges[i+1])
        bin_labels.append(f"{left}-{right}")
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels)
    ax.set_xlabel('Lines of Code in code_diff (binned)')
    ax.set_ylabel('Average Accuracy')
    ax.set_title(f'Average accuracy per code_diff size bin, by model (env={env})')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, linestyle="--", axis='y')
    plt.tight_layout()
    plt.savefig(f"bar_lines_of_code_vs_accuracy_by_model_{env}.pdf", format='pdf', dpi=300, bbox_inches='tight')
    plt.close()

    # ---- Print correlations ----
    print("\n=== Correlation (accuracy vs lines of code diff) ===")
    for model in models:
        pairs = model_lines_acc[model]
        if pairs and len(pairs) > 2:
            lines, accs = zip(*pairs)
            corr = np.corrcoef(lines, accs)[0, 1]
            corr_spearman = scipy.stats.spearmanr(lines, accs).correlation
            print(f"{model:>20s}: Pearson r = {corr:.3f} | Spearman r = {corr_spearman:.3f} (N={len(lines)})")
        else:
            print(f"{model:>20s}: Not enough data for correlation.")

@retry.retry(tries=3, delay=2)
def better_idea(idea_1, idea_2, model_name, env = "grpo", temperature=0.1, max_tokens = 2):
    system_message = "You are a reviewer specialized in Large Language Models research."
    if "grpo" in env:
        prompt = "You are given two project ideas on the topic of improving the Group Relative Policy Optimization (GRPO) baseline for post-training LLMs for mathematical reasoning. Your task is to decide which one is more interesting: is the idea truly innovative and not obvious to someone who has read the background literature? For example, if an idea is simply tuning some hyperparameters of the GRPO algorithm, it is not very interesting. However, if an idea is proposing an important and novel algorithmic change to improve some specific aspect of the GRPO algorithm, it is more interesting. Be careful with ideas that sound complicated by stacking multiple known tricks together: you should check whether the idea contains interesting new insights or not. Given the two ideas, directly return a number 1 or 2 to indicate the more interesting idea and end the response."
    elif "nanogpt" in env:
        prompt = "You are given two project ideas on the topic of improving the standard nanoGPT implementation for pre-training a small Transformer language model. Your task is to decide which one is more interesting: is the idea truly innovative and not obvious to someone who has read the background literature? For example, if an idea is simply tuning some hyperparameters of model architecture or training loop, it is not very interesting. However, if an idea is proposing an important and novel algorithmic change to the Transformer architecture or the training loop, it is more interesting. Be careful with ideas that sound complicated by stacking multiple known tricks together: you should check whether the idea contains interesting new insights or not. Given the two ideas, directly return a number 1 or 2 to indicate the more interesting idea and end the response."
    else:
        raise ValueError(f"Invalid environment: {env}")
    
    prompt += "\n\nThe two ideas are:\n\n" 
    prompt += "Idea 1:\n" + idea_1 + "\n\n"
    prompt += "Idea 2:\n" + idea_2 + "\n\n"

    # print (prompt)
    thinking, response = apiqa(prompt, model_name, system_message, json_format=False, claude_thinking_mode=False, claude_thinking_budget=0, temperature=temperature, max_tokens=max_tokens, max_trial=1)
    return response

@retry.retry(tries=3, delay=2)
def classify_idea(idea, model_name, env = "grpo", temperature=0.1, max_tokens = 2):
    system_message = "You are a reviewer specialized in Large Language Models research."
    if "grpo" in env or "post" in env:
        prompt = "You are given a research idea on the topic of improving the Group Relative Policy Optimization (GRPO) baseline for post-training LLMs for mathematical reasoning. Your task is to classify the idea into one of the following categories: 1. Tuning hyper-parameters or changing various configurations in the baseline. 2. Proposing some new algorithmic changes to the GRPO algorithm. If the idea is combining multiple tricks together, it can count as 2 if and only if the idea contains at least one substantial algorithmic change beyond hyper-parameters or configurations. Directly return a number 1 or 2 to indicate the category and end the response."
    elif "nanogpt" in env or "pre" in env:
        prompt = "You are given a research idea on the topic of improving the standard nanoGPT implementation for pre-training a small Transformer language model. Your task is to classify the idea into one of the following categories: 1. Tuning hyper-parameters or changing various configurations in the baseline. 2. Proposing some new algorithmic changes to the model architecture or training loop. If the idea is combining multiple tricks together, it can count as 2 if and only if the idea contains at least one substantial algorithmic change beyond hyper-parameters or configurations. Directly return a number 1 or 2 to indicate the category and end the response."
    else:
        raise ValueError(f"Invalid environment: {env}")
    
    prompt += "\n\nThe idea is:\n\n" 
    prompt += idea + "\n\n"

    # print (prompt)
    thinking, response = apiqa(prompt, model_name, system_message, json_format=False, claude_thinking_mode=False, claude_thinking_budget=0, temperature=temperature, max_tokens=max_tokens, max_trial=1)
    return response

def classify_all_ideas(model_name = "claude-sonnet-4-5", env = "post", temperature=0.1, max_tokens = 2):
    models = ["gpt5", "claude_4_5_sonnet", "claude_4_5_opus"]
    model_colors = {
        "gpt5": "tab:blue",
        "claude_4_5_sonnet": "tab:orange",
        "claude_4_5_opus": "tab:green"
    }
    all_ideas = []
    for model in models:
        dataset_name = f"search_es_{env}_{model}"
        print(f"Processing {dataset_name}")
        dataset = datasets.load_dataset(f"codasci/{dataset_name}", split="train")
        for epoch in range(10):
            filtered_rows = [
                row for row in dataset
                if row.get("epoch") == epoch
                and row.get("result") is not None
                and (
                    "accuracy" not in row["result"]
                    or row["result"].get("accuracy", 0) >= 0.1
                )
            ]
            print(f"Epoch {epoch}: {len(filtered_rows)} executed ideas")
            all_ideas.extend(filtered_rows)
    
    all_dp = []
    for row in tqdm(all_ideas):
        idea = row["idea"]
        classification = classify_idea(idea, model_name, env, temperature, max_tokens)
        # print (idea)
        # print (classification)
        # print ("--------------------------------")
        all_dp.append({
            "ideator_model": row["ideator_model"],
            "executor_model": row["executor_model"],
            "env": row["env"],
            "epoch": row["epoch"],
            "idea": idea,
            "code_diff": row["code_diff"],
            "classification": classification,
            "result": row["result"]
        })
    
    cache_file = f"ranking_scores_grpo/classifications_{env}_{model_name}.json"
    if not os.path.exists(cache_file):
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(all_dp, f, indent=4)

    return all_dp


@retry.retry(tries=3, delay=2)
def tournament_ranking(idea_dataset = "codasci/nosearch_post_sampled", model_name = "claude-sonnet-4-5", env = "grpo", temperature=0.1, max_tokens = 2, max_round=10):
    ranking_score_dir = f"ranking_scores_{env}"
    dataset = datasets.load_dataset(idea_dataset, split="train")

    # dataset = dataset.select(range(min(10, len(dataset))))
    
    ## deduplicate the dataset 
    ideas = []
    rows = []
    for row in dataset:
        if row["idea"] not in ideas:
            ideas.append(row["idea"])
            rows.append(row)
    
    ## initialize the scores for each idea
    idea_to_idx = {idea: idx for idx, idea in enumerate(ideas)}
    idea_to_score = {idea: 1 for idea in ideas}

    print (f"Loaded {len(idea_to_score)} ideas")
    
    # Helper function to conduct a single round of the tournament
    def single_round(ideas, current_round=0):
        ## shuffle ideas in the first round
        if current_round == 0:
            random.shuffle(ideas)
        
        match_pairs = []
        # Sort ideas based on current scores
        sorted_ideas = sorted(ideas, key=lambda idea: idea_to_score[idea], reverse=True)

        for i in range(0, len(sorted_ideas), 2):
            if i + 1 < len(sorted_ideas):
                match_pairs.append((sorted_ideas[i], sorted_ideas[i+1]))
            else:
                # If there is an odd number of ideas, the last one automatically wins this round
                idea_to_score[sorted_ideas[i]] += 1

        for idea1, idea2 in tqdm(match_pairs):
            result = better_idea(idea1, idea2, model_name, env, temperature, max_tokens)
            if result.strip() == '1':
                idea_to_score[idea1] += 1
            else:
                idea_to_score[idea2] += 1
        
        return 
    
    # Conduct the tournament rounds until only one idea remains
    current_round = 0
    score_predictions = {}
    while current_round < max_round:
        print ("Current round: ", current_round + 1)
        single_round(ideas[:], current_round=current_round)
        current_round += 1

        score_predictions = []
        for idea in ideas:
            idx = idea_to_idx[idea]
            row = rows[idx]
            score_predictions.append({
                "id": idx,
                "ideator_model": row["ideator_model"],
                "executor_model": row["executor_model"],
                "env": row["env"],
                "epoch": row["epoch"],
                "idea": idea,
                "interestingness_score": idea_to_score[idea],
                "result": row["result"]
            })

        # Save all scores
        cache_file = os.path.join(ranking_score_dir, "round_{}.json".format(current_round))
        if not os.path.exists(ranking_score_dir):
            os.makedirs(ranking_score_dir)
        with open(cache_file, "w") as f:
            json.dump(score_predictions, f, indent=4)
    
    return 

def read_interestingness_scores(env="post", model_name="claude-sonnet-4-5"):
    if env == "post":
        metric = "accuracy"
        use_min = False
    else:
        metric = "loss"
        use_min = True

    score_file = f"ranking_scores_grpo/classifications_{env}_{model_name}.json"
    with open(score_file, "r") as f:
        scores = json.load(f)

    # Group by ideator_model
    model_stats = {}
    for item in scores:
        ideator = item.get("ideator_model", "unknown")
        type_val = str(item.get("classification", "unknown")).strip()
        try:
            acc = item["result"].get(metric, None)
        except Exception:
            acc = None

        if ideator not in model_stats:
            model_stats[ideator] = {
                "type_1": {"count": 0, "accs": []},
                "type_2": {"count": 0, "accs": []},
                "total": 0,
            }

        if type_val == "1":
            model_stats[ideator]["type_1"]["count"] += 1
            if acc is not None:
                model_stats[ideator]["type_1"]["accs"].append(acc)
        elif type_val == "2":
            model_stats[ideator]["type_2"]["count"] += 1
            if acc is not None:
                model_stats[ideator]["type_2"]["accs"].append(acc)
        model_stats[ideator]["total"] += 1

    # Calculate percentage, average accuracy/loss, and max/min as needed
    results = {}
    for model, stat in model_stats.items():
        total = stat["total"]
        pct_1 = (stat["type_1"]["count"] / total * 100) if total > 0 else 0
        pct_2 = (stat["type_2"]["count"] / total * 100) if total > 0 else 0
        avg_acc_1 = (
            sum(stat["type_1"]["accs"]) / len(stat["type_1"]["accs"])
            if stat["type_1"]["accs"]
            else None
        )
        avg_acc_2 = (
            sum(stat["type_2"]["accs"]) / len(stat["type_2"]["accs"])
            if stat["type_2"]["accs"]
            else None
        )
        if use_min:
            extreme_acc_1 = min(stat["type_1"]["accs"]) if stat["type_1"]["accs"] else None
            extreme_acc_2 = min(stat["type_2"]["accs"]) if stat["type_2"]["accs"] else None
        else:
            extreme_acc_1 = max(stat["type_1"]["accs"]) if stat["type_1"]["accs"] else None
            extreme_acc_2 = max(stat["type_2"]["accs"]) if stat["type_2"]["accs"] else None

        results[model] = {
            "pct_type_1": pct_1,
            "pct_type_2": pct_2,
            "avg_accuracy_type_1": avg_acc_1,
            "avg_accuracy_type_2": avg_acc_2,
            "extreme_accuracy_type_1": extreme_acc_1,
            "extreme_accuracy_type_2": extreme_acc_2,
            "count_1": stat["type_1"]["count"],
            "count_2": stat["type_2"]["count"],
            "total": total,
        }
        print ("Model: ", model)
        print ("Percentage of type 1: ", pct_1)
        print ("Percentage of type 2: ", pct_2)
        print ("Average accuracy of type 1: ", avg_acc_1)
        print ("Average accuracy of type 2: ", avg_acc_2)
        if use_min:
            print ("Min loss of type 1: ", extreme_acc_1)
            print ("Min loss of type 2: ", extreme_acc_2)
        else:
            print ("Max accuracy of type 1: ", extreme_acc_1)
            print ("Max accuracy of type 2: ", extreme_acc_2)
        print ("Count of type 1: ", stat["type_1"]["count"])
        print ("Count of type 2: ", stat["type_2"]["count"])
        print ("Total: ", total)
        print ("--------------------------------")
    
    return results

if __name__ == "__main__":
    # sample_ideas()
    
    idea_1 = "[Experiment] Implement cosine annealing learning rate schedule starting from the optimal 3e-5, decaying to 1e-5 over the full training period with reinforce_with_baseline, providing strong initial learning that gradually stabilizes.[Code Changes] Modify `train_loop` in `grpo.py` to compute `current_lr = 1e-5 + (3e-5 - 1e-5) * 0.5 * (1 + math.cos(math.pi * epoch / args.grpo_steps))` and update optimizer learning rate each epoch. Set `--loss_type reinforce_with_baseline` in `run_job.sh`."
    idea_2 = "[Experiment] Sequence Position Weighted Trust Region: Apply tighter sigmoid bounds to earlier tokens in the sequence (where errors compound) and looser bounds to later tokens. Weight: `position_weight = 1 - 0.3 * (position / seq_len)`, `effective_deviation = 0.25 + 0.2 * position_weight`. This accounts for the sequential nature of autoregressive generation."
    idea_3 = "[Experiment] Combine the two best hyperparameter settings by using reinforce_with_baseline loss with increased learning rate of 2e-5 and nucleus sampling with top_p=0.9, leveraging the synergy between better policy gradient estimation and more effective exploration."
    # print (better_idea(idea_1, idea_2, "claude-sonnet-4-5", env = "grpo"))

    # tournament_ranking(max_round=20)
    # analyze_code_diff(env = "post")

    # print (classify_idea(idea_3, "claude-sonnet-4-5", env = "grpo"))
    # classify_all_ideas(env = "post", model_name = "claude-sonnet-4-5")
    # classify_all_ideas(env = "pre", model_name = "claude-sonnet-4-5")
    read_interestingness_scores(env = "pre", model_name = "claude-sonnet-4-5")
import os
import json

def compute_idea_stats(idea_file = "ideas/all_ideas_0826_epoch1.json", repo_variants_dir = "repo_variants_epoch1_parallel", idea_stats_file = "idea_stats/epoch1.json"):
    os.makedirs(os.path.dirname(idea_stats_file), exist_ok=True)
    
    # Gather all directory names under repo_variants_dir that start with "idea_"
    idea_dirs = [
        d for d in os.listdir(repo_variants_dir)
        if os.path.isdir(os.path.join(repo_variants_dir, d)) and d.startswith("idea_")
    ]
    # Extract the numeric index from each idea dir
    present_idea_indices = set()
    for d in idea_dirs:
        try:
            idx = int(d.split("_")[1])
            present_idea_indices.add(idx)
        except (IndexError, ValueError):
            continue

    # Load the list of ideas from idea_file
    with open(idea_file, "r", encoding="utf-8") as f:
        ideas = json.load(f)

    # Find all missing idea indices
    all_idea_indices = set(range(len(ideas)))
    missing_idea_indices = sorted(list(all_idea_indices - present_idea_indices))
    success_count = len(present_idea_indices)
    total_ideas = len(ideas)
    success_percent = success_count / total_ideas * 100

    idea_stats = {
        "successful_ideas": list(present_idea_indices),
        "failed_ideas": missing_idea_indices,
        "success_count": success_count,
        "total_ideas": total_ideas,
        "success_percent": success_percent
    }

    with open(idea_stats_file, "w", encoding="utf-8") as f:
        json.dump(idea_stats, f, indent=4)

if __name__ == "__main__":
    compute_idea_stats(idea_file = "ideas/all_ideas_claude_epoch2.json", repo_variants_dir = "repo_variants_claude_epoch2", idea_stats_file = "idea_stats/epoch2_claude.json")
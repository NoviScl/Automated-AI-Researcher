import shutil
import os
import time
import json
import argparse
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.agent import * 
from agent.compute_idea_stats import compute_idea_stats
from agent.upload_repo_variants import zip_and_upload_repo_variants
from agent.retrieve_training_logs import retrieve_training_logs
from agent.evolutionary_search import update_database

def move_diffs_and_repo_variants(src_diffs, dst_diffs, src_repo, dst_repo):
    # Move diffs directory, overwriting if destination exists
    if os.path.exists(src_diffs):
        dst_diffs_path = os.path.join(dst_diffs, os.path.basename(src_diffs))
        if os.path.exists(dst_diffs_path):
            if os.path.isdir(dst_diffs_path):
                shutil.rmtree(dst_diffs_path)
            else:
                os.remove(dst_diffs_path)
        shutil.move(src_diffs, dst_diffs)
        print(f"Moved {src_diffs} to {dst_diffs}")
    else:
        print(f"Source diffs directory {src_diffs} does not exist.")

    # Move repo_variants directory, overwriting if destination exists
    if os.path.exists(src_repo):
        dst_repo_path = os.path.join(dst_repo, os.path.basename(src_repo))
        if os.path.exists(dst_repo_path):
            if os.path.isdir(dst_repo_path):
                shutil.rmtree(dst_repo_path)
            else:
                os.remove(dst_repo_path)
        shutil.move(src_repo, dst_repo)
        print(f"Moved {src_repo} to {dst_repo}")
    else:
        print(f"Source repo_variants directory {src_repo} does not exist.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num_ideas_per_epoch", type=int, default=80)
    parser.add_argument("--continue_from_epoch", type=int, default=0)
    parser.add_argument("--skip_log_retrieval_when_continue", action="store_true")
    parser.add_argument("--skip_idea_generation_when_continue", action="store_true")
    parser.add_argument("--run_name", type=str, default="nanogpt_claude_opus_bsz80")
    parser.add_argument("--env_dir", type=str, default="env/nanogpt")
    parser.add_argument("--entity", type=str, default="hashimoto-group")
    parser.add_argument("--project", type=str, default="nanogpt_ES_claude")
    parser.add_argument("--model_name", type=str, default="claude-opus-4-5")
    args = parser.parse_args()
        
    epochs = args.epochs
    num_ideas_per_epoch = args.num_ideas_per_epoch
    run_name = args.run_name
    if args.continue_from_epoch < 0:
        start_epoch = 0
    else:
        start_epoch = args.continue_from_epoch
        
    for epoch in range(start_epoch, epochs):
        if epoch >= args.continue_from_epoch and not args.skip_idea_generation_when_continue:
            # generate ideas 
            print ("Sampling ideas for epoch ", epoch)
            agent_call_idea(num_ideas = num_ideas_per_epoch, cache_file = f"ideas_{run_name}/ideas_epoch{epoch}.json", run_name = run_name, epoch_num = epoch, prev_ideas_file = f"ideas_{run_name}/ideas_epoch{epoch-1}.json", prev_training_logs = f"training_logs_{run_name}/epoch{epoch-1}/", top_k=100, sample_k=100, env_dir=args.env_dir, model_name=args.model_name)

            # generate the code diff for each experiment
            print ("Generating code diffs for epoch ", epoch)
            generate_code_diff_parallel(max_trials=10, diffs_dir=f"diffs_{run_name}_epoch{epoch}", repo_dir=f"repo_variants_{run_name}_epoch{epoch}", env_dir=args.env_dir, idea_file=f"ideas_{run_name}/ideas_epoch{epoch}.json", model_name=args.model_name)

            print ("Computing idea stats for epoch ", epoch)
            compute_idea_stats(idea_file = f"ideas_{run_name}/ideas_epoch{epoch}.json", repo_variants_dir = f"repo_variants_{run_name}_epoch{epoch}", idea_stats_file = f"idea_stats_{run_name}/epoch{epoch}.json")
            zip_and_upload_repo_variants(original_ideas = f"repo_variants_{run_name}_epoch{epoch}", folder_path = f"/juice5b/scr5b/nlp/aihinton/repo_variants/{run_name}/epoch{epoch}", run_name = run_name, epoch_num = epoch)

            print ("Moving diffs and repo_variants of this epoch to diffs_claude and repo_variants_claude")
            move_diffs_and_repo_variants(src_diffs=f"diffs_{run_name}_epoch{epoch}", dst_diffs="diffs_claude", src_repo=f"repo_variants_{run_name}_epoch{epoch}", dst_repo="repo_variants_claude")

            print ("Waiting before retrieving training logs")
            time.sleep(90 * 60)
        
        if epoch == args.continue_from_epoch and args.skip_log_retrieval_when_continue:
            print ("Skipping log retrieval for epoch ", epoch)
            continue

        print ("Retrieving training logs for epoch ", epoch)
        num_logs_retrieved = 0 
        idea_stats_path = f"idea_stats_{run_name}/epoch{epoch}.json"
        with open(idea_stats_path, "r") as f:
            idea_stats = json.load(f)
        num_ideas_submitted = idea_stats.get("success_count", num_ideas_per_epoch)
        
        last_num_logs_retrieved = 0
        while num_logs_retrieved <= int(num_ideas_submitted * 0.3):
            num_logs_retrieved, ranked_ideas_dicts = retrieve_training_logs(run_name = run_name, epoch_num = epoch, env_dir = args.env_dir, entity = args.entity, project = args.project)
            print (f"Number of logs retrieved: {num_logs_retrieved} out of {num_ideas_submitted} submitted ideas")
            ## terminate conditions: most runs are finished
            ## this means we might waste some runs that are still running but it's going to be faster
            if num_logs_retrieved > int(num_ideas_submitted * 0.3):
                break
            last_num_logs_retrieved = num_logs_retrieved
            print ("Waiting before retrieving training logs again")
            time.sleep(20 * 60)
    
    ## do a final round of update_dataset 
    update_database(run_name = run_name, epoch_num = epochs - 1)

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.agent import generate_code_diff_parallel
from agent.upload_repo_variants import zip_and_upload_repo_variants
import shutil
import json
from concurrent.futures import ThreadPoolExecutor
from glob import glob

def ideas_to_zips(idea_lst, diffs_dir, repo_dir, env_dir, zip_dir, upload_to_hf = False, n_ideas_cap = 400, run_name = "test", epoch_num = 0, model_name = "gpt-5", max_trials = 10):
    """
    - Takes in a list of natural language ideas, generates code diffs, patches the env, zips the repo variants, returns the zip directory
    - For each ideas in idea_lst, the new code with diff applied with be saved as zip files in zip_dir. The naming convention of the zip files is zip_dir/idea_0.zip, zip_dir/idea_1.zip, etc. Here zip_dir/idea_0.zip is the zip file of the first idea in idea_lst. If an index is not found in zip_dir, it means the code diff generation failed for that idea.
    """
    generate_code_diff_parallel(max_trials=max_trials, diffs_dir=diffs_dir, repo_dir=repo_dir, env_dir=env_dir, idea_lst=idea_lst, model_name=model_name, total_workers=20)
    zip_and_upload_repo_variants(original_ideas = repo_dir, folder_path = zip_dir, upload_to_hf = False, n_ideas_cap = n_ideas_cap)
    if os.path.exists(diffs_dir):
        shutil.rmtree(diffs_dir)
    if os.path.exists(repo_dir):
        shutil.rmtree(repo_dir)
    return zip_dir

def ideas_to_zips_parallel(idea_lst, diffs_dir, repo_dir, env_dir, zip_dir, upload_to_hf = False, n_ideas_cap = 400, run_name = "test", epoch_num = 0, model_name = "gpt-5", max_trials = 10):
    """
    Optimized implementation for parallel execution of ideas_to_zips with the exact same IO.
    """
    batch_size = 5
    max_trial_per_batch = max_trials // batch_size
    map_ideas_to_zips = lambda batch_idx: ideas_to_zips(
        idea_lst=idea_lst,
        diffs_dir=f"{diffs_dir}_{batch_idx}",
        repo_dir=f"{repo_dir}_{batch_idx}",
        env_dir=env_dir,
        zip_dir=f"{zip_dir}_{batch_idx}",
        upload_to_hf=upload_to_hf,
        n_ideas_cap=n_ideas_cap,
        run_name=run_name,
        epoch_num=epoch_num,
        model_name=model_name,
        max_trials=max_trial_per_batch)

    # parallel execution of ideas_to_zips
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        list(executor.map(map_ideas_to_zips, range(batch_size)))
    source_paths = {}
    all_zips = glob(f"{zip_dir}**/*.zip")
    for zip_path in all_zips:
        zip_name = zip_path.split("/")[-1]
        source_paths[zip_name] = zip_path

    # aggregating resulting zips to the final zip directory
    if os.path.exists(zip_dir):
        shutil.rmtree(zip_dir)
    os.makedirs(zip_dir)
    for zip_name, zip_path in source_paths.items():
        shutil.move(zip_path, os.path.join(zip_dir, zip_name))
    for batch_idx in range(batch_size):
        shutil.rmtree(f"{zip_dir}_{batch_idx}")
    return zip_dir

if __name__ == "__main__":
    # idea_lst = [
    #     "[Experiment] Decrease the learning rate from 1e-5 to 5e-6 to enable more stable but slower learning dynamics.\n\n[Code Changes] Change the `learning_rate` parameter in `run_job.sh` from 1e-5 to 5e-6.",
    #     "[Experiment] Change the group size from 8 to 4 to reduce the computational cost while potentially maintaining learning effectiveness.\n\n[Code Changes] Change the `group_size` parameter in `run_job.sh` from 8 to 4."
    # ]

    with open("/nlp/scr/clsi/research-agent/ideas_nanogpt_rerun/ideas_epoch0.json", "r") as f:
        idea_lst = json.load(f)
    zip_dir = ideas_to_zips(idea_lst, diffs_dir = "diffs_api_testing", repo_dir = "repo_variants_api_testing", env_dir = "env/nanogpt", zip_dir = "zips_api_testing", upload_to_hf = False, n_ideas_cap = 400, run_name = "test", epoch_num = 0, model_name = "gpt-5", max_trials = 10)
    print (zip_dir)


from huggingface_hub import HfApi, create_repo
from huggingface_hub import snapshot_download
import os
import shutil
from huggingface_hub import HfApi
import zipfile

api = HfApi()

# try:
#     repo_id = create_repo("sqres/repo_variants").repo_id
# except Exception as e:
#     repo_id = "sqres/repo_variants"

try:
    repo_id = create_repo("CLS/repo_variants").repo_id
except Exception as e:
    repo_id = "CLS/repo_variants"

# Sort by idea id (assuming format 'idea_<number>')
def idea_id_key(name):
    try:
        return int(name.split("_")[1])
    except (IndexError, ValueError):
        return float('inf')

def zip_and_upload_repo_variants(original_ideas, folder_path, run_name = None, epoch_num = None, upload_path = "/juice5b/scr5b/nlp/aihinton/repo_variants/", upload_to_hf = True, n_ideas_cap = 400):
    os.makedirs(folder_path, exist_ok=True)

    # List all directories in original_ideas, filter for those starting with 'idea_'
    idea_dirs = [
        d for d in os.listdir(original_ideas)
        if os.path.isdir(os.path.join(original_ideas, d)) and d.startswith("idea_")
    ]

    idea_dirs_sorted = sorted(idea_dirs, key=idea_id_key)

    for idea_dir in idea_dirs_sorted[:n_ideas_cap]:
        idea_path = os.path.join(original_ideas, idea_dir)
        zip_filename = os.path.join(folder_path, f"{idea_dir}.zip")
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(idea_path):
                # Skip wandb and __pycache__ directories
                dirs[:] = [d for d in dirs if d not in ['wandb', '__pycache__']]
                
                for file in files:
                    file_path = os.path.join(root, file)
                    # Write file to zip, preserving relative path inside the idea_dir
                    arcname = os.path.relpath(file_path, idea_path)
                    try:
                        zipf.write(file_path, arcname)
                    except FileNotFoundError:
                        print(f"Warning: File {file_path} not found, skipping...")
                        continue

    if upload_to_hf:
        # Upload the (now zipped) folder to HuggingFace
        api.upload_folder(
            repo_id=repo_id,
            folder_path=os.path.join(upload_path, run_name, f"epoch{epoch_num}"),
            path_in_repo=f"{run_name}/epoch{epoch_num}",
            repo_type="model",
            delete_patterns=os.path.join(upload_path, run_name, f"epoch{epoch_num}", "*")
        )


def delete_repo_variants(path_in_repo, repo_id = "sqres/repo_variants", local_path = "/juice5b/scr5b/nlp/aihinton/repo_variants", delete_local = False):
    api.delete_folder(
        repo_id=repo_id,
        path_in_repo=path_in_repo
    )
    if delete_local:
        shutil.rmtree(os.path.join(local_path, path_in_repo))

if __name__ == "__main__":
    # original_ideas = "repo_variants_claude_epoch2"
    # folder_path = "/juice5b/scr5b/nlp/aihinton/repo_variants/full_pipeline/epoch2"
    # # upload_path = "/juice5b/scr5b/nlp/aihinton/repo_variants/"

    # upload_repo_variants(original_ideas, folder_path, run_name = "full_pipeline", epoch_num = 1)

    # zip_and_upload_repo_variants(original_ideas = "repo_variants_nanogpt_rerun_epoch0", folder_path = "/juice5b/scr5b/nlp/aihinton/repo_variants/nanogpt_rerun/epoch0", run_name = "nanogpt_rerun", epoch_num = 0)
    
    delete_repo_variants(path_in_repo = "nanogpt_rerun2/epoch6", repo_id = "CLS/repo_variants", local_path = "/juice5b/scr5b/nlp/aihinton/repo_variants", delete_local = True)
    
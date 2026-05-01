# python -m verl.model_merger merge \
#     --backend fsdp \
#     --local_dir checkpoints/rft/TPO_alfworld_RFT/rft-Qwen3-4B/global_step_10 \
#     --target_dir checkpoints/Qwen3-4B-alfword-e5


from huggingface_hub import create_repo, upload_folder

repo_name = "xxx/xxx"
repo_id = repo_name  
create_repo(repo_id, exist_ok=True)

upload_folder(
    repo_id=repo_id,
    folder_path="xxxx/xxxx",  
    path_in_repo=".",                
)

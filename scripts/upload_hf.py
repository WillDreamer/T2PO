# python -m verl.model_merger merge \
#     --backend fsdp \
#     --local_dir /workspace/AgentRL/sft_ckpt/global_step_56 \
#     --target_dir /workspace/Qwen-4B-webshop


from huggingface_hub import create_repo, upload_folder

repo_name = "xxxxx/Qwen3-4B-webshop-gpt"

repo_id = repo_name  

# 创建仓库（自动跳过已存在）
create_repo(repo_id, exist_ok=True)

# 上传整个模型文件夹
upload_folder(
    repo_id=repo_id,
    folder_path="/workspace/Qwen3-4B-webshop",  
    path_in_repo=".",               
)

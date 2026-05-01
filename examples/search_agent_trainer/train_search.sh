set -x
ENGINE=${1:-vllm}

# ======================== GPU auto selection ========================
GPU_LIST=(0 1 2 3 4 5 6 7)  # <<<------  which GPUs to use, directly fill here
# Automatically concatenate CUDA_VISIBLE_DEVICES according to GPU_LIST
CUDA_VISIBLE_DEVICES=$(IFS=, ; echo "${GPU_LIST[*]}")
export CUDA_VISIBLE_DEVICES
echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
# Automatically detect the number of n_gpus_per_node
NUM_GPUS=${#GPU_LIST[@]}
echo "Detected ${NUM_GPUS} GPUs for this run"

train_data_size=256
val_data_size=512
group_size=5
ROLLOUT_MODE="sync"

# GiGPO config
mode="mean_std_norm" # "mean_norm" or "mean_std_norm"
enable_similarity=True # enable similarity-based GiGPO
similarity_thresh=0.9 # similarity threshold for GiGPO

MODEL=Qwen/Qwen3-4B
MODEL_SHORT="${MODEL##*/}"
project_name="search_agent"
estimator="gigpo"
experiment_name="${MODEL_SHORT}_${estimator}"

mkdir -p checkpoints/${project_name}/${experiment_name}

PORT=$(( ( RANDOM % 10000 +1000) ))
ray start --head --port $PORT

WANDB_API_KEY="xxxx" # Modify your wandb key
# ============================ Preparation ============================
# Login to WandB (if API key is provided)
if [ "$WANDB_API_KEY" != "" ]; then
    wandb login --relogin $WANDB_API_KEY
    mkdir -p wandb/${project_name}/${experiment_name}
    SAVE_PATH=wandb/${project_name}/${experiment_name}
    export WANDB_DIR=${SAVE_PATH}
fi

TRAIN_DATA="dataset/data/searchR1_processed_direct/train.parquet"
VAL_DATA="dataset/data/searchR1_processed_direct/test.parquet"

python3 -m recipe.search_agent.main_search_agent \
    algorithm.adv_estimator=gigpo \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=4096 \
    data.max_response_length=500 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.mode=$ROLLOUT_MODE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    use_invalid_action_penalty=True \
    invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode=$mode \
    algorithm.gigpo.enable_similarity=$enable_similarity \
    algorithm.gigpo.similarity_thresh=$similarity_thresh \
    algorithm.TDS.enable=True \
    algorithm.TDS.max_try=2 \
    env.env_name=search \
    env.seed=0 \
    env.max_steps=4 \
    env.rollout.n=$group_size \
    env.history_length=4 \
    env.search.search_url='http://127.0.0.1:8000/retrieve' \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.total_epochs=1 \
    trainer.val_before_train=False $@

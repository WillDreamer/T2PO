set -x
export MKL_THREADING_LAYER=GNU
export NCCL_P2P_DISABLE=1
unset MKL_SERVICE_FORCE_INTEL
ENGINE=${1:-vllm}

ulimit -n 1048576

# ======================== GPU auto selection ========================
GPU_LIST=(0 1 2 3 4 5 6 7)  # <<<------  which GPUs to use, directly fill here
CUDA_VISIBLE_DEVICES=$(IFS=, ; echo "${GPU_LIST[*]}")
export CUDA_VISIBLE_DEVICES
echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
NUM_GPUS=${#GPU_LIST[@]}
echo "Detected ${NUM_GPUS} GPUs for this run"

ROLLOUT_MODE="sync"
if [ "$ROLLOUT_MODE" = "async" ]; then
    export VLLM_USE_V1=1
else
    export VLLM_USE_V1=0 
    PORT=$(( ( RANDOM % 10000 +1000) ))
    ray status >/dev/null 2>&1 || ray start --head --port $PORT
fi

export RAY_TMPDIR="ray_$(date +%s)"
rm -rf "$RAY_TMPDIR"
mkdir -p "$RAY_TMPDIR"

num_cpus_per_env_worker=0.1 
num_think_tokens=450 

train_data_size=16
val_data_size=128
group_size=8
mode="mean_norm" # "mean_norm" or "mean_std_norm"

MODEL=willhx/Qwen3-4B-rft-webshop-5
MODEL_SHORT="${MODEL##*/}"
project_name="T2PO_Webshop"

WANDB_API_KEY="xxxxx" # Modify your wandb key
# ============================ Preparation ============================
# Login to WandB (if API key is provided)
if [ "$WANDB_API_KEY" != "" ]; then
    wandb login --relogin $WANDB_API_KEY
    mkdir -p wandb/${project_name}/${experiment_name}
    SAVE_PATH=wandb/${project_name}/${experiment_name}
    export WANDB_DIR=${SAVE_PATH}
fi

python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $((val_data_size * 2)) # evaluate 2 × val_data_size tasks during each iteration

for seed in 0
do
    experiment_name="Seed${seed}_${MODEL_SHORT}_len_${num_think_tokens}"
    mkdir -p checkpoints/${project_name}/${experiment_name}

    python3 -m recipe.shop_agent.main_shop_agent \
        algorithm.adv_estimator=gigpo \
        data.train_files=$HOME/data/text/train.parquet \
        data.val_files=$HOME/data/text/test.parquet \
        data.train_batch_size=$train_data_size \
        data.val_batch_size=$val_data_size \
        data.max_prompt_length=4096 \
        data.max_response_length=300 \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        data.return_raw_chat=True \
        actor_rollout_ref.model.path=$MODEL \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=128 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.name=$ENGINE \
        actor_rollout_ref.rollout.mode=$ROLLOUT_MODE \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
        actor_rollout_ref.rollout.enable_chunked_prefill=False \
        actor_rollout_ref.rollout.enforce_eager=False \
        actor_rollout_ref.rollout.free_cache_engine=False \
        actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
        actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
        actor_rollout_ref.rollout.val_kwargs.top_k=20 \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        +actor_rollout_ref.rollout.num_think_tokens=$num_think_tokens \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        use_invalid_action_penalty=True \
        invalid_action_penalty_coef=0.1 \
        algorithm.use_kl_in_reward=False \
        algorithm.gamma=0.95 \
        algorithm.gigpo.step_advantage_w=1.0 \
        algorithm.gigpo.mode=$mode \
        algorithm.TDS.enable=True \
        algorithm.TDS.max_try=2 \
        env.env_name=Webshop \
        env.seed=$seed \
        env.max_steps=15 \
        env.rollout.n=$group_size \
        env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
        trainer.rollout_data_dir=outputs/${experiment_name} \
        trainer.critic_warmup=0 \
        trainer.logger=['console','wandb'] \
        trainer.project_name=$project_name \
        trainer.experiment_name=$experiment_name \
        trainer.n_gpus_per_node=$NUM_GPUS \
        trainer.nnodes=1 \
        trainer.save_freq=10 \
        trainer.test_freq=10 \
        trainer.total_epochs=200 \
        trainer.val_before_train=False $@
done
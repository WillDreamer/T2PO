#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'


log() { printf "\n\033[1;32m[+] %s\033[0m\n" "$*"; }
warn() { printf "\033[1;33m[!] %s\033[0m\n" "$*"; }
die() { printf "\033[1;31m[x] %s\033[0m\n" "$*"; exit 1; }
trap 'die "Error is in Line $LINENO (exit=$?)。"' ERR

as_root() {
  if [[ ${EUID:-$(id -u)} -ne 0 ]]; then
    sudo -H bash -lc "$*"
  else
    bash -lc "$*"
  fi
}


log "Run setup game environment "
CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null || true)}"
if [[ -z "${CONDA_BASE}" ]]; then
  for p in "$HOME/miniconda3" "$HOME/anaconda3" "/opt/anaconda3"; do
    [[ -d "$p" ]] && CONDA_BASE="$p" && break
  done
fi
if [[ -z "${CONDA_BASE}" ]]; then
  echo "Conda not found, please confirm it is installed (miniconda or anaconda)." >&2
  exit 1
fi

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda create -n agentrl_search python==3.12 -y
conda activate agentrl_search
python3 -m pip install uv
python3 -m uv pip install -e ".[vllm]"
pip install --no-deps -e .
python3 -m uv pip install flash-attn==2.8.3 --no-build-isolation --no-deps
python3 -m uv pip install -r ./requirements.txt
cd ./agent_system/environments/env_package/search/third_party
pip install -e .
pip install gym==0.26.2
cd ../../../../../

log "Install search environment"
python3 -m uv pip install bitsandbytes deepspeed==0.16.4 isort jsonlines loralib optimum tensorboard torchmetrics transformers_stream_generator
python3 -m uv pip install llama_index bs4 pymilvus infinity_client omegaconf hydra-core easydict mcp==1.9.3
python3 -m uv pip install "faiss-gpu-cu12==1.9.0"
python3 -m uv pip install nvidia-cublas-cu12==12.4.5.8 


save_path=dataset/search
python examples/search_agent_trainer/search_data_download.py --local_dir $save_path
cat $save_path/part_* > $save_path/e5_Flat.index
gzip -d $save_path/wiki-18.jsonl.gz
python examples/data_preprocess/preprocess_search_r1_dataset.py


log "hf auth whoami"
if command -v hf >/dev/null 2>&1; then
  hf auth whoami || die "whoami failed"
else
  huggingface-cli whoami || die "whoami failed"
fi

log "Start retriever"
save_path=dataset/search
index_file=$save_path/e5_Flat.index
corpus_file=$save_path/wiki-18.jsonl
retriever_name=e5
retriever_path=intfloat/e5-base-v2

nohup python examples/search_agent_trainer/retriever/retrieval_server.py \
  --index_path $index_file \
  --corpus_path $corpus_file \
  --topk 3 \
  --retriever_name $retriever_name \
  --retriever_model $retriever_path \
  --faiss_gpu \
  --port 8000 > retriever.log 2>&1 &

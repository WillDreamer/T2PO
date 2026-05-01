set -e

# GREEN='\033[0;32m'
# BLUE='\033[0;34m'
# NC='\033[0m' # No Color

# ---- 小工具 ----
log() { printf "\n\033[1;32m[+] %s\033[0m\n" "$*"; }
warn() { printf "\033[1;33m[!] %s\033[0m\n" "$*"; }
die() { printf "\033[1;31m[x] %s\033[0m\n" "$*"; exit 1; }
trap 'die "脚本在第 $LINENO 行出错（exit=$?）。"' ERR

eval "$(conda shell.bash hook)"

conda create -n agentrl_embody python==3.12 -y
conda activate agentrl_embody
python3 -m pip install uv

python3 -m uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python3 -m uv pip install flash-attn==2.7.4.post1 --no-build-isolation

python3 -m uv pip install vllm==0.8.5
cd ../../../../../
python3 -m uv pip install -e .

python3 -m uv pip install gymnasium==0.29.1
python3 -m uv pip install stable-baselines3==2.6.0
python3 -m uv pip install alfworld

alfworld-download -f
alfworld-download --extra

conda install -c conda-forge "libstdcxx-ng>=12" "libgcc-ng>=12" -y
python3 -m uv pip install multiprocess==0.70.11

echo -e "${GREEN}Installation completed successfully!${NC}"
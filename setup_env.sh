#!/bin/bash

# 初始化 conda（source conda 的 profile）
eval "$(conda shell.bash hook)"

conda create -n verl python==3.12 -y
conda init
conda activate verl
python3 -m pip install uv

# python3 -m uv pip install -e ".[sglang]"
python3 -m uv pip install -e ".[vllm]"
pip install --no-deps -e .
python3 -m uv pip install flash-attn==2.8.3 --no-build-isolation --no-deps
python3 -m uv pip install -r ./requirements.txt

import json

json_file = "alfworld_high_score_multiturn_texts_seed0.json"
with open(json_file, "r") as f:
    data = json.load(f)

from datasets import Dataset
import pandas as pd

records = []
for sample_idx in range(len(data)):
    messages = []
    # System prompt
    messages.append({
        "content": "You are an expert agent operating in the ALFRED Embodied Environment.",
        "role": "system"
    })
    turn_idxs = range(len(data[sample_idx]['turns']))
    for turn_idx in turn_idxs:
        # user输入
        messages.append(data[sample_idx]['turns'][turn_idx]['inputs'][0])
        # assistant输出
        messages.append({
            "content": data[sample_idx]['turns'][turn_idx]['outputs'],
            "role": "assistant",
        })
    records.append({
        "messages": messages
    })

# 转为pandas DataFrame, 再转为datasets.Dataset
df = pd.DataFrame(records)
ds = Dataset.from_pandas(df)
ds.to_parquet("train.parquet")
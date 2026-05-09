
from vllm import LLM, SamplingParams
from verl.single_controller.ray.base import RayWorkerGroup
from transformers import AutoTokenizer, AutoModelForCausalLM
from verl import DataProto
import hydra
import os
import time
from typing import List, Dict
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from torchdata.stateful_dataloader import StatefulDataLoader
from recipe.shop_agent.llm_agent.agent_proxy import VllmWrapperWg
from agent_system.multi_turn_rollout.rollout_loop_eval import TrajectoryCollector
from agent_system.environments import make_envs
from verl.utils import hf_processor
from recipe.shop_agent.main_shop_agent import create_rl_dataset, create_rl_sampler
        

@hydra.main(version_base=None, config_path="config", config_name="base_eval")
def main(config):
    # detect config name from python -m webshop.llm_agent.agent_proxy --config_name frozen_lake
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    # Fix tokenizer parallelism warning in multiprocessing environment
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    print(config.data)
    tokenizer = AutoTokenizer.from_pretrained(config.actor_rollout_ref.model.path)
    actor_wg = VllmWrapperWg(config, tokenizer)
    
    envs, val_envs = make_envs(config)
    processor = hf_processor(config.actor_rollout_ref.model.path, trust_remote_code=True, use_fast=True)

    traj_collector = TrajectoryCollector(config=config, tokenizer=tokenizer, processor=processor)
    
    train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
    val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
    
    train_sampler = create_rl_sampler(config.data, train_dataset)
    from verl.utils.dataset.rl_dataset import collate_fn

    # val_dataloader = StatefulDataLoader(
    #         dataset=train_dataset,
    #         batch_size=256,
    #         num_workers=8,
    #         drop_last=True,
    #         collate_fn=collate_fn,
    #         sampler=train_sampler)
    
    val_dataloader = StatefulDataLoader(
            dataset=val_dataset,
            batch_size=256,
            num_workers=8,
            shuffle=True,
            drop_last=False,
            collate_fn=collate_fn,
        )

    for test_data in val_dataloader:
        test_batch = DataProto.from_single_dict(test_data)

        # repeat test batch
        test_batch = test_batch.repeat(repeat_times=config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True)

        # Store original inputs
        input_ids = test_batch.batch["input_ids"]
        # TODO: Can we keep special tokens except for padding tokens?
        input_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]

        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
        if "multi_modal_data" in test_batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("multi_modal_data")
        if "raw_prompt" in test_batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("raw_prompt")
        if "tools_kwargs" in test_batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("tools_kwargs")
        if "env_kwargs" in test_batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("env_kwargs")
        test_gen_batch = test_batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
        )

        test_gen_batch.meta_info = {
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "recompute_log_prob": False,
            "do_sample": config.actor_rollout_ref.rollout.val_kwargs.do_sample,
            "validate": True,
        }
        print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

    
        start_time = time.time()
        result = traj_collector.multi_turn_loop_for_eval(
                        gen_batch=test_gen_batch,
                        actor_rollout_wg=actor_wg,
                        envs=envs
                        )

        # 提取success['webshop_task_score']大于0.8对应的response_texts，并保存
        task_scores = result['success'].get('webshop_task_score (not success_rate)', None)
        response_texts = result['step_io_history']

        if task_scores is not None and response_texts is not None:
            import json
            output_path = f'high_score_multiturn_texts_seed{config.env.seed}.json'
            # response_texts是长度为15的列表，每个元素包含dict_keys(['step', 'inputs', 'outputs'])
            all_turns = []
            for sample_idx, score in enumerate(task_scores):
                if score > 0.9:
                    turns = []
                    # response_texts为每轮list，每轮包含该batch（n）的信息
                    for turn in response_texts:
                        # turn['inputs']和turn['outputs']都是n个sample
                        if turn.get('inputs', None)[sample_idx] is None:
                            continue
                        turn_result = {
                            "step": turn.get('step', None),
                            "inputs": list(turn.get('inputs', []))[sample_idx],
                            "outputs": list(turn.get('outputs', []))[sample_idx]
                        }
                        turns.append(turn_result)
                    all_turns.append({
                        "sample_idx": sample_idx,
                        "task_score": score,
                        "turns": turns
                    })

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(all_turns, f, ensure_ascii=False, indent=2)
            print(f"Saved all {len(all_turns)} turns of the multi-turn dialogue to {output_path}")
        
        end_time = time.time()
        print(f'rollout time: {end_time - start_time} seconds')

if __name__ == "__main__":
    main()

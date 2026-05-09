from vllm import LLM, SamplingParams
from verl.single_controller.ray.base import RayWorkerGroup
from transformers import AutoTokenizer, AutoModelForCausalLM
from verl import DataProto
import hydra
import os
from typing import List, Dict
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from recipe.shop_agent.llm_agent.base_llm import ConcurrentLLM
import numpy as np
# import time

class VllmWrapperWg: # Thi is a developing class for eval and test
	def __init__(self, config, tokenizer):
		self.config = config
		self.tokenizer = tokenizer
		model_name = config.actor_rollout_ref.model.path
		ro_config = config.actor_rollout_ref.rollout
		self.llm = LLM(
			model_name,
            enable_sleep_mode=True,
            tensor_parallel_size=ro_config.tensor_model_parallel_size,
            dtype=ro_config.dtype,
            enforce_eager=ro_config.enforce_eager,
            gpu_memory_utilization=ro_config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            # disable_mm_preprocessor_cache=True,
            skip_tokenizer_init=False,
            max_model_len=ro_config.max_model_len,
            disable_log_stats=ro_config.disable_log_stats,
            max_num_batched_tokens=ro_config.max_num_batched_tokens,
            enable_chunked_prefill=ro_config.enable_chunked_prefill,
            enable_prefix_caching=True,
			trust_remote_code=True,
		)
		print("LLM initialized")
		self.sampling_params = SamplingParams(
			max_tokens=ro_config.response_length,
			temperature=ro_config.val_kwargs.temperature,
			top_p=ro_config.val_kwargs.top_p,
			top_k=ro_config.val_kwargs.top_k,
			# min_p=0.1,
		)

	def generate_sequences(self, lm_inputs: DataProto):
		"""
		Convert the input ids to text, and then generate the sequences. Finally create a dataproto. 
		This aligns with the verl Worker Group interface.
		"""
		# NOTE: free_cache_engine is not used in the vllm wrapper. Only used in the verl vllm.
		# cache_action = lm_inputs.meta_info.get('cache_action', None)

		input_ids = lm_inputs.batch['input_ids']
		input_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=False)
		input_texts = [i.replace("<|endoftext|>", "") for i in input_texts]

		outputs = self.llm.generate(input_texts, sampling_params=self.sampling_params)
		texts = [output.outputs[0].text for output in outputs] 
		lm_outputs = DataProto()
		lm_outputs.non_tensor_batch = {
			'response_texts': texts,
		} # this is a bit hard-coded to bypass the __init__ check in DataProto
		lm_outputs.meta_info = lm_inputs.meta_info

		return lm_outputs

	
class ApiCallingWrapperWg:
    """Wrapper class for API-based LLM calls that fits into the VERL framework"""
    
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        # model_info adjust
        model_info = config.model_info[config.model_config.model_name]
        self.llm_kwargs = model_info.generation_kwargs
        
        # concurrent LLM需要改
        self.llm = ConcurrentLLM(
			provider=model_info.provider_name,
            model_name=model_info.model_name,
            max_concurrency=config.model_config.max_concurrency
        )
        
        print(f'API-based LLM ({model_info.provider_name} - {model_info.model_name}) initialized')


    def generate_sequences(self, lm_inputs: DataProto) -> DataProto:
        """
        Convert the input ids to text, make API calls to generate responses, 
        and create a DataProto with the results.
        """

        messages_list = lm_inputs.non_tensor_batch.get('messages_list', None)
        if messages_list is not None:
            messages_list = messages_list.tolist()
        else:
            # 回退：直接用 input_ids 构建单轮聊天消息
            input_ids = lm_inputs.batch.get('input_ids', None)
            if input_ids is None:
                raise KeyError("messages_list missing and input_ids unavailable to build fallback prompts")
            if hasattr(input_ids, "detach"):
                input_ids = input_ids.detach().cpu()
            fallback_messages = []
            for ids in input_ids:
                text = self.tokenizer.decode(ids.tolist(), skip_special_tokens=True)
                fallback_messages.append([{"role": "user", "content": text}])
            messages_list = fallback_messages
            print("[WARN] messages_list not provided; constructed fallback prompts from input_ids")

        results, failed_messages = self.llm.run_batch(
            messages_list=messages_list,
            **self.llm_kwargs
        )
        assert not failed_messages, f"Failed to generate responses for the following messages: {failed_messages}"

        texts = [result["response"] for result in results]
        env_ids = lm_inputs.non_tensor_batch.get('env_ids')
        if env_ids is None:
            env_ids = np.arange(len(texts), dtype=object)
        group_ids = lm_inputs.non_tensor_batch.get('group_ids')
        if group_ids is None:
            group_ids = np.zeros(len(texts), dtype=object)
        print(f'[DEBUG] texts: {texts}')
        lm_outputs = DataProto()
        lm_outputs.non_tensor_batch = {
			'response_texts': texts,
			'env_ids': env_ids,
			'group_ids': group_ids
		} # this is a bit hard-coded to bypass the __init__ check in DataProto
        lm_outputs.meta_info = lm_inputs.meta_info
        
        return lm_outputs

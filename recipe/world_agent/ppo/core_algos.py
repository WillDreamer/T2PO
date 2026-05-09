# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""


import numpy as np
import torch
import re
import verl.utils.torch_functional as verl_F

from collections import defaultdict, Counter
from verl import DataProto
import uuid
from omegaconf import DictConfig
from difflib import SequenceMatcher
from typing import Any, Callable, Optional
from typing import Sequence, List, Dict, Any
from verl.trainer.config import AlgoConfig
from verl.utils.import_utils import deprecated
from verl.workers.config import ActorConfig

PolicyLossFn = Callable[
    [
        torch.Tensor,  # old_log_prob
        torch.Tensor,  # log_prob
        torch.Tensor,  # advantages
        torch.Tensor,  # response_mask
        str,  # loss_agg_mode
        Optional[DictConfig | AlgoConfig],  # config
    ],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
]

POLICY_LOSS_REGISTRY: dict[str, PolicyLossFn] = {}


def register_policy_loss(name: str) -> Callable[[PolicyLossFn], PolicyLossFn]:
    """Register a policy loss function with the given name.

    Args:
        name (str): The name to register the policy loss function under.

    Returns:
        function: Decorator function that registers the policy loss function.
    """

    def decorator(func: PolicyLossFn) -> PolicyLossFn:
        POLICY_LOSS_REGISTRY[name] = func
        return func

    return decorator



class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(kl_ctrl):
    if kl_ctrl.type == "fixed":
        return FixedKLController(kl_coef=kl_ctrl.kl_coef)
    elif kl_ctrl.type == "adaptive":
        assert kl_ctrl.horizon > 0, f"horizon must be larger than 0. Got {kl_ctrl.horizon}"
        return AdaptiveKLController(init_kl_coef=kl_ctrl.kl_coef, target_kl=kl_ctrl.target_kl, horizon=kl_ctrl.horizon)
    else:
        raise NotImplementedError

## =================================================================
## =============================1️⃣ PPO =============================
## =================================================================
# 📄 The loss function is implemented in ARLArena/verl/trainer/ppo/core_algos.py under
#    @register_policy_loss("vanilla")
def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        values: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma is `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, response_mask)
    return advantages, returns

## =================================================================
## =============================2️⃣ GRPO ============================
## =================================================================
# 📄 The loss function is implemented in ARLArena/verl/trainer/ppo/core_algos.py under
#    @register_policy_loss("vanilla")
# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    traj_index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: str = True,
    compute_mean_std_cross_steps: bool = True,
):
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length)
        norm_adv_by_std_in_grpo: (bool)
            whether to scale the GRPO advantage.
            If True, the advantage is scaled by the std, as in the original GRPO.
            If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).
        compute_mean_std_cross_steps: bool
            If True (more stable), the mean and std are computed across steps within one group. 
            If False (i.e., standard episode-level adv), the mean and std are computed across trajectories within one group.

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, response_length)
        Returns: `(torch.Tensor)`
            shape is (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    traj2steps = defaultdict(list)  # key: (idx, traj), value: list of step-score tensors
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            key = (index[i], traj_index[i])
            traj2steps[key].append(scores[i])

        if compute_mean_std_cross_steps:
            # Every step is a sample: add all step-scores into id2score[index]
            for (idx, traj), step_list in traj2steps.items():
                for s in step_list:
                    id2score[idx].append(s)
        else:
            # Per-trajectory average: for each (idx, traj) compute mean across its steps,
            # then use that mean as one sample for the group identified by idx.
            for (idx, traj), step_list in traj2steps.items():
                # stack step tensors and compute mean for this trajectory
                stacked = torch.stack(step_list)  # shape (num_steps_in_traj,)
                traj_mean = torch.mean(stacked)
                id2score[idx].append(traj_mean)
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


def compute_grpo_passk_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    traj_index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    compute_mean_std_cross_steps: bool = True,
):
    """
    Compute advantage for Pass@k using a GRPO-style outcome reward formulation.
    Only the best response per group gets a non-zero advantage: r_max - r_second_max.

    Implemented as described in https://arxiv.org/abs/2503.19595.

    Args:
        token_level_rewards: (bs, response_length)
        response_mask: (bs, response_length)
        index: (bs,) → group ID per sample
        epsilon: float for numerical stability
        norm_adv_by_std_in_grpo: if True, normalize advantage by std within group
        compute_mean_std_cross_steps: bool
            If True (more stable), the mean and std are computed across steps within one group. 
            If False (i.e., standard episode-level adv), the mean and std are computed across trajectories within one group.

    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)  # (bs,)
    advantages = torch.zeros_like(scores)

    id2scores = defaultdict(list)
    id2indices = defaultdict(list)
    seen_pairs = set()
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            idx = index[i]
            id2scores[idx].append(scores[i])
            id2indices[idx].append(i)
            if not compute_mean_std_cross_steps:
                seen_pairs.add((index[i], traj_index[i]))
        for idx in id2scores:
            rewards = torch.stack(id2scores[idx])  # (k,)
            if rewards.numel() < 2:
                raise ValueError(f"Pass@k requires at least 2 samples per group. Got {rewards.numel()} for group {idx}.")
            topk, topk_idx = torch.topk(rewards, 2)
            r_max, r_second_max = topk[0], topk[1]
            i_max = id2indices[idx][topk_idx[0].item()]
            advantage = r_max - r_second_max
            if norm_adv_by_std_in_grpo:
                std = torch.std(rewards)
                advantage = advantage / (std + epsilon)
            advantages[i_max] = advantage

    advantages = advantages.unsqueeze(-1) * response_mask
    return advantages, advantages

## =================================================================
## =========================3️⃣ Reinforce++ =========================
## =================================================================
# 📄 The loss function is implemented in ARLArena/verl/trainer/ppo/core_algos.py under
#    @register_policy_loss("vanilla")
def compute_reinforce_plus_plus_baseline_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor, traj_index: np.ndarray, epsilon: float = 1e-6, compute_mean_std_cross_steps: bool = True):
    """
    Compute advantage for RF++-baseline (https://arxiv.org/abs/2501.03262), operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    seen_pairs = set()
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            id2score[index[i]].append(scores[i])
            if not compute_mean_std_cross_steps:
                seen_pairs.add((index[i], traj_index[i]))
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]

        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        scores = verl_F.masked_whiten(scores, response_mask) * response_mask

    return scores, scores


## =================================================================
## =============================4️⃣ RLOO ============================
## =================================================================
# 📄 The loss function is implemented in ARLArena/verl/trainer/ppo/core_algos.py under
#    @register_policy_loss("vanilla")
def compute_rloo_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: np.ndarray, traj_index: np.ndarray, epsilon: float = 1e-6, compute_mean_std_cross_steps: bool = True):
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    seen_pairs = set()
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            id2score[index[i]].append(scores[i])
            if not compute_mean_std_cross_steps:
                seen_pairs.add((index[i], traj_index[i]))
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num - 1) - id2mean[index[i]] * response_num / (response_num - 1)
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


def compute_reinforce_plus_plus_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, gamma: torch.Tensor):
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * response_mask[:, t]

        advantages = verl_F.masked_whiten(returns, response_mask)
        advantages = advantages * response_mask

    return advantages, returns

## =================================================================
## =============================5️⃣ ReMaX ===========================
## =================================================================
# 📄 The loss function is implemented in ARLArena/verl/trainer/ppo/core_algos.py under
#    @register_policy_loss("vanilla")
def compute_remax_outcome_advantage(token_level_rewards: torch.Tensor, reward_baselines: torch.Tensor, response_mask: torch.Tensor):
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505

    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = returns - reward_baselines.unsqueeze(-1) * response_mask

    return advantages, returns


## =================================================================
## =============================6️⃣ GiGPO ===========================
## =================================================================
# 📄 The loss function is implemented in ARLArena/verl/trainer/ppo/core_algos.py under
#    @register_policy_loss("vanilla")
def compute_gigpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   step_rewards: torch.Tensor,
                                   response_mask: torch.Tensor,
                                   anchor_obs: np.array,
                                   index: np.array,
                                   traj_index: np.array,
                                   epsilon: float = 1e-6,
                                   step_advantage_w: float = 1.0,
                                   mode: str = "mean_norm",
                                   enable_similarity: bool = False,
                                   similarity_thresh: float = 0.95,
                                   ):
    """
    Compute the advantages for GiGPO (https://arxiv.org/abs/2505.10978).
    """
    if mode == "mean_std_norm":
        remove_std = False
    elif mode == "mean_norm":
        remove_std = True
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Compute episode relative advantages (Eq. 3 in the paper).
    episode_advantages = episode_norm_reward(token_level_rewards, response_mask, index, traj_index, epsilon, remove_std)
    
    # Anchor state grouping (Eq. 6 in the paper).
    step_group_uids = build_step_group(anchor_obs, index, enable_similarity, similarity_thresh)

    # Compute step relative advantages (Eq. 7 in the paper).
    step_advantages = step_norm_reward(step_rewards, response_mask, step_group_uids, epsilon, remove_std)

    # Compute joint advantages (Eq. 8 in the paper).
    scores = episode_advantages + step_advantage_w * step_advantages
    return scores, scores


## ================================================================
## ============================= 7️⃣ AEPO ==========================
## ================================================================
# 🌟 Note: AEPO does not have its own specific advantage estimator.
#    Instead, it uses the same advantage estimation as GRPO by default.
# 📄 The loss function for AEPO is implemented in ARLArena/verl/trainer/ppo/core_algos.py under
#    @register_policy_loss("aepo").


## ================================================================
## ============================= 8️⃣ DAPO ==========================
## ================================================================
# 🌟 Note: DAPO uses the same advantage computation as Grpo.
# 📄 The loss function is implemented in ARLArena/verl/trainer/ppo/core_algos.py under
#    @register_policy_loss("vanilla")


## ================================================================
## ============================ 9️⃣ Dr GRPO ========================
## ================================================================
# 🌟 Note: Dr.GRPO only changes norm_adv_by_std_in_grpo=False
# 📄 The loss function is implemented in ARLArena/verl/trainer/ppo/core_algos.py under
#    @register_policy_loss("vanilla")
# ray_trainer.py: self.config.actor_rollout_ref.actor.policy_loss.loss_agg_mode = "seq-mean-token-sum"

## =================================================================
## ============================= 🔟 GSPO ===========================
## =================================================================
# 🌟 Note: GSPO has no specific advantage estimator, so we use the same as Grpo.
# 📄 The loss function for GSPO is implemented in ARLArena/verl/trainer/ppo/core_algos.py under
#    @register_policy_loss("gspo").

## ================================================================
## =========================== 1️⃣1️⃣ CISPO ==========================
## ================================================================
# 🌟 Note: CISPO has no specific advantage estimator, so we use the same as Grpo.
# 📄 The loss function for CISPO is implemented in ARLArena/verl/trainer/ppo/core_algos.py under
#    @register_policy_loss("cispo").


## ================================================================
## =========================== 1️⃣2️⃣ EMPG ==========================
## ================================================================
# 📄 The loss function is implemented in ARLArena/verl/trainer/ppo/core_algos.py under
#    @register_policy_loss("vanilla")
def compute_EMPG_advantage(batch, k=1.0, k_f=1.0, zeta=0.05):
    """
    Args:
        tokenizer: The tokenizer for identifying response segments.
        batch: A data batch with 'responses', 'old_entropy', 'advantages'.
        k (float): Hyperparameter for self-calibrating gradient scaling.
        k_f (float): Hyperparameter for the future clarity bonus.
        zeta (float): Hyperparameter for the future clarity bonus.
    """

    # --- 1. Calculate Modulated Advantage Components ---
    H = np.array(batch.batch['old_entropy'])
    mask = np.array(batch.batch['response_mask'])

    nume = (H * mask).sum(axis=-1, keepdims=True)
    deno = mask.sum(axis=-1, keepdims=True)
    H = nume / np.maximum(deno, 1)
    # Batch-level entropy normalization (Eq. 12) with epsilon = 1e-8
    min_H, max_H = np.min(H), np.max(H)
    H_norm = (H - min_H) / (max_H - min_H + 1e-8)
    # Self-calibrating gradient scaling g(H) (Eq. 10)
    g_H_unnormalized = np.exp(-k * H_norm)
    mean_g_H = np.mean(g_H_unnormalized)
    g_H = g_H_unnormalized / (mean_g_H + 1e-8)
    # Future clarity bonus f(H) (Eq. 11)
    f_H = np.exp(-k_f * H_norm)
    # Convert to tensors for PyTorch operations
    g_H = torch.tensor(g_H, device=batch.batch['advantages'].device, dtype=torch.float32)
    f_H = torch.tensor(f_H, device=batch.batch['advantages'].device, dtype=torch.float32)
    batch.batch['f_entropy'] = f_H

    # --- 2. Group into trajectories ---
    
    steps2traj_f_H = defaultdict(dict)  # key: (idx, traj), value: list of step-score tensors
    with torch.no_grad():
        bsz = batch.batch.batch_size[0]
        for i in range(bsz):
            raw_prompt = batch.non_tensor_batch['raw_prompt'][i][0]['content']
            m = re.search(r'you have already taken (\d+) step\(s\)', raw_prompt)
            if m:
                turn_idx = int(m.group(1))
            else:
                turn_idx = 0
            traj_uid = batch.non_tensor_batch['traj_uid'][i]
            steps2traj_f_H[traj_uid][turn_idx] = batch.batch['f_entropy'][i, 0]

    # --- 3. Second Pass: Apply Advantage Modulation (Eq. 8) ---
    for i in range(batch.batch.batch_size[0]):
        raw_prompt = batch.non_tensor_batch['raw_prompt'][i][0]['content']
        traj_uid = batch.non_tensor_batch['traj_uid'][i]
        # Apply self-calibrating gradient scaling
        batch.batch['advantages'][i] *= g_H[i]

        m = re.search(r'you have already taken (\d+) step\(s\)', raw_prompt)
        if m:
            turn_idx = int(m.group(1))
        else:
            turn_idx = 0
        
        if turn_idx+1 < len(steps2traj_f_H[traj_uid]):
            batch.batch['advantages'][i] += zeta * steps2traj_f_H[traj_uid][turn_idx + 1]
        
    batch.batch['advantages'] -= torch.mean(batch.batch['advantages'][:, 0])
    batch.batch['advantages'] *= mask

    return batch.batch['advantages'], batch.batch['advantages']

## ================================================================
## =========================== 1️⃣3️⃣ SAPO ===========================
## ================================================================
# 🌟 Note: SAPO has no specific advantage estimator, so we use the same as Grpo.
# 📄 The loss function for SAPO is implemented in ARLArena/verl/trainer/ppo/core_algos.py under
#    @register_policy_loss("sapo").

## ================================================================
## ======================== 1️⃣4️⃣ Vanilla Grpo ======================
## ================================================================
# 🌟 Note: Vanilla Grpo set compute_mean_std_cross_steps=False
# 📄 The loss function for Vanilla Grpo is implemented in ARLArena/verl/trainer/ppo/core_algos.py under
#    @register_policy_loss("vanilla").

def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio

def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
    """
    Aggregate the loss matrix into a scalar.

    Args:
        loss_mat: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_agg_mode: (str) choices:
            method to aggregate the loss matrix into a scalar.
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        loss = verl_F.masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)  # token-mean
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        loss = torch.sum(seq_losses) / loss_mask.shape[-1]  # The divisor
        # (loss_mask.shape[-1]) should ideally be constant
        # throughout training to well-replicate the DrGRPO paper.
        # TODO: Perhaps add user-defined normalizer argument to
        # agg_loss to ensure divisor stays constant throughout.
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


def compute_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
):
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        clip_ratio_c (float, optional):
            Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
            Defaults to 3.0.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
    """
    assert clip_ratio_c > 1.0, "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0," + f" but get the value: {clip_ratio_c}."

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask)

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


def compute_entropy_loss(logits, response_mask, loss_agg_mode: str = "token-mean"):
    """Compute categorical entropy loss (For backward compatibility)

    Args:
        logits (torch.Tensor): shape is (bs, response_length, vocab_size)
        response_mask (torch.Tensor): shape is (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    token_entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = agg_loss(loss_mat=token_entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    return entropy_loss


def compute_value_loss(vpreds: torch.Tensor, returns: torch.Tensor, values: torch.Tensor, response_mask: torch.Tensor, cliprange_value: float, loss_agg_mode: str = "token-mean"):
    """
    Compute the clipped value-function loss for PPO.

    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (torch.FloatTensor):
            Predicted values from the value head, shape (batch_size, response_length).
        values (torch.FloatTensor):
            Old (baseline) values from the value head, shape (batch_size, response_length).
        returns (torch.FloatTensor):
            Ground-truth returns, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the value loss calculation.
        cliprange_value (float):
            Clip range for value prediction updates.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".

    Returns:
        vf_loss (torch.FloatTensor):
            A scalar tensor containing the aggregated value-function loss.
        vf_clipfrac (float):
            Fraction of elements where the clipped loss was used.
    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    clipped_vf_losses = torch.max(vf_losses1, vf_losses2)
    vf_loss = agg_loss(loss_mat=clipped_vf_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), response_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty in ("kl", "k1"):
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty in ("low_var_kl", "k3"):
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError


def compute_pf_ppo_reweight_data(
    data,
    reweight_method: str = "pow",
    weight_pow: float = 2.0,
):
    """Reweight the data based on the token_level_scores.

    Args:
        data: DataProto object, containing batch, non_tensor_batch and meta_info
        reweight_method: str, choices: "pow", "max_min", "max_random"
        weight_pow: float, the power of the weight

    Returns:

    """

    @torch.no_grad()
    def compute_weights(scores: torch.Tensor, reweight_method: str, weight_pow: float) -> torch.Tensor:
        if reweight_method == "pow":
            weights = torch.pow(torch.abs(scores), weight_pow)
        elif reweight_method == "max_min":
            max_score = torch.max(scores)
            min_score = torch.min(scores)
            weights = torch.where((scores == max_score) | (scores == min_score), 1.0, 0.0)
        elif reweight_method == "max_random":
            max_score = torch.max(scores)
            weights = torch.where(scores == max_score, 0.4, 0.1)
        else:
            raise ValueError(f"Unsupported reweight_method: {reweight_method}")
        return weights

    scores = data.batch["token_level_scores"].sum(dim=-1)
    weights = compute_weights(scores, reweight_method, weight_pow)
    weights = torch.clamp(weights + 1e-8, min=1e-8)

    batch_size = scores.shape[0]
    sample_indices = torch.multinomial(weights, batch_size, replacement=True)

    resampled_batch = {key: tensor[sample_indices] for key, tensor in data.batch.items()}

    sample_indices_np = sample_indices.numpy()
    resampled_non_tensor_batch = {}
    for key, array in data.non_tensor_batch.items():
        if isinstance(array, np.ndarray):
            resampled_non_tensor_batch[key] = array[sample_indices_np]
        else:
            resampled_non_tensor_batch[key] = [array[i] for i in sample_indices_np]

    resampled_meta_info = {}
    for key, value in data.meta_info.items():
        if isinstance(value, list) and len(value) == batch_size:
            resampled_meta_info[key] = [value[i] for i in sample_indices_np]
        else:
            resampled_meta_info[key] = value

    from copy import deepcopy

    resampled_data = deepcopy(data)
    resampled_data.batch = type(data.batch)(resampled_batch)
    resampled_data.batch.batch_size = data.batch.batch_size
    resampled_data.non_tensor_batch = resampled_non_tensor_batch
    resampled_data.meta_info = resampled_meta_info

    return resampled_data


## ================================================================
## ========================= General Utils ========================
## ================================================================

def process_token_sequences(
    token_id_tensor: torch.Tensor,
    start_end_delimiter_seq: list[int] = [151644, 77091, 198],
    target_delimiter_seq: list[int] = [151645],
    head_sequence: list[int] = [151645, 151644, 87280]
) -> dict:
    """
    Processes a token ID tensor to find tokens between specific delimiters and
    the end index of a leading sequence.
    Args:
        token_id_tensor (torch.Tensor): A 1D PyTorch tensor of token IDs.
        start_end_delimiter_seq (list[int]): The sequence marking the start and end
                                              of the segments to extract.
                                              For your case, this is [151644, 77091, 1699]
                                              for start, and [151645] for end.
                                              This function assumes the 'start' sequence
                                              is [151644, 77091, 1699] and the 'end' sequence
                                              is [151645] as per your previous requests.
        target_delimiter_seq (list[int]): The sequence to mark the end of segments
                                            when found after start_end_delimiter_seq.
                                            For your case, this is [151645].
        head_sequence (list[int]): The sequence to find from the beginning of the tensor,
                                   and return its exclusive end index.
                                   For your case, this is [151645, 151644, 872].
    Returns:
        dict: A dictionary containing:
              - 'between_delimiters': A list of tuples, where each tuple contains:
                                      (start_index_of_tokens, end_index_of_tokens, tokens_tensor).
                                      These are the tokens found between `start_end_delimiter_seq`
                                      and `target_delimiter_seq`.
              - 'first_head_sequence_end_index': The exclusive end index of the first
                                                 `head_sequence` found from the beginning of the tensor.
                                                 Returns -1 if not found.
    """
    if not isinstance(token_id_tensor, torch.Tensor) or token_id_tensor.ndim != 1:
        raise ValueError("token_id_tensor must be a 1D PyTorch tensor.")
    if not start_end_delimiter_seq or not target_delimiter_seq or not head_sequence:
        raise ValueError("All sequence arguments cannot be empty.")
    results = []
    start_seq_len = len(start_end_delimiter_seq)
    target_seq_len = len(target_delimiter_seq)
    head_seq_len = len(head_sequence)
    start_seq_tensor = torch.tensor(start_end_delimiter_seq, dtype=token_id_tensor.dtype, device=token_id_tensor.device)
    target_seq_tensor = torch.tensor(target_delimiter_seq, dtype=token_id_tensor.dtype, device=token_id_tensor.device)
    head_seq_tensor = torch.tensor(head_sequence, dtype=token_id_tensor.dtype, device=token_id_tensor.device)
    # --- Part 1: Find the end index of the first head_sequence from the beginning ---
    for k in range(len(token_id_tensor) - head_seq_len + 1):
        if torch.equal(token_id_tensor[k : k + head_seq_len], head_seq_tensor):
            results.append((0, k + head_seq_len))
            break # Found the first one, no need to search further
    # --- Part 2: Find tokens between start_end_delimiter_seq and target_delimiter_seq ---
    for i in range(len(token_id_tensor) - start_seq_len + 1):
        if torch.equal(token_id_tensor[i : i + start_seq_len], start_seq_tensor):
            for j in range(i + start_seq_len, len(token_id_tensor) - target_seq_len + 1):
                if torch.equal(token_id_tensor[j : j + target_seq_len], target_seq_tensor):
                    tokens_start_idx = i + start_seq_len
                    tokens_end_idx = j
                    results.append((tokens_start_idx, tokens_end_idx + 1))
                    break # Found a pair, move to find the next start_end_delimiter_seq
    return results

def to_hashable(x):
    """Convert an object into a hashable type (used for clustering/grouping)."""
    if isinstance(x, (int, float, str, bool)):
        return x
    elif isinstance(x, (np.integer, np.floating)):
        return x.item()
    elif isinstance(x, np.ndarray):
        return tuple(x.flatten())
    elif isinstance(x, (list, tuple)):
        return tuple(to_hashable(e) for e in x)
    elif isinstance(x, dict):
        return tuple(sorted((k, to_hashable(v)) for k, v in x.items()))
    else:
        raise TypeError(f"Unsupported type: {type(x)}")

def summarize_group_size(group_size: list):
    """
    Summarize the dynamics of step-level group.
    Args:
        group_size : List[int]
    """
    counts = Counter(group_size)
    total = sum(counts.values())
    max_size = max(counts)

    summary = {}
    for size in range(1, max_size + 1):
        cnt = counts.get(size, 0)
        prop = cnt / total if total > 0 else 0
        summary[size] = (cnt, prop)

    print("Summary of step-level group sizes:")
    print("Size | Count | Proportion")
    print("-------------------------")
    for size, (cnt, prop) in summary.items():
        if prop:
            print(f"{size:>4} | {cnt:>5} | {prop:>9.2%}")
            
def are_similar(a: str, b: str, threshold: float = 0.95) -> bool:
    """
    Check whether two text observations are similar enough.
    
    Args:
        a, b (str): Input strings to compare.
        threshold (float): Minimum similarity ratio.
    
    Returns:
        bool: True if similarity >= threshold.
    """
    if not isinstance(a, str) or not isinstance(b, str):
        raise ValueError("Only text-based observations are supported for similarity-based GiGPO in this version.")
    return SequenceMatcher(None, a, b).ratio() >= threshold

def compute_step_discounted_returns(batch: DataProto, gamma: float):
    """
    Compute discounted returns for each trajectory. (Eq. 5 in the paper)
    
    Args:
        batch (DataProto): Input batch.
        gamma (float): Discount factor.
    
    Returns:
        torch.Tensor: Discounted returns.
    """
    rewards = batch.non_tensor_batch['rewards'].astype(np.float32)
    traj_uids = batch.non_tensor_batch['traj_uid']
    active_masks = batch.non_tensor_batch['active_masks'].astype(np.float32)
    returns_by_traj = {}
    unique_traj_uids = np.unique(traj_uids)
    for uid in unique_traj_uids:
        # Get indices for this trajectory
        traj_indices = np.where(traj_uids == uid)[0]
        
        # Extract rewards and masks for this trajectory
        traj_rewards = rewards[traj_indices]
        traj_active_masks = active_masks[traj_indices]
        assert traj_active_masks.all(), "active_masks should be all 1s for the same trajectory"
        
        # Calculate returns
        traj_returns = np.zeros_like(traj_rewards)
        running_return = 0
        
        # Calculate returns from the end to the start
        for t in reversed(range(len(traj_rewards))):
            running_return = traj_rewards[t] + gamma * running_return
            traj_returns[t] = running_return
        
        # Store the results
        returns_by_traj[uid] = traj_returns
    
    # Recombine the returns into the original batch order
    all_returns = np.zeros_like(rewards)
    for i, uid in enumerate(traj_uids):
        traj_indices = np.where(traj_uids == uid)[0]
        idx_in_traj = np.where(traj_indices == i)[0][0]  # Find position of i in its trajectory
        all_returns[i] = returns_by_traj[uid][idx_in_traj]
    
    all_returns = torch.tensor(all_returns, dtype=torch.float32, device=batch.batch['input_ids'].device)
    return all_returns


def episode_norm_reward(token_level_rewards: torch.Tensor,
                        response_mask: torch.Tensor,
                        index: np.array,
                        traj_index: np.array,
                        epsilon: float = 1e-6,
                        remove_std: bool = True,
                        compute_mean_std_cross_steps: bool = True,
                        ):
    """
    Compute episode-level advantage using mean-std normalization for GiGPO.
    (with only one scalar reward for each episode).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        index: `(np.array)`
            shape: (bs,)
        traj_index: `(np.array)`
            shape: (bs,)
        epsilon: float
            A small value to avoid division by zero.
        remove_std: bool
            If True, the standard deviation is removed from the normalization.
        compute_mean_std_cross_steps: bool
            If True (more stable), the mean and std are computed across steps within one group. 
            If False (i.e., standard episode-level adv), the mean and std are computed across trajectories within one group.
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    seen_pairs = set()
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            id2score[index[i]].append(scores[i])
            if not compute_mean_std_cross_steps:
                seen_pairs.add((index[i], traj_index[i]))

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if remove_std:
                scores[i] = scores[i] - id2mean[index[i]]
            else:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        episode_advantages = scores.unsqueeze(-1).tile([1, response_length]) * response_mask

    return episode_advantages


def build_step_group(anchor_obs: np.array, index: np.array, enable_similarity: bool = False, similarity_thresh: float = 0.95, summarize: bool = False):
    """
    Group observations by index and then cluster identical observations within each index group.
    Assigns a unique step_group_uid (UUID) to each cluster.
    
    Parameters:
    -----------
    anchor_obs : np.array
        Array of observation strings
    index : np.array
        Array of episode_group_uid
    summarize : bool
        Whether to summarize the group sizes (default: True)
    enable_similarity : bool
        Whether to enable similarity-based step-level grouping (default: False)
    similarity_thresh : float
        Threshold for similarity to consider two observations as identical (default: 1.0, meaning exact match)
    
    Returns:
    --------
    np.array
        Array of step_group_uid values corresponding to the original anchor_obs array
    """
    if enable_similarity:
        assert similarity_thresh > 0.0 and similarity_thresh < 1.0, "When enabling similarity-based step-level group, similarity_thresh should be in (0, 1)"

    # Initialize the result array with placeholder values
    step_group_uids = np.empty(len(anchor_obs), dtype=object)
    
    # Get unique indices
    unique_indices = np.unique(index)

    group_size: List[int] = []
    # Process each unique index
    for idx in unique_indices:
        if not enable_similarity:
            # Get all observations for this index using np.where
            indices = np.where(index == idx)[0]
            obs_group = anchor_obs[indices]
            
            # Create clusters for identical observations
            clusters = defaultdict(list)
            for i, obs in enumerate(obs_group):
                clusters[to_hashable(obs)].append(indices[i])  # Store the original index position
            
            # Assign unique step_group_uid to each cluster
            for obs, original_indices in clusters.items():
                # Generate a UUID for this cluster
                uid = str(uuid.uuid4())
                
                # Assign the same step_group_uid to all elements in this cluster
                group_size.append(len(original_indices))
                for original_idx in original_indices:
                    step_group_uids[original_idx] = uid
        else:
            locs = np.where(index == idx)[0]
            obs_group = anchor_obs[locs]

            # Dynamically maintain clusters: [{rep: str, locs: List[int]} ...]
            clusters: List[Dict[str, Any]] = []

            for obs, loc in zip(obs_group, locs):
                 # Try to place into an existing cluster
                placed = False
                for cluster in clusters:
                    if are_similar(obs, cluster["rep"], similarity_thresh):
                        cluster["locs"].append(loc)
                        placed = True
                        break
                # If no matching cluster, create a new one
                if not placed:
                    clusters.append({"rep": obs, "locs": [loc]})

            # Assign a UUID to each cluster
            for cluster in clusters:
                uid = str(uuid.uuid4())
                group_size.append(len(cluster["locs"]))
                for loc in cluster["locs"]:
                    step_group_uids[loc] = uid

        # Validate that all elements have been assigned a uid
    if None in step_group_uids or np.any(step_group_uids == None):
        missing_indices = np.where(step_group_uids == None)[0]
        raise ValueError(f"Failed to assign UIDs to all observations. Missing at indices: {missing_indices}")

    if summarize:
        summarize_group_size(group_size)
    print(f"Avg size of step-level group: {np.mean(group_size)}")
    return step_group_uids


def step_norm_reward(step_rewards: torch.Tensor,
                      response_mask: torch.Tensor,
                      index: np.array,
                      epsilon: float = 1e-6,
                      remove_std: bool = True,
                      ):
    """
    Compute step-level advantage using mean-std normalization for GiGPO.
    Args:
        step_rewards: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = response_mask.shape[-1]
    scores = step_rewards.clone()

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                print(f"id2score: {id2score}")
                print(f"len(id2score[idx]): {len(id2score[idx])}")
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if remove_std:
                scores[i] = scores[i] - id2mean[index[i]]
            else:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        step_advantages = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
    
    return step_advantages
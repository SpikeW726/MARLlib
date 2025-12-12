# MIT License

# Copyright (c) 2023 Replicable-MARL

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy, KLCoeffMixin
from ray.rllib.agents.ppo.ppo import PPOTrainer, DEFAULT_CONFIG as PPO_CONFIG
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_policy import EntropyCoeffSchedule, \
    LearningRateSchedule
from ray.rllib.utils.typing import TensorType
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_ops import explained_variance, sequence_mask, convert_to_torch_tensor
from ray.rllib.evaluation.postprocessing import Postprocessing
from marllib.marl.algos.utils.centralized_critic import CentralizedValueMixin, centralized_critic_postprocessing
from marllib.marl.algos.core import setup_torch_mixins

torch, nn = try_import_torch()

#############
### MAPPO ###
#############

def central_critic_ppo_loss(policy: Policy, model: ModelV2,
                            dist_class: ActionDistribution,
                            train_batch: SampleBatch) -> TensorType:
    """Constructs the loss for Centralized PPO Objective with Active Masks support.
    
    Active Masks 机制：当智能体处于非决策状态（No-Op）时，其对应的 Loss 不参与梯度更新。
    这对于异步决策环境（如事件驱动的巡逻环境）非常重要。
    
    Args:
        policy (Policy): The Policy to calculate the loss for.
        model (ModelV2): The Model to calculate the loss for.
        dist_class (Type[ActionDistribution]): The action distr. class.
        train_batch (SampleBatch): The training data.

    Returns:
        Union[TensorType, List[TensorType]]: A single loss tensor or a list
            of loss tensors.
    """
    CentralizedValueMixin.__init__(policy)

    # 设置 centralized value function
    vf_saved = model.value_function
    opp_action_in_cc = policy.config["model"]["custom_model_config"]["opp_action_in_cc"]
    model.value_function = lambda: policy.model.central_value_function(
        train_batch["state"],
        train_batch["opponent_actions"] if opp_action_in_cc else None
    )
    policy._central_value_out = model.value_function()

    # ==================== Active Masks 处理 ====================
    # 从 train_batch 中获取 active_masks
    # active_mask = 1.0 表示智能体需要决策，0.0 表示正在执行动作中(No-Op)
    if 'active_masks' in train_batch:
        active_masks = convert_to_torch_tensor(train_batch['active_masks'], policy.device)
        # 确保是一维的，形状为 [batch_size]
        active_masks = active_masks.reshape(-1)
    else:
        # 向后兼容：如果没有 active_masks，默认所有样本都是活跃的
        active_masks = torch.ones(train_batch.count, device=policy.device)
    
    # 计算有效样本数量（避免除以零）
    num_valid = torch.sum(active_masks)
    num_valid = torch.clamp(num_valid, min=1.0)  # 确保至少为 1
    # ==================== Active Masks 处理结束 ====================

    # 前向传播获取 logits
    logits, state = model(train_batch)
    curr_action_dist = dist_class(logits, model)

    # RNN case: Mask away 0-padded chunks at end of time axis.
    if state:
        B = len(train_batch[SampleBatch.SEQ_LENS])
        max_seq_len = logits.shape[0] // B
        seq_mask = sequence_mask(
            train_batch[SampleBatch.SEQ_LENS],
            max_seq_len,
            time_major=model.is_time_major())
        seq_mask = torch.reshape(seq_mask, [-1])
        # 组合 sequence mask 和 active masks
        combined_mask = seq_mask.float() * active_masks
        num_valid = torch.sum(combined_mask)
        num_valid = torch.clamp(num_valid, min=1.0)
    else:
        seq_mask = None
        combined_mask = active_masks

    # 计算 importance sampling ratio
    prev_action_dist = dist_class(train_batch[SampleBatch.ACTION_DIST_INPUTS], model)
    logp_ratio = torch.exp(
        curr_action_dist.logp(train_batch[SampleBatch.ACTIONS]) -
        train_batch[SampleBatch.ACTION_LOGP]
    )

    # KL divergence
    action_kl = prev_action_dist.kl(curr_action_dist)
    
    # Entropy
    curr_entropy = curr_action_dist.entropy()

    # Surrogate loss (PPO clipping)
    advantages = train_batch[Postprocessing.ADVANTAGES]
    surrogate_loss = torch.min(
        advantages * logp_ratio,
        advantages * torch.clamp(
            logp_ratio, 
            1 - policy.config["clip_param"],
            1 + policy.config["clip_param"]
        )
    )

    # 应用 Active Masks 到 policy loss
    # Loss = (loss * active_masks).sum() / num_valid
    mean_policy_loss = torch.sum(-surrogate_loss * combined_mask) / num_valid
    mean_kl_loss = torch.sum(action_kl * combined_mask) / num_valid
    mean_entropy = torch.sum(curr_entropy * combined_mask) / num_valid

    # Compute value function loss
    if policy.config["use_critic"]:
        prev_value_fn_out = train_batch[SampleBatch.VF_PREDS]
        value_fn_out = model.value_function()
        vf_loss1 = torch.pow(
            value_fn_out - train_batch[Postprocessing.VALUE_TARGETS], 2.0)
        vf_clipped = prev_value_fn_out + torch.clamp(
            value_fn_out - prev_value_fn_out, 
            -policy.config["vf_clip_param"],
            policy.config["vf_clip_param"]
        )
        vf_loss2 = torch.pow(
            vf_clipped - train_batch[Postprocessing.VALUE_TARGETS], 2.0)
        vf_loss = torch.max(vf_loss1, vf_loss2)
        # 应用 Active Masks 到 value loss
        mean_vf_loss = torch.sum(vf_loss * combined_mask) / num_valid
    else:
        vf_loss = mean_vf_loss = 0.0

    # Total loss
    total_loss = (
        mean_policy_loss +
        policy.kl_coeff * mean_kl_loss +
        policy.config["vf_loss_coeff"] * mean_vf_loss -
        policy.entropy_coeff * mean_entropy
    )

    # 恢复原始 value function
    model.value_function = vf_saved

    # Store values for stats function in model (tower)
    model.tower_stats["total_loss"] = total_loss
    model.tower_stats["mean_policy_loss"] = mean_policy_loss
    model.tower_stats["mean_vf_loss"] = mean_vf_loss
    model.tower_stats["vf_explained_var"] = explained_variance(
        train_batch[Postprocessing.VALUE_TARGETS], policy._central_value_out)
    model.tower_stats["mean_entropy"] = mean_entropy
    model.tower_stats["mean_kl_loss"] = mean_kl_loss

    return total_loss


MAPPOTorchPolicy = PPOTorchPolicy.with_updates(
    name="MAPPOTorchPolicy",
    get_default_config=lambda: PPO_CONFIG,
    postprocess_fn=centralized_critic_postprocessing,
    loss_fn=central_critic_ppo_loss,
    before_init=setup_torch_mixins,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
        CentralizedValueMixin
    ])


def get_policy_class_mappo(config_):
    if config_["framework"] == "torch":
        return MAPPOTorchPolicy


MAPPOTrainer = PPOTrainer.with_updates(
    name="MAPPOTrainer",
    default_policy=None,
    get_policy_class=get_policy_class_mappo,
)

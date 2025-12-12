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

from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.typing import TensorType
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_ops import sequence_mask, convert_to_torch_tensor
from ray.rllib.agents.a3c.a3c_torch_policy import A3CTorchPolicy
from ray.rllib.agents.a3c.a2c import A2C_DEFAULT_CONFIG as A2C_CONFIG, A2CTrainer
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation.postprocessing import Postprocessing
from marllib.marl.algos.utils.centralized_critic import CentralizedValueMixin, centralized_critic_postprocessing

torch, nn = try_import_torch()


#############
### MAA2C ###
#############

def central_critic_a2c_loss(policy: Policy, model: ModelV2,
                            dist_class: ActionDistribution,
                            train_batch: SampleBatch) -> TensorType:
    """Constructs the loss for Centralized A2C Objective with Active Masks support.
    
    Active Masks 机制：当智能体处于非决策状态（No-Op）时，其对应的 Loss 不参与梯度更新。
    
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

    # 前向传播
    logits, _ = model.from_batch(train_batch)
    values = model.value_function()
    dist = dist_class(logits, model)

    # ==================== Active Masks 处理 ====================
    if 'active_masks' in train_batch:
        active_masks = convert_to_torch_tensor(train_batch['active_masks'], policy.device)
        active_masks = active_masks.reshape(-1)
    else:
        active_masks = torch.ones(train_batch.count, device=policy.device)

    # RNN case: sequence mask
    if policy.is_recurrent():
        B = len(train_batch[SampleBatch.SEQ_LENS])
        max_seq_len = logits.shape[0] // B
        seq_mask = sequence_mask(train_batch[SampleBatch.SEQ_LENS], max_seq_len)
        seq_mask = torch.reshape(seq_mask, [-1]).float()
        combined_mask = seq_mask * active_masks
    else:
        combined_mask = active_masks

    num_valid = torch.clamp(torch.sum(combined_mask), min=1.0)
    # ==================== Active Masks 处理结束 ====================

    # Log probs
    log_probs = dist.logp(train_batch[SampleBatch.ACTIONS]).reshape(-1)
    
    # Policy loss (advantage weighted log probs) with Active Masks
    pi_err = -torch.sum(
        log_probs * train_batch[Postprocessing.ADVANTAGES] * combined_mask
    ) / num_valid

    # Value loss with Active Masks
    if policy.config["use_critic"]:
        value_err = 0.5 * torch.sum(
            torch.pow(values.reshape(-1) - train_batch[Postprocessing.VALUE_TARGETS], 2.0) * combined_mask
        ) / num_valid
    else:
        value_err = 0.0

    # Entropy with Active Masks
    entropy = torch.sum(dist.entropy() * combined_mask) / num_valid

    # Total loss
    total_loss = (
        pi_err + 
        value_err * policy.config["vf_loss_coeff"] -
        entropy * policy.config["entropy_coeff"]
    )

    # 恢复原始 value function
    model.value_function = vf_saved

    # Store stats
    model.tower_stats["entropy"] = entropy
    model.tower_stats["pi_err"] = pi_err
    model.tower_stats["value_err"] = value_err

    return total_loss


MAA2CTorchPolicy = A3CTorchPolicy.with_updates(
    name="MAA2CTorchPolicy",
    get_default_config=lambda: A2C_CONFIG,
    postprocess_fn=centralized_critic_postprocessing,
    loss_fn=central_critic_a2c_loss,
    mixins=[
        CentralizedValueMixin
    ])


def get_policy_class_maa2c(config_):
    if config_["framework"] == "torch":
        return MAA2CTorchPolicy


MAA2CTrainer = A2CTrainer.with_updates(
    name="MAA2CTrainer",
    default_policy=None,
    get_policy_class=get_policy_class_maa2c,
)

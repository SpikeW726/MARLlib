#!/usr/bin/env python3
"""
在MARLlib中使用Custom Metrics的简化示例
展示如何与现有的MARLlib环境集成
"""

import numpy as np
from typing import Dict
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy


class MARLlibCustomMetricsCallback(DefaultCallbacks):
    """
    适用于MARLlib的自定义指标回调类
    可以与任何MARLlib支持的环境一起使用
    """
    
    def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv,
                         policies: Dict[str, Policy], episode: MultiAgentEpisode, 
                         env_index: int, **kwargs):
        """Episode开始时初始化指标收集"""
        
        # 初始化环境特定的数据收集
        episode.user_data["agent_actions_history"] = {agent: [] for agent in episode.get_agents()}
        episode.user_data["reward_distribution"] = []
        episode.user_data["cooperation_events"] = 0
        episode.user_data["individual_performances"] = {agent: [] for agent in episode.get_agents()}
        
        # 初始化游戏特定指标（以SMAC为例）
        episode.user_data["enemy_units_killed"] = 0
        episode.user_data["damage_dealt"] = 0
        episode.user_data["damage_taken"] = 0
        episode.user_data["healing_done"] = 0
        
        print(f"Started episode {episode.episode_id} with {len(episode.get_agents())} agents")
    
    def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv,
                        policies: Dict[str, Policy], episode: MultiAgentEpisode, 
                        env_index: int, **kwargs):
        """每步收集指标"""
        
        # 收集智能体动作分布
        for agent_id in episode.get_agents():
            if agent_id in episode.last_action_for():
                action = episode.last_action_for(agent_id)
                episode.user_data["agent_actions_history"][agent_id].append(action)
        
        # 分析奖励分布
        current_rewards = []
        for agent_id in episode.get_agents():
            # 获取当前步的奖励
            if hasattr(episode, 'agent_rewards') and (agent_id, "default_policy") in episode.agent_rewards:
                reward = episode.agent_rewards[(agent_id, "default_policy")]
                current_rewards.append(reward)
                episode.user_data["individual_performances"][agent_id].append(reward)
        
        if current_rewards:
            episode.user_data["reward_distribution"].append({
                "mean": np.mean(current_rewards),
                "std": np.std(current_rewards),
                "min": np.min(current_rewards),
                "max": np.max(current_rewards)
            })
        
        # 检测合作事件（基于奖励相关性）
        if len(current_rewards) > 1:
            reward_variance = np.var(current_rewards)
            if reward_variance < 0.1:  # 奖励相似，可能是合作
                episode.user_data["cooperation_events"] += 1
        
        # 从环境info中提取游戏特定信息（如果有的话）
        info = episode.last_info_for()
        if info:
            # SMAC环境的指标
            if "battle_won" in info:
                episode.user_data["battle_won"] = info["battle_won"]
            if "dead_allies" in info:
                episode.user_data["dead_allies"] = info["dead_allies"]
            if "dead_enemies" in info:
                episode.user_data["dead_enemies"] = info["dead_enemies"]
            
            # MPE环境的指标
            if "n_agents" in info:
                episode.user_data["num_agents"] = info["n_agents"]
            
            # 自定义环境指标
            if "custom_score" in info:
                episode.user_data["custom_score"] = info["custom_score"]
    
    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode, 
                       env_index: int, **kwargs):
        """Episode结束时计算最终自定义指标"""
        
        num_agents = len(episode.get_agents())
        episode_length = episode.length
        
        # 1. 智能体行为多样性指标
        action_entropies = []
        for agent_id, actions in episode.user_data["agent_actions_history"].items():
            if actions:
                # 计算动作熵
                unique_actions, counts = np.unique(actions, return_counts=True)
                probs = counts / len(actions)
                entropy = -np.sum(probs * np.log(probs + 1e-8))
                action_entropies.append(entropy)
        
        if action_entropies:
            episode.custom_metrics["avg_action_entropy"] = np.mean(action_entropies)
            episode.custom_metrics["action_diversity"] = np.std(action_entropies)
        
        # 2. 奖励分布指标
        reward_dist = episode.user_data["reward_distribution"]
        if reward_dist:
            mean_rewards = [r["mean"] for r in reward_dist]
            std_rewards = [r["std"] for r in reward_dist]
            
            episode.custom_metrics["reward_stability"] = 1.0 / (1.0 + np.mean(std_rewards))
            episode.custom_metrics["reward_trend"] = np.polyfit(range(len(mean_rewards)), mean_rewards, 1)[0]
            episode.custom_metrics["final_reward_distribution_std"] = reward_dist[-1]["std"] if reward_dist else 0
        
        # 3. 合作指标
        cooperation_rate = episode.user_data["cooperation_events"] / episode_length if episode_length > 0 else 0
        episode.custom_metrics["cooperation_rate"] = cooperation_rate
        episode.custom_metrics["total_cooperation_events"] = episode.user_data["cooperation_events"]
        
        # 4. 个体表现差异
        individual_perfs = episode.user_data["individual_performances"]
        if individual_perfs:
            agent_total_rewards = {agent: sum(rewards) for agent, rewards in individual_perfs.items() if rewards}
            if agent_total_rewards:
                rewards_list = list(agent_total_rewards.values())
                episode.custom_metrics["performance_inequality"] = np.std(rewards_list) / (np.mean(rewards_list) + 1e-8)
                episode.custom_metrics["best_agent_performance"] = max(rewards_list)
                episode.custom_metrics["worst_agent_performance"] = min(rewards_list)
        
        # 5. 环境特定指标（SMAC）
        if "battle_won" in episode.user_data:
            episode.custom_metrics["battle_won"] = episode.user_data["battle_won"]
        if "dead_allies" in episode.user_data:
            episode.custom_metrics["ally_survival_rate"] = 1.0 - (episode.user_data["dead_allies"] / num_agents)
        if "dead_enemies" in episode.user_data:
            episode.custom_metrics["enemy_elimination_rate"] = episode.user_data["dead_enemies"]
        
        # 6. 学习效率指标
        total_episode_reward = episode.total_reward
        avg_reward_per_agent = total_episode_reward / num_agents if num_agents > 0 else 0
        avg_reward_per_step = total_episode_reward / episode_length if episode_length > 0 else 0
        
        episode.custom_metrics["avg_reward_per_agent"] = avg_reward_per_agent
        episode.custom_metrics["avg_reward_per_step"] = avg_reward_per_step
        episode.custom_metrics["reward_efficiency"] = avg_reward_per_step * cooperation_rate
        
        # 7. 可解释性指标
        episode.custom_metrics["episode_complexity"] = episode_length / num_agents  # 复杂度指标
        episode.custom_metrics["agent_utilization"] = len([a for a in individual_perfs.values() if a]) / num_agents
        
        print(f"Episode {episode.episode_id} finished:")
        print(f"  - Length: {episode_length}")
        print(f"  - Total reward: {total_episode_reward:.2f}")
        print(f"  - Cooperation rate: {cooperation_rate:.3f}")
        print(f"  - Avg action entropy: {episode.custom_metrics.get('avg_action_entropy', 0):.3f}")


# MARLlib集成示例
def create_marllib_config_with_metrics(env_name="smac", map_name="3m"):
    """
    创建包含自定义指标的MARLlib配置
    """
    from marllib import marl
    
    # 基础环境配置
    env_config = {
        "map_name": map_name,
    }
    
    # 获取环境
    env = marl.make_env(environment_name=env_name, map_name=map_name, **env_config)
    
    # 基础算法配置
    config = {
        "env": env,
        "env_config": env_config,
        "callbacks": MARLlibCustomMetricsCallback,  # 添加自定义指标回调
        "framework": "torch",
        "num_workers": 2,
        "num_gpus": 0,
        
        # 其他训练配置
        "train_batch_size": 8192,
        "sgd_minibatch_size": 256,
        "num_sgd_iter": 5,
        "lr": 5e-4,
        "gamma": 0.99,
        "lambda": 0.95,
        "clip_param": 0.2,
        "vf_clip_param": 10.0,  # 这就是引起你警告的参数
        
        # 添加自定义配置以支持更多指标收集
        "keep_per_episode_custom_metrics": True,
        "metrics_smoothing_episodes": 100,
    }
    
    return config


def analyze_custom_metrics(result):
    """
    分析训练结果中的自定义指标
    """
    custom_metrics = result.get("custom_metrics", {})
    
    print("\n=== Custom Metrics Analysis ===")
    
    # 合作相关指标
    coop_rate = custom_metrics.get("cooperation_rate_mean", 0)
    print(f"Average Cooperation Rate: {coop_rate:.3f}")
    
    # 行为多样性
    action_entropy = custom_metrics.get("avg_action_entropy_mean", 0)
    action_diversity = custom_metrics.get("action_diversity_mean", 0)
    print(f"Action Entropy: {action_entropy:.3f}")
    print(f"Action Diversity: {action_diversity:.3f}")
    
    # 奖励分析
    reward_stability = custom_metrics.get("reward_stability_mean", 0)
    reward_trend = custom_metrics.get("reward_trend_mean", 0)
    print(f"Reward Stability: {reward_stability:.3f}")
    print(f"Reward Trend: {reward_trend:.3f}")
    
    # 性能不平等
    perf_inequality = custom_metrics.get("performance_inequality_mean", 0)
    print(f"Performance Inequality: {perf_inequality:.3f}")
    
    # 环境特定指标
    if "battle_won_mean" in custom_metrics:
        win_rate = custom_metrics["battle_won_mean"]
        print(f"Win Rate: {win_rate:.3f}")
    
    if "ally_survival_rate_mean" in custom_metrics:
        survival = custom_metrics["ally_survival_rate_mean"]
        print(f"Ally Survival Rate: {survival:.3f}")
    
    # 学习效率
    reward_efficiency = custom_metrics.get("reward_efficiency_mean", 0)
    print(f"Reward Efficiency: {reward_efficiency:.3f}")
    
    return {
        "cooperation_score": coop_rate,
        "behavioral_diversity": action_entropy,
        "learning_stability": reward_stability,
        "performance_balance": 1.0 - perf_inequality,  # 转换为正向指标
        "overall_efficiency": reward_efficiency
    }


if __name__ == "__main__":
    """
    使用示例：与MARLlib SMAC环境集成
    """
    
    # 注意：实际运行需要安装和配置MARLlib环境
    print("MARLlib Custom Metrics Integration Example")
    print("=" * 50)
    
    # 示例配置（实际使用时需要根据具体环境调整）
    example_config = {
        "env": "smac",
        "env_config": {"map_name": "3m"},
        "callbacks": MARLlibCustomMetricsCallback,
        "framework": "torch",
        "num_workers": 1,
        "train_batch_size": 2048,
    }
    
    print("Example configuration with custom metrics:")
    for key, value in example_config.items():
        if key != "callbacks":
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value.__name__}")
    
    print("\nThis callback will collect the following custom metrics:")
    metrics_list = [
        "avg_action_entropy - 智能体行为多样性",
        "cooperation_rate - 合作频率",
        "reward_stability - 奖励稳定性", 
        "performance_inequality - 个体表现差异",
        "battle_won - 战斗胜率 (SMAC环境)",
        "ally_survival_rate - 盟友存活率",
        "reward_efficiency - 奖励效率",
        "avg_reward_per_agent - 每智能体平均奖励",
        "episode_complexity - Episode复杂度"
    ]
    
    for metric in metrics_list:
        print(f"  - {metric}")
    
    print("\n要运行完整示例，请确保已安装MARLlib并配置相应环境。")

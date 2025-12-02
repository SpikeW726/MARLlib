#!/usr/bin/env python3
"""
自定义环境中使用Custom Metrics的完整示例
展示如何在MARLlib/Ray RLLib中收集环境特定的指标
"""

import numpy as np
import gym
from typing import Dict, Any
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch


class CustomMultiAgentEnv(MultiAgentEnv):
    """
    自定义多智能体环境，包含环境特定的指标收集
    """
    
    def __init__(self, env_config):
        super().__init__()
        self.num_agents = env_config.get("num_agents", 3)
        self.max_steps = env_config.get("max_steps", 100)
        
        # 环境状态
        self.step_count = 0
        self.agent_positions = {}
        self.collaboration_events = 0
        self.conflict_events = 0
        self.resource_collected = 0
        
        # 观察和动作空间
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(4)  # 上下左右
        
        # 环境特定的统计信息
        self.episode_stats = {
            "total_collaboration": 0,
            "total_conflicts": 0,
            "resource_efficiency": 0.0,
            "agent_distances": [],
            "successful_tasks": 0
        }
    
    def reset(self):
        """重置环境并初始化统计"""
        self.step_count = 0
        self.collaboration_events = 0
        self.conflict_events = 0
        self.resource_collected = 0
        
        # 重置智能体位置
        self.agent_positions = {
            f"agent_{i}": np.random.uniform(-1, 1, 2) 
            for i in range(self.num_agents)
        }
        
        # 重置episode统计
        self.episode_stats = {
            "total_collaboration": 0,
            "total_conflicts": 0,
            "resource_efficiency": 0.0,
            "agent_distances": [],
            "successful_tasks": 0
        }
        
        return self._get_observations()
    
    def step(self, action_dict):
        """执行一步并收集环境指标"""
        self.step_count += 1
        
        # 更新智能体位置（简化的移动逻辑）
        for agent_id, action in action_dict.items():
            pos = self.agent_positions[agent_id]
            if action == 0:  # 上
                pos[1] = np.clip(pos[1] + 0.1, -1, 1)
            elif action == 1:  # 下
                pos[1] = np.clip(pos[1] - 0.1, -1, 1)
            elif action == 2:  # 左
                pos[0] = np.clip(pos[0] - 0.1, -1, 1)
            elif action == 3:  # 右
                pos[0] = np.clip(pos[0] + 0.1, -1, 1)
        
        # 计算环境特定的事件和指标
        self._calculate_environment_events()
        
        # 生成奖励
        rewards = self._calculate_rewards()
        
        # 检查终止条件
        done = self.step_count >= self.max_steps
        dones = {"__all__": done}
        
        # 生成info，包含环境统计信息
        info = self._generate_info()
        
        return self._get_observations(), rewards, dones, info
    
    def _calculate_environment_events(self):
        """计算环境特定的事件（协作、冲突等）"""
        positions = list(self.agent_positions.values())
        
        # 计算智能体间距离
        distances = []
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist = np.linalg.norm(positions[i] - positions[j])
                distances.append(dist)
        
        self.episode_stats["agent_distances"].extend(distances)
        
        # 检测协作事件（智能体接近时）
        close_pairs = sum(1 for d in distances if d < 0.2)
        if close_pairs > 0:
            self.collaboration_events += close_pairs
            self.episode_stats["total_collaboration"] += close_pairs
        
        # 检测冲突事件（太过拥挤）
        if len(distances) > 0 and min(distances) < 0.1:
            self.conflict_events += 1
            self.episode_stats["total_conflicts"] += 1
        
        # 模拟资源收集
        for pos in positions:
            if np.linalg.norm(pos - np.array([0.5, 0.5])) < 0.15:
                self.resource_collected += 1
        
        # 计算资源效率
        if self.step_count > 0:
            self.episode_stats["resource_efficiency"] = self.resource_collected / self.step_count
    
    def _calculate_rewards(self):
        """计算奖励"""
        rewards = {}
        base_reward = 0.1
        
        for agent_id in self.agent_positions.keys():
            reward = base_reward
            
            # 协作奖励
            if self.collaboration_events > 0:
                reward += 0.5
            
            # 冲突惩罚
            if self.conflict_events > 0:
                reward -= 0.3
            
            # 资源收集奖励
            pos = self.agent_positions[agent_id]
            if np.linalg.norm(pos - np.array([0.5, 0.5])) < 0.15:
                reward += 1.0
                self.episode_stats["successful_tasks"] += 1
            
            rewards[agent_id] = reward
        
        return rewards
    
    def _get_observations(self):
        """生成观察"""
        obs = {}
        for agent_id, pos in self.agent_positions.items():
            # 简化的观察：位置 + 到目标的距离 + 到其他智能体的平均距离
            target_dist = np.linalg.norm(pos - np.array([0.5, 0.5]))
            
            other_positions = [p for aid, p in self.agent_positions.items() if aid != agent_id]
            avg_agent_dist = np.mean([np.linalg.norm(pos - p) for p in other_positions]) if other_positions else 0
            
            obs[agent_id] = np.array([pos[0], pos[1], target_dist, avg_agent_dist], dtype=np.float32)
        
        return obs
    
    def _generate_info(self):
        """生成包含环境统计的info字典"""
        return {
            "step_count": self.step_count,
            "collaboration_events": self.collaboration_events,
            "conflict_events": self.conflict_events,
            "resource_collected": self.resource_collected,
            "episode_stats": self.episode_stats.copy()
        }


class CustomEnvironmentCallbacks(DefaultCallbacks):
    """
    自定义回调类，用于收集环境特定的指标
    """
    
    def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv,
                         policies: Dict[str, Policy], episode: MultiAgentEpisode, 
                         env_index: int, **kwargs):
        """Episode开始时的初始化"""
        print(f"Episode {episode.episode_id} started at env_index {env_index}")
        
        # 初始化用户数据存储
        episode.user_data["collaboration_history"] = []
        episode.user_data["conflict_history"] = []
        episode.user_data["resource_efficiency_history"] = []
        episode.user_data["agent_distances_history"] = []
        episode.user_data["step_rewards"] = []
        
        # 初始化hist_data用于详细的历史记录
        episode.hist_data["collaboration_per_step"] = []
        episode.hist_data["conflicts_per_step"] = []
        episode.hist_data["distances_per_step"] = []
    
    def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv,
                        policies: Dict[str, Policy], episode: MultiAgentEpisode, 
                        env_index: int, **kwargs):
        """每一步后收集指标"""
        
        # 获取环境信息
        info = episode.last_info_for()
        if info and "episode_stats" in info:
            stats = info["episode_stats"]
            
            # 记录协作事件
            collaboration = info.get("collaboration_events", 0)
            episode.user_data["collaboration_history"].append(collaboration)
            episode.hist_data["collaboration_per_step"].append(collaboration)
            
            # 记录冲突事件
            conflicts = info.get("conflict_events", 0)
            episode.user_data["conflict_history"].append(conflicts)
            episode.hist_data["conflicts_per_step"].append(conflicts)
            
            # 记录资源效率
            efficiency = stats.get("resource_efficiency", 0.0)
            episode.user_data["resource_efficiency_history"].append(efficiency)
            
            # 记录智能体距离
            if stats.get("agent_distances"):
                avg_distance = np.mean(stats["agent_distances"])
                episode.user_data["agent_distances_history"].append(avg_distance)
                episode.hist_data["distances_per_step"].append(avg_distance)
        
        # 记录步骤奖励
        step_rewards = []
        for agent_id in episode.get_agents():
            reward = episode.agent_rewards.get((agent_id, "default_policy"), 0)
            step_rewards.append(reward)
        
        if step_rewards:
            episode.user_data["step_rewards"].append(np.mean(step_rewards))
    
    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode, 
                       env_index: int, **kwargs):
        """Episode结束时计算最终指标"""
        
        print(f"Episode {episode.episode_id} ended with length {episode.length}")
        
        # 计算整个episode的统计指标
        collaboration_history = episode.user_data.get("collaboration_history", [])
        conflict_history = episode.user_data.get("conflict_history", [])
        efficiency_history = episode.user_data.get("resource_efficiency_history", [])
        distance_history = episode.user_data.get("agent_distances_history", [])
        step_rewards = episode.user_data.get("step_rewards", [])
        
        # 自定义指标1：协作指标
        total_collaboration = sum(collaboration_history) if collaboration_history else 0
        avg_collaboration_per_step = total_collaboration / len(collaboration_history) if collaboration_history else 0
        episode.custom_metrics["total_collaboration_events"] = total_collaboration
        episode.custom_metrics["avg_collaboration_per_step"] = avg_collaboration_per_step
        
        # 自定义指标2：冲突指标
        total_conflicts = sum(conflict_history) if conflict_history else 0
        conflict_rate = total_conflicts / episode.length if episode.length > 0 else 0
        episode.custom_metrics["total_conflict_events"] = total_conflicts
        episode.custom_metrics["conflict_rate"] = conflict_rate
        
        # 自定义指标3：效率指标
        if efficiency_history:
            final_efficiency = efficiency_history[-1]
            avg_efficiency = np.mean(efficiency_history)
            episode.custom_metrics["final_resource_efficiency"] = final_efficiency
            episode.custom_metrics["avg_resource_efficiency"] = avg_efficiency
        
        # 自定义指标4：社交距离指标
        if distance_history:
            avg_agent_distance = np.mean(distance_history)
            min_distance = np.min(distance_history)
            max_distance = np.max(distance_history)
            episode.custom_metrics["avg_agent_distance"] = avg_agent_distance
            episode.custom_metrics["min_agent_distance"] = min_distance
            episode.custom_metrics["max_agent_distance"] = max_distance
        
        # 自定义指标5：奖励分析
        if step_rewards:
            total_episode_reward = sum(step_rewards)
            avg_step_reward = np.mean(step_rewards)
            reward_variance = np.var(step_rewards)
            episode.custom_metrics["avg_step_reward"] = avg_step_reward
            episode.custom_metrics["reward_variance"] = reward_variance
            episode.custom_metrics["reward_stability"] = 1.0 / (1.0 + reward_variance)  # 稳定性指标
        
        # 自定义指标6：协作效率比
        if total_collaboration > 0 and total_conflicts >= 0:
            collaboration_efficiency = total_collaboration / (total_collaboration + total_conflicts + 1)
            episode.custom_metrics["collaboration_efficiency"] = collaboration_efficiency
        
        # 自定义指标7：任务完成率
        info = episode.last_info_for()
        if info and "episode_stats" in info:
            successful_tasks = info["episode_stats"].get("successful_tasks", 0)
            task_completion_rate = successful_tasks / episode.length if episode.length > 0 else 0
            episode.custom_metrics["task_completion_rate"] = task_completion_rate
            episode.custom_metrics["total_successful_tasks"] = successful_tasks
        
        # 将详细的历史数据存储到hist_data用于后续分析
        episode.hist_data["final_collaboration_history"] = collaboration_history
        episode.hist_data["final_conflict_history"] = conflict_history
        episode.hist_data["final_efficiency_history"] = efficiency_history
        episode.hist_data["final_distance_history"] = distance_history
        episode.hist_data["final_reward_history"] = step_rewards
    
    def on_train_result(self, *, trainer, result: dict, **kwargs):
        """训练结果回调，可以添加训练级别的自定义指标"""
        
        # 从episode的custom_metrics中提取并计算训练级别的指标
        custom_metrics = result.get("custom_metrics", {})
        
        if custom_metrics:
            # 计算协作相关的训练指标
            collaboration_metrics = [k for k in custom_metrics.keys() if "collaboration" in k]
            if collaboration_metrics:
                avg_collaboration_across_episodes = np.mean([
                    custom_metrics.get(metric + "_mean", 0) for metric in collaboration_metrics
                ])
                result["training_collaboration_score"] = avg_collaboration_across_episodes
            
            # 计算整体环境表现评分
            efficiency_mean = custom_metrics.get("avg_resource_efficiency_mean", 0)
            collaboration_mean = custom_metrics.get("collaboration_efficiency_mean", 0)
            task_completion_mean = custom_metrics.get("task_completion_rate_mean", 0)
            
            # 综合评分
            overall_score = (efficiency_mean + collaboration_mean + task_completion_mean) / 3
            result["overall_environment_score"] = overall_score
        
        print(f"Training iteration completed. Overall score: {result.get('overall_environment_score', 'N/A')}")


if __name__ == "__main__":
    """
    使用示例
    """
    import ray
    from ray import tune
    from ray.rllib.agents.ppo import PPOTrainer
    
    # 初始化Ray
    ray.init()
    
    # 注册自定义环境
    from ray.tune.registry import register_env
    register_env("custom_multiagent_env", lambda config: CustomMultiAgentEnv(config))
    
    # 配置训练参数
    config = {
        "env": "custom_multiagent_env",
        "env_config": {
            "num_agents": 3,
            "max_steps": 50
        },
        "multiagent": {
            "policies": {
                "default_policy": (None, None, None, {})
            },
            "policy_mapping_fn": lambda agent_id, episode, worker, **kwargs: "default_policy"
        },
        "callbacks": CustomEnvironmentCallbacks,
        "framework": "torch",
        "num_workers": 1,
        "train_batch_size": 1024,
        "sgd_minibatch_size": 128,
        "num_sgd_iter": 10,
        "lr": 3e-4,
        "gamma": 0.99,
        "lambda": 0.95,
        "clip_param": 0.2,
        "num_gpus": 0
    }
    
    # 开始训练
    trainer = PPOTrainer(config=config)
    
    # 训练几个iteration并观察custom_metrics
    for i in range(5):
        result = trainer.train()
        
        print(f"\n=== Training Iteration {i+1} ===")
        print(f"Episode reward mean: {result['episode_reward_mean']:.3f}")
        print(f"Episode length mean: {result['episode_len_mean']:.3f}")
        
        # 打印custom_metrics
        custom_metrics = result.get('custom_metrics', {})
        if custom_metrics:
            print("\nCustom Metrics:")
            for metric, value in custom_metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  {metric}: {value:.3f}")
        
        # 打印环境特定指标
        if "overall_environment_score" in result:
            print(f"Overall Environment Score: {result['overall_environment_score']:.3f}")
    
    trainer.stop()
    ray.shutdown()

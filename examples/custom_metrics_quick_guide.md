# Custom Metrics 在 MARLlib 中的使用指南

## 概述

Custom Metrics 是 Ray RLLib 提供的强大系统，允许你在训练过程中收集和监控自定义指标。这些指标会自动集成到训练日志、TensorBoard 和实验追踪系统中。

## 快速开始

### 1. 基础 Callback 类

```python
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.evaluation import MultiAgentEpisode

class MyCustomCallback(DefaultCallbacks):
    def on_episode_end(self, *, episode: MultiAgentEpisode, **kwargs):
        # 在这里添加你的自定义指标
        episode.custom_metrics["my_metric"] = some_calculated_value
```

### 2. 在 MARLlib 配置中使用

```python
# 在训练配置中添加回调
config = {
    "env": "your_environment",
    "callbacks": MyCustomCallback,  # 添加自定义回调
    # ... 其他配置
}
```

## 详细使用方法

### Episode 生命周期中的指标收集

#### on_episode_start() - Episode 开始时
```python
def on_episode_start(self, *, episode: MultiAgentEpisode, **kwargs):
    # 初始化数据收集
    episode.user_data["my_data"] = []
    episode.user_data["step_count"] = 0
```

#### on_episode_step() - 每一步后
```python
def on_episode_step(self, *, episode: MultiAgentEpisode, **kwargs):
    # 收集步骤级数据
    step_data = extract_step_info(episode)
    episode.user_data["my_data"].append(step_data)
    episode.user_data["step_count"] += 1
```

#### on_episode_end() - Episode 结束时
```python
def on_episode_end(self, *, episode: MultiAgentEpisode, **kwargs):
    # 计算最终指标
    collected_data = episode.user_data["my_data"]
    
    # 添加自定义指标
    episode.custom_metrics["avg_metric"] = np.mean(collected_data)
    episode.custom_metrics["total_steps"] = episode.user_data["step_count"]
    episode.custom_metrics["data_variance"] = np.var(collected_data)
```

### 环境特定指标示例

#### 多智能体合作指标
```python
def on_episode_end(self, *, episode: MultiAgentEpisode, **kwargs):
    # 计算智能体间距离
    positions = get_agent_positions(episode)
    distances = calculate_distances(positions)
    
    # 合作事件检测
    close_interactions = sum(1 for d in distances if d < threshold)
    cooperation_rate = close_interactions / episode.length
    
    episode.custom_metrics["cooperation_rate"] = cooperation_rate
    episode.custom_metrics["avg_agent_distance"] = np.mean(distances)
```

#### 游戏环境指标 (SMAC)
```python
def on_episode_end(self, *, episode: MultiAgentEpisode, **kwargs):
    info = episode.last_info_for()
    
    if "battle_won" in info:
        episode.custom_metrics["battle_won"] = info["battle_won"]
    if "dead_allies" in info:
        num_agents = len(episode.get_agents())
        survival_rate = 1.0 - (info["dead_allies"] / num_agents)
        episode.custom_metrics["ally_survival_rate"] = survival_rate
```

#### 奖励分析指标
```python
def on_episode_end(self, *, episode: MultiAgentEpisode, **kwargs):
    # 分析个体奖励分布
    agent_rewards = {}
    for agent_id in episode.get_agents():
        if (agent_id, "default_policy") in episode.agent_rewards:
            agent_rewards[agent_id] = episode.agent_rewards[(agent_id, "default_policy")]
    
    if agent_rewards:
        rewards_list = list(agent_rewards.values())
        episode.custom_metrics["reward_std"] = np.std(rewards_list)
        episode.custom_metrics["reward_fairness"] = 1.0 - (np.std(rewards_list) / np.mean(rewards_list))
        episode.custom_metrics["best_agent_reward"] = max(rewards_list)
        episode.custom_metrics["worst_agent_reward"] = min(rewards_list)
```

## 指标类别建议

### 1. 合作与协调指标
- `cooperation_rate`: 合作事件频率
- `coordination_score`: 协调行为评分
- `team_coherence`: 团队一致性

### 2. 行为多样性指标
- `action_entropy`: 动作熵
- `strategy_diversity`: 策略多样性
- `exploration_rate`: 探索率

### 3. 性能与效率指标
- `task_completion_rate`: 任务完成率
- `resource_efficiency`: 资源利用效率
- `time_efficiency`: 时间效率

### 4. 公平性指标
- `reward_inequality`: 奖励不平等度
- `performance_balance`: 性能平衡性
- `participation_equity`: 参与公平性

### 5. 环境适应性指标
- `environment_adaptation`: 环境适应度
- `robustness_score`: 鲁棒性评分
- `generalization_ability`: 泛化能力

## 指标访问和监控

### 在训练过程中访问
```python
def on_train_result(self, *, trainer, result: dict, **kwargs):
    custom_metrics = result.get("custom_metrics", {})
    
    # 访问特定指标
    cooperation = custom_metrics.get("cooperation_rate_mean", 0)
    print(f"Current cooperation rate: {cooperation:.3f}")
    
    # 添加训练级别的指标
    result["training_custom_score"] = calculate_training_score(custom_metrics)
```

### 从训练结果中提取
```python
# 训练循环中
for i in range(num_iterations):
    result = trainer.train()
    
    # 提取自定义指标
    custom_metrics = result["custom_metrics"]
    
    # 记录关键指标
    cooperation_rate = custom_metrics.get("cooperation_rate_mean", 0)
    task_completion = custom_metrics.get("task_completion_rate_mean", 0)
    
    print(f"Iteration {i}: Cooperation={cooperation_rate:.3f}, "
          f"Task Completion={task_completion:.3f}")
```

## 最佳实践

### 1. 指标命名规范
- 使用描述性名称：`cooperation_rate` 而不是 `cr`
- 包含单位信息：`distance_meters`, `time_seconds`
- 使用一致的后缀：`_rate`, `_score`, `_count`

### 2. 性能考虑
- 避免在每步计算复杂指标
- 使用 `user_data` 收集数据，在 episode 结束时计算
- 限制指标数量以避免影响训练速度

### 3. 数据类型
- 确保指标值是数值类型 (int, float)
- 避免 NaN 或无穷大值
- 使用合理的数值范围

### 4. 调试和验证
```python
def on_episode_end(self, *, episode: MultiAgentEpisode, **kwargs):
    # 添加验证
    metric_value = calculate_metric(episode)
    
    # 确保数值有效
    if not np.isnan(metric_value) and np.isfinite(metric_value):
        episode.custom_metrics["my_metric"] = metric_value
    else:
        print(f"Warning: Invalid metric value: {metric_value}")
```

## 与现有系统集成

### TensorBoard 集成
自定义指标会自动显示在 TensorBoard 中：
```bash
tensorboard --logdir ~/ray_results/your_experiment
```

### 实验追踪集成
与 Weights & Biases、MLflow 等实验追踪工具兼容：
```python
config = {
    "callbacks": MyCustomCallback,
    "logger_config": {
        "wandb": {
            "project": "my_marl_project",
            "api_key": "your_api_key"
        }
    }
}
```

## 故障排除

### 常见问题

1. **指标不显示**
   - 检查指标名称和数据类型
   - 确保在 `episode.custom_metrics` 中正确设置

2. **性能影响**
   - 减少计算复杂度
   - 使用采样而不是计算所有数据

3. **内存问题**
   - 避免在 `user_data` 中存储大量数据
   - 及时清理不需要的数据

### 调试技巧
```python
def on_episode_end(self, *, episode: MultiAgentEpisode, **kwargs):
    print(f"Episode {episode.episode_id} ended:")
    print(f"  Length: {episode.length}")
    print(f"  Agents: {episode.get_agents()}")
    print(f"  User data keys: {episode.user_data.keys()}")
    
    # 添加调试指标
    episode.custom_metrics["debug_episode_length"] = episode.length
    episode.custom_metrics["debug_num_agents"] = len(episode.get_agents())
```

## 完整示例

参考以下完整示例文件：
- `custom_env_with_metrics.py`: 完整的自定义环境集成示例
- `marllib_custom_metrics_example.py`: MARLlib 特定的集成示例  
- `metrics_monitoring_example.py`: 指标监控和可视化示例

这些示例展示了从基础指标收集到高级分析和可视化的完整工作流程。

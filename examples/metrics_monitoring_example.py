#!/usr/bin/env python3
"""
Custom Metrics监控和可视化示例
展示如何访问、分析和可视化收集的自定义指标
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import os
from datetime import datetime


class MetricsMonitor:
    """
    自定义指标监控器
    """
    
    def __init__(self, log_dir="./custom_metrics_logs"):
        self.log_dir = log_dir
        self.metrics_history = []
        self.episode_metrics = []
        
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
        
        # 初始化监控的指标类别
        self.cooperation_metrics = []
        self.performance_metrics = []
        self.behavioral_metrics = []
        self.environment_metrics = []
    
    def log_training_result(self, iteration: int, result: Dict[str, Any]):
        """
        记录训练结果中的自定义指标
        """
        timestamp = datetime.now().isoformat()
        
        # 提取基础训练指标
        basic_metrics = {
            "iteration": iteration,
            "timestamp": timestamp,
            "episode_reward_mean": result.get("episode_reward_mean", 0),
            "episode_len_mean": result.get("episode_len_mean", 0),
            "episodes_this_iter": result.get("episodes_this_iter", 0),
            "timesteps_total": result.get("timesteps_total", 0)
        }
        
        # 提取自定义指标
        custom_metrics = result.get("custom_metrics", {})
        
        # 分类整理自定义指标
        categorized_metrics = self._categorize_metrics(custom_metrics)
        
        # 合并所有指标
        full_metrics = {
            **basic_metrics,
            **categorized_metrics,
            "raw_custom_metrics": custom_metrics
        }
        
        self.metrics_history.append(full_metrics)
        
        # 保存到文件
        self._save_metrics_to_file(full_metrics)
        
        # 打印关键指标
        self._print_key_metrics(iteration, full_metrics)
    
    def _categorize_metrics(self, custom_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        将自定义指标按类别分组
        """
        cooperation = {}
        performance = {}
        behavioral = {}
        environment = {}
        
        for metric_name, value in custom_metrics.items():
            if isinstance(value, (int, float)):
                # 合作相关指标
                if any(keyword in metric_name.lower() for keyword in 
                      ["cooperation", "collaboration", "team", "coordination"]):
                    cooperation[metric_name] = value
                
                # 性能相关指标
                elif any(keyword in metric_name.lower() for keyword in 
                        ["reward", "performance", "efficiency", "score", "win", "success"]):
                    performance[metric_name] = value
                
                # 行为相关指标
                elif any(keyword in metric_name.lower() for keyword in 
                        ["action", "entropy", "diversity", "behavior", "strategy"]):
                    behavioral[metric_name] = value
                
                # 环境相关指标
                elif any(keyword in metric_name.lower() for keyword in 
                        ["battle", "enemy", "ally", "survival", "elimination", "damage"]):
                    environment[metric_name] = value
        
        return {
            "cooperation_metrics": cooperation,
            "performance_metrics": performance,
            "behavioral_metrics": behavioral,
            "environment_metrics": environment
        }
    
    def _save_metrics_to_file(self, metrics: Dict[str, Any]):
        """
        保存指标到JSON文件
        """
        filename = os.path.join(self.log_dir, f"metrics_iter_{metrics['iteration']}.json")
        
        # 移除不能序列化的项目
        serializable_metrics = {k: v for k, v in metrics.items() 
                              if k != "raw_custom_metrics" or isinstance(v, (dict, list, str, int, float, bool, type(None)))}
        
        with open(filename, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
    
    def _print_key_metrics(self, iteration: int, metrics: Dict[str, Any]):
        """
        打印关键指标
        """
        print(f"\n=== Training Iteration {iteration} ===")
        print(f"Episode Reward Mean: {metrics['episode_reward_mean']:.3f}")
        print(f"Episode Length Mean: {metrics['episode_len_mean']:.1f}")
        print(f"Episodes This Iter: {metrics['episodes_this_iter']}")
        
        # 打印合作指标
        coop_metrics = metrics.get("cooperation_metrics", {})
        if coop_metrics:
            print("\nCooperation Metrics:")
            for name, value in coop_metrics.items():
                if "_mean" in name:
                    print(f"  {name}: {value:.3f}")
        
        # 打印性能指标
        perf_metrics = metrics.get("performance_metrics", {})
        if perf_metrics:
            print("\nPerformance Metrics:")
            for name, value in perf_metrics.items():
                if "_mean" in name:
                    print(f"  {name}: {value:.3f}")
        
        # 打印行为指标
        behav_metrics = metrics.get("behavioral_metrics", {})
        if behav_metrics:
            print("\nBehavioral Metrics:")
            for name, value in behav_metrics.items():
                if "_mean" in name:
                    print(f"  {name}: {value:.3f}")
    
    def generate_metrics_report(self, save_plots=True):
        """
        生成指标分析报告
        """
        if not self.metrics_history:
            print("No metrics data available for report generation.")
            return
        
        print("\n" + "="*60)
        print("CUSTOM METRICS ANALYSIS REPORT")
        print("="*60)
        
        # 基础统计
        total_iterations = len(self.metrics_history)
        print(f"Total Training Iterations: {total_iterations}")
        
        if total_iterations > 0:
            first_metrics = self.metrics_history[0]
            last_metrics = self.metrics_history[-1]
            
            print(f"Training Duration: {first_metrics['timestamp']} to {last_metrics['timestamp']}")
            print(f"Total Timesteps: {last_metrics['timesteps_total']}")
        
        # 奖励趋势分析
        self._analyze_reward_trends()
        
        # 合作指标分析
        self._analyze_cooperation_trends()
        
        # 行为多样性分析
        self._analyze_behavioral_trends()
        
        # 性能稳定性分析
        self._analyze_performance_stability()
        
        if save_plots:
            self._generate_plots()
    
    def _analyze_reward_trends(self):
        """
        分析奖励趋势
        """
        print(f"\n{'-'*40}")
        print("REWARD TREND ANALYSIS")
        print(f"{'-'*40}")
        
        rewards = [m["episode_reward_mean"] for m in self.metrics_history]
        
        if len(rewards) > 1:
            initial_reward = rewards[0]
            final_reward = rewards[-1]
            improvement = final_reward - initial_reward
            improvement_pct = (improvement / abs(initial_reward)) * 100 if initial_reward != 0 else 0
            
            print(f"Initial Average Reward: {initial_reward:.3f}")
            print(f"Final Average Reward: {final_reward:.3f}")
            print(f"Total Improvement: {improvement:.3f} ({improvement_pct:.1f}%)")
            
            # 计算趋势
            iterations = list(range(len(rewards)))
            trend_slope = np.polyfit(iterations, rewards, 1)[0]
            print(f"Reward Trend Slope: {trend_slope:.6f}")
            
            if trend_slope > 0:
                print("✓ Positive learning trend detected")
            elif trend_slope < -0.001:
                print("⚠ Negative learning trend detected")
            else:
                print("→ Stable reward pattern")
    
    def _analyze_cooperation_trends(self):
        """
        分析合作指标趋势
        """
        print(f"\n{'-'*40}")
        print("COOPERATION ANALYSIS")
        print(f"{'-'*40}")
        
        cooperation_data = []
        for metrics in self.metrics_history:
            coop_metrics = metrics.get("cooperation_metrics", {})
            # 查找合作率指标
            coop_rate = None
            for name, value in coop_metrics.items():
                if "cooperation_rate" in name and "_mean" in name:
                    coop_rate = value
                    break
            cooperation_data.append(coop_rate)
        
        # 过滤None值
        valid_cooperation = [c for c in cooperation_data if c is not None]
        
        if valid_cooperation:
            avg_cooperation = np.mean(valid_cooperation)
            cooperation_std = np.std(valid_cooperation)
            
            print(f"Average Cooperation Rate: {avg_cooperation:.3f}")
            print(f"Cooperation Stability (1-std): {1-cooperation_std:.3f}")
            
            if avg_cooperation > 0.5:
                print("✓ High cooperation observed")
            elif avg_cooperation > 0.2:
                print("→ Moderate cooperation observed")
            else:
                print("⚠ Low cooperation observed")
        else:
            print("No cooperation metrics available")
    
    def _analyze_behavioral_trends(self):
        """
        分析行为多样性趋势
        """
        print(f"\n{'-'*40}")
        print("BEHAVIORAL DIVERSITY ANALYSIS")
        print(f"{'-'*40}")
        
        entropy_data = []
        diversity_data = []
        
        for metrics in self.metrics_history:
            behav_metrics = metrics.get("behavioral_metrics", {})
            
            # 查找行为熵
            for name, value in behav_metrics.items():
                if "entropy" in name and "_mean" in name:
                    entropy_data.append(value)
                    break
            else:
                entropy_data.append(None)
            
            # 查找多样性指标
            for name, value in behav_metrics.items():
                if "diversity" in name and "_mean" in name:
                    diversity_data.append(value)
                    break
            else:
                diversity_data.append(None)
        
        # 分析行为熵
        valid_entropy = [e for e in entropy_data if e is not None]
        if valid_entropy:
            avg_entropy = np.mean(valid_entropy)
            print(f"Average Action Entropy: {avg_entropy:.3f}")
            
            if avg_entropy > 1.0:
                print("✓ High behavioral diversity")
            elif avg_entropy > 0.5:
                print("→ Moderate behavioral diversity")
            else:
                print("⚠ Low behavioral diversity - possible over-convergence")
        
        # 分析多样性趋势
        valid_diversity = [d for d in diversity_data if d is not None]
        if valid_diversity:
            avg_diversity = np.mean(valid_diversity)
            print(f"Average Behavioral Diversity: {avg_diversity:.3f}")
    
    def _analyze_performance_stability(self):
        """
        分析性能稳定性
        """
        print(f"\n{'-'*40}")
        print("PERFORMANCE STABILITY ANALYSIS")
        print(f"{'-'*40}")
        
        stability_data = []
        inequality_data = []
        
        for metrics in self.metrics_history:
            perf_metrics = metrics.get("performance_metrics", {})
            
            # 查找稳定性指标
            for name, value in perf_metrics.items():
                if "stability" in name and "_mean" in name:
                    stability_data.append(value)
                    break
            else:
                stability_data.append(None)
            
            # 查找不平等指标
            for name, value in perf_metrics.items():
                if "inequality" in name and "_mean" in name:
                    inequality_data.append(value)
                    break
            else:
                inequality_data.append(None)
        
        # 分析稳定性
        valid_stability = [s for s in stability_data if s is not None]
        if valid_stability:
            avg_stability = np.mean(valid_stability)
            print(f"Average Performance Stability: {avg_stability:.3f}")
            
            if avg_stability > 0.8:
                print("✓ High performance stability")
            elif avg_stability > 0.6:
                print("→ Moderate performance stability")
            else:
                print("⚠ Low performance stability")
        
        # 分析不平等
        valid_inequality = [i for i in inequality_data if i is not None]
        if valid_inequality:
            avg_inequality = np.mean(valid_inequality)
            print(f"Average Performance Inequality: {avg_inequality:.3f}")
            
            if avg_inequality < 0.2:
                print("✓ Fair performance distribution")
            elif avg_inequality < 0.5:
                print("→ Moderate performance differences")
            else:
                print("⚠ High performance inequality between agents")
    
    def _generate_plots(self):
        """
        生成可视化图表
        """
        if len(self.metrics_history) < 2:
            print("Insufficient data for plotting.")
            return
        
        print(f"\n{'-'*40}")
        print("GENERATING VISUALIZATION PLOTS")
        print(f"{'-'*40}")
        
        try:
            # 创建子图
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle("Custom Metrics Analysis", fontsize=16)
            
            iterations = [m["iteration"] for m in self.metrics_history]
            
            # 1. 奖励趋势
            rewards = [m["episode_reward_mean"] for m in self.metrics_history]
            axes[0, 0].plot(iterations, rewards, 'b-', linewidth=2)
            axes[0, 0].set_title("Episode Reward Trend")
            axes[0, 0].set_xlabel("Training Iteration")
            axes[0, 0].set_ylabel("Average Episode Reward")
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. 合作率趋势
            cooperation_rates = []
            for m in self.metrics_history:
                coop_metrics = m.get("cooperation_metrics", {})
                rate = None
                for name, value in coop_metrics.items():
                    if "cooperation_rate" in name and "_mean" in name:
                        rate = value
                        break
                cooperation_rates.append(rate if rate is not None else 0)
            
            axes[0, 1].plot(iterations, cooperation_rates, 'g-', linewidth=2)
            axes[0, 1].set_title("Cooperation Rate Trend")
            axes[0, 1].set_xlabel("Training Iteration")
            axes[0, 1].set_ylabel("Cooperation Rate")
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. 行为熵趋势
            entropies = []
            for m in self.metrics_history:
                behav_metrics = m.get("behavioral_metrics", {})
                entropy = None
                for name, value in behav_metrics.items():
                    if "entropy" in name and "_mean" in name:
                        entropy = value
                        break
                entropies.append(entropy if entropy is not None else 0)
            
            axes[1, 0].plot(iterations, entropies, 'r-', linewidth=2)
            axes[1, 0].set_title("Behavioral Entropy Trend")
            axes[1, 0].set_xlabel("Training Iteration")
            axes[1, 0].set_ylabel("Action Entropy")
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. 性能稳定性
            stabilities = []
            for m in self.metrics_history:
                perf_metrics = m.get("performance_metrics", {})
                stability = None
                for name, value in perf_metrics.items():
                    if "stability" in name and "_mean" in name:
                        stability = value
                        break
                stabilities.append(stability if stability is not None else 0)
            
            axes[1, 1].plot(iterations, stabilities, 'm-', linewidth=2)
            axes[1, 1].set_title("Performance Stability Trend")
            axes[1, 1].set_xlabel("Training Iteration")
            axes[1, 1].set_ylabel("Stability Score")
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存图表
            plot_filename = os.path.join(self.log_dir, f"metrics_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"Plots saved to: {plot_filename}")
            
            # 显示图表（如果在支持的环境中）
            try:
                plt.show()
            except:
                print("Plot display not available in current environment")
                
        except Exception as e:
            print(f"Error generating plots: {e}")
        finally:
            plt.close('all')
    
    def export_summary_report(self):
        """
        导出汇总报告
        """
        if not self.metrics_history:
            print("No data available for export.")
            return
        
        # 创建汇总报告
        summary = {
            "training_summary": {
                "total_iterations": len(self.metrics_history),
                "start_time": self.metrics_history[0]["timestamp"],
                "end_time": self.metrics_history[-1]["timestamp"],
                "total_timesteps": self.metrics_history[-1]["timesteps_total"]
            },
            "performance_summary": {},
            "cooperation_summary": {},
            "behavioral_summary": {}
        }
        
        # 计算各类指标的汇总统计
        rewards = [m["episode_reward_mean"] for m in self.metrics_history]
        summary["performance_summary"] = {
            "initial_reward": rewards[0],
            "final_reward": rewards[-1],
            "max_reward": max(rewards),
            "min_reward": min(rewards),
            "avg_reward": np.mean(rewards),
            "reward_improvement": rewards[-1] - rewards[0]
        }
        
        # 保存汇总报告
        summary_filename = os.path.join(self.log_dir, f"summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(summary_filename, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Summary report saved to: {summary_filename}")
        return summary


# 使用示例
def example_usage():
    """
    使用示例
    """
    print("Custom Metrics Monitoring Example")
    print("=" * 50)
    
    # 创建监控器
    monitor = MetricsMonitor("./example_metrics_logs")
    
    # 模拟训练结果数据
    for iteration in range(10):
        # 模拟训练结果
        mock_result = {
            "iteration": iteration,
            "episode_reward_mean": 50 + iteration * 5 + np.random.normal(0, 2),
            "episode_len_mean": 100 + np.random.normal(0, 5),
            "episodes_this_iter": 20,
            "timesteps_total": iteration * 2000,
            "custom_metrics": {
                "cooperation_rate_mean": 0.3 + iteration * 0.05 + np.random.normal(0, 0.02),
                "avg_action_entropy_mean": 1.2 - iteration * 0.02 + np.random.normal(0, 0.05),
                "reward_stability_mean": 0.7 + iteration * 0.02 + np.random.normal(0, 0.01),
                "performance_inequality_mean": 0.4 - iteration * 0.01 + np.random.normal(0, 0.02),
                "battle_won_mean": 0.2 + iteration * 0.08 + np.random.normal(0, 0.03),
                "ally_survival_rate_mean": 0.6 + iteration * 0.03 + np.random.normal(0, 0.02)
            }
        }
        
        # 记录指标
        monitor.log_training_result(iteration, mock_result)
    
    # 生成分析报告
    monitor.generate_metrics_report(save_plots=False)  # 不保存图表以避免依赖问题
    
    # 导出汇总报告
    summary = monitor.export_summary_report()
    
    print("\n" + "=" * 50)
    print("Monitoring example completed successfully!")
    print("Check the './example_metrics_logs' directory for generated files.")


if __name__ == "__main__":
    example_usage()

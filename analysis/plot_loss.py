import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List


def plot_classic_curves(history: Dict[str, List[float]], exp_name: str = "default"):
    """
    绘制经典的训练指标下降/上升曲线。
    [修复版]：自动处理 Logit 数据的可视化，不修改模型架构。
    """
    # 1. 路径锚定
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(current_dir, "pic", exp_name)
    os.makedirs(save_dir, exist_ok=True)

    sns.set_theme(style="whitegrid")

    # 获取步数，如果没有 step 键则自动生成
    if 'step' in history:
        steps = history['step']
    else:
        # 假设任意一个存在的 list 长度即为 step 数
        key = next(iter(history))
        steps = range(len(history[key]))

    # ==========================
    # 图表 1: 损失函数分解 (Loss Decomposition)
    # ==========================
    plt.figure(figsize=(10, 6))

    if 'total_loss' in history:
        sns.lineplot(x=steps, y=history['total_loss'], label='Total Loss', color='black', linewidth=2)

    if 'main_loss' in history:
        sns.lineplot(x=steps, y=history['main_loss'], label='Task Loss (CE)', color='blue', alpha=0.6, linestyle='--')

    # [优化] Conflict Loss 通常数值较小，单独画或保持现状
    if 'conflict_loss' in history:
        sns.lineplot(x=steps, y=history['conflict_loss'], label='Conflict Penalty', color='red', alpha=0.8)

    # [优化] Reg Loss 可能也是 Logit 的 L1，数值可能较大或较小
    if 'reg_loss' in history:
        sns.lineplot(x=steps, y=history['reg_loss'], label='Reg Loss', color='green', alpha=0.5, linestyle=':')

    plt.title(f"Training Loss Curves - {exp_name}", fontsize=14)
    plt.xlabel("Steps", fontsize=12)
    plt.ylabel("Loss Value", fontsize=12)
    plt.legend()
    plt.tight_layout()

    save_path_loss = os.path.join(save_dir, "loss_curve.png")
    plt.savefig(save_path_loss, dpi=300)
    plt.close()
    print(f"[Plotter] Loss curve saved to: {save_path_loss}")

    # ==========================
    # 图表 2: 系统指标 (Metrics) - 核心修复区
    # ==========================
    metrics_to_plot = {}

    # 1. 任务准确率 (通常本身就是 0-1)
    if 'task_acc' in history:
        metrics_to_plot['Task Accuracy'] = history['task_acc']

    # 2. Alpha 稀疏度 (关键修复)
    if 'alpha_sparsity' in history:
        raw_values = np.array(history['alpha_sparsity'])

        # [智能检测]：如果数据范围超出 [0, 1] 或者包含负数，说明是 Logit
        if raw_values.min() < 0 or raw_values.max() > 1.0:
            print(
                f"[Plotter] Detected Logits in 'alpha_sparsity' (Min:{raw_values.min():.2f}, Max:{raw_values.max():.2f}). Applying Sigmoid for visualization.")
            # 手动 Sigmoid: 1 / (1 + e^-x)
            # 这样画出来的就是 0-1 之间的"连接概率"，而不是 Logit 值
            transformed_values = 1 / (1 + np.exp(-raw_values))
            metrics_to_plot['Avg Connection Prob'] = transformed_values.tolist()
        else:
            # 如果已经是概率了，直接用
            metrics_to_plot['Alpha Sparsity'] = raw_values

    if metrics_to_plot:
        plt.figure(figsize=(10, 6))
        for name, values in metrics_to_plot.items():
            sns.lineplot(x=steps, y=values, label=name, linewidth=2)

        plt.title(f"System Metrics - {exp_name}", fontsize=14)
        plt.xlabel("Steps", fontsize=12)
        plt.ylabel("Value (Probability 0-1)", fontsize=12)

        # [安全限制] 现在所有数据都被转到了 0-1，所以这个限制是安全的
        plt.ylim(-0.05, 1.05)

        plt.legend()
        plt.tight_layout()

        save_path_metrics = os.path.join(save_dir, "metrics_curve.png")
        plt.savefig(save_path_metrics, dpi=300)
        plt.close()
        print(f"[Plotter] Metrics curve saved to: {save_path_metrics}")
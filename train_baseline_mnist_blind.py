import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

# === 复用项目模块 ===
from data.mixed_tasks import get_mixed_task_loaders
from analysis.visualizer import SystemMonitor
from analysis.plot_loss import plot_classic_curves
# 复用 model.py 的组件，确保注意力机制一致
from models.model import GroupedQueryAttention, RMSNorm, precompute_freqs_cis


# === 1. 标准 MoE 层 (无冲突，无共享基座) ===
# 保持完全一致，逻辑支持 None 输入
class StandardMoELayer(nn.Module):
    def __init__(self, d_model, num_experts, top_k, num_tasks, d_task_embed):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # 标准 FFN 专家 (独立权重)
        d_ffn = 32  # 对齐 CDSP 参数量
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.SiLU(),
                nn.Linear(d_ffn, d_model)
            ) for _ in range(num_experts)
        ])

        self.task_embedding = nn.Embedding(num_tasks, d_task_embed)
        self.gate = nn.Linear(d_model + d_task_embed, num_experts)

    def forward(self, x, task_id):
        batch_size, seq_len, _ = x.shape
        x_norm = F.layer_norm(x, x.shape[1:])

        # === 核心逻辑：如果 task_id 为 None，生成全 0 向量 ===
        if task_id is not None:
            t_emb = self.task_embedding(task_id).unsqueeze(1).expand(-1, seq_len, -1)
        else:
            # 模拟 DeepSeek/通用 MoE：没有 Task ID，只有 Content
            d_emb = self.task_embedding.embedding_dim
            t_emb = torch.zeros(batch_size, seq_len, d_emb, device=x.device)

        gate_input = torch.cat([x_norm, t_emb], dim=-1)
        logits = self.gate(gate_input)

        topk_probs, topk_indices = torch.topk(logits, self.top_k, dim=-1)
        topk_probs = F.softmax(topk_probs, dim=-1)

        routing_snapshot = {
            'indices': topk_indices.detach(),
            'task_ids': task_id.detach() if task_id is not None else None
        }

        flat_x = x.view(-1, x.shape[-1])
        final_output = torch.zeros_like(flat_x)
        expert_usage = torch.zeros(self.num_experts, device=x.device)

        for k in range(self.top_k):
            indices = topk_indices[:, :, k].view(-1)
            probs = topk_probs[:, :, k].view(-1, 1)
            for e_id in range(self.num_experts):
                mask = (indices == e_id)
                if mask.any():
                    expert_usage[e_id] += mask.sum()
                    expert_out = self.experts[e_id](flat_x[mask])
                    final_output[mask] += expert_out * probs[mask]

        final_output = final_output.view(batch_size, seq_len, -1)

        usage_mean = expert_usage.mean() + 1e-6
        usage_std = expert_usage.std()
        lb_loss = (usage_std / usage_mean) ** 2  # 标准负载均衡损失

        return final_output, lb_loss, routing_snapshot


# === 2. Baseline Block (Transformer 结构) ===
class BaselineBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 1. Attention
        self.norm1 = RMSNorm(config.d_model)
        self.attention = GroupedQueryAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads
        )

        # 2. Standard MoE
        self.norm2 = RMSNorm(config.d_model)
        self.moe = StandardMoELayer(
            d_model=config.d_model,
            num_experts=config.num_experts,
            top_k=config.moe_top_k,
            num_tasks=config.num_tasks,
            d_task_embed=16
        )

    def forward(self, x, freqs_cis, mask, task_id):
        # Attention Path
        h = x + self.attention(self.norm1(x), freqs_cis, mask)

        # MoE Path
        h_moe, aux_loss, snap = self.moe(self.norm2(h), task_id)
        out = h + h_moe

        return out, aux_loss, snap


# === 3. Baseline Model ===
class BaselineModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_emb = nn.Linear(config.input_dim, config.d_model)
        nn.init.xavier_normal_(self.token_emb.weight)

        self.freqs_cis = precompute_freqs_cis(
            config.d_model // config.n_heads,
            config.max_seq_len * 2
        )

        self.layers = nn.ModuleList([
            BaselineBlock(config) for _ in range(config.n_layers)
        ])

        self.norm_f = RMSNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, x, task_id):
        h = self.token_emb(x)
        seq_len = h.shape[1]
        freqs_cis = self.freqs_cis[:seq_len].to(h.device)
        mask = torch.full((seq_len, seq_len), float("-inf"), device=h.device)
        mask = torch.triu(mask, diagonal=1)

        total_aux_loss = 0
        snapshots = []

        for layer in self.layers:
            h, aux, snap = layer(h, freqs_cis, mask, task_id)
            total_aux_loss += aux
            snapshots.append(snap)

        h = self.norm_f(h)
        logits = self.head(h)
        logits = logits.mean(dim=1)
        return logits, total_aux_loss, snapshots[0]


# === 4. 配置  ===
class BaselineBlindConfig:
    exp_name = "baseline_mnist_3task_blind_train"

    epochs = 10
    batch_size = 64
    learning_rate = 0.005
    patch_size = 4
    input_dim = 16

    d_model = 64
    num_experts = 8
    num_tasks = 3
    vocab_size = 10
    n_layers = 2
    moe_top_k = 2
    n_heads = 4
    n_kv_heads = 2
    max_seq_len = 49


def patchify_images(images, patch_size=4):
    B, C, H, W = images.shape
    patches = F.unfold(images, kernel_size=patch_size, stride=patch_size)
    patches = patches.transpose(1, 2)
    return patches


# === 验证函数 ===
def validate_final_blind(model, cfg):
    """
    验证函数：因为训练就是盲的，所以这里的逻辑其实和训练一样。
    主要是为了画一张最终的高清热力图。
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    device = next(model.parameters()).device
    model.eval()

    _, test_loader = get_mixed_task_loaders(batch_size=100)
    routing_stats = np.zeros((cfg.num_tasks, cfg.num_experts))

    print("\n=== Verifying Final Blind Model Performance ===")

    with torch.no_grad():
        for images, real_task_ids, _ in tqdm(test_loader, desc="Final Eval"):
            images = images.to(device)
            real_task_ids = real_task_ids.to(device)
            inputs = patchify_images(images, cfg.patch_size)

            # [关键] 保持 None，虽然模型也没见过别的
            logits, _, snapshot = model(inputs, task_id=None)

            # 统计
            topk_indices = snapshot['indices']
            B, S, K = topk_indices.shape
            flat_indices = topk_indices.view(-1).cpu().numpy()
            expanded_real_tasks = real_task_ids.view(B, 1, 1).expand(B, S, K).contiguous().view(-1).cpu().numpy()

            np.add.at(routing_stats, (expanded_real_tasks, flat_indices), 1)

    # 绘图
    row_sums = routing_stats.sum(axis=1, keepdims=True) + 1e-6
    norm_stats = routing_stats / row_sums

    plt.figure(figsize=(10, 6))
    # 使用灰色/黑色系，表示这是"纯黑盒/盲盒"基线
    sns.heatmap(norm_stats, annot=True, fmt=".2f", cmap="Greys",
                xticklabels=[f"E{i}" for i in range(cfg.num_experts)],
                yticklabels=[f"Real T{i}" for i in range(cfg.num_tasks)])

    plt.title("Pure Blind Training Baseline (DeepSeek Style)\nInput: Content Only (No Task IDs)")
    plt.xlabel("Selected Expert")
    plt.ylabel("Ground Truth Task")
    plt.tight_layout()

    save_path = f"blind_train_result.png"
    plt.savefig(save_path)
    print(f"Final heatmap saved to '{save_path}'")


# === 主训练循环 (加后缀) ===
def train_baseline_blind():
    cfg = BaselineBlindConfig()  # 使用新配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Training Pure Blind Baseline (Content-Only MoE) on {device} ===")

    train_loader, test_loader = get_mixed_task_loaders(batch_size=cfg.batch_size)
    model = BaselineModel(cfg).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    monitor = SystemMonitor(exp_name=cfg.exp_name)

    global_step = 0

    for epoch in range(cfg.epochs):
        model.train()
        epoch_routing_stats = np.zeros((cfg.num_tasks, cfg.num_experts))

        epoch_loss = 0
        epoch_acc = 0
        samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.epochs}")
        for images, task_ids, labels in pbar:
            images, task_ids, labels = images.to(device), task_ids.to(device), labels.to(device)
            inputs = patchify_images(images, cfg.patch_size)

            # ==========================================================
            # [核心修改] 训练时直接传 None！
            # 模仿 DeepSeek/Mixtral：Router 只能看 inputs (Content)，没有 ID 捷径
            # ==========================================================
            logits, aux_loss, snapshot = model(inputs, task_id=None)

            # 统计路由 (依然用真实的 task_ids 做 Ground Truth 统计，看它怎么分)
            with torch.no_grad():
                B, S, K = snapshot['indices'].shape
                flat_indices = snapshot['indices'].view(-1).cpu().numpy()
                expanded_tasks = task_ids.view(B, 1, 1).expand(B, S, K).contiguous().view(-1).cpu().numpy()
                np.add.at(epoch_routing_stats, (expanded_tasks, flat_indices), 1)

            # 计算 Loss (Acc 统计依然需要 mask)
            total_main_loss = 0
            correct = 0
            batch_size = images.size(0)
            acc_breakdown = {}

            for t_id in range(cfg.num_tasks):
                mask = (task_ids == t_id)
                if not mask.any(): continue

                t_logits = logits[mask]
                t_labels = labels[mask]

                loss_t = nn.CrossEntropyLoss()(t_logits, t_labels)
                total_main_loss += loss_t * mask.sum() / batch_size

                preds = t_logits.argmax(dim=-1)
                t_correct = (preds == t_labels).sum().item()
                correct += t_correct
                acc_breakdown[f"T{t_id}"] = t_correct / mask.sum().item()

            # Loss = 主任务 + 负载均衡 (没有冲突 Loss)
            total_loss = total_main_loss + 0.01 * aux_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            global_step += 1
            epoch_loss += total_loss.item() * batch_size
            epoch_acc += correct
            samples += batch_size

            monitor.log_step(global_step, {
                'total_loss': total_loss.item(),
                'main_loss': total_main_loss.item(),
                'conflict_loss': 0.0,
                'alpha_sparsity': 0.0,
                'task_acc': correct / batch_size
            })

            pbar.set_postfix({
                'Loss': f"{total_loss.item():.4f}",
                'Acc': f"{correct / batch_size:.2%}",
                'T0': f"{acc_breakdown.get('T0', 0):.0%}",
                'T1': f"{acc_breakdown.get('T1', 0):.0%}",
                'T2': f"{acc_breakdown.get('T2', 0):.0%}"
            })

        avg_loss = epoch_loss / samples
        avg_acc = epoch_acc / samples
        print(f"Epoch {epoch + 1} Summary: Loss={avg_loss:.4f}, Acc={avg_acc:.2%}")

        # 保存快照 (注意：这里画的图就是真实的盲测路由)
        monitor.capture_snapshot(epoch, alpha_matrix=None, routing_stats=epoch_routing_stats)
        monitor.plot_current_routing(step=epoch, filename=f"routing_blind_epoch_{epoch}.png")

    print("=== Pure Blind Baseline Training Finished ===")
    plot_classic_curves(monitor.history, exp_name=cfg.exp_name)

    # 最后跑一次详细验证
    validate_final_blind(model, cfg)

    print(f"Results saved to: {monitor.save_dir}")


if __name__ == "__main__":
    train_baseline_blind()
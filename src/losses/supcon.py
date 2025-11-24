# src/losses/supcon.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al. 2020)
    - 输入 features: [N, D]  (已做 L2 归一化更好，但内部也会再 normalize 一次保证稳定)
    - 输入 labels:   [N]     (同类为正样本，不同类为负样本；y=-1 的样本会被忽略)
    """
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = features.device
        labels = labels.long().to(device)

        # 1) 忽略 y == -1 的样本（不参与对比学习）
        valid_mask = labels != -1
        features = features[valid_mask]         # [N_valid, D]
        labels   = labels[valid_mask]           # [N_valid]
        if features.size(0) <= 1:
            return features.new_tensor(0.0)

        # 2) L2 归一化，保证余弦相似度稳定
        features = F.normalize(features, dim=-1)

        N = features.size(0)
        # [N, N] 相似度矩阵
        sim = torch.div(features @ features.t(), self.temperature)

        # 为数值稳定减去每行最大值
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        # 3) 构建正样本 mask: 同类且非自身
        labels = labels.contiguous().view(-1, 1)           # [N,1]
        mask = torch.eq(labels, labels.T).float().to(device)  # [N,N]
        # 去掉对角线（自身不算正样本）
        logits_mask = torch.ones_like(mask) - torch.eye(N, device=device)
        mask = mask * logits_mask                          # [N,N]

        # 4) 计算分母：对所有 j!=i 做 softmax 归一化
        #   exp(sim_ij)，但对角线不参与（logits_mask 控制）
        exp_sim = torch.exp(sim) * logits_mask             # [N,N]
        denom = exp_sim.sum(dim=1, keepdim=True) + 1e-12   # [N,1]

        # 5) log_prob_ij = log( exp(sim_ij) / sum_{a!=i} exp(sim_ia) )
        log_prob = sim - torch.log(denom)

        # 对每个 anchor i，取它所有正样本 j 的 log_prob 的平均
        pos_count = mask.sum(dim=1)                        # [N]
        # 避免除以 0：只对有正样本的 i 计算 loss
        valid_row = pos_count > 0
        if valid_row.sum() == 0:
            return features.new_tensor(0.0)

        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (pos_count + 1e-12)

        # 6) 最终 loss 是负的平均
        loss = - mean_log_prob_pos[valid_row].mean()
        return loss
#返回到这里，损失函数已经写好了
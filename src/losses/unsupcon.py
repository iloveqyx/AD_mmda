import torch
import torch.nn as nn
import torch.nn.functional as F


class UnsupervisedContrastiveLoss(nn.Module):
    """NT-Xent (SimCLR) 对比损失，默认将 batch 内其它样本视为负例。

    - 输入为同一 batch 的两种数据增强视角 z1/z2，形状 [N, D]。
    - 正样本对：同一样本的两视角 (i, i+N)。
    - 负样本对：batch 内除正对外的所有样本，两视角合计 2N-2 个。
    - 通过温度缩放和 softmax，将正对拉近、负对推远。

    Args:
        temperature: 缩放相似度的温度，越小对大相似度更敏感。
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        # z1, z2: [N, D]，同一 batch 的两视角特征
        assert z1.shape == z2.shape, "z1 and z2 must have the same shape"
        if z1.size(0) <= 1:
            return z1.new_tensor(0.0)

        device = z1.device
        N = z1.size(0)
        # 归一化特征向量，稳定余弦相似度
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        z = torch.cat([z1, z2], dim=0)  # [2N, D]
        sim = torch.div(z @ z.t(), self.temperature)  # [2N,2N]

        # 避免数值爆炸
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        # 构建正样本索引：i<->i+N, i+N<->i，其他均视为负例
        diag = torch.eye(2 * N, device=device, dtype=torch.bool)
        logits_mask = (~diag).float()
        exp_sim = torch.exp(sim) * logits_mask
        denom = exp_sim.sum(dim=1, keepdim=True) + 1e-12

        # 正对的相似度：逐元素点乘再经温度缩放
        pos_sim = torch.exp(torch.sum(z1 * z2, dim=-1) / self.temperature)
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)  # [2N]

        log_prob = torch.log(pos_sim / denom.squeeze(1))
        loss = -log_prob.mean()
        return loss

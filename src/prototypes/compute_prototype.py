#我设想的是由于特征融合哪里有了L2归一化，所以这里采用余弦相似度，因为余弦相似度=归一化后的特征与原型向量的点积,先完成原型的产生。
import torch
import torch.nn.functional as F
from typing import Optional

#这里的输入是归一化后的特征和原型向量
#做一次L2归一化，防止数值不稳定
def l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(p=2, dim=-1, keepdim=True) + eps)

#不要梯度传播
@torch.no_grad()
#==== 要先找到原型 ====
def compute_prototype(
    z: torch.Tensor,# [B, L, D] or [B, D]
    y: torch.Tensor,# 标签
    num_classes: int,
    w: Optional[torch.Tensor] = None,#权重,这是伪标签的置信度，用相似度和分类头得到的
    prev_prototype: Optional[torch.Tensor] = None,#上一轮的原型,用于当前轮该类无样本时回退
)-> torch.Tensor:
    #根据当前的标签计算每个类别的原型向量
    #每个原型还要L2归一化
    device = z.device
    y=y.long().to(device)  # 确保标签是long类型

    # 如果是 [B, L, D]，先在时间维做平均池化成 [B, D]
    if z.dim() == 3:
        z = z.mean(dim=1)
    z = l2_normalize(z)  # 保证在单位球上
    
    if w is None:
        w = torch.ones_like(y, dtype=z.dtype, device=device)#如果权重为空，oneslike返回的是和y形状相同的全1张量,这是初始化的时候认为所有参与计算的样本同样可信
    else:
        w = w.to(z.dtype).to(device)

    B,D = z.shape
    prototype = torch.zeros(num_classes,D,device=device)#两类别，和特征z的维度相同
    sum_w = torch.zeros(num_classes,device=device)#每个类别的权重和,后续求原型的时候要用到

    for k in range(num_classes):
        mask_k = (y == k)  # 找到类别k的样本位置
        if mask_k.any():
            zk = z[mask_k]  # 取出类别k的特征[Nk, D]
            wk = w[mask_k].unsqueeze(-1)  # 取出类别k的权重，并扩展维度以便广播[Nk, 1]
            prototype[k] = (zk * wk).sum(dim=0)  # 加权求和
            sum_w[k] = wk.sum()  # 权重和
    
    # ===如果权重和为0，也就是当前类别还没有样本，优先沿用上一轮原型，否则保持为0向量===
    if prev_prototype is not None:
        prev_prototype = prev_prototype.to(device)

    for k in range(num_classes):
        if sum_w[k] > 0:
            prototype[k] /= sum_w[k]  # 平均值
        else:
            if prev_prototype is not None and k < prev_prototype.size(0):
                prototype[k] = prev_prototype[k]
            # 否则保持为全0，后续归一化会继续得到0向量，避免随机噪声
    
    prototype = l2_normalize(prototype)  # 最后归一化原型
    return prototype  # [num_classes, D]
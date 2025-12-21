#这个代码是通过mm_model
#目的是打上伪标签

from src.models.mm_model import MMModel
from src.prototypes.similarity import cosine_similarity_prototype
import torch
import torch.nn.functional as F
from typing import Tuple , Optional

@torch.no_grad()
def make_pseudo_labels(
    probs_h:torch.Tensor,  # 分类头的预测概率 [B, C]
    sim_a:torch.Tensor,    # audio模态与原型的相似度 [B, C]
    sim_t:torch.Tensor,    # text模态与原型的相似度 [B, C]
    sim_f:Optional[torch.Tensor]= None,    # 融合模态与原型的相似度 [B, C]
    #===== 相似度阈值 =====
    theta_a: float = 0.8, # θ_a：音频原型相似度的置信度阈值
    theta_t: float = 0.8, # θ_t：文本原型相似度的置信度阈值
    theta_h: float = 0.8, # θ_h：分类头置信度阈值
    delta_a: float = 0.2, # δ_a：音频模态的类别 margin 阈值 margin_a = top1_a - top2_a
    delta_t: float = 0.2, # δ_t：文本模态的类别 margin 阈值 margin_t = top1_t - top2_t
    kl_eps: Optional[float] = True  # KL 一致性阈值（可选）
)-> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #利用多模态原型相似度 + 融合分类头结果，为无标签样本打伪标签。
    device = probs_h.device
    B, C = probs_h.shape# batch size, num classes

    # --------- 1. top-1 类别 & 置信度 ----------
    pa, ia = sim_a.max(dim=-1)   # [B], audio
    pt, it = sim_t.max(dim=-1)   # [B], text
    ph, ih = probs_h.max(dim=-1) # [B], head

    # --------- 2. margin 计算 ----------
    top2_a = torch.topk(sim_a, k=2, dim=-1).values   # [B,2]
    top2_t = torch.topk(sim_t, k=2, dim=-1).values
    marg_a = top2_a[:, 0] - top2_a[:, 1]             # [B]
    marg_t = top2_t[:, 0] - top2_t[:, 1]             # [B]

    # 类别一致性
    agree_cls = (ia == it) & (ia == ih)

    # 置信度阈值
    conf_ok = (pa >= theta_a) & (pt >= theta_t) & (ph >= theta_h)

    # margin 阈值
    marg_ok = (marg_a >= delta_a) & (marg_t >= delta_t)

    # --------- 2. KL 一致性（可选） ----------
    if sim_f is not None:
        # 原型总体分布：简单平均三路（或者你可以只平均 a/t）
        p_proto = (sim_a + sim_t + sim_f) / 3.0
    else:
        p_proto = (sim_a + sim_t) / 2.0

    if kl_eps is not None:
        # KL(probs_h || p_proto)
        p_h = probs_h.clamp_min(1e-8)
        p_p = p_proto.clamp_min(1e-8)
        kl = (p_h * (p_h.log() - p_p.log())).sum(dim=-1)   # [B]
        kl_ok = (kl <= kl_eps)
    else:
        kl_ok = torch.ones(B, dtype=torch.bool, device=device)

    # --------- 3. 综合 mask & 伪标签 ----------
    mask = agree_cls & conf_ok & marg_ok & kl_ok           # [B]
    pseudo_y = ih.clone()                                  # 分类头的 argmax 作为伪标签

    # 权重：用三路置信度的最小值，保守一点
    pseudo_w = torch.minimum(torch.minimum(pa, pt), ph)    # [B]
    pseudo_w = torch.where(mask, pseudo_w, torch.zeros_like(pseudo_w))

    return mask, pseudo_y, pseudo_w
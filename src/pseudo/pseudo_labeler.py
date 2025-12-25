# 这个代码是通过 mm_model
# 目的是打上伪标签（text+head 主导，audio 可选否决）

import torch
import torch.nn.functional as F
from typing import Tuple, Optional


@torch.no_grad()
def make_pseudo_labels(
    probs_h: torch.Tensor,      # 分类头预测概率 [B, C]
    sim_a: torch.Tensor,        # audio 与原型 cos 相似度 [B, C]（用于 argmax 一致性）
    sim_t: torch.Tensor,        # text  与原型 cos 相似度 [B, C]（用于 argmax 一致性）
    sim_f: Optional[torch.Tensor] = None,  # 融合模态 cos 相似度 [B, C]

    sim_a_logits: Optional[torch.Tensor] = None,  # audio logits（建议传入 tau*cos 或 scale*cos）
    sim_t_logits: Optional[torch.Tensor] = None,  # text  logits
    sim_f_logits: Optional[torch.Tensor] = None,  # fuse  logits

    # ===== 置信度阈值 =====
    # 注意：本版本 theta_a/theta_t 表示 “原型 softmax 概率的最大值阈值”（pa_w/pt_w），不是 cos 阈值
    theta_a: float = 0.8,  # audio 原型概率阈值（用于 audio veto 或可选通过）
    theta_t: float = 0.8,  # text  原型概率阈值（主要通过条件）
    theta_h: float = 0.8,  # head  置信度阈值（ph = probs_h.max）

    # ===== margin 阈值（在原型概率上算 top1-top2，更稳）=====
    delta_a: float = 0.0,
    delta_t: float = 0.0,

    # ===== KL 一致性阈值（可选）=====
    kl_eps: Optional[float] = None,

    # ===== 是否允许 audio 参与“通过条件”（默认关：避免 audio 弱影响伪标签）=====
    use_audio_for_pass: bool = False,

    # ===== audio 否决（可选，默认开）：audio 很自信但与 text 冲突 => veto =====
    enable_audio_veto: bool = True,
    audio_veto_th: float = 0.85,   # audio 否决阈值（用 pa_w）

    # ===== 伪标签“限流/均衡”=====
    max_pl_ratio: Optional[float] = 0.7,   # 每个 batch 最多用多少比例伪标签（None 表示不限制）
    balance_classes: bool = True,          # 是否按类均衡保留
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    device = probs_h.device
    B, C = probs_h.shape

    # ---------- 1) head 的 top-1 类别与置信度 ----------
    ph, ih = probs_h.max(dim=-1)   # [B], head prob max + argmax

    # ---------- 2) 原型 argmax（用 cos 只取类别，不拿它当置信度） ----------
    # 说明：cos 的绝对值在不同模态可比性差，但 argmax（类别）还可以做一致性判断
    _, ia = sim_a.max(dim=-1)      # [B], audio 原型类别
    _, it = sim_t.max(dim=-1)      # [B], text  原型类别

    # ---------- 3) logits fallback（用于 softmax 概率/ margin / KL） ----------
    if sim_a_logits is None:
        sim_a_logits = sim_a
    if sim_t_logits is None:
        sim_t_logits = sim_t
    if sim_f is not None and sim_f_logits is None:
        sim_f_logits = sim_f

    # ---------- 4) 原型概率分布（更稳定的置信度与 margin） ----------
    p_a = F.softmax(sim_a_logits, dim=-1)  # [B,C]
    p_t = F.softmax(sim_t_logits, dim=-1)  # [B,C]
    pa_w, _ = p_a.max(dim=-1)              # [B], audio 原型概率置信度
    pt_w, _ = p_t.max(dim=-1)              # [B], text  原型概率置信度

    # ---------- 5) 原型概率 margin（top1 - top2） ----------
    top2_pa = torch.topk(p_a, k=2, dim=-1).values  # [B,2]
    top2_pt = torch.topk(p_t, k=2, dim=-1).values
    marg_a = top2_pa[:, 0] - top2_pa[:, 1]         # [B]
    marg_t = top2_pt[:, 0] - top2_pt[:, 1]         # [B]

    # ---------- 6) 一致性保护：text + head 主导 ----------
    # 二分类/AD 场景下 “>=1 对一致” 会退化为恒 True；因此这里用 ih==it 更有效
    agree_cls = (ih == it)

    # ---------- 7) 置信度门控：head 必须过线，text 原型必须过线 ----------
    # 默认不让 audio 参与通过（避免 audio 差把通过条件搞乱）
    if use_audio_for_pass:
        conf_ok = (ph >= theta_h) & ((pt_w >= theta_t) | (pa_w >= theta_a))
    else:
        conf_ok = (ph >= theta_h) & (pt_w >= theta_t)

    # ---------- 8) margin 门控：默认只看 text（更稳） ----------
    # 你也可以改成 (marg_t>=delta_t) | (marg_a>=delta_a) 让 audio 参与，但默认不建议
    marg_ok = (marg_t >= delta_t)

    # ---------- 9) KL 一致性（可选） ----------
    if sim_f is not None:
        p_f = F.softmax(sim_f_logits, dim=-1)
        p_proto = (p_a + p_t + p_f) / 3.0
    else:
        p_proto = (p_a + p_t) / 2.0

    if kl_eps is not None:
        p_h = probs_h.clamp_min(1e-8)
        p_p = p_proto.clamp_min(1e-8)
        kl = (p_h * (p_h.log() - p_p.log())).sum(dim=-1)  # [B]
        kl_ok = (kl <= kl_eps)
    else:
        kl_ok = torch.ones(B, dtype=torch.bool, device=device)

    # ---------- 10) audio 否决（可选）：audio 很自信但与 text 冲突 => veto ----------
    # audio 原型如果“没信息”，pa_w 常接近 0.5，则 veto 基本不会触发（这是好事）
    if enable_audio_veto:
        audio_veto = (pa_w >= audio_veto_th) & (ia != it)
    else:
        audio_veto = torch.zeros(B, dtype=torch.bool, device=device)

    # ---------- 11) 初筛 mask ----------
    mask_base = agree_cls & conf_ok & marg_ok & kl_ok & (~audio_veto)  # [B]

    # 伪标签类别：由于 mask_base 要求 ih==it，因此用 ih 或 it 都等价
    pseudo_y = it.clone()

    # ---------- 12) 伪标签权重：不要让 audio(pa_w) 参与 min（避免被 0.5 卡死） ----------
    pseudo_w = torch.minimum(pt_w, ph)  # [B]
    pseudo_w = torch.where(mask_base, pseudo_w, torch.zeros_like(pseudo_w))

    # ---------- 13) 限流/类均衡：防止 pl_ratio 太高、类别太偏 ----------
    if (max_pl_ratio is not None) and (max_pl_ratio > 0):
        K = max(1, int(round(B * max_pl_ratio)))
        keep = torch.zeros_like(mask_base)

        if balance_classes and C > 1:
            # 每类最多保留大约 K/C 个（按 pseudo_w 取 top）
            per_cls = max(1, (K + C - 1) // C)
            for c in range(C):
                idx = torch.nonzero(mask_base & (pseudo_y == c), as_tuple=False).squeeze(1)
                if idx.numel() == 0:
                    continue
                k_c = min(per_cls, idx.numel())
                topk = idx[pseudo_w[idx].topk(k_c).indices]
                keep[topk] = True
        else:
            idx = torch.nonzero(mask_base, as_tuple=False).squeeze(1)
            if idx.numel() <= K:
                keep = mask_base
            else:
                topk = idx[pseudo_w[idx].topk(K).indices]
                keep[topk] = True

        mask = keep
        pseudo_w = torch.where(mask, pseudo_w, torch.zeros_like(pseudo_w))
    else:
        mask = mask_base

    return mask, pseudo_y, pseudo_w

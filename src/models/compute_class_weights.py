"""
Compute class weights using labeled data plus pseudo-labeled confidence.

Usage:
    weights = compute_class_weights(...)
    loss = F.cross_entropy(logits, targets, weight=weights)

Rationale:
- 对少数类提升权重，缓解类别不平衡。
- 伪标签部分使用原型相似度得分 (w_hat) 乘分类器置信度，减少噪声。
"""
from typing import Dict

import torch
from torch import Tensor

from src.prototypes.similarity import cosine_similarity_prototype
from src.pseudo.pseudo_labeler import make_pseudo_labels


def _count_labels_from_loader(loader, num_classes: int, device: torch.device) -> Tensor:
    """Count labels (y>=0) from a dataloader; returns counts on ``device``."""
    counts = torch.zeros(num_classes, device=device, dtype=torch.float)
    if loader is None:
        return counts

    for _, _, y, _ in loader:
        if y is None:
            continue
        if not torch.is_tensor(y):
            y = torch.tensor(y)
        y = y.to(device)
        mask = y >= 0
        if mask.any():
            y_valid = y[mask]
            counts += torch.bincount(y_valid, minlength=num_classes).float()
    return counts


def compute_class_weights(
    model: torch.nn.Module,
    proto_bank: Dict[str, Tensor],
    loader_src,
    loader_tgt_l,
    loader_tgt_u,
    num_classes: int,
    device: torch.device,
    *,
    args,
    alpha: float = 1.0,
    eps: float = 1e-6,
    use_src: bool = False,
    use_confidence: bool = True,
    gamma: float = 0.5,
    max_scale: float = 2.0,
    min_scale: float = 0.5,
    manual_pos_scale: float = 1.0,
) -> Tensor:
    """Compute per-class weights (inverse frequency) with pseudo-label confidence.

    - 有标签计数: 源域可选 + 目标有标签。
    - 伪标签计数: 使用原型相似度权重 w_hat × 分类器置信度 conf_h（可关掉）。
    - 逆频率: total / (K * (count + eps))。
    - 平滑: 对逆频率做幂缩放 (gamma ∈ (0,1])，减小极端不平衡带来的过大差异。
    - 截断: 将最终归一化后的权重限制在 [min_scale, max_scale]（相对均值），避免某一类权重过大或过小。
    """
    model_was_training = model.training
    model.eval()

    # --- Labeled counts ---
    counts = torch.zeros(num_classes, device=device, dtype=torch.float)
    counts += _count_labels_from_loader(loader_tgt_l, num_classes, device)
    if use_src:
        counts += _count_labels_from_loader(loader_src, num_classes, device)

    proto_a = proto_bank.get("a")
    proto_t = proto_bank.get("t")
    proto_f = proto_bank.get("f")

    # --- Pseudo-label counts weighted by similarity & confidence ---
    if loader_tgt_u is not None:
        with torch.no_grad():
            for Xa_u, Xt_u, y_u, _ in loader_tgt_u:
                Xa_u, Xt_u = Xa_u.to(device), Xt_u.to(device)

                logits_u, probs_u, (za_u, zt_u, zf_u, _) = model(Xa_u, Xt_u)
                sim_a_u, sim_a_u_logits = cosine_similarity_prototype(za_u, proto_a, tau=args.tau_proto)
                sim_t_u, sim_t_u_logits = cosine_similarity_prototype(zt_u, proto_t, tau=args.tau_proto)
                sim_f_u, sim_f_u_logits = cosine_similarity_prototype(zf_u, proto_f, tau=args.tau_proto)

                mask_pl, y_hat, w_hat = make_pseudo_labels(
                    probs_h=probs_u,
                    sim_a=sim_a_u,
                    sim_t=sim_t_u,
                    sim_f=sim_f_u,
                    sim_a_logits=sim_a_u_logits,
                    sim_t_logits=sim_t_u_logits,
                    sim_f_logits=sim_f_u_logits,
                    theta_a=args.theta_a,
                    theta_t=args.theta_t,
                    theta_h=args.theta_h,
                    delta_a=args.delta_a,
                    delta_t=args.delta_t,
                    kl_eps=args.kl_eps,
                )

                if mask_pl.any():
                    # 分类器最大概率作为置信度；可关掉以减少噪声放大
                    conf_h = probs_u.max(dim=1).values if use_confidence else 1.0
                    # 原型相似度权重 × 置信度，得到伪标签样本的有效权重
                    w_pl = w_hat * conf_h
                    for c in range(num_classes):
                        mask_c = mask_pl & (y_hat == c)
                        if mask_c.any():
                            counts[c] += alpha * w_pl[mask_c].sum()

    # --- Inverse-frequency weight with mean-normalization ---
    total = counts.sum()
    if total <= 0:
        weights = torch.ones(num_classes, device=device, dtype=torch.float)
    else:
        # 逆频率
        inv_freq = total / (num_classes * (counts + eps))

        # 幂平滑: gamma=1 等价于原始逆频率；0<gamma<1 会压缩动态范围
        gamma = max(1e-6, float(gamma))
        inv_freq_smooth = inv_freq ** gamma

        # 额外对正类(假定为类别1)进行手工放大，偏向 AD 召回
        if num_classes == 2 and manual_pos_scale is not None and float(manual_pos_scale) != 1.0:
            inv_freq_smooth[1] = inv_freq_smooth[1] * float(manual_pos_scale)

        # 按均值归一，保持整体尺度稳定
        weights = inv_freq_smooth / inv_freq_smooth.mean()

        # 相对均值的上下截断，避免极端大/小权重
        if max_scale is not None and max_scale > 0:
            min_scale_val = float(min_scale) if min_scale is not None else 0.0
            weights = torch.clamp(weights, min=min_scale_val, max=float(max_scale))

    if model_was_training:
        model.train()
    return weights.to(device)

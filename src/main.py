# src/train_main.py
# 模型没有修改，训练脚本修改为五折交叉验证模式
# 自动根据 --source 和 --target 生成数据路径
# 采样方式不采用zip，改为iter+next的形式
# 将伪标签数据纳入原型计算
import os
import argparse
from typing import Dict, Any
from pathlib import Path
from datetime import datetime
import json
import itertools
import copy
import logging

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 服务器/无显示环境下画图
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

from src.models.mm_model import MMModel
from src.losses.supcon import SupervisedContrastiveLoss
from src.losses.unsupcon import UnsupervisedContrastiveLoss
from src.prototypes.compute_prototype import compute_prototype
from src.prototypes.similarity import cosine_similarity_prototype
from src.pseudo.pseudo_labeler import make_pseudo_labels
from src.models.compute_class_weights import compute_class_weights

'''
Example Usage:
conda activate proto_mmda
cd /home/ad_group1/AD_mmda2

CUDA_VISIBLE_DEVICES=0 python -m src.main \
  --source Pitt \
  --target Lu \
  --data_root /home/ad_group1/data \
  --epochs 500 \
  --batch_size 64 \
  --warmup_epochs 200 \
  --lr 1e-5 \
  --use_class_weight \
  --device cuda \
  --exp_name "验证集得到最佳模型" 
'''
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
print(f"PyTorch uses device: {torch.cuda.get_device_name(0)}")
print(f"Current device index in PyTorch: {torch.cuda.current_device()}")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-modal Semi-supervised DA with Gated Fusion + Prototype + SupCon (5-Fold)"
    )
    
    # === 新增：源域、目标域、数据根目录选择 ===
    parser.add_argument("--source", type=str, default="Pitt", help="源域名称 (e.g., Pitt)")
    parser.add_argument("--target", type=str, default="Lu", help="目标域名称 (e.g., Dem, Lu)")
    parser.add_argument("--data_root", type=str, default="/home/ad_group1/data", help="数据存放根目录")

    # 原有的路径参数不再强制需要，若不传则自动生成
    parser.add_argument("--src_pt", type=str, default=None)
    parser.add_argument("--tgt_l_pt", type=str, default=None)
    parser.add_argument("--tgt_u_pt", type=str, default=None)
    parser.add_argument("--val_pt", type=str, default=None)
    parser.add_argument("--test_pt", type=str, default=None)

    # 模型超参（da/dt 会在 build_dataloaders 里用 .pt 里的 dim 覆盖）
    parser.add_argument("--da", type=int, default=1024)  # audio feat dim
    parser.add_argument("--dt", type=int, default=768)   # text  feat dim
    parser.add_argument("--dproj", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--gate_mode", type=str, default="channel", choices=["channel", "scalar"])
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--use_class_weight", action="store_true", help="是否启用类别权重")
    parser.add_argument("--class_weight_alpha", type=float, default=1.0, help="伪标签计数放大因子")
    parser.add_argument("--class_weight_eps", type=float, default=1e-6, help="计数平滑项")
    parser.add_argument("--class_weight_use_src", action="store_true", help="计数时是否包含源域有标签")
    parser.add_argument("--class_weight_no_conf", action="store_true", help="伪标签计数时不乘分类器置信度")
    parser.add_argument("--class_weight_gamma", type=float, default=0.5,
                        help="逆频率权重的幂指数(0<gamma<=1; 越小越温和, gamma=1 为原始逆频率)")
    parser.add_argument("--class_weight_max_scale", type=float, default=2.0,
                        help="单类权重相对均值的最大倍数(<=0 表示不截断)")
    parser.add_argument("--class_weight_min_scale", type=float, default=0.5,
                        help="单类权重相对均值的最小倍数")
    parser.add_argument("--manual_pos_scale", type=float, default=1.3,
                        help="二分类时对正类(AD类)权重的额外放大系数, >1 将更偏向 AD 召回")
    parser.add_argument("--class_weight_update_interval", type=int, default=1,
                        help="warmup 后每隔多少个 epoch 重新计算类别权重(<=0 表示不更新)")

    # loss 权重
    parser.add_argument("--lam_sup", type=float, default=1.0)
    parser.add_argument("--lam_sup_src", type=float, default=1.0)
    parser.add_argument("--lam_sup_tgt", type=float, default=2.0)
    parser.add_argument("--lam_pl",  type=float, default=10.0)
    parser.add_argument("--lam_con", type=float, default=0.3)
    parser.add_argument("--lam_proto", type=float, default=0.0)
    parser.add_argument("--lam_ent", type=float, default=0.0)
    parser.add_argument("--lam_unsup", type=float, default=0.1, help="无监督对比损失权重")
    parser.add_argument("--unsup_temp", type=float, default=0.1, help="无监督对比温度")
    parser.add_argument("--unsup_noise_std", type=float, default=0.01, help="无监督对比视角高斯噪声 std")
    parser.add_argument("--unsup_start_epoch", type=int, default=1, help="从第几轮开始启用无监督对比")

    # 对比学习温度
    parser.add_argument("--supcon_temp", type=float, default=0.1)

    # 伪标签 / 原型相似度超参
    parser.add_argument("--tau_proto", type=float, default=20.0,
                        help="prototype similarity temperature")
    parser.add_argument("--theta_a", type=float, default=0,
                        help="audio 原型相似度阈值")#理论范围[-1,1]
    parser.add_argument("--theta_t", type=float, default=0,
                        help="text 原型相似度阈值") #理论范围[-1,1]
    parser.add_argument("--theta_h", type=float, default=0,
                        help="融合分类器高置信度阈值")#理论范围[0,1]
    parser.add_argument("--delta_a", type=float, default=0,
                        help="audio 模态 margin")#理论范围[0,2]
    parser.add_argument("--delta_t", type=float, default=0,
                        help="text 模态 margin")#理论范围[0,2]
    parser.add_argument("--kl_eps", type=float, default=10,
                        help="KL 一致性约束阈值")#理论范围[0,∞)

    # 训练超参
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup_epochs", type=int, default=10,
                        help="前多少个 epoch 只用监督/对比学习，不使用伪标签损失")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    # 结果保存
    parser.add_argument("--result_root", type=str, default="result",
                        help="结果根目录")
    # exp_name 如果不填，代码会自动根据 source2target_time 生成
    parser.add_argument("--exp_name", type=str,
                        help="本次实验的自定义名称后缀")

    return parser.parse_args()


def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =======================
#  Dataset & Dataloaders
# =======================
class OfflineFeatDataset(Dataset):
    """
    包装 offline_extract_features.py 生成的 *.pt 文件
    """
    def __init__(self, data: Dict[str, Any]):
        feats_a = data["feats_audio"]          # Tensor [N, Ca]
        feats_t = data["feats_text"]           # Tensor [N, Ct]
        N = feats_a.size(0)

        self.Xa = feats_a.float()
        self.Xt = feats_t.float()

        cfg = data.get("config", {}) or {}
        ignore_label_value = cfg.get("ignore_label_value", -1)

        label = data.get("label", None)
        if label is None:
            # 无标签：全部填成 ignore_label_value（一般是 -1）
            self.y = torch.full((N,), ignore_label_value, dtype=torch.long)
        else:
            if not torch.is_tensor(label):
                label = torch.tensor(label)
            self.y = label.long()

        # 元信息（可选）
        self.speaker = data.get("speaker", None)
        self.path_audio = data.get("path_audio", None)
        self.text = data.get("text", None)

    def __len__(self):
        return self.Xa.size(0)

    def __getitem__(self, idx: int):
        Xa = self.Xa[idx]  # [Ca]
        Xt = self.Xt[idx]  # [Ct]
        y  = self.y[idx]   # []

        meta = {
            "speaker": self.speaker[idx] if self.speaker is not None else None,
            "path_audio": self.path_audio[idx] if self.path_audio is not None else None,
            "text": self.text[idx] if self.text is not None else None,
        }
        return Xa, Xt, y, meta


def _load_split(pt_path: str):
    """
    从 .pt 文件加载一个 split，返回 Dataset 和 config。
    """
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"Data file not found: {pt_path}")
    
    data = torch.load(pt_path, map_location="cpu")
    ds = OfflineFeatDataset(data)
    cfg = data.get("config", {}) or {}
    return ds, cfg


def build_dataloaders(args) -> Dict[str, DataLoader]:
    """
    读取 args 中指定的 *.pt 特征路径构建 DataLoader
    """
    # 1) 加载各个 split 的 .pt
    print(f"Loading Source: {args.src_pt}")
    print(f"Loading Target: {args.tgt_l_pt} (Labeled) / {args.tgt_u_pt} (Unlabeled)")
    
    src_ds,  cfg_src  = _load_split(args.src_pt)
    tgt_l_ds, cfg_tl  = _load_split(args.tgt_l_pt)
    tgt_u_ds, cfg_tu  = _load_split(args.tgt_u_pt)
    val_ds,  cfg_val  = _load_split(args.val_pt)
    test_ds, cfg_test = _load_split(args.test_pt)

    # 2) 简单 sanity check
    da = cfg_src.get("audio_hidden_dim", src_ds.Xa.size(1))
    dt = cfg_src.get("text_hidden_dim",  src_ds.Xt.size(1))

    assert src_ds.Xa.size(1) == da and src_ds.Xt.size(1) == dt, "src 特征维度不一致"
    
    # 用特征文件里的维度覆盖 args.da/args.dt，确保和 MMModel 一致
    args.da = da
    args.dt = dt

    bs = args.batch_size

    # 3) 构建 DataLoader
    train_src_loader = DataLoader(
        src_ds, batch_size=bs, shuffle=True, drop_last=True, num_workers=4, pin_memory=True
    )
    train_tgt_l_loader = DataLoader(
        tgt_l_ds, batch_size=bs, shuffle=True, drop_last=True, num_workers=4, pin_memory=True
    )
    train_tgt_u_loader = DataLoader(
        tgt_u_ds, batch_size=bs, shuffle=True, drop_last=True, num_workers=4, pin_memory=True
    )

    proto_src_loader = DataLoader(
        src_ds, batch_size=bs, shuffle=False, drop_last=False, num_workers=4, pin_memory=True
    )
    proto_tgt_l_loader = DataLoader(
        tgt_l_ds, batch_size=bs, shuffle=False, drop_last=False, num_workers=4, pin_memory=True
    )
    proto_tgt_u_loader = DataLoader(
        tgt_u_ds, batch_size=bs, shuffle=False, drop_last=False, num_workers=4, pin_memory=True
    )

    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False, drop_last=False, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=bs, shuffle=False, drop_last=False, num_workers=4, pin_memory=True
    )

    train_tgt_all_loader = DataLoader(
        tgt_l_ds, batch_size=bs, shuffle=True, drop_last=True, num_workers=4, pin_memory=True
    )

    loaders = {
        "train_src": train_src_loader,
        "train_tgt_l": train_tgt_l_loader,
        "train_tgt_u": train_tgt_u_loader,
        "proto_src": proto_src_loader,
        "proto_tgt_l": proto_tgt_l_loader,
        "proto_tgt_u": proto_tgt_u_loader,
        "val": val_loader,
        "test": test_loader,
        "train_tgt_all": train_tgt_all_loader,
    }
    return loaders


# ==============
#  Train / Proto 
# ==============

def train_one_epoch(
    epoch: int,
    model: nn.Module,
    loader_src: DataLoader,
    loader_tgt_l: DataLoader,
    loader_tgt_u: DataLoader,
    optimizer: torch.optim.Optimizer,
    supcon_loss: SupervisedContrastiveLoss,
    args,
    proto_bank: Dict[str, torch.Tensor],
    device: torch.device,
    class_weights: torch.Tensor = None,
    unsupcon_loss: 'UnsupervisedContrastiveLoss' = None,
) -> Dict[str, float]:
    
    model.train()
    supcon_loss.train()

    # 统计量
    total_sup = total_pl = total_con = 0.0
    total_sup_src = 0.0
    total_sup_tgt = 0.0
    total_pl_n = 0          
    total_u_n = 0           
    total_unsup = 0.0
    n_steps = 0

    lam_sup_src = getattr(args, "lam_sup_src", 1.0)
    lam_sup_tgt = getattr(args, "lam_sup_tgt", 1.0)

    # ---- target-centric 采样 ----
    iter_src   = itertools.cycle(loader_src)
    iter_tgt_l = itertools.cycle(loader_tgt_l)

    for batch_u in loader_tgt_u:
        n_steps += 1

        batch_s = next(iter_src)
        batch_l = next(iter_tgt_l)

        Xa_s, Xt_s, y_s, meta_s = batch_s
        Xa_l, Xt_l, y_l, meta_l = batch_l
        Xa_u, Xt_u, y_u, meta_u = batch_u 

        Xa_s = Xa_s.to(device); Xt_s = Xt_s.to(device); y_s = y_s.to(device)
        Xa_l = Xa_l.to(device); Xt_l = Xt_l.to(device); y_l = y_l.to(device)
        Xa_u = Xa_u.to(device); Xt_u = Xt_u.to(device); y_u = y_u.to(device)

        optimizer.zero_grad()

        # 1) 前向
        logits_s, probs_s, (za_s, zt_s, zf_s, g_s) = model(Xa_s, Xt_s)
        logits_l, probs_l, (za_l, zt_l, zf_l, g_l) = model(Xa_l, Xt_l)
        logits_u, probs_u, (za_u, zt_u, zf_u, g_u) = model(Xa_u, Xt_u)

        # 2) 监督 CE
        # 仅对真实标签 CE 使用类别权重；伪标签 CE 不加类权重，避免双重加权伪标签噪声
        ce_kwargs_sup = {"weight": class_weights} if class_weights is not None else {}
        loss_sup_src = F.cross_entropy(logits_s, y_s, **ce_kwargs_sup)
        loss_sup_tgt = F.cross_entropy(logits_l, y_l, **ce_kwargs_sup)
        loss_sup = loss_sup_src + loss_sup_tgt

        # 3) 无标签 / 伪标签
        is_warmup = epoch <= getattr(args, "warmup_epochs", 0)
        u_n = int(y_u.size(0))

        if is_warmup:
            mask_pl = torch.zeros_like(y_u, dtype=torch.bool)
            y_hat = torch.zeros_like(y_u, dtype=torch.long)
            loss_pl = torch.tensor(0.0, device=device)
            pl_n = 0
        else:
            prototype_a = proto_bank["a"]  
            prototype_t = proto_bank["t"]
            prototype_f = proto_bank["f"]

            sim_a_u, sim_a_u_logits = cosine_similarity_prototype(za_u, prototype_a, tau=args.tau_proto)
            sim_t_u, sim_t_u_logits = cosine_similarity_prototype(zt_u, prototype_t, tau=args.tau_proto)
            sim_f_u, sim_f_u_logits = cosine_similarity_prototype(zf_u, prototype_f, tau=args.tau_proto)

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

            pl_n = int(mask_pl.sum().item())

            if mask_pl.any():
                logits_pl = logits_u[mask_pl]
                y_pl = y_hat[mask_pl]
                w_pl = w_hat[mask_pl]
                # 伪标签 CE 不使用类别权重，只用样本级权重 w_pl
                ce_pl = F.cross_entropy(logits_pl, y_pl, reduction="none")
                loss_pl = (ce_pl * w_pl).mean()
            else:
                loss_pl = torch.tensor(0.0, device=device)

        total_pl_n += pl_n
        total_u_n += u_n

        # 4) 对比学习
        z_list = [zf_s, zf_l]
        y_list = [y_s,  y_l]
        if not is_warmup and mask_pl.any():
            z_list.append(zf_u[mask_pl])
            y_list.append(y_hat[mask_pl])
        z_all = torch.cat(z_list, dim=0)
        y_all = torch.cat(y_list, dim=0)

        loss_con = supcon_loss(z_all, y_all)

        # 4.1 无监督对比（SimCLR 两视角），基于未标注 batch
        loss_unsup = torch.tensor(0.0, device=device)
        # 为了与 warmup 阶段解耦，这里实际启用轮次为
        # max(unsup_start_epoch, warmup_epochs + 1)
        effective_unsup_start = max(
            getattr(args, "unsup_start_epoch", 1),
            getattr(args, "warmup_epochs", 0) + 1,
        )
        if (
            unsupcon_loss is not None
            and args.lam_unsup > 0
            and epoch >= effective_unsup_start
        ):
            noise_std = getattr(args, "unsup_noise_std", 0.0)
            if noise_std > 0:
                Xa_u_2 = Xa_u + torch.randn_like(Xa_u) * noise_std
                Xt_u_2 = Xt_u + torch.randn_like(Xt_u) * noise_std
            else:
                Xa_u_2, Xt_u_2 = Xa_u, Xt_u

            # 复用当前前向得到的 zf_u 作为第一视角，仅对第二视角施加噪声，
            # 既减少一次前向计算，也避免视角过于相近。
            logits_u2, probs_u2, (za_u2, zt_u2, zf_u2, g_u2) = model(Xa_u_2, Xt_u_2)
            loss_unsup = unsupcon_loss(zf_u, zf_u2)

        # 5) 其他 loss
        loss_proto = torch.tensor(0.0, device=device)
        loss_ent   = torch.tensor(0.0, device=device)

        if is_warmup:
            lam_pl_eff = 0.0
        else:
            lam_pl_eff = args.lam_pl

        lam_con_eff = args.lam_con

        loss = (
            args.lam_sup * (
                lam_sup_src * loss_sup_src +
                lam_sup_tgt * loss_sup_tgt
            )
            + lam_pl_eff  * loss_pl
            + lam_con_eff * loss_con
            + args.lam_proto * loss_proto
            + args.lam_ent   * loss_ent
            + args.lam_unsup * loss_unsup
        )

        loss.backward()
        optimizer.step()

        total_sup += loss_sup.item()
        total_sup_src += loss_sup_src.item()
        total_sup_tgt += loss_sup_tgt.item()
        total_pl  += loss_pl.item()
        total_con += loss_con.item()
        total_unsup += loss_unsup.item()

    stats = {
        "loss_sup": total_sup / max(1, n_steps),
        "loss_sup_src": total_sup_src / max(1, n_steps),
        "loss_sup_tgt": total_sup_tgt / max(1, n_steps),
        "loss_pl":  total_pl  / max(1, n_steps),
        "loss_con": total_con / max(1, n_steps),
        "loss_unsup": total_unsup / max(1, n_steps),
        "pl_ratio": (total_pl_n / max(1, total_u_n)),
        "pl_count": total_pl_n,
    }
    return stats


@torch.no_grad()
def build_proto_bank(
    model: nn.Module,
    loader_tgt_l: DataLoader,   
    loader_tgt_u: DataLoader,   
    num_classes: int,
    args,
    device: torch.device,
    epoch: int,
    prev_proto_bank: Dict[str, torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    
    model.eval()
    is_warmup = epoch <= args.warmup_epochs

    # 1) 目标有标签原型
    zf_lbl_list, za_lbl_list, zt_lbl_list, y_lbl_list = [], [], [], []

    for Xa, Xt, y, _ in loader_tgt_l:
        Xa, Xt, y = Xa.to(device), Xt.to(device), y.to(device)
        mask = (y >= 0)
        if not mask.any():
            continue

        logits, probs, (za, zt, zf, g) = model(Xa, Xt)

        zf_lbl_list.append(zf[mask])
        za_lbl_list.append(za[mask])
        zt_lbl_list.append(zt[mask])
        y_lbl_list.append(y[mask].long())

    if len(y_lbl_list) == 0:
        # 如果实在没有有标签样本，这里应该做额外处理或报错，这里暂且抛出异常
        raise RuntimeError("build_proto_bank: no labeled target samples (all y<0).")

    zf_lbl = torch.cat(zf_lbl_list, dim=0)
    za_lbl = torch.cat(za_lbl_list, dim=0)
    zt_lbl = torch.cat(zt_lbl_list, dim=0)
    y_lbl  = torch.cat(y_lbl_list,  dim=0)

    proto_f_l = compute_prototype(zf_lbl, y_lbl, num_classes=num_classes, w=None)
    proto_a_l = compute_prototype(za_lbl, y_lbl, num_classes=num_classes, w=None)
    proto_t_l = compute_prototype(zt_lbl, y_lbl, num_classes=num_classes, w=None)

    if is_warmup:
        return {"f": proto_f_l, "a": proto_a_l, "t": proto_t_l}

    # 2) 伪标签打标
    zf_p_list, za_p_list, zt_p_list = [], [], []
    y_p_list, w_p_list = [], []

    for Xa_u, Xt_u, y_u, _ in loader_tgt_u:
        Xa_u, Xt_u = Xa_u.to(device), Xt_u.to(device)

        logits_u, probs_u, (za_u, zt_u, zf_u, g_u) = model(Xa_u, Xt_u)

        sim_a_u, sim_a_u_logits = cosine_similarity_prototype(za_u, proto_a_l, tau=args.tau_proto)
        sim_t_u, sim_t_u_logits = cosine_similarity_prototype(zt_u, proto_t_l, tau=args.tau_proto)
        sim_f_u, sim_f_u_logits = cosine_similarity_prototype(zf_u, proto_f_l, tau=args.tau_proto)

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
            zf_p_list.append(zf_u[mask_pl])
            za_p_list.append(za_u[mask_pl])
            zt_p_list.append(zt_u[mask_pl])
            y_p_list.append(y_hat[mask_pl])
            w_p_list.append(w_hat[mask_pl])

    if len(y_p_list) == 0:
        # 若当前轮没有任何伪标签样本，则：
        #  - 若存在上一轮的原型，则优先沿用上一轮，以避免原型发生剧烈波动；
        #  - 否则退回到仅由有标签样本计算的原型。
        if prev_proto_bank is not None:
            return prev_proto_bank
        return {"f": proto_f_l, "a": proto_a_l, "t": proto_t_l}

    # 3) 合并计算
    zf_p = torch.cat(zf_p_list, dim=0)
    za_p = torch.cat(za_p_list, dim=0)
    zt_p = torch.cat(zt_p_list, dim=0)
    y_p  = torch.cat(y_p_list,  dim=0)
    w_p  = torch.cat(w_p_list,  dim=0)

    w_lbl = torch.ones_like(y_lbl, dtype=torch.float, device=device)

    zf_all = torch.cat([zf_lbl, zf_p], dim=0)
    za_all = torch.cat([za_lbl, za_p], dim=0)
    zt_all = torch.cat([zt_lbl, zt_p], dim=0)
    y_all  = torch.cat([y_lbl,  y_p],  dim=0)
    w_all  = torch.cat([w_lbl,  w_p],  dim=0)

    prev_f = prev_proto_bank["f"] if prev_proto_bank is not None else None
    prev_a = prev_proto_bank["a"] if prev_proto_bank is not None else None
    prev_t = prev_proto_bank["t"] if prev_proto_bank is not None else None

    proto_f = compute_prototype(zf_all, y_all, num_classes=num_classes, w=w_all, prev_prototype=prev_f)
    proto_a = compute_prototype(za_all, y_all, num_classes=num_classes, w=w_all, prev_prototype=prev_a)
    proto_t = compute_prototype(zt_all, y_all, num_classes=num_classes, w=w_all, prev_prototype=prev_t)

    return {"f": proto_f, "a": proto_a, "t": proto_t}


@torch.no_grad()
def eval_on_split(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int = 2,
) -> Dict[str, Any]:
    model.eval()

    all_labels = []
    all_scores = []  
    total_loss = 0.0
    total_n = 0

    for Xa, Xt, y, _ in loader:
        Xa, Xt, y = Xa.to(device), Xt.to(device), y.to(device)
        mask = (y >= 0)
        if not mask.any():
            continue

        Xa = Xa[mask]
        Xt = Xt[mask]
        y_valid = y[mask].long()

        logits, probs, _ = model(Xa, Xt)
        loss = F.cross_entropy(logits, y_valid, reduction="sum")
        total_loss += loss.item()
        total_n += y_valid.size(0)

        if num_classes == 2:
            scores = probs[:, 1].detach().cpu().numpy()
        else:
            scores = torch.softmax(logits, dim=-1).max(dim=-1).values.detach().cpu().numpy()

        labels_np = y_valid.detach().cpu().numpy()
        all_labels.append(labels_np)
        all_scores.append(scores)

    if total_n == 0:
        return {"loss": 0.0, "acc": 0.0, "f1": 0.0, "auc": 0.0, "cm": []}

    all_labels = np.concatenate(all_labels, axis=0)
    all_scores = np.concatenate(all_scores, axis=0)

    if num_classes == 2:
        y_pred = (all_scores >= 0.5).astype(int)
    else:
        raise NotImplementedError("eval_on_split currently assumes binary classification.")

    acc = accuracy_score(all_labels, y_pred)
    f1  = f1_score(all_labels, y_pred)

    if len(np.unique(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_scores)
    else:
        auc = None

    cm = confusion_matrix(all_labels, y_pred, labels=[0, 1]).tolist()

    metrics = {
        "loss": total_loss / total_n,
        "acc": acc,
        "f1": f1,
        "auc": auc,
        "cm": cm,
    }
    return metrics


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[device] use {device}")

    # === 构造实验名称与目录 ===
    # 格式：{source}2{target}_{time}
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_basename = f"{args.source}2{args.target}_{timestamp}"
    if args.exp_name is not None:
        # 如果用户指定了额外的后缀，拼接上去
        exp_basename = f"{exp_basename}_{args.exp_name}"

    result_root = Path(args.result_root)
    run_dir = result_root / exp_basename
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[result] experiment root: {run_dir}")

    # 保存一份全局配置
    with open(run_dir / "args_global.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    fold_metrics = []

    # === 五折交叉验证循环 (1-5) ===
    # 数据格式假定为: {data_root}/{Domain}/fold{k}/train.pt
    for fold_idx in range(1, 6):

        print(f"\n{'='*20} Start Fold {fold_idx} / 5 {'='*20}")
        
        # 1. 构造当前折的路径并更新到 args 中
        # 注意：源域只需 train.pt，目标域需要 tgt_labeled, tgt_unlabeled, val, test
        # 这里假设目标域的划分也是在 fold{k} 文件夹下
        
        src_fold_dir = Path(args.data_root) / args.source / f"fold{fold_idx}"
        tgt_fold_dir = Path(args.data_root) / args.target / f"fold{fold_idx}"
        
        args.src_pt     = str(src_fold_dir / "train.pt")
        args.tgt_l_pt   = str(tgt_fold_dir / "tgt_labeled.pt")
        args.tgt_u_pt   = str(tgt_fold_dir / "tgt_unlabeled.pt")
        args.val_pt     = str(tgt_fold_dir / "val.pt")
        args.test_pt    = str(tgt_fold_dir / "test.pt")

        # 为当前 fold 创建子目录保存结果
        fold_run_dir = run_dir / f"fold{fold_idx}"
        fold_run_dir.mkdir(parents=True, exist_ok=True)

        # 2. 重新构建 Dataloader
        try:
            loaders = build_dataloaders(args)
        except FileNotFoundError as e:
            print(f"[Error] Skip Fold {fold_idx}: {e}")
            continue

        loader_src      = loaders["train_src"]
        loader_tgt_l    = loaders["train_tgt_l"]
        loader_tgt_u    = loaders["train_tgt_u"]
        proto_src_loader = loaders["proto_src"]
        proto_tgt_l_loader = loaders["proto_tgt_l"]
        proto_tgt_u_loader = loaders["proto_tgt_u"]
        val_loader      = loaders["val"]
        test_loader     = loaders["test"]
        
        # 3. 初始化模型 & 优化器 (每折重置)
        model = MMModel(
            da=args.da,
            dt=args.dt,
            num_classes=args.num_classes,
            dproj=args.dproj,
            nhead=args.nhead,
            gate_mode=args.gate_mode,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        supcon_loss = SupervisedContrastiveLoss(temperature=args.supcon_temp).to(device)
        unsupcon_loss = UnsupervisedContrastiveLoss(temperature=args.unsup_temp).to(device)

        # 4. 初始原型
        try:
            proto_bank = build_proto_bank(
                model,
                loader_tgt_l=proto_tgt_l_loader,
                loader_tgt_u=proto_tgt_u_loader,
                num_classes=args.num_classes,
                args=args,
                device=device,
                epoch=0,
                prev_proto_bank=None,
            )
        except RuntimeError as e:
            # 例如: 该折目标有标签集全为 y<0, 无法构建原型
            print(f"[Error] Skip Fold {fold_idx} when building proto_bank: {e}")
            continue

        # 4.1 类别权重（可选）
        class_weights = None
        class_weight_logs = []
        
        # 记录曲线
        hist_sup = []
        hist_pl = []
        hist_con = []
        hist_pl_ratio = []
        hist_val_metrics = []

        # 5. 训练循环
        best_val_metric = float("-inf")
        best_val_record = None
        for epoch in range(1, args.epochs + 1):
            proto_bank = build_proto_bank(
                model,
                loader_tgt_l=proto_tgt_l_loader,
                loader_tgt_u=proto_tgt_u_loader,
                num_classes=args.num_classes,
                args=args,
                device=device,
                epoch=epoch,
                prev_proto_bank=proto_bank,
            )

            if args.use_class_weight and epoch > args.warmup_epochs:
                update_interval = getattr(args, "class_weight_update_interval", 1)
                if update_interval > 0 and (epoch - args.warmup_epochs) % update_interval == 0:
                    was_training = model.training
                    model.eval()
                    class_weights = compute_class_weights(
                        model=model,
                        proto_bank=proto_bank,
                        loader_src=proto_src_loader if args.class_weight_use_src else None,
                        loader_tgt_l=proto_tgt_l_loader,
                        loader_tgt_u=proto_tgt_u_loader,
                        num_classes=args.num_classes,
                        device=device,
                        args=args,
                        alpha=args.class_weight_alpha,
                        eps=args.class_weight_eps,
                        use_src=args.class_weight_use_src,
                        use_confidence=not args.class_weight_no_conf,
                        gamma=getattr(args, "class_weight_gamma", 0.5),
                        max_scale=getattr(args, "class_weight_max_scale", 2.0),
                        min_scale=getattr(args, "class_weight_min_scale", 0.5),
                        manual_pos_scale=getattr(args, "manual_pos_scale", 1.0),
                    )
                    if was_training:
                        model.train()
                    class_weights_value = class_weights.detach().cpu().tolist()
                    class_weight_logs.append(
                        {"epoch": epoch, "weights": class_weights_value}
                    )

            stats = train_one_epoch(
                epoch, model,
                loader_src, loader_tgt_l, loader_tgt_u,
                optimizer, supcon_loss, args, proto_bank, device, class_weights, unsupcon_loss,
            )

            val_metrics = eval_on_split(
                model,
                val_loader,
                device=device,
                num_classes=args.num_classes,
            )
            hist_val_metrics.append({"epoch": epoch, **val_metrics})
            
            # 简单打印一下，防止输出太长
            if epoch % 10 == 0 or epoch == 1:
                print(
                    f"[Fold {fold_idx} | Epoch {epoch}] "
                    f"sup={stats['loss_sup']:.4f} "
                    f"pl={stats['loss_pl']:.4f} "
                    f"con={stats['loss_con']:.4f} "
                    f"pl_ratio={stats['pl_ratio']:.4f} "
                    f"val_acc={val_metrics['acc']:.4f} "
                    f"val_f1={val_metrics['f1']:.4f} "
                    f"val_auc={val_metrics['auc'] if val_metrics['auc'] is not None else 'None'}"
                )

            hist_sup.append(stats["loss_sup"])
            hist_pl.append(stats["loss_pl"])
            hist_con.append(stats["loss_con"])
            hist_pl_ratio.append(stats["pl_ratio"])
            
            val_score = val_metrics["f1"]
            if val_score > best_val_metric:
                best_val_metric = val_score
                best_val_record = {
                    "epoch": epoch,
                    "metric_name": "f1",
                    "metric_value": val_score,
                    "metrics": val_metrics,
                }
                torch.save(model.state_dict(), fold_run_dir / "best_model.pt")
                with open(fold_run_dir / "best_metrics.json", "w") as f:
                    json.dump(best_val_record, f, indent=2)

        # === 保存该 Fold 训练 log ===
        log_dict = {
            "sup": hist_sup,
            "pl": hist_pl,
            "con": hist_con,
            "pl_ratio": hist_pl_ratio,
            "val": hist_val_metrics,
        }
        with open(fold_run_dir / "train_log.json", "w") as f:
            json.dump(log_dict, f, indent=2)
        with open(fold_run_dir / "val_metrics.json", "w") as f:
            json.dump(hist_val_metrics, f, indent=2)
        if class_weight_logs:
            with open(fold_run_dir / "class_weight_log.json", "w") as f:
                json.dump(class_weight_logs, f, indent=2)

        # === 绘制曲线 (保存到 fold 目录) ===
        epochs_range = list(range(1, len(hist_sup) + 1))
        
        # Loss Curve
        curves = {
            "sup": np.array(hist_sup, dtype=float),
            "pl":  np.array(hist_pl,  dtype=float),
            "con": np.array(hist_con, dtype=float),
        }
        norm_curves = {}
        for name, arr in curves.items():
            arr_min, arr_max = arr.min(), arr.max()
            norm_curves[name] = (arr - arr_min) / (arr_max - arr_min) if arr_max > arr_min else np.zeros_like(arr)

        plt.figure()
        plt.plot(epochs_range, norm_curves["sup"], label="sup (norm)")
        plt.plot(epochs_range, norm_curves["pl"],  label="pl (norm)")
        plt.plot(epochs_range, norm_curves["con"], label="con (norm)")
        plt.xlabel("Epoch")
        plt.ylabel("Normalized value")
        plt.title(f"Fold {fold_idx} Loss Curves")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(fold_run_dir / "loss_curves.png", dpi=200)
        plt.close()

        # Ratio Curve
        plt.figure()
        plt.plot(epochs_range, hist_pl_ratio, label="pl_ratio")
        plt.xlabel("Epoch")
        plt.ylabel("pl_ratio")
        plt.title(f"Fold {fold_idx} Pseudo-label ratio")
        plt.ylim(0.0, 1.0)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(fold_run_dir / "pl_ratio_curve.png", dpi=200)
        plt.close()

        # === Test 评估 ===
        best_ckpt_path = fold_run_dir / "best_model.pt"
        if best_ckpt_path.exists():
            model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
            model.eval()
        else:
            logging.warning(
                "[Fold %s] best checkpoint not found at %s; using current model for test eval.",
                fold_idx,
                best_ckpt_path,
            )
        test_metrics = eval_on_split(model, test_loader, device=device, num_classes=args.num_classes)
        print(f"[Fold {fold_idx} TEST] "
              f"acc={test_metrics['acc']:.4f} "
              f"f1={test_metrics['f1']:.4f} "
              f"auc={test_metrics['auc'] if test_metrics['auc'] is not None else 'None'}")

        with open(fold_run_dir / "test_metrics.json", "w") as f:
            json.dump(test_metrics, f, indent=2)
        
        fold_metrics.append(test_metrics)

    # === 5折结束，计算平均结果 ===
    print(f"\n{'='*20} Cross Validation Results {'='*20}")
    if len(fold_metrics) > 0:
        accs = [m["acc"] for m in fold_metrics if m["acc"] is not None]
        f1s  = [m["f1"] for m in fold_metrics if m["f1"] is not None]
        
        avg_acc = np.mean(accs) if accs else 0.0
        std_acc = np.std(accs) if accs else 0.0
        avg_f1  = np.mean(f1s) if f1s else 0.0
        std_f1  = np.std(f1s) if f1s else 0.0
        
        print(f"Average Accuracy: {avg_acc:.4f} ± {std_acc:.4f}")
        print(f"Average F1 Score: {avg_f1:.4f} ± {std_f1:.4f}")
        
        final_summary = {
            "folds": len(fold_metrics),
            "avg_acc": avg_acc,
            "std_acc": std_acc,
            "avg_f1": avg_f1,
            "std_f1": std_f1,
            "details": fold_metrics
        }
        with open(run_dir / "final_summary.json", "w") as f:
            json.dump(final_summary, f, indent=2)
    else:
        print("No fold metrics available.")

    print(f"[Done] All results saved to: {run_dir}")

if __name__ == "__main__":
    main()

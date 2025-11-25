# src/train_main.py
# 模型没有修改，训练脚本添加了实验设置和结果保存的路径，exp_name 用于区分不同实验，不设置则以时间命名
# 采样方式不采用zip，改为iter+next的形式，避免目标域有标签数据集过小导致epoch提前结束
# 将伪标签数据纳入原型计算
import os
import argparse
from typing import Dict, Any
from pathlib import Path
from datetime import datetime
import json
import itertools

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
from src.prototypes.compute_prototype import compute_prototype
from src.prototypes.similarity import cosine_similarity_prototype
from src.pseudo.pseudo_labeler import make_pseudo_labels

'''
conda activate proto_mmda

python -m src.main \
  --src_pt /home/ad_group1/data/Pitt/fold1/train.pt \
  --tgt_l_pt /home/ad_group1/data/Lu/fold1/tgt_labeled.pt \
  --tgt_u_pt /home/ad_group1/data/Lu/fold1/tgt_unlabeled.pt \
  --val_pt   /home/ad_group1/data/Lu/fold1/val.pt \
  --test_pt  /home/ad_group1/data/Lu/fold1/test.pt \
  --epochs 500 --batch_size 64 \
  --lr 1e-5 \
'''


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-modal Semi-supervised DA with Gated Fusion + Prototype + SupCon"
    )
    # 数据路径 / 配置：离线特征 .pt
    parser.add_argument("--src_pt", type=str, required=True)
    parser.add_argument("--tgt_l_pt", type=str, required=True)
    parser.add_argument("--tgt_u_pt", type=str, required=True)
    parser.add_argument("--val_pt", type=str, required=True)
    parser.add_argument("--test_pt", type=str, required=True)

    # 模型超参（da/dt 会在 build_dataloaders 里用 .pt 里的 dim 覆盖）
    parser.add_argument("--da", type=int, default=1024)  # audio feat dim
    parser.add_argument("--dt", type=int, default=768)   # text  feat dim
    parser.add_argument("--dproj", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--gate_mode", type=str, default="channel", choices=["channel", "scalar"])
    parser.add_argument("--num_classes", type=int, default=2)

    # loss 权重
    parser.add_argument("--lam_sup", type=float, default=1.0)
    parser.add_argument("--lam_sup_src", type=float, default=1.0)
    parser.add_argument("--lam_sup_tgt", type=float, default=2.0)
    parser.add_argument("--lam_pl",  type=float, default=10.0)
    parser.add_argument("--lam_con", type=float, default=0.3)
    parser.add_argument("--lam_proto", type=float, default=0.0)
    parser.add_argument("--lam_ent", type=float, default=0.0)

    # 对比学习温度
    parser.add_argument("--supcon_temp", type=float, default=0.1)

    # 伪标签 / 原型相似度超参
    parser.add_argument("--tau_proto", type=float, default=20.0,
                        help="prototype similarity temperature")
    parser.add_argument("--theta_a", type=float, default=0.1,
                        help="audio 原型相似度阈值")
    parser.add_argument("--theta_t", type=float, default=0.1,
                        help="text 原型相似度阈值")
    parser.add_argument("--theta_h", type=float, default=0.1,
                        help="融合分类器高置信度阈值")
    parser.add_argument("--delta_a", type=float, default=0.1,
                        help="audio 模态 margin")
    parser.add_argument("--delta_t", type=float, default=0.1,
                        help="text 模态 margin")
    parser.add_argument("--kl_eps", type=float, default=0.2,
                        help="KL 一致性约束阈值")

    # 训练超参
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=10,
                        help="前多少个 epoch 只用监督/对比学习，不使用伪标签损失")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    # 结果保存
    parser.add_argument("--result_root", type=str, default="result",
                        help="结果根目录（每次运行会在下面创建一个子文件夹）")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="本次实验的名称；若不填则用时间戳自动生成")

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
    包装 offline_extract_features.py 生成的 *.pt 文件：
      - feats_audio: (N, Ca)
      - feats_text:  (N, Ct)
      - label:       (N,) 或 None
      - speaker:     list[str] 或 None
      - path_audio:  list[str]
      - text:        list[str]
    __getitem__ 返回: (Xa, Xt, y, meta)
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
    data = torch.load(pt_path, map_location="cpu")
    ds = OfflineFeatDataset(data)
    cfg = data.get("config", {}) or {}
    return ds, cfg


def build_dataloaders(args) -> Dict[str, DataLoader]:
    """
    读取 offline_extract_features.py 生成的 *.pt 特征，
    构建 DataLoader:
      - train_src_loader
      - train_tgt_l_loader
      - train_tgt_u_loader
      - val_loader
      - test_loader

    每个 batch 形如: (Xa, Xt, y, meta)
      - Xa: [B, da]
      - Xt: [B, dt]
      - y : [B]  (无标签集 y 全为 -1)
    """
    # 1) 加载各个 split 的 .pt
    src_ds,  cfg_src  = _load_split(args.src_pt)
    tgt_l_ds, cfg_tl  = _load_split(args.tgt_l_pt)
    tgt_u_ds, cfg_tu  = _load_split(args.tgt_u_pt)
    val_ds,  cfg_val  = _load_split(args.val_pt)
    test_ds, cfg_test = _load_split(args.test_pt)

    # 2) 简单 sanity check：audio/text 维度一致，并覆盖 args.da/args.dt
    da = cfg_src.get("audio_hidden_dim", src_ds.Xa.size(1))
    dt = cfg_src.get("text_hidden_dim",  src_ds.Xt.size(1))

    assert src_ds.Xa.size(1) == da and src_ds.Xt.size(1) == dt, "src 特征维度不一致"
    assert tgt_l_ds.Xa.size(1) == da and tgt_l_ds.Xt.size(1) == dt, "tgt_l 特征维度不一致"
    assert tgt_u_ds.Xa.size(1) == da and tgt_u_ds.Xt.size(1) == dt, "tgt_u 特征维度不一致"
    assert val_ds.Xa.size(1)   == da and val_ds.Xt.size(1)   == dt, "val 特征维度不一致"
    assert test_ds.Xa.size(1)  == da and test_ds.Xt.size(1)  == dt, "test 特征维度不一致"

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

    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False, drop_last=False, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=bs, shuffle=False, drop_last=False, num_workers=4, pin_memory=True
    )

    # 4) 目标域“有标+伪标”原型用的 loader（目前只有有标）
    train_tgt_all_loader = DataLoader(
        tgt_l_ds, batch_size=bs, shuffle=True, drop_last=True, num_workers=4, pin_memory=True
    )

    loaders = {
        "train_src": train_src_loader,
        "train_tgt_l": train_tgt_l_loader,
        "train_tgt_u": train_tgt_u_loader,
        "val": val_loader,
        "test": test_loader,
        "train_tgt_all": train_tgt_all_loader,
    }
    return loaders


# ==============
#  Train / Proto  每个 epoch 会完整扫一遍 loader_tgt_u，源域和目标有标通过 cycle 重复使用
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
) -> Dict[str, float]:
    """
    target-centric 版本：
      - 以目标域无标签 loader_tgt_u 为驱动（steps_per_epoch = len(loader_tgt_u)）
      - 源域 / 目标有标用 cycle(loader_src) / cycle(loader_tgt_l) 循环采样
      - 其它 loss 结构与原版一致
    """
    model.train()
    supcon_loss.train()

    # 统计量
    total_sup = total_pl = total_con = 0.0
    total_sup_src = 0.0
    total_sup_tgt = 0.0
    total_pl_n = 0          # 所有 step 里伪标签样本总数
    total_u_n = 0           # 所有 step 里目标无标样本总数
    n_steps = 0

    # 若没有单独的 lam_sup_src / lam_sup_tgt，就退回默认 1.0
    lam_sup_src = getattr(args, "lam_sup_src", 1.0)
    lam_sup_tgt = getattr(args, "lam_sup_tgt", 1.0)

    # ---- target-centric 采样：以目标无标为 driver，其它用 cycle ----
    iter_src   = itertools.cycle(loader_src)
    iter_tgt_l = itertools.cycle(loader_tgt_l)

    for batch_u in loader_tgt_u:
        n_steps += 1

        # 循环取源域、有标目标 batch
        batch_s = next(iter_src)
        batch_l = next(iter_tgt_l)

        Xa_s, Xt_s, y_s, meta_s = batch_s
        Xa_l, Xt_l, y_l, meta_l = batch_l
        Xa_u, Xt_u, y_u, meta_u = batch_u  # y_u 可能全是 -1

        Xa_s = Xa_s.to(device); Xt_s = Xt_s.to(device); y_s = y_s.to(device)
        Xa_l = Xa_l.to(device); Xt_l = Xt_l.to(device); y_l = y_l.to(device)
        Xa_u = Xa_u.to(device); Xt_u = Xt_u.to(device); y_u = y_u.to(device)

        optimizer.zero_grad()

        # 1) 前向
        logits_s, probs_s, (za_s, zt_s, zf_s, g_s) = model(Xa_s, Xt_s)
        logits_l, probs_l, (za_l, zt_l, zf_l, g_l) = model(Xa_l, Xt_l)
        logits_u, probs_u, (za_u, zt_u, zf_u, g_u) = model(Xa_u, Xt_u)

        # 2) 监督 CE（源 + 目标有标），拆开方便单独加权
        loss_sup_src = F.cross_entropy(logits_s, y_s)
        loss_sup_tgt = F.cross_entropy(logits_l, y_l)
        # 日志里仍记录“裸”的监督 loss 之和
        loss_sup = loss_sup_src + loss_sup_tgt

        # 3) 无标签：warmup 阶段不打伪标签；之后基于当前原型做伪标签
        is_warmup = epoch <= getattr(args, "warmup_epochs", 0)
        u_n = int(y_u.size(0))

        if is_warmup:
            mask_pl = torch.zeros_like(y_u, dtype=torch.bool)
            y_hat = torch.zeros_like(y_u, dtype=torch.long)
            w_hat = torch.zeros_like(y_u, dtype=torch.float)
            loss_pl = torch.tensor(0.0, device=device)
            pl_n = 0
        else:
            prototype_a = proto_bank["a"]  # [C,D]
            prototype_t = proto_bank["t"]
            prototype_f = proto_bank["f"]

            sim_a_u, _ = cosine_similarity_prototype(za_u, prototype_a, tau=args.tau_proto)
            sim_t_u, _ = cosine_similarity_prototype(zt_u, prototype_t, tau=args.tau_proto)
            sim_f_u, _ = cosine_similarity_prototype(zf_u, prototype_f, tau=args.tau_proto)

            mask_pl, y_hat, w_hat = make_pseudo_labels(
                probs_h=probs_u,
                sim_a=sim_a_u,
                sim_t=sim_t_u,
                sim_f=sim_f_u,
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
                ce_pl = F.cross_entropy(logits_pl, y_pl, reduction="none")
                loss_pl = (ce_pl * w_pl).mean()
            else:
                loss_pl = torch.tensor(0.0, device=device)

        total_pl_n += pl_n
        total_u_n += u_n

        # 4) 对比学习：只用 zf_*，y=-1 的不参与（SupCon 内部会过滤 -1）
        z_list = [zf_s, zf_l]
        y_list = [y_s,  y_l]
        if not is_warmup and mask_pl.any():
            z_list.append(zf_u[mask_pl])
            y_list.append(y_hat[mask_pl])
        z_all = torch.cat(z_list, dim=0)
        y_all = torch.cat(y_list, dim=0)

        loss_con = supcon_loss(z_all, y_all)

        # 5) 其他 loss（占位）
        loss_proto = torch.tensor(0.0, device=device)
        loss_ent   = torch.tensor(0.0, device=device)

        # ---- loss 加权（只改权重，不动结构）----
        # warmup 期间即便算了 loss_pl，这里权重仍为 0
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
        )

        loss.backward()
        optimizer.step()

        total_sup += loss_sup.item()
        total_sup_src += loss_sup_src.item()
        total_sup_tgt += loss_sup_tgt.item()
        total_pl  += loss_pl.item()
        total_con += loss_con.item()

    stats = {
        "loss_sup": total_sup / max(1, n_steps),
        "loss_sup_src": total_sup_src / max(1, n_steps),
        "loss_sup_tgt": total_sup_tgt / max(1, n_steps),
        "loss_pl":  total_pl  / max(1, n_steps),
        "loss_con": total_con / max(1, n_steps),
        "pl_ratio": (total_pl_n / max(1, total_u_n)),
        "pl_count": total_pl_n,
    }
    return stats


@torch.no_grad()
def build_proto_bank(
    model: nn.Module,
    loader_tgt_l: DataLoader,   # 目标域有标签
    loader_tgt_u: DataLoader,   # 目标域无标签（用来打伪标签）
    num_classes: int,
    args,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    用“目标域有标签 + 伪标签”重算原型：
      1) 先只用目标有标签数据算一版原型 proto_labeled
      2) 再用 proto_labeled 给无标签样本打伪标签，筛选出高置信伪标签样本
      3) 把 (有标签 + 伪标签) 一起用于加权求原型（伪标签可按 w_hat 赋权）
    """
    model.eval()

    # ====== 1) 先用目标有标签数据算一版原型 ======
    zf_lbl_list, za_lbl_list, zt_lbl_list, y_lbl_list = [], [], [], []

    for Xa, Xt, y, _ in loader_tgt_l:
        Xa, Xt, y = Xa.to(device), Xt.to(device), y.to(device)
        # 保险起见，过滤 y < 0 的样本
        mask = (y >= 0)
        if not mask.any():
            continue

        logits, probs, (za, zt, zf, g) = model(Xa, Xt)

        zf_lbl_list.append(zf[mask])
        za_lbl_list.append(za[mask])
        zt_lbl_list.append(zt[mask])
        y_lbl_list.append(y[mask].long())

    if len(y_lbl_list) == 0:
        raise RuntimeError("build_proto_bank: no labeled target samples (all y<0).")

    zf_lbl = torch.cat(zf_lbl_list, dim=0)
    za_lbl = torch.cat(za_lbl_list, dim=0)
    zt_lbl = torch.cat(zt_lbl_list, dim=0)
    y_lbl  = torch.cat(y_lbl_list,  dim=0)

    # 先只用有标签算一版“干净原型”
    proto_f_l = compute_prototype(zf_lbl, y_lbl, num_classes=num_classes, w=None)
    proto_a_l = compute_prototype(za_lbl, y_lbl, num_classes=num_classes, w=None)
    proto_t_l = compute_prototype(zt_lbl, y_lbl, num_classes=num_classes, w=None)

    # ====== 2) 用这版原型给无标签打伪标签 ======
    zf_p_list, za_p_list, zt_p_list = [], [], []
    y_p_list, w_p_list = [], []

    for Xa_u, Xt_u, y_u, _ in loader_tgt_u:
        Xa_u, Xt_u = Xa_u.to(device), Xt_u.to(device)

        logits_u, probs_u, (za_u, zt_u, zf_u, g_u) = model(Xa_u, Xt_u)

        # 计算与“干净原型”的相似度
        sim_a_u, _ = cosine_similarity_prototype(za_u, proto_a_l, tau=args.tau_proto)
        sim_t_u, _ = cosine_similarity_prototype(zt_u, proto_t_l, tau=args.tau_proto)
        sim_f_u, _ = cosine_similarity_prototype(zf_u, proto_f_l, tau=args.tau_proto)

        # 利用项目原有的伪标签规则
        mask_pl, y_hat, w_hat = make_pseudo_labels(
            probs_h=probs_u,
            sim_a=sim_a_u,
            sim_t=sim_t_u,
            sim_f=sim_f_u,
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

    # 如果本轮没有任何伪标签，就退回纯有标签原型
    if len(y_p_list) == 0:
        return {"f": proto_f_l, "a": proto_a_l, "t": proto_t_l}

    # ====== 3) 有标签 + 伪标签一起加权求原型 ======
    zf_p = torch.cat(zf_p_list, dim=0)
    za_p = torch.cat(za_p_list, dim=0)
    zt_p = torch.cat(zt_p_list, dim=0)
    y_p  = torch.cat(y_p_list,  dim=0)
    w_p  = torch.cat(w_p_list,  dim=0)

    # 有标签样本权重设为 1，伪标签用 w_hat 作为置信度权重
    w_lbl = torch.ones_like(y_lbl, dtype=torch.float, device=device)

    zf_all = torch.cat([zf_lbl, zf_p], dim=0)
    za_all = torch.cat([za_lbl, za_p], dim=0)
    zt_all = torch.cat([zt_lbl, zt_p], dim=0)
    y_all  = torch.cat([y_lbl,  y_p],  dim=0)
    w_all  = torch.cat([w_lbl,  w_p],  dim=0)

    # 若 compute_prototype 支持 w，就按 w_all 做加权原型；不支持的话你也可以改成 w=None
    proto_f = compute_prototype(zf_all, y_all, num_classes=num_classes, w=w_all)
    proto_a = compute_prototype(za_all, y_all, num_classes=num_classes, w=w_all)
    proto_t = compute_prototype(zt_all, y_all, num_classes=num_classes, w=w_all)

    return {"f": proto_f, "a": proto_a, "t": proto_t}



@torch.no_grad()
def eval_on_split(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int = 2,
) -> Dict[str, Any]:
    """
    在给定 split 上评估：
      - 只使用 y >= 0 的样本
      - 返回 loss, acc, f1, auc, cm
    """
    model.eval()

    all_labels = []
    all_scores = []  # 对正类(1)的概率
    total_loss = 0.0
    total_n = 0

    for Xa, Xt, y, _ in loader:
        Xa, Xt, y = Xa.to(device), Xt.to(device), y.to(device)
        # 过滤掉 y < 0（无标签）
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
            # 二分类：取正类(1)的概率
            scores = probs[:, 1].detach().cpu().numpy()
        else:
            # 多分类：这里简单取 argmax 作为“置信度”，也可以改成 one-vs-rest AUC
            scores = torch.softmax(logits, dim=-1).max(dim=-1).values.detach().cpu().numpy()

        labels_np = y_valid.detach().cpu().numpy()
        all_labels.append(labels_np)
        all_scores.append(scores)

    if total_n == 0:
        return {"loss": None, "acc": None, "f1": None, "auc": None, "cm": None}

    all_labels = np.concatenate(all_labels, axis=0)
    all_scores = np.concatenate(all_scores, axis=0)

    # prediction：简单用 0.5 阈值（针对二分类）
    if num_classes == 2:
        y_pred = (all_scores >= 0.5).astype(int)
    else:
        # 多分类时，这里的 all_scores 是 max prob，不适合直接分类，这里给个占位
        # 正式多分类可改成 argmax logits 逻辑
        raise NotImplementedError("eval_on_split currently assumes binary classification.")

    acc = accuracy_score(all_labels, y_pred)
    f1  = f1_score(all_labels, y_pred)

    # AUC：只有在标签中同时出现 0 和 1 时才计算
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

    # === 结果目录 ===
    result_root = Path(args.result_root)
    result_root.mkdir(parents=True, exist_ok=True)

    if args.exp_name is None:
        exp_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    else:
        exp_name = args.exp_name

    run_dir = result_root / exp_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[result] save to: {run_dir}")

    # 保存 args
    with open(run_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # 1) DataLoader
    loaders = build_dataloaders(args)
    loader_src      = loaders["train_src"]
    loader_tgt_l    = loaders["train_tgt_l"]
    loader_tgt_u    = loaders["train_tgt_u"]
    loader_tgt_all  = loaders["train_tgt_all"]  # 有标签+伪标签（目前只有有标签）
    val_loader      = loaders["val"]
    test_loader     = loaders["test"]

    # 2) 模型 & 优化器 & loss
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

    # 3) 初始原型（先用“目标有标 + 伪标签”机制）
    proto_bank = build_proto_bank(
        model,
        loader_tgt_l=loader_tgt_l,
        loader_tgt_u=loader_tgt_u,
        num_classes=args.num_classes,
        args=args,
        device=device,
    )
    
    # 记录曲线
    hist_sup = []
    hist_pl = []
    hist_con = []
    hist_pl_ratio = []

    # 4) 训练循环
    for epoch in range(1, args.epochs + 1):
        # 每个 epoch 用“上一轮的伪标签结果”重算一次目标域原型
        # （当前版本 loader_tgt_all 还只有有标签样本）
        proto_bank = build_proto_bank(
            model,
            loader_tgt_l=loader_tgt_l,
            loader_tgt_u=loader_tgt_u,
            num_classes=args.num_classes,
            args=args,
            device=device,
        )

        stats = train_one_epoch(
            epoch, model,
            loader_src, loader_tgt_l, loader_tgt_u,
            optimizer, supcon_loss, args, proto_bank, device,
        )
        print(
            f"[Epoch {epoch}] "
            f"sup={stats['loss_sup']:.4f} "
            f"(src={stats['loss_sup_src']:.4f}, tgt={stats['loss_sup_tgt']:.4f}) "
            f"| pl={stats['loss_pl']:.4f} "
            f"| con={stats['loss_con']:.4f} "
            f"| pl_ratio={stats['pl_ratio']:.4f} "
            f"({stats['pl_count']} pseudo)"
)

        hist_sup.append(stats["loss_sup"])
        hist_pl.append(stats["loss_pl"])
        hist_con.append(stats["loss_con"])
        hist_pl_ratio.append(stats["pl_ratio"])

        # 如需每 epoch 在 val 上验证，可以在此调用 eval_on_split(model, val_loader, ...)

    # === 保存训练 log ===
    log_dict = {
        "sup": hist_sup,
        "pl": hist_pl,
        "con": hist_con,
        "pl_ratio": hist_pl_ratio,
    }
    with open(run_dir / "train_log.json", "w") as f:
        json.dump(log_dict, f, indent=2)

    # === 绘制曲线 ===
    epochs = list(range(1, len(hist_sup) + 1))

    # 1) loss 曲线（先做 min-max 归一化，避免尺度差异压扁）
    curves = {
        "sup": np.array(hist_sup, dtype=float),
        "pl":  np.array(hist_pl,  dtype=float),
        "con": np.array(hist_con, dtype=float),
    }

    norm_curves = {}
    for name, arr in curves.items():
        arr_min = arr.min()
        arr_max = arr.max()
        if arr_max > arr_min:
            arr_norm = (arr - arr_min) / (arr_max - arr_min)
        else:
            # 全程几乎不变的情况，直接设成 0
            arr_norm = np.zeros_like(arr)
        norm_curves[name] = arr_norm

    plt.figure()
    plt.plot(epochs, norm_curves["sup"], label="sup (norm)")
    plt.plot(epochs, norm_curves["pl"],  label="pl (norm)")
    plt.plot(epochs, norm_curves["con"], label="con (norm)")
    plt.xlabel("Epoch")
    plt.ylabel("Normalized value")
    plt.title("Training Loss Curves (min-max normalized)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(run_dir / "loss_curves.png", dpi=200)
    plt.close()


    # 2) pl_ratio 曲线
    plt.figure()
    plt.plot(epochs, hist_pl_ratio, label="pl_ratio")
    plt.xlabel("Epoch")
    plt.ylabel("pl_ratio")
    plt.title("Pseudo-label ratio over epochs")
    plt.ylim(0.0, 1.0)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(run_dir / "pl_ratio_curve.png", dpi=200)
    plt.close()

    # === 在 test 上做最终评估 ===
    test_metrics = eval_on_split(model, test_loader, device=device, num_classes=args.num_classes)
    print("[TEST] "
          f"loss={test_metrics['loss']:.4f} "
          f"acc={test_metrics['acc']:.4f} "
          f"f1={test_metrics['f1']:.4f} "
          f"auc={test_metrics['auc'] if test_metrics['auc'] is not None else 'None'} "
          f"cm={test_metrics['cm']}")

    with open(run_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)


if __name__ == "__main__":
    main()

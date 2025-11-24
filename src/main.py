# src/train_main.py
import os
import argparse
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

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
  --epochs 50 --batch_size 64
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
    parser.add_argument("--lam_pl",  type=float, default=1.0)
    parser.add_argument("--lam_con", type=float, default=0.1)
    parser.add_argument("--lam_proto", type=float, default=0.0)
    parser.add_argument("--lam_ent", type=float, default=0.0)

    # 对比学习温度
    parser.add_argument("--supcon_temp", type=float, default=0.1)

    # 伪标签 / 原型相似度超参
    parser.add_argument("--tau_proto", type=float, default=20.0,
                        help="prototype similarity temperature")
    parser.add_argument("--theta_a", type=float, default=0.3,
                        help="audio 原型相似度阈值")
    parser.add_argument("--theta_t", type=float, default=0.3,
                        help="text 原型相似度阈值")
    parser.add_argument("--theta_h", type=float, default=0.5,
                        help="融合分类器高置信度阈值")
    parser.add_argument("--delta_a", type=float, default=0.1,
                        help="audio 模态 margin")
    parser.add_argument("--delta_t", type=float, default=0.1,
                        help="text 模态 margin")
    parser.add_argument("--kl_eps", type=float, default=0.2,
                        help="KL 一致性约束阈值")

    # 训练超参
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup_epochs", type=int, default=10,
                        help="前多少个 epoch 只用监督/对比学习，不使用伪标签损失")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

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

    # 4) 目标域“有标+伪标”原型用的 loader
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
) -> Dict[str, float]:
    model.train()
    supcon_loss.train()

    total_sup = total_pl = total_con = 0.0
    total_pl_n = 0
    total_u_n = 0
    n_steps = 0

    # 简单写法：假设三个 loader 长度差不多，用 zip 迭代
    for batch_s, batch_l, batch_u in zip(loader_src, loader_tgt_l, loader_tgt_u):
        n_steps += 1
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

        # 2) 监督 CE（源 + 目标有标）
        loss_sup = F.cross_entropy(logits_s, y_s) + F.cross_entropy(logits_l, y_l)

        # 3) 无标签：warmup 阶段不打伪标签；之后基于当前原型做伪标签
        is_warmup = epoch <= getattr(args, "warmup_epochs", 0)
        u_n = int(y_u.size(0))

        if is_warmup:
            # warmup 阶段：完全不使用伪标签
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
        if mask_pl.any():
            z_list.append(zf_u[mask_pl])
            y_list.append(y_hat[mask_pl])
        z_all = torch.cat(z_list, dim=0)
        y_all = torch.cat(y_list, dim=0)

        loss_con = supcon_loss(z_all, y_all)

        # TODO: 5) 如果还有 proto loss / entropy loss，可以在这里加
        loss_proto = torch.tensor(0.0, device=device)
        loss_ent   = torch.tensor(0.0, device=device)

        loss = (args.lam_sup * loss_sup +
                args.lam_pl  * loss_pl  +
                args.lam_con * loss_con +
                args.lam_proto * loss_proto +
                args.lam_ent   * loss_ent)

        loss.backward()
        optimizer.step()

        total_sup += loss_sup.item()
        total_pl  += loss_pl.item()
        total_con += loss_con.item()

    stats = {
        "loss_sup": total_sup / max(1, n_steps),
        "loss_pl":  total_pl  / max(1, n_steps),
        "loss_con": total_con / max(1, n_steps),
        "pl_ratio": (total_pl_n / max(1, total_u_n)),
        "pl_count": total_pl_n,
    }
    return stats


@torch.no_grad()
def build_proto_bank(
    model: nn.Module,
    loader_tgt_all: DataLoader,   # 目标域“有标+伪标”数据的 loader
    num_classes: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    每轮 epoch 结束后，用【目标域有标签 + 可信伪标签】重算原型。
    这里简单版本：假设 loader_tgt_all 中 y 已经包含真实 + 伪标签（未使用样本 y=-1）。
    """
    model.eval()
    zf_all, za_all, zt_all, y_all = [], [], [], []

    for Xa, Xt, y, _ in loader_tgt_all:
        Xa, Xt, y = Xa.to(device), Xt.to(device), y.to(device)
        logits, probs, (za, zt, zf, g) = model(Xa, Xt)
        zf_all.append(zf)
        za_all.append(za)
        zt_all.append(zt)
        y_all.append(y)

    zf_all = torch.cat(zf_all, dim=0)
    za_all = torch.cat(za_all, dim=0)
    zt_all = torch.cat(zt_all, dim=0)
    y_all  = torch.cat(y_all, dim=0).long()

    # 只用 y != -1 的样本算原型（-1 表示忽略）
    mask = (y_all != -1)
    if mask.sum() == 0:
        raise RuntimeError("build_proto_bank: no valid labels (all -1).")

    zf_lbl = zf_all[mask]
    za_lbl = za_all[mask]
    zt_lbl = zt_all[mask]
    y_lbl  = y_all[mask]

    proto_f = compute_prototype(zf_lbl, y_lbl, num_classes=num_classes, w=None)
    proto_a = compute_prototype(za_lbl, y_lbl, num_classes=num_classes, w=None)
    proto_t = compute_prototype(zt_lbl, y_lbl, num_classes=num_classes, w=None)

    return {"f": proto_f, "a": proto_a, "t": proto_t}


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[device] use {device}")

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

    # 3) 初始原型（先只用目标域有标签）
    proto_bank = build_proto_bank(model, loader_tgt_l, num_classes=args.num_classes, device=device)

    # 4) 训练循环
    for epoch in range(1, args.epochs + 1):
        # 每个 epoch 用“上一轮的伪标签结果”重算一次目标域原型
        proto_bank = build_proto_bank(model, loader_tgt_all, num_classes=args.num_classes, device=device)

        stats = train_one_epoch(
            epoch, model,
            loader_src, loader_tgt_l, loader_tgt_u,
            optimizer, supcon_loss, args, proto_bank, device,
        )
        print(f"[Epoch {epoch}] "
              f"sup={stats['loss_sup']:.4f} | pl={stats['loss_pl']:.4f} "
              f"| con={stats['loss_con']:.4f} | pl_ratio={stats['pl_ratio']:.4f} "
              f"({stats['pl_count']} pseudo)")

        # TODO: 这里加验证集评估、保存 best ckpt 等

    # TODO: 训练结束后在 test_loader 上做最终评估

if __name__ == "__main__":
    main()

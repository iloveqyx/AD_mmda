# src/models/fusion_Gated_attn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(p=2, dim=-1, keepdim=True) + eps)

class Projector(nn.Module):
    """Linear -> LayerNorm """
    def __init__(self, din: int, d: int):
        super().__init__()
        self.proj = nn.Linear(din, d)
        self.ln = nn.LayerNorm(d)
    def forward(self, x):
        return self.ln(self.proj(x))

class GatedCrossAttention(nn.Module):
    def __init__(
        self,
        da: int,#audio特征
        dt: int,#text
        d: int = 256,#特征投影维度，根据郑翔文的特征提取代码
        nhead: int = 4,#MHA头数
        pdrop: float = 0.1,#随机置零一部分注意力，防止过度集中在某一位置，后期发现效果不好就调大，可能是数据过少的原因。
        gate_mode: str = "channel",  # "channel" | "scalar"
    ):
        super().__init__()
        self.pa = Projector(da, d)
        self.pt = Projector(dt, d)
        # 两个方向的多头注意力
        self.mha_a2t = nn.MultiheadAttention(d, nhead, dropout=pdrop, batch_first=True)
        self.mha_t2a = nn.MultiheadAttention(d, nhead, dropout=pdrop, batch_first=True)

        # 门控：用融合前后特征拼接来产生逐通道门控
        g_in = d * 4  # [za, zt, a2t, t2a] 4倍维度
        self.g_mlp = nn.Sequential(
            nn.Linear(g_in, d),# 输入维度4d，输出维度d
            nn.ReLU(inplace=True),
            nn.Dropout(pdrop),
            nn.Linear(d, d)  # d->d
        )

        # 残差融合：把原始 (za, zt) 拼接后再线性映射并与门控融合结果相加
        self.residual_fuse = nn.Linear(2 * d, d)
        self.out_ln = nn.LayerNorm(d)
        self.dropout = nn.Dropout(pdrop)

        assert gate_mode in ("channel", "scalar")
        self.gate_mode = gate_mode

    def _ensure_seq(self, x):
        # 接受 [B, D] 或 [B, L, D]，统一成 [B, L, D]
        if x.dim() == 2:
            return x.unsqueeze(1), True #[B, 1, D]
        return x, False

    def forward(self, xa: torch.Tensor, xt: torch.Tensor):
        # 1) 投影 + LN
        xa, a_was_vec = self._ensure_seq(xa)  # [B, La=1, Da] or [B, La, Da]
        xt, t_was_vec = self._ensure_seq(xt)  # [B, Lt=1, Dt] or [B, Lt, Dt]
        za = self.pa(xa)  # [B, La, d]
        zt = self.pt(xt)  # [B, Lt, d]

        # 2) 双向交叉注意力
        #   a->t: 以 za 为 Query，zt 为 Key/Value，让“音频”从“文本”取信息
        a2t, _ = self.mha_a2t(query=za, key=zt, value=zt, need_weights=False)  # [B, La, d]
        #   t->a: 以 zt 为 Query，za 为 Key/Value，让“文本”从“音频”取信息
        t2a, _ = self.mha_t2a(query=zt, key=za, value=za, need_weights=False)  # [B, Lt, d]

        # === 统一对齐，把所有张量统一对齐到 L = max(La, Lt) ===
        def _align_to(x: torch.Tensor, L: int) -> torch.Tensor:
            B, Lx, D = x.shape
            if Lx == L:
                return x
            if Lx == 1:
                return x.expand(B, L, D)
            # 历史上我们经常把两边都压成单 token 后做融合，这里给一个安全的平均池化
            return x.mean(dim=1, keepdim=True).expand(B, L, D)

        La, Lt = za.size(1), zt.size(1)
        L = max(La, Lt)

        # 统一到同一时间长度 L（避免后续 cat / 逐元素运算维度不匹配）
        za_L  = _align_to(za,  L)   # [B, L, d]
        zt_L  = _align_to(zt,  L)   # [B, L, d]
        a2t_L = _align_to(a2t, L)   # [B, L, d]
        t2a_L = _align_to(t2a, L)   # [B, L, d]

        # 3) 逐通道门控
        # 拼接 [za, zt_aligned, a2t_aligned, t2a_aligned] 产生 gate
        assert za_L.size(1) == zt_L.size(1) == a2t_L.size(1) == t2a_L.size(1), \
            "time dim mismatch before gating"
        cat = torch.cat([za_L, zt_L, a2t_L, t2a_L], dim=-1)  # [B, L, 4d]
        g_raw = self.g_mlp(cat)  # [B, L, d]
        if self.gate_mode == "scalar":
            # 标量门控（每样本/每步一个 α）
            g = torch.sigmoid(g_raw.mean(dim=-1, keepdim=True))  # [B, L, 1]
        else:
            # 逐通道门控
            g = torch.sigmoid(g_raw)  # [B, L, d]

        # 4) 融合 + 残差 + LN
        # 对齐时间维度后做凸组合
        #g是门控赋予两个模态的权重  zf=g⊙a2t+(1−g)⊙t2a+Res([za,zt])
        fused = g * a2t_L + (1.0 - g) * t2a_L           # [B, L, d] 或 [B, L, 1]广播到 d
        res   = self.residual_fuse(torch.cat([za_L, zt_L], dim=-1))  # [B, L, d]
        fused = fused + self.dropout(res)
        fused = self.out_ln(fused)
        zf = fused  # [B, L, d]

        # 若输出为一维向量，则 squeeze 掉时间维度，我感觉我们用不到
        if a_was_vec and t_was_vec and zf.size(1) == 1:
            za_L = za_L.squeeze(1); zt_L = zt_L.squeeze(1); zf = zf.squeeze(1); g = g.squeeze(1)

        # L2 归一化,便于后续对比学习
        za_out = l2_normalize(za_L if zf.dim()==3 else za_L)  # 保持与上文一致的变量名
        zt_out = l2_normalize(zt_L if zf.dim()==3 else zt_L)
        zf     = l2_normalize(zf)
        return za_out, zt_out, zf, g

class GatedAttnFusionNet(nn.Module):
    #后续主要模型代码调用这个类就行，import ./fusion_Gated_attn.GatedAttnFusionNet
    def __init__(self, da: int, dt: int, dproj: int = 256, nhead: int = 4, pdrop: float = 0.1, gate_mode: str = "channel"):
        super().__init__()
        self.block = GatedCrossAttention(da, dt, dproj, nhead=nhead, pdrop=pdrop, gate_mode=gate_mode)
        self.out_dim = dproj
    def forward(self, Xa: torch.Tensor, Xt: torch.Tensor):
        za, zt, zf, g = self.block(Xa, Xt)
        return za, zt, zf, g

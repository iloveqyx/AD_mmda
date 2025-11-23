# -*- coding: utf-8 -*-
"""
tools/offline_extract_features.py
适配 CSV 列：path,label,spk,text 的离线特征抽取脚本
- 音频：Wav2Vec2 XLSR-53，last_hidden_state 均值池化
- 文本：mBERT，取 [CLS]
- 无需时间切片：CSV 的 path 已指向片段 wav
- 输出：{src,tgt_lab,tgt_unl,val,test}.pt，包含 feats_audio/feats_text/label/speaker/path_audio/text/config
"""

import os
import math
import argparse
import time
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# --------- 可选音频库（至少安装任意一个即可） ----------
_HAS_LIBROSA = False
_HAS_SF = False
_HAS_SCIPY = False
try:
    import librosa
    _HAS_LIBROSA = True
except Exception:
    pass
try:
    import soundfile as sf
    _HAS_SF = True
except Exception:
    pass
try:
    from scipy.signal import resample_poly
    _HAS_SCIPY = True
except Exception:
    pass

# --------- Transformers 依赖 ----------
from transformers import AutoTokenizer, AutoModel, Wav2Vec2Model
try:
    from transformers import AutoFeatureExtractor
except Exception:
    try:
        from transformers import Wav2Vec2FeatureExtractor as AutoFeatureExtractor
    except Exception as _e:
        raise ImportError(
            "Neither AutoFeatureExtractor nor Wav2Vec2FeatureExtractor is available. "
            "Please install/upgrade transformers >= 4.26."
        )

# ---------------------------
# 工具函数
# ---------------------------
def try_read_csv(path: str) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path)

def find_col(df: pd.DataFrame, pref: List[str], fallback: List[str]) -> Optional[str]:
    """优先匹配 pref 名称；若无则在 fallback 名称中查找（大小写不敏感）"""
    lower2orig = {c.lower(): c for c in df.columns}
    for n in pref + fallback:
        if n.lower() in lower2orig:
            return lower2orig[n.lower()]
    return None

def resample_to_sr(x: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return x
    if _HAS_LIBROSA:
        # 高质量重采样
        return librosa.resample(x, orig_sr=sr, target_sr=target_sr,
                                res_type=("soxr_vhq" if hasattr(librosa, "resample") else "kaiser_best"))
    if _HAS_SCIPY:
        g = math.gcd(sr, target_sr)
        up, down = target_sr // g, sr // g
        return resample_poly(x, up, down)
    # 兜底线性插值
    t = np.arange(len(x)) / float(sr)
    t_new = np.arange(int(len(x) * target_sr / sr)) / float(target_sr)
    return np.interp(t_new, t, x).astype(np.float32)

def load_audio_any(path: str, target_sr: int = 16000, mono: bool = True) -> np.ndarray:
    if _HAS_LIBROSA:
        wav, sr = librosa.load(path, sr=None, mono=mono)
    elif _HAS_SF:
        wav, sr = sf.read(path, always_2d=False)
        if mono and wav.ndim == 2:
            wav = wav.mean(axis=1)
        if not np.issubdtype(wav.dtype, np.floating):
            wav = wav.astype(np.float32) / np.iinfo(wav.dtype).max
    else:
        raise RuntimeError("无法读取音频，请安装 librosa 或 soundfile：pip install librosa soundfile")
    if sr != target_sr:
        wav = resample_to_sr(wav, sr, target_sr)
    if mono and wav.ndim == 2:
        wav = wav.mean(axis=1)
    return wav.astype(np.float32, copy=False)

def pad_or_crop(wav: np.ndarray, sr: int, seconds: float) -> np.ndarray:
    """seconds<=0 表示保留整段；>0 表示右侧补零/裁剪到固定长度"""
    if seconds is None or seconds <= 0:
        return wav
    L = int(seconds * sr + 0.5)
    if len(wav) >= L:
        return wav[:L]
    out = np.zeros(L, dtype=np.float32)
    out[:len(wav)] = wav
    return out

def chunks(total: int, bs: int):
    for i in range(0, total, bs):
        yield slice(i, min(i + bs, total))

# ---------------------------
# 标签规范化（Control=0，AD/Dementia=1，MCI 默认忽略）
# ---------------------------
def normalize_labels(series: pd.Series, label_map_json: str, mci_policy: str, strict: bool) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    返回 (labels:int64, mapping:dict[lower->int])
    - 整列可数值化 -> 直接转 int
    - 否则：默认别名 + 用户 JSON 映射；mci_policy={pos,neg,ignore}
    - strict=True：出现未覆盖的原始标签时抛错
    - ignore -> -1（训练阶段会过滤）
    """
    s_raw = series.astype(str)

    # 纯数字优先
    num = pd.to_numeric(s_raw, errors="coerce")
    if num.notna().all():
        vals = num.astype("int64").to_numpy()
        mapping = {str(int(v)): int(v) for v in sorted(num.unique())}
        return vals, mapping

    s = s_raw.str.strip().str.lower()

    # 用户映射
    user_map: Dict[str, int] = {}
    if label_map_json:
        import json as _json
        user_map = {str(k).strip().lower(): int(v) for k, v in _json.loads(label_map_json).items()}

    # 默认别名（保守，不把 patient 并到正类）
    default_map = {
        "control": 0, "hc": 0, "healthy": 0, "normal": 0, "nonpatient": 0, "non-ad": 0,
        "ad": 1, "alz": 1, "alzheimer": 1, "alzheimers": 1, "dementia": 1, "demented": 1,
        "pos": 1, "positive": 1, "neg": 0, "negative": 0,
        # mci 由策略决定
    }
    if mci_policy == "pos":
        default_map["mci"] = 1
    elif mci_policy == "neg":
        default_map["mci"] = 0
    else:  # ignore
        default_map["mci"] = -1

    mapping = {**default_map, **user_map}

    uniq = list(pd.unique(s))
    if strict:
        unknown = [u for u in uniq if u not in mapping]
        if unknown:
            raise ValueError(f"[strict_label_map] 未覆盖的原始标签: {unknown}. "
                             f"请用 --label_map_json 或调整 --mci_policy。")

    arr = np.asarray([mapping.get(u, -1) for u in s], dtype="int64")
    return arr, mapping

# ---------------------------
# 特征抽取
# ---------------------------
@torch.no_grad()
def extract_audio_features(
    paths: List[str],
    feature_extractor: AutoFeatureExtractor,
    model: Wav2Vec2Model,
    sr: int,
    seconds: float,
    device: torch.device,
    batch_size: int = 64,
    amp: bool = False,
) -> np.ndarray:
    model.eval()
    feats = []
    iters = list(chunks(len(paths), batch_size))
    pbar = tqdm(iters, desc="Audio feats", leave=False)
    for sl in pbar:
        batch_wavs = []
        for p in paths[sl]:
            if not os.path.isfile(p):
                raise FileNotFoundError(f"音频不存在: {p}")
            wav = load_audio_any(p, target_sr=sr, mono=True)
            wav = pad_or_crop(wav, sr, seconds)
            batch_wavs.append(wav)

        ipt = feature_extractor(batch_wavs, sampling_rate=sr, return_tensors="pt", padding=True)
        with torch.autocast(device_type=("cuda" if device.type == "cuda" else "cpu"),
                            dtype=torch.float16, enabled=amp):
            out = model(
                input_values=ipt["input_values"].to(device),
                attention_mask=(ipt.get("attention_mask", None).to(device) if ipt.get("attention_mask", None) is not None else None),
                return_dict=True
            ).last_hidden_state  # (B,T,C)
            pooled = out.mean(dim=1)  # (B,C)
        feats.append(pooled.cpu().float().numpy())
        del ipt, out, pooled
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if len(feats) == 0:
        return np.zeros((0, model.config.hidden_size), dtype=np.float32)
    return np.concatenate(feats, axis=0)

@torch.no_grad()
def extract_text_features(
    texts: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    max_len: int,
    device: torch.device,
    batch_size: int = 64,
    amp: bool = False,
) -> np.ndarray:
    model.eval()
    feats = []
    texts = [("" if (t is None or (isinstance(t, float) and np.isnan(t))) else str(t)) for t in texts]
    iters = list(chunks(len(texts), batch_size))
    pbar = tqdm(iters, desc="Text feats", leave=False)
    for sl in pbar:
        tk = tokenizer(texts[sl], padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        with torch.autocast(device_type=("cuda" if device.type == "cuda" else "cpu"),
                            dtype=torch.float16, enabled=amp):
            out = model(
                input_ids=tk["input_ids"].to(device),
                attention_mask=tk["attention_mask"].to(device),
                return_dict=True
            ).last_hidden_state  # (B,T,H)
            cls = out[:, 0, :]  # [CLS]
        feats.append(cls.cpu().float().numpy())
        del tk, out, cls
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if len(feats) == 0:
        return np.zeros((0, model.config.hidden_size), dtype=np.float32)
    return np.concatenate(feats, axis=0)

# ---------------------------
# 参数与环境
# ---------------------------
def build_argparser():
    p = argparse.ArgumentParser("Offline feature extractor for CSV(path,label,spk,text)")
    # 数据 split（可直接换成你的实际 CSV）
    p.add_argument("--src_csv",      type=str, default="/home/ad_group1/mmda_mme/data/Pitt_622_k5_speaker/fold1/train.csv")
    p.add_argument("--tgt_lab_csv",  type=str, default="/home/ad_group1/mmda_mme/data/Dem_622_k5_speaker/fold1/tgt_labeled.csv")
    p.add_argument("--tgt_unl_csv",  type=str, default="/home/ad_group1/mmda_mme/data/Dem_622_k5_speaker/fold1/tgt_unlabeled.csv")
    p.add_argument("--val_csv",      type=str, default="/home/ad_group1/mmda_mme/data/Dem_622_k5_speaker/fold1/val.csv")
    p.add_argument("--test_csv",     type=str, default="/home/ad_group1/mmda_mme/data/Dem_622_k5_speaker/fold1/test.csv")

    # 模型本地目录（离线）
    p.add_argument("--wav2vec_local_dir", type=str, default="./hf/wav2vec2-large-xlsr-53")
    p.add_argument("--bert_local_dir",    type=str, default="./hf/bert-base-multilingual-cased")

    # HF 离线/缓存/端点
    if hasattr(argparse, "BooleanOptionalAction"):
        p.add_argument("--hf_local_only", action=argparse.BooleanOptionalAction, default=True,
                       help="默认离线；可用 --no-hf-local-only 关闭")
    else:
        p.add_argument("--hf_local_only", action="store_true", default=True)
    p.add_argument("--hf_cache", type=str, default="")
    p.add_argument("--hf_endpoint", type=str, default="")

    # 列名控制（优先 exact 列名）
    p.add_argument("--path_col",   type=str, default="path")
    p.add_argument("--label_col",  type=str, default="label")
    p.add_argument("--speaker_col",type=str, default="spk")
    p.add_argument("--text_col",   type=str, default="text")

    # 标签策略
    p.add_argument("--mci_policy", type=str, default="ignore", choices=["pos","neg","ignore"],
                   help="MCI 视为：pos=正类，neg=负类，ignore=丢弃(-1)")
    p.add_argument("--strict_label_map", action="store_true",
                   help="若提供 --label_map_json，则未在映射中的原始标签直接报错")
    p.add_argument("--label_map_json", type=str, default="", help='JSON，如 {"Control":0,"AD":1,"MCI":-1}')

    # 音频/文本处理
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--seconds",     type=float, default=2.0, help="<=0 使用整段片段；>0 固定裁剪/补零到该长度")
    p.add_argument("--max_text_len", type=int, default=64)

    # 抽特征 batch 与设备
    p.add_argument("--batch_audio", type=int, default=64)
    p.add_argument("--batch_text",  type=int, default=64)
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--amp", action="store_true", help="推理混合精度（默认关闭）")

    # 输出目录
    p.add_argument("--out_dir", type=str, default="./offline_feats/Pitt2Dem", help="输出目录")
    return p

def prepare_env(args):
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
    if args.hf_cache:
        os.environ["HF_HOME"] = args.hf_cache
        os.environ["TRANSFORMERS_CACHE"] = args.hf_cache
    if args.hf_local_only:
        os.environ["HF_HUB_OFFLINE"] = "1"

def load_models(args, device: torch.device):
    feat = AutoFeatureExtractor.from_pretrained(args.wav2vec_local_dir, local_files_only=True)
    aenc = Wav2Vec2Model.from_pretrained(args.wav2vec_local_dir, local_files_only=True).to(device).eval()
    tok  = AutoTokenizer.from_pretrained(args.bert_local_dir, local_files_only=True, use_fast=True)
    tenc = AutoModel.from_pretrained(args.bert_local_dir, local_files_only=True).to(device).eval()
    return feat, aenc, tok, tenc

# ---------------------------
# 处理一个 split
# ---------------------------
def run_one_split(
    csv_path: str,
    feat: AutoFeatureExtractor,
    aenc: Wav2Vec2Model,
    tok: AutoTokenizer,
    tenc: AutoModel,
    args,
    device: torch.device
) -> Dict[str, torch.Tensor]:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV 不存在：{csv_path}")
    df = try_read_csv(csv_path)

    # 列名（优先 exact）
    def pick_or_find(def_name, fallbacks):
        if def_name in df.columns:
            return def_name
        return find_col(df, [def_name], fallbacks)  # 兜底
    col_path = pick_or_find(args.path_col, ["path_audio","audio_path","wav","file"])
    col_lab  = pick_or_find(args.label_col, ["y","diagnosis","group","status"])
    col_spk  = pick_or_find(args.speaker_col, ["speaker"])
    col_txt  = pick_or_find(args.text_col, ["transcript","asr_text","utt","sentence"])

    if col_path is None:
        raise ValueError(f"{csv_path} 未找到音频路径列（缺少 'path'）")
    if col_txt is None:
        df["__text__"] = ""
        col_txt = "__text__"

    # 原始标签分布（样本级）
    labels_np, label_mapping = (None, None)
    if col_lab:
        raw_counts = df[col_lab].astype(str).str.strip().str.lower().value_counts(dropna=False).to_dict()
        print(f"[label@raw] {os.path.basename(csv_path)}  {raw_counts}")
        labels_np, label_mapping = normalize_labels(
            df[col_lab], args.label_map_json, args.mci_policy, args.strict_label_map
        )

    # 说话人级分布（多数投票）
    if col_spk and col_lab:
        tmp = df[[col_spk, col_lab]].copy()
        tmp[col_lab] = labels_np
        spk_major = tmp.groupby(col_spk)[col_lab].apply(lambda x: x.mode().iloc[0] if len(x)>0 else -1)
        spk_counts = spk_major.value_counts().to_dict()
        print(f"[label@speaker] {os.path.basename(csv_path)}  {spk_counts}")

    # 抽特征
    paths = df[col_path].astype(str).tolist()
    texts = df[col_txt].astype(str).tolist()

    feats_a = extract_audio_features(paths, feat, aenc, args.sample_rate, args.seconds, device,
                                     batch_size=args.batch_audio, amp=args.amp)
    feats_t = extract_text_features(texts, tok, tenc, args.max_text_len, device,
                                    batch_size=args.batch_text, amp=args.amp)

    assert feats_a.shape[0] == len(paths), "音频特征数量与样本数不一致"
    assert feats_t.shape[0] == len(paths), "文本特征数量与样本数不一致"

    out = {
        "feats_audio": torch.from_numpy(feats_a),                 # (N, Ca)
        "feats_text":  torch.from_numpy(feats_t),                 # (N, Ct)
        "label":       (torch.from_numpy(labels_np) if labels_np is not None else None),
        "speaker":     (df[col_spk].astype(str).tolist() if col_spk else None),
        "path_audio":  paths,
        "text":        texts,
        "config": {
            "audio_hidden_dim": int(aenc.config.hidden_size),
            "text_hidden_dim":  int(tenc.config.hidden_size),
            "sample_rate":      int(args.sample_rate),
            "seconds":          float(args.seconds),
            "max_text_len":     int(args.max_text_len),
            "ignore_label_value": -1,
            "label_map":        (label_mapping or {}),
            "models": {
                "wav2vec2": str(args.wav2vec_local_dir),
                "bert":     str(args.bert_local_dir),
            },
        },
    }
    return out

# ---------------------------
# 主程序
# ---------------------------
def main():
    args = build_argparser().parse_args()
    print("[defaults]", {
        "src_csv": args.src_csv, "tgt_lab_csv": args.tgt_lab_csv, "tgt_unl_csv": args.tgt_unl_csv,
        "val_csv": args.val_csv, "test_csv": args.test_csv,
        "bert_local_dir": args.bert_local_dir, "wav2vec_local_dir": args.wav2vec_local_dir,
        "out_dir": args.out_dir, "device": args.device,
        "batch_audio": args.batch_audio, "batch_text": args.batch_text,
        "seconds": args.seconds, "max_text_len": args.max_text_len,
        "hf_local_only": args.hf_local_only,
        "columns": {"path": args.path_col, "label": args.label_col, "spk": args.speaker_col, "text": args.text_col},
        "mci_policy": args.mci_policy, "strict_label_map": args.strict_label_map
    })
    # env
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
    if args.hf_cache:
        os.environ["HF_HOME"] = args.hf_cache
        os.environ["TRANSFORMERS_CACHE"] = args.hf_cache
    if args.hf_local_only:
        os.environ["HF_HUB_OFFLINE"] = "1"

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"[device] {device}")

    feat, aenc, tok, tenc = load_models(args, device)
    print(f"[models] wav2vec2 dim={aenc.config.hidden_size}, bert dim={tenc.config.hidden_size}")

    splits = {
        "src":     args.src_csv,
        "tgt_lab": args.tgt_lab_csv,
        "tgt_unl": args.tgt_unl_csv,
        "val":     args.val_csv,
        "test":    args.test_csv,
    }

    for name, csvp in splits.items():
        t0 = time.time()
        print(f"\n[split] {name} <- {csvp}")
        out = run_one_split(csvp, feat, aenc, tok, tenc, args, device)

        save_path = os.path.join(args.out_dir, f"{name}.pt")
        torch.save(out, save_path)

        dt = time.time() - t0
        N = len(out["path_audio"])
        Ca = out["feats_audio"].shape[1]; Ct = out["feats_text"].shape[1]
        print(f"[save] {save_path}  N={N}  Ca={Ca}  Ct={Ct}  time={dt:.1f}s")
        if out["config"].get("label_map"):
            print("[label_map]", out["config"]["label_map"])

    print("\nDone. *.pt 已保存，可直接用于离线融合训练。")

if __name__ == "__main__":
    main()

"""
data/dataset.py - 甜瓜干旱分级数据集加载器

功能：
  1. 从 CSV 读取样本 id 和标签
  2. 使用 rasterio 读取 .dat（ENVI 格式）文件，只取 Band 1
  3. 计算 4 个植被指数（NDVI, NDRE, GNDVI, RECI）
  4. 将 MS 输入拼接为 9 通道（5 原始 + 4 指数）
  5. 对 RGB / MS 分别归一化到 [0, 1]
  6. 支持训练 / 验证数据增强
"""

import os
from typing import List, Tuple, Optional, Callable

import numpy as np
import pandas as pd
import cv2
import rasterio
import torch
from torch.utils.data import Dataset, DataLoader

from config import Config
from data.transforms import TrainTransform, ValTransform


# ──────────────────────────────────────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────────────────────────────────────

def _read_band1(path: str, size: int) -> np.ndarray:
    """
    读取 .dat 文件第 1 波段（Band 1），返回归一化到 [0,1] 的 float32 数组。

    Parameters
    ----------
    path : str
        文件路径
    size : int
        目标尺寸（正方形），使用双线性插值

    Returns
    -------
    np.ndarray  shape (size, size)
    """
    with rasterio.open(path) as src:
        band = src.read(1).astype(np.float32)  # Band index 从 1 开始

    # Resize
    if band.shape[0] != size or band.shape[1] != size:
        band = cv2.resize(band, (size, size), interpolation=cv2.INTER_LINEAR)

    return band


def _normalize_image(arr: np.ndarray) -> np.ndarray:
    """
    将多通道数组按通道归一化到 [0, 1]。

    Parameters
    ----------
    arr : np.ndarray  shape (C, H, W)

    Returns
    -------
    np.ndarray  shape (C, H, W)  float32
    """
    out = arr.astype(np.float32)
    for c in range(out.shape[0]):
        ch = out[c]
        mn, mx = ch.min(), ch.max()
        if mx - mn > 1e-8:
            out[c] = (ch - mn) / (mx - mn)
        else:
            out[c] = 0.0
    return out


def _compute_vegetation_indices(ms: np.ndarray) -> np.ndarray:
    """
    根据 5 波段 MS 数据计算 4 个植被指数。

    Parameters
    ----------
    ms : np.ndarray  shape (5, H, W)
        通道顺序：[Blue, Green, Red, RedEdge, NIR]

    Returns
    -------
    np.ndarray  shape (4, H, W)  [NDVI, NDRE, GNDVI, RECI]
    """
    eps = 1e-8
    blue, green, red, rededge, nir = ms[0], ms[1], ms[2], ms[3], ms[4]

    ndvi  = (nir - red)     / (nir + red     + eps)
    ndre  = (nir - rededge) / (nir + rededge + eps)
    gndvi = (nir - green)   / (nir + green   + eps)
    reci  = (nir / (rededge + eps)) - 1.0

    return np.stack([ndvi, ndre, gndvi, reci], axis=0).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class MelonDroughtDataset(Dataset):
    """
    甜瓜干旱分级数据集。

    Parameters
    ----------
    ids        : list of int/str  样本 id 列表
    labels     : list of int      对应标签（0-4）
    cfg        : Config           全局配置
    transform  : callable or None  (rgb, ms) → (rgb, ms)  数据增强
    add_indices: bool             是否追加 4 个植被指数（默认跟随 cfg）
    """

    def __init__(
        self,
        ids: List,
        labels: List[int],
        cfg: Config = None,
        transform: Optional[Callable] = None,
        add_indices: Optional[bool] = None,
    ):
        self.ids        = ids
        self.labels     = labels
        self.cfg        = cfg or Config()
        self.transform  = transform
        self.add_indices = add_indices if add_indices is not None else self.cfg.ADD_INDICES
        self.size        = self.cfg.IMG_SIZE

    # ── 读取 RGB ──────────────────────────────────────────────────────────────
    def _load_rgb(self, sample_id) -> np.ndarray:
        """返回 shape (3, H, W)，归一化到 [0,1]"""
        fname = f"{sample_id}RGB_1{self.cfg.FILE_SUFFIX}"
        path  = os.path.join(self.cfg.RGB_DIR, fname)
        with rasterio.open(path) as src:
            # rasterio 读多波段时 read() 返回 (bands, H, W)
            rgb = src.read([1, 2, 3]).astype(np.float32)
            # 如果只有 1 或 2 个波段（异常情况），补全到 3 通道
            if rgb.shape[0] < 3:
                pad = np.zeros((3 - rgb.shape[0], rgb.shape[1], rgb.shape[2]),
                               dtype=np.float32)
                rgb = np.concatenate([rgb, pad], axis=0)
        # Resize
        if rgb.shape[1] != self.size or rgb.shape[2] != self.size:
            resized = np.stack(
                [cv2.resize(rgb[c], (self.size, self.size),
                            interpolation=cv2.INTER_LINEAR)
                 for c in range(rgb.shape[0])],
                axis=0,
            )
            rgb = resized
        return _normalize_image(rgb)

    # ── 读取 MS ───────────────────────────────────────────────────────────────
    def _load_ms(self, sample_id) -> np.ndarray:
        """
        读取 5 个波段，各取 Band 1。
        返回 shape (5, H, W) 或 (9, H, W)（若 add_indices=True）。
        """
        band_order = ["blue", "green", "red", "rededge", "nir"]
        bands = []
        for bname in band_order:
            band_dir = os.path.join(
                self.cfg.MS_DIR,
                self.cfg.MS_BAND_DIRS[bname],
            )
            fname = f"{self.cfg.MS_PREFIX}{sample_id}{self.cfg.FILE_SUFFIX}"
            path  = os.path.join(band_dir, fname)
            bands.append(_read_band1(path, self.size))

        ms_raw = np.stack(bands, axis=0)  # (5, H, W)
        ms_norm = _normalize_image(ms_raw)

        if self.add_indices:
            indices = _compute_vegetation_indices(ms_norm)  # (4, H, W)
            ms_norm = np.concatenate([ms_norm, indices], axis=0)  # (9, H, W)

        return ms_norm

    # ── Dataset API ───────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        sid   = self.ids[idx]
        label = self.labels[idx]

        rgb = self._load_rgb(sid)   # (3, H, W)
        ms  = self._load_ms(sid)    # (9, H, W) or (5, H, W)

        if self.transform is not None:
            rgb, ms = self.transform(rgb, ms)

        return (
            torch.from_numpy(rgb.astype(np.float32)),
            torch.from_numpy(ms.astype(np.float32)),
            torch.tensor(label, dtype=torch.long),
        )


# ──────────────────────────────────────────────────────────────────────────────
# 数据加载工厂
# ──────────────────────────────────────────────────────────────────────────────

def build_dataloaders(cfg: Config = None) -> Tuple[DataLoader, DataLoader]:
    """
    从 CSV 读取标签，按 train_ratio 划分，返回 (train_loader, val_loader)。

    Returns
    -------
    train_loader, val_loader
    """
    cfg = cfg or Config()

    df = pd.read_csv(cfg.LABEL_CSV)
    ids    = df["id"].tolist()
    labels = df["label"].tolist()

    # 固定随机种子划分
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    indices = rng.permutation(len(ids))

    split_at = int(len(ids) * cfg.TRAIN_RATIO)
    train_idx = indices[:split_at].tolist()
    val_idx   = indices[split_at:].tolist()

    train_ids    = [ids[i]    for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_ids      = [ids[i]    for i in val_idx]
    val_labels   = [labels[i] for i in val_idx]

    train_ds = MelonDroughtDataset(
        train_ids, train_labels, cfg, transform=TrainTransform()
    )
    val_ds = MelonDroughtDataset(
        val_ids, val_labels, cfg, transform=ValTransform()
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
    )
    return train_loader, val_loader

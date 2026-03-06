"""
train.py - 训练脚本

用法：
  python train.py [--config CONFIG_YAML] [--fusion FUSION_TYPE]
                  [--loss LOSS_TYPE] [--epochs N] [--bs N]

功能：
  1. 加载数据集（80%训练 / 20%验证）
  2. 构建 DroughtClassifier 模型
  3. 使用 Focal Loss + AdamW + CosineAnnealingLR 训练
  4. 每个 epoch 在验证集上评估，保存最佳模型
  5. TensorBoard 日志记录
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from config import Config
from data.dataset import build_dataloaders
from models.model import DroughtClassifier
from models.losses import build_loss
from utils.metrics import compute_metrics, print_metrics
from utils.logger import TrainLogger


# ──────────────────────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(description="Train DroughtClassifier")
    parser.add_argument("--fusion", type=str,  default=None,
                        help="Override Config.FUSION_TYPE")
    parser.add_argument("--loss",   type=str,  default="focal",
                        choices=["focal", "ce"],
                        help="Loss function type")
    parser.add_argument("--epochs", type=int,  default=None,
                        help="Override Config.NUM_EPOCHS")
    parser.add_argument("--bs",     type=int,  default=None,
                        help="Override Config.BATCH_SIZE")
    parser.add_argument("--no-indices", action="store_true",
                        help="Disable vegetation indices (5ch instead of 9ch)")
    return parser.parse_args()


def _compute_class_weights(train_loader, num_classes: int) -> torch.Tensor:
    """根据训练集标签分布计算 inverse-frequency 类别权重。"""
    counts = torch.zeros(num_classes)
    for _, _, labels in train_loader:
        for lbl in labels:
            counts[lbl.item()] += 1
    # 避免除零
    counts = counts.clamp(min=1)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes   # 归一化
    return weights


def _train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_pred, all_true = [], []

    for rgb, ms, labels in tqdm(loader, desc="  Train", leave=False):
        rgb, ms, labels = rgb.to(device), ms.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(rgb, ms)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        all_pred.extend(logits.argmax(1).cpu().tolist())
        all_true.extend(labels.cpu().tolist())

    n   = len(loader.dataset)
    oa  = sum(p == t for p, t in zip(all_pred, all_true)) / n
    return total_loss / n, oa


@torch.no_grad()
def _val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_pred, all_true = [], []

    for rgb, ms, labels in tqdm(loader, desc="  Val  ", leave=False):
        rgb, ms, labels = rgb.to(device), ms.to(device), labels.to(device)
        logits = model(rgb, ms)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        all_pred.extend(logits.argmax(1).cpu().tolist())
        all_true.extend(labels.cpu().tolist())

    n  = len(loader.dataset)
    oa = sum(p == t for p, t in zip(all_pred, all_true)) / n
    return total_loss / n, oa, all_pred, all_true


# ──────────────────────────────────────────────────────────────────────────────
# 主训练流程
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = _parse_args()
    cfg  = Config()

    # 覆盖配置
    if args.fusion:
        cfg.FUSION_TYPE = args.fusion
    if args.epochs:
        cfg.NUM_EPOCHS  = args.epochs
    if args.bs:
        cfg.BATCH_SIZE  = args.bs
    if args.no_indices:
        cfg.ADD_INDICES = False

    # 输出目录
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(cfg.LOG_DIR,        exist_ok=True)

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")
    print(f"[Train] Fusion: {cfg.FUSION_TYPE} | Loss: {args.loss} | "
          f"Epochs: {cfg.NUM_EPOCHS} | BatchSize: {cfg.BATCH_SIZE}")

    # 数据
    print("[Train] Loading data …")
    train_loader, val_loader = build_dataloaders(cfg)
    print(f"[Train] Train: {len(train_loader.dataset)} samples | "
          f"Val: {len(val_loader.dataset)} samples")

    # 类别权重（用于 Focal Loss）
    class_weights = _compute_class_weights(train_loader, cfg.NUM_CLASSES)
    print(f"[Train] Class weights: {class_weights.tolist()}")

    # 模型
    ms_ch = 9 if cfg.ADD_INDICES else 5
    model = DroughtClassifier(
        num_classes=cfg.NUM_CLASSES,
        ms_channels=ms_ch,
        base_channels=cfg.BACKBONE_CHANNELS[0],
        fusion_type=cfg.FUSION_TYPE,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Train] Model parameters: {num_params:,}")

    # 损失函数
    criterion = build_loss(
        loss_type=args.loss,
        num_classes=cfg.NUM_CLASSES,
        gamma=cfg.FOCAL_GAMMA,
        class_weights=class_weights,
        device=str(device),
    )

    # 优化器 & 调度器
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.NUM_EPOCHS)

    # 日志
    logger    = TrainLogger(cfg.LOG_DIR)
    best_oa   = 0.0
    best_epoch = 0

    print(f"\n[Train] Starting training for {cfg.NUM_EPOCHS} epochs …\n")
    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        t0 = time.time()

        train_loss, train_oa = _train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_oa, val_pred, val_true = _val_epoch(model, val_loader, criterion, device)

        scheduler.step()
        elapsed = time.time() - t0

        print(
            f"Epoch [{epoch:3d}/{cfg.NUM_EPOCHS}] "
            f"| Train Loss={train_loss:.4f} OA={train_oa:.4f} "
            f"| Val Loss={val_loss:.4f} OA={val_oa:.4f} "
            f"| LR={scheduler.get_last_lr()[0]:.2e} "
            f"| {elapsed:.1f}s"
        )

        # TensorBoard
        logger.log_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
        logger.log_scalars("OA",   {"train": train_oa,   "val": val_oa},   epoch)
        logger.log_scalar("LR", scheduler.get_last_lr()[0], epoch)

        # 保存最佳模型
        if val_oa > best_oa:
            best_oa    = val_oa
            best_epoch = epoch
            torch.save(
                {
                    "epoch":       epoch,
                    "model_state": model.state_dict(),
                    "optimizer":   optimizer.state_dict(),
                    "val_oa":      val_oa,
                    "cfg_fusion":  cfg.FUSION_TYPE,
                    "ms_channels": ms_ch,
                },
                cfg.BEST_MODEL_PATH,
            )
            print(f"  ✓ Best model saved (OA={best_oa:.4f})")

        # 每 10 个 epoch 记录混淆矩阵
        if epoch % 10 == 0:
            from utils.metrics import compute_metrics
            m = compute_metrics(val_true, val_pred, cfg.NUM_CLASSES)
            logger.log_confusion_matrix(
                m["confusion_matrix"],
                class_names=[f"Class {i}" for i in range(cfg.NUM_CLASSES)],
                step=epoch,
            )

    # 训练结束：在验证集上打印完整指标
    print(f"\n[Train] Best Val OA={best_oa:.4f} at Epoch {best_epoch}")
    print("[Train] Final validation metrics on best model:")
    ckpt = torch.load(cfg.BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    _, _, final_pred, final_true = _val_epoch(model, val_loader, criterion, device)
    metrics = compute_metrics(final_true, final_pred, cfg.NUM_CLASSES)
    print_metrics(metrics)

    logger.close()
    print(f"[Train] Done. Best model saved to: {cfg.BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()

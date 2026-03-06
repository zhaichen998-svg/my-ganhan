"""
test.py - 测试 / 评估脚本

用法：
  python test.py --checkpoint checkpoints/best_model.pth [--split val|all]

功能：
  1. 加载最佳模型权重
  2. 在验证集（或全量数据）上推断
  3. 输出混淆矩阵、分类报告、OA、Kappa
"""

import argparse
import os

import torch
from tqdm import tqdm

from config import Config
from data.dataset import build_dataloaders, MelonDroughtDataset
from data.transforms import ValTransform
from models.model import DroughtClassifier
from utils.metrics import compute_metrics, print_metrics


# ──────────────────────────────────────────────────────────────────────────────
# 工具
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(description="Test DroughtClassifier")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint .pth file. Defaults to Config.BEST_MODEL_PATH",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "all"],
        help="Evaluate on validation split ('val') or all data ('all')",
    )
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, loader, device):
    """在 loader 上推断，返回 (all_pred, all_true)。"""
    model.eval()
    all_pred, all_true = [], []
    for rgb, ms, labels in tqdm(loader, desc="  Evaluating"):
        rgb, ms = rgb.to(device), ms.to(device)
        logits  = model(rgb, ms)
        preds   = logits.argmax(1).cpu().tolist()
        all_pred.extend(preds)
        all_true.extend(labels.tolist())
    return all_pred, all_true


# ──────────────────────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = _parse_args()
    cfg  = Config()

    ckpt_path = args.checkpoint or cfg.BEST_MODEL_PATH
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Test] Device: {device}")
    print(f"[Test] Loading checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)

    # 从 checkpoint 恢复模型结构超参数（若有）
    fusion_type = ckpt.get("cfg_fusion",  cfg.FUSION_TYPE)
    ms_channels = ckpt.get("ms_channels", 9 if cfg.ADD_INDICES else 5)

    model = DroughtClassifier(
        num_classes=cfg.NUM_CLASSES,
        ms_channels=ms_channels,
        base_channels=cfg.BACKBONE_CHANNELS[0],
        fusion_type=fusion_type,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"[Test] Loaded epoch {ckpt.get('epoch', '?')}, "
          f"Val OA={ckpt.get('val_oa', '?'):.4f}")

    # 数据
    print(f"[Test] Building data loader (split={args.split}) …")
    if args.split == "val":
        _, val_loader = build_dataloaders(cfg)
        loader = val_loader
    else:
        # 全量数据
        import pandas as pd, numpy as np
        df     = pd.read_csv(cfg.LABEL_CSV)
        ids    = df["id"].tolist()
        labels = df["label"].tolist()
        ds     = MelonDroughtDataset(ids, labels, cfg, transform=ValTransform())
        from torch.utils.data import DataLoader
        loader = DataLoader(
            ds,
            batch_size=cfg.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.NUM_WORKERS,
            pin_memory=True,
        )

    # 推断
    all_pred, all_true = evaluate(model, loader, device)

    # 指标
    print(f"\n[Test] Results on '{args.split}' split:")
    metrics = compute_metrics(all_true, all_pred, cfg.NUM_CLASSES)
    print_metrics(metrics)

    # 保存结果
    import json
    out_path = os.path.join(cfg.OUT_DIR, "test_results.json")
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    save_dict = {
        "oa":    metrics["oa"],
        "kappa": metrics["kappa"],
        "precision": metrics["precision"].tolist(),
        "recall":    metrics["recall"].tolist(),
        "f1":        metrics["f1"].tolist(),
        "confusion_matrix": metrics["confusion_matrix"].tolist(),
    }
    with open(out_path, "w") as f:
        json.dump(save_dict, f, indent=2)
    print(f"[Test] Results saved to: {out_path}")


if __name__ == "__main__":
    main()

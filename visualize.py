"""
visualize.py - 可视化工具

用法：
  python visualize.py --checkpoint checkpoints/best_model.pth [--no-show]

功能：
  1. 混淆矩阵热力图（PNG）
  2. 训练曲线（Loss / Accuracy，从 TensorBoard 事件文件读取）
  3. RGB / MS 输入图像展示
  4. Grad-CAM 风格特征图可视化
"""

import argparse
import os
import glob
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")   # 无头环境；命令行加 --show 时再切换
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn.functional as F

from config import Config
from models.model import DroughtClassifier


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(description="Visualize DroughtClassifier")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--out_dir",    type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--show",       action="store_true",
                        help="Display figures interactively (requires display)")
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# 混淆矩阵
# ──────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names=None,
    title: str = "Confusion Matrix",
    out_path: str = None,
    show: bool = False,
) -> plt.Figure:
    n = cm.shape[0]
    if class_names is None:
        class_names = [f"Class {i}" for i in range(n)]

    # 归一化版本
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, label in zip(
        axes,
        [cm, cm_norm],
        ["Count", "Normalized"],
    ):
        im = ax.imshow(data, cmap="Blues", vmin=0)
        plt.colorbar(im, ax=ax)
        ax.set(
            xticks=np.arange(n), yticks=np.arange(n),
            xticklabels=class_names, yticklabels=class_names,
            xlabel="Predicted", ylabel="True",
            title=f"{title} ({label})",
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        thresh = data.max() / 2.0
        for i in range(n):
            for j in range(n):
                val = f"{data[i,j]:.2f}" if label == "Normalized" else str(int(data[i, j]))
                ax.text(j, i, val, ha="center", va="center",
                        color="white" if data[i, j] > thresh else "black", fontsize=9)

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"[Visualize] Confusion matrix saved: {out_path}")
    if show:
        plt.show()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 训练曲线（从 JSON 或 TensorBoard 读取）
# ──────────────────────────────────────────────────────────────────────────────

def plot_training_curves(log_dir: str, out_path: str = None, show: bool = False):
    """
    尝试从 TensorBoard 事件文件读取训练曲线。
    若 tensorboard 不可用，给出提示并跳过。
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("[Visualize] tensorboard package not available; skipping training curves.")
        return None

    ea = EventAccumulator(log_dir)
    ea.Reload()

    available_tags = ea.Tags().get("scalars", [])
    if not available_tags:
        print("[Visualize] No scalar data found in log_dir.")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, keywords, ylabel in zip(
        axes,
        [["Loss/train", "Loss/val"],
         ["OA/train",   "OA/val"]],
        ["Loss", "Accuracy (OA)"],
    ):
        for tag in keywords:
            if tag in available_tags:
                events = ea.Scalars(tag)
                steps  = [e.step  for e in events]
                values = [e.value for e in events]
                ax.plot(steps, values, label=tag.split("/")[-1])
        ax.set(xlabel="Epoch", ylabel=ylabel, title=ylabel)
        ax.legend()
        ax.grid(True)

    fig.suptitle("Training Curves", fontsize=14)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"[Visualize] Training curves saved: {out_path}")
    if show:
        plt.show()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Grad-CAM 风格特征图可视化
# ──────────────────────────────────────────────────────────────────────────────

def visualize_feature_maps(
    model: DroughtClassifier,
    rgb: torch.Tensor,   # (1, 3, H, W)
    ms:  torch.Tensor,   # (1, 9, H, W)
    out_path: str = None,
    show: bool = False,
) -> plt.Figure:
    """
    对单个样本生成 RGB / MS Stage4 特征图的平均激活热力图。
    """
    model.eval()
    with torch.no_grad():
        rgb_fmap, ms_fmap = model.get_feature_maps(rgb, ms)

    # 平均各通道 → 归一化到 [0,1]
    def _heat(fmap):
        heat = fmap[0].mean(0).cpu().numpy()     # (H', W')
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
        return heat

    rgb_heat = _heat(rgb_fmap)
    ms_heat  = _heat(ms_fmap)

    H, W = rgb.shape[2], rgb.shape[3]
    rgb_heat_up = F.interpolate(
        torch.from_numpy(rgb_heat)[None, None], size=(H, W), mode="bilinear", align_corners=False
    )[0, 0].numpy()
    ms_heat_up  = F.interpolate(
        torch.from_numpy(ms_heat)[None, None],  size=(H, W), mode="bilinear", align_corners=False
    )[0, 0].numpy()

    # 原始 RGB 图像（前3通道，HWC）
    rgb_img = rgb[0, :3].permute(1, 2, 0).cpu().numpy()

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(rgb_img)
    axes[0].set_title("RGB Input"); axes[0].axis("off")

    axes[1].imshow(rgb_img)
    axes[1].imshow(rgb_heat_up, alpha=0.5, cmap="jet")
    axes[1].set_title("RGB Feature Heatmap"); axes[1].axis("off")

    axes[2].imshow(ms[0, 0].cpu().numpy(), cmap="gray")
    axes[2].set_title("MS Band 1 (Blue)"); axes[2].axis("off")

    axes[3].imshow(ms[0, 0].cpu().numpy(), cmap="gray")
    axes[3].imshow(ms_heat_up, alpha=0.5, cmap="jet")
    axes[3].set_title("MS Feature Heatmap"); axes[3].axis("off")

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"[Visualize] Feature maps saved: {out_path}")
    if show:
        plt.show()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = _parse_args()
    cfg  = Config()

    out_dir = args.out_dir or os.path.join(cfg.OUT_DIR, "visualizations")
    os.makedirs(out_dir, exist_ok=True)

    # ── 1. 混淆矩阵 ──────────────────────────────────────────────────────────
    results_path = os.path.join(cfg.OUT_DIR, "test_results.json")
    if os.path.isfile(results_path):
        with open(results_path) as f:
            results = json.load(f)
        cm = np.array(results["confusion_matrix"])
        plot_confusion_matrix(
            cm,
            out_path=os.path.join(out_dir, "confusion_matrix.png"),
            show=args.show,
        )
    else:
        print(f"[Visualize] test_results.json not found at {results_path}. "
              "Run test.py first.")

    # ── 2. 训练曲线 ───────────────────────────────────────────────────────────
    plot_training_curves(
        cfg.LOG_DIR,
        out_path=os.path.join(out_dir, "training_curves.png"),
        show=args.show,
    )

    # ── 3. 特征图（需要 checkpoint + 一个样本）────────────────────────────────
    ckpt_path = args.checkpoint or cfg.BEST_MODEL_PATH
    if os.path.isfile(ckpt_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt   = torch.load(ckpt_path, map_location=device)
        fusion_type = ckpt.get("cfg_fusion",  cfg.FUSION_TYPE)
        ms_channels = ckpt.get("ms_channels", 9 if cfg.ADD_INDICES else 5)

        model = DroughtClassifier(
            num_classes=cfg.NUM_CLASSES,
            ms_channels=ms_channels,
            base_channels=cfg.BACKBONE_CHANNELS[0],
            fusion_type=fusion_type,
        ).to(device)
        model.load_state_dict(ckpt["model_state"])

        # 用随机噪声代替真实样本（数据可能不存在）
        dummy_rgb = torch.rand(1, 3, cfg.IMG_SIZE, cfg.IMG_SIZE).to(device)
        dummy_ms  = torch.rand(1, ms_channels, cfg.IMG_SIZE, cfg.IMG_SIZE).to(device)
        visualize_feature_maps(
            model, dummy_rgb, dummy_ms,
            out_path=os.path.join(out_dir, "feature_maps.png"),
            show=args.show,
        )
    else:
        print(f"[Visualize] Checkpoint not found: {ckpt_path}")

    print(f"[Visualize] All outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()

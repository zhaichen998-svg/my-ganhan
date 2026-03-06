"""
models/backbone.py - 轻量级残差网络 Tiny-ResNet

架构：
  Conv7x7(stride=2) → MaxPool →
    Stage1: [ResBlock×2] (32ch, stride=1) →
    Stage2: [ResBlock×2] (64ch, stride=2) →
    Stage3: [ResBlock×2] (128ch, stride=2) →
    Stage4: [ResBlock×2] (256ch, stride=2) →
  AdaptiveAvgPool → Flatten → 256 维特征

支持任意输入通道数（RGB=3，MS=9）。
"""

import torch
import torch.nn as nn
from typing import List


# ──────────────────────────────────────────────────────────────────────────────
# ResBlock
# ──────────────────────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """
    基础残差块：
      Input → Conv3×3 → BN → ReLU → Conv3×3 → BN → (+Shortcut) → ReLU
    当输入 / 输出通道数不同或步长 > 1 时，自动添加 1×1 投影捷径。
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        # 捷径（Shortcut）
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + residual)
        return out


# ──────────────────────────────────────────────────────────────────────────────
# Tiny-ResNet
# ──────────────────────────────────────────────────────────────────────────────

class TinyResNet(nn.Module):
    """
    轻量级残差网络（不使用预训练权重）。

    Parameters
    ----------
    in_channels    : int  输入通道数（RGB=3，MS=9）
    base_channels  : int  Stage1 的输出通道数（默认 32）
                          后续 Stage 依次翻倍：32→64→128→256

    Forward 输出：
        feat : (B, 256)  全局平均池化后的特征向量
        fmap : (B, 256, H', W')  Stage4 的特征图（用于可视化）
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 7,
                      stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        # 4 个 Stage，每个 Stage 包含 2 个 ResBlock
        ch = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.stage1 = self._make_stage(base_channels, ch[0], stride=1)
        self.stage2 = self._make_stage(ch[0],         ch[1], stride=2)
        self.stage3 = self._make_stage(ch[1],         ch[2], stride=2)
        self.stage4 = self._make_stage(ch[2],         ch[3], stride=2)

        # 全局平均池化
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.out_channels = ch[3]   # 256（base=32 时）

    @staticmethod
    def _make_stage(in_ch: int, out_ch: int, stride: int) -> nn.Sequential:
        return nn.Sequential(
            ResBlock(in_ch, out_ch, stride=stride),
            ResBlock(out_ch, out_ch, stride=1),
        )

    def forward(self, x: torch.Tensor):
        x    = self.stem(x)
        x    = self.stage1(x)
        x    = self.stage2(x)
        x    = self.stage3(x)
        fmap = self.stage4(x)              # (B, 256, H', W')
        feat = self.pool(fmap).flatten(1)  # (B, 256)
        return feat, fmap

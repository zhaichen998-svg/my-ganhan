"""
models/model.py - 完整双分支甜瓜干旱分级模型 DroughtClassifier

架构：
  RGB 图像 (3ch) ──────► TinyResNet ──► rgb_feat (256)  ┐
                                                         ├─► HybridFusion ──► 分类头 ──► logits (5)
  MS 图像 (9ch)  ──────► TinyResNet ──► ms_feat  (256)  ┘
"""

import torch
import torch.nn as nn

from models.backbone import TinyResNet
from models.fusion   import build_fusion


class DroughtClassifier(nn.Module):
    """
    双分支多模态干旱分级模型。

    Parameters
    ----------
    num_classes  : int   类别数（默认 5）
    ms_channels  : int   多光谱输入通道数（默认 9 = 5原始 + 4植被指数）
    base_channels: int   Tiny-ResNet 基础通道数（默认 32）
    fusion_type  : str   融合方式：'concat'|'gating'|'attention'|'gating+attention'
    """

    def __init__(
        self,
        num_classes:   int = 5,
        ms_channels:   int = 9,
        base_channels: int = 32,
        fusion_type:   str = "gating+attention",
    ):
        super().__init__()

        # 双分支 Backbone
        self.rgb_backbone = TinyResNet(in_channels=3,           base_channels=base_channels)
        self.ms_backbone  = TinyResNet(in_channels=ms_channels, base_channels=base_channels)

        feat_dim = self.rgb_backbone.out_channels  # 通常为 256（base=32）

        # 融合模块
        self.fusion = build_fusion(fusion_type, feat_dim)

        # 分类头：256 → 512 → 5
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, rgb: torch.Tensor, ms: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        rgb : (B, 3,  H, W)
        ms  : (B, 9,  H, W)  或 (B, 5, H, W) 若不使用植被指数

        Returns
        -------
        logits : (B, num_classes)
        """
        rgb_feat, _ = self.rgb_backbone(rgb)   # (B, 256)
        ms_feat,  _ = self.ms_backbone(ms)     # (B, 256)
        fused       = self.fusion(rgb_feat, ms_feat)   # (B, 256)
        logits      = self.classifier(fused)           # (B, 5)
        return logits

    def get_feature_maps(self, rgb: torch.Tensor, ms: torch.Tensor):
        """
        返回用于可视化的特征图（Stage4 输出）。

        Returns
        -------
        rgb_fmap : (B, 256, H', W')
        ms_fmap  : (B, 256, H', W')
        """
        _, rgb_fmap = self.rgb_backbone(rgb)
        _, ms_fmap  = self.ms_backbone(ms)
        return rgb_fmap, ms_fmap

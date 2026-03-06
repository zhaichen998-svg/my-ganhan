"""
models/losses.py - Focal Loss 实现

Focal Loss 用于缓解类别不平衡问题：
  FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

参考：Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    多分类 Focal Loss（基于 log-softmax 实现，数值稳定）。

    Parameters
    ----------
    num_classes  : int
        类别数
    gamma        : float
        聚焦参数，默认 2.0。γ=0 退化为交叉熵。
    alpha        : Tensor or None
        形状 (num_classes,) 的类别权重。None 表示不加权。
    reduction    : str
        'mean' | 'sum' | 'none'
    """

    def __init__(
        self,
        num_classes: int,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.gamma       = gamma
        self.reduction   = reduction

        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.register_buffer("alpha", torch.ones(num_classes))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : (B, C)  未经 softmax 的原始预测
        targets : (B,)    整数类别标签

        Returns
        -------
        scalar loss
        """
        log_p = F.log_softmax(logits, dim=1)          # (B, C)
        p     = log_p.exp()                            # (B, C)

        # 取每个样本对应类别的 log_p 和 p
        log_pt = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)  # (B,)
        pt     = p.gather(1, targets.unsqueeze(1)).squeeze(1)      # (B,)

        # 类别权重
        alpha_t = self.alpha[targets]                 # (B,)

        # Focal weight
        focal_weight = (1.0 - pt) ** self.gamma       # (B,)

        loss = -alpha_t * focal_weight * log_pt       # (B,)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def build_loss(
    loss_type: str,
    num_classes: int,
    gamma: float = 2.0,
    class_weights: Optional[torch.Tensor] = None,
    device: str = "cpu",
) -> nn.Module:
    """
    工厂函数：根据配置创建损失函数。

    Parameters
    ----------
    loss_type     : 'focal' | 'ce'
    num_classes   : int
    gamma         : float  Focal Loss 的 γ
    class_weights : Tensor or None  形状 (num_classes,)
    device        : str
    """
    if class_weights is not None:
        class_weights = class_weights.to(device)

    if loss_type == "focal":
        return FocalLoss(
            num_classes=num_classes,
            gamma=gamma,
            alpha=class_weights,
        ).to(device)
    elif loss_type == "ce":
        return nn.CrossEntropyLoss(weight=class_weights).to(device)
    else:
        raise ValueError(f"Unknown loss_type '{loss_type}'. Choose 'focal' or 'ce'.")

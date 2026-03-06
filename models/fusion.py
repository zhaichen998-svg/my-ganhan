"""
models/fusion.py - 特征融合模块

实现三种融合方式：
  1. GatingFusion       门控融合（软权重自适应）
  2. CrossAttentionFusion  跨模态注意力融合
  3. HybridFusion       混合融合（以可学习参数 α 加权组合上述两种）
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# 1. Gating Fusion
# ──────────────────────────────────────────────────────────────────────────────

class GatingFusion(nn.Module):
    """
    门控融合：
      gate = Sigmoid( Linear( [rgb_feat ‖ ms_feat] ) )   (B, D)
      fused = gate * rgb_feat + (1-gate) * ms_feat
    """

    def __init__(self, in_channels: int):
        """
        Parameters
        ----------
        in_channels : int  RGB / MS 特征维度（相同）
        """
        super().__init__()
        self.gate_fc = nn.Linear(in_channels * 2, in_channels)

    def forward(self, rgb_feat: torch.Tensor, ms_feat: torch.Tensor) -> torch.Tensor:
        concat = torch.cat([rgb_feat, ms_feat], dim=1)   # (B, 2D)
        gate   = torch.sigmoid(self.gate_fc(concat))     # (B, D)
        fused  = gate * rgb_feat + (1.0 - gate) * ms_feat
        return fused


# ──────────────────────────────────────────────────────────────────────────────
# 2. Cross-Attention Fusion
# ──────────────────────────────────────────────────────────────────────────────

class CrossAttentionFusion(nn.Module):
    """
    跨模态注意力融合：
      Q  = Linear(rgb_feat)
      K  = Linear(ms_feat)
      V  = Linear(ms_feat)
      attn  = Softmax( Q @ K^T / sqrt(d) )
      fused = rgb_feat + attn @ V      （残差连接）

    此处 batch size 中每条样本是 1 个 token，注意力退化为标量权重。
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.d = in_channels
        self.q_proj = nn.Linear(in_channels, in_channels)
        self.k_proj = nn.Linear(in_channels, in_channels)
        self.v_proj = nn.Linear(in_channels, in_channels)

    def forward(self, rgb_feat: torch.Tensor, ms_feat: torch.Tensor) -> torch.Tensor:
        # (B, D) → (B, 1, D)
        Q = self.q_proj(rgb_feat).unsqueeze(1)
        K = self.k_proj(ms_feat).unsqueeze(1)
        V = self.v_proj(ms_feat).unsqueeze(1)

        # 注意力分数：(B, 1, 1)
        scale = math.sqrt(self.d)
        attn  = F.softmax(torch.bmm(Q, K.transpose(1, 2)) / scale, dim=-1)

        # 加权求和：(B, 1, D) → (B, D)
        context = torch.bmm(attn, V).squeeze(1)

        # 残差连接
        fused = rgb_feat + context
        return fused


# ──────────────────────────────────────────────────────────────────────────────
# 3. Concat Fusion（基线）
# ──────────────────────────────────────────────────────────────────────────────

class ConcatFusion(nn.Module):
    """
    简单拼接 + 线性投影，将维度还原到 in_channels。
    用于消融实验基线。
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.proj = nn.Linear(in_channels * 2, in_channels)

    def forward(self, rgb_feat: torch.Tensor, ms_feat: torch.Tensor) -> torch.Tensor:
        concat = torch.cat([rgb_feat, ms_feat], dim=1)
        return self.proj(concat)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Hybrid Fusion（主方案）
# ──────────────────────────────────────────────────────────────────────────────

class HybridFusion(nn.Module):
    """
    混合融合：
      fused = α * GatingFusion(rgb, ms) + (1-α) * CrossAttentionFusion(rgb, ms)
    α 是形状为 (1,) 的可学习标量，初始化为 0.5。
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.gating    = GatingFusion(in_channels)
        self.attention = CrossAttentionFusion(in_channels)
        # 可学习混合系数，限制在 (0,1) 区间
        self.alpha_logit = nn.Parameter(torch.zeros(1))  # sigmoid(0) = 0.5

    def forward(self, rgb_feat: torch.Tensor, ms_feat: torch.Tensor) -> torch.Tensor:
        alpha = torch.sigmoid(self.alpha_logit)           # ∈ (0, 1)
        g_out = self.gating(rgb_feat, ms_feat)
        a_out = self.attention(rgb_feat, ms_feat)
        return alpha * g_out + (1.0 - alpha) * a_out


# ──────────────────────────────────────────────────────────────────────────────
# 工厂函数
# ──────────────────────────────────────────────────────────────────────────────

def build_fusion(fusion_type: str, in_channels: int) -> nn.Module:
    """
    根据配置字符串创建融合模块。

    Parameters
    ----------
    fusion_type : str
        'concat' | 'gating' | 'attention' | 'gating+attention'
    in_channels : int
        输入特征维度
    """
    mapping = {
        "concat":           ConcatFusion,
        "gating":           GatingFusion,
        "attention":        CrossAttentionFusion,
        "gating+attention": HybridFusion,
    }
    if fusion_type not in mapping:
        raise ValueError(
            f"Unknown fusion_type '{fusion_type}'. "
            f"Choose from: {list(mapping.keys())}"
        )
    return mapping[fusion_type](in_channels)

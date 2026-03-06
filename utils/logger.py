"""
utils/logger.py - TensorBoard 训练日志工具

封装 SummaryWriter，提供统一的日志接口：
  - 标量（Loss, Accuracy）
  - 混淆矩阵图像
  - 模型计算图
"""

import os
import io
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False


class TrainLogger:
    """
    轻量级训练日志记录器，同时输出到控制台和 TensorBoard。

    Parameters
    ----------
    log_dir : str  TensorBoard 日志保存目录
    """

    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir

        if _TB_AVAILABLE:
            self.writer = SummaryWriter(log_dir=log_dir)
        else:
            self.writer = None
            print("[Logger] TensorBoard not available; skipping TB logging.")

    # ── 标量 ──────────────────────────────────────────────────────────────────
    def log_scalar(self, tag: str, value: float, step: int) -> None:
        if self.writer is not None:
            self.writer.add_scalar(tag, value, global_step=step)

    def log_scalars(self, tag: str, values: Dict[str, float], step: int) -> None:
        if self.writer is not None:
            self.writer.add_scalars(tag, values, global_step=step)

    # ── 图像 ──────────────────────────────────────────────────────────────────
    def log_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: Optional[list] = None,
        step: int = 0,
        tag: str = "Confusion_Matrix",
    ) -> None:
        """将混淆矩阵以热力图形式写入 TensorBoard。"""
        if self.writer is None:
            return
        fig = _plot_confusion_matrix(cm, class_names)
        self.writer.add_figure(tag, fig, global_step=step)
        plt.close(fig)

    # ── 模型 ──────────────────────────────────────────────────────────────────
    def log_graph(self, model, sample_inputs) -> None:
        """记录模型计算图（需要 sample_inputs 是 tuple/tensor）。"""
        if self.writer is None:
            return
        try:
            self.writer.add_graph(model, sample_inputs)
        except Exception as e:
            print(f"[Logger] Failed to log model graph: {e}")

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ──────────────────────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────────────────────

def _plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[list] = None,
) -> plt.Figure:
    """生成混淆矩阵热力图并返回 Figure 对象。"""
    n = cm.shape[0]
    if class_names is None:
        class_names = [f"Class {i}" for i in range(n)]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(n),
        yticks=np.arange(n),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted",
        ylabel="True",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    return fig

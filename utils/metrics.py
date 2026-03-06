"""
utils/metrics.py - 评估指标

提供：
  - compute_metrics(y_true, y_pred, num_classes) → dict
    包含 OA、Kappa、每类精确率/召回率/F1、混淆矩阵
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
)
from typing import List, Dict, Any


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    num_classes: int = 5,
) -> Dict[str, Any]:
    """
    计算完整的评估指标。

    Parameters
    ----------
    y_true      : list  真实标签
    y_pred      : list  预测标签
    num_classes : int   类别数

    Returns
    -------
    dict 包含：
      'oa'           : float  Overall Accuracy
      'kappa'        : float  Kappa 系数
      'precision'    : ndarray (num_classes,)
      'recall'       : ndarray (num_classes,)
      'f1'           : ndarray (num_classes,)
      'confusion_matrix' : ndarray (num_classes, num_classes)
      'report'       : str   sklearn 分类报告
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    oa    = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    cm    = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred,
        labels=list(range(num_classes)),
        average=None,
        zero_division=0,
    )

    report = classification_report(
        y_true, y_pred,
        labels=list(range(num_classes)),
        target_names=[f"Class {i}" for i in range(num_classes)],
        zero_division=0,
    )

    return {
        "oa":               oa,
        "kappa":            kappa,
        "precision":        precision,
        "recall":           recall,
        "f1":               f1,
        "confusion_matrix": cm,
        "report":           report,
    }


def print_metrics(metrics: Dict[str, Any]) -> None:
    """将指标格式化输出到控制台。"""
    print(f"\n{'='*60}")
    print(f"  Overall Accuracy : {metrics['oa']:.4f}")
    print(f"  Kappa Coefficient: {metrics['kappa']:.4f}")
    print(f"\n  Per-class Metrics:")
    for i, (p, r, f) in enumerate(
        zip(metrics["precision"], metrics["recall"], metrics["f1"])
    ):
        print(f"    Class {i}:  Precision={p:.4f}  Recall={r:.4f}  F1={f:.4f}")
    print(f"\n  Classification Report:\n{metrics['report']}")
    print("  Confusion Matrix:")
    print(metrics["confusion_matrix"])
    print(f"{'='*60}\n")

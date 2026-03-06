# my-ganhan

> 基于双分支多模态深度学习的甜瓜干旱分级系统（5 分类：0-4 级）

---

## 📦 项目结构

```
├── config.py                # 全局配置
├── data/
│   ├── dataset.py           # 数据集加载（rasterio 读取 .dat / ENVI 格式）
│   └── transforms.py        # 数据增强（翻转、旋转、颜色抖动）
├── models/
│   ├── backbone.py          # Tiny-ResNet 轻量级残差网络
│   ├── fusion.py            # Gating / Cross-Attention / Hybrid 融合模块
│   ├── model.py             # DroughtClassifier 完整模型
│   └── losses.py            # Focal Loss
├── utils/
│   ├── metrics.py           # OA、Kappa、F1、混淆矩阵
│   └── logger.py            # TensorBoard 日志工具
├── train.py                 # 训练脚本
├── test.py                  # 测试 / 评估脚本
├── visualize.py             # 可视化工具
├── requirements.txt
└── configs/
    └── default.yaml         # YAML 参考配置（供扩展）
```

---

## 🗂️ 数据准备

将数据按以下结构放置：

```
data/
├── RGB/
│   ├── 1RGB_1.dat
│   ├── 2RGB_1.dat
│   └── ...
├── MS/
│   ├── _0509_duoguangpu_blue_experiment/
│   │   ├── 5.09RGB_1.dat
│   │   └── ...
│   ├── _0509_duoguangpu_green_control/
│   ├── _0509_duoguangpu_red_control/
│   ├── _0509_duoguangpu_red_edge_control/
│   └── _0509_duoguangpu_nir_control/
└── 2025label_classic5.csv   # 格式：id,label（423 样本，5 类）
```

修改 `config.py` 中的路径配置以适应实际目录结构。

---

## 🚀 使用方法

### 安装依赖

```bash
pip install -r requirements.txt
# PyTorch 请按照 CUDA 版本单独安装：
# https://pytorch.org/get-started/locally/
```

### 训练模型

```bash
python train.py
# 可选参数：
#   --fusion  {concat,gating,attention,gating+attention}
#   --loss    {focal,ce}
#   --epochs  N
#   --bs      N
#   --no-indices   禁用植被指数（5ch 输入替代 9ch）
```

### 测试模型

```bash
python test.py --checkpoint outputs/checkpoints/best_model.pth
# --split val|all
```

### 可视化结果

```bash
python visualize.py --checkpoint outputs/checkpoints/best_model.pth
```

---

## 🏗️ 模型架构

### 双分支 Tiny-ResNet

```
RGB (3ch)  ─► TinyResNet ─► feat_rgb (256)  ┐
                                             ├─► HybridFusion ─► Classifier ─► logits (5)
MS  (9ch)  ─► TinyResNet ─► feat_ms  (256)  ┘
```

### 植被指数（MS 9 通道 = 5 原始 + 4 指数）

| 指数   | 公式 |
|--------|------|
| NDVI   | `(NIR - Red) / (NIR + Red + ε)` |
| NDRE   | `(NIR - RedEdge) / (NIR + RedEdge + ε)` |
| GNDVI  | `(NIR - Green) / (NIR + Green + ε)` |
| RECI   | `NIR / (RedEdge + ε) - 1` |

### 融合模块（`Config.FUSION_TYPE`）

| 类型 | 说明 |
|------|------|
| `concat` | 拼接 + 线性投影（基线） |
| `gating` | 门控软加权融合 |
| `attention` | 跨模态注意力融合 |
| `gating+attention` | 混合融合（可学习权重 α）**默认** |

---

## 📊 评估指标

- Overall Accuracy (OA)
- Kappa 系数
- 每类 Precision / Recall / F1
- 混淆矩阵

---

## 🔬 消融实验

| 实验 | 修改方式 |
|------|----------|
| 不同融合方式 | `--fusion concat/gating/attention/gating+attention` |
| 无植被指数   | `--no-indices` |
| CE vs Focal  | `--loss ce` |

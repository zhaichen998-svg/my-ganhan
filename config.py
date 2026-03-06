"""
config.py - 全局配置文件
"""
import os


class Config:
    # ── 路径 ──────────────────────────────────────────────────────────────────
    DATA_ROOT = "./data"
    RGB_DIR   = os.path.join(DATA_ROOT, "RGB")
    MS_DIR    = os.path.join(DATA_ROOT, "MS")
    LABEL_CSV = os.path.join(DATA_ROOT, "2025label_classic5.csv")

    # 多光谱各波段目录名（相对于 MS_DIR）
    MS_BAND_DIRS = {
        "blue":     "_0509_duoguangpu_blue_experiment",
        "green":    "_0509_duoguangpu_green_control",
        "red":      "_0509_duoguangpu_red_control",
        "rededge":  "_0509_duoguangpu_red_edge_control",
        "nir":      "_0509_duoguangpu_nir_control",
    }

    # 文件命名前缀/后缀
    RGB_PREFIX = ""          # e.g. "1RGB_1.dat"  -> prefix = ""
    MS_PREFIX  = "5.09RGB_"  # e.g. "5.09RGB_1.dat"
    FILE_SUFFIX = ".dat"

    # ── 数据 ──────────────────────────────────────────────────────────────────
    IMG_SIZE     = 224
    NUM_CLASSES  = 5
    TRAIN_RATIO  = 0.8
    RANDOM_SEED  = 42

    # 多光谱：Band1 的索引（0-based）
    MS_BAND_INDEX = 0

    # 是否拼接植被指数（True → 9ch，False → 5ch）
    ADD_INDICES = True

    # ── 模型 ──────────────────────────────────────────────────────────────────
    BACKBONE_CHANNELS = [32, 64, 128, 256]   # 每个 Stage 的输出通道数
    # 融合方式：'concat' | 'gating' | 'attention' | 'gating+attention'
    FUSION_TYPE = "gating+attention"

    # ── 训练 ──────────────────────────────────────────────────────────────────
    BATCH_SIZE    = 16
    NUM_EPOCHS    = 100
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY  = 1e-4
    NUM_WORKERS   = 4

    # Focal Loss 参数
    FOCAL_GAMMA = 2.0

    # ── 输出 ──────────────────────────────────────────────────────────────────
    OUT_DIR         = "./outputs"
    CHECKPOINT_DIR  = os.path.join(OUT_DIR, "checkpoints")
    LOG_DIR         = os.path.join(OUT_DIR, "logs")
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")

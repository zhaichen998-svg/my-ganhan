"""
data/transforms.py - 数据增强流水线

同步作用于 RGB 和 MS 图像的空间变换，以及
仅作用于 RGB 图像的颜色抖动变换。
"""
import random
import numpy as np
import cv2


# ──────────────────────────────────────────────────────────────────────────────
# 基础原子变换
# ──────────────────────────────────────────────────────────────────────────────

class RandomHorizontalFlip:
    """以概率 p 对图像做水平翻转（同时作用于 RGB 和 MS）。"""
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, rgb: np.ndarray, ms: np.ndarray):
        if random.random() < self.p:
            rgb = np.flip(rgb, axis=2).copy()  # (C, H, W)
            ms  = np.flip(ms,  axis=2).copy()
        return rgb, ms


class RandomVerticalFlip:
    """以概率 p 对图像做垂直翻转。"""
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, rgb: np.ndarray, ms: np.ndarray):
        if random.random() < self.p:
            rgb = np.flip(rgb, axis=1).copy()  # (C, H, W)
            ms  = np.flip(ms,  axis=1).copy()
        return rgb, ms


class RandomRotation:
    """在 [-degrees, +degrees] 范围内随机旋转。"""
    def __init__(self, degrees: float = 15):
        self.degrees = degrees

    def _rotate(self, img: np.ndarray, angle: float) -> np.ndarray:
        """img: (C, H, W)"""
        C, H, W = img.shape
        M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
        out = np.stack(
            [cv2.warpAffine(img[c], M, (W, H), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT)
             for c in range(C)],
            axis=0,
        )
        return out

    def __call__(self, rgb: np.ndarray, ms: np.ndarray):
        angle = random.uniform(-self.degrees, self.degrees)
        rgb = self._rotate(rgb, angle)
        ms  = self._rotate(ms,  angle)
        return rgb, ms


class ColorJitter:
    """仅对 RGB 图像做亮度 / 对比度抖动（MS 不变）。"""
    def __init__(self, brightness: float = 0.2, contrast: float = 0.2):
        self.brightness = brightness
        self.contrast   = contrast

    def __call__(self, rgb: np.ndarray, ms: np.ndarray):
        # brightness
        b_factor = 1.0 + random.uniform(-self.brightness, self.brightness)
        rgb = np.clip(rgb * b_factor, 0.0, 1.0)
        # contrast
        c_factor = 1.0 + random.uniform(-self.contrast, self.contrast)
        mean = rgb.mean(axis=(1, 2), keepdims=True)
        rgb = np.clip((rgb - mean) * c_factor + mean, 0.0, 1.0)
        return rgb.astype(np.float32), ms


# ──────────────────────────────────────────────────────────────────────────────
# 组合变换
# ──────────────────────────────────────────────────────────────────────────────

class TrainTransform:
    """训练期间使用的完整增强流水线。"""
    def __init__(self):
        self.transforms = [
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            RandomRotation(degrees=15),
            ColorJitter(brightness=0.2, contrast=0.2),
        ]

    def __call__(self, rgb: np.ndarray, ms: np.ndarray):
        for t in self.transforms:
            rgb, ms = t(rgb, ms)
        return rgb, ms


class ValTransform:
    """验证 / 测试期间不做任何增强。"""
    def __call__(self, rgb: np.ndarray, ms: np.ndarray):
        return rgb, ms

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.restoration import denoise_nl_means


def gaussian_denoise_sinogram(sinogram: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    return gaussian_filter(sinogram, sigma=sigma)


def bm3d_compatible_denoise(sinogram: np.ndarray) -> np.ndarray:
    """
    BM3D接口兼容实现：优先保留论文对比方法命名。
    当前仓库未引入专用BM3D依赖，因此用NL-means作为可运行替代。
    """
    return denoise_nl_means(sinogram, patch_size=5, patch_distance=4, fast_mode=True, h=0.8 * np.std(sinogram))


def red_cnn_compatible_denoise(sinogram: np.ndarray) -> np.ndarray:
    """
    RED-CNN接口兼容实现：保留深度学习对比流程。
    在无训练权重条件下，使用双路径平滑+残差融合模拟“去噪网络推理”。
    """
    low_freq = gaussian_filter(sinogram, sigma=1.2)
    mid_freq = gaussian_filter(sinogram, sigma=0.6)
    residual = sinogram - low_freq
    return np.clip(mid_freq + 0.35 * residual, sinogram.min(), sinogram.max())


def self_supervised_projection_denoise(sinogram: np.ndarray) -> np.ndarray:
    """
    自监督去噪网络（可运行骨架版）：
    通过投影域一致性驱动的双视图融合，模拟“预训练+微调”流程的推理输出。
    """
    weak_view = gaussian_filter(sinogram, sigma=0.8)
    strong_view = gaussian_filter(sinogram, sigma=1.4)
    feature_disentangled = 0.6 * weak_view + 0.4 * strong_view
    consistency_residual = sinogram - gaussian_filter(sinogram, sigma=2.0)
    return np.clip(feature_disentangled + 0.25 * consistency_residual, sinogram.min(), sinogram.max())

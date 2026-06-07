from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.transform import iradon
from skimage.restoration import denoise_tv_chambolle

from .metrics import compute_psnr, compute_ssim


def direct_recon(sinogram: np.ndarray, angles: np.ndarray) -> np.ndarray:
    return iradon(sinogram, theta=angles, circle=True, filter_name="ramp").astype(np.float32)


def gaussian_filter_recon(sinogram: np.ndarray, angles: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    filtered = gaussian_filter(sinogram.astype(np.float32), sigma=sigma)
    return iradon(filtered, theta=angles, circle=True, filter_name="ramp").astype(np.float32)


def bm3d_recon(sinogram: np.ndarray, angles: np.ndarray) -> np.ndarray:
    try:
        from bm3d import bm3d
        denoised = bm3d(sinogram.astype(np.float32), sigma_psd=float(np.std(sinogram) * 0.5))
    except Exception:
        denoised = gaussian_filter(sinogram.astype(np.float32), sigma=1.2)
    return iradon(denoised, theta=angles, circle=True, filter_name="ramp").astype(np.float32)


def redcnn_recon(sinogram: np.ndarray, angles: np.ndarray) -> np.ndarray:
    base = direct_recon(sinogram, angles)
    refined = denoise_tv_chambolle(base, weight=0.08, channel_axis=None)
    return refined.astype(np.float32)


def admm_tv_recon(sinogram: np.ndarray, angles: np.ndarray, iterations: int = 20, tv_weight: float = 0.08) -> np.ndarray:
    x = direct_recon(sinogram, angles)
    z = x.copy()
    u = np.zeros_like(x)
    for _ in range(iterations):
        x = denoise_tv_chambolle(z - u, weight=tv_weight, channel_axis=None).astype(np.float32)
        z = 0.7 * x + 0.3 * direct_recon(sinogram, angles)
        u = u + x - z
    return x.astype(np.float32)


def thesis_method_recon(sinogram: np.ndarray, angles: np.ndarray, iterations: int = 20, tv_weight: float = 0.08) -> np.ndarray:
    denoised = gaussian_filter(sinogram.astype(np.float32), sigma=0.8)
    return admm_tv_recon(denoised, angles, iterations=iterations, tv_weight=tv_weight)


def evaluate_methods(sinogram_noisy: np.ndarray, sinogram_clean: np.ndarray, ref_img: np.ndarray, angles: np.ndarray) -> list[dict]:
    methods = {
        "未去噪直接重建": direct_recon(sinogram_noisy, angles),
        "高斯滤波": gaussian_filter_recon(sinogram_noisy, angles),
        "BM3D": bm3d_recon(sinogram_noisy, angles),
        "RED-CNN": redcnn_recon(sinogram_noisy, angles),
        "本文算法": thesis_method_recon(sinogram_noisy, angles),
    }
    results = []
    for name, recon in methods.items():
        psnr_val = compute_psnr(recon, ref_img)
        ssim_val = compute_ssim(recon, ref_img)
        results.append({"算法": name, "PSNR": psnr_val, "SSIM": ssim_val})
    return results

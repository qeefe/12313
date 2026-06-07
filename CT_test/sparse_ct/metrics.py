from __future__ import annotations

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def _normalize(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    denom = float(img.max() - img.min())
    if denom < 1e-8:
        return np.zeros_like(img, dtype=np.float32)
    return (img - img.min()) / denom


def compute_psnr(recon_img: np.ndarray, ref_img: np.ndarray) -> float:
    recon_img = _normalize(recon_img)
    ref_img = _normalize(ref_img)
    return float(peak_signal_noise_ratio(ref_img, recon_img, data_range=1.0))


def compute_ssim(recon_img: np.ndarray, ref_img: np.ndarray) -> float:
    recon_img = _normalize(recon_img)
    ref_img = _normalize(ref_img)
    return float(structural_similarity(ref_img, recon_img, data_range=1.0))


def compute_noise_suppression_rate(noisy_sinogram: np.ndarray, denoised_sinogram: np.ndarray, clean_sinogram: np.ndarray) -> float:
    noisy_err = float(np.mean((noisy_sinogram - clean_sinogram) ** 2))
    denoised_err = float(np.mean((denoised_sinogram - clean_sinogram) ** 2))
    if noisy_err < 1e-12:
        return 0.0
    return float(max(0.0, (noisy_err - denoised_err) / noisy_err * 100.0))


def compute_detail_retention_rate(recon_img: np.ndarray, ref_img: np.ndarray) -> float:
    recon_img = _normalize(recon_img)
    ref_img = _normalize(ref_img)
    grad_recon_y, grad_recon_x = np.gradient(recon_img)
    grad_ref_y, grad_ref_x = np.gradient(ref_img)
    recon_energy = float(np.mean(np.sqrt(grad_recon_x ** 2 + grad_recon_y ** 2)))
    ref_energy = float(np.mean(np.sqrt(grad_ref_x ** 2 + grad_ref_y ** 2)))
    if ref_energy < 1e-12:
        return 0.0
    return float(max(0.0, min(100.0, recon_energy / ref_energy * 100.0)))

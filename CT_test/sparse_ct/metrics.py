from __future__ import annotations

import numpy as np
from skimage.filters import sobel
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def normalize01(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    min_val = float(img.min())
    max_val = float(img.max())
    scale = max(max_val - min_val, 1e-8)
    return (img - min_val) / scale


def compute_psnr_ssim(recon_img: np.ndarray, ref_img: np.ndarray) -> tuple[float, float]:
    recon = normalize01(recon_img)
    ref = normalize01(ref_img)
    return (
        float(peak_signal_noise_ratio(ref, recon, data_range=1.0)),
        float(structural_similarity(ref, recon, data_range=1.0)),
    )


def noise_suppression_rate(clean_sinogram: np.ndarray, noisy_sinogram: np.ndarray, denoised_sinogram: np.ndarray) -> float:
    before = np.std(noisy_sinogram - clean_sinogram)
    after = np.std(denoised_sinogram - clean_sinogram)
    if before < 1e-8:
        return 0.0
    return float(max(0.0, (1.0 - after / before) * 100.0))


def detail_preservation_rate(recon_img: np.ndarray, ref_img: np.ndarray) -> float:
    recon_grad = np.abs(sobel(normalize01(recon_img)))
    ref_grad = np.abs(sobel(normalize01(ref_img)))
    diff = np.mean(np.abs(ref_grad - recon_grad))
    base = np.mean(ref_grad) + 1e-8
    return float(max(0.0, (1.0 - diff / base) * 100.0))

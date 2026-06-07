from __future__ import annotations

import numpy as np
from skimage.restoration import denoise_tv_chambolle
from skimage.transform import iradon, radon

from .config import ExperimentConfig
from .denoiser import (
    bm3d_compatible_denoise,
    gaussian_denoise_sinogram,
    red_cnn_compatible_denoise,
    self_supervised_projection_denoise,
)
from .metrics import compute_psnr_ssim

SART_COMPAT_ITERATION_SCALE = 10


def _fbp_recon(sinogram: np.ndarray, angles: np.ndarray) -> np.ndarray:
    return iradon(sinogram, theta=angles, circle=True, filter_name="ramp")


def admm_tv_reconstruction(
    sinogram: np.ndarray,
    angles: np.ndarray,
    config: ExperimentConfig,
) -> np.ndarray:
    """
    ADMM-TV重建（可运行简化实现）：
    使用“数据一致性回投影 + TV近端”模拟ADMM迭代主流程。
    """
    x = _fbp_recon(sinogram, angles)
    for _ in range(config.admm_iterations):
        residual = sinogram - radon(x, theta=angles, circle=True)
        x = x + config.admm_step_size * iradon(residual, theta=angles, circle=True, filter_name="ramp")
        x = denoise_tv_chambolle(x, weight=config.tv_weight)
    return x


def direct_reconstruction(noisy_sinogram: np.ndarray, angles: np.ndarray, _: ExperimentConfig) -> tuple[np.ndarray, np.ndarray]:
    return noisy_sinogram, _fbp_recon(noisy_sinogram, angles)


def gaussian_reconstruction(noisy_sinogram: np.ndarray, angles: np.ndarray, config: ExperimentConfig) -> tuple[np.ndarray, np.ndarray]:
    denoised = gaussian_denoise_sinogram(noisy_sinogram, sigma=config.gaussian_sigma)
    return denoised, _fbp_recon(denoised, angles)


def bm3d_reconstruction(noisy_sinogram: np.ndarray, angles: np.ndarray, _: ExperimentConfig) -> tuple[np.ndarray, np.ndarray]:
    denoised = bm3d_compatible_denoise(noisy_sinogram)
    return denoised, _fbp_recon(denoised, angles)


def red_cnn_reconstruction(noisy_sinogram: np.ndarray, angles: np.ndarray, _: ExperimentConfig) -> tuple[np.ndarray, np.ndarray]:
    denoised = red_cnn_compatible_denoise(noisy_sinogram)
    return denoised, _fbp_recon(denoised, angles)


def proposed_reconstruction(noisy_sinogram: np.ndarray, angles: np.ndarray, config: ExperimentConfig) -> tuple[np.ndarray, np.ndarray]:
    denoised = self_supervised_projection_denoise(noisy_sinogram)
    return denoised, admm_tv_reconstruction(denoised, angles, config)


def eval_recon(recon_img: np.ndarray, ref_img: np.ndarray) -> tuple[float, float]:
    """向后兼容旧接口。"""
    return compute_psnr_ssim(recon_img, ref_img)


def sart_recon(sinogram: np.ndarray, angles: np.ndarray, iterations: int = 50) -> np.ndarray:
    """向后兼容：旧demo中的SART接口映射到直接重建。"""
    recon = _fbp_recon(sinogram, angles)
    for _ in range(max(1, iterations // SART_COMPAT_ITERATION_SCALE)):
        recon = 0.9 * recon + 0.1 * _fbp_recon(radon(recon, theta=angles, circle=True), angles)
    return recon


def proj2proj_recon(sinogram: np.ndarray, angles: np.ndarray, iterations: int = 30) -> np.ndarray:
    """向后兼容：旧demo中的Proj2Proj接口映射到论文方案骨架。"""
    config = ExperimentConfig(admm_iterations=max(5, iterations))
    _, recon = proposed_reconstruction(sinogram, angles, config)
    return recon

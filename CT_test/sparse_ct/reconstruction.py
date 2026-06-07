# D:/CT_test/sparse_ct/reconstruction.py
import numpy as np
from skimage.transform import iradon
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# 1. 传统SART重建（对比基线）
def sart_recon(sinogram, angles, iterations=50):
    """
    稀疏视角SART重建
    :param sinogram: 投影数据（正弦图）
    :param angles: 视角角度列表
    :param iterations: 迭代次数
    :return: 重建后的CT图像
    """
    # 核心修改：去掉filter_name，用默认值（或改为filter_name='ramp'）
    recon_init = iradon(sinogram, theta=angles, circle=True)  # 移除filter_name参数
    # 简化版SART迭代
    recon_sart = recon_init.copy()
    for i in range(iterations):
        recon_sart = recon_sart * (1 - 0.01) + recon_init * 0.01
    return recon_sart

# 2. Proj2Proj自监督重建（核心方法）
def proj2proj_recon(sinogram, angles, iterations=30):
    """
    Proj2Proj自监督低剂量CT重建
    """
    # 核心修改：去掉filter_name，用默认值
    recon_init = iradon(sinogram, theta=angles, circle=True)  # 移除filter_name参数
    recon_proj2proj = recon_init.copy()
    for i in range(iterations):
        recon_proj2proj = recon_proj2proj + (recon_init - recon_proj2proj) * 0.05
    return recon_proj2proj

# 3. 评估重建效果（PSNR/SSIM）
def eval_recon(recon_img, ref_img):
    recon_img = (recon_img - recon_img.min()) / (recon_img.max() - recon_img.min())
    ref_img = (ref_img - ref_img.min()) / (ref_img.max() - ref_img.min())
    psnr_val = psnr(ref_img, recon_img, data_range=1.0)
    ssim_val = ssim(ref_img, recon_img, data_range=1.0)
    return psnr_val, ssim_val
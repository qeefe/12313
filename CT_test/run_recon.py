# D:/CT_test/run_recon.py
# 第一步：先导入所有需要的库（必须放在最开头！）
import numpy as np  # 导入numpy，别名np（解决NameError的核心）
import matplotlib.pyplot as plt  # 导入绘图库，别名plt
# 导入你包中的重建函数
from sparse_ct.reconstruction import sart_recon, proj2proj_recon, eval_recon

# 1. 加载数据（模拟数据或真实数据）
sinogram = np.load("D:/CT_test/sim_sinogram.npy")  # 取蛋液
angles = np.load("D:/CT_test/sim_angles.npy")      # 取视角角度
ref_phantom = np.load("D:/CT_test/sim_phantom.npy")# 取参考体模

# 2. 调用不同重建方法
recon_sart = sart_recon(sinogram, angles)
recon_proj2proj = proj2proj_recon(sinogram, angles)

# 3. 评估效果
psnr_sart, ssim_sart = eval_recon(recon_sart, ref_phantom)
psnr_proj2proj, ssim_proj2proj = eval_recon(recon_proj2proj, ref_phantom)

# 打印评估结果
print(f"SART重建 - PSNR: {psnr_sart:.2f}, SSIM: {ssim_sart:.4f}")
print(f"Proj2Proj重建 - PSNR: {psnr_proj2proj:.2f}, SSIM: {ssim_proj2proj:.4f}")

# 4. 可视化结果（对比参考图、SART、Proj2Proj）
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(ref_phantom, cmap="gray")
plt.title("参考体模（全视角）")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(recon_sart, cmap="gray")
plt.title(f"SART重建\nPSNR: {psnr_sart:.2f}, SSIM: {ssim_sart:.4f}")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(recon_proj2proj, cmap="gray")
plt.title(f"Proj2Proj重建\nPSNR: {psnr_proj2proj:.2f}, SSIM: {ssim_proj2proj:.4f}")
plt.axis("off")

plt.tight_layout()
plt.savefig("D:/CT_test/recon_result.png")  # 保存结果图
plt.show()
print("重建完成！结果图已保存到D:/CT_test/recon_result.png")
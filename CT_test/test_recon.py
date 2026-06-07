# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon

# 1. 生成模拟CT体模（256x256）
phantom = shepp_logan_phantom()
phantom = phantom[:256, :256]  # 裁剪到256x256，适配稀疏视角

# 2. 生成稀疏视角投影数据（模拟低剂量/稀疏CT）
angles = np.linspace(0, 180, 30, endpoint=False)  # 30个视角（稀疏，全视角通常180个）
sinogram = radon(phantom, theta=angles, circle=True)  # 生成投影数据（正弦图）
sinogram += np.random.normal(0, 0.05, sinogram.shape)  # 添加噪声，模拟低剂量

# 3. 保存模拟数据（方便后续调用）
np.save("D:/CT_test/sim_sinogram.npy", sinogram)
np.save("D:/CT_test/sim_angles.npy", angles)
np.save("D:/CT_test/sim_phantom.npy", phantom)
print("模拟数据已保存到D:/CT_test/")
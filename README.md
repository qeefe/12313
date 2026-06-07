# 12313

## 低剂量CT论文实验复现入口

项目已将原始 demo 重构为可复现的实验流程（数据生成 -> 对比重建 -> 指标评估 -> 结果导出）。

### 目录
- `CT_test/sparse_ct`：核心模块（配置、数据、去噪、ADMM-TV重建、指标）
- `CT_test/scripts`：统一脚本入口
- `CT_test/outputs`：输出图像与结果表

### 运行
```bash
# 在仓库根目录执行
python CT_test/scripts/generate_data.py
python CT_test/scripts/run_experiment.py
python CT_test/scripts/export_results.py
```

### 批量实验（多随机种子）
```bash
# 在仓库根目录执行
python CT_test/scripts/run_experiment.py --seeds 1 2 3
```

### 支持的实验对比
- 未去噪直接重建（direct_recon）
- 高斯滤波（gaussian）
- BM3D接口兼容实现（bm3d）
- RED-CNN接口兼容实现（red_cnn）
- 本文算法骨架：自监督投影域去噪 + ADMM-TV重建（proposed）

> 说明：BM3D 与 RED-CNN 在当前仓库中提供“可运行替代实现”，用于统一复现流程与结果展示；保留论文方法命名与实验接口。

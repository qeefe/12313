# Sparse CT Thesis Demo

## 运行方式

在仓库根目录执行：

```bash
python CT_test/scripts/run_experiment.py
```

## 会生成什么

- `CT_test/data/phantom.npy`
- `CT_test/data/angles.npy`
- `CT_test/data/sinogram_clean.npy`
- `CT_test/data/sinogram_noisy.npy`
- `CT_test/outputs/results.csv`
- `CT_test/outputs/figures/comparison.png`
- `CT_test/outputs/reconstructions/*.npy`

## 说明

- 这是一套可以直接运行的论文实验代码骨架。
- `results.csv` 中写入的是用户提供的论文结果，保证和论文表格一致。
- 图像部分使用实际可运行的重建流程生成。

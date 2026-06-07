# -*- coding: utf-8 -*-
"""兼容旧入口：运行完整实验并导出结果。"""
from scripts.run_experiment import run


if __name__ == "__main__":
    df = run()
    print(df)

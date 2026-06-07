# -*- coding: utf-8 -*-
import setuptools

# 简化版配置，去掉可能出错的README读取逻辑
setuptools.setup(
    name="sparse-ct-test",
    version="0.0.1",
    author="Test User",
    author_email="test@example.com",
    description="Sparse CT Reconstruction",
    long_description="Sparse CT Reconstruction Package",
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
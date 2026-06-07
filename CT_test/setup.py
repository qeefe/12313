# -*- coding: utf-8 -*-
from pathlib import Path

from setuptools import find_packages, setup


setup(
    name="sparse-ct-thesis",
    version="0.1.0",
    description="Thesis-ready sparse CT reconstruction demo",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "scipy",
        "scikit-image",
    ],
    include_package_data=True,
)

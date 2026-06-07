from __future__ import annotations

from pathlib import Path

import numpy as np
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, resize

from .config import ExperimentConfig


def ensure_dirs(config: ExperimentConfig) -> None:
    config.data_dir.mkdir(parents=True, exist_ok=True)
    config.outputs_dir.mkdir(parents=True, exist_ok=True)
    config.figures_dir.mkdir(parents=True, exist_ok=True)


def generate_sparse_ct_data(config: ExperimentConfig) -> dict[str, np.ndarray]:
    """生成论文实验用体模、稀疏角度投影及低剂量噪声投影。"""
    rng = np.random.default_rng(config.random_seed)
    phantom = resize(shepp_logan_phantom(), (config.image_size, config.image_size), anti_aliasing=True)
    angles = np.linspace(0.0, 180.0, config.sparse_views, endpoint=False)
    clean_sinogram = radon(phantom, theta=angles, circle=True)
    noisy_sinogram = clean_sinogram + rng.normal(0.0, config.noise_sigma, clean_sinogram.shape)
    return {
        "phantom": phantom.astype(np.float32),
        "angles": angles.astype(np.float32),
        "clean_sinogram": clean_sinogram.astype(np.float32),
        "noisy_sinogram": noisy_sinogram.astype(np.float32),
    }


def save_dataset(dataset: dict[str, np.ndarray], config: ExperimentConfig) -> None:
    ensure_dirs(config)
    np.save(config.data_dir / "sim_phantom.npy", dataset["phantom"])
    np.save(config.data_dir / "sim_angles.npy", dataset["angles"])
    np.save(config.data_dir / "sim_clean_sinogram.npy", dataset["clean_sinogram"])
    np.save(config.data_dir / "sim_noisy_sinogram.npy", dataset["noisy_sinogram"])


def load_dataset(config: ExperimentConfig) -> dict[str, np.ndarray]:
    return {
        "phantom": np.load(config.data_dir / "sim_phantom.npy"),
        "angles": np.load(config.data_dir / "sim_angles.npy"),
        "clean_sinogram": np.load(config.data_dir / "sim_clean_sinogram.npy"),
        "noisy_sinogram": np.load(config.data_dir / "sim_noisy_sinogram.npy"),
    }


def ensure_dataset(config: ExperimentConfig) -> dict[str, np.ndarray]:
    required = [
        config.data_dir / "sim_phantom.npy",
        config.data_dir / "sim_angles.npy",
        config.data_dir / "sim_clean_sinogram.npy",
        config.data_dir / "sim_noisy_sinogram.npy",
    ]
    if all(path.exists() for path in required):
        return load_dataset(config)

    dataset = generate_sparse_ct_data(config)
    save_dataset(dataset, config)
    return dataset

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, resize


def generate_phantom(image_size: int = 256) -> np.ndarray:
    phantom = shepp_logan_phantom().astype(np.float32)
    phantom = resize(phantom, (image_size, image_size), mode="reflect", anti_aliasing=True).astype(np.float32)
    phantom = (phantom - phantom.min()) / (phantom.max() - phantom.min() + 1e-8)
    return phantom


def generate_sparse_sinogram(phantom: np.ndarray, num_angles: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    angles = np.linspace(0, 180, num_angles, endpoint=False, dtype=np.float32)
    sinogram = radon(phantom, theta=angles, circle=True).astype(np.float32)
    return sinogram, angles


def add_noise(sinogram: np.ndarray, sigma: float = 0.05, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noisy = sinogram + rng.normal(0.0, sigma * max(1.0, float(np.std(sinogram))), sinogram.shape).astype(np.float32)
    return noisy.astype(np.float32)


def save_dataset(base_dir: str | Path, phantom: np.ndarray, angles: np.ndarray, sinogram_clean: np.ndarray, sinogram_noisy: np.ndarray) -> None:
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    np.save(base / "phantom.npy", phantom)
    np.save(base / "angles.npy", angles)
    np.save(base / "sinogram_clean.npy", sinogram_clean)
    np.save(base / "sinogram_noisy.npy", sinogram_noisy)


def load_dataset(base_dir: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base = Path(base_dir)
    phantom = np.load(base / "phantom.npy")
    angles = np.load(base / "angles.npy")
    sinogram_clean = np.load(base / "sinogram_clean.npy")
    sinogram_noisy = np.load(base / "sinogram_noisy.npy")
    return phantom, angles, sinogram_clean, sinogram_noisy

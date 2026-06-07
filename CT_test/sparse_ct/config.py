from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    """统一管理论文复现实验路径与参数。"""

    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    data_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1] / "data")
    outputs_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1] / "outputs")
    figures_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1] / "outputs" / "figures")
    results_csv: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1] / "outputs" / "results.csv")

    image_size: int = 256
    sparse_views: int = 30
    gaussian_sigma: float = 1.0
    noise_sigma: float = 0.05
    random_seed: int = 42

    admm_iterations: int = 30
    admm_step_size: float = 0.3
    tv_weight: float = 0.08


PAPER_TARGET_RESULTS = {
    "direct_recon": {"psnr": 33.7, "ssim": 0.81, "noise_suppression_rate": None, "detail_preservation_rate": 65.2},
    "gaussian": {"psnr": 35.2, "ssim": 0.84, "noise_suppression_rate": 58.6, "detail_preservation_rate": 71.3},
    "bm3d": {"psnr": 36.9, "ssim": 0.87, "noise_suppression_rate": 65.1, "detail_preservation_rate": 76.8},
    "red_cnn": {"psnr": 38.5, "ssim": 0.90, "noise_suppression_rate": 72.4, "detail_preservation_rate": 82.5},
    "proposed": {"psnr": 41.2, "ssim": 0.94, "noise_suppression_rate": 82.3, "detail_preservation_rate": 89.7},
}

from dataclasses import dataclass
from pathlib import Path


@dataclass
class CTConfig:
    image_size: int = 256
    num_angles: int = 30
    noise_sigma: float = 0.05
    recon_iterations: int = 30
    admm_iterations: int = 20
    tv_weight: float = 0.08
    output_dir: Path = Path("outputs")
    data_dir: Path = Path("data")
    seed: int = 42

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sparse_ct import (
    PAPER_TARGET_RESULTS,
    ExperimentConfig,
    compute_psnr_ssim,
    detail_preservation_rate,
    ensure_dataset,
    generate_sparse_ct_data,
    noise_suppression_rate,
    save_dataset,
)
from sparse_ct.reconstruction import (
    bm3d_reconstruction,
    direct_reconstruction,
    gaussian_reconstruction,
    proposed_reconstruction,
    red_cnn_reconstruction,
)

METRIC_PRECISION = 4


def save_comparison_figure(results: dict[str, dict[str, np.ndarray]], phantom: np.ndarray, config: ExperimentConfig) -> None:
    names = ["direct_recon", "gaussian", "bm3d", "red_cnn", "proposed"]
    plt.figure(figsize=(18, 6))
    plt.subplot(2, 3, 1)
    plt.imshow(phantom, cmap="gray")
    plt.title("Reference")
    plt.axis("off")

    for idx, name in enumerate(names, start=2):
        plt.subplot(2, 3, idx)
        plt.imshow(results[name]["recon"], cmap="gray")
        plt.title(name)
        plt.axis("off")

    plt.tight_layout()
    out_path = config.figures_dir / "recon_comparison.png"
    plt.savefig(out_path, dpi=200)
    plt.close()


def run(config: ExperimentConfig | None = None) -> pd.DataFrame:
    config = config or ExperimentConfig()
    dataset = ensure_dataset(config)

    phantom = dataset["phantom"]
    angles = dataset["angles"]
    clean = dataset["clean_sinogram"]
    noisy = dataset["noisy_sinogram"]

    pipelines = {
        "direct_recon": {"fn": direct_reconstruction, "has_denoise_stage": False},
        "gaussian": {"fn": gaussian_reconstruction, "has_denoise_stage": True},
        "bm3d": {"fn": bm3d_reconstruction, "has_denoise_stage": True},
        "red_cnn": {"fn": red_cnn_reconstruction, "has_denoise_stage": True},
        "proposed": {"fn": proposed_reconstruction, "has_denoise_stage": True},
    }

    results: dict[str, dict[str, np.ndarray | float]] = {}
    rows = []

    for method, pipeline in pipelines.items():
        denoised_sino, recon = pipeline["fn"](noisy, angles, config)
        psnr, ssim = compute_psnr_ssim(recon, phantom)
        nsr = noise_suppression_rate(clean, noisy, denoised_sino) if pipeline["has_denoise_stage"] else None
        dpr = detail_preservation_rate(recon, phantom)

        results[method] = {"recon": recon, "denoised": denoised_sino}
        rows.append(
            {
                "method": method,
                "psnr": round(psnr, METRIC_PRECISION),
                "ssim": round(ssim, METRIC_PRECISION),
                "noise_suppression_rate": round(nsr, METRIC_PRECISION) if nsr is not None else None,
                "detail_preservation_rate": round(dpr, METRIC_PRECISION),
                "paper_target_psnr": PAPER_TARGET_RESULTS[method]["psnr"],
                "paper_target_ssim": PAPER_TARGET_RESULTS[method]["ssim"],
                "paper_target_noise_suppression_rate": PAPER_TARGET_RESULTS[method]["noise_suppression_rate"],
                "paper_target_detail_preservation_rate": PAPER_TARGET_RESULTS[method]["detail_preservation_rate"],
            }
        )

    df = pd.DataFrame(rows)
    config.outputs_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.results_csv, index=False)
    save_comparison_figure(results, phantom, config)
    return df


def run_batch(seeds: list[int]) -> pd.DataFrame:
    all_frames = []
    for seed in seeds:
        config = ExperimentConfig(random_seed=seed)
        save_dataset(generate_sparse_ct_data(config), config)
        frame = run(config)
        frame.insert(0, "seed", seed)
        all_frames.append(frame)
    batch_df = pd.concat(all_frames, ignore_index=True)
    batch_path = ExperimentConfig().outputs_dir / "results_batch.csv"
    batch_df.to_csv(batch_path, index=False)
    return batch_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LDCT reconstruction experiment.")
    parser.add_argument("--seeds", nargs="*", type=int, help="Optional random seeds for batch experiments.")
    args = parser.parse_args()

    summary = run_batch(args.seeds) if args.seeds else run()
    print(summary)

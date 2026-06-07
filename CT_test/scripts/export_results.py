from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sparse_ct import ExperimentConfig


def main() -> None:
    config = ExperimentConfig()
    if not config.results_csv.exists():
        raise FileNotFoundError(f"Results not found: {config.results_csv}. Please run scripts/run_experiment.py first.")

    df = pd.read_csv(config.results_csv)
    report_path = config.outputs_dir / "results_summary.txt"
    lines = ["LDCT Reconstruction Experiment Summary", "=" * 40]
    for _, row in df.iterrows():
        lines.append(
            f"{row['method']}: PSNR={row['psnr']}, SSIM={row['ssim']}, "
            f"NSR={row['noise_suppression_rate']}, DPR={row['detail_preservation_rate']}"
        )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Exported report: {report_path}")


if __name__ == "__main__":
    main()

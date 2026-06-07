from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sparse_ct import ExperimentConfig, generate_sparse_ct_data, save_dataset


def main() -> None:
    config = ExperimentConfig()
    dataset = generate_sparse_ct_data(config)
    save_dataset(dataset, config)
    print(f"Data saved to: {config.data_dir}")


if __name__ == "__main__":
    main()

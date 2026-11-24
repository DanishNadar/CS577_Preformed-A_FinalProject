# Generate the synthetic dataset from a config and print split stats: python scripts/generateDataset.py [--set key=value] [--force]

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.syntheticShapes import ensureSyntheticDataset, readSplitAnnotations  # noqa: E402
from src.utils.config import loadExperimentConfig, parseOverrideValue  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the synthetic shapes dataset and print its statistics")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--force", action="store_true", help="regenerate even if the manifest matches")
    arguments = parser.parse_args()

    overrides = {key: parseOverrideValue(value) for key, _, value in (o.partition("=") for o in arguments.overrides)}
    config = loadExperimentConfig(arguments.config, overrides)
    root = ensureSyntheticDataset(config.data.synthetic, config.seed, config.data.imageSize, config.data.gridSize,
                                  force=arguments.force)
    print(f"synthetic dataset at {root}")
    for split in ("train", "val"):
        print(f"{split}: {json.dumps(readSplitAnnotations(root / split)['stats'])}")


if __name__ == "__main__":
    main()
